from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from .config import settings
from .decoder import SNACDecoder
from .engine import OrpheusEngine
from .schemas import (
    HealthResponse,
    SpeechMetricsResponse,
    TTSRequest,
    VoiceInfo,
    VoicesResponse,
)

logger = logging.getLogger(__name__)

engine: OrpheusEngine | None = None
decoder: SNACDecoder | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global engine, decoder  # noqa: PLW0603

    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )

    logger.info("Starting Orpheus TTS server …")
    engine = OrpheusEngine(settings)
    decoder = SNACDecoder(settings.snac_model_name, device=settings.snac_device)
    logger.info("Server ready – accepting requests")

    yield

    logger.info("Shutting down …")
    engine = None
    decoder = None


app = FastAPI(
    title="Orpheus TTS",
    version="0.1.0",
    lifespan=lifespan,
    default_response_class=JSONResponse,
)


_STATIC_DIR = Path(__file__).parent / "static"


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def ui():
    return HTMLResponse((_STATIC_DIR / "index.html").read_text())


# ── helpers ───────────────────────────────────────────────────────


def _validate_voice(voice: str) -> str:
    if voice not in settings.voice_list:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown voice '{voice}'. Available: {settings.voice_list}",
        )
    return voice


async def _audio_stream(req: TTSRequest) -> AsyncIterator[bytes]:
    """Orchestrates the full pipeline: text → vLLM tokens → SNAC decode → PCM chunks."""
    assert engine is not None and decoder is not None  # noqa: S101

    token_gen = engine.generate_tokens(
        text=req.input,
        voice=req.voice,
        temperature=req.temperature,
        top_p=req.top_p,
        max_tokens=req.max_tokens,
        repetition_penalty=req.repetition_penalty,
    )

    chunk_idx = 0
    total_bytes = 0
    async for chunk in decoder.decode_stream(
        token_gen,
        min_frames_first=settings.min_frames_first,
        min_frames_subsequent=settings.min_frames_subsequent,
    ):
        total_bytes += len(chunk)
        chunk_idx += 1
        if chunk_idx <= 3 or chunk_idx % 20 == 0:
            logger.debug("Audio chunk #%d: %d bytes (total %d)", chunk_idx, len(chunk), total_bytes)
        yield chunk

    logger.info("Stream finished: %d chunks, %d bytes", chunk_idx, total_bytes)


async def _collect_speech_metrics(req: TTSRequest) -> SpeechMetricsResponse:
    """Run the full pipeline and return latency and throughput metrics."""
    assert engine is not None and decoder is not None  # noqa: S101

    request_id = uuid.uuid4().hex
    started = time.perf_counter()
    first_token_ms: float | None = None
    first_audio_chunk_ms: float | None = None
    token_deltas = 0
    codec_tokens = 0
    audio_chunks = 0
    audio_bytes = 0

    async def _instrumented_tokens() -> AsyncIterator[str]:
        nonlocal first_token_ms, token_deltas, codec_tokens

        async for delta in engine.generate_tokens(
            text=req.input,
            voice=req.voice,
            temperature=req.temperature,
            top_p=req.top_p,
            max_tokens=req.max_tokens,
            repetition_penalty=req.repetition_penalty,
            request_id=request_id,
        ):
            token_count = decoder.count_codec_tokens(delta)
            if token_count:
                token_deltas += 1
                codec_tokens += token_count
                if first_token_ms is None:
                    first_token_ms = (time.perf_counter() - started) * 1000
            yield delta

    async for chunk in decoder.decode_stream(
        _instrumented_tokens(),
        min_frames_first=settings.min_frames_first,
        min_frames_subsequent=settings.min_frames_subsequent,
    ):
        audio_chunks += 1
        audio_bytes += len(chunk)
        if first_audio_chunk_ms is None:
            first_audio_chunk_ms = (time.perf_counter() - started) * 1000

    total_generation_ms = (time.perf_counter() - started) * 1000
    logger.info(
        "Metrics request complete: id=%s first_token=%.1fms first_audio=%.1fms total=%.1fms",
        request_id,
        first_token_ms or -1.0,
        first_audio_chunk_ms or -1.0,
        total_generation_ms,
    )
    return SpeechMetricsResponse(
        request_id=request_id,
        status="completed",
        voice=req.voice,
        input_chars=len(req.input),
        token_deltas=token_deltas,
        codec_tokens=codec_tokens,
        audio_chunks=audio_chunks,
        audio_bytes=audio_bytes,
        first_token_ms=first_token_ms,
        first_audio_chunk_ms=first_audio_chunk_ms,
        total_generation_ms=total_generation_ms,
    )


# ── HTTP endpoints ────────────────────────────────────────────────


@app.post(
    "/v1/audio/speech/stream",
    response_class=StreamingResponse,
    summary="Stream TTS audio as raw PCM",
)
async def tts_stream(req: TTSRequest):
    """Returns a chunked ``audio/pcm`` stream (24 kHz · 16-bit · mono).

    Audio chunks are emitted as soon as the model has produced enough tokens,
    keeping time-to-first-byte as low as possible.
    """
    _validate_voice(req.voice)
    t0 = time.perf_counter()

    async def _generate():
        first = True
        async for chunk in _audio_stream(req):
            if first:
                logger.info(
                    "TTFB %.1f ms  voice=%s  text=%.60s…",
                    (time.perf_counter() - t0) * 1000,
                    req.voice,
                    req.input,
                )
                first = False
            yield chunk

    return StreamingResponse(
        _generate(),
        media_type="audio/pcm",
        headers={
            "X-Sample-Rate": str(SNACDecoder.SAMPLE_RATE),
            "X-Bit-Depth": str(SNACDecoder.BIT_DEPTH),
            "X-Channels": str(SNACDecoder.CHANNELS),
        },
    )


@app.post(
    "/v1/audio/speech",
    response_class=StreamingResponse,
    summary="Generate full TTS audio as WAV",
)
async def tts_full(req: TTSRequest):
    """Generates the complete audio and returns it as a WAV file."""
    import struct
    import io

    _validate_voice(req.voice)

    pcm_chunks: list[bytes] = []
    async for chunk in _audio_stream(req):
        pcm_chunks.append(chunk)

    pcm_data = b"".join(pcm_chunks)

    buf = io.BytesIO()
    num_samples = len(pcm_data) // 2
    data_size = num_samples * 2
    sample_rate = SNACDecoder.SAMPLE_RATE

    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + data_size))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack("<I", 16))
    buf.write(struct.pack("<HHIIHH", 1, 1, sample_rate, sample_rate * 2, 2, 16))
    buf.write(b"data")
    buf.write(struct.pack("<I", data_size))
    buf.write(pcm_data)

    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="audio/wav",
        headers={"Content-Disposition": 'attachment; filename="speech.wav"'},
    )


@app.post(
    "/v1/audio/speech/metrics",
    response_class=JSONResponse,
    response_model=SpeechMetricsResponse,
    summary="Generate speech and return per-request timing metrics",
)
async def tts_metrics(req: TTSRequest):
    _validate_voice(req.voice)
    return await _collect_speech_metrics(req)


# ── utility endpoints ─────────────────────────────────────────────


@app.get("/v1/voices", response_model=VoicesResponse)
async def list_voices():
    voices = [VoiceInfo(id=v, name=v.capitalize()) for v in settings.voice_list]
    return VoicesResponse(voices=voices, default=settings.default_voice, count=len(voices))


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok" if engine and decoder else "degraded",
        engine_ready=engine is not None and engine.is_ready,
        decoder_ready=decoder is not None,
    )
