from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

import orjson
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse

from .config import settings
from .decoder import SNACDecoder
from .engine import OrpheusEngine
from .schemas import HealthResponse, TTSRequest, VoiceInfo, VoicesResponse

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
    default_response_class=StreamingResponse,
)


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

    async for chunk in decoder.decode_stream(
        token_gen,
        min_frames_first=settings.min_frames_first,
        min_frames_subsequent=settings.min_frames_subsequent,
    ):
        yield chunk


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


# ── WebSocket endpoint ────────────────────────────────────────────


@app.websocket("/v1/audio/speech/ws")
async def tts_websocket(ws: WebSocket):
    """Bidirectional WebSocket endpoint.

    Send a JSON message matching :class:`TTSRequest` to start generation.
    The server streams back binary PCM frames and a final JSON
    ``{"done": true}`` message.
    """
    await ws.accept()
    try:
        while True:
            raw = await ws.receive_text()
            try:
                data = orjson.loads(raw)
                req = TTSRequest(**data)
            except Exception as exc:
                await ws.send_text(orjson.dumps({"error": str(exc)}).decode())
                continue

            if req.voice not in settings.voice_list:
                await ws.send_text(
                    orjson.dumps({"error": f"Unknown voice '{req.voice}'"}).decode()
                )
                continue

            async for chunk in _audio_stream(req):
                await ws.send_bytes(chunk)

            await ws.send_text(orjson.dumps({"done": True}).decode())
    except WebSocketDisconnect:
        logger.debug("WebSocket client disconnected")


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
