from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from .config import settings
from .decoder import SNACDecoder
from .engine import OrpheusEngine
from .models_registry import ModelProfile, ModelRegistry
from .schemas import (
    HealthResponse,
    ModelInfo,
    ModelsResponse,
    SpeechMetricsResponse,
    TTSRequest,
    VoiceInfo,
    VoicesResponse,
)

logger = logging.getLogger(__name__)

registry: ModelRegistry | None = None
engines: dict[str, OrpheusEngine] = {}
decoder: SNACDecoder | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global registry, decoder  # noqa: PLW0603

    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )

    logger.info("Starting Orpheus TTS server …")
    registry = ModelRegistry.from_config(
        settings.enabled_model_ids,
        overrides_file=settings.models_file or None,
    )

    profiles = registry.all()
    if settings.default_model not in registry:
        raise RuntimeError(
            f"DEFAULT_MODEL='{settings.default_model}' is not in enabled models: "
            f"{registry.ids()}"
        )

    per_model_gpu = settings.resolve_gpu_memory_utilization(len(profiles))
    logger.info(
        "Loading %d model(s) – %s – per-engine gpu_memory_utilization=%.2f",
        len(profiles),
        [p.id for p in profiles],
        per_model_gpu,
    )

    for profile in profiles:
        engines[profile.id] = OrpheusEngine(
            settings,
            profile,
            gpu_memory_utilization=per_model_gpu,
        )

    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        logger.info("CUDA sync OK after engine init – GPU state is clean")

    decoder = SNACDecoder(settings.snac_model_name, device=settings.snac_device)
    logger.info(
        "Server ready – default model=%s – accepting requests",
        settings.default_model,
    )

    yield

    logger.info("Shutting down …")
    engines.clear()
    decoder = None  # noqa: F841


app = FastAPI(
    title="Orpheus TTS",
    version="0.2.0",
    lifespan=lifespan,
    default_response_class=JSONResponse,
)


_STATIC_DIR = Path(__file__).parent / "static"


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def ui():
    return HTMLResponse((_STATIC_DIR / "index.html").read_text())


# ── helpers ───────────────────────────────────────────────────────


def _resolve_profile(model_id: str | None) -> ModelProfile:
    assert registry is not None  # noqa: S101
    target = model_id or settings.default_model
    if target not in registry:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown model '{target}'. Available: {registry.ids()}",
        )
    return registry.get(target)


def _resolve_voice(profile: ModelProfile, voice: str | None) -> str:
    try:
        profile.validate_voice(voice)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return profile.resolve_voice(voice)


def _engine_for(profile: ModelProfile) -> OrpheusEngine:
    try:
        return engines[profile.id]
    except KeyError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Engine for model '{profile.id}' is not ready",
        ) from exc


def _profile_to_info(profile: ModelProfile) -> ModelInfo:
    return ModelInfo(
        id=profile.id,
        display_name=profile.display_name,
        model_name=profile.model_name,
        language=profile.language,
        description=profile.description,
        multi_speaker=profile.is_multi_speaker,
        voices=[VoiceInfo(id=v, name=v.capitalize()) for v in profile.voices],
        default_voice=profile.default_voice,
    )


async def _audio_stream(
    req: TTSRequest,
    profile: ModelProfile,
    voice: str,
) -> AsyncIterator[bytes]:
    """Orchestrates the full pipeline: text → vLLM tokens → SNAC decode → PCM chunks."""
    assert decoder is not None  # noqa: S101
    engine = _engine_for(profile)

    token_gen = engine.generate_tokens(
        text=req.input,
        voice=voice,
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
            logger.debug(
                "Audio chunk #%d: %d bytes (total %d) model=%s",
                chunk_idx,
                len(chunk),
                total_bytes,
                profile.id,
            )
        yield chunk

    logger.info(
        "Stream finished: model=%s chunks=%d bytes=%d",
        profile.id,
        chunk_idx,
        total_bytes,
    )


async def _collect_speech_metrics(
    req: TTSRequest,
    profile: ModelProfile,
    voice: str,
) -> SpeechMetricsResponse:
    """Run the full pipeline and return latency and throughput metrics."""
    assert decoder is not None  # noqa: S101
    engine = _engine_for(profile)

    request_id = uuid.uuid4().hex
    started = time.perf_counter()
    ttft_ms: float | None = None
    ttfa_ms: float | None = None
    token_deltas = 0
    codec_tokens = 0
    audio_chunks = 0
    audio_bytes = 0

    async def _instrumented_tokens() -> AsyncIterator[str]:
        nonlocal ttft_ms, token_deltas, codec_tokens

        async for delta in engine.generate_tokens(
            text=req.input,
            voice=voice,
            temperature=req.temperature,
            top_p=req.top_p,
            max_tokens=req.max_tokens,
            repetition_penalty=req.repetition_penalty,
            request_id=request_id,
        ):
            if ttft_ms is None:
                ttft_ms = (time.perf_counter() - started) * 1000
            token_deltas += 1
            token_count = decoder.count_codec_tokens(delta)
            if token_count:
                codec_tokens += token_count
            yield delta

    async for chunk in decoder.decode_stream(
        _instrumented_tokens(),
        min_frames_first=settings.min_frames_first,
        min_frames_subsequent=settings.min_frames_subsequent,
    ):
        audio_chunks += 1
        audio_bytes += len(chunk)
        if ttfa_ms is None:
            ttfa_ms = (time.perf_counter() - started) * 1000

    total_generation_ms = (time.perf_counter() - started) * 1000
    logger.info(
        "Metrics request complete: id=%s model=%s ttft=%.1fms ttfa=%.1fms total=%.1fms",
        request_id,
        profile.id,
        ttft_ms or -1.0,
        ttfa_ms or -1.0,
        total_generation_ms,
    )
    return SpeechMetricsResponse(
        request_id=request_id,
        status="completed",
        model=profile.id,
        voice=voice or None,
        input_chars=len(req.input),
        token_deltas=token_deltas,
        codec_tokens=codec_tokens,
        audio_chunks=audio_chunks,
        audio_bytes=audio_bytes,
        ttft_ms=ttft_ms,
        ttfa_ms=ttfa_ms,
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
    profile = _resolve_profile(req.model)
    voice = _resolve_voice(profile, req.voice)

    t0 = time.perf_counter()

    async def _generate():
        first = True
        async for chunk in _audio_stream(req, profile, voice):
            if first:
                logger.info(
                    "TTFB %.1f ms  model=%s  voice=%s  text=%.60s…",
                    (time.perf_counter() - t0) * 1000,
                    profile.id,
                    voice or "-",
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
            "X-Model": profile.id,
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

    profile = _resolve_profile(req.model)
    voice = _resolve_voice(profile, req.voice)

    pcm_chunks: list[bytes] = []
    async for chunk in _audio_stream(req, profile, voice):
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
        headers={
            "Content-Disposition": 'attachment; filename="speech.wav"',
            "X-Model": profile.id,
        },
    )


@app.post(
    "/v1/audio/speech/metrics",
    response_class=JSONResponse,
    response_model=SpeechMetricsResponse,
    summary="Generate speech and return per-request timing metrics",
)
async def tts_metrics(req: TTSRequest):
    profile = _resolve_profile(req.model)
    voice = _resolve_voice(profile, req.voice)
    return await _collect_speech_metrics(req, profile, voice)


# ── utility endpoints ─────────────────────────────────────────────


@app.get("/v1/models", response_model=ModelsResponse)
async def list_models():
    assert registry is not None  # noqa: S101
    infos = [_profile_to_info(p) for p in registry.all()]
    return ModelsResponse(
        models=infos,
        default=settings.default_model,
        count=len(infos),
    )


@app.get("/v1/voices", response_model=VoicesResponse)
async def list_voices(
    model: str | None = Query(
        default=None,
        description="Model id to list voices for. Defaults to the server's default model.",
    ),
):
    profile = _resolve_profile(model)
    voices = [VoiceInfo(id=v, name=v.capitalize()) for v in profile.voices]
    return VoicesResponse(
        model=profile.id,
        voices=voices,
        default=profile.default_voice,
        count=len(voices),
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    ready = sum(1 for e in engines.values() if e.is_ready)
    total = len(engines)
    status = "ok" if ready == total and total > 0 and decoder is not None else "degraded"
    return HealthResponse(
        status=status,
        engines_ready=ready,
        engines_total=total,
        models=list(engines.keys()),
        decoder_ready=decoder is not None,
    )
