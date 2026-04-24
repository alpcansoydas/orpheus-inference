from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class TTSRequest(BaseModel):
    input: str = Field(..., min_length=1, max_length=4096, description="Text to synthesize.")
    model: str | None = Field(
        default=None,
        description=(
            "Model id to use (see ``GET /v1/models``). Defaults to the server's "
            "configured default model when omitted."
        ),
    )
    voice: str | None = Field(
        default=None,
        description=(
            "Speaker voice identifier. Required for multi-speaker models, "
            "ignored for single-speaker ones."
        ),
    )
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    max_tokens: int | None = Field(default=None, ge=1, le=4096)
    repetition_penalty: float | None = Field(default=None, ge=1.0, le=2.0)


class VoiceInfo(BaseModel):
    id: str
    name: str


class VoicesResponse(BaseModel):
    model: str
    voices: list[VoiceInfo]
    default: str | None
    count: int


class ModelInfo(BaseModel):
    id: str
    display_name: str
    model_name: str
    language: str
    description: str
    multi_speaker: bool
    voices: list[VoiceInfo]
    default_voice: str | None


class ModelsResponse(BaseModel):
    models: list[ModelInfo]
    default: str
    count: int


class HealthResponse(BaseModel):
    status: str
    engines_ready: int
    engines_total: int
    models: list[str]
    decoder_ready: bool


class SpeechMetricsResponse(BaseModel):
    request_id: str
    status: Literal["completed"]
    model: str
    voice: str | None
    input_chars: int
    token_deltas: int
    codec_tokens: int
    audio_chunks: int
    audio_bytes: int
    ttft_ms: float | None
    ttfa_ms: float | None
    total_generation_ms: float
