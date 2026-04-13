from __future__ import annotations

from pydantic import BaseModel, Field


class TTSRequest(BaseModel):
    input: str = Field(..., min_length=1, max_length=4096, description="Text to synthesize.")
    voice: str = Field(default="tara", description="Speaker voice identifier.")
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    max_tokens: int | None = Field(default=None, ge=1, le=4096)
    repetition_penalty: float | None = Field(default=None, ge=1.0, le=2.0)


class VoiceInfo(BaseModel):
    id: str
    name: str


class VoicesResponse(BaseModel):
    voices: list[VoiceInfo]
    default: str
    count: int


class HealthResponse(BaseModel):
    status: str
    engine_ready: bool
    decoder_ready: bool
