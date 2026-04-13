from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # ── vLLM model ────────────────────────────────────────────────
    model_name: str = "canopylabs/orpheus-3b-0.1-ft"
    tokenizer_name: str = "canopylabs/orpheus-3b-0.1-pretrained"
    dtype: str = "bfloat16"
    max_model_len: int = 2048
    gpu_memory_utilization: float = 0.90
    max_num_seqs: int = 64
    max_num_batched_tokens: int = 4096
    enable_chunked_prefill: bool = True
    enable_prefix_caching: bool = True
    block_size: int = 16

    # ── Sampling ──────────────────────────────────────────────────
    temperature: float = 0.4
    top_p: float = 0.9
    max_tokens: int = 1200
    repetition_penalty: float = 1.1
    stop_token_ids: str = "128258"

    # ── SNAC decoder ──────────────────────────────────────────────
    snac_model_name: str = "hubertsiuzdak/snac_24khz"
    snac_device: str = ""

    # ── Streaming tuning ──────────────────────────────────────────
    # Number of complete 7-token frames needed before emitting the
    # first audio chunk (lower → faster TTFB, potentially more artefacts).
    min_frames_first: int = 1
    # Sliding-window size (in frames) for subsequent chunks.
    min_frames_subsequent: int = 4

    # ── Voices ────────────────────────────────────────────────────
    available_voices: str = "tara,zoe,jess,zac,leo,mia,julia,leah"
    default_voice: str = "tara"

    # ── Server ────────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"

    # ── Derived helpers ───────────────────────────────────────────
    @property
    def voice_list(self) -> list[str]:
        return [v.strip() for v in self.available_voices.split(",")]

    @property
    def stop_token_id_list(self) -> list[int]:
        return [int(t.strip()) for t in self.stop_token_ids.split(",")]


settings = Settings()
