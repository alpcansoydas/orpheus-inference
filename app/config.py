from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # ── Models ────────────────────────────────────────────────────
    # Comma-separated list of model ids to load at startup. Each id must
    # resolve to an entry in the built-in ModelRegistry or in MODELS_FILE.
    enabled_models: str = "orpheus-tr"
    # Default model id used when a request does not specify ``model``.
    default_model: str = "orpheus-tr"
    # Optional JSON file with additional / overridden ``ModelProfile`` entries.
    models_file: str = ""

    # ── vLLM engine defaults (shared across models) ───────────────
    dtype: str = "bfloat16"
    max_model_len: int = 2048
    # Total VRAM fraction vLLM should use across *all* loaded models. The
    # actual per-engine value is derived by dividing this by the number of
    # enabled models unless ``per_model_gpu_memory_utilization`` is set.
    gpu_memory_utilization: float = 0.90
    per_model_gpu_memory_utilization: float = 0.0
    max_num_seqs: int = 64
    max_num_batched_tokens: int = 2048
    enable_chunked_prefill: bool = True
    enable_prefix_caching: bool = True
    block_size: int = 16
    # Disable CUDA graph capture. This is enabled by default because it trades
    # a bit of throughput for much lower VRAM usage and avoids startup crashes
    # on some GPUs / driver stacks.
    enforce_eager: bool = True

    # ── Legacy single-model overrides ─────────────────────────────
    # If set, these override the ``orpheus-en`` profile's model/tokenizer.
    # They are kept for backwards compatibility with old .env files.
    model_name: str = ""
    tokenizer_name: str = ""

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
    min_frames_first: int = 4
    min_frames_subsequent: int = 4

    # ── Server ────────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"

    # ── Derived helpers ───────────────────────────────────────────
    @property
    def enabled_model_ids(self) -> list[str]:
        return [m.strip() for m in self.enabled_models.split(",") if m.strip()]

    @property
    def stop_token_id_list(self) -> list[int]:
        return [int(t.strip()) for t in self.stop_token_ids.split(",")]

    def resolve_gpu_memory_utilization(self, num_models: int) -> float:
        """Per-engine VRAM fraction.

        If the operator has set an explicit ``per_model_gpu_memory_utilization``
        we honour it verbatim; otherwise we divide the total budget equally.
        """
        if self.per_model_gpu_memory_utilization > 0:
            return self.per_model_gpu_memory_utilization
        if num_models <= 0:
            return self.gpu_memory_utilization
        return self.gpu_memory_utilization / num_models


settings = Settings()
