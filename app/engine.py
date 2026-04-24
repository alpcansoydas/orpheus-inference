from __future__ import annotations

import logging
import uuid
from typing import AsyncIterator

from transformers import AutoConfig, AutoTokenizer
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine

from .config import Settings
from .models_registry import ModelProfile

logger = logging.getLogger(__name__)


# Orpheus special token IDs (fixed by the model vocabulary).
_TOKEN_START_OF_PROMPT = 128259
_TOKEN_EOT = 128009
_TOKEN_AUDIO_PREFIX_1 = 128260
_TOKEN_AUDIO_PREFIX_2 = 128261
_TOKEN_START_OF_AUDIO = 128257


class OrpheusEngine:
    """Thin wrapper around a vLLM ``AsyncLLMEngine`` that knows how to
    format Orpheus prompts and stream generated token text.

    One engine serves exactly one :class:`ModelProfile`; the registry in
    :mod:`app.main` keeps a mapping ``profile.id -> OrpheusEngine`` so
    requests can be routed to the correct checkpoint at dispatch time.
    """

    def __init__(
        self,
        cfg: Settings,
        profile: ModelProfile,
        *,
        gpu_memory_utilization: float | None = None,
    ) -> None:
        self._cfg = cfg
        self._profile = profile
        self._tokenizer = self._load_tokenizer(profile.tokenizer_name)
        self._engine = self._build_engine(
            cfg,
            profile,
            gpu_memory_utilization=gpu_memory_utilization
            or cfg.resolve_gpu_memory_utilization(1),
        )
        logger.info(
            "OrpheusEngine initialised – id=%s model=%s",
            profile.id,
            profile.model_name,
        )

    # ── public API ────────────────────────────────────────────────

    @property
    def profile(self) -> ModelProfile:
        return self._profile

    @property
    def is_ready(self) -> bool:
        return self._engine is not None

    async def generate_tokens(
        self,
        text: str,
        voice: str,
        *,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        repetition_penalty: float | None = None,
        request_id: str | None = None,
    ) -> AsyncIterator[str]:
        """Yield token-text deltas (e.g. ``<custom_token_10010>``) as they
        are produced by the model.  The caller is expected to feed these into
        :pymethod:`SNACDecoder.decode_stream`."""
        cfg = self._cfg
        prompt_ids = self._format_prompt_ids(text, voice)

        sampling = SamplingParams(
            temperature=temperature if temperature is not None else cfg.temperature,
            top_p=top_p if top_p is not None else cfg.top_p,
            max_tokens=max_tokens if max_tokens is not None else cfg.max_tokens,
            repetition_penalty=repetition_penalty if repetition_penalty is not None else cfg.repetition_penalty,
            stop_token_ids=cfg.stop_token_id_list,
        )

        request_id = request_id or uuid.uuid4().hex
        prev_len = 0

        async for output in self._engine.generate(
            {"prompt_token_ids": prompt_ids},
            sampling_params=sampling,
            request_id=request_id,
        ):
            new_text = output.outputs[0].text
            delta = new_text[prev_len:]
            prev_len = len(new_text)
            if delta:
                yield delta

    # ── prompt formatting ─────────────────────────────────────────

    def _format_prompt_ids(self, text: str, voice: str) -> list[int]:
        """Build the Orpheus prompt as token IDs (passed directly to vLLM).

        Layout: ``[START_PROMPT] {voice}: {text} [EOT][AUDIO_1][AUDIO_2][START_AUDIO]``
        The ``{voice}:`` prefix is omitted for single-speaker checkpoints.
        """
        raw = f"{voice}: {text}" if voice else text
        input_ids = self._tokenizer.encode(raw, add_special_tokens=False)

        return (
            [_TOKEN_START_OF_PROMPT]
            + input_ids
            + [_TOKEN_EOT, _TOKEN_AUDIO_PREFIX_1, _TOKEN_AUDIO_PREFIX_2, _TOKEN_START_OF_AUDIO]
        )

    # ── factory helpers ───────────────────────────────────────────

    @staticmethod
    def _load_tokenizer(name: str) -> AutoTokenizer:
        logger.info("Loading tokenizer: %s", name)
        return AutoTokenizer.from_pretrained(name)

    @staticmethod
    def _build_engine(
        cfg: Settings,
        profile: ModelProfile,
        *,
        gpu_memory_utilization: float,
    ) -> AsyncLLMEngine:
        logger.info(
            "Building vLLM engine – id=%s model=%s dtype=%s max_model_len=%d gpu_mem=%.2f",
            profile.id,
            profile.model_name,
            cfg.dtype,
            cfg.max_model_len,
            gpu_memory_utilization,
        )
        hf_overrides = OrpheusEngine._vocab_pad_overrides(
            profile.model_name,
            multiple=cfg.pad_vocab_to_multiple,
        )
        args = AsyncEngineArgs(
            model=profile.model_name,
            tokenizer=profile.tokenizer_name,
            dtype=cfg.dtype,
            max_model_len=cfg.max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            max_num_seqs=cfg.max_num_seqs,
            max_num_batched_tokens=cfg.max_num_batched_tokens,
            enable_chunked_prefill=cfg.enable_chunked_prefill,
            enable_prefix_caching=cfg.enable_prefix_caching,
            block_size=cfg.block_size,
            enforce_eager=cfg.enforce_eager,
            hf_overrides=hf_overrides,
        )
        return AsyncLLMEngine.from_engine_args(args)

    @staticmethod
    def _vocab_pad_overrides(model_name: str, *, multiple: int) -> dict:
        """Return optional ``hf_overrides`` that pad ``vocab_size``.

        This is disabled by default because recent vLLM releases expect the
        embedding weight shape to exactly match the configured vocab size.
        Some older stacks benefitted from padding misaligned vocab sizes to
        avoid profiling crashes, so we keep it as an explicit opt-in escape
        hatch.
        """
        if multiple <= 1:
            return {}

        model_cfg = AutoConfig.from_pretrained(model_name)
        vocab = model_cfg.vocab_size
        remainder = vocab % multiple
        if remainder == 0:
            return {}
        padded = vocab + (multiple - remainder)
        logger.warning(
            "Model vocab_size=%d is not a multiple of %d – padding to %d "
            "via hf_overrides. Enable this only if your vLLM stack needs the "
            "compatibility workaround and your checkpoint still loads cleanly.",
            vocab,
            multiple,
            padded,
        )
        return {"vocab_size": padded}
