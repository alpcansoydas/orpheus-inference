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

    _MIN_VOCAB_SIZE = max(
        _TOKEN_START_OF_PROMPT,
        _TOKEN_AUDIO_PREFIX_1,
        _TOKEN_AUDIO_PREFIX_2,
        _TOKEN_START_OF_AUDIO,
    ) + 1  # 128262 – embedding must cover all Orpheus control tokens

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
            self._tokenizer,
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
        tokenizer: AutoTokenizer,
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
        hf_overrides = OrpheusEngine._compute_hf_overrides(
            profile.model_name,
            tokenizer,
            pad_multiple=cfg.pad_vocab_to_multiple,
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

    # vLLM V1's Triton top-k/top-p sampler kernel requires vocab_size to
    # be aligned to its block stride.  Orpheus checkpoints often have an
    # awkward vocab_size (e.g. 128262 = 2×3×21377) that isn't divisible,
    # causing the kernel to read past the logits tensor.  Padding to 64
    # is the smallest alignment that satisfies current Triton block sizes.
    _VOCAB_PAD_MINIMUM = 64

    @staticmethod
    def _compute_hf_overrides(
        model_name: str,
        tokenizer: AutoTokenizer,
        *,
        pad_multiple: int,
    ) -> dict:
        """Return ``hf_overrides`` ensuring ``vocab_size`` is correct and
        aligned for vLLM's Triton sampling kernels.

        Two problems are handled:

        1. **Too-small vocab** – Some Unsloth/LoRA exports keep the base
           Llama 3 vocab_size (128256) even though the checkpoint includes
           embeddings for the Orpheus control tokens (128257–128261).

        2. **Triton alignment** – vLLM V1's ``_topk_topp_kernel`` processes
           logits in fixed-size blocks.  If ``vocab_size`` is not a multiple
           of the block stride the kernel reads past the tensor and crashes
           with ``CUDA: an illegal memory access``.  We pad to at least 64
           (overridable via ``PAD_VOCAB_TO_MULTIPLE``).
        """
        model_cfg = AutoConfig.from_pretrained(model_name)
        config_vocab = model_cfg.vocab_size

        required = max(
            config_vocab,
            len(tokenizer),
            OrpheusEngine._MIN_VOCAB_SIZE,
        )

        if pad_multiple > 0:
            pad = max(pad_multiple, OrpheusEngine._VOCAB_PAD_MINIMUM)
            remainder = required % pad
            if remainder:
                required += pad - remainder

        if required == config_vocab:
            return {}

        logger.warning(
            "Model config vocab_size=%d → overriding to %d "
            "(pad-to-%d alignment for Triton sampler + Orpheus control tokens).",
            config_vocab,
            required,
            pad,
        )
        return {"vocab_size": required}
