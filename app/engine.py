from __future__ import annotations

import logging
import uuid
from typing import AsyncIterator

import torch
from transformers import AutoTokenizer
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine

from .config import Settings

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

    All heavy lifting (continuous batching, PagedAttention, prefix
    caching, chunked prefill …) is handled by vLLM under the hood.
    """

    def __init__(self, cfg: Settings) -> None:
        self._cfg = cfg
        self._tokenizer = self._load_tokenizer(cfg.tokenizer_name)
        self._engine = self._build_engine(cfg)
        logger.info("OrpheusEngine initialised – model: %s", cfg.model_name)

    # ── public API ────────────────────────────────────────────────

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
    ) -> AsyncIterator[str]:
        """Yield token-text deltas (e.g. ``<custom_token_10010>``) as they
        are produced by the model.  The caller is expected to feed these into
        :pymethod:`SNACDecoder.decode_stream`."""
        cfg = self._cfg
        prompt = self._format_prompt(text, voice)

        sampling = SamplingParams(
            temperature=temperature if temperature is not None else cfg.temperature,
            top_p=top_p if top_p is not None else cfg.top_p,
            max_tokens=max_tokens if max_tokens is not None else cfg.max_tokens,
            repetition_penalty=repetition_penalty if repetition_penalty is not None else cfg.repetition_penalty,
            stop_token_ids=cfg.stop_token_id_list,
        )

        request_id = uuid.uuid4().hex
        prev_len = 0

        async for output in self._engine.generate(
            prompt=prompt,
            sampling_params=sampling,
            request_id=request_id,
        ):
            new_text = output.outputs[0].text
            delta = new_text[prev_len:]
            prev_len = len(new_text)
            if delta:
                yield delta

    # ── prompt formatting ─────────────────────────────────────────

    def _format_prompt(self, text: str, voice: str) -> str:
        """Build the Orpheus prompt string with special control tokens.

        Layout: ``[START_PROMPT] {voice}: {text} [EOT][AUDIO_1][AUDIO_2][START_AUDIO]``
        """
        raw = f"{voice}: {text}" if voice else text
        input_ids = self._tokenizer.encode(raw, add_special_tokens=False)

        full_ids = (
            [_TOKEN_START_OF_PROMPT]
            + input_ids
            + [_TOKEN_EOT, _TOKEN_AUDIO_PREFIX_1, _TOKEN_AUDIO_PREFIX_2, _TOKEN_START_OF_AUDIO]
        )
        return self._tokenizer.decode(full_ids)

    # ── factory helpers ───────────────────────────────────────────

    @staticmethod
    def _load_tokenizer(name: str) -> AutoTokenizer:
        logger.info("Loading tokenizer: %s", name)
        return AutoTokenizer.from_pretrained(name)

    @staticmethod
    def _build_engine(cfg: Settings) -> AsyncLLMEngine:
        logger.info(
            "Building vLLM engine – model=%s  dtype=%s  max_model_len=%d",
            cfg.model_name,
            cfg.dtype,
            cfg.max_model_len,
        )
        args = AsyncEngineArgs(
            model=cfg.model_name,
            tokenizer=cfg.tokenizer_name,
            dtype=cfg.dtype,
            max_model_len=cfg.max_model_len,
            gpu_memory_utilization=cfg.gpu_memory_utilization,
            max_num_seqs=cfg.max_num_seqs,
            max_num_batched_tokens=cfg.max_num_batched_tokens,
            enable_chunked_prefill=cfg.enable_chunked_prefill,
            enable_prefix_caching=cfg.enable_prefix_caching,
            block_size=cfg.block_size,
        )
        return AsyncLLMEngine.from_engine_args(args)
