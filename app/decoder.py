from __future__ import annotations

import logging
from typing import AsyncIterator

import numpy as np
import torch
from snac import SNAC

logger = logging.getLogger(__name__)

# Limit internal torch parallelism so it doesn't fight with the event loop.
torch.set_num_threads(1)

_CUSTOM_TOKEN_PREFIX = "<custom_token_"
_CUSTOM_TOKEN_SUFFIX = ">"


class SNACDecoder:
    """Decodes Orpheus audio-codec tokens into 24 kHz 16-bit PCM audio.

    The SNAC model is loaded once at startup, warmed up on the target device,
    and then reused across all requests.  All tensor operations inside
    :meth:`decode_tokens` use pre-allocated shapes and run under
    ``torch.inference_mode`` to minimise overhead.
    """

    SAMPLE_RATE: int = 24000
    BIT_DEPTH: int = 16
    CHANNELS: int = 1
    TOKENS_PER_FRAME: int = 7

    def __init__(self, model_name: str, device: str = "") -> None:
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        logger.info("Loading SNAC model %s on %s …", model_name, self._device)
        self._model: SNAC = SNAC.from_pretrained(model_name).eval().to(self._device)

        if self._device == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            self._warmup()

        logger.info("SNAC decoder ready on %s", self._device)

    # ── public API ────────────────────────────────────────────────

    # The SNAC decoder's causal convolutions produce warm-up artefacts
    # in the first ~2048 samples.  We skip them and take everything after.
    _WARMUP_SAMPLES: int = 2048

    def decode_tokens(self, token_values: list[int]) -> bytes | None:
        """Convert a list of SNAC code values (must be a multiple of 7) to
        raw 16-bit PCM bytes.  Returns ``None`` when the input is too short,
        contains out-of-range codes, or the SNAC output has no usable audio
        past the warm-up region."""
        n = len(token_values)
        if n < self.TOKENS_PER_FRAME:
            return None

        num_frames = n // self.TOKENS_PER_FRAME
        dev = self._device

        codes_0 = torch.empty((1, num_frames), dtype=torch.int32, device=dev)
        codes_1 = torch.empty((1, num_frames * 2), dtype=torch.int32, device=dev)
        codes_2 = torch.empty((1, num_frames * 4), dtype=torch.int32, device=dev)

        for i in range(num_frames):
            b = i * 7
            codes_0[0, i] = token_values[b]
            codes_1[0, i * 2] = token_values[b + 1]
            codes_1[0, i * 2 + 1] = token_values[b + 4]
            codes_2[0, i * 4] = token_values[b + 2]
            codes_2[0, i * 4 + 1] = token_values[b + 3]
            codes_2[0, i * 4 + 2] = token_values[b + 5]
            codes_2[0, i * 4 + 3] = token_values[b + 6]

        if (
            torch.any(codes_0 < 0) or torch.any(codes_0 > 4096)
            or torch.any(codes_1 < 0) or torch.any(codes_1 > 4096)
            or torch.any(codes_2 < 0) or torch.any(codes_2 > 4096)
        ):
            return None

        codes = [codes_0, codes_1, codes_2]

        with torch.inference_mode():
            audio_hat = self._model.decode(codes)
            total_samples = audio_hat.shape[-1]

            if total_samples <= self._WARMUP_SAMPLES:
                return None

            audio_slice = audio_hat[:, :, self._WARMUP_SAMPLES:]

            if self._device == "cuda":
                pcm = (audio_slice * 32767.0).round().to(torch.int16)
                return pcm.cpu().numpy().tobytes()

            arr = audio_slice.detach().numpy()
            return (arr * 32767.0).round().astype(np.int16).tobytes()

    async def decode_stream(
        self,
        token_text_gen: AsyncIterator[str],
        *,
        min_frames_first: int = 2,
        min_frames_subsequent: int = 4,
    ) -> AsyncIterator[bytes]:
        """Async generator: consumes ``<custom_token_…>`` text deltas and
        yields PCM audio chunks as soon as enough tokens accumulate.

        *min_frames_first* controls the minimum complete frames (×7 tokens)
        before attempting the **first** audio chunk.  The actual first chunk
        may use more frames if the SNAC output is still too short.
        *min_frames_subsequent* sets the sliding-window size for every
        chunk after that.
        """
        buffer: list[int] = []
        count = 0
        first_chunk_sent = False
        first_threshold = min_frames_first * self.TOKENS_PER_FRAME
        window_size = min_frames_subsequent * self.TOKENS_PER_FRAME

        async for text_delta in token_text_gen:
            for raw_number in self._parse_all_tokens(text_delta):
                code = raw_number - 10 - ((count % 7) * 4096)
                if code < 0:
                    continue

                buffer.append(code)
                count += 1

                if count % self.TOKENS_PER_FRAME != 0:
                    continue

                if not first_chunk_sent:
                    if count < first_threshold:
                        continue
                    audio = self.decode_tokens(buffer)
                    if audio:
                        first_chunk_sent = True
                        logger.debug(
                            "First audio chunk: %d bytes from %d tokens (%d frames)",
                            len(audio), count, count // self.TOKENS_PER_FRAME,
                        )
                        yield audio
                else:
                    audio = self.decode_tokens(buffer[-window_size:])
                    if audio:
                        yield audio

        if buffer and not first_chunk_sent:
            logger.warning("Stream ended with %d tokens but no audio was emitted", count)

    # ── internals ─────────────────────────────────────────────────

    @staticmethod
    def _parse_all_tokens(text: str) -> list[int]:
        """Extract all ``<custom_token_N>`` numbers from a text delta.

        A single vLLM delta may contain more than one token when the
        event-loop batches updates, so we must capture every occurrence.
        """
        results: list[int] = []
        search_from = 0
        while True:
            start = text.find(_CUSTOM_TOKEN_PREFIX, search_from)
            if start == -1:
                break
            end = text.find(_CUSTOM_TOKEN_SUFFIX, start + len(_CUSTOM_TOKEN_PREFIX))
            if end == -1:
                break
            try:
                number = int(text[start + len(_CUSTOM_TOKEN_PREFIX) : end])
                results.append(number)
            except (ValueError, TypeError):
                pass
            search_from = end + len(_CUSTOM_TOKEN_SUFFIX)
        return results

    def _warmup(self) -> None:
        dummy = [
            torch.randint(0, 4096, (1, 1), dtype=torch.int32, device=self._device),
            torch.randint(0, 4096, (1, 2), dtype=torch.int32, device=self._device),
            torch.randint(0, 4096, (1, 4), dtype=torch.int32, device=self._device),
        ]
        with torch.inference_mode():
            _ = self._model.decode(dummy)
