from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, replace
from typing import Iterable

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelProfile:
    """Static description of an Orpheus-compatible TTS model.

    A profile carries everything that is model-specific: the HF repo ids,
    whether it is a multi-speaker checkpoint, the list of voices it was
    trained on, and a language tag used purely for UI labelling.
    """

    id: str
    display_name: str
    model_name: str
    tokenizer_name: str
    voices: tuple[str, ...] = field(default_factory=tuple)
    default_voice: str | None = None
    language: str = "en"
    description: str = ""

    @property
    def is_multi_speaker(self) -> bool:
        return bool(self.voices)

    def resolve_voice(self, voice: str | None) -> str:
        """Pick the voice to actually send to the model.

        - Single-speaker profiles always return an empty string (no prefix).
        - Multi-speaker profiles fall back to ``default_voice`` when the
          caller omits one.
        """
        if not self.is_multi_speaker:
            return ""
        if voice:
            return voice
        return self.default_voice or self.voices[0]

    def validate_voice(self, voice: str | None) -> None:
        if not self.is_multi_speaker:
            return
        resolved = voice or self.default_voice
        if resolved not in self.voices:
            raise ValueError(
                f"Unknown voice '{voice}' for model '{self.id}'. "
                f"Available: {list(self.voices)}"
            )


# Built-in profiles. Additional profiles may be loaded from a JSON file via
# the ``MODELS_FILE`` setting, or overridden piecewise via env vars.
_BUILTIN_PROFILES: tuple[ModelProfile, ...] = (
    ModelProfile(
        id="orpheus-en",
        display_name="Orpheus 3B (English, multi-speaker)",
        model_name="canopylabs/orpheus-3b-0.1-ft",
        tokenizer_name="canopylabs/orpheus-3b-0.1-pretrained",
        voices=("tara", "zoe", "jess", "zac", "leo", "mia", "julia", "leah"),
        default_voice="tara",
        language="en",
        description="Canopy Labs Orpheus 3B fine-tune with 8 English voices.",
    ),
    ModelProfile(
        id="orpheus-tr",
        display_name="Orpheus 3B (Turkish, Unsloth fine-tune)",
        model_name="yaltay/tmp_tmp_smp",
        tokenizer_name="yaltay/tmp_tmp_smp",
        voices=(),
        default_voice=None,
        language="tr",
        description=(
            "Unsloth Turkish fine-tune of Orpheus 3B. Single-speaker checkpoint "
            "– the voice field is ignored for this model."
        ),
    ),
)


class ModelRegistry:
    """In-memory registry of :class:`ModelProfile` objects.

    Resolution order when looking up by id:
    1. overrides loaded from ``MODELS_FILE`` (JSON)
    2. built-in profiles

    The registry is intentionally a simple, read-mostly object – engines and
    routing logic treat it as a source of truth for what models exist.
    """

    def __init__(self, profiles: Iterable[ModelProfile]) -> None:
        self._profiles: dict[str, ModelProfile] = {p.id: p for p in profiles}
        if not self._profiles:
            raise ValueError("ModelRegistry requires at least one profile")

    # ── introspection ────────────────────────────────────────────

    def __contains__(self, model_id: str) -> bool:
        return model_id in self._profiles

    def ids(self) -> list[str]:
        return list(self._profiles.keys())

    def all(self) -> list[ModelProfile]:
        return list(self._profiles.values())

    def get(self, model_id: str) -> ModelProfile:
        try:
            return self._profiles[model_id]
        except KeyError as exc:
            raise KeyError(
                f"Unknown model '{model_id}'. Available: {self.ids()}"
            ) from exc

    # ── factories ────────────────────────────────────────────────

    @classmethod
    def from_config(
        cls,
        enabled_ids: Iterable[str],
        *,
        overrides_file: str | None = None,
    ) -> "ModelRegistry":
        """Build a registry from the built-ins, optionally merging a JSON
        override file and then filtering to the explicitly enabled ids.
        """
        profiles: dict[str, ModelProfile] = {p.id: p for p in _BUILTIN_PROFILES}

        if overrides_file:
            try:
                with open(overrides_file, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
            except FileNotFoundError:
                logger.warning("MODELS_FILE '%s' not found – ignoring", overrides_file)
                data = []
            else:
                for raw in data:
                    profile = _profile_from_dict(raw, existing=profiles.get(raw.get("id", "")))
                    profiles[profile.id] = profile

        enabled = [mid.strip() for mid in enabled_ids if mid.strip()]
        missing = [mid for mid in enabled if mid not in profiles]
        if missing:
            raise ValueError(
                f"Enabled models not found in registry: {missing}. "
                f"Known: {list(profiles.keys())}"
            )

        chosen = [profiles[mid] for mid in enabled] if enabled else list(profiles.values())
        return cls(chosen)


def _profile_from_dict(
    raw: dict, *, existing: ModelProfile | None = None
) -> ModelProfile:
    if not raw.get("id"):
        raise ValueError(f"Model profile missing 'id': {raw}")

    voices_raw = raw.get("voices", existing.voices if existing else ())
    voices = tuple(voices_raw) if voices_raw is not None else ()

    base = existing or ModelProfile(
        id=raw["id"],
        display_name=raw.get("display_name", raw["id"]),
        model_name=raw["model_name"],
        tokenizer_name=raw.get("tokenizer_name", raw["model_name"]),
    )

    return replace(
        base,
        id=raw["id"],
        display_name=raw.get("display_name", base.display_name),
        model_name=raw.get("model_name", base.model_name),
        tokenizer_name=raw.get("tokenizer_name", base.tokenizer_name),
        voices=voices,
        default_voice=raw.get("default_voice", base.default_voice),
        language=raw.get("language", base.language),
        description=raw.get("description", base.description),
    )
