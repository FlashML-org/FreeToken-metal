"""Runtime knobs of the multimodal path; the architecture side lives in the checkpoint config.

Port of FreeToken mm/config.py. Divergences: no embed_cache_device or encoder_weights (the
tower stays resident in unified memory) and no processor_kwargs (no transformers processor to
forward them to).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, get_args

# The inputs an encoder tower serves.
Modality = Literal["image"]
# Encoder tower kinds a family can register; --mm-disable <kind> leaves that tower unbuilt and
# refuses its inputs.
EncoderKind = Literal["vision", "audio"]
ENCODER_KINDS: tuple[EncoderKind, ...] = get_args(EncoderKind)
# Checkpoint config sections that describe encoder towers; the parser never sees the section
# of a tower this process does not build.
ENCODER_SECTIONS = ("vision_config", "audio_config")


@dataclass(frozen=True, kw_only=True)
class MultimodalConfig:
    # Encoder kinds whose tower this process does not build; --text-model-only names them all.
    disabled_encoders: frozenset[EncoderKind] = frozenset()
    # Per-image token budget; the family's MMProcessor converts it to its image processor's
    # own limits. None keeps the checkpoint's, except a max the processor has a default for.
    image_min_tokens: int | None = None
    image_max_tokens: int | None = None

    @property
    def text_model_only(self) -> bool:
        return set(ENCODER_KINDS) <= self.disabled_encoders


__all__ = ["ENCODER_KINDS", "ENCODER_SECTIONS", "EncoderKind", "Modality", "MultimodalConfig"]
