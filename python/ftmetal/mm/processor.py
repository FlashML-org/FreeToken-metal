"""Per-family multimodal processors and the encoders a checkpoint serves; MMProcessor.apply is
the image half of tokenization.

Port of FreeToken mm/processor.py. Divergences: each family preprocesses with its own numpy
and Pillow port of the transformers PIL image processor (resize_and_rescale is their shared
pixel path), reading the checkpoint's processor config once at construction, so there is no
lazily loaded image processor or get_mm_processor_kwargs; items carry numpy features (a
bf16 one as uint16 bits) and positions are int32 numpy; dummy_items takes no dtype or
device; max_patches sizes the encoder's activation reserve; an image that fails to decode
raises ValueError; active_encoders is computed here, for get_mm_processor and EngineConfig
alike, and a checkpoint without config.json or with an unknown architecture serves no
encoder, nor one whose tower weights are missing or stored under another prefix (it is
served text only); an MMProcessor may set default_image_max_tokens, which get_mm_processor
applies when --image-max-tokens is absent (FreeToken keeps the checkpoint's limit).
"""

from __future__ import annotations

import hashlib
import importlib
import io
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from PIL import Image

from ftmetal.mm import check_mm_pad_shift
from ftmetal.mm.config import MultimodalConfig
from ftmetal.mm.item import MMItem
from ftmetal.models.loader import (
    checkpoint_tensor_names,
    config_dict,
    config_int,
    config_str_list,
    read_model_config,
)
from ftmetal.models.register import MODEL_REGISTRY, EncoderSpec, ModelSpec


@dataclass(kw_only=True)
class PromptReplacement:
    """The token sequence that replaces one placeholder; is_embed marks the positions that
    take image embeddings (None: all of them)."""

    full: list[int]
    is_embed: list[bool] | None = None

    @classmethod
    def select_token_id(cls, full: list[int], embed_token_id: int) -> PromptReplacement:
        return cls(full=full, is_embed=[token == embed_token_id for token in full])

    def embed_spans(self) -> list[tuple[int, int]]:
        """Half-open [start, end) runs of embedding positions, relative to the start of full."""
        mask = self.is_embed if self.is_embed is not None else [True] * len(self.full)
        spans: list[tuple[int, int]] = []
        for i, is_embed in enumerate(mask):
            if not is_embed:
                continue
            elif spans and spans[-1][1] == i:
                spans[-1] = (spans[-1][0], i + 1)
            else:
                spans.append((i, i + 1))
        return spans


@dataclass(kw_only=True)
class MMResult:
    input_ids: np.ndarray
    mm_items: list[MMItem]
    mrope_positions: np.ndarray | None
    mrope_delta: int


def find_all(ids: list[int], target: list[int]) -> list[int]:
    """Start index of every non-overlapping occurrence of target."""
    found: list[int] = []
    i = 0
    while i <= len(ids) - len(target):
        if ids[i : i + len(target)] == target:
            found.append(i)
            i += len(target)
        else:
            i += 1
    return found


def read_processor_config(model_path: str, file_name: str) -> dict[str, object]:
    """One of the checkpoint's processor config files."""
    value = json.loads((Path(model_path) / file_name).read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{file_name} of {model_path} is not a JSON object")
    return value


# Every image is converted to RGB before preprocessing.
RGB_CHANNELS = 3


def channel_vector(config: dict[str, object], key: str) -> np.ndarray:
    """A per-channel image processor constant (image_mean, image_std), float32 [channels, 1, 1]."""
    values = config.get(key)
    assert isinstance(values, list) and all(isinstance(value, int | float) for value in values), (
        f"processor config {key!r} must be a list of numbers"
    )
    return np.array(values, dtype=np.float32)[:, None, None]


def resize_and_rescale(
    image: Image.Image,
    height: int,
    width: int,
    resample: Image.Resampling,
    rescale_factor: float,
    normalization: tuple[np.ndarray, np.ndarray] | None,
) -> np.ndarray:
    """float32 [channels, height, width] pixels, bit for bit what the transformers PIL image
    processors compute; normalization is the (mean, std) pair, None to skip it."""
    pixels = np.asarray(image.resize((width, height), resample)).transpose(2, 0, 1)
    # The reference rescales in float64 and rounds once to float32.
    values = (pixels.astype(np.float64) * rescale_factor).astype(np.float32)
    if normalization is None:
        return values
    else:
        mean, std = normalization
        return (values - mean) / std


class MMProcessor(ABC):
    """How a checkpoint's images become items, tokens and positions; apply is the
    family-agnostic driver, subclasses fill the hooks.

    Built from the checkpoint files alone and stateless afterwards, so the tokenizer worker
    and the frontend share one. Preprocessing stays one fixed CPU implementation: radix keys
    are content hashes of the features, and resize results differ across backends.
    """

    # The token sequence the chat template renders once per image.
    placeholder: list[int]
    # The --image-max-tokens get_mm_processor applies when the flag is absent; None keeps the
    # checkpoint's limit.
    default_image_max_tokens: int | None = None

    @abstractmethod
    def __init__(self, config: dict[str, object], model_path: str, mm: MultimodalConfig) -> None:
        """Token ids from config.json, pixel parameters from the processor config file and
        the token budget from mm; refuses a budget the family cannot honor."""

    @abstractmethod
    def process(self, images: list[Image.Image]) -> list[MMItem]:
        """One MMItem per RGB image with feature and hash; offsets are assigned by apply."""

    @abstractmethod
    def prompt_replacement(self, item: MMItem) -> PromptReplacement:
        """The token sequence that stands in for this item's placeholder in the prompt."""

    def positions(self, length: int, items: list[MMItem]) -> tuple[np.ndarray, int] | None:
        """[3, length] t/h/w rope positions and the decode delta; None for 1-D rope families."""
        return None

    @abstractmethod
    def dummy_items(self) -> list[MMItem]:
        """The smallest encodable item of each modality the family serves, for warming the
        encoders."""

    @property
    @abstractmethod
    def max_patches(self) -> int:
        """The feature rows of the largest image the token budget lets through."""

    def apply(self, input_ids: np.ndarray, images: list[bytes]) -> MMResult:
        """Swap each image placeholder for its replacement sequence and precompute rope
        positions."""
        ids: list[int] = input_ids.tolist()
        slots = find_all(ids, self.placeholder)
        if len(slots) != len(images):
            raise ValueError(
                f"prompt renders {len(slots)} image placeholders but the request "
                f"carries {len(images)} images"
            )

        rgb_images: list[Image.Image] = []
        for raw in images:
            # Pillow's decoders raise more than OSError on malformed bytes; any of it is this
            # request's failure, which the tokenizer worker reports instead of dying.
            try:
                rgb_images.append(Image.open(io.BytesIO(raw)).convert("RGB"))
            except Exception as exc:
                raise ValueError(f"could not decode image: {exc}") from exc
        items = self.process(rgb_images)
        expanded: list[int] = []
        cursor = 0
        for slot, item in zip(slots, items, strict=True):
            replacement = self.prompt_replacement(item)
            expanded.extend(ids[cursor:slot])
            base = len(expanded)
            full = list(replacement.full)
            spans = replacement.embed_spans()
            # Embedding slots carry the content pad id, so radix keys and the model's scatter
            # mask see the image.
            for start, end in spans:
                full[start:end] = [item.pad_value] * (end - start)
            expanded.extend(full)
            item.offsets = [(base + start, base + end) for start, end in spans]
            item.validate()
            cursor = slot + len(self.placeholder)
        expanded.extend(ids[cursor:])

        mrope_positions, mrope_delta = self.positions(len(expanded), items) or (None, 0)
        return MMResult(
            input_ids=np.array(expanded, dtype=input_ids.dtype),
            mm_items=items,
            mrope_positions=mrope_positions,
            mrope_delta=mrope_delta,
        )


def image_positions(length: int, blocks: list[tuple[int, int, int, int]]) -> tuple[np.ndarray, int]:
    """t/h/w positions: text advances all three rows, an image freezes t and spreads h/w over
    its grid, then the running position jumps by max(h, w).

    blocks are (offset, n_tokens, grid_h, grid_w) in prompt order; delta shifts decode
    positions (sequence index + delta)."""
    positions = np.empty((3, length), dtype=np.int32)
    start = 0
    cursor = 0
    for offset, num_tokens, grid_rows, grid_cols in blocks:
        positions[:, cursor:offset] = np.arange(start, start + offset - cursor, dtype=np.int32)
        start += offset - cursor
        index = np.arange(num_tokens, dtype=np.int32)
        positions[0, offset : offset + num_tokens] = start
        positions[1, offset : offset + num_tokens] = start + index // grid_cols
        positions[2, offset : offset + num_tokens] = start + index % grid_cols
        start += max(grid_rows, grid_cols)
        cursor = offset + num_tokens
    positions[:, cursor:] = np.arange(start, start + length - cursor, dtype=np.int32)
    delta = int(positions.max()) + 1 - length
    return positions, delta


def content_hash(feature: np.ndarray, extra: bytes = b"") -> int:
    """62-bit hash of the feature bytes plus family extras (e.g. the grid, so transposed
    images differ)."""
    hasher = hashlib.sha256(np.ascontiguousarray(feature).data)
    hasher.update(extra)
    return int.from_bytes(hasher.digest()[:8], "little") % (1 << 62)


def architecture_spec(config: dict[str, object]) -> ModelSpec | None:
    """The registry entry of the checkpoint config's architecture; None for an unknown one."""
    architectures = config_str_list(config, "architectures")
    return MODEL_REGISTRY.get(architectures[0]) if architectures else None


def active_encoders(
    model_path: str, config: dict[str, object], mm: MultimodalConfig
) -> tuple[EncoderSpec, ...]:
    """The encoder towers this process builds: the family registers them, the checkpoint
    config carries their section and its weights their tensors, --mm-disable did not name
    them. A directory without weights (the tokenizer and the processor need none) is judged
    by its config."""
    spec = architecture_spec(config)
    if spec is None:
        return ()
    else:
        has_weights = any(Path(model_path).glob("*.safetensors"))
        tensor_names = checkpoint_tensor_names(model_path) if has_weights else []
        return tuple(
            encoder
            for encoder in spec.encoders
            if config.get(encoder.config_key) is not None
            and encoder.kind not in mm.disabled_encoders
            and (
                not has_weights
                or any(name.startswith(encoder.weight_prefix) for name in tensor_names)
            )
        )


def get_mm_processor(model_path: str, mm: MultimodalConfig) -> MMProcessor | None:
    """A processor for the checkpoint under the mm knobs, or None when no encoder is left to
    serve; callers share the instance. Without --image-max-tokens the processor's default
    applies."""
    has_config = (Path(model_path) / "config.json").is_file()
    config = read_model_config(model_path) if has_config else {}
    spec = architecture_spec(config)
    if spec is None or spec.mm_processor is None or not active_encoders(model_path, config, mm):
        return None
    else:
        check_mm_pad_shift(config_int(config_dict(config, "text_config") or config, "vocab_size"))
        module_name, _, class_name = spec.mm_processor.partition(":")
        processor_class = getattr(importlib.import_module(module_name), class_name)
        assert issubclass(processor_class, MMProcessor)
        default_max_tokens = processor_class.default_image_max_tokens
        if mm.image_max_tokens is not None or default_max_tokens is None:
            budget = mm
        elif mm.image_min_tokens is not None and mm.image_min_tokens > default_max_tokens:
            raise ValueError(
                f"--image-min-tokens {mm.image_min_tokens} exceeds this image processor's "
                f"default --image-max-tokens {default_max_tokens}; pass --image-max-tokens"
            )
        else:
            budget = replace(mm, image_max_tokens=default_max_tokens)
        return processor_class(config, model_path, budget)


__all__ = [
    "RGB_CHANNELS",
    "MMProcessor",
    "MMResult",
    "PromptReplacement",
    "active_encoders",
    "architecture_spec",
    "channel_vector",
    "content_hash",
    "find_all",
    "get_mm_processor",
    "image_positions",
    "read_processor_config",
    "resize_and_rescale",
]
