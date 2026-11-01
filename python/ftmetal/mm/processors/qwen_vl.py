"""Qwen VL family image processor (Qwen3.6, Qwen3.8-Flash-Next): patch-grid ViT with a
spatial merger, 3-axis rope positions.

Port of FreeToken mm/processors/qwen_vl.py, with the pixel path of transformers'
Qwen2VLImageProcessorPil (smart_resize, patchify) ported to numpy and Pillow. Divergences:
the patch and merge sizes come from the preprocessor config alone (FreeToken counts tokens
with vision_config's); the bf16 feature is held as its uint16 bit pattern (numpy has no
bf16); images only, so t is always 1 and there is no video check; no image_grid; the default
--image-max-tokens is 4096, not the checkpoint's 16384.
"""

from __future__ import annotations

import math
import struct
from pathlib import Path

import numpy as np
from PIL import Image

from ftmetal.mm import mm_pad_value
from ftmetal.mm.config import MultimodalConfig
from ftmetal.mm.item import MMItem
from ftmetal.mm.processor import (
    RGB_CHANNELS,
    MMProcessor,
    PromptReplacement,
    channel_vector,
    content_hash,
    image_positions,
    read_processor_config,
    resize_and_rescale,
)
from ftmetal.models.loader import config_bool, config_dict, config_float, config_int

PREPROCESSOR_CONFIG_FILE = "preprocessor_config.json"
# A checkpoint without the preprocessor config keeps the image processor's as a section here.
PROCESSOR_CONFIG_FILE = "processor_config.json"
# Qwen2VLImageProcessorPil's values for the keys the checkpoint's preprocessor config omits.
DEFAULT_RESAMPLE = Image.Resampling.BICUBIC
DEFAULT_RESCALE_FACTOR = 1 / 255
# smart_resize refuses images more elongated than this.
MAX_ASPECT_RATIO = 200


def smart_resize(
    height: int, width: int, factor: int, min_pixels: int, max_pixels: int
) -> tuple[int, int]:
    """The resized (height, width): both sides multiples of factor, the area within
    [min_pixels, max_pixels] and the aspect ratio kept as closely as possible."""
    aspect_ratio = max(height, width) / min(height, width)
    if aspect_ratio > MAX_ASPECT_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_ASPECT_RATIO}, got {aspect_ratio}"
        )
    rounded_height = round(height / factor) * factor
    rounded_width = round(width / factor) * factor
    if rounded_height * rounded_width > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        return (
            max(factor, math.floor(height / beta / factor) * factor),
            max(factor, math.floor(width / beta / factor) * factor),
        )
    elif rounded_height * rounded_width < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        return (
            math.ceil(height * beta / factor) * factor,
            math.ceil(width * beta / factor) * factor,
        )
    else:
        return rounded_height, rounded_width


class QwenVLMMProcessor(MMProcessor):
    # Measured on an M4 Pro, Qwen3.6's tower takes 7 s and a 0.6 GiB reserve for a 4096-token
    # image (2048x2048), 98 s and 2.1 GiB at the checkpoint's 16384 (4096x4096).
    default_image_max_tokens = 4096

    def __init__(self, config: dict[str, object], model_path: str, mm: MultimodalConfig) -> None:
        text_config = config_dict(config, "text_config")
        assert text_config is not None, "a Qwen VL config.json has a text_config"
        self.image_token_id = config_int(config, "image_token_id")
        self.placeholder = [self.image_token_id]
        self.is_mrope = "mrope_section" in (config_dict(text_config, "rope_parameters") or {})

        if (Path(model_path) / PREPROCESSOR_CONFIG_FILE).is_file():
            preprocessor = read_processor_config(model_path, PREPROCESSOR_CONFIG_FILE)
        else:
            processor_config = read_processor_config(model_path, PROCESSOR_CONFIG_FILE)
            image_processor = config_dict(processor_config, "image_processor")
            if image_processor is None:
                raise ValueError(f"{PROCESSOR_CONFIG_FILE} of {model_path} has no image_processor")
            preprocessor = image_processor
        size = config_dict(preprocessor, "size")
        assert size is not None, "the image processor config has no size"
        self.patch_size = config_int(preprocessor, "patch_size")
        self.temporal_patch_size = config_int(preprocessor, "temporal_patch_size")
        self.merge_size = config_int(preprocessor, "merge_size")
        pixels_per_token = (self.patch_size * self.merge_size) ** 2
        self.resample = Image.Resampling(config_int(preprocessor, "resample", DEFAULT_RESAMPLE))
        self.rescale_factor = config_float(preprocessor, "rescale_factor", DEFAULT_RESCALE_FACTOR)
        self.normalization = (
            (channel_vector(preprocessor, "image_mean"), channel_vector(preprocessor, "image_std"))
            if config_bool(preprocessor, "do_normalize", True)
            else None
        )
        # The budget travels as pixel areas, one token per (patch * merge)^2 pixels; None keeps
        # the checkpoint's limit.
        if mm.image_min_tokens is None:
            self.min_pixels = config_int(size, "shortest_edge")
        else:
            self.min_pixels = mm.image_min_tokens * pixels_per_token
        if mm.image_max_tokens is None:
            self.max_pixels = config_int(size, "longest_edge")
        else:
            self.max_pixels = mm.image_max_tokens * pixels_per_token

    def process(self, images: list[Image.Image]) -> list[MMItem]:
        patch = self.patch_size
        merge = self.merge_size
        items: list[MMItem] = []
        for image in images:
            height, width = smart_resize(
                image.height, image.width, patch * merge, self.min_pixels, self.max_pixels
            )
            pixels = resize_and_rescale(
                image, height, width, self.resample, self.rescale_factor, self.normalization
            )
            # bf16 rounded to nearest even, as the tower's first cast would round the float32.
            bits = pixels.view(np.uint32)
            pixels = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype(np.uint16)
            channels = pixels.shape[0]
            grid_h, grid_w = height // patch, width // patch
            # [grid_h / merge, grid_w / merge, merge, merge, channels, patch, patch]: patches in
            # merge-block order, as the tower's merger reads them.
            patches = pixels.reshape(
                channels, grid_h // merge, merge, patch, grid_w // merge, merge, patch
            ).transpose(1, 4, 2, 5, 0, 3, 6)
            # One image is one temporal patch: its frame repeats temporal_patch_size times.
            patches = np.broadcast_to(
                patches[:, :, :, :, :, None],
                (*patches.shape[:5], self.temporal_patch_size, patch, patch),
            )
            feature = patches.reshape(
                grid_h * grid_w, channels * self.temporal_patch_size * patch * patch
            )
            grid_thw = (1, grid_h, grid_w)
            item_hash = content_hash(feature, struct.pack("<3i", *grid_thw))
            items.append(
                MMItem(
                    modality="image",
                    hash=item_hash,
                    pad_value=mm_pad_value(item_hash),
                    offsets=[],
                    feature=feature,
                    grid_thw=grid_thw,
                )
            )
        return items

    def dummy_items(self) -> list[MMItem]:
        merge = self.merge_size
        patch_dim = RGB_CHANNELS * self.temporal_patch_size * self.patch_size**2
        return [
            MMItem(
                modality="image",
                hash=0,
                pad_value=0,
                offsets=[(0, 1)],
                feature=np.zeros((merge * merge, patch_dim), dtype=np.uint16),
                grid_thw=(1, merge, merge),
            )
        ]

    @property
    def max_patches(self) -> int:
        return self.max_pixels // self.patch_size**2

    def prompt_replacement(self, item: MMItem) -> PromptReplacement:
        assert item.grid_thw is not None
        t, h, w = item.grid_thw
        return PromptReplacement(full=[self.image_token_id] * (t * h * w // self.merge_size**2))

    def positions(self, length: int, items: list[MMItem]) -> tuple[np.ndarray, int] | None:
        if not self.is_mrope:
            return None
        else:
            blocks: list[tuple[int, int, int, int]] = []
            for item in items:
                assert item.grid_thw is not None
                ((start, end),) = item.offsets
                _, h, w = item.grid_thw
                blocks.append((start, end - start, h // self.merge_size, w // self.merge_size))
            return image_positions(length, blocks)


__all__ = ["QwenVLMMProcessor", "smart_resize"]
