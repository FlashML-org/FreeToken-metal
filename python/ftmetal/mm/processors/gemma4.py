"""Gemma4 image processor: aspect-ratio-preserving 16x16 patches, pooled 3x3 into soft tokens,
1-D rope.

Port of FreeToken mm/processors/gemma4.py, with the pixel path of transformers'
Gemma4ImageProcessorPil (get_aspect_ratio_preserving_size, convert_image_to_patches) ported
to numpy and Pillow. Divergences: only the real patches are produced, so there is no
padding to the budget to strip; no Gemma4UnifiedMMProcessor (the unified release is not
served).
"""

from __future__ import annotations

import math
import struct

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
    read_processor_config,
    resize_and_rescale,
)
from ftmetal.models.loader import config_bool, config_dict, config_float, config_int

PROCESSOR_CONFIG_FILE = "processor_config.json"
# The only soft-token budgets the image processor accepts.
SOFT_TOKEN_BUDGETS = (70, 140, 280, 560, 1120)


def soft_token_budget(max_tokens: int) -> int:
    """Largest accepted budget within max_tokens."""
    return max(budget for budget in SOFT_TOKEN_BUDGETS if budget <= max_tokens)


def get_aspect_ratio_preserving_size(
    height: int, width: int, patch_size: int, max_patches: int, pooling_kernel_size: int
) -> tuple[int, int]:
    """The largest (height, width) of the image's aspect ratio that patchifies into at most
    max_patches patches, both sides multiples of pooling_kernel_size * patch_size."""
    target_pixels = max_patches * patch_size**2
    factor = math.sqrt(target_pixels / (height * width))
    side_multiple = pooling_kernel_size * patch_size
    floor_height = math.floor(factor * height / side_multiple) * side_multiple
    floor_width = math.floor(factor * width / side_multiple) * side_multiple
    max_side_length = (max_patches // pooling_kernel_size**2) * side_multiple
    if floor_height == 0 and floor_width == 0:
        raise ValueError(
            "Attempting to resize to a 0 x 0 image. Resized height should be divisible by "
            f"pooling_kernel_size * patch_size = {side_multiple}."
        )
    elif floor_height == 0:
        target = (side_multiple, min(math.floor(width / height) * side_multiple, max_side_length))
    elif floor_width == 0:
        target = (min(math.floor(height / width) * side_multiple, max_side_length), side_multiple)
    else:
        target = (floor_height, floor_width)
    if target[0] * target[1] > target_pixels:
        raise ValueError(
            f"Resizing [{height}x{width}] to [{target[0]}x{target[1]}] "
            f"but this exceeds {max_patches} patches with patch_size {patch_size}"
        )
    return target


class Gemma4MMProcessor(MMProcessor):
    def __init__(self, config: dict[str, object], model_path: str, mm: MultimodalConfig) -> None:
        if mm.image_max_tokens is not None and mm.image_max_tokens < SOFT_TOKEN_BUDGETS[0]:
            raise ValueError(
                f"--image-max-tokens {mm.image_max_tokens} is below the smallest soft-token "
                f"budget this image processor accepts ({SOFT_TOKEN_BUDGETS[0]})"
            )
        self.image_token_id = config_int(config, "image_token_id")
        self.boi_token_id = config_int(config, "boi_token_id")
        self.eoi_token_id = config_int(config, "eoi_token_id")
        self.placeholder = [self.image_token_id]

        processor_config = read_processor_config(model_path, PROCESSOR_CONFIG_FILE)
        image_processor = config_dict(processor_config, "image_processor")
        assert image_processor is not None, f"{PROCESSOR_CONFIG_FILE} has no image_processor"
        self.patch_size = config_int(image_processor, "patch_size")
        self.pooling_kernel_size = config_int(image_processor, "pooling_kernel_size")
        self.resample = Image.Resampling(config_int(image_processor, "resample"))
        self.rescale_factor = config_float(image_processor, "rescale_factor")
        self.normalization = (
            (
                channel_vector(image_processor, "image_mean"),
                channel_vector(image_processor, "image_std"),
            )
            if config_bool(image_processor, "do_normalize", False)
            else None
        )
        # Every image is scaled to its budget as far as the aspect ratio allows, so a lower
        # bound has no effect.
        if mm.image_max_tokens is None:
            self.max_soft_tokens = config_int(image_processor, "max_soft_tokens")
        else:
            self.max_soft_tokens = soft_token_budget(mm.image_max_tokens)

    def process(self, images: list[Image.Image]) -> list[MMItem]:
        patch = self.patch_size
        pooled_patches = self.pooling_kernel_size**2
        items: list[MMItem] = []
        for image in images:
            height, width = get_aspect_ratio_preserving_size(
                image.height,
                image.width,
                patch,
                self.max_soft_tokens * pooled_patches,
                self.pooling_kernel_size,
            )
            pixels = resize_and_rescale(
                image, height, width, self.resample, self.rescale_factor, self.normalization
            )
            channels = pixels.shape[0]
            rows, cols = height // patch, width // patch
            # [rows * cols, patch * patch * channels], patches in row-major order.
            feature = (
                pixels.reshape(channels, rows, patch, cols, patch)
                .transpose(1, 3, 2, 4, 0)
                .reshape(rows * cols, patch * patch * channels)
            )
            # Each patch's (x, y).
            grid_x, grid_y = np.meshgrid(
                np.arange(cols, dtype=np.int32), np.arange(rows, dtype=np.int32), indexing="xy"
            )
            position_ids = np.stack([grid_x, grid_y], axis=-1).reshape(rows * cols, 2)
            num_soft_tokens = rows * cols // pooled_patches
            item_hash = content_hash(
                feature, position_ids.tobytes() + struct.pack("<i", num_soft_tokens)
            )
            items.append(
                MMItem(
                    modality="image",
                    hash=item_hash,
                    pad_value=mm_pad_value(item_hash),
                    offsets=[],
                    feature=feature,
                    position_ids=position_ids,
                    num_soft_tokens=num_soft_tokens,
                )
            )
        return items

    def dummy_items(self) -> list[MMItem]:
        kernel = self.pooling_kernel_size
        grid_x, grid_y = np.meshgrid(
            np.arange(kernel, dtype=np.int32), np.arange(kernel, dtype=np.int32), indexing="xy"
        )
        return [
            MMItem(
                modality="image",
                hash=0,
                pad_value=0,
                offsets=[(0, 1)],
                feature=np.zeros(
                    (kernel * kernel, RGB_CHANNELS * self.patch_size**2), dtype=np.float32
                ),
                position_ids=np.stack([grid_x, grid_y], axis=-1).reshape(kernel * kernel, 2),
                num_soft_tokens=1,
            )
        ]

    @property
    def max_patches(self) -> int:
        return self.max_soft_tokens * self.pooling_kernel_size**2

    def prompt_replacement(self, item: MMItem) -> PromptReplacement:
        assert item.num_soft_tokens is not None
        full = [self.boi_token_id, *[self.image_token_id] * item.num_soft_tokens, self.eoi_token_id]
        return PromptReplacement.select_token_id(full, self.image_token_id)


__all__ = [
    "SOFT_TOKEN_BUDGETS",
    "Gemma4MMProcessor",
    "get_aspect_ratio_preserving_size",
    "soft_token_budget",
]
