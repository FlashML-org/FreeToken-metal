"""The Qwen VL and Gemma4 pixel paths against the transformers PIL image processors, bit for
bit (Qwen VL's in bf16), on images built in memory and decoded the way FreeToken decodes them.

The image processor configs are the real checkpoints': the checkpoint directories when
FTMETAL_TEST_QWEN_VL_PATH / FTMETAL_TEST_GEMMA4_PATH name them, copies of their files
otherwise.
"""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Literal

import mlx.core as mx
import numpy as np
import pytest
from PIL import Image

from ftmetal.mm.config import MultimodalConfig
from ftmetal.mm.processors.gemma4 import Gemma4MMProcessor, soft_token_budget
from ftmetal.mm.processors.qwen_vl import QwenVLMMProcessor
from ftmetal.models.loader import read_model_config
from tiny_vl_model import write_gemma4_model, write_qwen_vl_model

qwen_reference = pytest.importorskip("transformers.models.qwen2_vl.image_processing_pil_qwen2_vl")
gemma_reference = pytest.importorskip("transformers.models.gemma4.image_processing_pil_gemma4")

QWEN_VL_PATH = os.environ.get("FTMETAL_TEST_QWEN_VL_PATH")
GEMMA4_PATH = os.environ.get("FTMETAL_TEST_GEMMA4_PATH")
# One Qwen VL token is a 32x32 pixel area: patch 16, merged 2x2.
QWEN_PIXELS_PER_TOKEN = 32 * 32

Family = Literal["qwen", "gemma"]

# (family, width, height, mode, format, --image-max-tokens): every case must match exactly.
PREPROCESS_CASES: list[tuple[Family, int, int, str, str, int | None]] = [
    ("qwen", 512, 512, "RGB", "PNG", None),
    # Smaller than the minimum area: scaled up.
    ("qwen", 20, 12, "RGB", "PNG", None),
    # Sides that are not multiples of 32.
    ("qwen", 333, 777, "RGB", "JPEG", None),
    # Extreme aspect ratio; the short side rounds to zero before the minimum lifts it.
    ("qwen", 3000, 16, "RGB", "PNG", None),
    # Larger than the checkpoint's 16384-token cap, and than an explicit one: scaled down.
    ("qwen", 6000, 3000, "RGB", "JPEG", None),
    ("qwen", 900, 700, "RGB", "PNG", 256),
    ("qwen", 300, 200, "RGBA", "PNG", None),
    ("qwen", 300, 200, "P", "PNG", None),
    ("qwen", 300, 200, "L", "JPEG", None),
    ("qwen", 300, 200, "LA", "PNG", None),
    ("gemma", 512, 512, "RGB", "PNG", None),
    ("gemma", 20, 12, "RGB", "PNG", None),
    ("gemma", 333, 777, "RGB", "JPEG", None),
    # The short side floors to zero: one pooled row, the long side capped.
    ("gemma", 3000, 10, "RGB", "PNG", None),
    ("gemma", 2400, 2000, "RGB", "JPEG", None),
    # A budget below the default picks the 70 soft-token tier.
    ("gemma", 640, 400, "RGB", "PNG", 100),
    ("gemma", 300, 200, "RGBA", "PNG", None),
    ("gemma", 300, 200, "P", "PNG", None),
    ("gemma", 300, 200, "L", "JPEG", None),
    ("gemma", 300, 200, "LA", "PNG", None),
]


def encode_image(width: int, height: int, mode: str, image_format: str) -> bytes:
    """Random pixels in the given mode, with random alpha where the mode has one, encoded."""
    rgba = np.random.default_rng(width * height).integers(0, 256, (height, width, 4), np.uint8)
    source = Image.fromarray(rgba)
    # A palette made from RGBA keeps a transparency table, which Pillow warns about.
    image = source.convert("RGB").convert("P") if mode == "P" else source.convert(mode)
    buffer = io.BytesIO()
    image.save(buffer, format=image_format)
    return buffer.getvalue()


def checkpoint_dir(family: Family, tiny_model_dir: Path) -> Path:
    """The real checkpoint when its environment variable names one, else the tiny copy."""
    if family == "qwen" and QWEN_VL_PATH is not None:
        return Path(QWEN_VL_PATH)
    elif family == "gemma" and GEMMA4_PATH is not None:
        return Path(GEMMA4_PATH)
    elif family == "qwen":
        write_qwen_vl_model(tiny_model_dir)
        return tiny_model_dir
    else:
        write_gemma4_model(tiny_model_dir)
        return tiny_model_dir


def check_against_transformers(
    family: Family, model_dir: Path, raw: bytes, image_max_tokens: int | None
) -> None:
    config = read_model_config(str(model_dir))
    mm = MultimodalConfig(image_max_tokens=image_max_tokens)
    # FreeToken's decode, which the reference processes.
    rgb = Image.open(io.BytesIO(raw)).convert("RGB")
    if family == "qwen":
        processor = QwenVLMMProcessor(config, str(model_dir), mm)
        reference = qwen_reference.Qwen2VLImageProcessorPil.from_pretrained(model_dir)
        if image_max_tokens is None:
            expected = reference(images=rgb)
        else:
            size = {
                "shortest_edge": reference.size.shortest_edge,
                "longest_edge": image_max_tokens * QWEN_PIXELS_PER_TOKEN,
            }
            expected = reference(images=rgb, size=size)
        (item,) = processor.apply(np.array(processor.placeholder, np.int32), [raw]).mm_items
        assert item.grid_thw == tuple(expected["image_grid_thw"][0].tolist())
        # The reference's float32 pixels in bf16, by MLX's own rounding.
        bf16 = mx.array(expected["pixel_values"]).astype(mx.bfloat16, stream=mx.cpu)
        assert item.feature is not None and item.feature.dtype == np.uint16
        assert np.array_equal(item.feature, np.array(bf16.view(mx.uint16)))
    else:
        processor = Gemma4MMProcessor(config, str(model_dir), mm)
        reference = gemma_reference.Gemma4ImageProcessorPil.from_pretrained(model_dir)
        budget = (
            {}
            if image_max_tokens is None
            else {"max_soft_tokens": soft_token_budget(image_max_tokens)}
        )
        expected = reference(images=rgb, **budget)
        positions = expected["image_position_ids"][0]
        valid = (positions != -1).all(axis=-1)
        (item,) = processor.apply(np.array(processor.placeholder, np.int32), [raw]).mm_items
        assert item.num_soft_tokens == expected["num_soft_tokens_per_image"][0]
        assert item.position_ids is not None and item.position_ids.dtype == np.int32
        assert np.array_equal(item.position_ids, positions[valid])
        assert item.feature is not None and item.feature.dtype == np.float32
        assert np.array_equal(item.feature, expected["pixel_values"][0][valid])


@pytest.mark.parametrize(
    ("family", "width", "height", "mode", "image_format", "image_max_tokens"), PREPROCESS_CASES
)
def test_pixels_match_the_transformers_processor_exactly(
    tiny_model_dir: Path,
    family: Family,
    width: int,
    height: int,
    mode: str,
    image_format: str,
    image_max_tokens: int | None,
) -> None:
    raw = encode_image(width, height, mode, image_format)
    model_dir = checkpoint_dir(family, tiny_model_dir)
    check_against_transformers(family, model_dir, raw, image_max_tokens)
