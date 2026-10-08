"""MMProcessor on tiny Qwen VL and Gemma4 checkpoints: placeholder expansion into content pad
ids, half-open offsets, mrope positions, content hashes, the token budgets and the
architecture lookup."""

from __future__ import annotations

import io
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from PIL import Image

from ftmetal.mm import MM_PAD_SHIFT_VALUE, mm_pad_value
from ftmetal.mm.config import ENCODER_KINDS, MultimodalConfig
from ftmetal.mm.processor import MMProcessor, get_mm_processor, image_positions
from ftmetal.mm.processors.gemma4 import Gemma4MMProcessor
from ftmetal.mm.processors.qwen_vl import QwenVLMMProcessor
from ftmetal.models.loader import read_model_config
from tiny_vl_model import (
    GEMMA4_BOI_TOKEN_ID,
    GEMMA4_EOI_TOKEN_ID,
    GEMMA4_IMAGE_TOKEN_ID,
    register_vision_encoder,
    write_gemma4_model,
    write_qwen_vl_model,
)

PURPLE = (120, 30, 200)


def png(width: int, height: int, color: tuple[int, int, int] = PURPLE) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buffer, format="PNG")
    return buffer.getvalue()


def qwen_processor(model_dir: Path, mm: MultimodalConfig | None = None) -> QwenVLMMProcessor:
    config = read_model_config(str(model_dir))
    return QwenVLMMProcessor(config, str(model_dir), mm or MultimodalConfig())


def gemma_processor(model_dir: Path, mm: MultimodalConfig | None = None) -> Gemma4MMProcessor:
    config = read_model_config(str(model_dir))
    return Gemma4MMProcessor(config, str(model_dir), mm or MultimodalConfig())


def image_tokens(processor: MMProcessor, image: bytes) -> int:
    """The prompt tokens one image takes."""
    ids = np.array(processor.placeholder, np.int32)
    (item,) = processor.apply(ids, [image]).mm_items
    return sum(end - start for start, end in item.offsets)


def test_qwen_image_becomes_one_span_of_its_content_pad_id(tiny_model_dir: Path) -> None:
    image_token_id = write_qwen_vl_model(tiny_model_dir)
    ids = np.array([1, 2, image_token_id, 3], np.int32)
    result = qwen_processor(tiny_model_dir).apply(ids, [png(512, 512)])
    (item,) = result.mm_items
    # 512x512 is a 32x32 patch grid, merged 2x2 into 256 tokens.
    assert item.grid_thw == (1, 32, 32) and item.offsets == [(2, 258)]
    assert item.pad_value == mm_pad_value(item.hash) >= MM_PAD_SHIFT_VALUE
    assert result.input_ids.dtype == np.int32
    assert result.input_ids.tolist() == [1, 2, *[item.pad_value] * 256, 3]


def test_qwen_mrope_positions_spread_the_image_over_its_merged_grid(tiny_model_dir: Path) -> None:
    image_token_id = write_qwen_vl_model(tiny_model_dir)
    ids = np.array([1, 2, image_token_id, 3, 4], np.int32)
    result = qwen_processor(tiny_model_dir).apply(ids, [png(512, 256)])
    positions = result.mrope_positions
    assert positions is not None and positions.dtype == np.int32
    # 512x256 merges into 8 rows of 16 tokens at offsets [2, 130).
    assert positions.shape == (3, 132)
    # Text rows advance all three axes together.
    assert positions[:, :2].tolist() == [[0, 1]] * 3
    assert positions[0, 2:130].tolist() == [2] * 128
    assert positions[1, 2:130].tolist() == [2 + i // 16 for i in range(128)]
    assert positions[2, 2:130].tolist() == [2 + i % 16 for i in range(128)]
    # The text after it resumes past the longer side.
    assert positions[:, 130:].tolist() == [[18, 19]] * 3
    assert result.mrope_delta == int(positions.max()) + 1 - 132 == 19 + 1 - 132


def test_image_positions_freeze_t_and_advance_by_the_longer_side() -> None:
    # [text x2][image 2x3][text x1][image 3x1][text x2]: 14 tokens.
    positions, delta = image_positions(14, [(2, 6, 2, 3), (9, 3, 3, 1)])
    assert positions[:, :2].tolist() == [[0, 1]] * 3
    assert positions[:, 2:8].tolist() == [[2] * 6, [2, 2, 2, 3, 3, 3], [2, 3, 4, 2, 3, 4]]
    assert positions[:, 8].tolist() == [5, 5, 5]
    assert positions[:, 9:12].tolist() == [[6] * 3, [6, 7, 8], [6] * 3]
    assert positions[:, 12:].tolist() == [[9, 10]] * 3
    assert delta == 10 + 1 - 14


def test_same_image_same_hash_and_ids(tiny_model_dir: Path) -> None:
    image_token_id = write_qwen_vl_model(tiny_model_dir)
    processor = qwen_processor(tiny_model_dir)
    ids = np.array([5, image_token_id, 6], np.int32)
    first = processor.apply(ids, [png(64, 64)])
    again = processor.apply(ids, [png(64, 64)])
    recolored = processor.apply(ids, [png(64, 64, (0, 0, 0))])
    assert first.mm_items[0].hash == again.mm_items[0].hash
    assert np.array_equal(first.input_ids, again.input_ids)
    assert recolored.mm_items[0].hash != first.mm_items[0].hash
    assert not np.array_equal(recolored.input_ids, first.input_ids)


# (placeholders in the prompt, images, the error)
REFUSED_INPUT_CASES: list[tuple[int, list[bytes], str]] = [
    # Counted before any image is decoded.
    (2, [b"not an image"], "2 image placeholders but the request carries 1"),
    (0, [png(8, 8)], "0 image placeholders but the request carries 1"),
    (1, [b"not an image"], "could not decode image"),
]


@pytest.mark.parametrize(("placeholders", "images", "error"), REFUSED_INPUT_CASES)
def test_mismatched_or_undecodable_images_raise_value_error(
    tiny_model_dir: Path, placeholders: int, images: list[bytes], error: str
) -> None:
    image_token_id = write_qwen_vl_model(tiny_model_dir)
    ids = np.array([7, *[image_token_id] * placeholders], np.int32)
    with pytest.raises(ValueError, match=error):
        qwen_processor(tiny_model_dir).apply(ids, images)


# (--image-min-tokens, --image-max-tokens, image width, height, the tokens it takes)
QWEN_BUDGET_CASES: list[tuple[int | None, int | None, int, int, int]] = [
    (None, None, 512, 512, 256),
    # The checkpoint's minimum is 64 tokens.
    (None, None, 64, 64, 64),
    # The checkpoint's maximum of 16384 tokens lets a 2400x2000 image through at 1984x2400.
    (None, None, 2400, 2000, 4650),
    (None, 64, 1024, 1024, 64),
    (300, None, 64, 64, 324),
]


@pytest.mark.parametrize(
    ("min_tokens", "max_tokens", "width", "height", "tokens"), QWEN_BUDGET_CASES
)
def test_qwen_token_budget_bounds_the_image(
    tiny_model_dir: Path,
    min_tokens: int | None,
    max_tokens: int | None,
    width: int,
    height: int,
    tokens: int,
) -> None:
    write_qwen_vl_model(tiny_model_dir)
    mm = MultimodalConfig(image_min_tokens=min_tokens, image_max_tokens=max_tokens)
    assert image_tokens(qwen_processor(tiny_model_dir, mm), png(width, height)) == tokens


# (--image-min-tokens, --image-max-tokens, the tokens a 3000x3000 image takes through
# get_mm_processor): Qwen VL's default cap of 4096 holds until the flag replaces it.
QWEN_DEFAULT_CAP_CASES: list[tuple[int | None, int | None, int]] = [
    (None, None, 4096),
    (1024, None, 4096),
    # A flag above the default lifts it, here to the checkpoint's own 16384.
    (None, 16384, 8836),
    (None, 1024, 1024),
]


@pytest.mark.parametrize(("min_tokens", "max_tokens", "tokens"), QWEN_DEFAULT_CAP_CASES)
def test_qwen_default_max_tokens_yields_to_the_flag(
    tiny_model_dir: Path, min_tokens: int | None, max_tokens: int | None, tokens: int
) -> None:
    write_qwen_vl_model(tiny_model_dir)
    mm = MultimodalConfig(image_min_tokens=min_tokens, image_max_tokens=max_tokens)
    processor = get_mm_processor(str(tiny_model_dir), mm)
    assert processor is not None
    assert image_tokens(processor, png(3000, 3000)) == tokens


def test_min_tokens_above_the_default_max_needs_the_flag(tiny_model_dir: Path) -> None:
    write_qwen_vl_model(tiny_model_dir)
    with pytest.raises(ValueError, match="default --image-max-tokens 4096"):
        get_mm_processor(str(tiny_model_dir), MultimodalConfig(image_min_tokens=8192))


def test_gemma_wraps_the_soft_tokens_in_boi_eoi(tiny_model_dir: Path) -> None:
    write_gemma4_model(tiny_model_dir)
    ids = np.array([1, GEMMA4_IMAGE_TOKEN_ID, 2], np.int32)
    result = gemma_processor(tiny_model_dir).apply(ids, [png(640, 400)])
    (item,) = result.mm_items
    # 640x400 at the default 280 budget: 39 x 63 patches pooled 3x3 into 13 x 21.
    assert item.num_soft_tokens == 273 and item.offsets == [(2, 275)]
    assert item.position_ids is not None and item.position_ids.shape == (39 * 63, 2)
    assert item.position_ids[64].tolist() == [1, 1]
    assert result.input_ids.tolist() == [
        1,
        GEMMA4_BOI_TOKEN_ID,
        *[item.pad_value] * 273,
        GEMMA4_EOI_TOKEN_ID,
        2,
    ]
    assert result.mrope_positions is None and result.mrope_delta == 0


# (--image-max-tokens, the soft tokens a 640x400 image takes): the largest accepted budget
# within the maximum.
GEMMA_BUDGET_CASES: list[tuple[int | None, int]] = [
    (None, 273),
    (910, 522),
    (100, 60),
    (70, 60),
]


@pytest.mark.parametrize(("max_tokens", "soft_tokens"), GEMMA_BUDGET_CASES)
def test_gemma_maximum_picks_a_soft_token_budget(
    tiny_model_dir: Path, max_tokens: int | None, soft_tokens: int
) -> None:
    write_gemma4_model(tiny_model_dir)
    processor = gemma_processor(tiny_model_dir, MultimodalConfig(image_max_tokens=max_tokens))
    assert image_tokens(processor, png(640, 400)) == soft_tokens


def test_gemma_maximum_below_the_smallest_budget_is_refused(tiny_model_dir: Path) -> None:
    write_gemma4_model(tiny_model_dir)
    with pytest.raises(ValueError, match="below the smallest soft-token budget"):
        gemma_processor(tiny_model_dir, MultimodalConfig(image_max_tokens=60))


def test_processor_lookup_needs_a_served_encoder(
    tiny_model_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_qwen_vl_model(tiny_model_dir)
    gemma_dir = tmp_path / "gemma"
    gemma_dir.mkdir()
    write_gemma4_model(gemma_dir)
    assert isinstance(get_mm_processor(str(tiny_model_dir), MultimodalConfig()), QwenVLMMProcessor)
    text_model_only = MultimodalConfig(disabled_encoders=frozenset(ENCODER_KINDS))
    assert get_mm_processor(str(tiny_model_dir), text_model_only) is None
    vision_disabled = MultimodalConfig(disabled_encoders=frozenset({"vision"}))
    assert get_mm_processor(str(tiny_model_dir), vision_disabled) is None
    audio_disabled = MultimodalConfig(disabled_encoders=frozenset({"audio"}))
    assert isinstance(get_mm_processor(str(tiny_model_dir), audio_disabled), QwenVLMMProcessor)
    # Gemma4 registers no vision encoder yet.
    assert get_mm_processor(str(gemma_dir), MultimodalConfig()) is None
    register_vision_encoder(monkeypatch, "Gemma4ForConditionalGeneration")
    assert isinstance(get_mm_processor(str(gemma_dir), MultimodalConfig()), Gemma4MMProcessor)
    # A checkpoint without config.json, or without its vision section, serves no images.
    assert get_mm_processor(str(tmp_path), MultimodalConfig()) is None
    # Nor one whose weights hold no tower under the prefix the loader reads.
    weights = tiny_model_dir / "model.safetensors"
    mx.save_safetensors(str(weights), {"vision_tower.proj.weight": mx.zeros((1,))})
    assert get_mm_processor(str(tiny_model_dir), MultimodalConfig()) is None
    mx.save_safetensors(str(weights), {"model.visual.proj.weight": mx.zeros((1,))})
    assert isinstance(get_mm_processor(str(tiny_model_dir), MultimodalConfig()), QwenVLMMProcessor)
    write_qwen_vl_model(tiny_model_dir, has_vision=False)
    assert get_mm_processor(str(tiny_model_dir), MultimodalConfig()) is None
