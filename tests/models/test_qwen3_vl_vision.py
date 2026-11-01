"""The Qwen3.6 vision tower against a float64 numpy reference on tiny random weights.

The reference follows FreeToken's torch tower step by step (the patch conv, F.interpolate of
the position table with align_corners, 2-D rope, attention per image, the merger) and is
written apart from the module's host tables. The cases cover one image, two images of
different grids in one call (attention must not cross images), a non-square grid, grids
smaller and larger than the position table's side, head_dim 72 on the fused attention
kernel, and the MLP in row chunks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten, tree_unflatten

from ftmetal.models.qwen3_vl import vision
from ftmetal.models.qwen3_vl.vision import Qwen3VLVisionModel

# The real vision_config keys at toy sizes: head_dim 32, a 4 x 4 position table.
TINY_VISION_CONFIG: dict[str, object] = {
    "deepstack_visual_indexes": [],
    "depth": 2,
    "hidden_act": "gelu_pytorch_tanh",
    "hidden_size": 64,
    "in_channels": 3,
    "intermediate_size": 128,
    "num_heads": 2,
    "num_position_embeddings": 16,
    "out_hidden_size": 32,
    "patch_size": 16,
    "spatial_merge_size": 2,
    "temporal_patch_size": 2,
}
# Qwen3.6's head_dim 72, which only the fused full-attention kernel serves past 8 queries.
HEAD_DIM_72_CONFIG: dict[str, object] = {
    **TINY_VISION_CONFIG,
    "hidden_size": 144,
    "intermediate_size": 256,
    "num_position_embeddings": 36,
}
# Worst |error| relative to the largest |expected| value. fp32 differs from the float64
# reference by summation order; bf16 rounds every activation and the rope frequencies.
TOLERANCE = {mx.float32: 1e-5, mx.bfloat16: 2**-5}


@dataclass(frozen=True)
class Case:
    grids: tuple[tuple[int, int, int], ...]
    vision_config: dict[str, object]
    dtype: mx.Dtype = mx.float32

    def __str__(self) -> str:
        label = "+".join(f"{h}x{w}" for _, h, w in self.grids)
        label += f"-D{self.head_dim}"
        return label + ("-bf16" if self.dtype == mx.bfloat16 else "-fp32")

    @property
    def head_dim(self) -> int:
        hidden_size, num_heads = self.vision_config["hidden_size"], self.vision_config["num_heads"]
        assert isinstance(hidden_size, int) and isinstance(num_heads, int)
        return hidden_size // num_heads


CASES = [
    Case(((1, 4, 4),), TINY_VISION_CONFIG),  # one image, grid == table side
    Case(((1, 4, 6), (1, 8, 4)), TINY_VISION_CONFIG),  # two images, no attention across
    Case(((1, 4, 10),), TINY_VISION_CONFIG),  # non-square grid
    Case(((1, 12, 12),), TINY_VISION_CONFIG),  # grid larger than the table side
    Case(((1, 2, 2), (1, 6, 2)), TINY_VISION_CONFIG),  # smaller than the side, 4 queries
    Case(((1, 8, 6), (1, 4, 4)), HEAD_DIM_72_CONFIG),
    Case(((1, 8, 6), (1, 4, 4)), HEAD_DIM_72_CONFIG, mx.bfloat16),
]


def config_value(vision_config: dict[str, object], key: str) -> int:
    value = vision_config[key]
    assert isinstance(value, int)
    return value


def random_weights(model: Qwen3VLVisionModel, seed: int) -> dict[str, np.ndarray]:
    """fp32 weights for every parameter: unit-gain-ish norms, fan-in scaled projections."""
    rng = np.random.default_rng(seed)
    weights: dict[str, np.ndarray] = {}
    for name, placeholder in dict(tree_flatten(model.parameters())).items():
        shape = tuple(placeholder.shape)
        if "norm" in name and name.endswith("weight"):
            value = 1 + 0.1 * rng.standard_normal(shape)
        elif name.endswith("bias"):
            value = 0.1 * rng.standard_normal(shape)
        elif name == "pos_embed.weight":
            value = rng.standard_normal(shape)
        else:
            value = rng.standard_normal(shape) / math.sqrt(math.prod(shape[1:]))
        weights[name] = value.astype(np.float32)
    return weights


def linear(x: np.ndarray, weights: dict[str, np.ndarray], name: str) -> np.ndarray:
    return x @ weights[f"{name}.weight"].T + weights[f"{name}.bias"]


def layer_norm(x: np.ndarray, weights: dict[str, np.ndarray], name: str) -> np.ndarray:
    centered = x - x.mean(-1, keepdims=True)
    normed = centered / np.sqrt((centered**2).mean(-1, keepdims=True) + 1e-6)
    return normed * weights[f"{name}.weight"] + weights[f"{name}.bias"]


def interpolation_matrix(size: int, side: int) -> np.ndarray:
    """[size, side]: F.interpolate(mode="linear", align_corners=True) along one axis."""
    matrix = np.zeros((size, side))
    for i in range(size):
        source = i * (side - 1) / max(size - 1, 1)
        low = math.floor(source)
        matrix[i, low] += 1 - (source - low)
        matrix[i, min(low + 1, side - 1)] += source - low
    return matrix


def merge_order(grid: np.ndarray, merge: int) -> np.ndarray:
    """[h, w, ...] in raster order -> [h * w, ...] in the processor's merge-block order."""
    height, width = grid.shape[:2]
    blocks = grid.reshape(height // merge, merge, width // merge, merge, *grid.shape[2:])
    return blocks.swapaxes(1, 2).reshape(height * width, *grid.shape[2:])


def rope(x: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """2-D NeoX rope of x [patches, heads, head_dim]: rows take the first quarter of the
    frequencies of each half, columns the second."""
    half = x.shape[-1] // 2
    inv_freq = 1.0 / 10000.0 ** (np.arange(0, half, 2) / half)
    freqs = (positions[:, :, None] * inv_freq).reshape(len(x), half)
    angles = np.concatenate([freqs, freqs], axis=-1)[:, None]
    rotated = np.concatenate([-x[..., half:], x[..., :half]], axis=-1)
    return x * np.cos(angles) + rotated * np.sin(angles)


def reference_tower(
    weights: dict[str, np.ndarray],
    vision_config: dict[str, object],
    pixel_values: np.ndarray,
    grids: tuple[tuple[int, int, int], ...],
) -> np.ndarray:
    w = {name: value.astype(np.float64) for name, value in weights.items()}
    num_heads = config_value(vision_config, "num_heads")
    merge = config_value(vision_config, "spatial_merge_size")
    side = math.isqrt(config_value(vision_config, "num_position_embeddings"))
    kernel = w["patch_embed.proj.weight"]
    patches = pixel_values.astype(np.float64).reshape(-1, *kernel.shape[1:])
    x = np.einsum("nctij,hctij->nh", patches, kernel) + w["patch_embed.proj.bias"]

    table = w["pos_embed.weight"].reshape(side, side, -1)
    position_rows, positions = [], []
    for _, height, width in grids:
        resampled = np.einsum(
            "ha,wb,abd->hwd",
            interpolation_matrix(height, side),
            interpolation_matrix(width, side),
            table,
        )
        position_rows.append(merge_order(resampled, merge))
        rows_cols = np.stack(np.meshgrid(np.arange(height), np.arange(width), indexing="ij"), -1)
        positions.append(merge_order(rows_cols, merge))
    x = x + np.concatenate(position_rows)
    patch_positions = np.concatenate(positions)

    num_patches = len(x)
    bounds = np.cumsum([0] + [height * width for _, height, width in grids])
    for i in range(config_value(vision_config, "depth")):
        prefix = f"blocks.{i}"
        qkv = linear(layer_norm(x, w, f"{prefix}.norm1"), w, f"{prefix}.attn.qkv")
        qkv = qkv.reshape(num_patches, 3, num_heads, -1)
        q, k, v = rope(qkv[:, 0], patch_positions), rope(qkv[:, 1], patch_positions), qkv[:, 2]
        attended = np.zeros_like(q)
        for start, end in zip(bounds[:-1], bounds[1:]):
            scores = np.einsum("qhd,khd->hqk", q[start:end], k[start:end]) / math.sqrt(q.shape[-1])
            probs = np.exp(scores - scores.max(-1, keepdims=True))
            probs /= probs.sum(-1, keepdims=True)
            attended[start:end] = np.einsum("hqk,khd->qhd", probs, v[start:end])
        x = x + linear(attended.reshape(num_patches, -1), w, f"{prefix}.attn.proj")
        hidden = linear(layer_norm(x, w, f"{prefix}.norm2"), w, f"{prefix}.mlp.linear_fc1")
        inner = math.sqrt(2 / math.pi) * (hidden + 0.044715 * hidden**3)
        x = x + linear(0.5 * hidden * (1 + np.tanh(inner)), w, f"{prefix}.mlp.linear_fc2")

    merged = layer_norm(x, w, "merger.norm").reshape(num_patches // merge**2, -1)
    hidden = linear(merged, w, "merger.linear_fc1")
    hidden = 0.5 * hidden * (1 + np.vectorize(math.erf)(hidden / math.sqrt(2)))
    return linear(hidden, w, "merger.linear_fc2")


def check_against_reference(case: Case) -> None:
    model = Qwen3VLVisionModel(case.vision_config)
    weights = random_weights(model, seed=0)
    model.update(tree_unflatten([(name, mx.array(value)) for name, value in weights.items()]))
    model.set_dtype(case.dtype)
    num_patches = sum(height * width for _, height, width in case.grids)
    patch_features = 3 * 2 * 16 * 16
    pixel_values = np.random.default_rng(1).uniform(-1, 1, (num_patches, patch_features))
    pixel_values = pixel_values.astype(np.float32)

    out = model(mx.array(pixel_values), list(case.grids))
    expected = reference_tower(weights, case.vision_config, pixel_values, case.grids)
    out_hidden_size = config_value(case.vision_config, "out_hidden_size")
    assert out.shape == (num_patches // 4, out_hidden_size) and out.dtype == case.dtype
    error = np.abs(np.array(out.astype(mx.float32)) - expected).max()
    assert error / np.abs(expected).max() < TOLERANCE[case.dtype]


@pytest.mark.parametrize("case", CASES, ids=str)
def test_matches_numpy_reference(case: Case) -> None:
    check_against_reference(case)


def test_mlp_row_chunks_match_the_numpy_reference(monkeypatch: pytest.MonkeyPatch) -> None:
    """An image past MLP_ROWS patches runs its MLP in row chunks, the last one short: 144
    patches in chunks of 10."""
    monkeypatch.setattr(vision, "MLP_ROWS", 10)
    check_against_reference(Case(((1, 12, 12),), TINY_VISION_CONFIG))


def test_parameters_follow_the_checkpoint_names() -> None:
    """The loader maps model.visual.<name> to <name>: names, shapes and the bf16 dtype."""
    model = Qwen3VLVisionModel(TINY_VISION_CONFIG)
    top = {
        "patch_embed.proj.weight": (64, 3, 2, 16, 16),
        "patch_embed.proj.bias": (64,),
        "pos_embed.weight": (16, 64),
        "merger.norm.weight": (64,),
        "merger.norm.bias": (64,),
        "merger.linear_fc1.weight": (256, 256),
        "merger.linear_fc1.bias": (256,),
        "merger.linear_fc2.weight": (32, 256),
        "merger.linear_fc2.bias": (32,),
    }
    block = {
        "norm1.weight": (64,),
        "norm1.bias": (64,),
        "norm2.weight": (64,),
        "norm2.bias": (64,),
        "attn.qkv.weight": (192, 64),
        "attn.qkv.bias": (192,),
        "attn.proj.weight": (64, 64),
        "attn.proj.bias": (64,),
        "mlp.linear_fc1.weight": (128, 64),
        "mlp.linear_fc1.bias": (128,),
        "mlp.linear_fc2.weight": (64, 128),
        "mlp.linear_fc2.bias": (64,),
    }
    blocks = {f"blocks.{i}.{name}": shape for i in range(2) for name, shape in block.items()}
    expected = top | blocks
    parameters = dict(tree_flatten(model.parameters()))
    assert {name: tuple(value.shape) for name, value in parameters.items()} == expected
    assert all(value.dtype == mx.bfloat16 for value in parameters.values())


def test_refuses_a_deepstack_tower() -> None:
    with pytest.raises(ValueError, match="DeepStack"):
        Qwen3VLVisionModel({**TINY_VISION_CONFIG, "deepstack_visual_indexes": [0]})
