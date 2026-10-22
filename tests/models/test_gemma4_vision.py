"""The Gemma4 vision tower and its projection against an fp32 reference on tiny random weights.

The reference is an independent forward of the tower for one unpadded image, written apart
from the module (separate q / k / v and gate / up projections, rope per axis half, a
one-hot pooling by patch position, standardize, the unscaled pre-projection norm), and reads
the checkpoint's own tensor names, so the engine side also goes through the weight rules.
The cases cover a non-square grid in a patch order that is not row-major, grouped kv heads at
Gemma4's head_dim 72, the bf16 tower, a 4-bit projection, a tower whose q / k / v and gate / up
the checkpoint quantizes unevenly (so the loader leaves them apart), and the MLP in row chunks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from ftmetal.layers.quantization.config import QuantConfig
from ftmetal.models.gemma4 import vision
from ftmetal.models.gemma4.config import parse_config
from ftmetal.models.gemma4.vision import (
    Gemma4MultimodalEmbedder,
    Gemma4VisionModel,
    parse_vision_config,
)
from ftmetal.models.loader import apply_rules, load_state
from ftmetal.models.register import model_weight_rules
from tiny_vl_model import TINY_GEMMA4_VISION_CONFIG, gemma4_tower_weights

TEXT_HIDDEN_SIZE = 48
TEXT_CONFIG: dict[str, object] = {
    "hidden_size": TEXT_HIDDEN_SIZE,
    "num_hidden_layers": 2,
    "intermediate_size": 64,
    "num_attention_heads": 2,
    "head_dim": 32,
    "num_key_value_heads": 1,
    "sliding_window": 8,
    "vocab_size": 128,
    "max_position_embeddings": 512,
}
# Gemma4's head_dim 72, four query heads on two kv heads.
GROUPED_72_CONFIG = {
    **TINY_GEMMA4_VISION_CONFIG,
    "hidden_size": 144,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 72,
}
QUANT_GROUP_SIZE = 32
PROJECTION = "embed_vision.embedding_projection"
# Bits per tower layer module in the uneven case; v_proj stays bf16 beside its 4- and 8-bit q / k.
UNEVEN_LAYER_BITS = {
    "self_attn.q_proj": 4,
    "self_attn.k_proj": 8,
    "mlp.gate_proj": 4,
    "mlp.up_proj": 8,
}
# Worst |error| relative to the largest |expected| value: fp32 differs by summation order,
# bf16 rounds the weights and every activation.
TOLERANCE = {mx.float32: 1e-5, mx.bfloat16: 2**-5}


@dataclass(frozen=True)
class Case:
    rows: int
    columns: int
    vision_config: dict[str, object]
    dtype: mx.Dtype = mx.float32
    # projection: a 4-bit projection; uneven: also UNEVEN_LAYER_BITS in every tower layer.
    quantization: Literal["none", "projection", "uneven"] = "none"
    # The patches in a random order instead of row-major.
    shuffled: bool = False
    # vision.MLP_ROWS for the case; None keeps it.
    mlp_rows: int | None = None

    def __str__(self) -> str:
        heads = self.vision_config["num_attention_heads"], self.vision_config["head_dim"]
        label = f"{self.rows}x{self.columns}-H{heads[0]}-D{heads[1]}"
        label += "-bf16" if self.dtype == mx.bfloat16 else "-fp32"
        label += {"none": "", "projection": "-q4", "uneven": "-uneven"}[self.quantization]
        label += "-shuffled" if self.shuffled else ""
        return label + ("" if self.mlp_rows is None else f"-mlp{self.mlp_rows}")


CASES = [
    Case(6, 12, TINY_GEMMA4_VISION_CONFIG, shuffled=True),
    Case(9, 6, GROUPED_72_CONFIG),
    Case(6, 12, TINY_GEMMA4_VISION_CONFIG, mx.bfloat16),
    Case(6, 6, TINY_GEMMA4_VISION_CONFIG, quantization="projection"),
    Case(6, 6, TINY_GEMMA4_VISION_CONFIG, quantization="uneven"),
    # 54 patches in MLP chunks of 10, the last one short.
    Case(6, 9, TINY_GEMMA4_VISION_CONFIG, mlp_rows=10),
]


def rms_norm(x: mx.array, weight: mx.array | None, eps: float) -> mx.array:
    normed = x * mx.rsqrt(mx.mean(x**2, axis=-1, keepdims=True) + eps)
    return normed if weight is None else normed * weight


def multidimensional_rope(x: mx.array, positions: mx.array, base: float) -> mx.array:
    """x [L, N, D], positions [L, 2]: each axis rotates its own half of the head with
    rotate_half inside it."""
    channels = x.shape[-1] // 2
    exponents = (2.0 / channels) * mx.arange(0, channels // 2).astype(mx.float32)
    parts = []
    for axis in range(2):
        part = x[..., axis * channels : (axis + 1) * channels]
        angles = positions[:, axis : axis + 1].astype(mx.float32) / mx.power(base, exponents)
        cos = mx.concatenate([mx.cos(angles), mx.cos(angles)], axis=-1)[:, None]
        sin = mx.concatenate([mx.sin(angles), mx.sin(angles)], axis=-1)[:, None]
        rotated = mx.concatenate([-part[..., channels // 2 :], part[..., : channels // 2]], -1)
        parts.append(part * cos + rotated * sin)
    return mx.concatenate(parts, axis=-1)


def reference_embeddings(
    weights: dict[str, mx.array],
    vision_config: dict[str, object],
    pixel_values: mx.array,
    positions: mx.array,
) -> mx.array:
    """The tower then the projection, fp32, for one image."""
    params = {name: value.astype(mx.float32) for name, value in weights.items()}
    tower_config = parse_vision_config(vision_config)
    heads, kv_heads = tower_config.num_heads, tower_config.num_kv_heads
    head_dim, theta = tower_config.head_dim, tower_config.rope_theta
    eps = tower_config.rms_norm_eps
    tower = "vision_tower"
    x = (2 * (pixel_values - 0.5)) @ params[f"{tower}.patch_embedder.input_proj.weight"].T
    table = params[f"{tower}.patch_embedder.position_embedding_table"]
    one_hot = (positions[:, :, None] == mx.arange(table.shape[1])).astype(mx.float32)
    x = x + (one_hot.transpose(1, 0, 2) @ table).sum(axis=0)
    length = x.shape[0]
    for layer_id in range(tower_config.num_layers):
        prefix = f"{tower}.encoder.layers.{layer_id}"

        def linear(inputs: mx.array, name: str) -> mx.array:
            return inputs @ params[f"{prefix}.{name}.linear.weight"].T

        normed = rms_norm(x, params[f"{prefix}.input_layernorm.weight"], eps)
        q = linear(normed, "self_attn.q_proj").reshape(length, heads, head_dim)
        k = linear(normed, "self_attn.k_proj").reshape(length, kv_heads, head_dim)
        v = linear(normed, "self_attn.v_proj").reshape(length, kv_heads, head_dim)
        q = multidimensional_rope(
            rms_norm(q, params[f"{prefix}.self_attn.q_norm.weight"], eps),
            positions,
            theta,
        )
        k = multidimensional_rope(
            rms_norm(k, params[f"{prefix}.self_attn.k_norm.weight"], eps),
            positions,
            theta,
        )
        v = rms_norm(v, None, eps)
        attended = mx.fast.scaled_dot_product_attention(
            q.transpose(1, 0, 2)[None],
            k.transpose(1, 0, 2)[None],
            v.transpose(1, 0, 2)[None],
            scale=1.0,
        )
        attended = linear(attended[0].transpose(1, 0, 2).reshape(length, -1), "self_attn.o_proj")
        x = x + rms_norm(attended, params[f"{prefix}.post_attention_layernorm.weight"], eps)
        normed = rms_norm(x, params[f"{prefix}.pre_feedforward_layernorm.weight"], eps)
        gated = nn.gelu_approx(linear(normed, "mlp.gate_proj")) * linear(normed, "mlp.up_proj")
        feed_forward = linear(gated, "mlp.down_proj")
        x = x + rms_norm(feed_forward, params[f"{prefix}.post_feedforward_layernorm.weight"], eps)

    kernel = tower_config.pooling_kernel_size
    num_soft_tokens = length // kernel**2
    max_x = positions[:, 0].max() + 1
    kernel_ids = positions // kernel
    kernel_ids = kernel_ids[:, 0] + (max_x // kernel) * kernel_ids[:, 1]
    pool = (kernel_ids[:, None] == mx.arange(num_soft_tokens)).astype(mx.float32) / kernel**2
    pooled = (pool.T @ x) * tower_config.hidden_size**0.5
    pooled = (pooled - params[f"{tower}.std_bias"]) * params[f"{tower}.std_scale"]
    return rms_norm(pooled, None, eps) @ params[f"{PROJECTION}.weight"].T


def check_against_reference(case: Case) -> None:
    weights = gemma4_tower_weights(case.vision_config, TEXT_HIDDEN_SIZE, seed=0)
    weights = {name: value.astype(case.dtype) for name, value in weights.items()}
    # The reference reads each quantized module's dequantized weight.
    reference = dict(weights)
    bits_by_module: dict[str, int] = {}
    if case.quantization != "none":
        bits_by_module[PROJECTION] = 4
    if case.quantization == "uneven":
        for layer_id in range(parse_vision_config(case.vision_config).num_layers):
            prefix = f"vision_tower.encoder.layers.{layer_id}"
            for module, bits in UNEVEN_LAYER_BITS.items():
                bits_by_module[f"{prefix}.{module}.linear"] = bits
    quantization: dict[str, object] | None = None
    if bits_by_module:
        quantization = {"group_size": QUANT_GROUP_SIZE, "bits": 4, "mode": "affine"}
        quantization |= {
            module: {"group_size": QUANT_GROUP_SIZE, "bits": bits}
            for module, bits in bits_by_module.items()
            if bits != 4
        }
    for module, bits in bits_by_module.items():
        quantized, scales, biases = mx.quantize(
            weights[f"{module}.weight"], QUANT_GROUP_SIZE, bits=bits
        )
        reference[f"{module}.weight"] = mx.dequantize(
            quantized, scales, biases, QUANT_GROUP_SIZE, bits=bits
        )
        weights |= {
            f"{module}.weight": quantized,
            f"{module}.scales": scales,
            f"{module}.biases": biases,
        }
    config = parse_config(
        {
            "architectures": ["Gemma4ForConditionalGeneration"],
            "text_config": TEXT_CONFIG,
            "vision_config": case.vision_config,
        },
        QuantConfig.from_checkpoint(quantization, list(weights)),
    )
    assert config.vision_config is not None
    tower_config = parse_vision_config(config.vision_config)
    tower = Gemma4VisionModel(config, tower_config, dtype=case.dtype)
    embedder = Gemma4MultimodalEmbedder(config, tower_config, dtype=case.dtype)
    # Loaded through the weight rules, keys, shapes and dtypes strict.
    state = apply_rules(weights, model_weight_rules("Gemma4ForConditionalGeneration", config))
    for module, prefix in ((tower, "vision_tower."), (embedder, "embed_vision.")):
        load_state(
            module,
            {name.removeprefix(prefix): state[name] for name in state if name.startswith(prefix)},
        )

    grid_x, grid_y = np.meshgrid(np.arange(case.columns), np.arange(case.rows), indexing="xy")
    position_ids = np.stack([grid_x, grid_y], axis=-1).reshape(-1, 2).astype(np.int32)
    rng = np.random.default_rng(1)
    pixels = rng.uniform(0, 1, (len(position_ids), 3 * 16 * 16)).astype(np.float32)
    if case.shuffled:
        order = rng.permutation(len(position_ids))
        pixels, position_ids = pixels[order], position_ids[order]
    num_soft_tokens = case.rows * case.columns // 9

    out = embedder.forward(tower.forward(mx.array(pixels), position_ids, num_soft_tokens))
    expected = reference_embeddings(
        reference, case.vision_config, mx.array(pixels), mx.array(position_ids)
    )
    assert out.shape == (num_soft_tokens, TEXT_HIDDEN_SIZE) and out.dtype == case.dtype
    error = mx.abs(out.astype(mx.float32) - expected).max()
    assert float(error / mx.abs(expected).max()) < TOLERANCE[case.dtype]


@pytest.mark.parametrize("case", CASES, ids=str)
def test_matches_the_reference_forward(case: Case, monkeypatch: pytest.MonkeyPatch) -> None:
    if case.mlp_rows is not None:
        monkeypatch.setattr(vision, "MLP_ROWS", case.mlp_rows)
    check_against_reference(case)


def test_refuses_clipped_linears() -> None:
    with pytest.raises(ValueError, match="clipped linears"):
        parse_vision_config({**TINY_GEMMA4_VISION_CONFIG, "use_clipped_linears": True})
