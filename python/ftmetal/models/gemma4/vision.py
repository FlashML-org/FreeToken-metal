"""Gemma4 vision tower and the projection of its soft tokens into the text model's space.

Port of FreeToken models/gemma4/vision.py. Divergences: one image per call, so there are no
padding patches to mask or strip; the q / k / v and gate / up projections are fused and the
MLP is the text model's GeGLU, run in row chunks past MLP_ROWS patches; the 2-D rope is
mx.fast.rope on each half of a head, a patch's x and y as per-row offsets; the pooling is a
scatter-add by the patches' host-side kernel indices; no weight streaming and no
unified-release embedder.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from ftmetal.layers.base import BaseOP
from ftmetal.layers.linear import Linear, LinearMerged
from ftmetal.layers.norm import RMSNorm
from ftmetal.models.config import ModelConfig, scheme_of, schemes_of
from ftmetal.models.gemma4.moe import Gemma4GatedMLP
from ftmetal.models.gemma4.weight import GATE_UP_PARTS, QKV_PARTS
from ftmetal.models.loader import (
    config_bool,
    config_dict,
    config_float,
    config_int,
    config_optional_int,
    config_str,
)

RGB_CHANNELS = 3
# 63 MiB per MLP chunk, and the default 280-token image (2520 patches) runs in one.
MLP_ROWS = 2560
# Rope rotates the head's first half by the patch's x and its second half by its y.
NUM_AXES = 2


@dataclass(frozen=True, kw_only=True)
class VisionConfig:
    hidden_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    intermediate_size: int
    patch_size: int
    pooling_kernel_size: int
    position_embedding_size: int
    rope_theta: float
    rms_norm_eps: float
    standardize: bool


def parse_vision_config(vision_config: dict[str, object]) -> VisionConfig:
    """The checkpoint's vision_config; clipped linears or another activation are refused."""
    if config_bool(vision_config, "use_clipped_linears", False):
        raise ValueError("Gemma4 vision towers with clipped linears are not served")
    activation = config_str(vision_config, "hidden_activation", "gelu_pytorch_tanh")
    if activation != "gelu_pytorch_tanh":
        raise ValueError(f"unsupported Gemma4 vision activation {activation!r}")
    num_heads = config_int(vision_config, "num_attention_heads")
    hidden_size = config_int(vision_config, "hidden_size")
    rope_parameters = config_dict(vision_config, "rope_parameters") or {}
    return VisionConfig(
        hidden_size=hidden_size,
        num_layers=config_int(vision_config, "num_hidden_layers"),
        num_heads=num_heads,
        num_kv_heads=config_optional_int(vision_config, "num_key_value_heads") or num_heads,
        head_dim=config_optional_int(vision_config, "head_dim") or hidden_size // num_heads,
        intermediate_size=config_int(vision_config, "intermediate_size"),
        patch_size=config_int(vision_config, "patch_size"),
        pooling_kernel_size=config_int(vision_config, "pooling_kernel_size"),
        position_embedding_size=config_int(vision_config, "position_embedding_size"),
        rope_theta=config_float(rope_parameters, "rope_theta", 100.0),
        rms_norm_eps=config_float(vision_config, "rms_norm_eps", 1e-6),
        standardize=config_bool(vision_config, "standardize", False),
    )


def rotate_axes(x: mx.array, positions: mx.array, base: float) -> mx.array:
    """x [P, heads, D]: the first half of each head rotated by its patch's x, the second by its
    y (rotate_half within each half); positions int32 [P, 2]."""
    half = x.shape[-1] // 2
    rows = x[:, :, None]
    first = mx.fast.rope(
        rows[..., :half], half, traditional=False, base=base, scale=1.0, offset=positions[:, 0]
    )
    second = mx.fast.rope(
        rows[..., half:], half, traditional=False, base=base, scale=1.0, offset=positions[:, 1]
    )
    return mx.concatenate([first, second], axis=-1)[:, :, 0]


class Gemma4VisionAttention(BaseOP):
    """Bidirectional attention over one image's patches, q / k / v RMSNorm and 2-D rope;
    the scores are unscaled."""

    def __init__(
        self, config: ModelConfig, vision: VisionConfig, *, prefix: str, dtype: mx.Dtype
    ) -> None:
        super().__init__()
        self.num_heads = vision.num_heads
        self.num_kv_heads = vision.num_kv_heads
        self.head_dim = vision.head_dim
        self.rope_theta = vision.rope_theta
        q_dim = self.num_heads * self.head_dim
        kv_dim = self.num_kv_heads * self.head_dim
        self.qkv_proj = LinearMerged(
            vision.hidden_size,
            [q_dim, kv_dim, kv_dim],
            schemes=schemes_of(config, prefix, tuple(f"{part}.linear" for part in QKV_PARTS)),
            part_names=QKV_PARTS,
            dtype=dtype,
        )
        self.o_proj = Linear(
            q_dim,
            vision.hidden_size,
            scheme=scheme_of(config, f"{prefix}.o_proj.linear"),
            dtype=dtype,
        )
        eps = vision.rms_norm_eps
        self.q_norm = RMSNorm(self.head_dim, eps=eps, dtype=dtype)
        self.k_norm = RMSNorm(self.head_dim, eps=eps, dtype=dtype)
        self.v_norm = RMSNorm(self.head_dim, eps=eps, dtype=dtype, with_scale=False)

    def forward(self, x: mx.array, positions: mx.array) -> mx.array:
        num_patches = x.shape[0]
        q, k, v = self.qkv_proj.forward_split(x)
        q = rotate_axes(
            self.q_norm.forward(q.reshape(num_patches, self.num_heads, self.head_dim)),
            positions,
            self.rope_theta,
        )
        k = rotate_axes(
            self.k_norm.forward(k.reshape(num_patches, self.num_kv_heads, self.head_dim)),
            positions,
            self.rope_theta,
        )
        v = self.v_norm.forward(v.reshape(num_patches, self.num_kv_heads, self.head_dim))
        out = mx.fast.scaled_dot_product_attention(
            q.transpose(1, 0, 2)[None],
            k.transpose(1, 0, 2)[None],
            v.transpose(1, 0, 2)[None],
            scale=1.0,
        )
        return self.o_proj.forward(out[0].transpose(1, 0, 2).reshape(num_patches, -1))


class Gemma4VisionEncoderLayer(BaseOP):
    """A ViT block with the four-norm sandwich: each sublayer's output is normed before the
    residual add."""

    def __init__(
        self, config: ModelConfig, vision: VisionConfig, *, prefix: str, dtype: mx.Dtype
    ) -> None:
        super().__init__()
        self.self_attn = Gemma4VisionAttention(
            config, vision, prefix=f"{prefix}.self_attn", dtype=dtype
        )
        mlp = f"{prefix}.mlp"
        self.mlp = Gemma4GatedMLP(
            vision.hidden_size,
            vision.intermediate_size,
            gate_up_schemes=schemes_of(
                config, mlp, tuple(f"{part}.linear" for part in GATE_UP_PARTS)
            ),
            down_scheme=scheme_of(config, f"{mlp}.down_proj.linear"),
            dtype=dtype,
        )
        hidden, eps = vision.hidden_size, vision.rms_norm_eps
        self.input_layernorm = RMSNorm(hidden, eps=eps, dtype=dtype)
        self.post_attention_layernorm = RMSNorm(hidden, eps=eps, dtype=dtype)
        self.pre_feedforward_layernorm = RMSNorm(hidden, eps=eps, dtype=dtype)
        self.post_feedforward_layernorm = RMSNorm(hidden, eps=eps, dtype=dtype)

    def forward(self, x: mx.array, positions: mx.array) -> mx.array:
        attended = self.self_attn.forward(self.input_layernorm.forward(x), positions)
        x = x + self.post_attention_layernorm.forward(attended)
        normed = self.pre_feedforward_layernorm.forward(x)
        if normed.shape[0] <= MLP_ROWS:
            feed_forward = self.mlp.forward(normed)
        else:
            # Evaluated chunk by chunk: a lazy graph would run every chunk's gate_up first.
            parts: list[mx.array] = []
            for start in range(0, normed.shape[0], MLP_ROWS):
                parts.append(self.mlp.forward(normed[start : start + MLP_ROWS]))
                mx.eval(parts[-1])
            feed_forward = mx.concatenate(parts)
        return x + self.post_feedforward_layernorm.forward(feed_forward)


class Gemma4VisionEncoder(BaseOP):
    def __init__(self, config: ModelConfig, vision: VisionConfig, *, dtype: mx.Dtype) -> None:
        super().__init__()
        self.layers = [
            Gemma4VisionEncoderLayer(
                config, vision, prefix=f"vision_tower.encoder.layers.{layer_id}", dtype=dtype
            )
            for layer_id in range(vision.num_layers)
        ]

    def forward(self, x: mx.array, positions: mx.array) -> mx.array:
        for layer in self.layers:
            x = layer.forward(x, positions)
            # One layer at a time: the whole tower's lazy graph holds many layers' activations.
            mx.eval(x)
        return x


class Gemma4VisionPatchEmbedder(BaseOP):
    """Each patch's pixels, rescaled to [-1, 1] and projected, plus its x and y table rows."""

    def __init__(self, config: ModelConfig, vision: VisionConfig, *, dtype: mx.Dtype) -> None:
        super().__init__()
        self.input_proj = Linear(
            RGB_CHANNELS * vision.patch_size**2,
            vision.hidden_size,
            scheme=scheme_of(config, "vision_tower.patch_embedder.input_proj"),
            dtype=dtype,
        )
        self.position_embedding_table = mx.zeros(
            (NUM_AXES, vision.position_embedding_size, vision.hidden_size), dtype=dtype
        )

    def forward(self, pixel_values: mx.array, position_ids: mx.array) -> mx.array:
        projected = self.input_proj.forward(
            (2 * (pixel_values - 0.5)).astype(self.position_embedding_table.dtype)
        )
        table = self.position_embedding_table
        return projected + (table[0][position_ids[:, 0]] + table[1][position_ids[:, 1]])


class Gemma4VisionModel(BaseOP):
    """One image's patches -> its soft tokens [num_soft_tokens, hidden], in the tower dtype."""

    def __init__(self, config: ModelConfig, vision: VisionConfig, *, dtype: mx.Dtype) -> None:
        super().__init__()
        assert vision.head_dim % (2 * NUM_AXES) == 0, "each axis rotates whole halves"
        self.pooling_kernel_size = vision.pooling_kernel_size
        self.root_hidden_size = vision.hidden_size**0.5
        self.standardize = vision.standardize
        self.patch_embedder = Gemma4VisionPatchEmbedder(config, vision, dtype=dtype)
        self.encoder = Gemma4VisionEncoder(config, vision, dtype=dtype)
        if vision.standardize:
            self.std_bias = mx.zeros((vision.hidden_size,), dtype=dtype)
            self.std_scale = mx.ones((vision.hidden_size,), dtype=dtype)

    def forward(
        self, pixel_values: mx.array, position_ids: np.ndarray, num_soft_tokens: int
    ) -> mx.array:
        """pixel_values float32 [P, patch**2 * 3] in [0, 1]; position_ids int32 [P, 2] as (x, y)
        on a grid whose sides are multiples of the pooling kernel."""
        kernel = self.pooling_kernel_size
        num_patches = pixel_values.shape[0]
        assert position_ids.shape == (num_patches, NUM_AXES)
        assert num_patches == num_soft_tokens * kernel**2, (num_patches, num_soft_tokens)
        positions = mx.array(position_ids)
        x = self.patch_embedder.forward(pixel_values, positions)
        x = self.encoder.forward(x, positions)

        # Average each kernel x kernel block of patches, blocks in row-major order.
        columns = int(position_ids[:, 0].max()) + 1
        block = position_ids // kernel
        block_ids = mx.array(block[:, 0] + columns // kernel * block[:, 1])
        pooled = mx.zeros((num_soft_tokens, x.shape[1]), dtype=mx.float32)
        pooled = pooled.at[block_ids].add(x.astype(mx.float32)) / kernel**2
        # The average rounds to the tower dtype before the fp32 sqrt(hidden) scaling.
        pooled = pooled.astype(x.dtype).astype(mx.float32) * self.root_hidden_size
        if self.standardize:
            pooled = (pooled - self.std_bias.astype(mx.float32)) * self.std_scale.astype(mx.float32)
        return pooled.astype(x.dtype)


class Gemma4MultimodalEmbedder(BaseOP):
    """Soft tokens -> text embeddings: an unscaled RMSNorm, then the projection."""

    def __init__(self, config: ModelConfig, vision: VisionConfig, *, dtype: mx.Dtype) -> None:
        super().__init__()
        self.embedding_pre_projection_norm = RMSNorm(
            vision.hidden_size, eps=vision.rms_norm_eps, dtype=dtype, with_scale=False
        )
        self.embedding_projection = Linear(
            vision.hidden_size,
            config.hidden_size,
            scheme=scheme_of(config, "embed_vision.embedding_projection"),
            dtype=dtype,
        )

    def forward(self, x: mx.array) -> mx.array:
        return self.embedding_projection.forward(self.embedding_pre_projection_norm.forward(x))


__all__ = [
    "MLP_ROWS",
    "Gemma4MultimodalEmbedder",
    "Gemma4VisionModel",
    "VisionConfig",
    "parse_vision_config",
]
