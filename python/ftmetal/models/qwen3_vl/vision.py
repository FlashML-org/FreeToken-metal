"""Qwen VL vision tower: ViT blocks with bidirectional attention per image and a 2x2 patch merger.

Port of FreeToken models/qwen3_vl/vision.py. Divergences: the patch conv runs as one matmul
(kernel == stride), its kernel stored channels last when the checkpoint is an MLX
conversion; no DeepStack, weight streaming, tensor parallelism or position-table row
chunking; the tower is evaluated stage by stage (the embedding, each block, each MLP row
chunk) and there is no environment override of the chunk size; the position-table taps and
the rope positions are built in numpy on the host.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import partial
from itertools import accumulate

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ftmetal.layers.base import BaseOP
from ftmetal.layers.embedding import Embedding
from ftmetal.layers.linear import Linear
from ftmetal.mm.item import MMItem
from ftmetal.models.loader import config_int, config_optional_str

LAYER_NORM_EPS = 1e-6
ROPE_THETA = 10000.0
# 4096 rows x 4304 x bf16 = 34 MiB per MLP temporary while the matmuls still fill the GPU.
MLP_ROWS = 4096


@dataclass(frozen=True, kw_only=True)
class VisionConfig:
    hidden_size: int
    depth: int
    num_heads: int
    intermediate_size: int
    patch_size: int
    temporal_patch_size: int
    spatial_merge_size: int
    num_position_embeddings: int
    out_hidden_size: int
    in_channels: int


def parse_vision_config(vision_config: dict[str, object]) -> VisionConfig:
    """The checkpoint's vision_config; a DeepStack tower or another activation is refused."""
    if vision_config.get("deepstack_visual_indexes"):
        raise ValueError("DeepStack vision towers are not supported")
    hidden_act = config_optional_str(vision_config, "hidden_act")
    if hidden_act != "gelu_pytorch_tanh":
        raise ValueError(f"unsupported vision hidden_act {hidden_act!r}")
    return VisionConfig(
        hidden_size=config_int(vision_config, "hidden_size"),
        depth=config_int(vision_config, "depth"),
        num_heads=config_int(vision_config, "num_heads"),
        intermediate_size=config_int(vision_config, "intermediate_size"),
        patch_size=config_int(vision_config, "patch_size"),
        temporal_patch_size=config_int(vision_config, "temporal_patch_size"),
        spatial_merge_size=config_int(vision_config, "spatial_merge_size"),
        num_position_embeddings=config_int(vision_config, "num_position_embeddings"),
        out_hidden_size=config_int(vision_config, "out_hidden_size"),
        in_channels=config_int(vision_config, "in_channels"),
    )


def interpolation_taps(index: np.ndarray, size: int, side: int) -> tuple[np.ndarray, np.ndarray]:
    """Bilinear (align_corners) taps into a side-long table axis for the positions index on a
    size-long axis, and their fp32 weights."""
    source = index.astype(np.float32) * np.float32(side - 1) / np.float32(max(size - 1, 1))
    floor = np.floor(source)
    offsets = np.arange(2, dtype=np.float32)
    taps = np.minimum(floor[:, None] + offsets, side - 1).astype(np.int32)
    weights = np.maximum(1 - np.abs(source[:, None] - floor[:, None] - offsets), 0)
    return taps, weights


# Shaped: a shapeless trace of this graph splits into several kernels with fp32 temporaries.
@mx.compile
def gelu_tanh(x: mx.array) -> mx.array:
    """gelu_pytorch_tanh in fp32, rounded once to the input dtype."""
    return nn.gelu_approx(x.astype(mx.float32)).astype(x.dtype)


@partial(mx.compile, shapeless=True)
def gelu_erf(x: mx.array) -> mx.array:
    """Exact gelu in fp32, rounded once to the input dtype."""
    return nn.gelu(x.astype(mx.float32)).astype(x.dtype)


@mx.compile
def add_position_rows(x: mx.array, corner_rows: mx.array, corner_weights: mx.array) -> mx.array:
    """x plus the bilinear sum of its four table rows [patches, 4, hidden], taken in fp32 one
    corner at a time as FreeToken does, rounded once to x's dtype before the add."""
    position_rows = corner_rows[:, 0].astype(mx.float32) * corner_weights[:, :1]
    for corner in range(1, 4):
        corner_row = corner_rows[:, corner].astype(mx.float32)
        position_rows = position_rows + corner_row * corner_weights[:, corner : corner + 1]
    return x + position_rows.astype(x.dtype)


# One output per call: a two-output kernel would keep its shared fp32 operands in memory.
@partial(mx.compile, shapeless=True)
def rope_rotate(x: mx.array, partner: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """x * cos - partner * sin in fp32, rounded once to x's dtype: one half of the NeoX rope."""
    return (x.astype(mx.float32) * cos - partner.astype(mx.float32) * sin).astype(x.dtype)


class VisionLayerNorm(BaseOP):
    def __init__(self, size: int) -> None:
        super().__init__()
        self.weight = mx.ones((size,), dtype=mx.bfloat16)
        self.bias = mx.zeros((size,), dtype=mx.bfloat16)

    def forward(self, x: mx.array) -> mx.array:
        return mx.fast.layer_norm(x, self.weight, self.bias, LAYER_NORM_EPS)


class VisionAttention(BaseOP):
    def __init__(self, config: VisionConfig) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.qkv = Linear(config.hidden_size, 3 * config.hidden_size, has_bias=True)
        self.proj = Linear(config.hidden_size, config.hidden_size, has_bias=True)

    def forward(
        self, x: mx.array, rope_cos: mx.array, rope_sin: mx.array, image_offsets: list[int]
    ) -> mx.array:
        """Rope, then bidirectional attention within each image; image_offsets are the patch
        rows where the second and later images start."""
        num_patches = x.shape[0]
        half = self.head_dim // 2
        qkv = self.qkv.forward(x).reshape(num_patches, 3, self.num_heads, self.head_dim)
        qkv = qkv.transpose(1, 2, 0, 3)
        first, second = qkv[:2, ..., :half], qkv[:2, ..., half:]
        qk = mx.concatenate(
            [
                rope_rotate(first, second, rope_cos, rope_sin),
                rope_rotate(second, first, rope_cos, -rope_sin),
            ],
            axis=-1,
        )
        # range slices stay views; an integer index plus a new axis copies a strided operand
        q, k, v = qk[0:1], qk[1:2], qkv[2:3]
        outputs = [
            mx.fast.scaled_dot_product_attention(
                q_image, k_image, v_image, scale=self.head_dim**-0.5
            ).transpose(0, 2, 1, 3)
            for q_image, k_image, v_image in zip(
                mx.split(q, image_offsets, axis=2),
                mx.split(k, image_offsets, axis=2),
                mx.split(v, image_offsets, axis=2),
            )
        ]
        out = mx.concatenate(outputs, axis=1) if len(outputs) > 1 else outputs[0]
        return self.proj.forward(out.reshape(num_patches, -1))


class VisionMLP(BaseOP):
    def __init__(self, config: VisionConfig) -> None:
        super().__init__()
        self.linear_fc1 = Linear(config.hidden_size, config.intermediate_size, has_bias=True)
        self.linear_fc2 = Linear(config.intermediate_size, config.hidden_size, has_bias=True)

    def forward(self, x: mx.array) -> mx.array:
        if x.shape[0] <= MLP_ROWS:
            return self.linear_fc2.forward(gelu_tanh(self.linear_fc1.forward(x)))
        else:
            # A per-token op: row chunks bound the two [rows, intermediate] temporaries while the
            # matmuls stay large. Each chunk is evaluated before the next is built: MLX runs a
            # lazy graph breadth first, every chunk's fc1 before any fc2.
            parts: list[mx.array] = []
            for start in range(0, x.shape[0], MLP_ROWS):
                rows = x[start : start + MLP_ROWS]
                parts.append(self.linear_fc2.forward(gelu_tanh(self.linear_fc1.forward(rows))))
                mx.eval(parts[-1])
            return mx.concatenate(parts)


class VisionBlock(BaseOP):
    def __init__(self, config: VisionConfig) -> None:
        super().__init__()
        self.norm1 = VisionLayerNorm(config.hidden_size)
        self.norm2 = VisionLayerNorm(config.hidden_size)
        self.attn = VisionAttention(config)
        self.mlp = VisionMLP(config)

    def forward(
        self, x: mx.array, rope_cos: mx.array, rope_sin: mx.array, image_offsets: list[int]
    ) -> mx.array:
        x = x + self.attn.forward(self.norm1.forward(x), rope_cos, rope_sin, image_offsets)
        return x + self.mlp.forward(self.norm2.forward(x))


class Conv3dParams(nn.Module):
    """The patch conv's weight [hidden, *kernel_shape] and bias."""

    def __init__(self, hidden_size: int, kernel_shape: tuple[int, int, int, int]) -> None:
        super().__init__()
        self.weight = mx.zeros((hidden_size, *kernel_shape), dtype=mx.bfloat16)
        self.bias = mx.zeros((hidden_size,), dtype=mx.bfloat16)


class VisionPatchEmbed(BaseOP):
    """The patch conv (kernel == stride) as one matmul over each flattened patch; a
    channels_last checkpoint stores its kernel [hidden, temporal, patch, patch, channels]."""

    def __init__(self, config: VisionConfig, channels_last: bool) -> None:
        super().__init__()
        temporal, patch = config.temporal_patch_size, config.patch_size
        channels = config.in_channels
        self.channels_last = channels_last
        if channels_last:
            kernel_shape = (temporal, patch, patch, channels)
        else:
            kernel_shape = (channels, temporal, patch, patch)
        self.proj = Conv3dParams(config.hidden_size, kernel_shape)

    def forward(self, pixel_values: mx.array) -> mx.array:
        # The processor flattens each patch channels first.
        if self.channels_last:
            weight = self.proj.weight.transpose(0, 4, 1, 2, 3)
        else:
            weight = self.proj.weight
        x = pixel_values.astype(weight.dtype)
        return x @ weight.reshape(weight.shape[0], -1).T + self.proj.bias


class VisionPatchMerger(BaseOP):
    """LayerNorm on each patch, then an MLP over each merge x merge block's patches."""

    def __init__(self, config: VisionConfig) -> None:
        super().__init__()
        self.merged_size = config.hidden_size * config.spatial_merge_size**2
        self.norm = VisionLayerNorm(config.hidden_size)
        self.linear_fc1 = Linear(self.merged_size, self.merged_size, has_bias=True)
        self.linear_fc2 = Linear(self.merged_size, config.out_hidden_size, has_bias=True)

    def forward(self, x: mx.array) -> mx.array:
        merged = self.norm.forward(x).reshape(-1, self.merged_size)
        return self.linear_fc2.forward(gelu_erf(self.linear_fc1.forward(merged)))


class Qwen3VLVisionModel(nn.Module):
    """Pixels -> soft tokens [sum(h * w) / merge**2, out_hidden_size], images in order.

    pixel_values are the processor's patches [sum(h * w), channels * temporal * patch**2] in
    merge-block order, grid_thw one (1, h, w) per image; the output is in the weights' dtype.
    channels_last: the checkpoint's patch conv kernel is channels last (an MLX conversion's).
    """

    def __init__(self, vision_config: dict[str, object], channels_last: bool) -> None:
        super().__init__()
        config = parse_vision_config(vision_config)
        self.config = config
        self.head_dim = config.hidden_size // config.num_heads
        assert self.head_dim % 4 == 0, "rows and columns each take half of every rope half"
        self.grid_side = math.isqrt(config.num_position_embeddings)
        assert self.grid_side**2 == config.num_position_embeddings, "the position table is square"
        self.patch_embed = VisionPatchEmbed(config, channels_last)
        self.pos_embed = Embedding(config.num_position_embeddings, config.hidden_size)
        self.blocks = [VisionBlock(config) for _ in range(config.depth)]
        self.merger = VisionPatchMerger(config)

    def __call__(self, pixel_values: mx.array, grid_thw: list[tuple[int, int, int]]) -> mx.array:
        config = self.config
        merge = config.spatial_merge_size
        side = self.grid_side
        for t, height, width in grid_thw:
            assert t == 1 and height % merge == 0 and width % merge == 0, (t, height, width)
        num_patches = sum(height * width for _, height, width in grid_thw)
        patch_features = config.in_channels * config.temporal_patch_size * config.patch_size**2
        assert pixel_values.shape == (num_patches, patch_features), pixel_values.shape

        positions: list[np.ndarray] = []
        taps: list[np.ndarray] = []
        weights: list[np.ndarray] = []
        for _, height, width in grid_thw:
            rows, cols = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
            blocks = (height // merge, merge, width // merge, merge)
            rows = rows.reshape(blocks).transpose(0, 2, 1, 3).reshape(-1)
            cols = cols.reshape(blocks).transpose(0, 2, 1, 3).reshape(-1)
            row_taps, row_weights = interpolation_taps(rows, height, side)
            col_taps, col_weights = interpolation_taps(cols, width, side)
            positions.append(np.stack([rows, cols], axis=-1))
            taps.append((row_taps[:, :, None] * side + col_taps[:, None, :]).reshape(-1, 4))
            weights.append((row_weights[:, :, None] * col_weights[:, None, :]).reshape(-1, 4))

        # no local holds the [patches, 4, hidden] corner rows: they die with the add
        x = add_position_rows(
            self.patch_embed.forward(pixel_values),
            self.pos_embed.forward(mx.array(np.concatenate(taps))),
            mx.array(np.concatenate(weights)),
        )

        dtype = self.pos_embed.weight.dtype
        inv_dim = self.head_dim // 2
        inv_freq = 1.0 / (ROPE_THETA ** (mx.arange(0, inv_dim, 2, dtype=mx.float32) / inv_dim))
        # frequencies and their cos / sin stay in the tower dtype on purpose: FreeToken's bf16
        # reference rounds them there
        patch_positions = mx.array(np.concatenate(positions)).astype(dtype)
        freqs = patch_positions[:, :, None] * inv_freq.astype(dtype)
        freqs = freqs.reshape(num_patches, inv_dim)
        rope_cos = mx.cos(freqs).astype(mx.float32)
        rope_sin = mx.sin(freqs).astype(mx.float32)

        image_offsets = list(accumulate(height * width for _, height, width in grid_thw))[:-1]
        # The tower runs stage by stage: one lazy graph of it runs breadth first and holds the
        # embedding's and several blocks' activations at once.
        mx.eval(x, rope_cos, rope_sin)
        for block in self.blocks:
            x = block.forward(x, rope_cos, rope_sin, image_offsets)
            mx.eval(x)
        return self.merger.forward(x)

    def encode(self, item: MMItem) -> mx.array:
        """[item.num_tokens, out_hidden_size] soft tokens of one image."""
        assert item.feature is not None and item.grid_thw is not None
        return self(mx.array(item.feature).view(mx.bfloat16), [item.grid_thw])


__all__ = ["MLP_ROWS", "Qwen3VLVisionModel", "VisionConfig", "parse_vision_config"]
