"""Model configuration: attention groups and the per-model geometry.

Port of FreeToken models/config.py, reduced to the FULL, SWA and LINEAR group kinds and
the fields Qwen3.5-style hybrid MoE and Gemma4 models use. Divergences: the quantization
config is the MLX checkpoint's affine scheme map; no tensor parallelism or MTP; the MoE
strategy is fused or offload and every layer of a MoE model routes experts; the
LINEAR group emits a KVCacheGroupSpec too (its per-slot conv and state shapes), so every
pool is sized from a spec; the rope variant is a rope_type field, not a scaling
dictionary; vision_config is the checkpoint's raw section, which the tower parses.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar, Literal

import mlx.core as mx

from ftmetal.attention.metadata import AttnType
from ftmetal.layers.quantization.config import QuantConfig
from ftmetal.layers.rotary import MRopeLayout


@dataclass(frozen=True)
class RotaryConfig:
    head_dim: int
    rotary_dim: int
    max_position: int
    base: float
    # proportional: the first rotary_dim / 2 frequency pairs of a head_dim-wide rope rotate,
    # the rest pass through (Gemma4's global layers); default: the first rotary_dim dims.
    rope_type: Literal["default", "proportional"] = "default"
    # 3-axis rope sections; None keeps the 1-D rope path, set only when the model serves
    # vision.
    mrope_section: tuple[int, int, int] | None = None
    mrope_layout: MRopeLayout = "contiguous"


def mrope_layout_from_rope_params(rope_params: dict[str, object]) -> MRopeLayout:
    """rope_parameters flags -> layout name: mrope_interleaved_glm, else mrope_interleaved,
    else contiguous."""
    if rope_params.get("mrope_interleaved_glm"):
        return "interleaved_glm"
    elif rope_params.get("mrope_interleaved"):
        return "interleaved"
    else:
        return "contiguous"


@dataclass(frozen=True, kw_only=True)
class KVCacheGroupSpec:
    """The storage view of one attention group: all its pool needs to size and allocate."""

    name: str
    attn_type: AttnType
    layer_ids: tuple[int, ...]
    # Paged K/V geometry (FULL, SWA).
    num_kv_heads: int = 0
    head_dim: int = 0
    # The keys a query attends to, its own included (SWA).
    sliding_window: int | None = None
    # Per-slot state per layer (LINEAR); () when the type has none.
    conv_shape: tuple[int, ...] = ()
    state_shape: tuple[int, ...] = ()
    # The recurrent state's dtype; the conv state and K/V use the engine dtype.
    state_dtype: mx.Dtype = mx.float32

    @property
    def num_layers(self) -> int:
        return len(self.layer_ids)

    def token_bytes(self, dtype: mx.Dtype) -> int:
        """K and V bytes per token over the group's layers; 0 for per-slot state."""
        return 2 * self.num_kv_heads * self.head_dim * dtype.size * self.num_layers

    def slot_bytes(self, dtype: mx.Dtype) -> int:
        """Conv and recurrent state bytes per slot over the group's layers; 0 for paged K/V."""
        conv_bytes = math.prod(self.conv_shape) * dtype.size if self.conv_shape else 0
        state_bytes = math.prod(self.state_shape) * self.state_dtype.size if self.state_shape else 0
        return (conv_bytes + state_bytes) * self.num_layers


@dataclass(frozen=True)
class BaseAttentionGroupConfig:
    name: str
    layer_ids: tuple[int, ...]

    def owns_layer(self, layer_id: int) -> bool:
        return layer_id in self.layer_ids


@dataclass(frozen=True)
class FullAttentionGroupConfig(BaseAttentionGroupConfig):
    kind: ClassVar[Literal["full"]] = "full"

    num_kv_heads: int
    head_dim: int
    rotary_config: RotaryConfig
    # V is the K projection's output, normed on its own (Gemma4's global layers).
    k_eq_v: bool = False


@dataclass(frozen=True)
class SWAAttentionGroupConfig(BaseAttentionGroupConfig):
    kind: ClassVar[Literal["swa"]] = "swa"

    num_kv_heads: int
    head_dim: int
    rotary_config: RotaryConfig
    sliding_window: int


@dataclass(frozen=True)
class LinearGatedDeltaGroupConfig(BaseAttentionGroupConfig):
    kind: ClassVar[Literal["linear_gated_delta"]] = "linear_gated_delta"

    num_key_heads: int
    num_value_heads: int
    key_head_dim: int
    value_head_dim: int
    conv_kernel_dim: int
    output_gate: str = "silu"


AttentionGroupConfig = (
    FullAttentionGroupConfig | SWAAttentionGroupConfig | LinearGatedDeltaGroupConfig
)
# How the routed experts are served: fused keeps every expert resident, offload reads them
# from the checkpoint into a slot cache (moe/offload_cache.py).
MoeStrategy = Literal["fused", "offload"]


@dataclass(frozen=True)
class ModelConfig:
    num_layers: int
    num_qo_heads: int
    num_kv_heads: int
    head_dim: int
    hidden_size: int
    vocab_size: int
    intermediate_size: int
    rms_norm_eps: float
    rotary_config: RotaryConfig
    hidden_act: str
    tie_word_embeddings: bool
    num_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    norm_topk_prob: bool
    model_type: str
    architectures: tuple[str, ...]
    attention_groups: tuple[AttentionGroupConfig, ...]
    quant: QuantConfig | None = None
    moe_enabled: bool = False
    shared_expert_intermediate_size: int = 0
    use_qk_norm: bool = False
    attn_output_gate: bool = False
    # None: head_dim ** -0.5 of each attention group.
    attn_sm_scale: float | None = None
    final_logit_softcapping: float | None = None
    embedding_scale: float | None = None
    # The checkpoint's vision_config section, None when this process builds no tower.
    vision_config: dict[str, object] | None = None
    image_token_id: int | None = None
    # Set by the engine before it builds the model; the checkpoint does not say.
    moe_strategy: MoeStrategy = "fused"
    # Set with moe_strategy: routed gate and up apart in every layer, under offload when the
    # checkpoint stores some layer's in two schemes.
    is_expert_gate_up_apart: bool = False

    @property
    def is_moe(self) -> bool:
        return self.moe_enabled

    @property
    def num_moe_layers(self) -> int:
        """Decoder layers with routed experts: all of them in the served MoE families."""
        return self.num_layers if self.moe_enabled else 0

    @property
    def is_multimodal(self) -> bool:
        return self.vision_config is not None

    @property
    def is_mrope(self) -> bool:
        """Whether attention takes 3-axis (t/h/w) rope positions."""
        return self.rotary_config.mrope_section is not None

    @property
    def has_linear_attention(self) -> bool:
        return self.linear_attention_group() is not None

    def attention_group_for_layer(self, layer_id: int) -> AttentionGroupConfig:
        matches = [group for group in self.attention_groups if group.owns_layer(layer_id)]
        if len(matches) != 1:
            raise ValueError(f"expected exactly one attention group for layer {layer_id}")
        return matches[0]

    def full_attention_group(self) -> FullAttentionGroupConfig | None:
        groups = [g for g in self.attention_groups if isinstance(g, FullAttentionGroupConfig)]
        if len(groups) > 1:
            raise ValueError("expected at most one full attention group")
        return groups[0] if groups else None

    def swa_attention_group(self) -> SWAAttentionGroupConfig | None:
        groups = [g for g in self.attention_groups if isinstance(g, SWAAttentionGroupConfig)]
        if len(groups) > 1:
            raise ValueError("expected at most one sliding-window attention group")
        return groups[0] if groups else None

    def linear_attention_group(self) -> LinearGatedDeltaGroupConfig | None:
        groups = [g for g in self.attention_groups if isinstance(g, LinearGatedDeltaGroupConfig)]
        if len(groups) > 1:
            raise ValueError("expected at most one linear-gated-delta attention group")
        return groups[0] if groups else None

    def is_linear_layer(self, layer_id: int) -> bool:
        return isinstance(self.attention_group_for_layer(layer_id), LinearGatedDeltaGroupConfig)

    def attn_type_for_layer(self, layer_id: int) -> AttnType:
        group = self.attention_group_for_layer(layer_id)
        if isinstance(group, LinearGatedDeltaGroupConfig):
            return AttnType.LINEAR
        elif isinstance(group, SWAAttentionGroupConfig):
            return AttnType.SWA
        else:
            return AttnType.FULL

    def kv_cache_group_specs(self) -> tuple[KVCacheGroupSpec, ...]:
        specs: list[KVCacheGroupSpec] = []
        for group in self.attention_groups:
            if isinstance(group, FullAttentionGroupConfig):
                spec = KVCacheGroupSpec(
                    name=group.name,
                    attn_type=AttnType.FULL,
                    layer_ids=group.layer_ids,
                    num_kv_heads=group.num_kv_heads,
                    head_dim=group.head_dim,
                )
            elif isinstance(group, SWAAttentionGroupConfig):
                spec = KVCacheGroupSpec(
                    name=group.name,
                    attn_type=AttnType.SWA,
                    layer_ids=group.layer_ids,
                    num_kv_heads=group.num_kv_heads,
                    head_dim=group.head_dim,
                    sliding_window=group.sliding_window,
                )
            else:
                # A conv ring over q, k and v; the state is [Hv, Dv, Dk] as the GDN kernels read it.
                conv_dim = 2 * group.num_key_heads * group.key_head_dim
                conv_dim += group.num_value_heads * group.value_head_dim
                spec = KVCacheGroupSpec(
                    name=group.name,
                    attn_type=AttnType.LINEAR,
                    layer_ids=group.layer_ids,
                    conv_shape=(group.conv_kernel_dim, conv_dim),
                    state_shape=(group.num_value_heads, group.value_head_dim, group.key_head_dim),
                )
            specs.append(spec)
        assert len({spec.attn_type for spec in specs}) == len(specs), "one group per attention type"
        return tuple(specs)


__all__ = [
    "AttentionGroupConfig",
    "BaseAttentionGroupConfig",
    "FullAttentionGroupConfig",
    "KVCacheGroupSpec",
    "LinearGatedDeltaGroupConfig",
    "ModelConfig",
    "MoeStrategy",
    "RotaryConfig",
    "SWAAttentionGroupConfig",
    "mrope_layout_from_rope_params",
]
