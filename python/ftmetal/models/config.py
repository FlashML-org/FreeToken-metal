"""Model configuration: attention groups and the per-model geometry.

Port of FreeToken models/config.py, reduced to the FULL and LINEAR group kinds and
the fields Qwen3.5-style hybrid MoE models use. Divergences: the quantization config
is the MLX checkpoint's affine scheme map; no tensor parallelism, vision or MTP.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Literal

from ftmetal.attention.base import AttnType
from ftmetal.layers.quantization.config import QuantConfig


@dataclass(frozen=True)
class RotaryConfig:
    head_dim: int
    rotary_dim: int
    max_position: int
    base: float


@dataclass(frozen=True)
class KVCacheGroupSpec:
    name: str
    layer_ids: tuple[int, ...]
    num_kv_heads: int
    head_dim: int
    attn_type: AttnType = AttnType.FULL

    @property
    def num_layers(self) -> int:
        return len(self.layer_ids)


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


@dataclass(frozen=True)
class LinearGatedDeltaGroupConfig(BaseAttentionGroupConfig):
    kind: ClassVar[Literal["linear_gated_delta"]] = "linear_gated_delta"

    num_key_heads: int
    num_value_heads: int
    key_head_dim: int
    value_head_dim: int
    conv_kernel_dim: int
    output_gate: str = "silu"


AttentionGroupConfig = FullAttentionGroupConfig | LinearGatedDeltaGroupConfig


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

    @property
    def is_moe(self) -> bool:
        return self.moe_enabled

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

    def linear_attention_group(self) -> LinearGatedDeltaGroupConfig | None:
        groups = [g for g in self.attention_groups if isinstance(g, LinearGatedDeltaGroupConfig)]
        if len(groups) > 1:
            raise ValueError("expected at most one linear-gated-delta attention group")
        return groups[0] if groups else None

    def is_linear_layer(self, layer_id: int) -> bool:
        return isinstance(self.attention_group_for_layer(layer_id), LinearGatedDeltaGroupConfig)

    def attn_type_for_layer(self, layer_id: int) -> AttnType:
        if self.is_linear_layer(layer_id):
            return AttnType.LINEAR
        return AttnType.FULL

    def kv_cache_group_specs(self) -> tuple[KVCacheGroupSpec, ...]:
        return tuple(
            KVCacheGroupSpec(
                name=group.name,
                layer_ids=group.layer_ids,
                num_kv_heads=group.num_kv_heads,
                head_dim=group.head_dim,
            )
            for group in self.attention_groups
            if isinstance(group, FullAttentionGroupConfig)
        )


__all__ = [
    "AttentionGroupConfig",
    "BaseAttentionGroupConfig",
    "FullAttentionGroupConfig",
    "KVCacheGroupSpec",
    "LinearGatedDeltaGroupConfig",
    "ModelConfig",
    "RotaryConfig",
]
