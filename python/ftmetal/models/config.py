"""Model configuration: attention groups and the per-model geometry.

Port of FreeToken models/config.py, reduced to the FULL, SWA, LINEAR and QSA group kinds
and the fields Qwen3.5-style hybrid MoE and Gemma4 models use. Divergences: the quantization
config is the checkpoint's MLX scheme map; no tensor parallelism or MTP; the MoE
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
from ftmetal.layers.quantization.config import QuantConfig, QuantScheme
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


def mrope_section_from_rope_params(
    rope_params: dict[str, object], vision_config: dict[str, object] | None
) -> tuple[int, int, int] | None:
    """rope_parameters' [t, h, w] mrope_section; None without one, and without vision: a
    text-only model keeps the 1-D partial rope and its decode-graph layout."""
    raw_section = rope_params.get("mrope_section") if vision_config is not None else None
    if raw_section is None:
        return None
    elif not isinstance(raw_section, list):
        raise ValueError(f"mrope_section must be three ints, got {raw_section!r}")
    else:
        sections = [value for value in raw_section if isinstance(value, int)]
        if len(sections) != 3 or len(raw_section) != 3:
            raise ValueError(f"mrope_section must be three ints, got {raw_section!r}")
        else:
            return (sections[0], sections[1], sections[2])


@dataclass(frozen=True, kw_only=True)
class SlotStateSpec:
    """A model's own per-request state kept in the GDN state pool's slots, beside the GDN
    states, so snapshots, restores and clears carry it (FreeToken's slot_states)."""

    name: str
    shape: tuple[int, ...]
    # None: the engine dtype.
    dtype: mx.Dtype | None = None

    def nbytes(self, dtype: mx.Dtype) -> int:
        return math.prod(self.shape) * (self.dtype or dtype).size


@dataclass(frozen=True, kw_only=True)
class KVCacheGroupSpec:
    """The storage view of one attention group: all its pool needs to size and allocate."""

    name: str
    attn_type: AttnType
    layer_ids: tuple[int, ...]
    # Paged K/V geometry (FULL, SWA, QSA).
    num_kv_heads: int = 0
    head_dim: int = 0
    # The keys a query attends to, its own included (SWA).
    sliding_window: int | None = None
    # Image tokens also attend to their image's later tokens at prefill (batch.mm_block_ends).
    bidirectional_mm_blocks: bool = False
    # The block indexer (QSA): one index key per token, blocks of index_block_size keys, and
    # the complete blocks a query attends beside its own incomplete one.
    index_head_dim: int = 0
    index_block_size: int = 0
    index_topk_blocks: int = 0
    # Every token's 3-axis rope position at its slot, once for the group's layers: a block's
    # key ropes at its first token's, which an earlier forward may have written (QSA, mrope).
    has_rope_positions: bool = False
    # Per-slot state per layer (LINEAR); () when the type has none.
    conv_shape: tuple[int, ...] = ()
    state_shape: tuple[int, ...] = ()
    # The recurrent state's dtype; the conv state and K/V use the engine dtype.
    state_dtype: mx.Dtype = mx.float32
    # Per-slot states of the model's own, once per slot (LINEAR).
    slot_states: tuple[SlotStateSpec, ...] = ()

    @property
    def num_layers(self) -> int:
        return len(self.layer_ids)

    def token_bytes(self, dtype: mx.Dtype) -> int:
        """K, V and index key bytes per token over the group's layers, and its rope position;
        0 for per-slot state."""
        per_layer = (2 * self.num_kv_heads * self.head_dim + self.index_head_dim) * dtype.size
        rope_bytes = 3 * mx.int32.size if self.has_rope_positions else 0
        return per_layer * self.num_layers + rope_bytes

    def slot_bytes(self, dtype: mx.Dtype) -> int:
        """Conv, recurrent and model state bytes per slot over the group's layers; 0 for paged
        K/V."""
        conv_bytes = math.prod(self.conv_shape) * dtype.size if self.conv_shape else 0
        state_bytes = math.prod(self.state_shape) * self.state_dtype.size if self.state_shape else 0
        extra_bytes = sum(state.nbytes(dtype) for state in self.slot_states)
        return (conv_bytes + state_bytes) * self.num_layers + extra_bytes


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
    # Image tokens also attend to their image's later tokens at prefill (Gemma4's window layers).
    bidirectional_mm_blocks: bool = False


@dataclass(frozen=True)
class QSAAttentionGroupConfig(BaseAttentionGroupConfig):
    """Full attention over the keys a block indexer selects: each query attends its
    index_topk_blocks best-scoring complete blocks of index_block_size keys and its own
    incomplete block, every key while its complete blocks fit."""

    kind: ClassVar[Literal["qsa"]] = "qsa"

    num_kv_heads: int
    head_dim: int
    rotary_config: RotaryConfig
    index_num_heads: int
    index_head_dim: int
    index_block_size: int
    index_topk_blocks: int


GatedDeltaOutputGate = Literal["silu", "sigmoid"]


@dataclass(frozen=True)
class LinearGatedDeltaGroupConfig(BaseAttentionGroupConfig):
    kind: ClassVar[Literal["linear_gated_delta"]] = "linear_gated_delta"

    num_key_heads: int
    num_value_heads: int
    key_head_dim: int
    value_head_dim: int
    conv_kernel_dim: int
    output_gate: GatedDeltaOutputGate = "silu"
    slot_states: tuple[SlotStateSpec, ...] = ()


AttentionGroupConfig = (
    FullAttentionGroupConfig
    | SWAAttentionGroupConfig
    | LinearGatedDeltaGroupConfig
    | QSAAttentionGroupConfig
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
    # A prefill step's peak above the pools, (fixed bytes, bytes per token), measured on the
    # model; None: the engine's estimate (engine/cache_budget.py).
    prefill_activation_bytes: tuple[int, int] | None = None
    # Prefill tokens per batch when the server is not given a budget.
    max_extend_tokens: int = 2048
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
    def has_bidirectional_mm_blocks(self) -> bool:
        """Whether some layer attends within an image in both directions at prefill."""
        return any(
            isinstance(group, SWAAttentionGroupConfig) and group.bidirectional_mm_blocks
            for group in self.attention_groups
        )

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

    def qsa_attention_group(self) -> QSAAttentionGroupConfig | None:
        groups = [g for g in self.attention_groups if isinstance(g, QSAAttentionGroupConfig)]
        if len(groups) > 1:
            raise ValueError("expected at most one QSA attention group")
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
        elif isinstance(group, QSAAttentionGroupConfig):
            return AttnType.QSA
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
                    bidirectional_mm_blocks=group.bidirectional_mm_blocks,
                )
            elif isinstance(group, QSAAttentionGroupConfig):
                spec = KVCacheGroupSpec(
                    name=group.name,
                    attn_type=AttnType.QSA,
                    layer_ids=group.layer_ids,
                    num_kv_heads=group.num_kv_heads,
                    head_dim=group.head_dim,
                    index_head_dim=group.index_head_dim,
                    index_block_size=group.index_block_size,
                    index_topk_blocks=group.index_topk_blocks,
                    has_rope_positions=group.rotary_config.mrope_section is not None,
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
                    slot_states=group.slot_states,
                )
            specs.append(spec)
        assert len({spec.attn_type for spec in specs}) == len(specs), "one group per attention type"
        return tuple(specs)


def scheme_of(config: ModelConfig, checkpoint_module: str) -> QuantScheme | None:
    """The scheme of a module by its checkpoint path; None means bf16."""
    return None if config.quant is None else config.quant.scheme_for(checkpoint_module)


def schemes_of(
    config: ModelConfig, parent_module: str, parts: tuple[str, ...]
) -> list[QuantScheme | None]:
    """The scheme of each part (a checkpoint module name under parent_module) of a fused
    module; they fuse only when all agree."""
    if config.quant is None:
        return [None] * len(parts)
    else:
        return config.quant.schemes_for([f"{parent_module}.{part}" for part in parts])


__all__ = [
    "AttentionGroupConfig",
    "BaseAttentionGroupConfig",
    "FullAttentionGroupConfig",
    "GatedDeltaOutputGate",
    "KVCacheGroupSpec",
    "LinearGatedDeltaGroupConfig",
    "ModelConfig",
    "MoeStrategy",
    "QSAAttentionGroupConfig",
    "RotaryConfig",
    "SWAAttentionGroupConfig",
    "SlotStateSpec",
    "mrope_layout_from_rope_params",
    "mrope_section_from_rope_params",
    "scheme_of",
    "schemes_of",
]
