"""Qwen3.8-Flash-Next (qwen4_exp) config parsing from a checkpoint's config.json.

Port of FreeToken models/qwen4_exp/config.py. Divergences: reads the MLX checkpoint's
text_config and scheme map; the full-attention layers are a QSA group (their indexer's
geometry on the group, not on every full group); each PLE layer's n-gram context and
short-conv window are slot states of the GDN group; served text only (the 1-D partial
rope; no vision tower, no MTP head).
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx

from ftmetal.layers.quantization.config import QuantConfig
from ftmetal.models.config import (
    LinearGatedDeltaGroupConfig,
    ModelConfig,
    QSAAttentionGroupConfig,
    RotaryConfig,
    SlotStateSpec,
)
from ftmetal.models.loader import (
    config_bool,
    config_dict,
    config_float,
    config_int,
    config_optional_int,
    config_optional_str,
    config_str,
    config_str_list,
)

# A prefill step's peak above the pools, the experts offloaded (M4 Pro, 32k-token prompt,
# 2026-10-27): 1161 / 1804 / 3001 MiB at 2048 / 4096 / 8192-token chunks, rounded up.
PREFILL_ACTIVATION_FIXED_BYTES = 640 << 20
PREFILL_ACTIVATION_BYTES_PER_TOKEN = 320 << 10
# The offload cache reads each layer's absent experts once per chunk: 4096-token chunks
# prefill 1.9x as fast as 2048 at 2k-32k with decode within noise (M4 Pro, 2026-10-27).
MAX_EXTEND_TOKENS = 4096


def ple_context_state(layer_id: int) -> str:
    """The slot state holding a PLE layer's n-gram context (the last ngram_size - 1 ids)."""
    return f"ple_context.{layer_id}"


def ple_conv_state(layer_id: int) -> str:
    """The slot state holding a PLE layer's short-conv window (its last normed inputs)."""
    return f"ple_conv.{layer_id}"


@dataclass(frozen=True, kw_only=True)
class Qwen4ExpConfig(ModelConfig):
    # Residual streams and the rank of their mixing projections.
    hc_count: int
    hc_lowrank: int
    # Zero-based decoder layers with a PLE branch.
    ple_layer_ids: tuple[int, ...]
    ple_embed_dim: int
    ple_conv_kernel_size: int
    ngram_size: int
    heads_per_ngram: int
    # The n-gram hash restarts after this token.
    ngram_eos_token_id: int

    @property
    def hc_hidden_size(self) -> int:
        return self.hc_count * self.hidden_size

    @property
    def ple_conv_state_len(self) -> int:
        """Inputs the dilated PLE conv reads before a token: (kernel - 1) * ngram_size."""
        return (self.ple_conv_kernel_size - 1) * self.ngram_size


def parse_config(config: dict[str, object], quant: QuantConfig | None = None) -> ModelConfig:
    text = config_dict(config, "text_config") or config
    num_heads = config_int(text, "num_attention_heads")
    hidden_size = config_int(text, "hidden_size")
    head_dim = config_optional_int(text, "head_dim") or hidden_size // num_heads
    num_kv_heads = config_int(text, "num_key_value_heads", num_heads)
    rope_params = config_dict(text, "rope_parameters") or {}
    rotary = RotaryConfig(
        head_dim=head_dim,
        # FreeToken parity: int(), not round().
        rotary_dim=int(head_dim * config_float(rope_params, "partial_rotary_factor", 1.0)),
        max_position=config_int(text, "max_position_embeddings"),
        base=config_float(rope_params, "rope_theta", 10000.0),
    )
    num_layers = config_int(text, "num_hidden_layers")
    types = config_str_list(text, "layer_types")
    if types is None:
        interval = config_int(text, "full_attention_interval", 4)
        types = [
            "full_attention" if (i + 1) % interval == 0 else "linear_attention"
            for i in range(num_layers)
        ]
    qsa_ids = tuple(i for i, t in enumerate(types) if t in ("full_attention", "indexed_attention"))
    linear_ids = tuple(i for i, t in enumerate(types) if t == "linear_attention")
    if len(qsa_ids) + len(linear_ids) != num_layers:
        raise ValueError(f"unsupported layer types {sorted(set(types))}")

    hc_count = config_int(text, "hc_count")
    ngram_size = config_int(text, "ngram_size")
    if ngram_size != 3:
        raise ValueError(f"the PLE hashes bigrams and trigrams: ngram_size 3, not {ngram_size}")
    ple_conv_kernel_size = config_int(text, "ple_conv_kernel_size")
    raw_ple_ids = text.get("ple_layer_ids") or []
    if not isinstance(raw_ple_ids, list) or not all(isinstance(i, int) for i in raw_ple_ids):
        raise ValueError(f"ple_layer_ids must be a list of layer numbers, got {raw_ple_ids!r}")
    # The checkpoint counts layers from 1.
    ple_layer_ids = tuple(sorted(i - 1 for i in raw_ple_ids))
    if any(i not in linear_ids for i in ple_layer_ids):
        raise ValueError(f"PLE layers {ple_layer_ids} must be linear-attention layers")
    slot_states = tuple(
        state
        for layer_id in ple_layer_ids
        for state in (
            SlotStateSpec(
                name=ple_context_state(layer_id), shape=(ngram_size - 1,), dtype=mx.int32
            ),
            SlotStateSpec(
                name=ple_conv_state(layer_id),
                shape=((ple_conv_kernel_size - 1) * ngram_size, hc_count * hidden_size),
            ),
        )
    )
    if config_int(text, "indexer_kv_heads", 1) != 1:
        raise ValueError("QSA's indexer has one key head")
    block_size = config_int(text, "indexer_compress_ratio")
    budget = config_int(text, "indexer_budget")
    if budget % block_size != 0:
        raise ValueError(f"indexer_budget {budget} is not whole {block_size}-key blocks")
    # The checkpoint's output_gate_type, else its hidden_act.
    output_gate = config_optional_str(text, "output_gate_type") or config_str(
        text, "hidden_act", "silu"
    )
    if output_gate not in ("sigmoid", "silu"):
        raise ValueError(f"unsupported GDN output gate {output_gate!r}")
    groups = (
        QSAAttentionGroupConfig(
            name="qsa",
            layer_ids=qsa_ids,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            rotary_config=rotary,
            index_num_heads=config_int(text, "indexer_n_heads"),
            index_head_dim=config_int(text, "indexer_head_dim"),
            index_block_size=block_size,
            index_topk_blocks=budget // block_size,
        ),
        LinearGatedDeltaGroupConfig(
            name="linear",
            layer_ids=linear_ids,
            num_key_heads=config_int(text, "linear_num_key_heads"),
            num_value_heads=config_int(text, "linear_num_value_heads"),
            key_head_dim=config_int(text, "linear_key_head_dim"),
            value_head_dim=config_int(text, "linear_value_head_dim"),
            conv_kernel_dim=config_int(text, "linear_conv_kernel_dim"),
            output_gate="sigmoid" if output_gate == "sigmoid" else "silu",
            slot_states=slot_states,
        ),
    )
    eos = text.get("eos_token_id", config.get("eos_token_id"))
    eos_token_id = eos[0] if isinstance(eos, list) else eos
    if not isinstance(eos_token_id, int):
        raise ValueError("the n-gram hash needs an eos_token_id")
    num_experts = config_int(text, "num_experts")
    architectures = config_str_list(config, "architectures") or ["Qwen4ExpForCausalLM"]
    return Qwen4ExpConfig(
        num_layers=num_layers,
        num_qo_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        hidden_size=hidden_size,
        vocab_size=config_int(text, "vocab_size"),
        intermediate_size=0,
        rms_norm_eps=config_float(text, "rms_norm_eps"),
        rotary_config=rotary,
        hidden_act=config_str(text, "hidden_act", "silu"),
        tie_word_embeddings=config_bool(text, "tie_word_embeddings", False),
        num_experts=num_experts,
        num_experts_per_tok=config_int(text, "num_experts_per_tok"),
        moe_intermediate_size=config_int(text, "moe_intermediate_size"),
        norm_topk_prob=config_bool(text, "norm_topk_prob", True),
        model_type=config_str(config, "model_type", "qwen4_exp"),
        architectures=tuple(architectures),
        attention_groups=groups,
        quant=quant,
        moe_enabled=num_experts > 0,
        shared_expert_intermediate_size=config_int(text, "shared_expert_intermediate_size"),
        use_qk_norm=True,
        attn_output_gate=True,
        prefill_activation_bytes=(
            PREFILL_ACTIVATION_FIXED_BYTES,
            PREFILL_ACTIVATION_BYTES_PER_TOKEN,
        ),
        max_extend_tokens=MAX_EXTEND_TOKENS,
        hc_count=hc_count,
        hc_lowrank=config_int(text, "hc_lowrank"),
        ple_layer_ids=ple_layer_ids,
        ple_embed_dim=config_optional_int(text, "ple_embed_dim") or hidden_size,
        ple_conv_kernel_size=ple_conv_kernel_size,
        ngram_size=ngram_size,
        heads_per_ngram=config_int(text, "heads_per_ngram"),
        ngram_eos_token_id=eos_token_id,
    )


__all__ = ["Qwen4ExpConfig", "parse_config", "ple_context_state", "ple_conv_state"]
