"""A tiny random Qwen3.8-Flash-Next (qwen4_exp) checkpoint and its plain reference forward.

No reference library serves the family, so reference_logits is the model's math written
out over a whole sequence in fp32, without caches: hashed n-gram rows through the PLE
branch, gated delta rule layers (mlx-lm's ops recurrence), QSA layers attending the keys
the block indexer selects (the top blocks by summed relu scores, ties to the lower block,
plus the query's incomplete block; every key while its complete blocks fit), softmax top-k
experts with a gated shared expert, all over hyper-connected residual streams.
save_checkpoint writes the weights in the MLX layout, the PLE table as affine 4-bit shards.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.models.gated_delta import gated_delta_ops

from ftmetal.models.loader import config_dict, config_float, config_int, config_str_list

EOS_TOKEN_ID = 7
# Small primes per n-gram head: the hash's per-head table sizes.
HEAD_SIZES = (101, 103, 107, 109)
NUM_SHARDS = 4
TINY_QWEN4_CONFIG: dict[str, object] = {
    "model_type": "qwen4_exp_text",
    "hidden_size": 64,
    "num_hidden_layers": 4,
    "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 64,
    "rms_norm_eps": 1e-6,
    "vocab_size": 128,
    "max_position_embeddings": 4096,
    # The 3-axis sections are the checkpoint's; text only, the rope is 1-D.
    "rope_parameters": {
        "type": "default",
        "rope_theta": 10000,
        "partial_rotary_factor": 0.25,
        "mrope_section": [3, 3, 2],
        "mrope_interleaved": True,
    },
    "linear_num_key_heads": 4,
    "linear_num_value_heads": 12,
    "linear_key_head_dim": 32,
    "linear_value_head_dim": 32,
    "linear_conv_kernel_dim": 4,
    "output_gate_type": "sigmoid",
    "num_experts": 8,
    "num_experts_per_tok": 2,
    "moe_intermediate_size": 32,
    "shared_expert_intermediate_size": 32,
    "hc_count": 4,
    "hc_lowrank": 16,
    "indexer_n_heads": 2,
    "indexer_kv_heads": 1,
    "indexer_head_dim": 32,
    "indexer_compress_ratio": 4,
    # 4 blocks: a query past its 16th key attends a subset.
    "indexer_budget": 16,
    "ple_layer_ids": [2],
    # 32-wide rows: the table is stored as affine 4-bit shards in groups of 32, as the real one.
    "ple_embed_dim": 128,
    "ple_conv_kernel_size": 4,
    "ngram_size": 3,
    "heads_per_ngram": 2,
    "eos_token_id": EOS_TOKEN_ID,
    "tie_word_embeddings": False,
}


def ple_layer_ids(text: dict[str, object]) -> list[int]:
    """The zero-based PLE layers of a config that counts them from 1."""
    raw = text["ple_layer_ids"]
    assert isinstance(raw, list)
    return [i - 1 for i in raw if isinstance(i, int)]


def random_weights(seed: int, config: dict[str, object] | None = None) -> dict[str, mx.array]:
    """Random fp32 weights under the MLX layout's checkpoint names."""
    text = TINY_QWEN4_CONFIG if config is None else config
    rng = np.random.default_rng(seed)
    hidden = config_int(text, "hidden_size")
    hc = config_int(text, "hc_count")
    width = hc * hidden
    lowrank = config_int(text, "hc_lowrank")
    num_heads = config_int(text, "num_attention_heads")
    num_kv = config_int(text, "num_key_value_heads")
    head_dim = config_int(text, "head_dim")
    hk, hv = config_int(text, "linear_num_key_heads"), config_int(text, "linear_num_value_heads")
    dk, dv = config_int(text, "linear_key_head_dim"), config_int(text, "linear_value_head_dim")
    conv_dim = 2 * hk * dk + hv * dv
    experts = config_int(text, "num_experts")
    moe = config_int(text, "moe_intermediate_size")
    shared = config_int(text, "shared_expert_intermediate_size")
    index_heads, index_dim = (
        config_int(text, "indexer_n_heads"),
        config_int(text, "indexer_head_dim"),
    )
    vocab = config_int(text, "vocab_size")
    embed_dim = config_int(text, "ple_embed_dim")
    num_ngram_heads = (config_int(text, "ngram_size") - 1) * config_int(text, "heads_per_ngram")
    ple_layers = ple_layer_ids(text)
    types = config_str_list(text, "layer_types")
    assert types is not None

    def linear(out_size: int, in_size: int) -> np.ndarray:
        return rng.normal(0.0, 1.0 / math.sqrt(in_size), (out_size, in_size))

    def centred(size: int) -> np.ndarray:
        return rng.normal(0.0, 0.1, (size,))

    out: dict[str, np.ndarray] = {
        "language_model.model.embed_tokens.weight": rng.normal(0.0, 1.0, (vocab, hidden)),
        "language_model.lm_head.weight": linear(vocab, hidden),
    }

    def hyper_connection(prefix: str, with_inject: bool) -> None:
        out[f"{prefix}.hc_norm.weight"] = centred(width)
        out[f"{prefix}.input_mix_weight_down.weight"] = linear(lowrank, width)
        out[f"{prefix}.input_mix_weight_up.weight"] = linear(width, lowrank)
        if with_inject:
            out[f"{prefix}.block_inject_weight.weight"] = linear(hc, width)

    hyper_connection("language_model.model.hyper_connection_mixer", with_inject=False)
    for layer_id, layer_type in enumerate(types):
        p = f"language_model.model.layers.{layer_id}"
        hyper_connection(f"{p}.attn_hyper_connection", with_inject=True)
        hyper_connection(f"{p}.mlp_hyper_connection", with_inject=True)
        if layer_type == "linear_attention":
            out[f"{p}.linear_attn.in_proj_qkv.weight"] = linear(conv_dim, hidden)
            out[f"{p}.linear_attn.in_proj_z.weight"] = linear(hv * dv, hidden)
            out[f"{p}.linear_attn.in_proj_b.weight"] = linear(hv, hidden)
            out[f"{p}.linear_attn.in_proj_a.weight"] = linear(hv, hidden)
            out[f"{p}.linear_attn.conv1d.weight"] = rng.normal(0.0, 0.5, (conv_dim, 4, 1))
            out[f"{p}.linear_attn.A_log"] = rng.uniform(-1.0, 1.0, (hv,))
            out[f"{p}.linear_attn.dt_bias"] = rng.uniform(-1.0, 1.0, (hv,))
            out[f"{p}.linear_attn.norm.weight"] = 1.0 + centred(dv)
            out[f"{p}.linear_attn.out_proj.weight"] = linear(hidden, hv * dv)
        else:
            out[f"{p}.self_attn.q_proj.weight"] = linear(2 * num_heads * head_dim, hidden)
            out[f"{p}.self_attn.k_proj.weight"] = linear(num_kv * head_dim, hidden)
            out[f"{p}.self_attn.v_proj.weight"] = linear(num_kv * head_dim, hidden)
            out[f"{p}.self_attn.o_proj.weight"] = linear(hidden, num_heads * head_dim)
            out[f"{p}.self_attn.q_norm.weight"] = centred(head_dim)
            out[f"{p}.self_attn.k_norm.weight"] = centred(head_dim)
            out[f"{p}.self_attn.indexer.index_qk_proj.weight"] = linear(
                (index_heads + 1) * index_dim, hidden
            )
            out[f"{p}.self_attn.indexer.q_layernorm.weight"] = centred(index_dim)
            out[f"{p}.self_attn.indexer.k_layernorm.weight"] = centred(index_dim)
        out[f"{p}.mlp.gate.weight"] = linear(experts, hidden)
        out[f"{p}.mlp.switch_mlp.gate_proj.weight"] = np.stack(
            [linear(moe, hidden) for _ in range(experts)]
        )
        out[f"{p}.mlp.switch_mlp.up_proj.weight"] = np.stack(
            [linear(moe, hidden) for _ in range(experts)]
        )
        out[f"{p}.mlp.switch_mlp.down_proj.weight"] = np.stack(
            [linear(hidden, moe) for _ in range(experts)]
        )
        out[f"{p}.mlp.shared_expert.gate_proj.weight"] = linear(shared, hidden)
        out[f"{p}.mlp.shared_expert.up_proj.weight"] = linear(shared, hidden)
        out[f"{p}.mlp.shared_expert.down_proj.weight"] = linear(hidden, shared)
        out[f"{p}.mlp.shared_expert_gate.weight"] = linear(1, hidden)
        if layer_id in ple_layers:
            out[f"{p}.ple.key_proj.weight"] = linear(width, embed_dim)
            out[f"{p}.ple.value_proj.weight"] = linear(hidden, embed_dim)
            for norm in ("norm_key", "norm_query", "norm_conv"):
                out[f"{p}.ple.{norm}.weight"] = centred(width)
            out[f"{p}.ple.conv1d.weight"] = rng.normal(0.0, 0.5, (width, 4, 1))
            num_rows = sum(HEAD_SIZES)
            table = rng.normal(0.0, 1.0, (num_rows, embed_dim // num_ngram_heads))
            prefix = f"{p}.ple.ple_embedding"
            for shard, rows in enumerate(np.array_split(table, NUM_SHARDS)):
                out[f"{prefix}.ngram_embedding.shards.{shard}.weight"] = rows
    weights: dict[str, mx.array] = {}
    for key, value in out.items():
        tensor = mx.array(value.astype(np.float32))
        if ".ngram_embedding.shards." in key:
            module = key.removesuffix(".weight")
            packed, scales, biases = mx.quantize(tensor, group_size=32, bits=4)
            weights[key], weights[f"{module}.scales"], weights[f"{module}.biases"] = (
                packed,
                scales,
                biases,
            )
        else:
            weights[key] = tensor
    for layer_id in ple_layers:
        prefix = f"language_model.model.layers.{layer_id}.ple.ple_embedding"
        # Large enough that the hash products wrap around int64.
        multipliers = 2 * rng.integers(1 << 59, 1 << 61, size=3) + 1
        weights[f"{prefix}.layer_multipliers"] = mx.array(multipliers.astype(np.int64))
        weights[f"{prefix}.ngram_heads_vocab_sizes"] = mx.array(np.array(HEAD_SIZES, np.int64))
        offsets = np.concatenate([[0], np.cumsum(HEAD_SIZES)[:-1]]).astype(np.int64)
        weights[f"{prefix}.ngram_heads_offsets"] = mx.array(offsets)
    return weights


# Wide enough for the one-token kernels: 4 streams of 128 make the hyper-connection's 512-wide
# inputs qmv_rows takes, and 32-wide PLE rows quantize in groups of 32.
TINY_QWEN4_WIDE_CONFIG: dict[str, object] = {
    **TINY_QWEN4_CONFIG,
    "hidden_size": 128,
    "hc_lowrank": 32,
    "ple_embed_dim": 128,
}
# The real checkpoint's recipe: affine 4-bit in groups of 32, the routers 8-bit in groups of 64.
QUANTIZATION: dict[str, object] = {"group_size": 32, "bits": 4, "mode": "affine"}
ROUTER_QUANTIZATION: dict[str, object] = {"group_size": 64, "bits": 8, "mode": "affine"}


def quantize_weights(
    weights: dict[str, mx.array],
) -> tuple[dict[str, mx.array], dict[str, mx.array], dict[str, object]]:
    """(the checkpoint's tensors, the weights they dequantize to, its quantization section):
    every matrix whose rows split into groups is quantized, the PLE table shards too."""
    stored: dict[str, mx.array] = {}
    dequantized: dict[str, mx.array] = {}
    quantization = dict(QUANTIZATION)
    for key, value in weights.items():
        module = key.removesuffix(".weight")
        is_router = module.endswith(("mlp.gate", "mlp.shared_expert_gate"))
        scheme = ROUTER_QUANTIZATION if is_router else QUANTIZATION
        group_size, bits = config_int(scheme, "group_size"), config_int(scheme, "bits")
        if (
            not key.endswith(".weight")
            or not mx.issubdtype(value.dtype, mx.floating)
            or value.ndim < 2
            or value.shape[-1] % group_size != 0
        ):
            stored[key] = value
            dequantized[key] = value
            continue
        packed, scales, biases = mx.quantize(value, group_size=group_size, bits=bits)
        stored[key], stored[f"{module}.scales"], stored[f"{module}.biases"] = packed, scales, biases
        dequantized[key] = mx.dequantize(packed, scales, biases, group_size=group_size, bits=bits)
        if is_router:
            quantization[module] = dict(scheme)
    return stored, dequantized, quantization


def save_checkpoint(
    weights: dict[str, mx.array],
    path: Path,
    config: dict[str, object] | None = None,
    quantization: dict[str, object] | None = None,
) -> None:
    mx.save_safetensors(str(path / "model.safetensors"), weights)
    full_config: dict[str, object] = {
        "architectures": ["Qwen4ExpForConditionalGeneration"],
        "model_type": "qwen4_exp",
        "text_config": TINY_QWEN4_CONFIG if config is None else config,
        "eos_token_id": [127, EOS_TOKEN_ID],
    }
    if quantization is not None:
        # MLX checkpoints write the section twice, the second time as quantization_config.
        full_config["quantization"] = quantization
        full_config["quantization_config"] = quantization
    (path / "config.json").write_text(json.dumps(full_config))


def centred_norm(x: mx.array, weight: mx.array, eps: float, group_size: int) -> mx.array:
    y = x.reshape(*x.shape[:-1], x.shape[-1] // group_size, group_size)
    y = y * mx.rsqrt(mx.mean(y * y, axis=-1, keepdims=True) + eps)
    return y.reshape(x.shape) * (1.0 + weight)


def rope(x: mx.array, positions: mx.array, rotary_dim: int, base: float) -> mx.array:
    """NeoX rotation of the first rotary_dim dims of x [L, heads, D] at positions [L]."""
    half = rotary_dim // 2
    inv_freq = base ** (-mx.arange(0, rotary_dim, 2, dtype=mx.float32) / rotary_dim)
    angles = positions.astype(mx.float32)[:, None] * inv_freq[None]
    cos, sin = mx.cos(angles)[:, None], mx.sin(angles)[:, None]
    x1, x2 = x[..., :half], x[..., half:rotary_dim]
    return mx.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin, x[..., rotary_dim:]], axis=-1)


def dilated_conv(x: mx.array, weight: mx.array, dilation: int) -> mx.array:
    """Causal depthwise conv of x [L, C] by weight [C, K, 1]: sum over taps k of
    weight[:, k] * x[t - (K - 1 - k) * dilation], zero before the sequence."""
    taps = weight.shape[1]
    span = (taps - 1) * dilation
    padded = mx.concatenate([mx.zeros((span, x.shape[1])), x])
    length = x.shape[0]
    out = mx.zeros_like(x)
    for k in range(taps):
        out = out + weight[:, k, 0] * padded[k * dilation : k * dilation + length]
    return out


def topk_lowest_ties(scores: np.ndarray, topk: int) -> np.ndarray:
    """Indices of the topk highest finite scores, ties to the lower index."""
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    return np.array([i for i in order[:topk] if np.isfinite(scores[i])], dtype=np.int64)


def reference_logits(
    weights: dict[str, mx.array], tokens: list[int], config: dict[str, object] | None = None
) -> mx.array:
    """fp32 logits [L, vocab] of the whole sequence tokens."""
    text = TINY_QWEN4_CONFIG if config is None else config
    eps = config_float(text, "rms_norm_eps")
    hidden = config_int(text, "hidden_size")
    hc = config_int(text, "hc_count")
    head_dim = config_int(text, "head_dim")
    num_heads, num_kv = (
        config_int(text, "num_attention_heads"),
        config_int(text, "num_key_value_heads"),
    )
    rope_params = config_dict(text, "rope_parameters")
    assert rope_params is not None
    rotary_dim = int(head_dim * config_float(rope_params, "partial_rotary_factor"))
    base = config_float(rope_params, "rope_theta")
    hk, hv = config_int(text, "linear_num_key_heads"), config_int(text, "linear_num_value_heads")
    dk, dv = config_int(text, "linear_key_head_dim"), config_int(text, "linear_value_head_dim")
    index_heads, index_dim = (
        config_int(text, "indexer_n_heads"),
        config_int(text, "indexer_head_dim"),
    )
    block = config_int(text, "indexer_compress_ratio")
    topk = config_int(text, "indexer_budget") // block
    top_experts = config_int(text, "num_experts_per_tok")
    ngram_size = config_int(text, "ngram_size")
    heads_per_ngram = config_int(text, "heads_per_ngram")
    ple_layers = ple_layer_ids(text)
    types = config_str_list(text, "layer_types")
    assert types is not None
    length = len(tokens)
    positions = mx.arange(length)
    w = weights

    def proj(x: mx.array, key: str) -> mx.array:
        return x @ w[key].T

    def gated_residual(streams: mx.array, prefix: str) -> tuple[mx.array, mx.array | None]:
        normed = centred_norm(streams, w[f"{prefix}.hc_norm.weight"], eps, hidden)
        down = nn.silu(proj(normed, f"{prefix}.input_mix_weight_down.weight") / hc)
        mix = mx.sigmoid(proj(down, f"{prefix}.input_mix_weight_up.weight")).reshape(
            length, hc, hidden
        )
        mixed = mx.mean(mix * normed.reshape(length, hc, hidden), axis=1)
        key = f"{prefix}.block_inject_weight.weight"
        if key not in w:
            return mixed, None
        return mixed, 2 * mx.sigmoid(proj(normed, key) / hc)

    def ple(streams: mx.array, p: str) -> mx.array:
        prefix = f"{p}.ple.ple_embedding"
        multipliers = np.array(w[f"{prefix}.layer_multipliers"]).astype(np.int64)
        sizes = np.array(w[f"{prefix}.ngram_heads_vocab_sizes"]).astype(np.int64)
        offsets = np.array(w[f"{prefix}.ngram_heads_offsets"]).astype(np.int64)
        table = mx.concatenate(
            [
                mx.dequantize(
                    *(
                        w[f"{prefix}.ngram_embedding.shards.{shard}.{part}"]
                        for part in ("weight", "scales", "biases")
                    ),
                    group_size=32,
                    bits=4,
                )
                for shard in range(NUM_SHARDS)
            ]
        )
        history = np.array([EOS_TOKEN_ID] * (ngram_size - 1) + tokens, dtype=np.int64)
        # The id shift positions back, or eos when that crosses an eos or the start.
        shifted = []
        for shift in range(ngram_size):
            column = []
            for t in range(len(history)):
                earlier_eos = [j for j in range(t) if history[j] == EOS_TOKEN_ID]
                last_eos = earlier_eos[-1] if earlier_eos else -1
                is_valid = t - shift >= 0 and t - shift > last_eos
                column.append(history[t - shift] if is_valid else EOS_TOKEN_ID)
            shifted.append(np.array(column, dtype=np.int64)[-length:])
        bigram = shifted[0] * multipliers[0] ^ shifted[1] * multipliers[1]
        trigram = bigram ^ shifted[2] * multipliers[2]
        mixed = np.stack([bigram] * heads_per_ngram + [trigram] * heads_per_ngram, axis=1)
        ids = mixed % sizes[None] + offsets[None]
        embedded = table[mx.array(ids)].reshape(length, -1)
        keys = centred_norm(
            proj(embedded, f"{p}.ple.key_proj.weight"), w[f"{p}.ple.norm_key.weight"], eps, hidden
        )
        values = proj(embedded, f"{p}.ple.value_proj.weight")
        queries = centred_norm(streams, w[f"{p}.ple.norm_query.weight"], eps, hidden)
        scores = mx.sum(
            keys.reshape(length, hc, hidden) * queries.reshape(length, hc, hidden),
            axis=-1,
            keepdims=True,
        ) / math.sqrt(hidden)
        scores = mx.sign(scores) * mx.sqrt(mx.maximum(mx.abs(scores), 1e-6))
        gated = (mx.sigmoid(scores) * values[:, None, :]).reshape(length, hc * hidden)
        normed = centred_norm(gated, w[f"{p}.ple.norm_conv.weight"], eps, hidden)
        return gated + nn.silu(dilated_conv(normed, w[f"{p}.ple.conv1d.weight"], ngram_size))

    def gdn(x: mx.array, p: str) -> mx.array:
        qkv = nn.silu(dilated_conv(proj(x, f"{p}.in_proj_qkv.weight"), w[f"{p}.conv1d.weight"], 1))
        q = qkv[:, : hk * dk].reshape(1, length, hk, dk)
        k = qkv[:, hk * dk : 2 * hk * dk].reshape(1, length, hk, dk)
        v = qkv[:, 2 * hk * dk :].reshape(1, length, hv, dv)
        q = q * mx.rsqrt(mx.sum(q * q, axis=-1, keepdims=True) + 1e-6) * dk**-0.5
        k = k * mx.rsqrt(mx.sum(k * k, axis=-1, keepdims=True) + 1e-6)
        beta = mx.sigmoid(proj(x, f"{p}.in_proj_b.weight"))[None]
        a = proj(x, f"{p}.in_proj_a.weight")
        g = mx.exp(-mx.exp(w[f"{p}.A_log"]) * nn.softplus(a + w[f"{p}.dt_bias"]))[None]
        y, _ = gated_delta_ops(q, k, v, g, beta)
        y = y[0]
        z = proj(x, f"{p}.in_proj_z.weight").reshape(length, hv, dv)
        normed = y * mx.rsqrt(mx.mean(y * y, axis=-1, keepdims=True) + eps) * w[f"{p}.norm.weight"]
        return proj((normed * mx.sigmoid(z)).reshape(length, hv * dv), f"{p}.out_proj.weight")

    def qsa(x: mx.array, p: str) -> mx.array:
        qg = proj(x, f"{p}.q_proj.weight").reshape(length, num_heads, 2 * head_dim)
        q, gate = qg[..., :head_dim], qg[..., head_dim:]
        k = proj(x, f"{p}.k_proj.weight").reshape(length, num_kv, head_dim)
        v = proj(x, f"{p}.v_proj.weight").reshape(length, num_kv, head_dim)
        q = rope(
            centred_norm(q, w[f"{p}.q_norm.weight"], eps, head_dim), positions, rotary_dim, base
        )
        k = rope(
            centred_norm(k, w[f"{p}.k_norm.weight"], eps, head_dim), positions, rotary_dim, base
        )
        qk = proj(x, f"{p}.indexer.index_qk_proj.weight").reshape(
            length, index_heads + 1, index_dim
        )
        index_queries = rope(
            centred_norm(qk[:, :index_heads], w[f"{p}.indexer.q_layernorm.weight"], eps, index_dim),
            positions,
            rotary_dim,
            base,
        )
        num_blocks = length // block
        pooled = mx.mean(
            qk[: num_blocks * block, index_heads].reshape(num_blocks, block, index_dim), axis=1
        )
        block_keys = rope(
            centred_norm(pooled[:, None], w[f"{p}.indexer.k_layernorm.weight"], eps, index_dim),
            mx.arange(num_blocks) * block,
            rotary_dim,
            base,
        )[:, 0]
        scores = mx.maximum(mx.einsum("thd,nd->thn", index_queries, block_keys), 0).sum(axis=1)
        scores = np.array(scores / math.sqrt(index_dim))
        visible = np.zeros((length, length), dtype=bool)
        for t in range(length):
            complete = (t + 1) // block
            if complete <= topk:
                visible[t, : t + 1] = True
                continue
            for b in topk_lowest_ties(scores[t, :complete], topk):
                visible[t, b * block : (b + 1) * block] = True
            visible[t, complete * block : t + 1] = True
        repeat = num_heads // num_kv
        keys = mx.repeat(k, repeat, axis=1).transpose(1, 0, 2)
        values = mx.repeat(v, repeat, axis=1).transpose(1, 0, 2)
        logits = (q.transpose(1, 0, 2) @ keys.transpose(0, 2, 1)) * head_dim**-0.5
        logits = mx.where(mx.array(visible)[None], logits, -mx.inf)
        out = (mx.softmax(logits, axis=-1) @ values).transpose(1, 0, 2)
        out = out * mx.sigmoid(gate)
        return proj(out.reshape(length, num_heads * head_dim), f"{p}.o_proj.weight")

    def moe(x: mx.array, p: str) -> mx.array:
        probs = np.array(mx.softmax(proj(x, f"{p}.gate.weight"), axis=-1))
        routed = []
        for t in range(length):
            chosen = topk_lowest_ties(probs[t], top_experts)
            weights_t = probs[t, chosen] / probs[t, chosen].sum()
            total = mx.zeros((hidden,))
            for expert, weight in zip(chosen.tolist(), weights_t.tolist(), strict=True):
                gate = x[t] @ w[f"{p}.switch_mlp.gate_proj.weight"][expert].T
                up = x[t] @ w[f"{p}.switch_mlp.up_proj.weight"][expert].T
                down = (nn.silu(gate) * up) @ w[f"{p}.switch_mlp.down_proj.weight"][expert].T
                total = total + weight * down
            routed.append(total)
        shared = proj(
            nn.silu(proj(x, f"{p}.shared_expert.gate_proj.weight"))
            * proj(x, f"{p}.shared_expert.up_proj.weight"),
            f"{p}.shared_expert.down_proj.weight",
        )
        return mx.stack(routed) + mx.sigmoid(proj(x, f"{p}.shared_expert_gate.weight")) * shared

    streams = mx.tile(w["language_model.model.embed_tokens.weight"][mx.array(tokens)], (1, hc))
    for layer_id, layer_type in enumerate(types):
        p = f"language_model.model.layers.{layer_id}"
        if layer_id in ple_layers:
            streams = streams + ple(streams, p)
        mixed, gates = gated_residual(streams, f"{p}.attn_hyper_connection")
        if layer_type == "linear_attention":
            branch = gdn(mixed, f"{p}.linear_attn")
        else:
            branch = qsa(mixed, f"{p}.self_attn")
        assert gates is not None
        streams = (
            streams.reshape(length, hc, hidden) + gates[:, :, None] * branch[:, None]
        ).reshape(length, hc * hidden)
        mixed, gates = gated_residual(streams, f"{p}.mlp_hyper_connection")
        branch = moe(mixed, f"{p}.mlp")
        assert gates is not None
        streams = (
            streams.reshape(length, hc, hidden) + gates[:, :, None] * branch[:, None]
        ).reshape(length, hc * hidden)
    final, _ = gated_residual(streams, "language_model.model.hyper_connection_mixer")
    return proj(final, "language_model.lm_head.weight")


def reference_greedy(
    weights: dict[str, mx.array],
    prompt: list[int],
    steps: int,
    config: dict[str, object] | None = None,
) -> tuple[mx.array, list[int]]:
    """(the prompt's logits, steps greedy tokens), each step a whole-sequence forward."""
    tokens = list(prompt)
    prompt_logits = reference_logits(weights, tokens, config)
    generated = [int(mx.argmax(prompt_logits[-1]))]
    for _ in range(steps - 1):
        tokens.append(generated[-1])
        generated.append(int(mx.argmax(reference_logits(weights, tokens, config)[-1])))
    return prompt_logits, generated
