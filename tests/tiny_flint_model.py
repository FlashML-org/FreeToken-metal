"""A tiny random Flint checkpoint and its reference model.

The pinned mlx-lm predates the family, so Model is its definition written out over mlx-lm's caches,
masks and expert layer: full layers without rope, sliding layers with rope and a window, a learned
sink per query head, two leading dense MLP layers, then sigmoid-routed experts beside an ungated
shared expert. Sinks and norms are perturbed, since an initial zero sink would hide a kernel that
drops it. save_checkpoint writes the MLX layout; quantized, as flint-int4 stores it: group-64
affine 4-bit, the embedding in 8 bits only its export section declares, the router unquantized and
the packed weights as bytes. save_conversion writes it as the conversions that nest the text model
under language_model. do.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.cache import KVCache, RotatingKVCache
from mlx_lm.models.switch_layers import SwitchGLU

from ftmetal.models.loader import config_int

EOS_TOKEN_ID = 127
WINDOW = 8
GROUP_SIZE = 64
BITS = 4
EMBEDDING_BITS = 8
# The conversions quantize the router, in 8 bits, and nest the text model under this prefix.
ROUTER_BITS = 8
CONVERSION_PREFIX = "language_model."
SINK_SCALE = 2.0
NORM_NOISE = 0.1
LAYER_TYPES = [
    "full_attention",
    "full_attention",
    "sliding_attention",
    "sliding_attention",
    "full_attention",
    "sliding_attention",
]
TINY_FLINT_CONFIG: dict[str, object] = {
    "architectures": ["FlintForCausalLM"],
    "model_type": "flint",
    "hidden_size": 128,
    "num_hidden_layers": len(LAYER_TYPES),
    "layer_types": LAYER_TYPES,
    "no_rope_layer_indices": [0, 1, 4],
    "mlp_only_layers": [0, 1],
    "sliding_window": WINDOW,
    "intermediate_size": 192,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 64,
    "num_experts": 8,
    "num_experts_per_tok": 2,
    "moe_intermediate_size": 64,
    "shared_expert_intermediate_size": 128,
    "rms_norm_eps": 1e-6,
    "vocab_size": 128,
    "max_position_embeddings": 4096,
    "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"},
    "hidden_act": "silu",
    "router_score_function": "sigmoid",
    "softmax_type": "learnable",
    "norm_topk_prob": True,
    "tie_word_embeddings": False,
    "eos_token_id": [EOS_TOKEN_ID],
}


class Attention(nn.Module):
    def __init__(self, config: dict[str, object], layer_id: int) -> None:
        super().__init__()
        hidden, head_dim = config_int(config, "hidden_size"), config_int(config, "head_dim")
        self.num_q, self.num_kv = (
            config_int(config, "num_attention_heads"),
            config_int(config, "num_key_value_heads"),
        )
        self.scale = head_dim**-0.5
        self.q_proj = nn.Linear(hidden, self.num_q * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden, self.num_kv * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, self.num_kv * head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_q * head_dim, hidden, bias=False)
        self.q_norm = nn.RMSNorm(head_dim, eps=1e-6)
        self.k_norm = nn.RMSNorm(head_dim, eps=1e-6)
        self.softmax_offset = mx.zeros((self.num_q,))
        self.rope = (
            nn.RoPE(head_dim, traditional=False, base=10000.0)
            if (LAYER_TYPES[layer_id] == "sliding_attention")
            else None
        )

    def __call__(
        self, x: mx.array, mask: mx.array | str | None, cache: KVCache | RotatingKVCache
    ) -> mx.array:
        batch, length, _ = x.shape
        q = self.q_norm(self.q_proj(x).reshape(batch, length, self.num_q, -1)).transpose(0, 2, 1, 3)
        k = self.k_norm(self.k_proj(x).reshape(batch, length, self.num_kv, -1)).transpose(
            0, 2, 1, 3
        )
        v = self.v_proj(x).reshape(batch, length, self.num_kv, -1).transpose(0, 2, 1, 3)
        if self.rope is not None:
            q = self.rope(q, offset=cache.offset)
            k = self.rope(k, offset=cache.offset)
        k, v = cache.update_and_fetch(k, v)
        assert k is not None and v is not None
        out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.scale, mask=mask, sinks=self.softmax_offset
        )
        return self.o_proj(out.transpose(0, 2, 1, 3).reshape(batch, length, -1))


class MLP(nn.Module):
    def __init__(self, hidden: int, intermediate: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class SparseMoeBlock(nn.Module):
    def __init__(self, config: dict[str, object]) -> None:
        super().__init__()
        hidden, num_experts = config_int(config, "hidden_size"), config_int(config, "num_experts")
        self.top_k = config_int(config, "num_experts_per_tok")
        self.gate = nn.Linear(hidden, num_experts, bias=False)
        self.switch_mlp = SwitchGLU(
            hidden, config_int(config, "moe_intermediate_size"), num_experts
        )
        self.shared_expert = MLP(hidden, config_int(config, "shared_expert_intermediate_size"))

    def __call__(self, x: mx.array) -> mx.array:
        scores = mx.sigmoid(self.gate(x))
        ids = mx.argpartition(-scores, kth=self.top_k - 1, axis=-1)[..., : self.top_k]
        weights = mx.take_along_axis(scores, ids, axis=-1)
        weights = weights / weights.sum(axis=-1, keepdims=True)
        routed = (self.switch_mlp(x, ids) * weights[..., None]).sum(axis=-2)
        return routed + self.shared_expert(x)


class DecoderLayer(nn.Module):
    def __init__(self, config: dict[str, object], layer_id: int) -> None:
        super().__init__()
        hidden = config_int(config, "hidden_size")
        self.self_attn = Attention(config, layer_id)
        dense_ids = config["mlp_only_layers"]
        assert isinstance(dense_ids, list)
        self.mlp = (
            MLP(hidden, config_int(config, "intermediate_size"))
            if layer_id in dense_ids
            else SparseMoeBlock(config)
        )
        self.input_layernorm = nn.RMSNorm(hidden, eps=1e-6)
        self.post_attention_layernorm = nn.RMSNorm(hidden, eps=1e-6)

    def __call__(
        self, x: mx.array, mask: mx.array | str | None, cache: KVCache | RotatingKVCache
    ) -> mx.array:
        h = x + self.self_attn(self.input_layernorm(x), mask, cache)
        return h + self.mlp(self.post_attention_layernorm(h))


class Body(nn.Module):
    def __init__(self, config: dict[str, object]) -> None:
        super().__init__()
        hidden = config_int(config, "hidden_size")
        self.embed_tokens = nn.Embedding(config_int(config, "vocab_size"), hidden)
        self.layers = [DecoderLayer(config, i) for i in range(len(LAYER_TYPES))]
        self.norm = nn.RMSNorm(hidden, eps=1e-6)


class Model(nn.Module):
    def __init__(self, config: dict[str, object]) -> None:
        super().__init__()
        self.model = Body(config)
        self.lm_head = nn.Linear(
            config_int(config, "hidden_size"), config_int(config, "vocab_size"), bias=False
        )

    def make_cache(self) -> list[KVCache | RotatingKVCache]:
        return [
            RotatingKVCache(max_size=WINDOW) if kind == "sliding_attention" else KVCache()
            for kind in LAYER_TYPES
        ]

    def __call__(self, tokens: mx.array, cache: list[KVCache | RotatingKVCache]) -> mx.array:
        h = self.model.embed_tokens(tokens)
        full_mask = create_attention_mask(h, cache[LAYER_TYPES.index("full_attention")])
        window_mask = create_attention_mask(
            h, cache[LAYER_TYPES.index("sliding_attention")], window_size=WINDOW
        )
        for layer, kind, layer_cache in zip(self.model.layers, LAYER_TYPES, cache, strict=True):
            h = layer(h, window_mask if kind == "sliding_attention" else full_mask, layer_cache)
        return self.lm_head(self.model.norm(h))


def build_reference(seed: int, config: dict[str, object] | None = None) -> Model:
    """A random fp32 reference with every norm and sink away from its initial value."""
    mx.random.seed(seed)
    model = Model(TINY_FLINT_CONFIG if config is None else config)
    params: dict[str, mx.array] = {}
    for key, value in tree_flatten(model.parameters()):
        assert isinstance(value, mx.array)
        if key.endswith("softmax_offset"):
            value = SINK_SCALE * mx.random.normal(value.shape)
        elif key.endswith("norm.weight"):
            value = value + NORM_NOISE * mx.random.normal(value.shape)
        params[key] = value
    model.load_weights(list(params.items()))
    model.set_dtype(mx.float32)
    mx.eval(model.parameters())
    return model


def quantize(model: Model, router_bits: int | None = None) -> None:
    """The flint-int4 recipe: 4-bit in groups of 64, the embedding in 8, the router unquantized,
    or in router_bits."""

    def predicate(path: str, module: nn.Module) -> bool | dict[str, object]:
        if not hasattr(module, "to_quantized") or (
            path.endswith("mlp.gate") and router_bits is None
        ):
            return False
        if path.endswith("embed_tokens"):
            return {"group_size": GROUP_SIZE, "bits": EMBEDDING_BITS}
        if path.endswith("mlp.gate"):
            return {"group_size": GROUP_SIZE, "bits": router_bits}
        return True

    nn.quantize(model, GROUP_SIZE, BITS, class_predicate=predicate)
    mx.eval(model.parameters())


def save_checkpoint(
    model: Model, path: Path, is_quantized: bool, config: dict[str, object] | None = None
) -> None:
    """The MLX layout, the quantized one with its packed weights as bytes."""
    state = dict(tree_flatten(model.parameters()))
    saved_config = dict(TINY_FLINT_CONFIG if config is None else config)
    if is_quantized:
        state = {
            key: value.view(mx.uint8)
            if key.endswith(".weight") and f"{key.removesuffix('.weight')}.scales" in state
            else value
            for key, value in state.items()
        }
        saved_config["quantization"] = {"group_size": GROUP_SIZE, "bits": BITS}
        saved_config["mlx_export"] = {"bits": BITS, "embedding_bits": EMBEDDING_BITS}
    mx.save_safetensors(str(path / "model.safetensors"), state, metadata={"format": "mlx"})
    (path / "config.json").write_text(json.dumps(saved_config))


def save_conversion(model: Model, path: Path, config: dict[str, object] | None = None) -> None:
    """The MLX layout as conversions store it: every name under language_model., a zero router
    bias per MoE layer, packed weights as words and, quantized, the modules not in the default
    scheme named under the prefix in the quantization section, with no export section."""
    saved_config = dict(TINY_FLINT_CONFIG if config is None else config)
    state = {CONVERSION_PREFIX + key: value for key, value in tree_flatten(model.parameters())}
    dense_ids = saved_config["mlp_only_layers"]
    assert isinstance(dense_ids, list)
    for layer_id in range(config_int(saved_config, "num_hidden_layers")):
        if layer_id not in dense_ids:
            bias = mx.zeros((config_int(saved_config, "num_experts"),))
            state[f"{CONVERSION_PREFIX}model.layers.{layer_id}.mlp.expert_bias"] = bias
    quantized = [
        (module_path, module)
        for module_path, module in model.named_modules()
        if isinstance(module, (nn.QuantizedLinear, nn.QuantizedEmbedding))
    ]
    if quantized:
        quantization: dict[str, object] = {"group_size": GROUP_SIZE, "bits": BITS}
        for module_path, module in quantized:
            if module.bits != BITS:
                entry = {"group_size": module.group_size, "bits": module.bits}
                quantization[CONVERSION_PREFIX + module_path] = entry
        saved_config["quantization"] = quantization
    mx.save_safetensors(str(path / "model.safetensors"), state, metadata={"format": "mlx"})
    (path / "config.json").write_text(json.dumps(saved_config))


def reference_greedy(model: Model, prompt: list[int], steps: int) -> tuple[mx.array, list[int]]:
    """The prefill's logits [prompt, vocab] and the greedy tokens."""
    cache = model.make_cache()
    logits = model(mx.array(prompt)[None], cache)
    prefill_logits = logits[0]
    token = int(mx.argmax(logits[0, -1]))
    tokens = [token]
    for _ in range(steps - 1):
        logits = model(mx.array([[token]]), cache)
        token = int(mx.argmax(logits[0, -1]))
        tokens.append(token)
    return prefill_logits, tokens
