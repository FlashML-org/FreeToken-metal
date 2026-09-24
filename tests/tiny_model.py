"""A tiny random Qwen3.5 MoE model shared by the parity and scheduler tests.

The mlx-lm model is the reference; save_checkpoint writes it in the MLX checkpoint
layout ftmetal loads, and reference_greedy is the contiguous-cache greedy decode.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_lm.models import qwen3_5 as reference

TINY_TEXT_CONFIG = {
    "model_type": "qwen3_5_moe_text",
    "hidden_size": 64,
    "intermediate_size": 96,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 64,
    "rms_norm_eps": 1e-6,
    "vocab_size": 128,
    "max_position_embeddings": 512,
    "linear_num_value_heads": 4,
    "linear_num_key_heads": 2,
    "linear_key_head_dim": 32,
    "linear_value_head_dim": 32,
    "linear_conv_kernel_dim": 4,
    "full_attention_interval": 2,
    "num_experts": 4,
    "num_experts_per_tok": 2,
    "shared_expert_intermediate_size": 96,
    "moe_intermediate_size": 96,
    "norm_topk_prob": True,
    "tie_word_embeddings": False,
    "attention_bias": False,
    "rope_parameters": {"type": "default", "rope_theta": 10000, "partial_rotary_factor": 0.25},
}


def build_reference(seed: int) -> reference.Model:
    mx.random.seed(seed)
    model = reference.Model(
        reference.ModelArgs(model_type="qwen3_5_moe", text_config=TINY_TEXT_CONFIG)
    )
    # Random init leaves the norm weights at one; perturb them so the parity is not trivial.
    params = {
        key: (value + 0.05 * mx.random.normal(value.shape))
        if key.endswith("norm.weight")
        else value
        for key, value in tree_flatten(model.parameters())
    }
    model.load_weights(list(params.items()))
    model.set_dtype(mx.float32)
    mx.eval(model.parameters())
    return model


def save_checkpoint(model: reference.Model, path: Path, quantization: dict | None) -> None:
    weights = dict(tree_flatten(model.parameters()))
    mx.save_safetensors(str(path / "model.safetensors"), weights)
    config = {
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "model_type": "qwen3_5_moe",
        "text_config": TINY_TEXT_CONFIG,
        "eos_token_id": [127],
    }
    if quantization is not None:
        config["quantization"] = quantization
    (path / "config.json").write_text(json.dumps(config))


def reference_greedy(
    model: reference.Model, prompt: list[int], steps: int
) -> tuple[mx.array, list[int]]:
    cache = model.make_cache()
    logits = model(mx.array(prompt)[None], cache=cache)
    prefill_logits = logits[0]
    token = int(mx.argmax(logits[0, -1]).item())
    tokens = [token]
    for _ in range(steps - 1):
        logits = model(mx.array([[token]]), cache=cache)
        token = int(mx.argmax(logits[0, -1]).item())
        tokens.append(token)
    return prefill_logits, tokens


__all__ = ["TINY_TEXT_CONFIG", "build_reference", "reference_greedy", "save_checkpoint"]
