"""A tiny random Qwen3.5 MoE model shared by the parity and scheduler tests.

The mlx-lm model is the reference; save_checkpoint writes it in the MLX checkpoint
layout, save_hf_checkpoint in the Hugging Face layout (hf_layout_weights), and
reference_greedy is the contiguous-cache greedy decode.
TINY_FULL_ATTENTION_CONFIG drops the linear layers, so the engine has no GDN state pool.
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

TINY_FULL_ATTENTION_CONFIG = {**TINY_TEXT_CONFIG, "full_attention_interval": 1}


def build_reference(seed: int, text_config: dict[str, object] | None = None) -> reference.Model:
    mx.random.seed(seed)
    model = reference.Model(
        reference.ModelArgs(
            model_type="qwen3_5_moe",
            text_config=TINY_TEXT_CONFIG if text_config is None else text_config,
        )
    )
    # Random init leaves the norm weights at one; perturb them so the parity is not trivial.
    params: dict[str, mx.array] = {}
    for key, value in tree_flatten(model.parameters()):
        assert isinstance(value, mx.array)
        if key.endswith("norm.weight"):
            value = value + 0.05 * mx.random.normal(value.shape)
        params[key] = value
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
        "text_config": model.args.text_config,
        "eos_token_id": [127],
    }
    if quantization is not None:
        # mlx-lm writes the section twice, the second time as quantization_config.
        config["quantization"] = quantization
        config["quantization_config"] = quantization
    (path / "config.json").write_text(json.dumps(config))


# The norms the MLX layout stores shifted by one (mlx-lm's qwen3_5 sanitize).
SHIFTED_NORMS = (
    ".input_layernorm.weight",
    ".post_attention_layernorm.weight",
    "model.norm.weight",
    ".q_norm.weight",
    ".k_norm.weight",
)


def hf_layout_weights(
    weights: dict[str, mx.array], text_prefix: str, *, has_stacked_experts: bool = False
) -> dict[str, mx.array]:
    """MLX layout weights (the reference's, or quantized tensors under its module names) as a
    Hugging Face checkpoint stores them: text_prefix names, one tensor per expert (or, as the
    bf16 release stores them, each layer's gate and up stacked in one bank and down in
    another), a [conv_dim, 1, kernel] conv, unshifted norms, and an MTP head the engine does
    not build, its experts stacked as every export stores them."""
    out: dict[str, mx.array] = {
        "mtp.fc.weight": mx.zeros((4, 4)),
        "mtp.layers.0.mlp.experts.gate_up_proj": mx.zeros((4, 8, 4)),
        "mtp.layers.0.mlp.experts.down_proj": mx.zeros((4, 4, 4)),
    }
    for mlx_key, value in weights.items():
        if mlx_key.startswith("language_model.model."):
            key = text_prefix + mlx_key.removeprefix("language_model.model.")
        else:
            key = mlx_key.removeprefix("language_model.")
        parent, _, leaf = key.partition(".switch_mlp.")
        if leaf and has_stacked_experts:
            if leaf == "gate_proj.weight":
                up = weights[mlx_key.replace(".gate_proj.", ".up_proj.")]
                out[f"{parent}.experts.gate_up_proj"] = mx.concatenate([value, up], axis=-2)
            elif leaf == "down_proj.weight":
                out[f"{parent}.experts.down_proj"] = value
            else:
                assert leaf == "up_proj.weight", f"{mlx_key}: stacked experts are bf16"
        elif leaf:
            for expert_id in range(value.shape[0]):
                out[f"{parent}.experts.{expert_id}.{leaf}"] = value[expert_id]
        elif key.endswith(".conv1d.weight"):
            out[key] = value.moveaxis(1, 2)
        elif key.endswith(SHIFTED_NORMS):
            out[key] = value - 1
        else:
            out[key] = value
    return out


def save_hf_checkpoint(
    weights: dict[str, mx.array],
    model: reference.Model,
    path: Path,
    architecture: str,
    quantization_config: dict[str, object] | None = None,
) -> None:
    mx.save_safetensors(str(path / "model.safetensors"), weights)
    config: dict[str, object] = {
        "architectures": [architecture],
        "model_type": "qwen3_5_moe",
        "text_config": model.args.text_config,
        "eos_token_id": [127],
    }
    if quantization_config is not None:
        config["quantization_config"] = quantization_config
    (path / "config.json").write_text(json.dumps(config))


def reference_greedy(
    model: reference.Model, prompt: list[int], steps: int
) -> tuple[mx.array, list[int]]:
    cache = model.make_cache()
    logits = model(mx.array(prompt)[None], cache=cache)
    prefill_logits = logits[0]
    token = int(mx.argmax(logits[0, -1]))
    tokens = [token]
    for _ in range(steps - 1):
        logits = model(mx.array([[token]]), cache=cache)
        token = int(mx.argmax(logits[0, -1]))
        tokens.append(token)
    return prefill_logits, tokens


def uncached_greedy(model: reference.Model, prompt: list[int], steps: int) -> list[int]:
    """Greedy decode rerunning the whole sequence per token: mlx-lm 0.31's contiguous cache
    path builds its linear-attention mask from layer 0's cache, so it fails without one."""
    sequence = list(prompt)
    for _ in range(steps):
        sequence.append(int(mx.argmax(model(mx.array(sequence)[None])[0, -1])))
    return sequence[len(prompt) :]


__all__ = [
    "TINY_FULL_ATTENTION_CONFIG",
    "TINY_TEXT_CONFIG",
    "build_reference",
    "hf_layout_weights",
    "reference_greedy",
    "save_checkpoint",
    "save_hf_checkpoint",
    "uncached_greedy",
]
