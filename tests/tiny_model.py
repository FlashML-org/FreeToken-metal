"""A tiny random Qwen3.5 MoE model shared by the parity and scheduler tests.

The mlx-lm model is the reference; save_checkpoint writes it in the MLX checkpoint
layout, save_hf_checkpoint in the Hugging Face layout (hf_layout_weights), save_export as a
ModelOpt or compressed-tensors export, its weights quantized by quantize_export (which takes
any family's MLX layout weights) the way each stores them (NVFP4: e2m1 codes packed low
nibble first, e4m3 scales per 16 and a per-tensor global scale; FP8: e4m3 bytes and a
per-tensor scale) from the format definitions rather than MLX's kernels, with the activation
and KV-cache scales the exports carry; reference_greedy is the contiguous-cache greedy decode.
TINY_FULL_ATTENTION_CONFIG drops the linear layers, so the engine has no GDN state pool.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import mlx.core as mx
import numpy as np
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


E2M1_VALUES = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
E2M1_MAX = 6.0
E4M3_MAX = 448.0
NVFP4_GROUP_SIZE = 16
LINEAR_ATTENTION = r"\.linear_attn\.(in_proj_qkv|in_proj_z|out_proj)$"
FULL_ATTENTION = r"\.self_attn\.[qkvo]_proj$"
EXPERTS = r"\.mlp\.(switch_mlp|shared_expert)\.(gate|up|down)_proj$"
# Per dialect, the modules (MLX layout names) stored FP8 and NVFP4 as the real exports store
# them: ModelOpt's mixed precision, and compressed-tensors with an FP8 group beside NVFP4.
EXPORT_FORMATS = {
    "modelopt": (f"{LINEAR_ATTENTION}|{FULL_ATTENTION}", f"{EXPERTS}|lm_head$"),
    "compressed-tensors": (LINEAR_ATTENTION, f"{EXPERTS}|{FULL_ATTENTION}"),
}
NVFP4_WEIGHTS = {"num_bits": 4, "type": "float", "strategy": "tensor_group", "group_size": 16}
FP8_WEIGHTS = {"num_bits": 8, "type": "float", "strategy": "tensor"}
EXPORT_CONFIGS: dict[str, dict[str, object]] = {
    "modelopt": {
        "quant_method": "modelopt",
        "quant_algo": "MIXED_PRECISION",
        "quantized_layers": {"lm_head": {"quant_algo": "W4A16_NVFP4", "group_size": 16}},
    },
    "compressed-tensors": {
        "quant_method": "compressed-tensors",
        "config_groups": {
            "group_0": {"weights": NVFP4_WEIGHTS},
            "group_1": {"weights": FP8_WEIGHTS},
        },
        # Empty, as in a real export: no transforms, no sparsity.
        "sparsity_config": {},
        "transform_config": {},
    },
}


def e4m3(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Round to e4m3: (the bytes, their values)."""
    encoded = mx.to_fp8(mx.array(values, dtype=mx.float32))
    return np.array(encoded), np.array(mx.from_fp8(encoded, dtype=mx.float32))


def nvfp4(weight: np.ndarray, headroom: float) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """[out, in] -> (packed uint8 [out, in / 2], e4m3 scale bytes [out, in / 16], the
    dequantization-side global scale, the dequantized weight); the global scale is headroom
    times the least that reaches the weight's amax."""
    out_size, in_size = weight.shape
    global_scale = float(np.abs(weight).max()) * headroom / (E2M1_MAX * E4M3_MAX)
    blocks = weight.reshape(out_size, in_size // NVFP4_GROUP_SIZE, NVFP4_GROUP_SIZE)
    scale_bytes, scales = e4m3(np.abs(blocks).max(axis=-1) / E2M1_MAX / global_scale)
    step = np.maximum(scales * global_scale, np.finfo(np.float32).tiny)[..., None]
    magnitude = np.argmin(np.abs(np.abs(blocks / step)[..., None] - E2M1_VALUES), axis=-1)
    codes = (magnitude | np.where(blocks < 0, 8, 0)).astype(np.uint8).reshape(out_size, in_size)
    packed = codes[:, 0::2] | (codes[:, 1::2] << 4)
    values = np.where(codes >= 8, -1.0, 1.0) * E2M1_VALUES[codes & 7]
    dequantized = values.reshape(blocks.shape) * step
    return packed, scale_bytes, global_scale, dequantized.reshape(out_size, in_size)


def fp8(weight: np.ndarray, headroom: float) -> tuple[np.ndarray, float, np.ndarray]:
    """[out, in] -> (e4m3 bytes, the per-tensor scale, the dequantized weight); the scale is
    headroom times the least that reaches the weight's amax."""
    scale = float(np.abs(weight).max()) * headroom / E4M3_MAX
    encoded, values = e4m3(weight / scale)
    return encoded, scale, values * scale


def quantize_export(
    weights: dict[str, mx.array],
    dialect: str,
    formats: tuple[str, str],
    dtype: mx.Dtype,
    seed: int,
) -> tuple[dict[str, mx.array], dict[str, mx.array]]:
    """MLX layout weights as a ModelOpt or compressed-tensors export stores them, under the same
    module names: the modules formats (FP8 and NVFP4 module patterns) names quantized, with the
    activation and KV-cache scales the dialect stores, the other tensors in dtype; and the
    dequantized weights by key."""
    fp8_modules, nvfp4_modules = formats
    is_modelopt = dialect == "modelopt"
    dequantized_weights: dict[str, mx.array] = {}
    stored: dict[str, mx.array] = {}
    # The tiny model's tensors of a shape share one amax: a headroom drawn per tensor gives each
    # per-tensor scale its own value, so one read from the wrong expert or fused part shows.
    rng = np.random.default_rng(seed)
    for key, value in weights.items():
        module = key.removesuffix(".weight")
        # A bank is quantized expert by expert; its per-tensor scales stack per expert.
        tensors = np.array(value, dtype=np.float32).reshape(-1, *value.shape[-2:])
        # One per linear: ModelOpt's a scalar, compressed-tensors' of shape [1].
        scale_shape = value.shape[:-2] if is_modelopt else (*value.shape[:-2], 1)
        ones = mx.ones(scale_shape, dtype=mx.float32)
        headrooms = rng.uniform(1.0, 4.0, size=len(tensors))
        if re.search(fp8_modules, module):
            encoded, scales, dequantized = zip(*map(fp8, tensors, headrooms), strict=True)
            stored[f"{module}.weight"] = mx.array(np.stack(encoded).reshape(value.shape))
            stored[f"{module}.weight_scale"] = mx.array(
                np.array(scales, np.float32).reshape(scale_shape)
            )
            stored[f"{module}.input_scale"] = ones
        elif re.search(nvfp4_modules, module):
            packed, scale_bytes, global_scales, dequantized = zip(
                *map(nvfp4, tensors, headrooms), strict=True
            )
            weight_name = "weight" if is_modelopt else "weight_packed"
            packed_shape = (*value.shape[:-1], value.shape[-1] // 2)
            stored[f"{module}.{weight_name}"] = mx.array(np.stack(packed).reshape(packed_shape))
            stored[f"{module}.weight_scale"] = mx.array(
                np.stack(scale_bytes).reshape(*value.shape[:-1], -1)
            )
            global_array = np.array(global_scales, np.float32).reshape(scale_shape)
            if is_modelopt:
                stored[f"{module}.weight_scale_2"] = mx.array(global_array)
                stored[f"{module}.input_scale"] = ones
            else:
                stored[f"{module}.weight_global_scale"] = mx.array(1.0 / global_array)
                stored[f"{module}.input_global_scale"] = ones
        else:
            stored[key] = value.astype(dtype)
            dequantized_weights[key] = value
            continue
        dequantized_weights[key] = mx.array(np.stack(dequantized).reshape(value.shape))
    # The attention layers' FP8 KV-cache scales.
    for key in list(stored):
        if key.endswith("self_attn.k_proj.weight"):
            attention = key.removesuffix(".k_proj.weight")
            kv_scale = f"{attention}.k_proj.k_scale" if is_modelopt else f"{attention}.k_scale"
            stored[kv_scale] = mx.ones(() if is_modelopt else (1,), dtype=mx.float32)
    return stored, dequantized_weights


def save_export(
    dialect: str,
    seed: int,
    path: Path,
    dtype: mx.Dtype,
    *,
    text_config: dict[str, object] | None = None,
    formats: tuple[str, str] | None = None,
    is_hf_layout: bool = True,
) -> dict[str, mx.array]:
    """Write the tiny model (of text_config) as a ModelOpt or compressed-tensors export
    (quantize_export of its weights, formats the dialect's EXPORT_FORMATS by default), in the
    Hugging Face layout or, without is_hf_layout, under the MLX layout's names; returns the
    dequantized weights by reference key."""
    model = build_reference(seed, text_config)
    stored, dequantized_weights = quantize_export(
        dict(tree_flatten(model.parameters())),
        dialect,
        EXPORT_FORMATS[dialect] if formats is None else formats,
        dtype,
        seed,
    )
    weights = hf_layout_weights(stored, "model.language_model.") if is_hf_layout else stored
    architecture = "Qwen3_5MoeForConditionalGeneration"
    save_hf_checkpoint(weights, model, path, architecture, EXPORT_CONFIGS[dialect])
    return dequantized_weights


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
    "EXPERTS",
    "EXPORT_CONFIGS",
    "EXPORT_FORMATS",
    "FP8_WEIGHTS",
    "FULL_ATTENTION",
    "LINEAR_ATTENTION",
    "TINY_FULL_ATTENTION_CONFIG",
    "TINY_TEXT_CONFIG",
    "build_reference",
    "hf_layout_weights",
    "quantize_export",
    "reference_greedy",
    "save_checkpoint",
    "save_export",
    "save_hf_checkpoint",
    "uncached_greedy",
]
