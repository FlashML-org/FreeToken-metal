"""ModelOpt and compressed-tensors exports rewritten into MLX quantized tensors, and the global
scales an MLX nvfp4 checkpoint keeps as ModelOpt does rewritten into output scales.

Port of the weight-only part of FreeToken layers/quantization/configs/modelopt.py and
compressed_tensors.py. Divergences: the export's NVFP4 and per-tensor FP8 modules are rewritten
at load into MLX's nvfp4 (as stored) and mxfp8 (under unit e8m0 scales) tensors, each per-tensor
scale becoming an output_scale, instead of read by FP4 / FP8 kernels; each module's format is
read from the tensors it stores in the MLX layout, not from the config's module lists, which
name Hugging Face modules, and the config is only checked for the formats, transforms and
sparsity this serves; activation and KV-cache scales are dropped, the engine serving the weights
alone; the checkpoint bytes of each expert tensor carry over to the tensor rewritten from it.
"""

from __future__ import annotations

import mlx.core as mx

from ftmetal.layers.quantization.config import FpBlockScheme, QuantScheme
from ftmetal.models.loader import (
    UNUSED_SCALE_SUFFIXES,
    ExpertBytes,
    ExpertFill,
    config_dict,
    config_int,
    config_str,
    stacked_slices,
)

# The e8m0 byte of a scale of one.
E8M0_ONE = 127
NVFP4_SCHEME = FpBlockScheme("nvfp4", has_output_scale=True)
FP8_SCHEME = FpBlockScheme("mxfp8", has_output_scale=True)
MODELOPT_ALGOS = ("NVFP4", "W4A16_NVFP4", "FP8")
# The compressed-tensors weights served, (bits, type, strategy, group size): NVFP4 and
# per-tensor FP8.
COMPRESSED_TENSORS_WEIGHTS = (
    (4, "float", "tensor_group", NVFP4_SCHEME.group_size),
    (8, "float", "tensor", None),
)
# The compressed-tensors fields refused when set, with the weights each marks.
COMPRESSED_TENSORS_UNSERVED_FIELDS = {
    "transform_config": "rotated weights",
    "sparsity_config": "sparse weights",
}


def output_scale(module: str, scale: mx.array, weight: mx.array) -> mx.array:
    """A per-tensor scale as the module's output_scale in float32: a Linear's [out, 1], one per
    output row, so fused parts concatenate on -2; a bank's [experts, 1, 1], one per expert and
    part."""
    experts = weight.shape[:-2]
    if scale.size != (experts[0] if experts else 1):
        raise ValueError(f"{module}: {scale.shape} scales are not one per tensor")
    elif experts:
        return scale.astype(mx.float32).reshape(*experts, 1, 1)
    else:
        return mx.full((weight.shape[0], 1), scale.astype(mx.float32).reshape(()))


def rewrite_global_scales(
    state: dict[str, mx.array],
) -> tuple[dict[str, mx.array], frozenset[str]]:
    """An MLX checkpoint whose nvfp4 modules each store the dequantization-side global scale
    ModelOpt NVFP4 does (weight_scale_2: one per tensor, or per expert of a bank), with each such
    scale as its module's output_scale; and the modules rewritten, for QuantConfig to check that
    each is declared nvfp4."""
    out: dict[str, mx.array] = {}
    modules: set[str] = set()
    for key, tensor in state.items():
        if key.endswith(".weight_scale_2"):
            module = key.removesuffix(".weight_scale_2")
            out[f"{module}.output_scale"] = output_scale(module, tensor, state[f"{module}.weight"])
            modules.add(module)
        else:
            out[key] = tensor
    return out, frozenset(modules)


def rewrite_export(
    state: dict[str, mx.array],
    quantization_config: dict[str, object],
    expert_bytes: dict[str, ExpertBytes],
) -> tuple[dict[str, mx.array], dict[str, QuantScheme], dict[str, ExpertBytes]]:
    """An export's NVFP4 and per-tensor FP8 modules (an MLX layout state) as MLX nvfp4 / mxfp8
    weight, scales and output_scale tensors, the scheme of each module rewritten, and
    expert_bytes (the state's experts' checkpoint bytes) for the result; raises unless every
    format the config names is NVFP4 or per-tensor FP8 with no transform_config or
    sparsity_config."""
    method = config_str(quantization_config, "quant_method", "")
    if method == "modelopt":
        quant_algo = config_str(quantization_config, "quant_algo", "").upper()
        algos = {quant_algo} if quant_algo != "MIXED_PRECISION" else set()
        for name, layer in (config_dict(quantization_config, "quantized_layers") or {}).items():
            if not isinstance(layer, dict):
                raise ValueError(f"ModelOpt quantized_layers[{name!r}] is not an object")
            algos.add(config_str(layer, "quant_algo", "").upper())
            if config_int(layer, "group_size", NVFP4_SCHEME.group_size) != NVFP4_SCHEME.group_size:
                raise ValueError(f"ModelOpt {name}: NVFP4 is served in groups of 16 only")
        unserved = sorted(algos - set(MODELOPT_ALGOS))
        if unserved:
            raise ValueError(f"ModelOpt quant_algo {unserved} is not served, only {MODELOPT_ALGOS}")
    elif method == "compressed-tensors":
        for field, holds in COMPRESSED_TENSORS_UNSERVED_FIELDS.items():
            if quantization_config.get(field):
                raise ValueError(f"compressed-tensors {field} ({holds}) is not served")
        for name, group in (config_dict(quantization_config, "config_groups") or {}).items():
            if not isinstance(group, dict):
                raise ValueError(f"compressed-tensors config_groups[{name!r}] is not an object")
            weights = config_dict(group, "weights") or {}
            stored = (
                config_int(weights, "num_bits", 0),
                config_str(weights, "type", ""),
                config_str(weights, "strategy", ""),
                weights.get("group_size"),
            )
            if stored not in COMPRESSED_TENSORS_WEIGHTS:
                raise ValueError(
                    f"compressed-tensors group {name!r} stores {stored}; only NVFP4 and "
                    f"per-tensor FP8 {COMPRESSED_TENSORS_WEIGHTS} are served"
                )
    else:
        raise ValueError(f"quantization method {method!r} is not served")
    out: dict[str, mx.array] = {}
    out_expert_bytes: dict[str, ExpertBytes] = {}
    schemes: dict[str, QuantScheme] = {}
    consumed: set[str] = set()

    def carry(out_key: str, key: str) -> None:
        """A rewritten expert tensor's checkpoint bytes: its source's as recorded, or the source
        stored stacked under its own name."""
        if key in expert_bytes:
            out_expert_bytes[out_key] = expert_bytes[key]
        elif state[key].ndim == 3:
            out_expert_bytes[out_key] = stacked_slices(key, state[key])

    for key in state:
        if key.endswith(".weight_scale_2"):
            # ModelOpt NVFP4 stores the dequantization-side global scale.
            module = key.removesuffix(".weight_scale_2")
            weight_key = f"{module}.weight"
            global_scale = state[key]
        elif key.endswith(".weight_global_scale"):
            # compressed-tensors NVFP4 stores the quantization side.
            module = key.removesuffix(".weight_global_scale")
            weight_key = f"{module}.weight_packed"
            global_scale = 1.0 / state[key].astype(mx.float32)
        else:
            continue
        weight, scales = state[weight_key], state[f"{module}.weight_scale"]
        if scales.shape[-1] * NVFP4_SCHEME.group_size != weight.shape[-1] * 2:
            raise ValueError(f"{module}: NVFP4 scales are not one per 16 weights")
        out[f"{module}.weight"] = weight.view(mx.uint32)
        out[f"{module}.scales"] = scales
        out[f"{module}.output_scale"] = output_scale(module, global_scale, weight)
        carry(f"{module}.weight", weight_key)
        carry(f"{module}.scales", f"{module}.weight_scale")
        schemes[module] = NVFP4_SCHEME
        consumed.update((key, weight_key, f"{module}.weight_scale"))
    for key in state:
        if key.endswith(".weight_scale") and key not in consumed:
            # Per-tensor FP8: e4m3 bytes and the dequantization scale.
            module = key.removesuffix(".weight_scale")
            weight = state[f"{module}.weight"]
            if weight.dtype != mx.uint8:
                raise ValueError(f"{module}: an FP8 weight is stored as bytes, not {weight.dtype}")
            groups = weight.shape[-1] // FP8_SCHEME.group_size
            out[f"{module}.weight"] = weight.view(mx.uint32)
            out[f"{module}.scales"] = mx.full((*weight.shape[:-1], groups), E8M0_ONE, mx.uint8)
            out[f"{module}.output_scale"] = output_scale(module, state[key], weight)
            carry(f"{module}.weight", f"{module}.weight")
            if weight.ndim == 3:
                out_expert_bytes[f"{module}.scales"] = ExpertFill(value=E8M0_ONE)
            schemes[module] = FP8_SCHEME
            consumed.update((key, f"{module}.weight"))
    for key, tensor in state.items():
        if key not in consumed and not key.endswith(UNUSED_SCALE_SUFFIXES):
            out[key] = tensor
            if key in expert_bytes:
                out_expert_bytes[key] = expert_bytes[key]
    return out, schemes, out_expert_bytes


__all__ = ["rewrite_export", "rewrite_global_scales"]
