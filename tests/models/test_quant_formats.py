"""How checkpoints' quantization formats read as schemes and serve.

An MLX quantization section names affine or a float block mode per entry; an entry that names
no mode is affine (mlx-lm's reading), and a float mode takes only its own bits and group size.
A ModelOpt or compressed-tensors export of the tiny Qwen3.5 MoE model is quantized here the way
each stores it (NVFP4: e2m1 codes packed low nibble first, e4m3 scales per 16 and a per-tensor
global scale; FP8: e4m3 bytes and a per-tensor scale), from the format definitions rather than
MLX's kernels, in the Hugging Face layout with the activation and KV-cache scales the exports
carry; the engine serving it must match the mlx-lm reference on the dequantized weights.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from ftmetal.core import SamplingParams
from ftmetal.engine.config import EngineConfig
from ftmetal.layers.embedding import Embedding
from ftmetal.layers.quantization.config import AffineScheme, FpBlockScheme, QuantConfig, QuantScheme
from ftmetal.models.quant_export import rewrite_export
from ftmetal.models.register import read_checkpoint
from ftmetal.scheduler.scheduler import Scheduler
from tiny_model import build_reference, hf_layout_weights, reference_greedy, save_hf_checkpoint

# (the quantization section's default entry, the scheme it reads as or the refusal)
SCHEME_CASES: list[tuple[dict[str, object], QuantScheme | str]] = [
    ({"group_size": 64, "bits": 4}, AffineScheme(4, 64)),
    ({"group_size": 32, "bits": 3, "mode": "affine"}, AffineScheme(3, 32)),
    ({"group_size": 32, "bits": 4, "mode": "mxfp4"}, FpBlockScheme("mxfp4")),
    ({"group_size": 16, "bits": 4, "mode": "nvfp4"}, FpBlockScheme("nvfp4")),
    ({"group_size": 32, "bits": 8, "mode": "mxfp8"}, FpBlockScheme("mxfp8")),
    (
        {"group_size": 32, "bits": 4, "mode": "nvfp4"},
        "nvfp4 stores 4-bit elements in groups of 16, not 4-bit in groups of 32",
    ),
    (
        {"group_size": 32, "bits": 4, "mode": "mxfp8"},
        "mxfp8 stores 8-bit elements in groups of 32, not 4-bit in groups of 32",
    ),
    ({"group_size": 32, "bits": 4, "mode": "fp4"}, "unsupported quantization mode 'fp4'"),
]


@pytest.mark.parametrize(("entry", "expected"), SCHEME_CASES)
def test_default_entry_scheme(entry: dict[str, object], expected: QuantScheme | str) -> None:
    if isinstance(expected, str):
        with pytest.raises(ValueError, match=re.escape(expected) + "$"):
            QuantConfig.from_checkpoint(entry, ["proj.scales"])
    else:
        assert QuantConfig.from_checkpoint(entry, ["proj.scales"]).scheme_for("proj") == expected


def test_module_entries_without_a_mode_are_affine() -> None:
    quantization: dict[str, object] = {
        "group_size": 16,
        "bits": 4,
        "mode": "nvfp4",
        "mlp.gate": {"group_size": 64, "bits": 8},
        "shared_expert_gate": {"group_size": 32, "bits": 8, "mode": "mxfp8"},
    }
    names = ["proj.scales", "mlp.gate.scales", "shared_expert_gate.scales", "norm.weight"]
    quant = QuantConfig.from_checkpoint(quantization, names)
    assert quant.scheme_for("proj") == FpBlockScheme("nvfp4")
    assert quant.scheme_for("mlp.gate") == AffineScheme(8, 64)
    assert quant.scheme_for("shared_expert_gate") == FpBlockScheme("mxfp8")
    assert quant.scheme_for("norm") is None


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


def save_export(dialect: str, seed: int, path: Path, dtype: mx.Dtype) -> dict[str, mx.array]:
    """Write the tiny model as a ModelOpt or compressed-tensors export, its unquantized
    tensors in dtype; returns the dequantized weights by reference key."""
    model = build_reference(seed)
    fp8_modules, nvfp4_modules = EXPORT_FORMATS[dialect]
    is_modelopt = dialect == "modelopt"
    dequantized_weights: dict[str, mx.array] = {}
    stored: dict[str, mx.array] = {}
    # The tiny model's tensors of a shape share one amax: a headroom drawn per tensor gives each
    # per-tensor scale its own value, so one read from the wrong expert or fused part shows.
    rng = np.random.default_rng(seed)
    for key, value in tree_flatten(model.parameters()):
        assert isinstance(value, mx.array)
        module = key.removesuffix(".weight")
        # A bank is quantized expert by expert; its per-tensor scales stack per expert.
        tensors = np.array(value, dtype=np.float32).reshape(-1, *value.shape[-2:])
        ones = mx.ones((len(tensors),), dtype=mx.float32)
        headrooms = rng.uniform(1.0, 4.0, size=len(tensors))
        if re.search(fp8_modules, module):
            encoded, scales, dequantized = zip(*map(fp8, tensors, headrooms), strict=True)
            stored[f"{module}.weight"] = mx.array(np.stack(encoded).reshape(value.shape))
            stored[f"{module}.weight_scale"] = mx.array(np.array(scales, np.float32))
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
            global_array = np.array(global_scales, np.float32)
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
    # The full-attention layers' FP8 KV-cache scales.
    for key in list(stored):
        if key.endswith("self_attn.k_proj.weight"):
            attention = key.removesuffix(".k_proj.weight")
            kv_scale = f"{attention}.k_proj.k_scale" if is_modelopt else f"{attention}.k_scale"
            stored[kv_scale] = mx.ones((1,), dtype=mx.float32)
    # ModelOpt stores one scalar per linear.
    for key, tensor in stored.items():
        if is_modelopt and tensor.ndim == 1 and tensor.size == 1 and "switch_mlp" not in key:
            stored[key] = tensor.reshape(())
    weights = hf_layout_weights(stored, "model.language_model.")
    architecture = "Qwen3_5MoeForConditionalGeneration"
    save_hf_checkpoint(weights, model, path, architecture, EXPORT_CONFIGS[dialect])
    return dequantized_weights


def serve(path: Path, dtype: mx.Dtype, attention_backend: str) -> Scheduler:
    config = EngineConfig(
        model_path=str(path),
        dtype=dtype,
        max_running_req=2,
        page_size=4,
        attention_backend=attention_backend,
        num_page_override=64,
    )
    return Scheduler(config, eos_token_ids={127})


def prefill_logits(scheduler: Scheduler, prompt: list[int]) -> mx.array:
    """The logits of a one-chunk prompt's last token; the request then runs to its end."""
    scheduler.add_request(0, np.array(prompt, dtype=np.int32), SamplingParams(max_tokens=1))
    forward_input = scheduler.schedule_next_batch()
    assert forward_input is not None
    logits = scheduler.engine.model.forward(forward_input.batch)[0]
    # Before the pools' in-place writes of the steps that follow.
    mx.eval(logits)
    scheduler.process_last_data(scheduler.forward(forward_input))
    scheduler.run_until_idle()
    return logits


@pytest.mark.parametrize("attention_backend", ["mlx", "metal"])
@pytest.mark.parametrize("dialect", list(EXPORT_FORMATS))
def test_export_matches_its_dequantized_weights(
    tmp_path: Path, dialect: str, attention_backend: str
) -> None:
    dequantized = save_export(dialect, seed=7, path=tmp_path, dtype=mx.float32)
    model = build_reference(seed=7)
    model.load_weights(list(dequantized.items()))
    prompt = [int(t) for t in np.random.default_rng(7).integers(1, 120, size=37)]
    steps = 12
    expected_logits, expected = reference_greedy(model, prompt, steps)

    scheduler = serve(tmp_path, mx.float32, attention_backend)
    assert mx.allclose(prefill_logits(scheduler, prompt), expected_logits[-1], atol=1e-4, rtol=1e-4)
    params = SamplingParams(temperature=0.0, max_tokens=steps, ignore_eos=True)
    assert scheduler.generate(np.array(prompt, dtype=np.int32), params) == expected


@pytest.mark.parametrize("dialect", list(EXPORT_FORMATS))
def test_export_serves_in_bf16(tmp_path: Path, dialect: str) -> None:
    """The output scales are float32; the forward around them stays in bf16."""
    dequantized = save_export(dialect, seed=8, path=tmp_path, dtype=mx.bfloat16)
    model = build_reference(seed=8)
    model.load_weights(list(dequantized.items()))
    prompt = [int(t) for t in np.random.default_rng(8).integers(1, 120, size=37)]
    expected_logits, _ = reference_greedy(model, prompt, 1)

    logits = prefill_logits(serve(tmp_path, mx.bfloat16, "metal"), prompt)
    assert logits.dtype == mx.bfloat16
    assert float(mx.max(mx.abs(logits.astype(mx.float32) - expected_logits[-1]))) < 0.25


# (an export's quantization_config this does not serve, the refusal)
UNSERVED_EXPORTS: list[tuple[dict[str, object], str]] = [
    ({"quant_method": "modelopt", "quant_algo": "FP8_PB_WO"}, "quant_algo ['FP8_PB_WO']"),
    (
        {
            "quant_method": "modelopt",
            "quant_algo": "MIXED_PRECISION",
            "quantized_layers": {"lm_head": {"quant_algo": "NVFP4", "group_size": 32}},
        },
        "NVFP4 is served in groups of 16 only",
    ),
    (
        {
            "quant_method": "compressed-tensors",
            "config_groups": {
                "group_0": {"weights": {"num_bits": 4, "type": "int", "strategy": "group"}}
            },
        },
        "stores (4, 'int', 'group', None)",
    ),
    (
        {
            "quant_method": "compressed-tensors",
            "config_groups": {"group_0": {"weights": {**FP8_WEIGHTS, "strategy": "channel"}}},
        },
        "stores (8, 'float', 'channel', None)",
    ),
    ({"quant_method": "gptq"}, "quantization method 'gptq' is not served"),
]


@pytest.mark.parametrize(("quantization_config", "refusal"), UNSERVED_EXPORTS)
def test_unserved_exports_are_refused(quantization_config: dict[str, object], refusal: str) -> None:
    with pytest.raises(ValueError, match=re.escape(refusal)):
        rewrite_export({}, quantization_config)


def test_export_of_a_family_without_mlx_layout_is_refused(tmp_path: Path) -> None:
    mx.save_safetensors(str(tmp_path / "model.safetensors"), {"lm_head.weight": mx.zeros((4, 4))})
    config = {
        "architectures": ["Gemma4ForCausalLM"],
        "quantization_config": EXPORT_CONFIGS["modelopt"],
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(
        ValueError, match="Gemma4ForCausalLM serves MLX checkpoints only, not a 'modelopt' export"
    ):
        read_checkpoint("Gemma4ForCausalLM", str(tmp_path))


def test_exported_embedding_table_is_refused() -> None:
    with pytest.raises(ValueError, match="exported embedding table"):
        Embedding(128, 64, scheme=FpBlockScheme("nvfp4", has_output_scale=True))
