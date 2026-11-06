"""How checkpoints' quantization formats read as schemes and serve.

An MLX quantization section names affine or a float block mode per entry; an entry that names
no mode is affine (mlx-lm's reading), and a float mode takes only its own bits and group size.
A ModelOpt or compressed-tensors export of the tiny Qwen3.5 MoE model (tiny_model.save_export)
must serve like the mlx-lm reference on its dequantized weights.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from ftmetal.core import SamplingParams
from ftmetal.engine.config import EngineConfig
from ftmetal.layers.embedding import Embedding
from ftmetal.layers.quantization.config import AffineScheme, FpBlockScheme, QuantConfig, QuantScheme
from ftmetal.models.quant_export import rewrite_export
from ftmetal.models.register import read_checkpoint
from ftmetal.scheduler.scheduler import Scheduler
from tiny_model import (
    EXPORT_CONFIGS,
    EXPORT_FORMATS,
    FP8_WEIGHTS,
    build_reference,
    reference_greedy,
    save_export,
)

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
    (
        {
            "quant_method": "compressed-tensors",
            "config_groups": {"group_0": {"weights": FP8_WEIGHTS}},
            "transform_config": {
                "config_groups": {
                    "u": {
                        "type": "hadamard",
                        "apply": [{"targets": ["Linear"], "location": "input"}],
                    }
                }
            },
        },
        "compressed-tensors transform_config",
    ),
    (
        {
            "quant_method": "compressed-tensors",
            "config_groups": {"group_0": {"weights": FP8_WEIGHTS}},
            "sparsity_config": {"format": "sparse-24-bitmask", "sparsity_structure": "2:4"},
        },
        "compressed-tensors sparsity_config",
    ),
    ({"quant_method": "gptq"}, "quantization method 'gptq' is not served"),
]


@pytest.mark.parametrize(("quantization_config", "refusal"), UNSERVED_EXPORTS)
def test_unserved_exports_are_refused(quantization_config: dict[str, object], refusal: str) -> None:
    with pytest.raises(ValueError, match=re.escape(refusal)):
        rewrite_export({}, quantization_config, {})


def test_export_of_a_family_without_mlx_layout_is_refused(tmp_path: Path) -> None:
    mx.save_safetensors(str(tmp_path / "model.safetensors"), {"lm_head.weight": mx.zeros((4, 4))})
    config = {
        "architectures": ["Qwen4ExpForCausalLM"],
        "quantization_config": EXPORT_CONFIGS["modelopt"],
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(
        ValueError, match="Qwen4ExpForCausalLM serves MLX checkpoints only, not a 'modelopt' export"
    ):
        read_checkpoint("Qwen4ExpForCausalLM", str(tmp_path))


def test_exported_embedding_table_is_refused() -> None:
    with pytest.raises(ValueError, match="exported embedding table"):
        Embedding(128, 64, scheme=FpBlockScheme("nvfp4", has_output_scale=True))
