"""Fuse rules on checkpoints that quantize the parts of a fused module differently.

mixed_fuses must name exactly the fused modules whose parts differ in scheme (a bf16 part
among quantized ones included), and apply_rules must then load those parts apart under
the fused module while every other module still fuses. Qwen3.5 checkpoint names.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import mlx.core as mx

from ftmetal.layers.quantization.config import QuantConfig
from ftmetal.models.loader import FuseRule, WeightRules, apply_rules, mixed_fuses

RULES = WeightRules(
    renames=(("language_model.model.", "model."),),
    segments=(("switch_mlp", "experts"),),
    fuses=(
        FuseRule("qkv_proj", ("q_proj", "k_proj", "v_proj")),
        FuseRule("gate_up_proj", ("gate_proj", "up_proj")),
        FuseRule("in_proj_ba", ("in_proj_b", "in_proj_a")),
    ),
    expert_segment="experts",
)
PREFIX = "language_model.model.layers"
# Checkpoint modules of a two-layer model; layer 0 linear attention, layer 1 full attention.
MODULES = (
    f"{PREFIX}.0.linear_attn.in_proj_b",
    f"{PREFIX}.0.linear_attn.in_proj_a",
    f"{PREFIX}.1.self_attn.q_proj",
    f"{PREFIX}.1.self_attn.k_proj",
    f"{PREFIX}.1.self_attn.v_proj",
    *(
        f"{PREFIX}.{layer}.mlp.switch_mlp.{part}"
        for layer in (0, 1)
        for part in ("gate_proj", "up_proj")
    ),
)
EIGHT_BIT = {"bits": 8, "group_size": 32}


@dataclass(frozen=True)
class Case:
    name: str
    # Per checkpoint module: an override, or None for a module left bf16 (no scales).
    overrides: dict[str, dict[str, int] | None]
    unfused: frozenset[str]


CASES = [
    Case("uniform", {}, frozenset()),
    Case(
        "expert-up-8-bit",
        {f"{PREFIX}.1.mlp.switch_mlp.up_proj": EIGHT_BIT},
        frozenset({"model.layers.1.mlp.experts.gate_up_proj"}),
    ),
    Case(
        "k-8-bit",
        {f"{PREFIX}.1.self_attn.k_proj": EIGHT_BIT},
        frozenset({"model.layers.1.self_attn.qkv_proj"}),
    ),
    Case(
        "a-bf16-b-quantized",
        {f"{PREFIX}.0.linear_attn.in_proj_a": None},
        frozenset({"model.layers.0.linear_attn.in_proj_ba"}),
    ),
    Case(
        "ba-both-bf16",
        {f"{PREFIX}.0.linear_attn.in_proj_a": None, f"{PREFIX}.0.linear_attn.in_proj_b": None},
        frozenset(),
    ),
]


def check(case: Case) -> None:
    quantized = [module for module in MODULES if case.overrides.get(module, {}) is not None]
    quantization: dict[str, object] = {"bits": 4, "group_size": 32}
    quantization.update({m: o for m, o in case.overrides.items() if o is not None})
    quant = QuantConfig.from_checkpoint(quantization, [f"{m}.scales" for m in quantized])
    rules = replace(RULES, unfused=mixed_fuses(RULES, quant))
    assert rules.unfused == case.unfused, case.name

    # Each part's weight: 4 output rows, its bits' packing of 64 inputs (bf16: 64 wide).
    state: dict[str, mx.array] = {}
    for module in MODULES:
        override = case.overrides.get(module, {})
        width = 64 if override is None else 64 * override.get("bits", 4) // 32
        state[f"{module}.weight"] = mx.zeros((4, width), dtype=mx.uint32)
    loaded = apply_rules(state, rules)
    for module in MODULES:
        key = rules.rename(f"{module}.weight")
        assert key is not None
        parent, _, leaf = key.removesuffix(".weight").rpartition(".")
        rule, _ = rules.part_index[leaf]
        fused = f"{parent}.{rule.fused}"
        if fused in case.unfused:
            assert loaded[f"{fused}.{leaf}.weight"].shape == state[f"{module}.weight"].shape
        else:
            assert loaded[f"{fused}.weight"].shape[0] == 4 * len(rule.parts), (case.name, fused)


def test_mixed_fuses_load_apart() -> None:
    for case in CASES:
        check(case)
