"""Memory budget on the Qwen3.6-35B-A3B geometry: report lines, page and GDN slot counts,
for the hybrid radix cache and naive; and Gemma4-26B-A4B's window pool, a fixed cost beside
the global pages for naive and for the SWA radix cache. MoE offload on Qwen3.6-35B-A3B's
checkpoint: what auto picks, the cache size each flag gives, and the experts line, on this
box's budget and on fractions of it standing for 24 GB and 32 GB Macs; refused cache sizes
and configs; the strategy resolution on a tiny checkpoint, auto keeping a family without
offload resident; Gemma4's refusal of offload. The
vision towers' reserves, and that the budget and the image processor import no model package.

The expected numbers were computed by the engine's sizing before pools were created per
attention type; the dispatch must reproduce them exactly from the same fixed budget.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Literal, NamedTuple

import mlx.core as mx
import mlx.nn as nn
import pytest

from ftmetal.attention.metadata import AttnType
from ftmetal.engine.cache_budget import (
    GIB,
    IN_FLIGHT_STEPS,
    OFFLOAD_IN_FLIGHT_STEPS,
    BudgetReport,
    activation_reserve_bytes,
    choose_moe_strategy,
    encoder_reserve_bytes,
    graph_reserve_bytes,
    kv_reserve_pages,
    memory_budget_bytes,
    plan_budget,
    resolve_moe_cache_size,
    solve_num_pages,
    victim_cache_bytes,
)
from ftmetal.engine.config import EngineConfig, MoeStrategyChoice
from ftmetal.engine.engine import resolve_moe_strategy
from ftmetal.kvcache import pool_costs
from ftmetal.kvcache.base import PoolCost
from ftmetal.models.config import MoeStrategy
from ftmetal.models.gemma4 import vision as gemma4_vision
from ftmetal.models.gemma4.config import parse_config as parse_gemma4_config
from ftmetal.models.gemma4.model import Gemma4ForCausalLM
from ftmetal.models.qwen3_5_moe.config import parse_config
from ftmetal.models.qwen3_vl import vision as qwen_vision
from ftmetal.models.register import (
    GEMMA4_VISION,
    QWEN_VISION,
    EncoderSpec,
    model_weight_rules,
    read_checkpoint,
)
from ftmetal.moe.offload_cache import STAGING_BYTES
from tiny_model import TINY_TEXT_CONFIG, build_reference, save_checkpoint

# The text_config fields of Qwen3.6-35B-A3B that its attention groups are parsed from.
QWEN3_6_TEXT_CONFIG: dict[str, object] = {
    "num_attention_heads": 16,
    "num_key_value_heads": 2,
    "head_dim": 256,
    "hidden_size": 2048,
    "num_hidden_layers": 40,
    "full_attention_interval": 4,
    "vocab_size": 248320,
    "rms_norm_eps": 1e-6,
    "max_position_embeddings": 262144,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 32,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4,
    "num_experts": 256,
    "rope_parameters": {"type": "default", "rope_theta": 10000000, "partial_rotary_factor": 0.25},
}
MIB = 1 << 20
WORKING_SET_BYTES = 40_200_896_512  # max_recommended_working_set_size, M4 Pro 48 GB
WEIGHTS_BYTES = 20 * GIB
# 30 GDN layers x (conv ring 4 x 8192 bf16 + state 32 x 128 x 128 fp32).
GDN_SLOT_BYTES = 64_880_640


class BudgetCase(NamedTuple):
    page_size: int
    max_running_req: int
    gdn_snapshots: int | None
    max_extend_tokens: int
    num_page_override: int | None
    num_pages: int
    num_slots: int
    lines: list[str]
    pages_line: str
    cache_type: Literal["naive", "radix"] = "radix"


CASES = [
    BudgetCase(
        page_size=1,
        max_running_req=4,
        gdn_snapshots=None,
        max_extend_tokens=2048,
        num_page_override=None,
        num_pages=507478,
        num_slots=25,
        lines=[
            "budget         36.44 GiB",
            "weights        20.00 GiB",
            "gdn slots       1.51 GiB",
            "activations     5.25 GiB",
            "kv pages        9.68 GiB",
        ],
        pages_line="KV cache: 507478 pages x 1 tokens, 9.68 GiB",
    ),
    BudgetCase(
        page_size=32,
        max_running_req=8,
        gdn_snapshots=16,
        max_extend_tokens=2048,
        num_page_override=None,
        num_pages=13482,
        num_slots=49,
        lines=[
            "budget         36.44 GiB",
            "weights        20.00 GiB",
            "gdn slots       2.96 GiB",
            "activations     5.25 GiB",
            "kv pages        8.23 GiB",
        ],
        pages_line="KV cache: 13482 pages x 32 tokens, 8.23 GiB",
    ),
    BudgetCase(
        page_size=16,
        max_running_req=2,
        gdn_snapshots=4,
        max_extend_tokens=1024,
        num_page_override=1000,
        num_pages=1000,
        num_slots=13,
        lines=[
            "budget         36.44 GiB",
            "weights        20.00 GiB",
            "gdn slots       0.79 GiB",
            "activations     3.06 GiB",
            "kv pages       12.59 GiB",
        ],
        pages_line="KV cache: 1000 pages x 16 tokens, 0.31 GiB",
    ),
    # Naive: a live slot per running request and the padding slot; the 20 hybrid radix slots
    # it drops go to the pages.
    BudgetCase(
        page_size=1,
        max_running_req=4,
        gdn_snapshots=None,
        max_extend_tokens=2048,
        num_page_override=None,
        num_pages=570838,
        num_slots=5,
        lines=[
            "budget         36.44 GiB",
            "weights        20.00 GiB",
            "gdn slots       0.30 GiB",
            "activations     5.25 GiB",
            "kv pages       10.89 GiB",
        ],
        pages_line="KV cache: 570838 pages x 1 tokens, 10.89 GiB",
        cache_type="naive",
    ),
]


@pytest.mark.parametrize("case", CASES, ids=["page1", "page32", "override", "naive"])
def test_qwen3_6_budget_reproduces_pages_and_slots(
    case: BudgetCase, caplog: pytest.LogCaptureFixture
) -> None:
    model_config = parse_config(QWEN3_6_TEXT_CONFIG)
    config = EngineConfig(
        model_path="unused",
        page_size=case.page_size,
        max_running_req=case.max_running_req,
        gdn_snapshots=case.gdn_snapshots,
        max_extend_tokens_override=case.max_extend_tokens,
        num_page_override=case.num_page_override,
        cache_type=case.cache_type,
    )
    costs = pool_costs(model_config.kv_cache_group_specs(), config)
    assert costs[AttnType.LINEAR] == PoolCost(0, case.num_slots * GDN_SLOT_BYTES)

    activations = activation_reserve_bytes(
        config.max_extend_tokens, model_config.num_qo_heads, IN_FLIGHT_STEPS
    )
    budget = memory_budget_bytes(WORKING_SET_BYTES, None)
    report = plan_budget(budget, WEIGHTS_BYTES, activations, 0, costs)
    assert report.lines() == case.lines
    caplog.set_level(logging.INFO)
    num_pages = solve_num_pages(report, config.page_size, config.num_page_override)
    assert num_pages == case.num_pages
    assert caplog.messages == [case.pages_line]


def test_graph_reserve_comes_off_the_paged_pools() -> None:
    model_config = parse_config(QWEN3_6_TEXT_CONFIG)
    config = EngineConfig(model_path="unused", max_running_req=8, max_extend_tokens_override=2048)
    costs = pool_costs(model_config.kv_cache_group_specs(), config)
    activations = activation_reserve_bytes(
        config.max_extend_tokens, model_config.num_qo_heads, IN_FLIGHT_STEPS
    )
    budget = memory_budget_bytes(WORKING_SET_BYTES, None)
    graphs = graph_reserve_bytes(8)
    without = plan_budget(budget, WEIGHTS_BYTES, activations, 0, costs)
    with_graphs = plan_budget(budget, WEIGHTS_BYTES, activations, graphs, costs)
    assert without.paged_bytes - with_graphs.paged_bytes == graphs
    assert "graphs" not in "".join(without.lines())
    lines = with_graphs.lines()
    assert lines.index("graphs          0.01 GiB") == lines.index("activations     5.25 GiB") + 1


# The vision_config fields the encoder reserve reads, the same in Qwen3.6-35B-A3B's and
# Gemma-4-26B-A4B's towers, and both processors' feature row: 1536 bf16 values (Qwen) or 768
# float32 ones (Gemma4).
VISION_CONFIG: dict[str, object] = {"hidden_size": 1152, "intermediate_size": 4304}
FEATURE_BYTES = 3072


# (tower, patches, the measured peak the reserve must cover). Qwen3.6's in bf16 above the
# weights and the float32 pixels, plus their bf16 cast it holds; Gemma4's evaluated layer by
# layer, above the weights, with 5% to spare.
ENCODER_PEAK_CASES: list[tuple[str, EncoderSpec, int, float]] = [
    ("qwen", QWEN_VISION, 16384, 482.6 * MIB + 16384 * FEATURE_BYTES),
    ("qwen", QWEN_VISION, 65536, 1802.4 * MIB + 65536 * FEATURE_BYTES),
    ("gemma4", GEMMA4_VISION, 2520, 1.05 * 319 * MIB),
    ("gemma4", GEMMA4_VISION, 10080, 1.05 * 715 * MIB),
]


@pytest.mark.parametrize(("label", "encoder", "patches", "peak"), ENCODER_PEAK_CASES)
def test_encoder_reserve_covers_the_measured_peak(
    label: str, encoder: EncoderSpec, patches: int, peak: float
) -> None:
    reserve = encoder_reserve_bytes(encoder, VISION_CONFIG, mx.bfloat16, patches, FEATURE_BYTES)
    assert reserve >= peak, label


def test_encoder_reserve_comes_off_the_paged_pools() -> None:
    """Qwen3.6's 16384 tokens of 2x2 patches: per patch a bf16 1536-wide feature and 13 bf16
    1152-wide activations, then one 4096-row chunk of two 4304-wide ones; Gemma4's 280 tokens
    of 3x3 patches: per patch a float32 768-wide feature and 24 activations, one 2560-row chunk
    of three, and 160 MiB."""
    reserve = encoder_reserve_bytes(
        QWEN_VISION, VISION_CONFIG, mx.bfloat16, 16384 * 4, FEATURE_BYTES
    )
    assert reserve == 65536 * (2 * 1536 + 2 * 13 * 1152) + 2 * 2 * 4096 * 4304 == 2_234_777_600
    gemma4 = encoder_reserve_bytes(GEMMA4_VISION, VISION_CONFIG, mx.bfloat16, 2520, FEATURE_BYTES)
    assert gemma4 == 2520 * (4 * 768 + 2 * 24 * 1152) + 2 * 3 * 2560 * 4304 + (160 << 20)
    model_config = parse_config(QWEN3_6_TEXT_CONFIG)
    config = EngineConfig(model_path="unused", max_running_req=8, max_extend_tokens_override=2048)
    costs = pool_costs(model_config.kv_cache_group_specs(), config)
    budget = memory_budget_bytes(WORKING_SET_BYTES, None)
    without = plan_budget(budget, WEIGHTS_BYTES, 0, 0, costs)
    with_encoder = plan_budget(budget, WEIGHTS_BYTES, 0, 0, costs, reserve)
    assert without.paged_bytes - with_encoder.paged_bytes == reserve
    assert "encoder         2.08 GiB" in with_encoder.lines()


@pytest.mark.parametrize(
    ("encoder", "tower"), [(QWEN_VISION, qwen_vision), (GEMMA4_VISION, gemma4_vision)]
)
def test_encoder_spec_mlp_rows_are_the_towers(encoder: EncoderSpec, tower: ModuleType) -> None:
    assert encoder.mlp_rows == tower.MLP_ROWS


# What the frontend and the budget must not import: the model packages and their kernels.
MODEL_MODULES = (
    "ftmetal.models.gemma4",
    "ftmetal.models.qwen3_5_moe",
    "ftmetal.kernel.attention.gdn.gdn_chunk",
    "ftmetal.kernel.attention.gdn.gdn_decode",
    "ftmetal.kernel.attention.gdn.gdn_prefill",
    "ftmetal.kernel.moe.moe_topk",
)


@pytest.mark.parametrize("module", ["ftmetal.mm.processor", "ftmetal.engine.cache_budget"])
def test_registry_users_load_no_model_package(module: str) -> None:
    probe = (
        f"import json, sys, {module}\n"
        f"print(json.dumps([name for name in {MODEL_MODULES!r} if name in sys.modules]))"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    assert json.loads(result.stdout) == []


# The text_config fields of Gemma4-26B-A4B that its attention groups are parsed from.
GEMMA4_26B_TEXT_CONFIG: dict[str, object] = {
    "num_attention_heads": 16,
    "num_key_value_heads": 8,
    "num_global_key_value_heads": 2,
    "attention_k_eq_v": True,
    "head_dim": 256,
    "global_head_dim": 512,
    "hidden_size": 2816,
    "intermediate_size": 2112,
    "num_hidden_layers": 30,
    "sliding_window": 1024,
    "layer_types": [
        "full_attention" if (i + 1) % 6 == 0 else "sliding_attention" for i in range(30)
    ],
    "vocab_size": 262144,
    "max_position_embeddings": 262144,
    "rope_parameters": {
        "full_attention": {
            "partial_rotary_factor": 0.25,
            "rope_theta": 1e6,
            "rope_type": "proportional",
        },
        "sliding_attention": {"rope_theta": 1e4, "rope_type": "default"},
    },
}


class WindowBudgetCase(NamedTuple):
    cache_type: Literal["naive", "radix"]
    window_cost: PoolCost
    lines: list[str]


# 5 global layers x 2 x 2 heads x 512 x bf16 per token; 25 window layers x 2 x 8 heads x 256
# x bf16 per window slot.
WINDOW_BUDGET_CASES = [
    # 5 x align32(1024 + 2048 + 1) slots, fixed.
    WindowBudgetCase(
        "naive",
        PoolCost(0, 15520 * 200 * 1024),
        [
            "budget         36.44 GiB",
            "weights        13.00 GiB",
            "swa window      2.96 GiB",
            "activations     0.00 GiB",
            "kv pages       20.48 GiB",
        ],
    ),
    # The floor, 4 x (1040 + 1024 + 3) + 2 x 2048 slots, in 13 windows of 1024, 8 windows more
    # and the sentinel, fixed.
    WindowBudgetCase(
        "radix",
        PoolCost(0, 21505 * 200 * 1024),
        [
            "budget         36.44 GiB",
            "weights        13.00 GiB",
            "swa window      4.10 GiB",
            "activations     0.00 GiB",
            "kv pages       19.34 GiB",
        ],
    ),
]


def test_gemma4_window_pool_beside_the_global_pages() -> None:
    model_config = parse_gemma4_config(GEMMA4_26B_TEXT_CONFIG)
    for case in WINDOW_BUDGET_CASES:
        config = EngineConfig(
            model_path="unused",
            max_running_req=4,
            max_extend_tokens_override=2048,
            cache_type=case.cache_type,
        )
        costs = pool_costs(model_config.kv_cache_group_specs(), config)
        assert costs == {AttnType.FULL: PoolCost(20 * 1024, 0), AttnType.SWA: case.window_cost}
        report = plan_budget(memory_budget_bytes(WORKING_SET_BYTES, None), 13 * GIB, 0, 0, costs)
        assert report.lines() == case.lines, case.cache_type


def test_gemma4_refuses_moe_offload() -> None:
    model_config = parse_gemma4_config(
        {**GEMMA4_26B_TEXT_CONFIG, "enable_moe_block": True, "num_experts": 128}
    )
    with pytest.raises(ValueError, match="not supported for Gemma4 yet"):
        Gemma4ForCausalLM(dataclasses.replace(model_config, moe_strategy="offload"))


# Qwen3.6-35B-A3B's checkpoint, from its safetensors headers: the weights besides the routed
# experts, the experts, and one affine 4-bit group-64 expert over the gate_up and down banks.
NON_EXPERT_BYTES = 1_378_867_456
EXPERT_BYTES = 18_119_393_280
EXPERT_SLOT_BYTES = 1_769_472
NUM_LAYERS = 40
NUM_EXPERTS = 256
TOP_K = 8
TOTAL_EXPERTS = NUM_LAYERS * NUM_EXPERTS


class OffloadBudgetCase(NamedTuple):
    memory_ratio: float | None
    moe_cache_size: int
    moe_cache_rate: float | None
    moe_prefill_overlap: bool
    # What auto picks with no cache size flag.
    auto_strategy: MoeStrategy
    cache_size: int
    lines: list[str]


OFFLOAD_BUDGET_CASES = [
    # A 24 GB Mac's budget: the experts do not fit resident; cache-auto gives the KV pools
    # 8192 tokens and the experts the rest.
    OffloadBudgetCase(
        0.43,
        0,
        None,
        True,
        "offload",
        6342,
        [
            "budget         16.10 GiB",
            "weights         1.28 GiB",
            "experts        10.51 GiB",
            "gdn slots       1.51 GiB",
            "activations     2.62 GiB",
            "graphs          0.01 GiB",
            "kv pages        0.16 GiB",
        ],
    ),
    OffloadBudgetCase(
        0.43,
        0,
        0.5,
        True,
        "offload",
        5120,
        [
            "budget         16.10 GiB",
            "weights         1.28 GiB",
            "experts         8.50 GiB",
            "gdn slots       1.51 GiB",
            "activations     2.62 GiB",
            "graphs          0.01 GiB",
            "kv pages        2.17 GiB",
        ],
    ),
    # A 32 GB Mac's budget at rate 0.7.
    OffloadBudgetCase(
        0.58,
        0,
        0.7,
        True,
        "offload",
        7168,
        [
            "budget         21.72 GiB",
            "weights         1.28 GiB",
            "experts        11.88 GiB",
            "gdn slots       1.51 GiB",
            "activations     2.62 GiB",
            "graphs          0.01 GiB",
            "kv pages        4.41 GiB",
        ],
    ),
    # Without prefill overlap two steps of activations count.
    OffloadBudgetCase(
        0.43,
        4000,
        None,
        False,
        "offload",
        4000,
        [
            "budget         16.10 GiB",
            "weights         1.28 GiB",
            "experts         6.65 GiB",
            "gdn slots       1.51 GiB",
            "activations     5.25 GiB",
            "graphs          0.01 GiB",
            "kv pages        1.39 GiB",
        ],
    ),
    # The floors: one token's experts per layer (40 x 8) and, with prefill overlap, two whole
    # layers of buffers over them (2 x 256).
    OffloadBudgetCase(
        0.17,
        0,
        None,
        True,
        "offload",
        512,
        [
            "budget          6.36 GiB",
            "weights         1.28 GiB",
            "experts         0.91 GiB",
            "gdn slots       1.51 GiB",
            "activations     2.62 GiB",
            "graphs          0.01 GiB",
            "kv pages        0.03 GiB",
        ],
    ),
    OffloadBudgetCase(
        0.235,
        0,
        None,
        False,
        "offload",
        320,
        [
            "budget          8.80 GiB",
            "weights         1.28 GiB",
            "experts         0.59 GiB",
            "gdn slots       1.51 GiB",
            "activations     5.25 GiB",
            "graphs          0.01 GiB",
            "kv pages        0.15 GiB",
        ],
    ),
    # This box keeps every expert resident under auto; an offload's cache-auto takes them all.
    OffloadBudgetCase(
        None,
        0,
        None,
        True,
        "fused",
        TOTAL_EXPERTS,
        [
            "budget         36.44 GiB",
            "weights         1.28 GiB",
            "experts        16.94 GiB",
            "gdn slots       1.51 GiB",
            "activations     2.62 GiB",
            "graphs          0.01 GiB",
            "kv pages       14.07 GiB",
        ],
    ),
]


def offload_budget(
    config: EngineConfig,
) -> tuple[MoeStrategy, int, BudgetReport]:
    """What auto picks, the offload cache size and the offload's budget report, as the
    engine composes them for Qwen3.6-35B-A3B under config."""
    model_config = parse_config(QWEN3_6_TEXT_CONFIG)
    costs = pool_costs(model_config.kv_cache_group_specs(), config)
    budget = memory_budget_bytes(WORKING_SET_BYTES, config.memory_ratio)
    graphs = graph_reserve_bytes(config.max_running_req)
    reserve_pages = kv_reserve_pages(config.kv_reserve_tokens, config.page_size)
    resident = BudgetReport(
        budget=budget,
        weights=NON_EXPERT_BYTES + EXPERT_BYTES,
        activations=activation_reserve_bytes(
            config.max_extend_tokens, model_config.num_qo_heads, IN_FLIGHT_STEPS
        ),
        graphs=graphs,
        costs=costs,
    )
    in_flight_steps = OFFLOAD_IN_FLIGHT_STEPS if config.moe_prefill_overlap else IN_FLIGHT_STEPS
    activations = activation_reserve_bytes(
        config.max_extend_tokens, model_config.num_qo_heads, in_flight_steps
    )
    cache_size = resolve_moe_cache_size(
        config.moe_cache_size_flag(TOTAL_EXPERTS),
        report=BudgetReport(
            budget=budget,
            weights=NON_EXPERT_BYTES,
            activations=activations,
            graphs=graphs,
            costs=costs,
        ),
        reserve_pages=reserve_pages,
        slot_bytes=EXPERT_SLOT_BYTES,
        staging_bytes=STAGING_BYTES,
        num_layers=NUM_LAYERS,
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
        total_experts=TOTAL_EXPERTS,
        prefill_overlap=config.moe_prefill_overlap,
        buffer_placement=config.moe_prefill_buffer,
    )
    report = plan_budget(
        budget,
        NON_EXPERT_BYTES,
        activations,
        graphs,
        costs,
        experts=cache_size * EXPERT_SLOT_BYTES + STAGING_BYTES,
    )
    return choose_moe_strategy(resident, reserve_pages), cache_size, report


@pytest.mark.parametrize("case", OFFLOAD_BUDGET_CASES)
def test_qwen3_6_offload_budget(case: OffloadBudgetCase) -> None:
    config = EngineConfig(
        model_path="unused",
        max_extend_tokens_override=2048,
        memory_ratio=case.memory_ratio,
        moe_cache_size=case.moe_cache_size,
        moe_cache_rate=case.moe_cache_rate,
        moe_prefill_overlap=case.moe_prefill_overlap,
    )
    auto_strategy, cache_size, report = offload_budget(config)
    assert (auto_strategy, cache_size) == (case.auto_strategy, case.cache_size)
    assert report.lines() == case.lines


# (EngineConfig fields, the refusal)
REFUSED_OFFLOAD_CASES: list[tuple[dict[str, float | int | bool | None], str]] = [
    ({"moe_cache_size": 300}, "and the two prefill buffers: give at least 512"),
    ({"moe_cache_size": 200, "moe_prefill_overlap": False}, "give at least 320 slots$"),
    ({"memory_ratio": 0.15, "moe_prefill_overlap": False}, "does not fit the memory budget"),
    ({"moe_cache_size": 512, "moe_cache_rate": 0.5}, "moe_cache_size and moe_cache_rate are"),
    ({"moe_cache_rate": 0.5, "moe_cache_auto": True}, "moe_cache_rate and moe_cache_auto are"),
    ({"moe_cache_rate": 1.5}, r"moe_cache_rate must be in \[0, 1\]"),
    ({"memory_ratio": 0.0}, r"memory_ratio must be in \(0, 1\]"),
    ({"moe_victim_cache_gib": -1.0}, "moe_victim_cache_gib must be auto or a finite"),
    ({"moe_victim_cache_gib": float("nan")}, "moe_victim_cache_gib must be auto or a finite"),
]


@pytest.mark.parametrize(("fields", "error"), REFUSED_OFFLOAD_CASES)
def test_refused_offload_budgets(fields: dict[str, float | int | bool | None], error: str) -> None:
    with pytest.raises((ValueError, RuntimeError), match=error):
        offload_budget(EngineConfig(model_path="unused", max_extend_tokens_override=2048, **fields))


MEMORY_BYTES = 51_539_607_552  # hw.memsize, M4 Pro 48 GB
# (--moe-victim-cache-gib, hw.memsize, the victim cache's bytes, the warning logged): the
# memory beyond the working set is 10.56 GiB on this box.
VICTIM_CACHE_CASES: list[tuple[float | Literal["auto"], int, int, str]] = [
    (0, MEMORY_BYTES, 0, ""),
    # 4.56 GiB: all of it less the 6 GiB host reserve.
    ("auto", MEMORY_BYTES, 4_896_260_096, ""),
    ("auto", WORKING_SET_BYTES + 4 * GIB, 0, ""),
    (2.5, MEMORY_BYTES, 2_684_354_560, ""),
    (16, MEMORY_BYTES, 11_338_711_040, "--moe-victim-cache-gib 16 is capped at 10.56 GiB"),
]


@pytest.mark.parametrize(("requested", "memory_size", "expected", "warning"), VICTIM_CACHE_CASES)
def test_victim_cache_bytes(
    caplog: pytest.LogCaptureFixture,
    requested: float | Literal["auto"],
    memory_size: int,
    expected: int,
    warning: str,
) -> None:
    with caplog.at_level(logging.WARNING):
        assert victim_cache_bytes(requested, memory_size, WORKING_SET_BYTES) == expected
    assert warning in caplog.text and (warning != "") == bool(caplog.records)


# (--moe-strategy, EngineConfig flags, budget, the checkpoint's quantization mode or None for
# bf16, the strategy resolved) on a tiny checkpoint: its weights take a few MiB beside an
# activation reserve of two 0.75 GiB steps, and 8192 tokens of KV take 8 MiB. auto resolves
# alike whatever the experts' format.
STRATEGY_CASES: list[
    tuple[MoeStrategyChoice, dict[str, float | bool], int, str | None, MoeStrategy]
] = [
    ("auto", {}, 2 * GIB, "affine", "fused"),
    ("auto", {}, 1 * GIB, "affine", "offload"),
    ("auto", {"kv_reserve_tokens": 1 << 20}, 2 * GIB, None, "offload"),
    ("auto", {"kv_reserve_tokens": 1 << 20}, 2 * GIB, "mxfp4", "offload"),
    ("auto", {"moe_cache_rate": 0.5}, 2 * GIB, None, "offload"),
    ("auto", {"moe_cache_auto": True}, 2 * GIB, None, "offload"),
    ("fused", {"moe_cache_rate": 0.5}, 1 * GIB, "affine", "fused"),
    ("offload", {}, 2 * GIB, None, "offload"),
]
# Group size and bits of each mode the cases quantize in.
QUANT_SHAPES = {"affine": (32, 4), "mxfp4": (32, 4)}


def save_tiny_checkpoint(path: Path, mode: str | None) -> None:
    """The tiny checkpoint in bf16, or with every linear and the embedding in mode."""
    model = build_reference(seed=0)
    if mode is None:
        save_checkpoint(model, path, quantization=None)
    else:
        group_size, bits = QUANT_SHAPES[mode]
        nn.quantize(model, group_size, bits, mode=mode)
        mx.eval(model.parameters())
        quantization = {"group_size": group_size, "bits": bits, "mode": mode}
        save_checkpoint(model, path, quantization=quantization)


def resolve_tiny_strategy(
    path: Path, strategy: MoeStrategyChoice, flags: dict[str, float | bool], budget: int
) -> MoeStrategy:
    config = EngineConfig(
        model_path=str(path), max_extend_tokens_override=256, moe_strategy=strategy, **flags
    )
    model_config = config.model_config
    return resolve_moe_strategy(
        config,
        model_weight_rules(config.architecture, model_config),
        read_checkpoint(config.architecture, config.model_path).state,
        budget=budget,
        graphs=0,
        costs=pool_costs(model_config.kv_cache_group_specs(), config),
        encoder=0,
    )


@pytest.mark.parametrize(("strategy", "flags", "budget", "mode", "expected"), STRATEGY_CASES)
def test_moe_strategy_resolution(
    tmp_path: Path,
    strategy: MoeStrategyChoice,
    flags: dict[str, float | bool],
    budget: int,
    mode: str | None,
    expected: MoeStrategy,
) -> None:
    save_tiny_checkpoint(tmp_path, mode)
    assert resolve_tiny_strategy(tmp_path, strategy, flags, budget) == expected


def test_auto_keeps_experts_resident_without_offload(tmp_path: Path) -> None:
    """Gemma4 has no offload cache: where auto would offload, its experts stay resident beside
    less KV, or the model is refused before any weight is read when they leave no pages."""
    expert_bank = "language_model.model.layers.0.experts.switch_glu.gate_proj.weight"
    mx.save_safetensors(str(tmp_path / "model.safetensors"), {expert_bank: mx.zeros((128, 8, 8))})
    text_config = {**GEMMA4_26B_TEXT_CONFIG, "enable_moe_block": True, "num_experts": 128}
    checkpoint_config = {
        "architectures": ["Gemma4ForConditionalGeneration"],
        "model_type": "gemma4",
        "text_config": text_config,
    }
    (tmp_path / "config.json").write_text(json.dumps(checkpoint_config))
    # Its window pool alone takes 3.3 GiB beside the 1.5 GiB activation reserve.
    flags: dict[str, float | bool] = {"kv_reserve_tokens": 1 << 20}
    assert resolve_tiny_strategy(tmp_path, "auto", flags, 8 * GIB) == "fused"
    with pytest.raises(RuntimeError, match="model does not fit the memory budget"):
        resolve_tiny_strategy(tmp_path, "auto", {}, 4 * GIB)


def test_dense_model_refuses_moe_offload(tmp_path: Path) -> None:
    save_checkpoint(build_reference(seed=0), tmp_path, quantization=None)
    checkpoint_config = json.loads((tmp_path / "config.json").read_text())
    checkpoint_config["text_config"] = {**TINY_TEXT_CONFIG, "num_experts": 0}
    (tmp_path / "config.json").write_text(json.dumps(checkpoint_config))
    assert resolve_tiny_strategy(tmp_path, "auto", {}, 2 * GIB) == "fused"
    with pytest.raises(ValueError, match="routes no experts"):
        resolve_tiny_strategy(tmp_path, "offload", {}, 2 * GIB)
