"""Memory budget on the Qwen3.6-35B-A3B geometry: report lines, page and GDN slot counts;
and Gemma4-26B-A4B's window pool as a fixed cost.

The expected numbers were computed by the engine's sizing before pools were created per
attention type; the dispatch must reproduce them exactly from the same fixed budget.
"""

from __future__ import annotations

import logging
from typing import NamedTuple

import pytest

from ftmetal.attention.metadata import AttnType
from ftmetal.engine.cache_budget import (
    GIB,
    activation_reserve_bytes,
    graph_reserve_bytes,
    memory_budget_bytes,
    plan_budget,
    solve_num_pages,
)
from ftmetal.engine.config import EngineConfig
from ftmetal.kvcache import pool_costs
from ftmetal.kvcache.base import PoolCost
from ftmetal.models.gemma4.config import parse_config as parse_gemma4_config
from ftmetal.models.qwen3_5_moe.config import parse_config

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
WORKING_SET_BYTES = 40_200_896_512  # max_recommended_working_set_size, M4 Pro 48 GB
WEIGHTS_BYTES = 20 * GIB
# 30 GDN layers x (conv ring 4 x 8192 bf16 + state 32 x 128 x 128 fp32).
GDN_SLOT_BYTES = 64_880_640


class BudgetCase(NamedTuple):
    page_size: int
    max_running_req: int
    linear_state_cache_ratio: float
    max_extend_tokens: int
    num_page_override: int | None
    num_pages: int
    num_slots: int
    lines: list[str]
    pages_line: str


CASES = [
    BudgetCase(
        page_size=1,
        max_running_req=4,
        linear_state_cache_ratio=2.0,
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
        linear_state_cache_ratio=2.0,
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
        linear_state_cache_ratio=0.5,
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
]


@pytest.mark.parametrize("case", CASES, ids=["page1", "page32", "override"])
def test_qwen3_6_budget_reproduces_pages_and_slots(
    case: BudgetCase, caplog: pytest.LogCaptureFixture
) -> None:
    model_config = parse_config(QWEN3_6_TEXT_CONFIG)
    config = EngineConfig(
        model_path="unused",
        page_size=case.page_size,
        max_running_req=case.max_running_req,
        linear_state_cache_ratio=case.linear_state_cache_ratio,
        max_extend_tokens=case.max_extend_tokens,
        num_page_override=case.num_page_override,
    )
    costs = pool_costs(model_config.kv_cache_group_specs(), config)
    assert costs[AttnType.LINEAR] == PoolCost(0, case.num_slots * GDN_SLOT_BYTES)

    activations = activation_reserve_bytes(config.max_extend_tokens, model_config.num_qo_heads)
    budget = memory_budget_bytes(WORKING_SET_BYTES)
    report = plan_budget(budget, WEIGHTS_BYTES, activations, 0, costs)
    assert report.lines() == case.lines
    caplog.set_level(logging.INFO)
    num_pages = solve_num_pages(report, config.page_size, config.num_page_override)
    assert num_pages == case.num_pages
    assert caplog.messages == [case.pages_line]


def test_graph_reserve_comes_off_the_paged_pools() -> None:
    model_config = parse_config(QWEN3_6_TEXT_CONFIG)
    config = EngineConfig(model_path="unused", max_running_req=8)
    costs = pool_costs(model_config.kv_cache_group_specs(), config)
    activations = activation_reserve_bytes(config.max_extend_tokens, model_config.num_qo_heads)
    budget = memory_budget_bytes(WORKING_SET_BYTES)
    graphs = graph_reserve_bytes(8)
    without = plan_budget(budget, WEIGHTS_BYTES, activations, 0, costs)
    with_graphs = plan_budget(budget, WEIGHTS_BYTES, activations, graphs, costs)
    assert without.paged_bytes - with_graphs.paged_bytes == graphs
    assert "graphs" not in "".join(without.lines())
    lines = with_graphs.lines()
    assert lines.index("graphs          0.01 GiB") == lines.index("activations     5.25 GiB") + 1


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


def test_gemma4_window_pool_is_a_fixed_cost_beside_the_global_pages() -> None:
    model_config = parse_gemma4_config(GEMMA4_26B_TEXT_CONFIG)
    config = EngineConfig(model_path="unused", max_running_req=4, max_extend_tokens=2048)
    costs = pool_costs(model_config.kv_cache_group_specs(), config)
    # 5 global layers x 2 x 2 heads x 512 x bf16 per token; 25 window layers x 2 x 8 heads
    # x 256 x bf16 per slot, for 5 x align32(1024 + 2048 + 1) slots.
    assert costs == {
        AttnType.FULL: PoolCost(20 * 1024, 0),
        AttnType.SWA: PoolCost(0, 15520 * 200 * 1024),
    }
    report = plan_budget(memory_budget_bytes(WORKING_SET_BYTES), 13 * GIB, 0, 0, costs)
    assert "swa window      2.96 GiB" in report.lines()
