"""Unified-memory budget: what the pools may take after the weights are resident.

Port of FreeToken engine/cache_budget.py and BaseKVCachePool.solve_num_pages. Divergences:
the budget is Metal's recommended working set minus a fixed headroom, or memory_ratio of
it; every pool prices its own spec as (bytes per page, fixed bytes), the fixed costs come
off first and the paged pools share the rest (FreeToken's engine adds the GDN state pool to
the KV family's fixed cost by hand); the lines are the ones this engine has (weights, the
offload cache's experts, one per pool, activations, the vision encoder's activations, the
decode graphs' arena); --moe-cache-auto (plan_cache_budget) gives the KV pools
kv_reserve_tokens first and the experts the rest, out of what the other lines leave; the
auto strategy keeps the experts resident when they fit beside that KV reserve (FreeToken's
auto always offloads a MoE model off unified memory).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import mlx.core as mx

from ftmetal.attention.metadata import AttnType
from ftmetal.kvcache.base import PoolCost
from ftmetal.models.config import MoeStrategy
from ftmetal.models.loader import config_int
from ftmetal.models.qwen3_5_moe.vision import MLP_ROWS
from ftmetal.moe.slot_pools import (
    MoePoolGrouping,
    PrefillBufferPlacement,
    min_cache_size,
)

logger = logging.getLogger(__name__)

GIB = 1 << 30
HEADROOM_FRACTION = 0.02
HEADROOM_MIN_BYTES = 1 * GIB
# Prefill peak above the resident pools, measured on Qwen3.6 (chunks 256..3072 tokens):
# a fixed part, a linear part (MoE gather rows dominate) and the bf16 attention scores.
ACTIVATION_FIXED_BYTES = 512 << 20
ACTIVATION_BYTES_PER_TOKEN = 1 << 20
ATTENTION_SCORE_BYTES = 2
# Two steps can be in flight under the overlap loop.
IN_FLIGHT_STEPS = 2
# A MoE offload's overlap prefill waits for the step before it, so one step's activations
# are in flight (an inference, to check against the real model's peak memory).
OFFLOAD_IN_FLIGHT_STEPS = 1
# The decode CommandGraphs' shared arena, rounded up from Qwen3.6-35B-A3B at a 256k-token
# page table: 1.05 MiB at batch size 1 and 8.30 MiB at 8, nearly all flash-decode
# partials (their partition count follows the page-table width).
GRAPH_FIXED_BYTES = 4 << 20
GRAPH_BYTES_PER_ROW = 5 << 18
# Hidden widths per patch live in a vision tower block as it attends: the residual stream, the
# normed copy, qkv and its bias-add copy (6), the rotated q and k halves and their
# concatenation (4), and the attention output. Measured on Qwen3.6's tower in bf16, the peak
# above the weights and the float32 pixels is 12.2 widths per patch plus 43 MiB (482.6 MiB at
# 16384 patches, 1802.4 MiB at 65536).
VISION_ACTIVATION_WIDTHS = 13
# Each pool's budget line; the only LINEAR pool so far holds GDN state.
POOL_LABELS: dict[AttnType, str] = {
    AttnType.FULL: "kv pages",
    AttnType.SWA: "swa window",
    AttnType.LINEAR: "gdn slots",
}


def memory_budget_bytes(recommended_working_set: int, memory_ratio: float | None) -> int:
    """memory_ratio of the working set (FreeToken's --memory-ratio), else the working set
    less its headroom."""
    if memory_ratio is not None:
        return int(recommended_working_set * memory_ratio)
    else:
        headroom = max(int(recommended_working_set * HEADROOM_FRACTION), HEADROOM_MIN_BYTES)
        return recommended_working_set - headroom


def activation_reserve_bytes(
    max_extend_tokens: int, num_qo_heads: int, in_flight_steps: int
) -> int:
    per_step = ACTIVATION_FIXED_BYTES + max_extend_tokens * ACTIVATION_BYTES_PER_TOKEN
    per_step += num_qo_heads * max_extend_tokens * max_extend_tokens * ATTENTION_SCORE_BYTES
    return in_flight_steps * per_step


def encoder_reserve_bytes(
    vision_config: dict[str, object], dtype: mx.Dtype, max_patches: int
) -> int:
    """The vision tower's peak above its weights for the largest image. Per patch: its bf16
    feature and, in the tower dtype, VISION_ACTIVATION_WIDTHS hidden widths; then one MLP row
    chunk's fc1 output and its gelu. A [patches, patches] score matrix is not counted: the
    tower attends with a fused kernel."""
    patch_dim = (
        config_int(vision_config, "in_channels")
        * config_int(vision_config, "temporal_patch_size")
        * config_int(vision_config, "patch_size") ** 2
    )
    hidden = config_int(vision_config, "hidden_size")
    intermediate = config_int(vision_config, "intermediate_size")
    per_patch = mx.bfloat16.size * patch_dim + dtype.size * VISION_ACTIVATION_WIDTHS * hidden
    mlp_chunk = dtype.size * 2 * MLP_ROWS * intermediate
    return max_patches * per_patch + mlp_chunk


def graph_reserve_bytes(max_graph_bs: int) -> int:
    """What decode graphs up to batch size max_graph_bs keep: the arena they share, and
    the few buffers they pin."""
    return GRAPH_FIXED_BYTES + max_graph_bs * GRAPH_BYTES_PER_ROW


def budget_line(label: str, num_bytes: int) -> str:
    return f"{label:<12}{num_bytes / GIB:8.2f} GiB"


@dataclass(frozen=True, kw_only=True)
class BudgetReport:
    budget: int
    weights: int
    activations: int
    graphs: int
    costs: dict[AttnType, PoolCost]
    # The vision encoder's activations; 0 without a tower.
    encoder: int = 0
    # The offload cache: its expert slots and IO staging; 0 while the experts are resident
    # (they are weights then).
    experts: int = 0

    @property
    def bytes_per_page(self) -> int:
        return sum(cost.bytes_per_page for cost in self.costs.values())

    @property
    def paged_bytes(self) -> int:
        """What the paged pools share once the weights, activations, graphs and fixed costs
        are off."""
        fixed_bytes = sum(cost.fixed_bytes for cost in self.costs.values())
        reserved = self.weights + self.experts + self.activations + self.encoder + self.graphs
        return self.budget - reserved - fixed_bytes

    def holds_pages(self, num_pages: int) -> bool:
        """Whether the paged pools get num_pages pages."""
        return self.paged_bytes >= num_pages * self.bytes_per_page

    def lines(self) -> list[str]:
        """The deductions in order, then each paged pool's share of what is left."""
        fixed_lines = [
            budget_line(POOL_LABELS[attn_type], cost.fixed_bytes)
            for attn_type, cost in self.costs.items()
            if cost.fixed_bytes > 0
        ]
        paged_lines = [
            budget_line(
                POOL_LABELS[attn_type],
                self.paged_bytes * cost.bytes_per_page // self.bytes_per_page,
            )
            for attn_type, cost in self.costs.items()
            if cost.bytes_per_page > 0
        ]
        expert_lines = [budget_line("experts", self.experts)] if self.experts > 0 else []
        encoder_lines = [budget_line("encoder", self.encoder)] if self.encoder > 0 else []
        graph_lines = [budget_line("graphs", self.graphs)] if self.graphs > 0 else []
        return [
            budget_line("budget", self.budget),
            budget_line("weights", self.weights),
            *expert_lines,
            *fixed_lines,
            budget_line("activations", self.activations),
            *encoder_lines,
            *graph_lines,
            *paged_lines,
        ]


def kv_reserve_pages(kv_reserve_tokens: int, page_size: int) -> int:
    return -(-kv_reserve_tokens // page_size)


def choose_moe_strategy(resident: BudgetReport, reserve_pages: int) -> MoeStrategy:
    """--moe-strategy auto with no cache size flag: fused when the report of every weight
    resident leaves the paged pools reserve_pages, else offload."""
    if resident.holds_pages(reserve_pages):
        return "fused"
    else:
        return "offload"


def resolve_moe_cache_size(
    requested: int,
    *,
    report: BudgetReport,
    reserve_pages: int,
    slot_bytes: int,
    staging_bytes: int,
    num_layers: int,
    num_experts: int,
    top_k: int,
    total_experts: int,
    grouping: MoePoolGrouping,
    prefill_overlap: bool,
    buffer_placement: PrefillBufferPlacement,
) -> int:
    """The offload cache's slots: requested when set (at least one token's experts per layer,
    and the two prefill buffers with prefill overlap: min_cache_size), else --moe-cache-auto:
    what the paged pools of report (which has no experts line) leave past reserve_pages, from
    that floor to every expert."""
    floor = min_cache_size(
        num_layers,
        num_experts,
        top_k,
        grouping=grouping,
        prefill_overlap=prefill_overlap,
        buffer_placement=buffer_placement,
    )
    if requested > 0 and requested < floor:
        reason = (
            "one token's experts per layer and the two prefill buffers"
            if prefill_overlap
            else "one token's experts per layer"
        )
        hint = " or pass --disable-moe-prefill-overlap" if prefill_overlap else ""
        raise ValueError(
            f"--moe-cache-size {requested} cannot hold {reason}: give at least {floor} slots{hint}"
        )
    elif requested > 0:
        return requested
    else:
        spare = report.paged_bytes - reserve_pages * report.bytes_per_page - staging_bytes
        return max(floor, min(spare // slot_bytes, total_experts))


def plan_budget(
    budget: int,
    weights: int,
    activations: int,
    graphs: int,
    costs: dict[AttnType, PoolCost],
    encoder: int = 0,
    experts: int = 0,
) -> BudgetReport:
    report = BudgetReport(
        budget=budget,
        weights=weights,
        activations=activations,
        graphs=graphs,
        costs=costs,
        encoder=encoder,
        experts=experts,
    )
    assert report.bytes_per_page > 0, "no paged pool to give the budget to"
    if report.paged_bytes <= 0:
        raise RuntimeError("model does not fit the memory budget:\n" + "\n".join(report.lines()))
    return report


def solve_num_pages(report: BudgetReport, page_size: int, num_page_override: int | None) -> int:
    """Usable pages the paged pools get: their share of the budget unless overridden."""
    if num_page_override is None:
        num_pages = report.paged_bytes // report.bytes_per_page
    else:
        num_pages = num_page_override
    if num_pages <= 1:
        raise RuntimeError("not enough memory for the KV cache; lower --max-seq-len")
    logger.info(
        f"KV cache: {num_pages} pages x {page_size} tokens, "
        f"{num_pages * report.bytes_per_page / GIB:.2f} GiB"
    )
    return num_pages


__all__ = [
    "ACTIVATION_BYTES_PER_TOKEN",
    "ACTIVATION_FIXED_BYTES",
    "ATTENTION_SCORE_BYTES",
    "GRAPH_BYTES_PER_ROW",
    "GRAPH_FIXED_BYTES",
    "IN_FLIGHT_STEPS",
    "OFFLOAD_IN_FLIGHT_STEPS",
    "POOL_LABELS",
    "VISION_ACTIVATION_WIDTHS",
    "BudgetReport",
    "activation_reserve_bytes",
    "choose_moe_strategy",
    "encoder_reserve_bytes",
    "graph_reserve_bytes",
    "kv_reserve_pages",
    "memory_budget_bytes",
    "plan_budget",
    "resolve_moe_cache_size",
    "solve_num_pages",
]
