"""Unified-memory budget: what the pools may take after the weights are resident.

Port of FreeToken engine/cache_budget.py and BaseKVCachePool.solve_num_pages. Divergences:
the budget is Metal's recommended working set minus a fixed headroom; every pool prices
its own spec as (bytes per page, fixed bytes), the fixed costs come off first and the
paged pools share the rest (FreeToken's engine adds the GDN state pool to the KV
family's fixed cost by hand); the lines are the ones this engine has (weights, one per
pool, activations); no MoE expert cache.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from ftmetal.attention.metadata import AttnType
from ftmetal.kvcache.base import PoolCost

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
# Each pool's budget line; the only LINEAR pool so far holds GDN state.
POOL_LABELS: dict[AttnType, str] = {AttnType.FULL: "kv pages", AttnType.LINEAR: "gdn slots"}


def memory_budget_bytes(recommended_working_set: int) -> int:
    headroom = max(int(recommended_working_set * HEADROOM_FRACTION), HEADROOM_MIN_BYTES)
    return recommended_working_set - headroom


def activation_reserve_bytes(max_extend_tokens: int, num_qo_heads: int) -> int:
    per_step = ACTIVATION_FIXED_BYTES + max_extend_tokens * ACTIVATION_BYTES_PER_TOKEN
    per_step += num_qo_heads * max_extend_tokens * max_extend_tokens * ATTENTION_SCORE_BYTES
    return IN_FLIGHT_STEPS * per_step


def budget_line(label: str, num_bytes: int) -> str:
    return f"{label:<12}{num_bytes / GIB:8.2f} GiB"


@dataclass(frozen=True, kw_only=True)
class BudgetReport:
    budget: int
    weights: int
    activations: int
    costs: dict[AttnType, PoolCost]

    @property
    def bytes_per_page(self) -> int:
        return sum(cost.bytes_per_page for cost in self.costs.values())

    @property
    def paged_bytes(self) -> int:
        """What the paged pools share once the weights, activations and fixed costs are off."""
        fixed_bytes = sum(cost.fixed_bytes for cost in self.costs.values())
        return self.budget - self.weights - self.activations - fixed_bytes

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
        return [
            budget_line("budget", self.budget),
            budget_line("weights", self.weights),
            *fixed_lines,
            budget_line("activations", self.activations),
            *paged_lines,
        ]


def plan_budget(
    budget: int, weights: int, activations: int, costs: dict[AttnType, PoolCost]
) -> BudgetReport:
    report = BudgetReport(budget=budget, weights=weights, activations=activations, costs=costs)
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
    "IN_FLIGHT_STEPS",
    "POOL_LABELS",
    "BudgetReport",
    "activation_reserve_bytes",
    "memory_budget_bytes",
    "plan_budget",
    "solve_num_pages",
]
