"""Unified-memory budget: what the pools may take after the weights are resident.

Port of FreeToken engine/cache_budget.py. Divergences: the budget is Metal's
recommended working set minus a fixed headroom, and the lines are the ones this
engine has (weights, KV pages, GDN slots, activations, the MLX allocator cache).
"""

from __future__ import annotations

from dataclasses import dataclass

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


def memory_budget_bytes(recommended_working_set: int) -> int:
    headroom = max(int(recommended_working_set * HEADROOM_FRACTION), HEADROOM_MIN_BYTES)
    return recommended_working_set - headroom


def activation_reserve_bytes(max_extend_tokens: int, num_qo_heads: int) -> int:
    per_step = ACTIVATION_FIXED_BYTES + max_extend_tokens * ACTIVATION_BYTES_PER_TOKEN
    per_step += num_qo_heads * max_extend_tokens * max_extend_tokens * ATTENTION_SCORE_BYTES
    return IN_FLIGHT_STEPS * per_step


@dataclass(frozen=True)
class BudgetReport:
    budget: int
    weights: int
    linear_state: int
    activations: int
    kv_available: int

    def lines(self) -> list[str]:
        return [
            f"budget      {self.budget / GIB:8.2f} GiB",
            f"weights     {self.weights / GIB:8.2f} GiB",
            f"gdn slots   {self.linear_state / GIB:8.2f} GiB",
            f"activations {self.activations / GIB:8.2f} GiB",
            f"kv pages    {self.kv_available / GIB:8.2f} GiB",
        ]


def plan_budget(budget: int, weights: int, linear_state: int, activations: int) -> BudgetReport:
    kv_available = budget - weights - linear_state - activations
    report = BudgetReport(budget, weights, linear_state, activations, kv_available)
    if kv_available <= 0:
        raise RuntimeError("model does not fit the memory budget:\n" + "\n".join(report.lines()))
    return report


__all__ = [
    "ACTIVATION_BYTES_PER_TOKEN",
    "ACTIVATION_FIXED_BYTES",
    "ATTENTION_SCORE_BYTES",
    "IN_FLIGHT_STEPS",
    "BudgetReport",
    "activation_reserve_bytes",
    "memory_budget_bytes",
    "plan_budget",
]
