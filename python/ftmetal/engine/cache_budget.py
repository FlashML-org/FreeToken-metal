"""Unified-memory budget: what the pools may take after the weights are resident.

Port of FreeToken engine/cache_budget.py and BaseKVCachePool.solve_num_pages. Divergences:
the budget is Metal's recommended working set minus a fixed headroom, or memory_ratio of
it; every pool prices its own spec as (bytes per page, fixed bytes), the fixed costs come
off first and the paged pools share the rest (FreeToken's engine adds the GDN state pool to
the KV family's fixed cost by hand); the lines are the ones this engine has (weights, the
offload cache's experts, one per pool, activations, the vision encoder's activations, the
decode graphs' arena); an offload without a cache size keeps the KV pools the context first
(--num-token, --max-seq-len or 64K tokens, not FreeToken's --kv-reserve-tokens) and gives
the experts the rest, out of what the other lines leave (no --moe-cache-auto); the auto
strategy keeps the experts resident when they fit beside that context (FreeToken's auto
always offloads a MoE model off unified memory); a --num-token or --max-seq-len the budget
cannot hold is refused at startup; the offload's victim cache lies outside the budget, in
the host memory beyond the working set (victim_cache_bytes).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import mlx.core as mx

from ftmetal.attention.metadata import AttnType
from ftmetal.kvcache.base import PoolCost
from ftmetal.models.config import MoeStrategy
from ftmetal.models.loader import config_int
from ftmetal.models.register import EncoderSpec
from ftmetal.moe.slot_pools import PrefillBufferPlacement, min_cache_size

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
# Host memory beyond Metal's working set that an auto victim cache leaves the system. On the
# 48 GB M4 Pro (10.56 GiB beyond it) the 4.56 GiB this leaves the cache ran the 2026-11-02
# victim cache A/B (single-turn cells, agentic replay) without swap and nearly no purges; PR #3's
# unpinned victim sweep on that box reached free ~0.1 GiB at 6 GiB and swap at 7.
VICTIM_CACHE_HOST_RESERVE_BYTES = 6 * GIB
# Each pool's budget line; the only LINEAR pool so far holds GDN state.
POOL_LABELS: dict[AttnType, str] = {
    AttnType.FULL: "kv pages",
    AttnType.SWA: "swa window",
    AttnType.LINEAR: "gdn slots",
    AttnType.QSA: "qsa kv pages",
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
    max_extend_tokens: int,
    num_qo_heads: int,
    in_flight_steps: int,
    measured: tuple[int, int] | None = None,
) -> int:
    """What in_flight_steps prefill steps of max_extend_tokens hold above the pools: the model's
    measured (fixed bytes, bytes per token) when it has one, else the Qwen3.6 estimate."""
    if measured is not None:
        fixed_bytes, bytes_per_token = measured
        return in_flight_steps * (fixed_bytes + max_extend_tokens * bytes_per_token)
    per_step = ACTIVATION_FIXED_BYTES + max_extend_tokens * ACTIVATION_BYTES_PER_TOKEN
    per_step += num_qo_heads * max_extend_tokens * max_extend_tokens * ATTENTION_SCORE_BYTES
    return in_flight_steps * per_step


def encoder_reserve_bytes(
    encoder: EncoderSpec,
    vision_config: dict[str, object],
    dtype: mx.Dtype,
    max_patches: int,
    feature_bytes: int,
) -> int:
    """The vision tower's peak above its weights for the largest image: per patch its
    feature_bytes-long feature row and the encoder's activation widths of hidden in the tower
    dtype, then one MLP chunk of its intermediate widths, and its fixed part. A [patches,
    patches] score matrix is not counted: the towers attend with a fused kernel."""
    hidden = config_int(vision_config, "hidden_size")
    intermediate = config_int(vision_config, "intermediate_size")
    per_patch = feature_bytes + dtype.size * encoder.activation_widths * hidden
    mlp_chunk = dtype.size * encoder.mlp_widths * encoder.mlp_rows * intermediate
    return encoder.fixed_bytes + max_patches * per_patch + mlp_chunk


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


def kv_reserve_pages(reserved_context_tokens: int, page_size: int) -> int:
    return -(-reserved_context_tokens // page_size)


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
    prefill_overlap: bool,
    buffer_placement: PrefillBufferPlacement,
) -> int:
    """The offload cache's slots: requested when set (at least one token's experts per layer,
    and the two prefill buffers with prefill overlap: min_cache_size), else what the paged
    pools of report (which has no experts line) leave past reserve_pages, from that floor to
    every expert."""
    floor = min_cache_size(
        num_layers,
        num_experts,
        top_k,
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


def victim_cache_bytes(
    requested_gib: float | Literal["auto"], memory_size: int, recommended_working_set: int
) -> int:
    """The MoE offload's victim cache, in the host memory beyond Metal's working set: auto
    takes it less VICTIM_CACHE_HOST_RESERVE_BYTES, a size is capped at all of it."""
    beyond = max(0, memory_size - recommended_working_set)
    if requested_gib == "auto":
        return max(0, beyond - VICTIM_CACHE_HOST_RESERVE_BYTES)
    elif int(requested_gib * GIB) > beyond:
        logger.warning(
            f"--moe-victim-cache-gib {requested_gib} is capped at {beyond / GIB:.2f} GiB, the "
            f"memory beyond Metal's working set"
        )
        return beyond
    else:
        return int(requested_gib * GIB)


def solve_num_pages(
    report: BudgetReport,
    page_size: int,
    num_page_override: int | None,
    *,
    num_token_override: int | None,
    context_tokens: int,
    is_context_requested: bool,
) -> int:
    """Usable pages the paged pools get: --num-pages as given, --num-token in whole pages, else
    their share of the budget. A budget short of the context refuses a requested one and warns
    about the default."""
    budget_pages = report.paged_bytes // report.bytes_per_page
    if num_page_override is not None:
        num_pages = num_page_override
    elif num_token_override is not None:
        num_pages = min(budget_pages, kv_reserve_pages(num_token_override, page_size))
    else:
        num_pages = budget_pages
    num_tokens = num_pages * page_size
    is_short_of_context = num_page_override is None and num_tokens < context_tokens
    if num_pages <= 1:
        raise RuntimeError(
            "not enough memory for the KV cache; raise --memory-ratio, lower "
            "--max-prefill-length or shrink the MoE cache"
        )
    elif is_short_of_context and is_context_requested:
        raise ValueError(
            f"a context of {context_tokens} tokens does not fit the memory budget: the KV pools "
            f"get {num_tokens} tokens beside the weights and any MoE cache"
        )
    elif is_short_of_context:
        logger.warning(
            f"the memory budget gives the KV pools {num_tokens} tokens, less than the default "
            f"context of {context_tokens}"
        )
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
    "VICTIM_CACHE_HOST_RESERVE_BYTES",
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
    "victim_cache_bytes",
]
