"""GraphRunner: decode steps replayed from one CommandGraph capture per batch-size bucket.

Port of FreeToken engine/graph.py. Divergences: a graph is a CommandGraph capture
(kernel/command_graph.py) instead of a CUDA graph, so there is no GraphCaptureBuffer:
the step inputs (input_ids, positions, out_loc and the attention metadata a backend's
prepare_for_replay builds at the bucket's fixed shapes) are rebound by every replay,
and so are the pools, which the replay takes and returns; the bs ladder ends at
max_running_req, on the ladder or not; the intermediates and temporaries of every
capture are planned into one arena all buckets share, allocated after the captures,
and a temporary a replay reads before writing (poison test) stays pinned instead; a
bucket whose capture or arena check fails is not replayed, pad_batch pads its batches
to the next captured bucket instead.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, replace

import mlx.core as mx
from mlx.utils import tree_flatten

from ftmetal.attention.base import BaseAttnBackend
from ftmetal.attention.metadata import AttnType
from ftmetal.core import Batch, Req
from ftmetal.kernel._ext import _C
from ftmetal.kernel.command_graph import (
    ArenaPlan,
    capture,
    place_blocks,
    plan_graph_arena,
    replay,
)
from ftmetal.kvcache import pool_arrays, set_pool_arrays
from ftmetal.kvcache.base import BasePool
from ftmetal.kvcache.linear_state_pool import LinearStatePool
from ftmetal.models.blocks import BaseLLMModel

logger = logging.getLogger(__name__)

MIB = 1 << 20

# Batch-size buckets: 1, 2, 4, then every multiple of this.
GRAPH_BS_STRIDE = 8


def determine_graph_bs(max_running_req: int) -> list[int]:
    """The ladder below max_running_req, then max_running_req itself: a full batch replays."""
    candidates = [1, 2, 4, *range(GRAPH_BS_STRIDE, max_running_req, GRAPH_BS_STRIDE)]
    return [bs for bs in candidates if bs < max_running_req] + [max_running_req]


def undonated_copies(kernel_names: list[str]) -> int:
    """Same-dtype vector copies (v_copy<T><T>): an eager step donates their input and skips
    them, a capture holds the input; in Qwen3.5 decode the partial-rotary RoPE of q and k."""
    count = 0
    for name in kernel_names:
        dtypes = name.removeprefix("v_copy")
        half = len(dtypes) // 2
        count += name.startswith("v_copy") and dtypes[:half] == dtypes[half:]
    return count


def bit_equal(first: mx.array, second: mx.array) -> bool:
    return (
        first.shape == second.shape
        and first.dtype == second.dtype
        and mx.array_equal(first.view(mx.uint8), second.view(mx.uint8)).item() is True
    )


@dataclass(frozen=True, kw_only=True)
class ReplayCheck:
    """A bucket's check batch: dummy rows on their own GDN slots (the dummy rows of a
    capture share the padding slot and race on it), and a clean replay's logits."""

    step_inputs: list[mx.array]
    slots: list[int]
    logits: mx.array


class GraphRunner:
    def __init__(
        self,
        model: BaseLLMModel,
        attn_backends: dict[AttnType, BaseAttnBackend],
        pools: dict[AttnType, BasePool],
        dummy_req: Req,
        graph_bs_list: list[int],
        max_seq_len: int,
        reserve_bytes: int,
    ) -> None:
        """An empty graph_bs_list turns replay off; max_seq_len is the page-table row width;
        the graphs may keep reserve_bytes (their arena and pinned buffers)."""
        self.model = model
        self.attn_backends = attn_backends
        self.pools = pools
        self.dummy_req = dummy_req
        self.graph_bs_list = sorted(graph_bs_list)
        self.reserve_bytes = reserve_bytes
        self.graphs: dict[int, _C.Graph] = {}
        self.plans: dict[int, ArenaPlan] = {}
        self.arena: mx.array | None = None
        linear_state_pool = pools.get(AttnType.LINEAR)
        assert linear_state_pool is None or isinstance(linear_state_pool, LinearStatePool)
        self.linear_state_pool = linear_state_pool
        if self.graph_bs_list:
            self.capture_all(max_seq_len)
        else:
            logger.info("CommandGraph is off: decode runs eagerly")

    def capture_all(self, max_seq_len: int) -> None:
        """Capture, poison-test and plan every bucket on dummy requests, largest first, then
        bind all to one arena and check each replay on it against its capture's."""
        for backend in self.attn_backends.values():
            backend.init_capture_graph(max_seq_len, self.graph_bs_list)
        started = time.perf_counter()
        pool = self.linear_state_pool
        check_slots = pool.alloc(max(self.graph_bs_list)) if pool is not None else []
        checks: dict[int, ReplayCheck] = {}
        for bs in sorted(self.graph_bs_list, reverse=True):
            bucket_started = time.perf_counter()
            try:
                graph = self.capture(self.dummy_batch([self.dummy_req] * bs))
                check = self.clean_check(graph, bs, check_slots[:bs])
                pinned = self.poisoned_temporaries(graph, check)
                plan = plan_graph_arena(graph, pinned)
                place_blocks(graph, plan)
            except (RuntimeError, ValueError) as error:
                logger.warning(f"CommandGraph capture of batch size {bs} failed, eager: {error}")
                # A capture refused before its eval leaves the dummy step's pools lazy.
                mx.eval(*pool_arrays(self.pools))
                continue
            self.graphs[bs], self.plans[bs], checks[bs] = graph, plan, check
            stats = graph.stats
            elapsed = time.perf_counter() - bucket_started
            logger.info(
                f"captured batch size {bs} in {elapsed:.2f} s: {stats.num_dispatches} dispatches "
                f"({undonated_copies(graph.dispatch_names)} same-dtype vector copies), "
                f"{len(stats.temporaries)} temporaries ({len(pinned)} fail the poison test), "
                f"arena {plan.arena_bytes / MIB:.2f} MiB (lower bound "
                f"{plan.lower_bound / MIB:.2f} MiB), pinned "
                f"{stats.capture_pinned_bytes / MIB:.1f} -> {stats.pinned_bytes / MIB:.3f} MiB"
            )
        while self.graphs and self.graph_bytes() > self.reserve_bytes:
            bs = max(self.graphs)
            logger.warning(
                f"CommandGraph: {self.graph_bytes() / MIB:.2f} MiB of arena and pinned buffers "
                f"exceed the {self.reserve_bytes / MIB:.2f} MiB graph reserve: batch size {bs} "
                f"runs eagerly"
            )
            self.graphs.pop(bs)
            self.plans.pop(bs)
        if self.graphs:
            self.bind_arena(checks)
        if pool is not None:
            pool.free(check_slots)
        pinned_bytes = sum(graph.stats.pinned_bytes for graph in self.graphs.values())
        arena_bytes = 0 if self.arena is None else self.arena.nbytes
        logger.info(
            f"CommandGraph: batch sizes {sorted(self.graphs)} captured in "
            f"{time.perf_counter() - started:.1f} s; arena {arena_bytes / MIB:.2f} MiB, "
            f"pinned {pinned_bytes / MIB:.3f} MiB, reserve {self.reserve_bytes / MIB:.2f} MiB"
        )

    def graph_bytes(self) -> int:
        """What the graphs keep once bound: the arena they share and their pinned buffers."""
        arena_bytes = max(self.plans[bs].arena_bytes for bs in self.graphs)
        return arena_bytes + sum(graph.stats.pinned_bytes for graph in self.graphs.values())

    def bind_arena(self, checks: dict[int, ReplayCheck]) -> None:
        """One arena for every graph; a graph whose replay on the arena filled with 0xFF
        differs from its clean replay stays eager."""
        arena = mx.zeros((max(plan.arena_bytes for plan in self.plans.values()),), mx.uint8)
        mx.eval(arena)
        self.arena = arena
        for bs in sorted(self.graphs, reverse=True):
            graph = self.graphs[bs]
            _C.bind_arena(graph, arena)
            _C.poison_arena(graph)
            check = checks[bs]
            if not bit_equal(self.replay_rows(graph, check.step_inputs, check.slots), check.logits):
                logger.warning(f"CommandGraph batch size {bs} differs on the arena, eager")
                self.graphs.pop(bs)
                self.plans.pop(bs)

    def dummy_batch(self, reqs: list[Req]) -> Batch:
        """A decode batch of dummy requests with its metadata for a capture."""
        batch = Batch(reqs=reqs, phase="decode")
        batch.input_ids = mx.zeros((len(reqs),), dtype=mx.int32)
        batch.positions = mx.full((len(reqs),), self.dummy_req.cached_len, dtype=mx.int32)
        for backend in self.attn_backends.values():
            backend.prepare_for_capture(batch)
        return batch

    def clean_check(self, graph: _C.Graph, bs: int, slots: list[int]) -> ReplayCheck:
        """The check batch of bucket bs (its GDN rows on slots), and a replay of it."""
        if self.linear_state_pool is None:
            reqs = [self.dummy_req] * bs
        else:
            reqs = [replace(self.dummy_req, linear_slot_idx=slot) for slot in slots]
        step_inputs = self.step_inputs(self.dummy_batch(reqs))
        mx.eval(*step_inputs)
        logits = self.replay_rows(graph, step_inputs, slots)
        return ReplayCheck(step_inputs=step_inputs, slots=slots, logits=logits)

    def replay_rows(
        self, graph: _C.Graph, step_inputs: list[mx.array], slots: list[int]
    ) -> mx.array:
        """Logits of a replay of a check batch, its GDN slots zeroed first."""
        if self.linear_state_pool is not None:
            self.linear_state_pool.clear_slots(slots)
        logits, pools = replay(graph, step_inputs, pool_arrays(self.pools))
        set_pool_arrays(self.pools, pools)
        mx.eval(logits, *pools)
        return logits

    def poisoned_temporaries(self, graph: _C.Graph, check: ReplayCheck) -> list[int]:
        """Temporaries a replay reads before it writes them (host-initialized constants):
        filled with 0xFF, each changes the logits. One replay poisons them all; only when
        that one differs is each tested alone."""
        temporaries = graph.temporary_blocks

        def logits_change(blocks: list[int]) -> bool:
            _C.poison_blocks(graph, blocks)
            logits = self.replay_rows(graph, check.step_inputs, check.slots)
            changed = not bit_equal(logits, check.logits)
            _C.restore_blocks(graph)
            return changed

        if not temporaries or not logits_change(temporaries):
            return []
        pinned = [block for block in temporaries if logits_change([block])]
        labels = [graph.block_labels[block] for block in pinned]
        logger.warning(f"CommandGraph: {len(pinned)} temporaries stay pinned: {labels}")
        return pinned

    def capture(self, batch: Batch) -> _C.Graph:
        """One real forward of a padded decode batch, its metadata prepared, under the recorder."""
        parameters = tree_flatten(self.model.parameters())
        externals = [value for _, value in parameters if isinstance(value, mx.array)]
        step_inputs = self.step_inputs(batch)
        mx.eval(*step_inputs)
        pools = pool_arrays(self.pools)
        logits = self.model.forward(batch)
        return capture([logits, *pool_arrays(self.pools)], step_inputs, pools, externals)

    def step_inputs(self, batch: Batch) -> list[mx.array]:
        """input_ids, positions, out_loc, then each backend's metadata arrays, in capture order."""
        arrays = [batch.input_ids, batch.positions, batch.out_loc]
        for attn_type in self.attn_backends:
            arrays += batch.attn_metadata[attn_type].step_inputs()
        return arrays

    def can_replay(self, batch: Batch) -> bool:
        return batch.is_decode and batch.padded_size in self.graphs

    def replay(self, batch: Batch) -> mx.array:
        """Logits [batch.size, vocab] of a padded decode batch; the pools are rebound."""
        assert self.can_replay(batch)
        for backend in self.attn_backends.values():
            backend.prepare_for_replay(batch)
        graph = self.graphs[batch.padded_size]
        logits, pools = replay(graph, self.step_inputs(batch), pool_arrays(self.pools))
        set_pool_arrays(self.pools, pools)
        return logits[: batch.size]

    def pad_batch(self, batch: Batch) -> None:
        """Pad a decode batch with dummy requests to the smallest captured bucket it fits."""
        bucket = min((bs for bs in self.graphs if bs >= batch.size), default=None)
        if batch.is_decode and bucket is not None:
            padded_size = bucket
        else:
            padded_size = batch.size
        batch.padded_reqs = batch.reqs + [self.dummy_req] * (padded_size - batch.size)


__all__ = ["GraphRunner", "determine_graph_bs"]
