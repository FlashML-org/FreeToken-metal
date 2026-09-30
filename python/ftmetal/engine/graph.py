"""GraphRunner: decode steps replayed from one CommandGraph capture per batch-size bucket.

Port of FreeToken engine/graph.py. Divergences: a graph is a CommandGraph capture
(kernel/command_graph.py) instead of a CUDA graph, so there is no GraphCaptureBuffer:
the step inputs (input_ids, positions, out_loc and the metadata arrays a backend's
prepare_for_replay leaves at the bucket's fixed shapes) are rebound by every replay,
and so are the pools, which the replay takes and returns; the bs ladder ends at
max_running_req, on the ladder or not; the intermediates and temporaries of every
capture are planned into one arena all buckets share, allocated after the captures,
and a temporary a replay reads before writing (poison test) stays pinned instead; each
capture batch is replayed on the poisoned arena and must give its clean replay's
logits, and graphs outgrowing the memory budget's graph reserve fail the startup like
a failed capture.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

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
    """A bucket's capture batch as step inputs, and the logits of a clean replay of it."""

    step_inputs: list[mx.array]
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
        self.max_graph_bs = max(graph_bs_list, default=0)
        self.reserve_bytes = reserve_bytes
        self.graphs: dict[int, _C.Graph] = {}
        self.plans: dict[int, ArenaPlan] = {}
        self.arena: mx.array | None = None
        if self.graph_bs_list:
            self.capture_all(max_seq_len)
        else:
            logger.info("CommandGraph is off: decode runs eagerly")

    def capture_all(self, max_seq_len: int) -> None:
        """Capture, poison-test and plan every bucket on dummy requests, largest first, then
        bind all to one arena and check each replay on it against its clean replay."""
        for backend in self.attn_backends.values():
            backend.init_capture_graph(max_seq_len, self.graph_bs_list)
        started = time.perf_counter()
        checks: dict[int, ReplayCheck] = {}
        for bs in reversed(self.graph_bs_list):
            bucket_started = time.perf_counter()
            batch = self.capture_batch(bs)
            graph = self.capture(batch)
            step_inputs = self.step_inputs(batch)
            check = ReplayCheck(
                step_inputs=step_inputs, logits=self.check_replay(graph, step_inputs)
            )
            pinned = self.poisoned_temporaries(graph, check)
            plan = plan_graph_arena(graph, pinned)
            place_blocks(graph, plan)
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
        graph_bytes = self.graph_bytes()
        if graph_bytes > self.reserve_bytes:
            raise RuntimeError(
                f"CommandGraph: {graph_bytes / MIB:.2f} MiB of arena and pinned buffers exceed "
                f"the {self.reserve_bytes / MIB:.2f} MiB graph reserve of the memory budget"
            )
        self.bind_arena(checks)
        for backend in self.attn_backends.values():
            backend.release_capture()
        logger.info(
            f"CommandGraph: batch sizes {self.graph_bs_list} captured in "
            f"{time.perf_counter() - started:.1f} s; arena and pinned buffers "
            f"{graph_bytes / MIB:.2f} MiB of the {self.reserve_bytes / MIB:.2f} MiB reserve"
        )

    def graph_bytes(self) -> int:
        """What the graphs keep once bound: the arena they share and their pinned buffers."""
        arena_bytes = max(plan.arena_bytes for plan in self.plans.values())
        return arena_bytes + sum(graph.stats.pinned_bytes for graph in self.graphs.values())

    def bind_arena(self, checks: dict[int, ReplayCheck]) -> None:
        """One arena for every graph; each graph's replay on the arena filled with 0xFF must
        give its clean replay's logits."""
        arena = mx.zeros((max(plan.arena_bytes for plan in self.plans.values()),), mx.uint8)
        mx.eval(arena)
        self.arena = arena
        for bs, graph in self.graphs.items():
            _C.bind_arena(graph, arena)
            _C.poison_arena(graph)
            check = checks[bs]
            if not bit_equal(self.check_replay(graph, check.step_inputs), check.logits):
                raise RuntimeError(
                    f"CommandGraph batch size {bs}: a replay on the poisoned arena differs "
                    f"from its clean replay"
                )

    def capture_batch(self, bs: int) -> Batch:
        """A decode batch of bs dummy requests with its capture metadata."""
        batch = Batch(reqs=[self.dummy_req] * bs, phase="decode")
        batch.input_ids = mx.zeros((bs,), dtype=mx.int32)
        batch.positions = mx.full((bs,), self.dummy_req.cached_len, dtype=mx.int32)
        for backend in self.attn_backends.values():
            backend.prepare_metadata(batch)
            backend.prepare_for_capture(batch)
        return batch

    def check_replay(self, graph: _C.Graph, step_inputs: list[mx.array]) -> mx.array:
        """Logits of a replay of a capture batch, from the state reset_capture_state leaves."""
        for backend in self.attn_backends.values():
            backend.reset_capture_state()
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
            logits = self.check_replay(graph, check.step_inputs)
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
        """One real forward of a padded decode batch, its metadata at its bucket's shapes,
        under the recorder."""
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
        return batch.is_decode and batch.size <= self.max_graph_bs

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
        """Pad a decode batch with dummy requests to the smallest bucket it fits."""
        if self.can_replay(batch):
            padded_size = next(bs for bs in self.graph_bs_list if bs >= batch.size)
        else:
            padded_size = batch.size
        batch.padded_reqs = batch.reqs + [self.dummy_req] * (padded_size - batch.size)


__all__ = ["GraphRunner", "determine_graph_bs"]
