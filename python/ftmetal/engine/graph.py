"""GraphRunner: decode steps replayed from one CommandGraph capture per batch-size bucket.

Port of FreeToken engine/graph.py. Divergences: a graph is a CommandGraph capture
(ftmetal.command_graph) instead of a CUDA graph, so there is no GraphCaptureBuffer:
the step inputs (input_ids, positions, out_loc, the [3, bs] mrope positions of a model
serving vision, which join the sequence positions the GDN decode reads, and the metadata
arrays a backend's prepare_for_replay leaves at the bucket's fixed shapes) are rebound by
every replay, and so are the pools, which the replay takes and returns; the bs ladder ends
at max_running_req, on the ladder or not; the arena is the graphs' shared pool, owned by
GraphArena: every capture's intermediates and temporaries are planned into it, it is
allocated after the captures, and each capture batch's replay on it must give its clean
replay's logits; graphs outgrowing the memory budget's graph reserve fail the startup
like a failed capture.
"""

from __future__ import annotations

import logging
import time
from functools import partial

import mlx.core as mx
from mlx.utils import tree_flatten

from ftmetal.attention.base import BaseAttnBackend
from ftmetal.attention.metadata import AttnType
from ftmetal.command_graph import _C, capture, replay
from ftmetal.command_graph.arena import GraphArena
from ftmetal.core import Batch, Req
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
        is_mrope: bool,
    ) -> None:
        """An empty graph_bs_list turns replay off; max_seq_len is the page-table row width;
        the graphs may keep reserve_bytes (their arena and pinned buffers); an mrope model
        takes [3, bs] rope positions as a step input."""
        self.model = model
        self.is_mrope = is_mrope
        self.attn_backends = attn_backends
        self.pools = pools
        self.dummy_req = dummy_req
        self.graph_bs_list = sorted(graph_bs_list)
        self.max_graph_bs = max(graph_bs_list, default=0)
        self.reserve_bytes = reserve_bytes
        self.graphs: dict[int, _C.Graph] = {}
        self.arena = GraphArena()
        if self.graph_bs_list:
            self.capture_all(max_seq_len)
        else:
            logger.info("CommandGraph is off: decode runs eagerly")

    def capture_all(self, max_seq_len: int) -> None:
        """Capture every bucket on dummy requests, largest first, into the arena they share,
        then bind them to it."""
        for backend in self.attn_backends.values():
            backend.init_capture_graph(max_seq_len, self.graph_bs_list)
        started = time.perf_counter()
        for bs in reversed(self.graph_bs_list):
            batch = self.capture_batch(bs)
            graph = self.capture(batch)
            self.arena.add(graph, partial(self.check_replay, graph, self.step_inputs(batch)))
            self.graphs[bs] = graph
        graph_bytes = self.arena.nbytes
        if graph_bytes > self.reserve_bytes:
            raise RuntimeError(
                f"CommandGraph: {graph_bytes / MIB:.2f} MiB of arena and pinned buffers exceed "
                f"the {self.reserve_bytes / MIB:.2f} MiB graph reserve of the memory budget"
            )
        self.arena.bind()
        for backend in self.attn_backends.values():
            backend.release_capture()
        logger.info(
            f"CommandGraph: batch sizes {self.graph_bs_list} captured in "
            f"{time.perf_counter() - started:.1f} s; arena and pinned buffers "
            f"{graph_bytes / MIB:.2f} MiB of the {self.reserve_bytes / MIB:.2f} MiB reserve"
        )

    def capture_batch(self, bs: int) -> Batch:
        """A decode batch of bs dummy requests with its capture metadata."""
        batch = Batch(reqs=[self.dummy_req] * bs, phase="decode")
        batch.input_ids = mx.zeros((bs,), dtype=mx.int32)
        batch.positions = mx.full((bs,), self.dummy_req.cached_len, dtype=mx.int32)
        if self.is_mrope:
            batch.mrope_positions = mx.full((3, bs), self.dummy_req.cached_len, dtype=mx.int32)
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
        """input_ids, positions, out_loc, the mrope positions of an mrope model, then each
        backend's metadata arrays, in capture order."""
        arrays = [batch.input_ids, batch.positions, batch.out_loc]
        if self.is_mrope:
            assert batch.mrope_positions is not None
            arrays.append(batch.mrope_positions)
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
