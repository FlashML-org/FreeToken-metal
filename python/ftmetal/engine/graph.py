"""GraphRunner: decode steps replayed from one CommandGraph capture per batch-size bucket.

Port of FreeToken engine/graph.py. Divergences: a graph is a CommandGraph capture
(kernel/command_graph.py) instead of a CUDA graph, so there is no GraphCaptureBuffer:
the step inputs (input_ids, positions, out_loc and the attention metadata a backend's
prepare_for_replay builds at the bucket's fixed shapes) are rebound by every replay,
and so are the pools, which the replay takes and returns; the bs ladder is capped by
max_running_req; a bucket whose capture fails stays eager (pad_batch does not pad to it).
"""

from __future__ import annotations

import logging
import time

import mlx.core as mx
from mlx.utils import tree_flatten

from ftmetal.attention.base import AttnType, BaseAttnBackend
from ftmetal.attention.full.common import FullAttnMetadata
from ftmetal.attention.metadata import LinearAttnMetadata
from ftmetal.core import Batch, Req
from ftmetal.kernel._ext import _C
from ftmetal.kernel.command_graph import capture, replay
from ftmetal.kvcache.base import BasePool
from ftmetal.models.blocks import BaseLLMModel

logger = logging.getLogger(__name__)

# FreeToken's ladder: 1, 2, 4, then every multiple of 8.
GRAPH_BS_STRIDE = 8


def determine_graph_bs(max_running_req: int) -> list[int]:
    """FreeToken's _determine_cuda_graph_bs with max_running_req as the largest batch."""
    candidates = [1, 2, 4, *range(GRAPH_BS_STRIDE, max_running_req + 1, GRAPH_BS_STRIDE)]
    return [bs for bs in candidates if bs <= max_running_req]


def undonated_copies(kernel_names: list[str]) -> int:
    """Same-dtype vector copies (v_copy<T><T>): an eager step donates their input and skips
    them, a capture holds the input; in Qwen3.5 decode the partial-rotary RoPE of q and k."""
    count = 0
    for name in kernel_names:
        dtypes = name.removeprefix("v_copy")
        half = len(dtypes) // 2
        count += name.startswith("v_copy") and dtypes[:half] == dtypes[half:]
    return count


def pool_arrays(pools: dict[AttnType, BasePool]) -> list[mx.array]:
    return [array for pool in pools.values() for array in pool.arrays()]


def set_pool_arrays(pools: dict[AttnType, BasePool], arrays: list[mx.array]) -> None:
    """Rebind every pool to its part of arrays, in pool_arrays order."""
    offset = 0
    for pool in pools.values():
        count = len(pool.arrays())
        pool.set_arrays(arrays[offset : offset + count])
        offset += count
    assert offset == len(arrays)


class GraphRunner:
    def __init__(
        self,
        model: BaseLLMModel,
        attn_backends: dict[AttnType, BaseAttnBackend],
        pools: dict[AttnType, BasePool],
        dummy_req: Req,
        graph_bs_list: list[int],
        max_seq_len: int,
    ) -> None:
        """An empty graph_bs_list turns replay off; max_seq_len is the page-table row width."""
        self.model = model
        self.attn_backends = attn_backends
        self.pools = pools
        self.dummy_req = dummy_req
        self.graph_bs_list = sorted(graph_bs_list)
        self.graphs: dict[int, _C.Graph] = {}
        if self.graph_bs_list:
            self.capture_all(max_seq_len)
        else:
            logger.info("CommandGraph is off: decode runs eagerly")

    def capture_all(self, max_seq_len: int) -> None:
        """Capture every bucket on a batch of dummy requests, largest first."""
        for backend in self.attn_backends.values():
            backend.init_capture_graph(max_seq_len, self.graph_bs_list)
        started = time.perf_counter()
        for bs in sorted(self.graph_bs_list, reverse=True):
            batch = Batch(reqs=[self.dummy_req] * bs, phase="decode")
            batch.input_ids = mx.zeros((bs,), dtype=mx.int32)
            batch.positions = mx.full((bs,), self.dummy_req.cached_len, dtype=mx.int32)
            for backend in self.attn_backends.values():
                backend.prepare_for_capture(batch)
            try:
                graph = self.capture(batch)
            except (RuntimeError, ValueError) as error:
                logger.warning(f"CommandGraph capture of batch size {bs} failed, eager: {error}")
                # A capture refused before its eval leaves the dummy step's pools lazy.
                mx.eval(*pool_arrays(self.pools))
            else:
                self.graphs[bs] = graph
                stats = graph.stats
                logger.info(
                    f"captured batch size {bs}: {stats.num_dispatches} dispatches "
                    f"({undonated_copies(graph.dispatch_names)} same-dtype vector copies), "
                    f"{len(stats.temporaries)} temporaries, "
                    f"pinned {stats.pinned_bytes / 2**20:.1f} MiB"
                )
        logger.info(
            f"CommandGraph: batch sizes {sorted(self.graphs)} captured in "
            f"{time.perf_counter() - started:.1f} s"
        )

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
            metadata = batch.attn_metadata[attn_type]
            if isinstance(metadata, FullAttnMetadata):
                arrays += [metadata.cu_seqlens_q, metadata.seq_lens, metadata.page_table]
            elif isinstance(metadata, LinearAttnMetadata):
                arrays += [metadata.cu_seqlens, metadata.cache_indices, metadata.track_slots]
            else:
                raise TypeError(f"no step inputs for {type(metadata).__name__}")
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
        """Pad a decode batch with dummy requests to its bucket when that bucket was captured."""
        bucket = next((bs for bs in self.graph_bs_list if bs >= batch.size), None)
        if batch.is_decode and bucket is not None and bucket in self.graphs:
            padded_size = bucket
        else:
            padded_size = batch.size
        batch.padded_reqs = batch.reqs + [self.dummy_req] * (padded_size - batch.size)


__all__ = ["GraphRunner", "determine_graph_bs", "pool_arrays", "set_pool_arrays"]
