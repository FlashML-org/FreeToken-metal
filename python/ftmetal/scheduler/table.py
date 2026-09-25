"""Request slots, the host page table and the device token pool.

Port of FreeToken scheduler/table.py. Divergences: the page table is a host numpy
array (the backends slice it into per-step metadata); the token pool is a device
array written only through the in-place row primitive. Row max_running_reqs of the
token pool is scratch for sampled tokens nobody reads.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from ftmetal.kernel.inplace import scatter_rows


class TableManager:
    def __init__(self, max_running_reqs: int, page_table: np.ndarray) -> None:
        self.max_running_reqs = max_running_reqs
        self.free_slots = list(range(max_running_reqs))
        self.page_table = page_table
        self.aligned_max_seq_len = page_table.shape[1]
        self.scratch_row = max_running_reqs
        self.token_pool = mx.zeros(
            ((max_running_reqs + 1) * self.aligned_max_seq_len, 1), dtype=mx.int32
        )
        mx.eval(self.token_pool)

    @property
    def available_size(self) -> int:
        return len(self.free_slots)

    def allocate(self) -> int:
        return self.free_slots.pop()

    def free(self, slot: int) -> None:
        self.free_slots.append(slot)

    def flat_ids(self, table_idx: int, start: int, stop: int) -> np.ndarray:
        base = table_idx * self.aligned_max_seq_len
        return np.arange(base + start, base + stop, dtype=np.int32)

    def write_tokens(self, table_idx: int, start: int, tokens: np.ndarray) -> None:
        """Host tokens into the pool row of table_idx from position start."""
        ids = mx.array(self.flat_ids(table_idx, start, start + len(tokens)))
        rows = mx.array(np.asarray(tokens, dtype=np.int32))[:, None]
        self.token_pool = scatter_rows(self.token_pool, rows, ids)

    def write_sampled(self, next_tokens: mx.array, flat_ids: mx.array) -> None:
        """Lazy sampled tokens [size] into flat pool positions [size]."""
        self.token_pool = scatter_rows(self.token_pool, next_tokens[:, None], flat_ids)

    def gather_tokens(self, flat_ids: mx.array) -> mx.array:
        return self.token_pool[flat_ids, 0]


__all__ = ["TableManager"]
