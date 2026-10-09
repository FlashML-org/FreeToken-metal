"""OffloadMoeCache: a slot cache of routed experts read on demand from the checkpoint.

Port of FreeToken moe/offload_cache.py (OffloadMoeCache). Divergences: memory is unified,
so there are no host banks: one bank per layout role holds cache_size expert rows, filled
straight from the checkpoint files by _C.ExpertService (in C++: the token-granular LRU
over the slots, the IO threads, the two prefill windows); slot_of and last_use are shared
arrays the C++ side writes in place; ensure_experts takes the forward's step and returns
the slots; prefetch_prefill_layer returns the layer's slot map (resident experts keep
their slots, only the absent are read into the window) instead of copying a whole layer.
"""

from __future__ import annotations

import mlx.core as mx

from ftmetal.kernel._ext import _C
from ftmetal.layers.quantization.config import BankSpec
from ftmetal.moe.source import ExpertSource

# The instance NVMe saturates near queue depth 4 (io_bench); 8 threads keep it there while
# others copy out of their staging buffers.
IO_THREADS = 8


class OffloadMoeCache:
    def __init__(
        self,
        *,
        num_layers: int,
        num_experts: int,
        cache_size: int,
        prefill_overlap: bool,
        layout: dict[str, BankSpec],
        source: ExpertSource,
    ) -> None:
        assert not prefill_overlap or cache_size >= 2 * num_experts, (
            "Prefill overlap borrows two full expert-layer windows from the cache, so "
            "cache_size must be at least 2 * num_experts"
        )
        assert cache_size >= num_experts, "cache_size must hold one layer's experts"
        assert (source.num_layers, source.num_experts) == (num_layers, num_experts), (
            f"the checkpoint has {source.num_layers} MoE layers of {source.num_experts} experts"
        )
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.cache_size = cache_size
        self.prefill_overlap = prefill_overlap
        self.window_slots = 2 * num_experts if prefill_overlap else 0
        self.layout = layout
        self.banks = {
            role: mx.zeros((cache_size, *spec.shape), dtype=spec.dtype)
            for role, spec in layout.items()
        }
        self.slot_of = mx.full((num_layers * num_experts,), -1, dtype=mx.int32)
        self.last_use = mx.zeros((cache_size,), dtype=mx.uint32)
        mx.eval(list(self.banks.values()), self.slot_of, self.last_use)
        self.service = _C.ExpertService(
            num_layers,
            num_experts,
            cache_size,
            self.window_slots,
            self.slot_of,
            self.last_use,
            list(self.banks.values()),
            source.files,
            source.segment_table(layout),
            IO_THREADS,
        )

    @property
    def nbytes(self) -> int:
        return sum(bank.nbytes for bank in self.banks.values())

    def fill_initial(self) -> None:
        """Read the lowest-numbered experts of every layer into the slots outside the windows."""
        flat_ids = [
            layer_id * self.num_experts + expert
            for expert in range(self.num_experts)
            for layer_id in range(self.num_layers)
        ]
        self.service.fill_initial(flat_ids[: self.cache_size - self.window_slots])

    def ensure_experts(self, layer_id: int, expert_ids: list[int], step: int) -> list[int]:
        """The slot of each of the layer's distinct experts at step (from 1); the misses are
        read before it returns."""
        return self.service.ensure_experts(layer_id, expert_ids, step)

    def begin_prefill(self) -> None:
        self.service.begin_prefill()

    def prefetch_prefill_layer(self, layer_id: int) -> mx.array:
        """Start reading the layer's absent experts into its window; returns the slot of every
        expert, int32 [num_experts]."""
        return mx.array(self.service.prefetch_prefill_layer(layer_id), dtype=mx.int32)

    def wait_prefill_layer(self, layer_id: int) -> None:
        self.service.wait_prefill_layer(layer_id)

    def bank_views(self) -> dict[str, mx.array]:
        """Layout role -> its [cache_size, ...] bank."""
        return dict(self.banks)

    def stats(self) -> _C.ExpertServiceStats:
        return self.service.stats()


__all__ = ["IO_THREADS", "OffloadMoeCache"]
