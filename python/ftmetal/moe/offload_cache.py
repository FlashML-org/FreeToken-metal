"""OffloadMoeCache: a slot cache of routed experts read on demand from the checkpoint.

Port of FreeToken moe/offload_cache.py (OffloadMoeCache). Divergences: memory is unified,
so there are no host banks: one bank per layout role holds cache_size expert rows, filled
straight from the checkpoint files by _C.ExpertService (in C++: the token-granular LRU
over the slots, the IO threads, the two prefill windows); slot_of and last_use are shared
arrays the C++ side writes in place; ensure_experts takes the forward's step and returns
the slots; prefetch_prefill_layer returns the layer's slot map (resident experts keep
their slots, only the absent are read into the window) instead of copying a whole layer.
Decode and short prefills look their experts up on the GPU (kernel/moe_lookup.py): misses
go to the C++ coordinator through a mailbox instead of FreeToken's per-layer ready/done
flags, with a watchdog (FreeToken cpu_executor.py) failing an unanswered request.
"""

from __future__ import annotations

import mlx.core as mx

from ftmetal.kernel._ext import _C
from ftmetal.kernel.moe_lookup import moe_lookup
from ftmetal.layers.quantization.config import BankSpec
from ftmetal.moe.source import ExpertSource

# The instance NVMe saturates near queue depth 4 (io_bench); 8 threads keep it there while
# others copy out of their staging buffers.
IO_THREADS = 8
# FreeToken's watchdog threshold. Metal let a lookup spin 60 s on the M4 Pro without failing
# its command buffer (timeout_probe), so the threshold is a policy choice, not a limit.
STALL_TIMEOUT_S = 10.0


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
        stall_timeout_s: float = STALL_TIMEOUT_S,
        run_coordinator: bool = True,
    ) -> None:
        """Without run_coordinator nobody answers the lookups (a stalled coordinator, for
        tests); the watchdog fails a request unanswered for stall_timeout_s."""
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
        # The lookups update these in place and return them; lookup rebinds them, and a
        # command graph replay passes and returns them as pools.
        self.last_use = mx.zeros((cache_size,), dtype=mx.uint32)
        self.mailbox = mx.zeros((_C.MAILBOX_WORDS,), dtype=mx.uint32)
        self.hit_stats = mx.zeros((num_layers, 2), dtype=mx.uint32)
        mx.eval(
            list(self.banks.values()), self.slot_of, self.last_use, self.mailbox, self.hit_stats
        )
        self.service = _C.ExpertService(
            num_layers,
            num_experts,
            cache_size,
            self.window_slots,
            self.slot_of,
            self.last_use,
            self.mailbox,
            self.hit_stats,
            list(self.banks.values()),
            source.files,
            source.segment_table(layout),
            IO_THREADS,
            stall_timeout_s,
            run_coordinator,
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

    def lookup(
        self, layer_id: int, topk_ids: mx.array, moe_step: mx.array, kind: _C.MoeLookupKind
    ) -> tuple[mx.array, mx.array]:
        """The slots int32 [T, top_k] and slot map int32 [num_experts] of the layer's routed
        experts topk_ids int32 [T, top_k]; moe_step int32 [2] is (real rows, step)."""
        outputs = moe_lookup(
            topk_ids,
            moe_step,
            self.slot_of,
            self.mailbox,
            self.last_use,
            self.hit_stats,
            layer_id=layer_id,
            kind=kind,
        )
        self.mailbox, self.last_use, self.hit_stats = (
            outputs.mailbox,
            outputs.last_use,
            outputs.hit_stats,
        )
        return outputs.slots, outputs.slot_map

    def arm(self) -> None:
        """Keep the coordinator spinning through the forward about to be submitted."""
        self.service.arm()

    def check_healthy(self) -> None:
        """Raise the first failure of the cache: a read, a broken or unanswered lookup
        request, the shutdown. Once per forward (FreeToken raise_if_unhealthy)."""
        self.service.check_healthy()

    def probe(self) -> None:
        """One lookup round trip through the coordinator; raises if it was not answered."""
        self.arm()
        slots, _ = self.lookup(
            0,
            mx.zeros((1, 1), dtype=mx.int32),
            mx.zeros((2,), dtype=mx.int32),
            _C.MoeLookupKind.PROBE,
        )
        mx.eval(slots, self.mailbox)
        self.check_healthy()

    def shutdown(self) -> None:
        """Answer a waiting lookup with SHUTDOWN, refuse later ones and stop the threads."""
        self.service.shutdown()

    def stats(self) -> _C.ExpertServiceStats:
        return self.service.stats()


__all__ = ["IO_THREADS", "OffloadMoeCache", "STALL_TIMEOUT_S"]
