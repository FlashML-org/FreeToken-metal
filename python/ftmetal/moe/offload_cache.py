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
flags, with a watchdog (FreeToken cpu_executor.py) failing an unanswered request. Each
forward is prepared like an attention backend's metadata: a decode, or a prefill below
prefill_overlap_min_tokens, looks its experts up (a copy-miss prefill: its misses join the
LRU, where FreeToken's prefill streams whole layers); a longer prefill reads each layer's
absent experts into a window one layer ahead, ordered by the host. No slot a step reads may
be rewritten within the step, since the GPU caches may keep its old row: a forward whose
experts may not all fit the cache runs as several steps, each its own command buffer.
"""

from __future__ import annotations

import math
from typing import Literal

import mlx.core as mx

from ftmetal.core import Batch
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
# The IO threads' staging buffers, beside the banks.
STAGING_BYTES = IO_THREADS * _C.EXPERT_STAGING_BYTES
# moe_step carries the step as int32.
MAX_STEP = 2**31 - 1

# What a forward does with the routed experts: looks them up (a decode, or a short prefill:
# copy-miss) or reads each layer's absent ones into a window ahead of it (a long prefill).
MoeForwardMode = Literal["decode", "copy_miss", "overlap"]


def slot_bytes(layout: dict[str, BankSpec]) -> int:
    """One expert's row over every bank of layout."""
    return sum(math.prod(spec.shape) * spec.dtype.size for spec in layout.values())


class OffloadMoeCache:
    def __init__(
        self,
        *,
        num_layers: int,
        num_experts: int,
        top_k: int,
        cache_size: int,
        prefill_overlap: bool,
        prefill_overlap_min_tokens: int,
        layout: dict[str, BankSpec],
        source: ExpertSource,
        stall_timeout_s: float = STALL_TIMEOUT_S,
        run_coordinator: bool = True,
    ) -> None:
        """A prefill of prefill_overlap_min_tokens or more reads its layers ahead when
        prefill_overlap is on. Without run_coordinator nobody answers the lookups (a stalled
        coordinator, for tests); the watchdog fails a request unanswered for stall_timeout_s."""
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
        self.top_k = top_k
        self.cache_size = cache_size
        self.prefill_overlap = prefill_overlap
        self.prefill_overlap_min_tokens = prefill_overlap_min_tokens
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
        # The forward being built (prepare): its mode and real rows, the layers per step, the
        # last step (from 1) and moe_step int32 [2] = (real rows, step), the lookups' input.
        self.mode: MoeForwardMode = "decode"
        self.num_rows = 0
        self.layers_per_step = num_layers
        self.step = 0
        self.moe_step = mx.zeros((2,), dtype=mx.int32)
        # An overlap prefill's slot map of each prefetched layer, and the MoE output of the
        # layer before, which must land before the next layer is read into its window.
        self.prefill_slot_maps: dict[int, mx.array] = {}
        self.prefill_previous_output: mx.array | None = None

    @property
    def nbytes(self) -> int:
        return sum(bank.nbytes for bank in self.banks.values())

    def prepare(self, batch: Batch) -> None:
        """Choose the forward's mode and start its first step; before the model builds it."""
        if batch.is_decode:
            self.begin_lookups("decode", batch.size)
        elif self.prefill_overlap and batch.num_tokens >= self.prefill_overlap_min_tokens:
            self.mode = "overlap"
        else:
            self.begin_lookups("copy_miss", batch.num_tokens)

    def prepare_for_capture(self) -> None:
        """A decode graph's capture: its rows are dummies, so its lookups post nothing."""
        self.begin_lookups("decode", 0)

    def begin_lookups(self, mode: MoeForwardMode, num_rows: int) -> None:
        self.mode = mode
        self.num_rows = num_rows
        self.layers_per_step = self.max_layers_per_step(num_rows)
        self.advance_step()

    def max_layers_per_step(self, num_rows: int) -> int:
        """The layers one step of num_rows real rows may take: a slot the step reads keeps its
        expert until the step ends, so the step's layers must hold their worst case at once."""
        worst_experts = min(self.num_experts, num_rows * self.top_k)
        if worst_experts == 0:
            return self.num_layers
        else:
            return min(self.num_layers, self.cache_size // worst_experts)

    def fits_one_step(self, num_rows: int) -> bool:
        """Whether a decode of num_rows real rows runs as one step, as a graph replay must."""
        return self.max_layers_per_step(num_rows) == self.num_layers

    def advance_step(self) -> None:
        assert self.step < MAX_STEP, "the step counter outgrew moe_step's int32"
        self.step += 1
        self.moe_step = mx.array([self.num_rows, self.step], dtype=mx.int32)

    def lookup_slots(self, layer_id: int, topk_ids: mx.array, hidden_states: mx.array) -> mx.array:
        """The bank slot int32 [T, top_k] of each routed pair topk_ids of a decode or copy-miss
        layer whose input is hidden_states. A layer that starts a new step evaluates the layers
        before first, ending their step with their command buffer."""
        if layer_id > 0 and layer_id % self.layers_per_step == 0:
            mx.eval(hidden_states)
            self.advance_step()
            self.arm()
        if self.mode == "decode":
            kind = _C.MoeLookupKind.DECODE
        else:
            kind = _C.MoeLookupKind.PREFILL_COPY_MISS
        slots, _ = self.lookup(layer_id, topk_ids, self.moe_step, kind)
        return slots

    def begin_prefill_forward(self) -> None:
        """Layer 0 of an overlap prefill: wait for the GPU (the forward before may still read
        the windows), take the windows back and start reading layers 0 and 1."""
        mx.synchronize()
        self.begin_prefill()
        self.prefill_previous_output = None
        self.prefill_slot_maps = {
            layer_id: self.prefetch_prefill_layer(layer_id)
            for layer_id in range(min(2, self.num_layers))
        }

    def prefill_slot_map(self, layer_id: int) -> mx.array:
        """An overlap prefill layer's slot map int32 [num_experts], once its window is read."""
        self.wait_prefill_layer(layer_id)
        return self.prefill_slot_maps.pop(layer_id)

    def end_prefill_layer(self, layer_id: int, output: mx.array) -> None:
        """After an overlap prefill layer's MoE output is submitted: wait for the layer
        before's, whose window the next layer is read into, then start that read."""
        next_layer = layer_id + 1
        if next_layer == self.num_layers:
            self.prefill_previous_output = None
        elif next_layer in self.prefill_slot_maps:
            self.prefill_previous_output = output
        else:
            assert self.prefill_previous_output is not None, "layer 0 prefetches layer 1"
            mx.eval(self.prefill_previous_output)
            self.prefill_slot_maps[next_layer] = self.prefetch_prefill_layer(next_layer)
            self.prefill_previous_output = output

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

    def graph_externals(self) -> list[mx.array]:
        """The banks and slot_of: a graph binds them in place, the CPU writes them."""
        return [*self.banks.values(), self.slot_of]

    def table_arrays(self) -> list[mx.array]:
        """mailbox, last_use and hit_stats: the lookups update them in place, so a graph
        replay takes and returns them like the pools."""
        return [self.mailbox, self.last_use, self.hit_stats]

    def set_table_arrays(self, arrays: list[mx.array]) -> None:
        self.mailbox, self.last_use, self.hit_stats = arrays

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

    def hit_counts(self) -> tuple[int, int]:
        """(experts the lookups found resident, experts they missed) so far."""
        stats = self.stats()
        return sum(stats.lookup_hits), sum(stats.misses)


__all__ = [
    "IO_THREADS",
    "MAX_STEP",
    "STAGING_BYTES",
    "STALL_TIMEOUT_S",
    "MoeForwardMode",
    "OffloadMoeCache",
    "slot_bytes",
]
