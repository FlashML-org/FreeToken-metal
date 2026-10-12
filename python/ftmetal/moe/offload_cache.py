"""OffloadMoeCache: a slot cache of routed experts read on demand from the checkpoint.

Port of FreeToken moe/offload_cache.py (OffloadMoeCache). Divergences: memory is unified,
so there are no host banks: one raw byte buffer per bank role holds the cache's expert rows
(each pool's in its layers' layout), filled straight from the checkpoint files by
_C.ExpertService (in C++: a token-granular LRU per pool of slots, the IO threads, the two
prefill buffers), laid out by moe/slot_pools.py (FreeToken's slot pool design, its pools of
one slot width placed smallest first, not in layer order); a layer whose gate and up the
checkpoint stores in two formats gets them in the wider one (align_gate_up); slot_of and
last_use are shared arrays the C++ side writes in place; ensure_experts takes the forward's
step and returns the slots; prefetch_prefill_layer returns where each of the layer's
experts lies (resident experts in their slots, only the absent read into the prefill
buffer) instead of copying a whole layer, which the expert-sorted gather reads at byte
offsets. Decode and short prefills look their experts up on the GPU (kernel/moe_lookup.py):
misses go to the C++ coordinator through a mailbox instead of FreeToken's per-layer
ready/done flags, with a watchdog (FreeToken cpu_executor.py) failing an unanswered
request. Each forward is prepared like an attention backend's metadata: a decode, or a
prefill below prefill_overlap_min_tokens, looks its experts up (a copy-miss prefill: its
misses join the LRU, where FreeToken's prefill streams whole layers); a longer prefill
reads each layer's absent experts into a prefill buffer one layer ahead, ordered by the
host. No slot a step reads may be rewritten within the step, since the GPU caches may keep
its old row: a decode's rows are capped so that each pool holds its layers' experts of the
whole decode at once (FreeToken caps per layer), one step as a graph replay must, and a
copy-miss prefill whose experts may not all fit the pools at once runs as several steps,
each its own command buffer.
"""

from __future__ import annotations

from typing import Literal

import mlx.core as mx
import numpy as np

from ftmetal.core import Batch
from ftmetal.kernel._ext import _C
from ftmetal.kernel.moe_lookup import moe_lookup
from ftmetal.moe.slot_pools import SlotLayout, initial_experts
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
# copy-miss) or reads each layer's absent ones into a prefill buffer ahead of it (a long
# prefill).
MoeForwardMode = Literal["decode", "copy_miss", "overlap"]


class OffloadMoeCache:
    def __init__(
        self,
        *,
        slot_layout: SlotLayout,
        source: ExpertSource,
        prefill_overlap: bool,
        prefill_overlap_min_tokens: int,
        expert_counts: np.ndarray | None = None,
        stall_timeout_s: float = STALL_TIMEOUT_S,
        run_coordinator: bool = True,
    ) -> None:
        """slot_layout from plan_slot_layout, with prefill buffers when prefill_overlap is on,
        then a prefill of prefill_overlap_min_tokens or more reads its layers ahead;
        expert_counts int64 [num_layers, num_experts], the hot-expert table it was planned
        with (None: none), picks the experts the pools start with. Without run_coordinator
        nobody answers the lookups (a stalled coordinator, for tests); the watchdog fails a
        request unanswered for stall_timeout_s."""
        self.slot_layout = slot_layout
        self.layer_pools = slot_layout.layer_pools
        num_layers, num_experts = len(self.layer_pools), slot_layout.num_experts
        assert (source.num_layers, source.num_experts) == (num_layers, num_experts), (
            f"the checkpoint has {source.num_layers} MoE layers of {source.num_experts} experts"
        )
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.expert_counts = expert_counts
        # The pools' slots; dedicated prefill buffers lie beside them.
        self.cache_size = slot_layout.num_slots
        self.prefill_overlap = prefill_overlap
        self.prefill_overlap_min_tokens = prefill_overlap_min_tokens
        pools = slot_layout.pools
        layouts = [pools[pool_id].layout for pool_id in self.layer_pools]
        widths = slot_layout.bank_widths
        self.banks = {
            role: mx.zeros((num_bytes // widths[role], widths[role]), dtype=mx.uint8)
            for role, num_bytes in slot_layout.bank_bytes.items()
        }
        self.slot_of = mx.full((num_layers * num_experts,), -1, dtype=mx.int32)
        # The lookups update these in place and return them; lookup rebinds them, and a
        # command graph replay passes and returns them as pools.
        self.last_use = mx.zeros((self.cache_size,), dtype=mx.uint32)
        self.mailbox = mx.zeros((_C.MAILBOX_WORDS,), dtype=mx.uint32)
        self.hit_stats = mx.zeros((num_layers, 2), dtype=mx.uint32)
        mx.eval(
            list(self.banks.values()), self.slot_of, self.last_use, self.mailbox, self.hit_stats
        )
        # Each pool's rows in its layout, views of the banks.
        self.pool_banks = [
            {
                role: self.banks[role][
                    pool.offsets[role] // widths[role] : (
                        pool.offsets[role] + pool.num_slots * spec.nbytes
                    )
                    // widths[role]
                ]
                .view(spec.dtype)
                .reshape(pool.num_slots, *spec.shape)
                for role, spec in pool.layout.items()
            }
            for pool in pools
        ]
        mx.eval(self.pool_banks)
        # [num_layers, num_banks, 2]: the byte offset of slot 0's row (slot s at offset + s *
        # row) and the row's bytes, and [num_banks] each prefill buffer's bytes.
        self.destinations = np.array(
            [
                [
                    (pool.offsets[role] - pool.first_slot * spec.nbytes, spec.nbytes)
                    for role, spec in pool.layout.items()
                ]
                for pool in (pools[pool_id] for pool_id in self.layer_pools)
            ],
            dtype=np.int64,
        )
        self.buffer_bytes = np.array(list(slot_layout.buffer_bytes.values()), np.int64)
        # Each projection's weight, scales and biases banks, by their columns of destinations.
        self.projection_banks: dict[str, list[int]] = {}
        for index, role in enumerate(self.banks):
            self.projection_banks.setdefault(role.rpartition(".")[0], []).append(index)
        self.service = _C.ExpertService(
            num_layers,
            num_experts,
            slot_layout.buffer_size,
            self.layer_pools,
            slot_layout.pool_bounds,
            [pool.volatile_slots for pool in pools],
            self.slot_of,
            self.last_use,
            self.mailbox,
            self.hit_stats,
            list(self.banks.values()),
            source.files,
            source.segment_table(layouts),
            self.destinations,
            self.buffer_bytes.tolist(),
            IO_THREADS,
            stall_timeout_s,
            run_coordinator,
        )
        # The forward being built (prepare): its mode and real rows, the layers that start a
        # new step, the last step (from 1) and moe_step int32 [2] = (real rows, step), the
        # lookups' input.
        self.mode: MoeForwardMode = "decode"
        self.num_rows = 0
        self.step_starts: list[int] = []
        self.step = 0
        self.moe_step = mx.zeros((2,), dtype=mx.int32)
        # An overlap prefill's expert byte offsets of each prefetched layer, and the MoE output
        # of the layer before, which must land before the next layer is read into its buffer.
        self.prefill_expert_bytes: dict[int, dict[str, mx.array]] = {}
        self.prefill_previous_output: mx.array | None = None

    @property
    def nbytes(self) -> int:
        return sum(bank.nbytes for bank in self.banks.values())

    def layer_banks(self, layer_id: int) -> dict[str, mx.array]:
        """Role -> the [num_slots, ...] rows of the layer's pool, which its lookups' slots
        index."""
        return self.pool_banks[self.layer_pools[layer_id]]

    def prepare(self, batch: Batch) -> None:
        """Choose the forward's mode and start its first step; before the model builds it. A
        prefill too long to look up reads its layers ahead whatever its length."""
        max_tokens = self.slot_layout.max_lookup_tokens
        if batch.is_decode:
            self.begin_lookups("decode", batch.size)
        elif self.prefill_overlap and (
            batch.num_tokens >= self.prefill_overlap_min_tokens
            or (max_tokens is not None and batch.num_tokens > max_tokens)
        ):
            self.mode = "overlap"
        else:
            self.begin_lookups("copy_miss", batch.num_tokens)

    def prepare_for_capture(self) -> None:
        """A decode graph's capture: its rows are dummies, so its lookups post nothing."""
        self.begin_lookups("decode", 0)

    def begin_lookups(self, mode: MoeForwardMode, num_rows: int) -> None:
        if mode == "decode":
            max_rows = self.slot_layout.max_decode_rows
        else:
            max_rows = self.slot_layout.max_lookup_tokens
        if max_rows is not None and num_rows > max_rows:
            raise RuntimeError(
                f"a {mode} forward of {num_rows} rows may route more experts than the cache's "
                f"pools hold; the scheduler caps its rows at {max_rows}"
            )
        self.mode = mode
        self.num_rows = num_rows
        # The decode cap keeps a decode one step.
        self.step_starts = [] if mode == "decode" else self.slot_layout.step_starts(num_rows)
        self.advance_step()

    def advance_step(self) -> None:
        assert self.step < MAX_STEP, "the step counter outgrew moe_step's int32"
        self.step += 1
        self.moe_step = mx.array([self.num_rows, self.step], dtype=mx.int32)

    def lookup_slots(self, layer_id: int, topk_ids: mx.array, hidden_states: mx.array) -> mx.array:
        """The pool slot int32 [T, top_k] of each routed pair topk_ids of a decode or copy-miss
        layer whose input is hidden_states. A layer that starts a new step evaluates the layers
        before first, ending their step with their command buffer."""
        if layer_id in self.step_starts:
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
        the prefill buffers' slots), take the buffers back and start reading layers 0 and 1."""
        mx.synchronize()
        self.begin_prefill()
        self.prefill_previous_output = None
        self.prefill_expert_bytes = {
            layer_id: self.prefetch_prefill_layer(layer_id)
            for layer_id in range(min(2, self.num_layers))
        }

    def expert_bytes(self, layer_id: int) -> dict[str, mx.array]:
        """An overlap prefill layer's experts per projection, once its prefill buffer is read:
        int64 [num_experts, 3], each expert's byte offsets into the projection's weight, scales
        and biases banks."""
        self.wait_prefill_layer(layer_id)
        return self.prefill_expert_bytes.pop(layer_id)

    def end_prefill_layer(self, layer_id: int, output: mx.array) -> None:
        """After an overlap prefill layer's MoE output is submitted: wait for the layer
        before's, whose prefill buffer the next layer is read into, then start that read."""
        next_layer = layer_id + 1
        if next_layer == self.num_layers:
            self.prefill_previous_output = None
        elif next_layer in self.prefill_expert_bytes:
            self.prefill_previous_output = output
        else:
            assert self.prefill_previous_output is not None, "layer 0 prefetches layer 1"
            mx.eval(self.prefill_previous_output)
            self.prefill_expert_bytes[next_layer] = self.prefetch_prefill_layer(next_layer)
            self.prefill_previous_output = output

    def fill_initial(self) -> None:
        """Fill every pool with its layers' most routed experts (the lowest-numbered without a
        hot-expert table)."""
        self.service.fill_initial(initial_experts(self.slot_layout, self.expert_counts))

    def ensure_experts(self, layer_id: int, expert_ids: list[int], step: int) -> list[int]:
        """The slot of each of the layer's distinct experts at step (from 1); the misses are
        read before it returns."""
        return self.service.ensure_experts(layer_id, expert_ids, step)

    def begin_prefill(self) -> None:
        self.service.begin_prefill()

    def prefetch_prefill_layer(self, layer_id: int) -> dict[str, mx.array]:
        """Start reading the layer's absent experts into its prefill buffer; returns each
        expert's byte offsets per projection (expert_bytes), in bank order."""
        slot_map = np.array(self.service.prefetch_prefill_layer(layer_id), dtype=np.int64)
        offsets, rows = self.destinations[layer_id, :, 0], self.destinations[layer_id, :, 1]
        buffer = (layer_id % 2) * self.buffer_bytes
        slots = slot_map[:, None]
        # A resident expert's slot, else -1 - its row in the buffer.
        expert_bytes = np.where(slots >= 0, offsets + slots * rows, buffer + (-1 - slots) * rows)
        return {
            projection: mx.array(expert_bytes[:, columns])
            for projection, columns in self.projection_banks.items()
        }

    def wait_prefill_layer(self, layer_id: int) -> None:
        self.service.wait_prefill_layer(layer_id)

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
        """The slots int32 [T, top_k] and slot map int32 [num_experts] in the layer's pool
        (layer_banks) of its routed experts topk_ids int32 [T, top_k]; moe_step int32 [2] is
        (real rows, step)."""
        outputs = moe_lookup(
            topk_ids,
            moe_step,
            self.slot_of,
            self.mailbox,
            self.last_use,
            self.hit_stats,
            layer_id=layer_id,
            slot_base=self.slot_layout.pools[self.layer_pools[layer_id]].first_slot,
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
]
