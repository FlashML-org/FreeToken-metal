"""OffloadMoeCache: a slot cache of routed experts read on demand from the checkpoint. Port of
FreeToken moe/offload_cache.py (OffloadMoeCache). Divergences: memory is unified, so there are
no host banks: one raw byte buffer per bank role holds the cache's expert rows (each pool's in
its layers' layout), filled straight from the checkpoint files by _C.ExpertService (in C++: a
token-granular LRU per pool of slots, the IO threads, the two prefill buffers), laid out by
moe/slot_pools.py (FreeToken's slot pool design, its pools of one slot width placed smallest
first, not in layer order; the layers' layouts share their banks); slot_of and last_use are
shared arrays the C++ side writes in place; ensure_experts takes the forward's step and returns
the slots; prefetch_prefill_layer returns where each of the layer's experts lies (resident
experts in their slots, only the absent read into the prefill buffer) instead of copying a whole
layer, which the expert-sorted gather reads at byte offsets. Decode and short prefills look
their experts up on the GPU (kernel/moe/moe_lookup.py): misses go to the C++ coordinator through
a mailbox instead of FreeToken's per-layer ready/done flags, with a watchdog (FreeToken
cpu_executor.py) failing an unanswered request. A decode layer may issue its lookup instead
(split_lookup): the misses are posted without the wait, which the kernels reading their rows do,
so the layer's resident pairs and shared expert run during the read. Each forward is prepared
like an attention backend's metadata: a decode, or a prefill below prefill_overlap_min_tokens,
looks its experts up (a copy-miss prefill: its misses join the LRU, where FreeToken's prefill
streams whole layers); a longer prefill reads each layer's absent experts into a prefill buffer
ahead of its routing, the most often routed first, drops the reads its routing does not need
once that is on the host, and starts the next layer's then (PR #3's route-drop). No slot a step
reads may be rewritten within the step, since the GPU caches may keep its old row: a decode's
rows are capped so that each pool holds its layers' experts of the whole decode at once
(FreeToken caps per layer), one step as a graph replay must, and a copy-miss prefill whose
experts may not all fit the pools at once runs as several steps, each its own command buffer.
Below the slots, a victim cache in pageable host memory outside the GPU budget
(victim_cache_bytes, 0: none; FreeToken has none) keeps the experts that leave their slots,
purgeable under memory pressure: a later miss, or an overlap prefill's absent expert, is copied
from it instead of read from the checkpoint. Only where a slot's bytes come from changes; the
lookups and slot_of never see it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, NamedTuple

import mlx.core as mx
import numpy as np

from ftmetal.core import Batch
from ftmetal.env import read_env
from ftmetal.kernel._ext import _C
from ftmetal.kernel.moe.moe_lookup import moe_lookup, moe_lookup_issue
from ftmetal.moe.slot_pools import MoeCachePolicy, SlotLayout, VictimMemoryKind, initial_experts
from ftmetal.moe.source import ExpertSource

# The instance NVMe saturates near queue depth 4 (io_bench); 8 threads keep it there while
# others copy out of their staging buffers.
IO_THREADS = 8
# FreeToken's watchdog threshold. Metal let a lookup spin 60 s on the M4 Pro without failing
# its command buffer (timeout_probe), so the threshold is a policy choice, not a limit; a GPU wait
# gives up on its own some time later (MAILBOX_SPIN_LIMIT), when the CPU is gone.
STALL_TIMEOUT_S = 10.0
# The IO threads' staging buffers, beside the banks.
STAGING_BYTES = IO_THREADS * _C.EXPERT_STAGING_BYTES
# moe_step carries the step as int32.
MAX_STEP = 2**31 - 1
# A pool's spare slots for deferred victim stores (ExpertService): a decode lookup's victim hit
# that evicts a resident takes a spare, and the evicted expert is stored once the lookup is
# answered, so the GPU waits for one copy instead of two (PR #3's 4). Only in pools far larger
# than one step's experts: the scheduler's cap on a step's rows counts every slot of a pool.
DEFERRED_STORE_SPARES = 4
DEFERRED_STORE_MIN_POOL_SLOTS = 4096
# --moe-cache-policy lrfu's half-life in steps (PR #3's): an expert a conversation keeps coming
# back to outlasts one-off ones; longer ones lost long replies there, whose new working set they
# evicted while earlier turns' frequent experts held their slots.
LRFU_HALF_LIFE = 20.0
VICTIM_MEMORY = {
    "purgeable": _C.VictimMemory.PURGEABLE,
    "reserved": _C.VictimMemory.ANONYMOUS,
}
# An overlap prefill reads each layer's absent experts ahead of its routing, the most often
# routed first: each layer's routed rate, decayed by this at each of its prefill routings (PR #3's
# 0.75), orders them, so that the reads its routing then drops (route_prefill_layer) are the
# queue's tail.
PREFILL_ROUTED_DECAY = 0.75
# With a prediction of its routing (the next layer's router on this layer's MoE input:
# set_routers), a layer reads ahead only its absent predicted experts, the predicted at least this
# many (PR #3's 48); the rest wait for its routing, so that the reads it drops never begin.
PREFILL_READS_AHEAD = 48
# An overlap prefill of fewer tokens computes each layer's routed experts that lie in slots while
# its absent ones are read (FTMETAL_MOE_RESIDENT_FIRST_TOKENS, 0: never; PR #3's 3072): less GPU
# work is left after the reads, so the next layer's routing lands sooner and drops more of its
# reads. A longer one's MoE outlasts its reads anyway. Only when each pair runs changes.
RESIDENT_FIRST_TOKENS = 3072


class ResidentFirst(NamedTuple):
    """An overlap prefill layer's MoE in two passes (MoEMethod.apply_sorted_first): the experts
    in order, its routed ones in slots first (num_experts of them, num_rows of its pairs)."""

    order: np.ndarray
    num_experts: int
    num_rows: int


# What a forward does with the routed experts: looks them up (a decode, or a short prefill:
# copy-miss) or reads each layer's absent ones into a prefill buffer ahead of it (a long
# prefill).
MoeForwardMode = Literal["decode", "copy_miss", "overlap"]


class IssuedLookup(NamedTuple):
    """A decode layer's lookup posted without its wait (moe_lookup_issue), for the kernels that
    wait for it (affine_glu_pending)."""

    # int32 [T, top_k]: each pair's slot in the layer's pool, negative where it is pending.
    slots: mx.array
    issued: mx.array
    slot_of: mx.array
    mailbox: mx.array
    layer_id: int
    slot_base: int


class OffloadMoeCache:
    def __init__(
        self,
        *,
        slot_layout: SlotLayout,
        source: ExpertSource,
        prefill_overlap: bool,
        prefill_overlap_min_tokens: int,
        expert_counts: np.ndarray | None = None,
        victim_cache_bytes: int,
        pairs: list[mx.array] | None,
        split_lookup: bool,
        victim_memory: VictimMemoryKind = "purgeable",
        cache_policy: MoeCachePolicy = "lru",
        ssd_keepalive_us: int = 0,
        deferred_store_spares: list[int] | None = None,
        stall_timeout_s: float = STALL_TIMEOUT_S,
        run_coordinator: bool = True,
    ) -> None:
        """slot_layout from plan_slot_layout, with prefill buffers when prefill_overlap is on,
        then a prefill of prefill_overlap_min_tokens or more reads its layers ahead;
        expert_counts int64 [num_layers, num_experts], the hot-expert table it was planned
        with (None: none), picks the experts the pools start with; victim_cache_bytes is the
        host victim cache's budget, split over the pools by their slots' bytes (0: none); pairs:
        each layer's pair table when the layouts hold pair codes (an expert pack's), else None;
        split_lookup: a decode layer whose method can issue its lookup (issue_lookup) and
        computes its resident pairs while the CPU reads the absent ones. victim_memory: the victim
        cache's memory kind; cache_policy: the slots' eviction order; ssd_keepalive_us: the SSD
        keepalive's read interval (0: none); deferred_store_spares: each pool's spare slots for
        deferred victim stores (None: DEFERRED_STORE_SPARES in each pool of
        DEFERRED_STORE_MIN_POOL_SLOTS or more). Without run_coordinator nobody answers the
        lookups (a stalled coordinator, for tests); the watchdog fails a request unanswered for
        stall_timeout_s."""
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
        self.pairs = pairs
        self.split_lookup = split_lookup
        pools = slot_layout.pools
        # Each layer's first slot, which its lookups' slots count from.
        self.slot_bases = [pools[pool_id].first_slot for pool_id in self.layer_pools]
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
            list(self.banks.values()),
            self.slot_of,
            self.last_use,
            self.mailbox,
            self.hit_stats,
            self.pairs or [],
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
        # Each projection's banks (weight, then any scales and biases: the layout's
        # order, the expert-sorted kernels' expert_bytes columns), by their columns of
        # destinations.
        self.projection_banks: dict[str, list[int]] = {}
        for index, role in enumerate(self.banks):
            self.projection_banks.setdefault(role.rpartition(".")[0], []).append(index)
        segments, locations = source.segment_table(layouts)
        if deferred_store_spares is None:
            deferred_store_spares = [
                DEFERRED_STORE_SPARES if pool.num_slots >= DEFERRED_STORE_MIN_POOL_SLOTS else 0
                for pool in pools
            ]
        self.cache_policy = cache_policy
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
            segments,
            locations,
            self.destinations,
            self.buffer_bytes.tolist(),
            victim_cache_bytes,
            IO_THREADS,
            stall_timeout_s,
            run_coordinator,
            victim_memory=VICTIM_MEMORY[victim_memory],
            pool_spares=deferred_store_spares,
            lrfu_half_life=LRFU_HALF_LIFE if cache_policy == "lrfu" else 0.0,
            ssd_keepalive_us=ssd_keepalive_us,
        )
        # The forward being built (prepare): its mode and real rows, the layers that start a
        # new step, the last step (from 1) and moe_step int32 [2] = (real rows, step), the
        # lookups' input.
        self.mode: MoeForwardMode = "decode"
        self.num_rows = 0
        self.step_starts: list[int] = []
        self.step = 0
        self.moe_step = mx.zeros((2,), dtype=mx.int32)
        # An overlap prefill's expert byte offsets of each prefetched layer, the MoE output of
        # the layer before, which must land before the next layer is read into its buffer, and
        # each layer's routed rate float64 [num_layers, num_experts], its read-ahead order.
        self.prefill_expert_bytes: dict[int, dict[str, mx.array]] = {}
        self.prefill_previous_output: mx.array | None = None
        self.prefill_routed_rate = np.zeros((num_layers, num_experts))
        # The overlap prefill's tokens, each prefetched layer's slot map and the layers whose
        # routing splits their MoE (resident_first).
        self.prefill_tokens = 0
        self.resident_first_tokens = int(
            read_env("MOE_RESIDENT_FIRST_TOKENS") or RESIDENT_FIRST_TOKENS
        )
        self.prefill_slot_maps: dict[int, np.ndarray] = {}
        self.prefill_resident_first: dict[int, ResidentFirst] = {}
        # Each layer's router, predicting its routing from an input (set_routers); none: the
        # read-ahead goes by routed rate alone.
        self.routers: dict[int, Callable[[mx.array], mx.array]] = {}

    @property
    def nbytes(self) -> int:
        return sum(bank.nbytes for bank in self.banks.values())

    def layer_banks(self, layer_id: int) -> dict[str, mx.array]:
        """Role -> the [num_slots, ...] rows of the layer's pool, which its lookups' slots
        index, and its layer_pairs."""
        return {**self.pool_banks[self.layer_pools[layer_id]], **self.layer_pairs(layer_id)}

    def layer_pairs(self, layer_id: int) -> dict[str, mx.array]:
        """pairs -> the layer's pair table [pairs, 2] when its rows hold pair codes, else none."""
        return {} if self.pairs is None else {"pairs": self.pairs[layer_id]}

    def prepare(self, batch: Batch) -> None:
        """Choose the forward's mode and start its first step; before the model builds it. A
        prefill too long to look up reads its layers ahead whatever its length."""
        max_tokens = self.slot_layout.max_lookup_tokens
        if (
            self.cache_policy == "lrfu"
            and not batch.is_decode
            and all(req.cached_len == 0 for req in batch.reqs)
        ):
            # A request that starts with nothing cached (a new conversation): the use counts
            # back to recency.
            self.service.reset_scores()
        if batch.is_decode:
            self.begin_lookups("decode", batch.size)
        elif self.prefill_overlap and (
            batch.num_tokens >= self.prefill_overlap_min_tokens
            or (max_tokens is not None and batch.num_tokens > max_tokens)
        ):
            self.mode = "overlap"
            self.prefill_tokens = batch.num_tokens
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

    def issue_lookup(self, layer_id: int, topk_ids: mx.array) -> IssuedLookup:
        """A decode layer's lookup of its routed pairs topk_ids int32 [T, top_k] that posts the
        absent experts and returns, for the waiters to resolve their pairs."""
        assert self.mode == "decode", f"a {self.mode} forward looks its experts up"
        outputs = moe_lookup_issue(
            topk_ids,
            self.moe_step,
            self.slot_of,
            self.mailbox,
            self.last_use,
            self.hit_stats,
            layer_id=layer_id,
            slot_base=self.slot_bases[layer_id],
        )
        self.mailbox, self.last_use, self.hit_stats = (
            outputs.mailbox,
            outputs.last_use,
            outputs.hit_stats,
        )
        return IssuedLookup(
            outputs.slots,
            outputs.issued,
            self.slot_of,
            outputs.mailbox,
            layer_id,
            self.slot_bases[layer_id],
        )

    def end_issued_lookup(self, mailbox: mx.array) -> None:
        """Take the mailbox as an issued lookup's waits return it: the next post, a later forward's
        too, runs after them."""
        self.mailbox = mailbox

    def set_routers(self, routers: dict[int, Callable[[mx.array], mx.array]]) -> None:
        """Each MoE layer's router: its routing int32 [T, top_k] of an input, which predicts it
        from the layer before's MoE input during an overlap prefill."""
        self.routers = routers

    def begin_prefill_forward(self) -> None:
        """Layer 0 of an overlap prefill: wait for the GPU (the forward before may still read
        the prefill buffers' slots), take the buffers back and start reading layer 0."""
        mx.synchronize()
        self.begin_prefill()
        self.prefill_previous_output = None
        self.prefill_slot_maps = {}
        self.prefill_resident_first = {}
        self.prefill_expert_bytes = {0: self.prefetch_prefill_layer(0)}

    def predict_prefill_routing(self, layer_id: int, hidden_states: mx.array) -> mx.array | None:
        """The next layer's routing as its router predicts it from this overlap prefill layer's
        MoE input hidden_states (lazy), or None without a router or a next layer."""
        router = self.routers.get(layer_id + 1)
        return None if router is None else router(hidden_states)

    def route_prefill_layer(
        self, layer_id: int, topk_ids: mx.array, predicted: mx.array | None = None
    ) -> None:
        """An overlap prefill layer's routing topk_ids int32 [T, top_k], once its router ran:
        drop the reads of its unrouted experts not yet begun and start its routed ones not read
        ahead, then start reading the next layer (predicted: its routing as predicted) into the
        prefill buffer of the layer before, whose MoE output has landed (a layer's routing
        depends on it, so the wait is free)."""
        ids = np.array(topk_ids)
        routed = np.unique(ids)
        self.service.route_prefill_layer(layer_id, routed.tolist())
        resident = routed[self.prefill_slot_maps.pop(layer_id)[routed] >= 0]
        if self.prefill_tokens < self.resident_first_tokens and 0 < resident.size < routed.size:
            order = np.concatenate([resident, np.setdiff1d(np.arange(self.num_experts), resident)])
            num_rows = int(np.isin(ids, resident).sum())
            self.prefill_resident_first[layer_id] = ResidentFirst(order, resident.size, num_rows)
        rate = self.prefill_routed_rate[layer_id]
        rate *= PREFILL_ROUTED_DECAY
        rate[routed] += 1 - PREFILL_ROUTED_DECAY
        next_layer = layer_id + 1
        if next_layer < self.num_layers and next_layer not in self.prefill_expert_bytes:
            if self.prefill_previous_output is not None:
                mx.eval(self.prefill_previous_output)
            counts = (
                None
                if predicted is None
                else np.bincount(np.array(predicted).ravel(), minlength=self.num_experts)
            )
            self.prefill_expert_bytes[next_layer] = self.prefetch_prefill_layer(next_layer, counts)

    def end_prefill_layer(self, layer_id: int, output: mx.array) -> None:
        """An overlap prefill layer's MoE output, submitted: the next layer's routing reads the
        layer after it into this layer's prefill buffer once it lands (the last layer's is not
        kept)."""
        self.prefill_previous_output = output if layer_id + 1 < self.num_layers else None

    def resident_first(self, layer_id: int) -> tuple[ResidentFirst, dict[str, mx.array]] | None:
        """After route_prefill_layer: how an overlap prefill layer splits its MoE to compute its
        experts in slots first, with its expert_bytes ahead of its reads, or None."""
        first = self.prefill_resident_first.pop(layer_id, None)
        return None if first is None else (first, self.prefill_expert_bytes[layer_id])

    def expert_bytes(self, layer_id: int) -> dict[str, mx.array]:
        """An overlap prefill layer's experts per projection, once its routed experts are in its
        prefill buffer: int64 [num_experts, banks], each expert's byte offsets into the
        projection's banks (weight, then any scales and biases)."""
        self.wait_prefill_layer(layer_id)
        return self.prefill_expert_bytes.pop(layer_id)

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

    def prefetch_prefill_layer(
        self, layer_id: int, predicted: np.ndarray | None = None
    ) -> dict[str, mx.array]:
        """Start reading the layer's absent experts into its prefill buffer: with predicted (how
        often its predicted routing names each expert, int64 [num_experts]) the predicted ones,
        the most often named first, else all, the most often routed first; returns each expert's
        byte offsets per projection (expert_bytes), in bank order."""
        rate = self.prefill_routed_rate[layer_id]
        if predicted is None:
            order, reads_ahead = np.argsort(-rate, kind="stable"), -1
        else:
            # The predicted experts lead the order: its head that many long, at least
            # PREFILL_READS_AHEAD, is read ahead (the absent of it).
            order = np.lexsort((-rate, -predicted))
            reads_ahead = max(PREFILL_READS_AHEAD, int(np.count_nonzero(predicted)))
        slot_map = np.array(
            self.service.prefetch_prefill_layer(layer_id, order.tolist(), reads_ahead),
            dtype=np.int64,
        )
        self.prefill_slot_maps[layer_id] = slot_map
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
        """The banks and slot_of, which a graph binds in place and the CPU writes, and any pair
        tables."""
        return [*self.banks.values(), self.slot_of, *(self.pairs or [])]

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
            slot_base=self.slot_bases[layer_id],
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
    "IssuedLookup",
    "MoeForwardMode",
    "OffloadMoeCache",
]
