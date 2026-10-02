"""Admission of pending prompts into prefill batches, chunked by a token budget.

Port of FreeToken scheduler/prefill.py. Admission matches the prefix cache, locks the
match and reserves the GDN slots (1 live + 2 ping-pong, evicting tree snapshots if short);
a hit carries the snapshot slot to restore from; a continuation chunk inherits the window
slots freed so far. Divergences: no multimodal currency, and no window-slot admission
check or chunk cap (the naive window pool holds every running request's window and one
step's tokens); a miss zeroes its live slot here; the cached part of the token pool row
is not written (only the extended tokens are ever gathered); a continuation chunk
inherits the pending track snapshot of the chunk before it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ftmetal.core import Batch, Req, SamplingParams
from ftmetal.kvcache.hybrid_radix_cache import HybridCacheHandle
from ftmetal.kvcache.radix_cache import RadixCacheHandle, align_down
from ftmetal.scheduler.cache import CacheManager, div_ceil
from ftmetal.scheduler.decode import DecodeManager
from ftmetal.scheduler.table import TableManager

# GDN slots a request takes at admission: the live slot and the two ping-pong track slots.
ADMISSION_LINEAR_SLOTS = 3


@dataclass
class PendingReq:
    uid: int
    input_ids: np.ndarray
    sampling_params: SamplingParams
    chunked_req: ChunkedReq | None = None

    @property
    def input_len(self) -> int:
        return len(self.input_ids)

    @property
    def output_len(self) -> int:
        return self.sampling_params.max_tokens


class ChunkedReq(Req):
    """A prefill chunk that is not the prompt's last: never sampled, never decoded."""

    def alloc_ids_buf(self) -> None:
        self.ids_buf = self.input_ids

    def append_host(self, next_token: int) -> None:
        raise NotImplementedError("ChunkedReq should not be sampled")

    @property
    def can_decode(self) -> bool:
        return False


@dataclass
class AdmittedResources:
    """What a request holds from admission on: its locked match, table row and GDN slots."""

    cache_handle: RadixCacheHandle | HybridCacheHandle | None
    table_idx: int
    linear_slot_idx: int | None
    ping_pong: tuple[int, int] | None
    # The tree snapshot to copy into the live slot before the first chunk.
    restore_src: int | None


@dataclass
class PrefillAdder:
    token_budget: int
    reserved_size: int
    cache_manager: CacheManager
    table_manager: TableManager

    def kv_reservation_size(self, total_len: int, cached_len: int) -> int:
        page_size = self.cache_manager.page_size
        return (div_ceil(total_len, page_size) - div_ceil(cached_len, page_size)) * page_size

    def try_allocate_one(self, pending: PendingReq) -> AdmittedResources | None:
        if self.table_manager.available_size == 0:
            return None
        match = self.cache_manager.match_req(pending.input_ids)
        handle = match.handle
        cached_len = 0 if handle is None else handle.cached_len
        estimated_size = self.kv_reservation_size(
            pending.input_len + pending.output_len, cached_len
        )
        if estimated_size + self.reserved_size > self.cache_manager.available_size:
            return None
        # Locking takes the matched pages out of the evictable budget: check again.
        self.cache_manager.lock(handle)
        if estimated_size + self.reserved_size > self.cache_manager.available_size:
            self.cache_manager.unlock(handle)
            return None
        pool = self.cache_manager.linear_state_pool
        if pool is not None:
            if pool.num_free_slots < ADMISSION_LINEAR_SLOTS:
                self.cache_manager.ensure_mamba_slots(ADMISSION_LINEAR_SLOTS)
            if pool.num_free_slots < ADMISSION_LINEAR_SLOTS:
                self.cache_manager.unlock(handle)
                return None
        table_idx = self.table_manager.allocate()
        if handle is not None:
            self.table_manager.page_table[table_idx, :cached_len] = handle.kv_indices
        linear_slot_idx = ping_pong = None
        if pool is not None:
            linear_slot_idx = pool.alloc(1)[0]
            first, second = pool.alloc(2)
            ping_pong = (first, second)
            if match.mamba_value is None:
                pool.clear_slots([linear_slot_idx])
        return AdmittedResources(handle, table_idx, linear_slot_idx, ping_pong, match.mamba_value)

    def add_one_req(
        self,
        pending: PendingReq,
        cached_len: int,
        resources: AdmittedResources,
        next_track_idx: int,
        last_track_seqlen: int | None,
        swa_evicted_seqlen: int,
    ) -> Req:
        remain_len = pending.input_len - cached_len
        chunk_size = min(self.token_budget, remain_len)
        align = self.cache_manager.prefill_chunk_align
        if align > 1 and 0 < chunk_size < remain_len:
            # An unaligned end only loses this prompt's snapshot boundaries: keep it when the
            # budget left cannot fill one unit rather than stall the request.
            aligned = align_down(cached_len + chunk_size, align) - cached_len
            chunk_size = aligned if aligned > 0 else chunk_size
        is_chunked = chunk_size < remain_len
        self.token_budget -= chunk_size
        self.reserved_size += self.kv_reservation_size(
            pending.input_len + pending.output_len, cached_len
        )
        self.table_manager.write_tokens(
            resources.table_idx,
            cached_len,
            pending.input_ids[cached_len : cached_len + chunk_size],
        )
        req_cls = ChunkedReq if is_chunked else Req
        req = req_cls(
            input_ids=pending.input_ids[: cached_len + chunk_size],
            table_idx=resources.table_idx,
            cached_len=cached_len,
            output_len=pending.output_len,
            uid=pending.uid,
            sampling_params=pending.sampling_params,
            cache_handle=resources.cache_handle,
        )
        req.linear_slot_idx = resources.linear_slot_idx
        req.mamba_ping_pong = resources.ping_pong
        req.mamba_next_track_idx = next_track_idx
        req.mamba_last_track_seqlen = last_track_seqlen
        req.mamba_restore_src = resources.restore_src
        req.swa_evicted_seqlen = swa_evicted_seqlen
        return req

    def try_add_one(self, pending: PendingReq) -> Req | None:
        if self.token_budget <= 0:
            return None
        chunked = pending.chunked_req
        if chunked is not None:
            # A continuation already holds its live state: nothing to restore.
            resources = AdmittedResources(
                cache_handle=chunked.cache_handle,
                table_idx=chunked.table_idx,
                linear_slot_idx=chunked.linear_slot_idx,
                ping_pong=chunked.mamba_ping_pong,
                restore_src=None,
            )
            # An earlier chunk's snapshot stays pending until a deeper one replaces it, so a
            # last chunk too short to track still commits one.
            return self.add_one_req(
                pending,
                chunked.cached_len,
                resources,
                chunked.mamba_next_track_idx,
                chunked.mamba_last_track_seqlen,
                chunked.swa_evicted_seqlen,
            )
        resources = self.try_allocate_one(pending)
        if resources is None:
            return None
        cached_len = 0 if resources.cache_handle is None else resources.cache_handle.cached_len
        return self.add_one_req(
            pending,
            cached_len,
            resources,
            next_track_idx=0,
            last_track_seqlen=None,
            swa_evicted_seqlen=0,
        )


@dataclass
class PrefillManager:
    cache_manager: CacheManager
    table_manager: TableManager
    decode_manager: DecodeManager
    pending_list: list[PendingReq] = field(default_factory=list)

    def add_one_req(self, uid: int, input_ids: np.ndarray, sampling_params: SamplingParams) -> None:
        self.pending_list.append(
            PendingReq(uid, np.asarray(input_ids, dtype=np.int32), sampling_params)
        )

    def schedule_next_batch(self, prefill_budget: int) -> Batch | None:
        if not self.pending_list:
            return None
        adder = PrefillAdder(
            token_budget=prefill_budget,
            reserved_size=self.decode_manager.inflight_tokens,
            cache_manager=self.cache_manager,
            table_manager=self.table_manager,
        )
        reqs: list[Req] = []
        chunked_list: list[PendingReq] = []
        for pending in self.pending_list:
            req = adder.try_add_one(pending)
            if req is None:
                break
            pending.chunked_req = None
            if isinstance(req, ChunkedReq):
                pending.chunked_req = req
                chunked_list.append(pending)
            reqs.append(req)
        if not reqs:
            return None
        self.pending_list = chunked_list + self.pending_list[len(reqs) :]
        return Batch(reqs=reqs, phase="prefill")

    def abort_req(self, uid: int) -> Req | None:
        for i, pending in enumerate(self.pending_list):
            if pending.uid == uid:
                self.pending_list.pop(i)
                return pending.chunked_req
        return None

    @property
    def runnable(self) -> bool:
        return len(self.pending_list) > 0


__all__ = [
    "ADMISSION_LINEAR_SLOTS",
    "AdmittedResources",
    "ChunkedReq",
    "PendingReq",
    "PrefillAdder",
    "PrefillManager",
]
