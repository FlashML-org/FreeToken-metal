"""Admission of pending prompts into prefill batches, chunked by a token budget.

Port of FreeToken scheduler/prefill.py. Divergences: no prefix cache in round 1 (the
match always misses), no multimodal or sliding-window currencies; the GDN live slot
is allocated and zeroed at admission.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ftmetal.core import Batch, Req, SamplingParams
from ftmetal.kvcache.linear_state_pool import LinearStatePool
from ftmetal.scheduler.cache import CacheManager, div_ceil
from ftmetal.scheduler.decode import DecodeManager
from ftmetal.scheduler.table import TableManager


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
class PrefillAdder:
    token_budget: int
    reserved_size: int
    cache_manager: CacheManager
    table_manager: TableManager
    linear_state_pool: LinearStatePool | None

    def kv_reservation_size(self, total_len: int, cached_len: int) -> int:
        page_size = self.cache_manager.page_size
        return (div_ceil(total_len, page_size) - div_ceil(cached_len, page_size)) * page_size

    def try_allocate_one(self, pending: PendingReq) -> tuple[int, int, int | None] | None:
        if self.table_manager.available_size == 0:
            return None
        cached_len = self.cache_manager.match_len(pending.input_ids)
        estimated_size = self.kv_reservation_size(
            pending.input_len + pending.output_len, cached_len
        )
        if estimated_size + self.reserved_size > self.cache_manager.available_size:
            return None
        if self.linear_state_pool is not None and self.linear_state_pool.num_free_slots < 1:
            return None
        table_idx = self.table_manager.allocate()
        linear_slot_idx = None
        if self.linear_state_pool is not None:
            linear_slot_idx = self.linear_state_pool.alloc(1)[0]
            self.linear_state_pool.clear_slots([linear_slot_idx])
        return cached_len, table_idx, linear_slot_idx

    def add_one_req(
        self,
        pending: PendingReq,
        table_idx: int,
        cached_len: int,
        linear_slot_idx: int | None,
    ) -> Req:
        remain_len = pending.input_len - cached_len
        chunk_size = min(self.token_budget, remain_len)
        is_chunked = chunk_size < remain_len
        self.token_budget -= chunk_size
        self.reserved_size += self.kv_reservation_size(
            pending.input_len + pending.output_len, cached_len
        )
        self.table_manager.write_tokens(
            table_idx, cached_len, pending.input_ids[cached_len : cached_len + chunk_size]
        )
        req_cls = ChunkedReq if is_chunked else Req
        req = req_cls(
            input_ids=pending.input_ids[: cached_len + chunk_size],
            table_idx=table_idx,
            cached_len=cached_len,
            output_len=pending.output_len,
            uid=pending.uid,
            sampling_params=pending.sampling_params,
        )
        req.linear_slot_idx = linear_slot_idx
        return req

    def try_add_one(self, pending: PendingReq) -> Req | None:
        if self.token_budget <= 0:
            return None
        chunked = pending.chunked_req
        if chunked is not None:
            return self.add_one_req(
                pending, chunked.table_idx, chunked.cached_len, chunked.linear_slot_idx
            )
        resource = self.try_allocate_one(pending)
        if resource is None:
            return None
        cached_len, table_idx, linear_slot_idx = resource
        return self.add_one_req(pending, table_idx, cached_len, linear_slot_idx)


@dataclass
class PrefillManager:
    cache_manager: CacheManager
    table_manager: TableManager
    decode_manager: DecodeManager
    linear_state_pool: LinearStatePool | None
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
            linear_state_pool=self.linear_state_pool,
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


__all__ = ["ChunkedReq", "PendingReq", "PrefillAdder", "PrefillManager"]
