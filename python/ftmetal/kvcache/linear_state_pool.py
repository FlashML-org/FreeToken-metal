"""Per-request recurrent state (conv + SSM) for GDN layers.

Port of FreeToken kvcache/linear_state_pool.py. Divergences: one array per layer
for the conv states and the recurrent states [num_slots, num_v_heads,
value_head_dim, key_head_dim] (the layouts the GDN kernels read directly); clearing
and copying slots are the in-place row primitives; no extra slot_states and no
runtime rebuild. The conv state of a slot is a ring [kernel, conv_dim] of the last
kernel conv inputs stored at position mod kernel, so the fused decode step can
update it in the same launch that reads it (a shift register would race). A prefill
snapshot for the hybrid radix cache is a whole slot, written into a request's
ping-pong slot at the deepest page boundary inside its chunk. The pool is built from
the LINEAR group's spec and prices itself (FreeToken's engine adds state_pool_bytes to
the KV family's fixed cost instead).
"""

from __future__ import annotations

from typing import Self

import mlx.core as mx

from ftmetal.kernel.inplace import copy_rows, zero_rows
from ftmetal.kvcache.base import BasePool, PoolConfig, PoolCost
from ftmetal.models.config import KVCacheGroupSpec

PADDING_SLOT = 0
# Slots a running request can hold unevictably: its live slot, two ping-pong track slots
# and the committed snapshot its cache handle locks through decode.
SLOTS_PER_RUNNING_REQ = 4
MIN_SNAPSHOT_CACHE_SLOTS = 4


def linear_pool_num_slots(max_running_req: int, linear_state_cache_ratio: float) -> int:
    """The running requests' slots, a cross-request snapshot cache and the padding slot."""
    snapshot_cache = max(MIN_SNAPSHOT_CACHE_SLOTS, int(linear_state_cache_ratio * max_running_req))
    return SLOTS_PER_RUNNING_REQ * max_running_req + snapshot_cache + 1


class LinearStatePool(BasePool):
    def __init__(self, spec: KVCacheGroupSpec, num_slots: int, dtype: mx.Dtype) -> None:
        self._num_slots = num_slots
        self.conv_dtype = dtype
        self.state_dtype = spec.state_dtype
        self.conv_states = [
            mx.zeros((num_slots, *spec.conv_shape), dtype=dtype) for _ in spec.layer_ids
        ]
        self.recurrent_states = [
            mx.zeros((num_slots, *spec.state_shape), dtype=spec.state_dtype) for _ in spec.layer_ids
        ]
        mx.eval(*self.conv_states, *self.recurrent_states)
        self.local_index_map = {layer_id: i for i, layer_id in enumerate(spec.layer_ids)}
        self.padding_slot = PADDING_SLOT
        self.free_slots: list[int] = list(range(1, num_slots))

    @classmethod
    def kv_cost(cls, spec: KVCacheGroupSpec, config: PoolConfig) -> PoolCost:
        num_slots = linear_pool_num_slots(config.max_running_req, config.linear_state_cache_ratio)
        return PoolCost(0, num_slots * spec.slot_bytes(config.dtype))

    @classmethod
    def from_spec(cls, spec: KVCacheGroupSpec, config: PoolConfig, num_pages: int) -> Self:
        # Slots follow the concurrency, not the page count.
        num_slots = linear_pool_num_slots(config.max_running_req, config.linear_state_cache_ratio)
        return cls(spec, num_slots, config.dtype)

    @property
    def num_free_slots(self) -> int:
        return len(self.free_slots)

    def alloc(self, n: int = 1) -> list[int]:
        if n > len(self.free_slots):
            raise RuntimeError(f"LinearStatePool exhausted: need {n}, have {len(self.free_slots)}")
        return [self.free_slots.pop() for _ in range(n)]

    def free(self, slots: list[int]) -> None:
        self.free_slots.extend(slots)

    def clear_slots(self, slots: list[int]) -> None:
        """Zero conv + recurrent state at slots across all linear layers."""
        ids = mx.array(slots, dtype=mx.int32)
        for i in range(len(self.conv_states)):
            self.conv_states[i] = zero_rows(self.conv_states[i], ids)
            self.recurrent_states[i] = zero_rows(self.recurrent_states[i], ids)

    def copy_from(self, src: int, dst: int) -> None:
        """Copy a whole-sequence snapshot (all layers) from slot src to slot dst."""
        src_ids = mx.array([src], dtype=mx.int32)
        dst_ids = mx.array([dst], dtype=mx.int32)
        for i in range(len(self.conv_states)):
            self.conv_states[i] = copy_rows(self.conv_states[i], src_ids, dst_ids)
            self.recurrent_states[i] = copy_rows(self.recurrent_states[i], src_ids, dst_ids)

    def is_linear_layer(self, layer_id: int) -> bool:
        return layer_id in self.local_index_map

    def local_index(self, layer_id: int) -> int:
        return self.local_index_map[layer_id]

    def arrays(self) -> list[mx.array]:
        return [*self.conv_states, *self.recurrent_states]

    def set_arrays(self, arrays: list[mx.array]) -> None:
        count = len(self.conv_states)
        assert len(arrays) == 2 * count
        self.conv_states, self.recurrent_states = arrays[:count], arrays[count:]

    @property
    def num_linear_layers(self) -> int:
        return len(self.local_index_map)

    @property
    def num_slots(self) -> int:
        return self._num_slots

    def bytes_per_slot(self) -> int:
        conv = self.conv_states[0]
        rec = self.recurrent_states[0]
        per_layer = conv.size // conv.shape[0] * conv.dtype.size
        per_layer += rec.size // rec.shape[0] * rec.dtype.size
        return per_layer * len(self.conv_states)


__all__ = [
    "MIN_SNAPSHOT_CACHE_SLOTS",
    "PADDING_SLOT",
    "SLOTS_PER_RUNNING_REQ",
    "LinearStatePool",
    "linear_pool_num_slots",
]
