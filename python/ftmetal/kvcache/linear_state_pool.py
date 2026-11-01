"""Per-request recurrent state (conv + SSM) for GDN layers.

Port of FreeToken kvcache/linear_state_pool.py. Divergences: one array per layer for the
conv states and the recurrent states [num_slots, num_v_heads, value_head_dim, key_head_dim]
(the layouts the GDN kernels read directly), and one array per slot state a model declares
(FreeToken's slot_states) [num_slots, *shape]; clearing and copying slots are the in-place
row primitives; no runtime rebuild. The conv state of a slot is a ring [kernel, conv_dim] of
the last kernel conv inputs stored at position mod kernel, so the fused decode step can
update it in the same launch that reads it (a shift register would race). A prefill snapshot
for the hybrid radix cache is a whole slot, written into a request's ping-pong slot at the
deepest stride boundary inside its chunk and at most its prompt's reuse_len. The pool is
built from the LINEAR group's spec and prices itself (FreeToken's engine adds
state_pool_bytes to the KV family's fixed cost instead). The hybrid radix cache's snapshot
cache is gdn_snapshots slots on top of the running requests' (FreeToken's is max(4,
linear_state_cache_ratio x max_running_req)). Naive holds only each running request's live
slot and the padding slot, FreeToken's naive size; the live slot is a free-list slot the
request takes at admission and returns at its finish or abort (SGLang's
HybridReqToTokenPool), where FreeToken's naive keys it by table_idx.
"""

from __future__ import annotations

from typing import Self

import mlx.core as mx

from ftmetal.kernel.inplace import copy_rows, zero_rows
from ftmetal.kvcache.base import BasePool, PoolConfig, PoolCost
from ftmetal.models.config import KVCacheGroupSpec

PADDING_SLOT = 0
# Slots a running request can hold unevictably: its live slot under every cache type; under
# the hybrid radix cache also two ping-pong track slots and the committed snapshot its cache
# handle locks through decode.
LIVE_SLOTS = 1
TRACK_SLOTS = 2
LOCKED_SNAPSHOT_SLOTS = 1
# Snapshots a finished conversation leaves in the tree: its prompt's track and its live state.
CONVERSATION_SNAPSHOTS = 2
# The hybrid radix cache's default snapshot cache, about four conversations. 8 is arbitrary.
DEFAULT_GDN_SNAPSHOTS = 8


def gdn_snapshot_slots(gdn_snapshots: int | None) -> int:
    """The hybrid radix cache's snapshot cache: gdn_snapshots, by default
    DEFAULT_GDN_SNAPSHOTS; fewer than one conversation's snapshots are refused."""
    if gdn_snapshots is None:
        return DEFAULT_GDN_SNAPSHOTS
    elif gdn_snapshots < CONVERSATION_SNAPSHOTS:
        raise ValueError(
            f"--gdn-snapshots {gdn_snapshots} holds fewer than the {CONVERSATION_SNAPSHOTS} "
            f"snapshots of one conversation: pass --gdn-snapshots {CONVERSATION_SNAPSHOTS} or more"
        )
    else:
        return gdn_snapshots


def linear_pool_num_slots(config: PoolConfig) -> int:
    """The running requests' slots and the padding slot; radix (the hybrid radix cache) adds
    its snapshot cache."""
    max_running_req = config.max_running_req
    if config.cache_type == "naive":
        # Enough for the decode graph captures too: they borrow their max_running_req scratch
        # slots at startup and return them before the first admission.
        return LIVE_SLOTS * max_running_req + 1
    else:
        slots_per_req = LIVE_SLOTS + TRACK_SLOTS + LOCKED_SNAPSHOT_SLOTS
        snapshot_cache = gdn_snapshot_slots(config.gdn_snapshots)
        return slots_per_req * max_running_req + snapshot_cache + 1


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
        # A model's own per-slot states by name, zeroed with the slot.
        self.slot_states = {
            state.name: mx.zeros((num_slots, *state.shape), dtype=state.dtype or dtype)
            for state in spec.slot_states
        }
        mx.eval(*self.conv_states, *self.recurrent_states, *self.slot_states.values())
        self.local_index_map = {layer_id: i for i, layer_id in enumerate(spec.layer_ids)}
        self.padding_slot = PADDING_SLOT
        self.free_slots: list[int] = list(range(1, num_slots))

    @classmethod
    def kv_cost(cls, spec: KVCacheGroupSpec, config: PoolConfig) -> PoolCost:
        return PoolCost(0, linear_pool_num_slots(config) * spec.slot_bytes(config.dtype))

    @classmethod
    def from_spec(cls, spec: KVCacheGroupSpec, config: PoolConfig, num_pages: int) -> Self:
        # Slots follow the concurrency, not the page count.
        return cls(spec, linear_pool_num_slots(config), config.dtype)

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
        for name, state in self.slot_states.items():
            self.slot_states[name] = zero_rows(state, ids)

    def copy_from(self, src: int, dst: int) -> None:
        """Copy a whole-sequence snapshot (all layers) from slot src to slot dst."""
        src_ids = mx.array([src], dtype=mx.int32)
        dst_ids = mx.array([dst], dtype=mx.int32)
        for i in range(len(self.conv_states)):
            self.conv_states[i] = copy_rows(self.conv_states[i], src_ids, dst_ids)
            self.recurrent_states[i] = copy_rows(self.recurrent_states[i], src_ids, dst_ids)
        for name, state in self.slot_states.items():
            self.slot_states[name] = copy_rows(state, src_ids, dst_ids)

    def is_linear_layer(self, layer_id: int) -> bool:
        return layer_id in self.local_index_map

    def local_index(self, layer_id: int) -> int:
        return self.local_index_map[layer_id]

    def arrays(self) -> list[mx.array]:
        return [*self.conv_states, *self.recurrent_states, *self.slot_states.values()]

    def set_arrays(self, arrays: list[mx.array]) -> None:
        count = len(self.conv_states)
        assert len(arrays) == 2 * count + len(self.slot_states)
        self.conv_states, self.recurrent_states = arrays[:count], arrays[count : 2 * count]
        self.slot_states = dict(zip(self.slot_states, arrays[2 * count :], strict=True))

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
        extra = sum(state.nbytes // state.shape[0] for state in self.slot_states.values())
        return per_layer * len(self.conv_states) + extra


__all__ = [
    "CONVERSATION_SNAPSHOTS",
    "DEFAULT_GDN_SNAPSHOTS",
    "LIVE_SLOTS",
    "LOCKED_SNAPSHOT_SLOTS",
    "PADDING_SLOT",
    "TRACK_SLOTS",
    "LinearStatePool",
    "gdn_snapshot_slots",
    "linear_pool_num_slots",
]
