"""Per-request recurrent state (conv + SSM) for GDN layers.

Port of FreeToken kvcache/linear_state_pool.py. Divergences: one array per layer
for the conv states and the recurrent states [num_slots, num_v_heads,
value_head_dim, key_head_dim] (the layouts the GDN kernels read directly); clearing
and copying slots are the in-place row primitives; no extra slot_states and no
runtime rebuild. The conv state of a slot is a ring [kernel, conv_dim] of the last
kernel conv inputs stored at position mod kernel, so the fused decode step can
update it in the same launch that reads it (a shift register would race).
"""

from __future__ import annotations

import mlx.core as mx

from ftmetal.kernel.inplace import copy_rows, zero_rows
from ftmetal.models.config import LinearGatedDeltaGroupConfig

PADDING_SLOT = 0


class LinearStatePool:
    def __init__(
        self,
        group: LinearGatedDeltaGroupConfig,
        num_slots: int,
        dtype: mx.Dtype,
        state_dtype: mx.Dtype = mx.float32,
    ) -> None:
        self.group = group
        self._num_slots = num_slots
        self.conv_dtype = dtype
        self.state_dtype = state_dtype
        conv_dim = 2 * group.num_key_heads * group.key_head_dim
        conv_dim += group.num_value_heads * group.value_head_dim
        self.conv_states = [
            mx.zeros((num_slots, group.conv_kernel_dim, conv_dim), dtype=dtype)
            for _ in group.layer_ids
        ]
        self.recurrent_states = [
            mx.zeros(
                (num_slots, group.num_value_heads, group.value_head_dim, group.key_head_dim),
                dtype=state_dtype,
            )
            for _ in group.layer_ids
        ]
        mx.eval(*self.conv_states, *self.recurrent_states)
        self.local_index_map = {layer_id: i for i, layer_id in enumerate(group.layer_ids)}
        self.padding_slot = PADDING_SLOT
        self.free_slots: list[int] = list(range(1, num_slots))

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


def linear_state_bytes_per_req(
    group: LinearGatedDeltaGroupConfig, dtype: mx.Dtype, state_dtype: mx.Dtype = mx.float32
) -> int:
    conv_dim = 2 * group.num_key_heads * group.key_head_dim
    conv_dim += group.num_value_heads * group.value_head_dim
    conv_bytes = conv_dim * group.conv_kernel_dim * dtype.size
    rec_bytes = group.num_value_heads * group.key_head_dim * group.value_head_dim
    rec_bytes *= state_dtype.size
    return len(group.layer_ids) * (conv_bytes + rec_bytes)


def linear_pool_num_slots(max_running_req: int) -> int:
    """Round 1 (no prefix cache): one live slot per running request plus the padding slot."""
    return max_running_req + 1


__all__ = [
    "PADDING_SLOT",
    "LinearStatePool",
    "linear_pool_num_slots",
    "linear_state_bytes_per_req",
]
