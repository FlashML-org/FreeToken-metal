"""MoE slot lookup: the bank slot of each routed expert of one layer, on the GPU.

The offload cache keeps routed experts in bank slots that slot_of maps. The lookup takes the
distinct experts of the real rows (moe_step's first entry); when some are absent it posts them to
the mailbox and waits in the kernel until the CPU has read them into victim slots (or gives up on a
CPU gone silent past the spin limit), so a forward with lookups needs no host synchronization and
replays as a command graph. A probe posts no experts and only checks that the CPU answers.
"""

from __future__ import annotations

from typing import NamedTuple

import mlx.core as mx

from ftmetal.kernel._ext import _C


class MoeLookupOutputs(NamedTuple):
    # int32 [T, top_k]: each pair's slot in its layer's pool; padding rows get a slot the
    # step already uses.
    slots: mx.array
    # int32 [E]: each expert's slot in the pool; unused experts as padding rows.
    slot_map: mx.array
    # The updated shared tables; they alias the arguments, callers rebind them.
    mailbox: mx.array
    last_use: mx.array
    hit_stats: mx.array


def moe_lookup(
    topk_ids: mx.array,
    moe_step: mx.array,
    slot_of: mx.array,
    mailbox: mx.array,
    last_use: mx.array,
    hit_stats: mx.array,
    *,
    layer_id: int,
    slot_base: int,
    kind: _C.MoeLookupKind,
) -> MoeLookupOutputs:
    """Slots of layer_id's routed experts topk_ids int32 [T, top_k] in its pool, whose first
    slot is slot_base.

    moe_step int32 [2] is (real rows, step), steps counting from 1; slot_of int32
    [num_layers * E], mailbox uint32 [MAILBOX_WORDS], last_use uint32 [num_slots] and
    hit_stats uint32 [num_layers, 2] are the offload cache's shared tables.
    """
    slots, slot_map, mailbox, last_use, hit_stats = _C.moe_lookup(
        topk_ids, moe_step, slot_of, mailbox, last_use, hit_stats, layer_id, slot_base, kind
    )
    return MoeLookupOutputs(slots, slot_map, mailbox, last_use, hit_stats)


__all__ = ["MoeLookupOutputs", "moe_lookup"]
