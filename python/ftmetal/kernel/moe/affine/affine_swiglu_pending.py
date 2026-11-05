"""affine_swiglu_pending: the offload decode's silu_and_mul of one MoE layer's routed gate_up rows,
waiting inside the kernel for the experts moe_lookup_issue posted.

No FreeToken donor. A decode layer under offload runs moe_lookup_issue (posts the absent experts
and returns), the resident gate_up pass (affine_gather_qmv, or affine_gather_qmv_pair_codes with
skip_pending) and the shared expert while the CPU reads the absent ones, then this kernel: a
resident pair's act is silu_and_mul of its resident gate_up rows, a pending pair's threadgroups
wait for the answer, take its slot as moe_lookup would have (a padding pair and an unanswered
miss the first pair's) and compute its gate_up rows at it. Every gate_up row equals
mx.gather_qmm's (or the pair-code gather's) at the pair's slot and act equals silu_and_mul's, bit
for bit. Gate and up are one bank projection [2I, H], gate's rows first.
"""

from __future__ import annotations

from typing import NamedTuple

import mlx.core as mx

from ftmetal.kernel._ext import _C
from ftmetal.kernel.moe.affine.affine_gather_qmv_pair_codes import (
    GATHER_QMV_FAST_INPUT_STEP,
    QMV_TILE_ROWS,
)

# The act rows of each simdgroup: the intermediate size is a multiple of them.
ACT_ROWS_PER_SIMDGROUP = _C.ACT_ROWS_PER_SIMDGROUP


class AffineScales(NamedTuple):
    """An affine bank's scales and biases [S, N, K / group_size] in the input's dtype."""

    scales: mx.array
    biases: mx.array


class PairCodeScales(NamedTuple):
    """An affine bank's pair-code rows uint8 [S, row bytes] and the layer's pair table
    [2^code_bits, 2] in the input's dtype (affine_gather_qmv_pair_codes)."""

    codes: mx.array
    pairs: mx.array

    @property
    def code_bits(self) -> int:
        return self.pairs.shape[0].bit_length() - 1


ExpertScales = AffineScales | PairCodeScales


class SwigluPendingOutputs(NamedTuple):
    # [T, top_k, 1, I] in x's dtype.
    act: mx.array
    # int32 [T, top_k]: every pair's slot in the layer's pool, as moe_lookup returns them.
    slots: mx.array
    # The mailbox, aliased: the next post takes it, so that it runs after every wait.
    mailbox: mx.array


def affine_swiglu_pending(
    x: mx.array,
    w: mx.array,
    scales: ExpertScales,
    gate_up_resident: mx.array,
    slots: mx.array,
    issued: mx.array,
    slot_of: mx.array,
    mailbox: mx.array,
    after: mx.array,
    *,
    layer_id: int,
    slot_base: int,
    num_experts: int,
    group_size: int,
    bits: int,
) -> SwigluPendingOutputs:
    """silu_and_mul of every routed pair's gate_up rows: x [T, 1, 1, H] through the gate_up bank
    w [S, 2I, H * bits / 32] uint32 and its scales, the resident pairs' rows gate_up_resident
    [T, top_k, 1, 2I], moe_lookup_issue's slots and issued.

    slot_of and mailbox are the offload cache's (the mailbox as moe_lookup_issue returned it);
    after is never read, only evaluated first (the shared expert's output).
    """
    output_size, input_size = w.shape[1], x.shape[-1]
    is_fast = (
        output_size % QMV_TILE_ROWS == 0 and input_size % GATHER_QMV_FAST_INPUT_STEP[bits] == 0
    )
    if isinstance(scales, PairCodeScales):
        scales_or_codes, biases_or_pairs, code_bits = scales.codes, scales.pairs, scales.code_bits
    else:
        scales_or_codes, biases_or_pairs, code_bits = scales.scales, scales.biases, 0
    act, resolved_slots, mailbox = _C.affine_swiglu_pending(
        x,
        w,
        scales_or_codes,
        biases_or_pairs,
        gate_up_resident,
        slots,
        issued,
        slot_of,
        mailbox,
        after,
        layer_id,
        slot_base,
        num_experts,
        group_size,
        bits,
        code_bits,
        is_fast,
    )
    return SwigluPendingOutputs(act, resolved_slots, mailbox)


__all__ = [
    "ACT_ROWS_PER_SIMDGROUP",
    "AffineScales",
    "ExpertScales",
    "PairCodeScales",
    "SwigluPendingOutputs",
    "affine_swiglu_pending",
]
