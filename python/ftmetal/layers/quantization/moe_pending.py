"""An affine offload decode on its issued lookup: the resident pairs' gate_up and the shared expert
run while the CPU reads the absent experts, which affine_swiglu_pending waits for in the kernel.

No FreeToken donor. A decode layer whose lookup moe_lookup_issue posted without its wait (an
OffloadMoeCache under split_lookup) runs four kernels where AffineMoEMethod.apply_slots and the
shared expert's gated add run six or more: the resident gate_up pass (affine_gather_qmv, or
affine_gather_qmv_pair_codes with skip_pending), the shared expert, affine_swiglu_pending (the
absent pairs' gate_up once read, silu_and_mul for all) and affine_down_combine (down, the routing
weighted sum and the shared expert's gated add). The output equals the other path's bit for bit.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import NamedTuple

import mlx.core as mx

from ftmetal.kernel.moe.affine.affine_down_combine import (
    MAX_TOP_K,
    ROWS_PER_THREADGROUP,
    affine_down_combine,
)
from ftmetal.kernel.moe.affine.affine_gather_qmv import affine_gather_qmv
from ftmetal.kernel.moe.affine.affine_gather_qmv_pair_codes import (
    GATHER_QMV_FAST_INPUT_STEP,
    GROUP_SIZES,
    affine_gather_qmv_pair_codes,
)
from ftmetal.kernel.moe.affine.affine_swiglu_pending import (
    ACT_ROWS_PER_SIMDGROUP,
    AffineScales,
    ExpertScales,
    PairCodeScales,
    affine_swiglu_pending,
)
from ftmetal.layers.quantization.config import AffineScheme
from ftmetal.layers.quantization.moe import AffineMoEMethod, SharedExpertForward
from ftmetal.moe.offload_cache import IssuedLookup


class PendingOutputs(NamedTuple):
    # [T, H]: the routed experts' output, plus the shared expert's gated output when given.
    out: mx.array
    # int32 [T, top_k]: each pair's slot in the layer's pool, as the waiting lookup resolves it.
    slots: mx.array
    # The mailbox as the waits return it: the next post takes it, so that it runs after them.
    mailbox: mx.array


def supports_pending(method: AffineMoEMethod, x: mx.array, topk_weights: mx.array) -> bool:
    """Whether apply_pending runs this forward: silu over one gate_up projection and down, both
    affine in the kernels' bits and group sizes, at shapes they take, the routing weights in x's
    dtype, the banks'."""
    cfg = method.cfg
    return (
        cfg.activation == "silu"
        and list(method.projections) == ["gate_up_proj", "down_proj"]
        and all(
            isinstance(scheme, AffineScheme)
            and scheme.bits in GATHER_QMV_FAST_INPUT_STEP
            and scheme.group_size in GROUP_SIZES
            for _, _, scheme in method.projections.values()
        )
        and cfg.top_k <= MAX_TOP_K
        and cfg.hidden % ROWS_PER_THREADGROUP == 0
        and cfg.intermediate % ACT_ROWS_PER_SIMDGROUP == 0
        and x.dtype == topk_weights.dtype == cfg.dtype
    )


def apply_pending(
    method: AffineMoEMethod,
    banks: Mapping[str, mx.array],
    x: mx.array,
    topk_weights: mx.array,
    lookup: IssuedLookup,
    shared_expert: SharedExpertForward | None,
) -> PendingOutputs:
    """method.apply_slots of x [T, H] over the issued lookup's slots in banks (the layer's pool
    rows and resident arrays), plus the shared expert's gated output when given."""

    def expert_scales(projection: str) -> ExpertScales:
        if f"{projection}.codes" in banks:
            return PairCodeScales(banks[f"{projection}.codes"], banks["pairs"])
        else:
            return AffineScales(banks[f"{projection}.scales"], banks[f"{projection}.biases"])

    gate_up_scheme = method.projections["gate_up_proj"][2]
    down_scheme = method.projections["down_proj"][2]
    assert isinstance(gate_up_scheme, AffineScheme) and isinstance(down_scheme, AffineScheme)
    h = mx.expand_dims(x, (-2, -3))
    gate_up_weight = banks["gate_up_proj.weight"]
    gate_up_scales = expert_scales("gate_up_proj")
    if isinstance(gate_up_scales, PairCodeScales):
        gate_up_resident = affine_gather_qmv_pair_codes(
            h,
            gate_up_weight,
            gate_up_scales.codes,
            gate_up_scales.pairs,
            lookup.slots,
            group_size=gate_up_scheme.group_size,
            bits=gate_up_scheme.bits,
            code_bits=gate_up_scales.code_bits,
            skip_pending=True,
        )
    else:
        gate_up_resident = affine_gather_qmv(
            h,
            gate_up_weight,
            gate_up_scales.scales,
            gate_up_scales.biases,
            lookup.slots,
            group_size=gate_up_scheme.group_size,
            bits=gate_up_scheme.bits,
        )
    if shared_expert is None:
        shared = shared_gate = None
        after = gate_up_resident
    else:
        # note (jason-fxz): MLX encodes the deepest chains first: without these the shared expert
        # would run before the lookup posts, not during the read.
        (shared_x,) = mx.depends([x], [lookup.issued])
        shared, shared_gate = shared_expert(shared_x)
        (after,) = mx.depends([shared], [shared_gate])
    act, slots, mailbox = affine_swiglu_pending(
        h,
        gate_up_weight,
        gate_up_scales,
        gate_up_resident,
        lookup.slots,
        lookup.issued,
        lookup.slot_of,
        lookup.mailbox,
        after,
        layer_id=lookup.layer_id,
        slot_base=lookup.slot_base,
        num_experts=method.cfg.num_experts,
        group_size=gate_up_scheme.group_size,
        bits=gate_up_scheme.bits,
    )
    out = affine_down_combine(
        act,
        banks["down_proj.weight"],
        expert_scales("down_proj"),
        slots,
        topk_weights,
        group_size=down_scheme.group_size,
        bits=down_scheme.bits,
        shared=shared,
        shared_gate=shared_gate,
    )
    return PendingOutputs(out, slots, mailbox)


__all__ = ["PendingOutputs", "apply_pending", "supports_pending"]
