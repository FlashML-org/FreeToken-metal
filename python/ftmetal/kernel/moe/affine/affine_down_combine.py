"""affine_down_combine: the offload decode's down projection of one MoE layer's routed pairs, their
routing-weighted sum and the shared expert's gated add, in one launch.

No FreeToken donor. After affine_swiglu_pending: each pair's act row through the down bank at its
resolved slot, as mx.gather_qmm computes it (affine_slot_qmv.h, on the impl MLX's gather_qmv
picks), then (y.squeeze(-2) * weights[..., None]).sum(axis=-2) in the dtype, its products and
partial sums rounded and ordered as MLX's multiply and column reduction round and order them,
then, given the shared expert's output and gate logit, layers/activation.py's sigmoid_gated_add.
The output equals that sequence of ops' bit for bit. One threadgroup per output rows computes every
pair's rows and combines them, except on pair codes off the fast step whose last part-block
QMV_HALF_REM does not split (kernel/_ext.py's switch off, or a tail over half a block): there
affine_gather_qmv_pair_codes gathers the rows and affine_down_combine_gathered combines them.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx

from ftmetal.kernel._ext import HEADER_DEFINES, metal_kernel_header, metal_source
from ftmetal.kernel.custom import MetalKernel, TemplateValue, metal_kernel
from ftmetal.kernel.moe.affine.affine_gather_qmv_pair_codes import (
    GATHER_QMV_FAST_INPUT_STEP,
    GROUP_SIZES,
    QMV_TILE_ROWS,
    SIMD_WIDTH,
    affine_gather_qmv_pair_codes,
)
from ftmetal.kernel.moe.affine.affine_swiglu_pending import ExpertScales, PairCodeScales

# Output rows per threadgroup; its simdgroups are the token's top_k pairs.
ROWS_PER_THREADGROUP = 4
# MLX sums fewer than 32 rows per column with col_reduce_small, the order the kernel keeps.
MAX_TOP_K = 31
# affine_down_combine_gathered's threads per threadgroup.
GATHERED_THREADS = 256
# affine_slot_qmv.h's half-lane split of the off-fast-step last part-block (FTMETAL_QMV_HALF_REM).
QMV_HALF_REM = bool(HEADER_DEFINES["moe/affine/affine_slot_qmv.h"]["QMV_HALF_REM"])
# MLX's qmv_impl values per lane per block, by bit width (its pack factor).
QMV_VALUES_PER_LANE = {2: 16, 3: 8, 4: 8, 5: 8, 6: 4, 8: 4}


def splits_last_block(intermediate_size: int, group_size: int, bits: int) -> bool:
    """Whether affine_slot_qmv.h's qmv_rows runs its last part-block on both lane halves."""
    values = QMV_VALUES_PER_LANE[bits]
    block = SIMD_WIDTH * values
    tail = intermediate_size - (intermediate_size - 1) // block * block
    return QMV_HALF_REM and (SIMD_WIDTH // 2) % (group_size // values) == 0 and tail <= block // 2


SOURCE = metal_source(Path(__file__).with_name("affine_down_combine.metal"))
GATHERED_SOURCE = metal_source(Path(__file__).with_name("affine_down_combine_gathered.metal"))
HEADER = metal_kernel_header(Path(__file__).with_name("affine_down_combine.h"))
AFFINE_SCALES = (
    "AffineScales<T>{scales + size_t(slot) * N * (K / GROUP_SIZE), "
    "biases + size_t(slot) * N * (K / GROUP_SIZE)}"
)
PAIR_CODE_SCALES = (
    "PairCodeScales<T, CODE_BITS, decltype(pairs)>{"
    "(const device uint32_t*)(codes + size_t(slot) * CODE_ROW_BYTES), pairs}"
)
GATED_ADD = "sigmoid_gated_add(total, shared[size_t(token) * N + row], shared_gate[token])"


def combine_kernel(is_pair_codes: bool, has_shared: bool) -> MetalKernel:
    """The kernel over one scale source, with or without the gated add."""
    scale_names = ["codes", "pairs"] if is_pair_codes else ["scales", "biases"]
    source = SOURCE.replace("{{SCALES}}", PAIR_CODE_SCALES if is_pair_codes else AFFINE_SCALES)
    source = source.replace("{{OUTPUT}}", GATED_ADD if has_shared else "total")
    name = "affine_down_combine" + ("_pair_codes" if is_pair_codes else "")
    return metal_kernel(
        name=name + ("_shared" if has_shared else ""),
        input_names=[
            "act",
            "w",
            *scale_names,
            "slots",
            "weights",
            *(["shared", "shared_gate"] if has_shared else []),
        ],
        output_names=["out"],
        source=source,
        header=HEADER,
    )


# (pair codes, shared expert) -> kernel.
KERNELS = {
    (is_pair_codes, has_shared): combine_kernel(is_pair_codes, has_shared)
    for is_pair_codes in (False, True)
    for has_shared in (False, True)
}
# shared expert -> affine_down_combine_gathered.
GATHERED_KERNELS = {
    has_shared: metal_kernel(
        name="affine_down_combine_gathered" + ("_shared" if has_shared else ""),
        input_names=["y", "weights", *(["shared", "shared_gate"] if has_shared else [])],
        output_names=["out"],
        source=GATHERED_SOURCE.replace("{{OUTPUT}}", GATED_ADD if has_shared else "total"),
        header=HEADER,
    )
    for has_shared in (False, True)
}


def affine_down_combine(
    act: mx.array,
    w: mx.array,
    scales: ExpertScales,
    slots: mx.array,
    weights: mx.array,
    *,
    group_size: int,
    bits: int,
    shared: mx.array | None = None,
    shared_gate: mx.array | None = None,
) -> mx.array:
    """The routed experts' output [T, H] in act's dtype: act [T, top_k, 1, I] through the down
    bank w [S, H, I * bits / 32] uint32 and its scales at slots int32 [T, top_k], summed by
    weights [T, top_k] in act's dtype; plus shared [T, H] gated by sigmoid(shared_gate [T, 1])
    when given (both in act's dtype)."""
    tokens, top_k, _, intermediate_size = act.shape
    num_slots, hidden_size, words = w.shape
    assert act.dtype in (mx.float32, mx.float16, mx.bfloat16), act.dtype
    assert act.shape[2] == 1 and 1 <= top_k <= MAX_TOP_K, act.shape
    assert slots.shape == weights.shape == (tokens, top_k), (slots.shape, weights.shape)
    assert slots.dtype == mx.int32 and weights.dtype == act.dtype, (slots.dtype, weights.dtype)
    assert bits in GATHER_QMV_FAST_INPUT_STEP and group_size in GROUP_SIZES, (bits, group_size)
    assert intermediate_size % group_size == 0, (intermediate_size, group_size)
    assert w.dtype == mx.uint32 and words * 32 == intermediate_size * bits, (w.dtype, w.shape)
    assert hidden_size % ROWS_PER_THREADGROUP == 0, hidden_size
    is_fast = (
        hidden_size % QMV_TILE_ROWS == 0
        and intermediate_size % GATHER_QMV_FAST_INPUT_STEP[bits] == 0
    )
    template: list[tuple[str, TemplateValue]] = [
        ("T", act.dtype),
        ("GROUP_SIZE", group_size),
        ("BITS", bits),
        ("K", intermediate_size),
        ("N", hidden_size),
        ("TOP_K", top_k),
        ("ROWS", ROWS_PER_THREADGROUP),
        ("FAST", is_fast),
    ]
    if isinstance(scales, PairCodeScales):
        assert scales.codes.dtype == mx.uint8 and scales.codes.shape[0] == num_slots
        assert scales.pairs.ndim == 2 and scales.pairs.shape[1] == 2
        assert scales.pairs.dtype == act.dtype, scales.pairs.dtype
        template += [("CODE_BITS", scales.code_bits), ("CODE_ROW_BYTES", scales.codes.shape[1])]
        scale_inputs = [scales.codes, scales.pairs]
    else:
        group_shape = (num_slots, hidden_size, intermediate_size // group_size)
        assert scales.scales.shape == scales.biases.shape == group_shape
        assert scales.scales.dtype == scales.biases.dtype == act.dtype, scales.scales.dtype
        scale_inputs = [scales.scales, scales.biases]
    if shared is None:
        assert shared_gate is None, "a shared gate without a shared expert"
        shared_inputs: list[mx.array] = []
    else:
        assert shared_gate is not None, "a shared expert without its gate"
        assert shared.shape == (tokens, hidden_size) and shared_gate.shape == (tokens, 1)
        assert shared.dtype == shared_gate.dtype == act.dtype, (shared.dtype, shared_gate.dtype)
        shared_inputs = [shared, shared_gate]
    if (
        isinstance(scales, PairCodeScales)
        and not is_fast
        and not splits_last_block(intermediate_size, group_size, bits)
    ):
        # note (jason-fxz): on this impl, its last part-block on half the lanes, one threadgroup
        # over the pairs ran 8-14 % slower than the gather and a combine pass (Qwen3.8's down: 65
        # against 59 us at T 1, 423 against 366 at T 8); split over both halves it matches the
        # gather alone (320 against 313 us at T 8). Plain scales and the fast impl lose nothing.
        y = affine_gather_qmv_pair_codes(
            act,
            w,
            scales.codes,
            scales.pairs,
            slots,
            group_size=group_size,
            bits=bits,
            code_bits=scales.code_bits,
        )
        (out,) = GATHERED_KERNELS[shared is not None](
            inputs=[y, weights, *shared_inputs],
            template=[("T", act.dtype), ("N", hidden_size), ("TOP_K", top_k)],
            grid=(tokens * hidden_size, 1, 1),
            threadgroup=(min(GATHERED_THREADS, tokens * hidden_size), 1, 1),
            output_shapes=[(tokens, hidden_size)],
            output_dtypes=[act.dtype],
        )
        return out
    kernel = KERNELS[(isinstance(scales, PairCodeScales), shared is not None)]
    (out,) = kernel(
        inputs=[act, w, *scale_inputs, slots, weights, *shared_inputs],
        template=template,
        grid=(SIMD_WIDTH * (hidden_size // ROWS_PER_THREADGROUP), top_k, tokens),
        threadgroup=(SIMD_WIDTH, top_k, 1),
        output_shapes=[(tokens, hidden_size)],
        output_dtypes=[act.dtype],
    )
    return out


__all__ = ["affine_down_combine"]
