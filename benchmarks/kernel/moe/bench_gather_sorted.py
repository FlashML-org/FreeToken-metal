"""MoE prefill's sorted expert gather on a large shared expert bank, three ways, per call.

    uv run python -m benchmarks.kernel.moe.bench_gather_sorted [--rows 1024,8192,16384,32768]
        [--banks 5120,10240] [--projections gate_up,down] [--repeats 25]
    uv run python -m benchmarks.kernel.moe.bench_gather_sorted --pair-codes
        [--models qwen3.6,qwen3.8]
        [--tokens 512,2048,4096] [--projections gate_up,down] [--repeats 25]

Qwen3.6-35B-A3B's expert projections in affine 4-bit, group 64, bf16: gate_up (2048 -> 1024)
and down (512 -> 2048). The B rows are (token, expert) pairs with ascending expert ids drawn
uniformly over 256 experts. The variants:
  A  mx.gather_qmm(..., sorted_indices=True) on a resident 256-row bank (MLX's rhs kernel
     from B = 1024 up), what MoELayer runs today;
  B  copy the used experts' rows from their bank slots into a 256-row prefill buffer, then
     A on the buffer: all 256 experts copied (worst) or 128 (half already in the buffer).
     The copy runs three ways, timed alone: mx.take into fresh arrays (always all 256), a
     slice update of a preallocated buffer, and ftmetal's in-place copy_rows into a buffer
     that is the bank's first 256 rows; the totals take the fastest;
  C  gather_qmm_sorted on the large bank at the experts' byte offsets (and on the resident bank);
  E  mx.gather_qmm on the large bank with ascending slot ids (rhs kernel only at B / E >= 4,
     else one matvec per row).
The large bank tiles the resident one; expert e sits at a random slot past the prefill
buffer for B and C, and at an ascending one for E. Each figure is the median ms per call
(the call and its eval) over the repeats, the calls interleaved, with its ratio to A.
exact: C, B and E equal A bit for bit.

--pair-codes instead times gather_qmm_sorted_pair_codes on pair-code rows (--code-bits wide)
against gather_qmm_sorted on the scales and biases they stand for, at a prefill step's tokens
routed to top_k distinct experts each, uniformly: Qwen3.6-35B-A3B (256 experts, top 8, group 64)
and Qwen3.8-Flash-Next (512 experts, top 10, group 32), affine 4-bit bf16, the experts at their
own rows of one bank with random codes into a table of 2^code_bits pairs. Each figure is the
median ms per call over the repeats, the calls interleaved. exact: equal bit for bit.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from collections.abc import Sequence
from typing import NamedTuple, Protocol

import mlx.core as mx
import numpy as np

from ftmetal.kernel.kvcache.pool_rows import copy_rows, scatter_rows
from ftmetal.kernel.moe.gather_sorted import gather_qmm_sorted, gather_qmm_sorted_pair_codes
from ftmetal.kernel.ref import pair_code_bank_ref

NUM_EXPERTS = 256
BITS = 4
GROUP_SIZE = 64
DTYPE = mx.bfloat16
# (input size, output size) of Qwen3.6-35B-A3B's routed expert projections.
PROJECTIONS = {"gate_up": (2048, 1024), "down": (512, 2048)}
WARMUP_CALLS = 3
QUANTIZATION = {"transpose": True, "group_size": GROUP_SIZE, "bits": BITS}
# Experts encoded per pair_code_bank_ref call, which unpacks every code bit as one value.
ENCODE_EXPERTS = 64


class ExpertShapes(NamedTuple):
    num_experts: int
    top_k: int
    group_size: int
    # (input size, output size) of each routed expert projection.
    projections: dict[str, tuple[int, int]]


PAIR_CODE_MODELS = {
    "qwen3.6": ExpertShapes(256, 8, 64, {"gate_up": (2048, 1024), "down": (512, 2048)}),
    "qwen3.8": ExpertShapes(512, 10, 32, {"gate_up": (2560, 1280), "down": (640, 2560)}),
}


class Bank(NamedTuple):
    weight: mx.array
    scales: mx.array
    biases: mx.array


class ProjectionCall(Protocol):
    def __call__(self) -> mx.array: ...


class BufferCopy(Protocol):
    def __call__(self) -> tuple[mx.array, ...]: ...


def random_bank(input_size: int, output_size: int) -> Bank:
    """A resident bank of random 4-bit weights; every bit of the packed words is random."""
    shape = (NUM_EXPERTS, output_size, input_size * BITS // 32)
    high = mx.random.randint(0, 2**16, shape, dtype=mx.uint32)
    low = mx.random.randint(0, 2**16, shape, dtype=mx.uint32)
    group_shape = (NUM_EXPERTS, output_size, input_size // GROUP_SIZE)
    bank = Bank(
        weight=(high << 16) | low,
        scales=(0.02 * mx.random.normal(group_shape)).astype(DTYPE),
        biases=(0.02 * mx.random.normal(group_shape)).astype(DTYPE),
    )
    mx.eval(bank)
    return bank


def large_bank(resident: Bank, bank_size: int, slot_lists: list[mx.array]) -> Bank:
    """bank_size rows tiled from resident, with resident row e also at slots[e] of each list."""
    copies = (bank_size // NUM_EXPERTS, 1, 1)
    large = Bank(*(mx.tile(array, copies) for array in resident))
    for slots in slot_lists:
        large = Bank(*(scatter_rows(rows, array, slots) for rows, array in zip(large, resident)))
    mx.eval(large)
    return large


def median_ms(calls: Sequence[ProjectionCall | BufferCopy], repeats: int) -> list[float]:
    """Median ms per call of each call, the calls interleaved so drift hits them alike."""
    times: list[list[float]] = [[] for _ in calls]
    for repeat in range(WARMUP_CALLS + repeats):
        for call, call_times in zip(calls, times):
            mx.synchronize()
            started = time.perf_counter()
            mx.eval(call())
            if repeat >= WARMUP_CALLS:
                call_times.append(time.perf_counter() - started)
    return [statistics.median(call_times) * 1e3 for call_times in times]


def bench_bank(
    name: str,
    resident: Bank,
    bank_size: int,
    row_counts: list[int],
    repeats: int,
    rng: np.random.Generator,
) -> None:
    """The copy methods once, then one table row per B, against a large bank; freed on return."""
    # Slots past the in-place prefill buffer (rows [0, 256)): a random one per expert for B and C,
    # an ascending one for E.
    chosen = NUM_EXPERTS + rng.choice(bank_size - NUM_EXPERTS, 2 * NUM_EXPERTS, replace=False)
    random_slots = mx.array(chosen[:NUM_EXPERTS].astype(np.int32))
    ascending_slots = mx.array(np.sort(chosen[NUM_EXPERTS:]).astype(np.int32))
    large = large_bank(resident, bank_size, [random_slots, ascending_slots])
    buffer = Bank(*(mx.zeros_like(array) for array in resident))
    buffer_rows = mx.arange(NUM_EXPERTS, dtype=mx.int32)
    half = mx.array(np.sort(rng.permutation(NUM_EXPERTS)[: NUM_EXPERTS // 2]).astype(np.int32))
    mx.eval(buffer, half)

    def take_buffer() -> tuple[mx.array, ...]:
        return tuple(mx.take(array, random_slots, axis=0) for array in large)

    def update_buffer(experts: mx.array) -> tuple[mx.array, ...]:
        source_slots = random_slots[experts]
        for buffer_array, array in zip(buffer, large):
            buffer_array[experts] = mx.take(array, source_slots, axis=0)
        return buffer

    def bank_buffer() -> tuple[mx.array, ...]:
        return tuple(array[:NUM_EXPERTS] for array in large)

    def copy_into_bank(experts: mx.array) -> tuple[mx.array, ...]:
        nonlocal large
        source_slots = random_slots[experts]
        large = Bank(*(copy_rows(array, source_slots, experts) for array in large))
        return bank_buffer()

    # Each returns the filled prefill buffer; mx.take always copies all 256 experts.
    copies: dict[str, BufferCopy] = {
        "take 256": take_buffer,
        "update 256": lambda: update_buffer(buffer_rows),
        "update 128": lambda: update_buffer(half),
        "copy_rows 256": lambda: copy_into_bank(buffer_rows),
        "copy_rows 128": lambda: copy_into_bank(half),
    }
    copy_ms = dict(zip(copies, median_ms(list(copies.values()), repeats)))
    print(
        f"copy {name} E={bank_size}: "
        + ", ".join(f"{method} {ms:.2f} ms" for method, ms in copy_ms.items()),
        flush=True,
    )
    worst_copy = min(["take 256", "update 256", "copy_rows 256"], key=copy_ms.__getitem__)
    half_copy = min(["take 256", "update 128", "copy_rows 128"], key=copy_ms.__getitem__)

    input_size = resident.weight.shape[2] * 32 // BITS
    output_size = resident.weight.shape[1]

    def slot_bytes(slots: mx.array) -> mx.array:
        """Each slot's byte offsets into the weight, scales and biases."""
        row_bytes = mx.array([array[0].nbytes for array in resident], dtype=mx.int64)
        return slots.astype(mx.int64)[:, None] * row_bytes

    random_bytes = slot_bytes(random_slots)
    buffer_bytes = slot_bytes(buffer_rows)
    mx.eval(random_bytes, buffer_bytes)
    for rows in row_counts:
        experts = np.sort(rng.integers(0, NUM_EXPERTS, rows))
        offsets = np.searchsorted(experts, np.arange(NUM_EXPERTS + 1))
        expert_ids = mx.array(experts.astype(np.uint32))
        expert_offsets = mx.array(offsets.astype(np.int32))
        slot_ids = ascending_slots.astype(mx.uint32)[expert_ids]
        x = mx.random.normal((rows, 1, input_size)).astype(DTYPE)
        mx.eval(x, expert_ids, expert_offsets, slot_ids)

        def buffer_gather(arrays: tuple[mx.array, ...]) -> mx.array:
            return mx.gather_qmm(
                x, *arrays, rhs_indices=expert_ids, sorted_indices=True, **QUANTIZATION
            )

        calls: list[ProjectionCall] = [
            lambda: buffer_gather(resident),
            lambda: gather_qmm_sorted(
                x,
                *large,
                expert_offsets,
                random_bytes,
                n=output_size,
                group_size=GROUP_SIZE,
                bits=BITS,
            ),
            lambda: gather_qmm_sorted(
                x,
                *resident,
                expert_offsets,
                buffer_bytes,
                n=output_size,
                group_size=GROUP_SIZE,
                bits=BITS,
            ),
            lambda: mx.gather_qmm(
                x, *large, rhs_indices=slot_ids, sorted_indices=True, **QUANTIZATION
            ),
            lambda: buffer_gather(bank_buffer()),
            lambda: buffer_gather(copies[worst_copy]()),
            lambda: buffer_gather(copies[half_copy]()),
        ]
        reference = calls[0]()
        exact = [bool(mx.array_equal(reference, calls[i]())) for i in (1, 4, 3)]
        a_ms, c_ms, c_resident_ms, e_ms, gather_ms, worst_ms, half_ms = median_ms(calls, repeats)
        e_path = "rhs" if rows // bank_size >= 4 else "qmv"
        print(
            f"| {name} | {rows} | {bank_size} | {a_ms:.2f} "
            f"| {c_ms:.2f} | {c_ms / a_ms:.3f} | {c_resident_ms:.2f} | {c_resident_ms / a_ms:.3f} "
            f"| {copy_ms[worst_copy]:.2f} {worst_copy} | {gather_ms:.2f} "
            f"| {worst_ms:.2f} | {worst_ms / a_ms:.2f} "
            f"| {copy_ms[half_copy]:.2f} {half_copy} | {half_ms:.2f} | {half_ms / a_ms:.2f} "
            f"| {e_ms:.2f} | {e_ms / a_ms:.2f} | {e_path} "
            f"| {'/'.join('y' if is_exact else 'n' for is_exact in exact)} |",
            flush=True,
        )


def bench_pair_codes(
    model: str,
    projection: str,
    token_counts: list[int],
    code_bits: int,
    repeats: int,
    rng: np.random.Generator,
) -> None:
    """One table row per token count, over a bank of the model's experts built once."""
    shapes = PAIR_CODE_MODELS[model]
    num_experts, top_k, group_size = shapes.num_experts, shapes.top_k, shapes.group_size
    input_size, output_size = shapes.projections[projection]
    w_shape = (num_experts, output_size, input_size * BITS // 32)
    w = mx.random.randint(0, 2**16, w_shape, dtype=mx.uint32) << 16
    w = w | mx.random.randint(0, 2**16, w_shape, dtype=mx.uint32)
    num_pairs = 1 << code_bits
    pairs = mx.array(0.02 * rng.standard_normal((num_pairs, 2), dtype=np.float32)).astype(DTYPE)
    num_codes = output_size * input_size // group_size
    parts = []
    for first in range(0, num_experts, ENCODE_EXPERTS):
        count = min(ENCODE_EXPERTS, num_experts - first)
        codes = rng.integers(0, num_pairs, (count, num_codes)).astype(np.uint16)
        parts.append(pair_code_bank_ref(codes, pairs, output_size, code_bits))
        mx.eval(parts[-1])
    rows, scales, biases = (mx.concatenate(list(arrays)) for arrays in zip(*parts))
    mx.eval(w, rows, scales, biases)
    experts = np.arange(num_experts, dtype=np.int64)[:, None]
    affine_bytes = mx.array(experts * np.array([w[0].nbytes, scales[0].nbytes, biases[0].nbytes]))
    pair_codes_bytes = mx.array(experts * np.array([w[0].nbytes, rows[0].nbytes]))
    for tokens in token_counts:
        routed = np.stack([rng.choice(num_experts, top_k, replace=False) for _ in range(tokens)])
        expert_ids = np.sort(routed.reshape(-1))
        offsets = np.searchsorted(expert_ids, np.arange(num_experts + 1))
        expert_offsets = mx.array(offsets.astype(np.int32))
        x = mx.random.normal((expert_ids.size, 1, input_size)).astype(DTYPE)
        mx.eval(expert_offsets, x)
        calls: list[ProjectionCall] = [
            lambda: gather_qmm_sorted(
                x,
                w,
                scales,
                biases,
                expert_offsets,
                affine_bytes,
                n=output_size,
                group_size=group_size,
                bits=BITS,
            ),
            lambda: gather_qmm_sorted_pair_codes(
                x,
                w,
                rows,
                pairs,
                expert_offsets,
                pair_codes_bytes,
                n=output_size,
                group_size=group_size,
                bits=BITS,
                code_bits=code_bits,
            ),
        ]
        is_exact = bool(mx.array_equal(calls[0](), calls[1]()))
        affine_ms, pair_codes_ms = median_ms(calls, repeats)
        print(
            f"| {model} | {projection} | {tokens} | {expert_ids.size} | {num_experts} "
            f"| {affine_ms:.2f} | {pair_codes_ms:.2f} | {pair_codes_ms / affine_ms:.3f} "
            f"| {'y' if is_exact else 'n'} |",
            flush=True,
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", default="1024,8192,16384,32768")
    parser.add_argument("--banks", default="5120,10240")
    parser.add_argument("--projections", default="gate_up,down")
    parser.add_argument("--repeats", type=int, default=25)
    parser.add_argument("--pair-codes", action="store_true")
    parser.add_argument("--code-bits", type=int, default=13)
    parser.add_argument("--models", default="qwen3.6,qwen3.8")
    parser.add_argument("--tokens", default="512,2048,4096")
    args = parser.parse_args()
    if args.pair_codes:
        token_counts = [int(tokens) for tokens in args.tokens.split(",")]
        rng = np.random.default_rng(0)
        mx.random.seed(0)
        print(
            "| model | proj | tokens | B | E | gather_qmm_sorted ms | pair_codes ms "
            "| pair_codes / affine | exact |"
        )
        print("|---|---|---:|---:|---:|---:|---:|---:|---|")
        for model in args.models.split(","):
            for projection in args.projections.split(","):
                bench_pair_codes(model, projection, token_counts, args.code_bits, args.repeats, rng)
                mx.clear_cache()
        return 0
    row_counts = [int(rows) for rows in args.rows.split(",")]
    bank_sizes = [int(bank_size) for bank_size in args.banks.split(",")]
    assert all(bank_size % NUM_EXPERTS == 0 for bank_size in bank_sizes)
    rng = np.random.default_rng(0)
    mx.random.seed(0)
    print(
        "| proj | B | E | A ms | C ms | C/A | C resident ms | C res/A | B copy 256 ms "
        "| B gather ms | B worst ms | B worst/A | B copy 128 ms | B 50% ms | B 50%/A "
        "| E ms | E/A | E path | exact C/B/E |"
    )
    print(
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---|---:|---:|---:|---:|---|---|"
    )
    for name in args.projections.split(","):
        resident = random_bank(*PROJECTIONS[name])
        for bank_size in bank_sizes:
            bench_bank(name, resident, bank_size, row_counts, args.repeats, rng)
            mx.clear_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
