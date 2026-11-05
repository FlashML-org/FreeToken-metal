"""affine_swiglu_pending against the offload decode's ops with a waiting lookup, exactly, and its
wait on the mailbox (common/mailbox.h).

First the kernel alone, after moe_lookup_issue and the resident gate_up pass over a pool of random
affine experts (plain, or pair codes of 13 and 16 bits), with the CPU's answer written into the
mailbox and slot_of by the test, before the kernel runs or while its threadgroups wait: the resolved
slots must be moe_lookup's for the same request (an answered miss its new slot, a padding pair and
an unanswered miss the first pair's, or the pool's first slot) and act must equal silu_and_mul of
mx.gather_qmm's gate_up rows at them, bit for bit. Requests are answered DONE, answered FAILED,
released by the shutdown word, refused by it before the post, given up on past the mailbox's spin
limit (after which the next lookup posts nothing), or not needed (no expert absent, no real row).
Shapes: Qwen3.6's and Qwen3.8-Flash-Next's gate_up (whose hidden 2560 is K) at 1 and 22 rows, the
test models', bits 2-6 and 8, each dtype.

Then the four kernels of a decode MoE layer (moe_lookup_issue, the resident pass,
affine_swiglu_pending, affine_down_combine with the shared expert's gated add) on an offload cache
whose C++ coordinator answers: a chain of layers per step, each layer's ids depending on the layer
before, must give the outputs, slots, slot_of and last_use of a twin cache running the waiting
lookup's path (moe_lookup, AffineMoEMethod.apply_slots, sigmoid_gated_add), eagerly and as a
replayed command graph, in plain and pair-code rows. Last the wait's edges on a cache: a coordinator
in the test that reads the misses into their slots while the waiters spin, a request the watchdog
fails, a shutdown releasing the waiters, a failed read; the unanswered fall back as moe_lookup's do.
And apply_pending's order: a probe in the shared expert's place must run while the request is
posted and unanswered, after the issue and before the waiters.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import mlx.core as mx
import numpy as np
import numpy.typing as npt
import pytest

from ftmetal.command_graph import capture, replay
from ftmetal.kernel._ext import _C
from ftmetal.kernel.custom import metal_kernel
from ftmetal.kernel.moe.affine.affine_gather_qmv import affine_gather_qmv
from ftmetal.kernel.moe.affine.affine_gather_qmv_pair_codes import affine_gather_qmv_pair_codes
from ftmetal.kernel.moe.affine.affine_swiglu_pending import (
    AffineScales,
    ExpertScales,
    PairCodeScales,
    affine_swiglu_pending,
)
from ftmetal.kernel.moe.moe_lookup import moe_lookup_issue
from ftmetal.kernel.ref import pair_code_bank_ref
from ftmetal.layers.activation import sigmoid_gated_add, silu_and_mul
from ftmetal.layers.quantization.config import AffineScheme
from ftmetal.layers.quantization.moe import AffineMoEMethod, MoEConfig, moe_method
from ftmetal.layers.quantization.moe_pending import apply_pending, supports_pending
from ftmetal.moe.offload_cache import OffloadMoeCache
from offload_harness import CheckpointSpec, build_cache

Word = _C.MailboxWord
State = _C.MailboxState
Issued = _C.IssuedWord
Answer = Literal["done", "failed", "shutdown", "refused", "abandoned"]
# The kernel tests look up layer 1 of 2, whose pool starts at slot FIRST_SLOT.
NUM_LAYERS = 2
LAYER_ID = 1
FIRST_SLOT = 7
# A late answer comes this long after the post; the CPU's writes are this far apart, which orders
# them for the GPU as the coordinator's release stores do.
ANSWER_DELAY_S = 0.1
WRITE_GAP_S = 0.005
POST_DEADLINE_S = 10.0
# A spin limit an unanswered wait reaches in milliseconds.
TEST_SPIN_LIMIT = 100_000


def mailbox_words(mailbox: mx.array) -> npt.NDArray[np.uint32]:
    """A view of the evaluated mailbox the GPU and the CPU share."""
    return np.asarray(memoryview(mailbox)).view(np.uint32)


def wait_for_post(mailbox: npt.NDArray[np.uint32], sequence: int) -> None:
    deadline = time.monotonic() + POST_DEADLINE_S
    while not (mailbox[Word.SEQUENCE] == sequence and mailbox[Word.STATE] == State.POSTED):
        assert time.monotonic() < deadline, f"request {sequence} was never posted"
        time.sleep(0.001)


def answer_request(mailbox: npt.NDArray[np.uint32], sequence: int, state: _C.MailboxState) -> None:
    """Answer the posted request as the coordinator does: the state, then the acknowledgement."""
    time.sleep(WRITE_GAP_S)
    mailbox[Word.STATE] = state
    time.sleep(WRITE_GAP_S)
    mailbox[Word.ACKNOWLEDGED] = sequence


@dataclass(frozen=True)
class Case:
    """tokens x top_k pairs over a pool of experts of [2 * intermediate_size, hidden_size] gate_up;
    the first num_real_rows rows are real, num_absent of their experts absent (the first pair's
    when is_first_absent), the request answered as answer, late: while the kernel waits."""

    tokens: int
    top_k: int
    hidden_size: int
    intermediate_size: int
    group_size: int = 64
    bits: int = 4
    dtype: mx.Dtype = mx.bfloat16
    # None: plain scales and biases.
    code_bits: int | None = None
    num_experts: int = 256
    num_real_rows: int | None = None
    num_absent: int = 3
    is_first_absent: bool = False
    answer: Answer = "done"
    is_late: bool = False

    def __str__(self) -> str:
        dtype = {mx.bfloat16: "bf16", mx.float16: "fp16", mx.float32: "fp32"}[self.dtype]
        scales = "plain" if self.code_bits is None else f"c{self.code_bits}"
        real = self.tokens if self.num_real_rows is None else self.num_real_rows
        return (
            f"T{self.tokens}r{real}-k{self.top_k}-H{self.hidden_size}-I{self.intermediate_size}"
            f"-g{self.group_size}-b{self.bits}-{dtype}-{scales}-E{self.num_experts}"
            f"-a{self.num_absent}{'f' if self.is_first_absent else ''}-{self.answer}"
            f"{'-late' if self.is_late else ''}"
        )


CASES = [
    # Qwen3.6 (hidden 2048, width 512, group 64, top 8): one decode row, the decode cap with
    # padding rows, an absent first pair, an answer while the kernel waits, pair codes.
    Case(1, 8, 2048, 512),
    Case(22, 8, 2048, 512, num_real_rows=19, num_absent=6, is_late=True),
    Case(4, 8, 2048, 512, num_real_rows=3, is_first_absent=True, code_bits=13),
    Case(3, 8, 2048, 512, dtype=mx.float16, is_first_absent=True, is_late=True),
    # Qwen3.8-Flash-Next (hidden 2560, width 640, group 32, top 10, 512 experts).
    Case(1, 10, 2560, 640, group_size=32, num_experts=512, code_bits=13),
    Case(22, 10, 2560, 640, group_size=32, num_experts=512, num_real_rows=20, is_late=True),
    Case(2, 10, 2560, 640, group_size=32, dtype=mx.float32, num_experts=512, code_bits=16),
    # tests/tiny_model.py and tests/offload_harness.py.
    Case(5, 2, 64, 96, group_size=32, dtype=mx.float32, num_experts=8, num_absent=2),
    Case(4, 4, 64, 256, group_size=32, num_experts=8, num_absent=2, code_bits=13),
    Case(3, 2, 128, 64, num_experts=32, num_real_rows=2, is_first_absent=True),
    Case(9, 2, 192, 128, bits=8, dtype=mx.float16, num_experts=32, is_late=True),
    Case(2, 2, 2048, 2048, num_experts=32, num_absent=2),
    # Other bit widths, on the fast impl and the other.
    Case(3, 8, 512, 256, bits=2, group_size=32),
    Case(3, 8, 384, 128, bits=3, dtype=mx.float16),
    Case(3, 8, 512, 256, bits=5, group_size=128, code_bits=13),
    Case(3, 8, 768, 128, bits=6, dtype=mx.float32),
    Case(3, 8, 320, 128, bits=8, group_size=32),
    # Failed, released by a shutdown (also while waiting), refused before the post.
    Case(3, 8, 2048, 512, num_real_rows=2, answer="failed"),
    Case(3, 8, 2048, 512, num_real_rows=2, is_first_absent=True, answer="failed"),
    Case(2, 8, 2048, 512, is_first_absent=True, answer="shutdown"),
    Case(2, 8, 2048, 512, answer="shutdown", is_late=True),
    Case(3, 4, 128, 64, num_experts=32, is_first_absent=True, answer="refused"),
    # Nobody answers, as when the CPU is gone: the waiters give up past the spin limit.
    Case(2, 8, 2048, 512, answer="abandoned"),
    Case(3, 4, 128, 64, num_experts=32, is_first_absent=True, answer="abandoned", code_bits=13),
    # Nothing to wait for: no expert absent, no real row.
    Case(3, 8, 2048, 512, num_real_rows=2, num_absent=0),
    Case(4, 8, 2048, 512, num_real_rows=0, num_absent=0),
]


def gate_up_bank(
    case: Case, num_slots: int, rng: np.random.Generator
) -> tuple[mx.array, ExpertScales, mx.array, mx.array]:
    """The pool's gate_up weight, its scale source, and the scales and biases it stands for."""
    shape = (num_slots, 2 * case.intermediate_size, case.hidden_size)
    weight = mx.array(0.05 * rng.standard_normal(shape, dtype=np.float32)).astype(case.dtype)
    w, scales, biases = mx.quantize(weight, case.group_size, case.bits)
    if case.code_bits is None:
        return w, AffineScales(scales, biases), scales, biases
    else:
        pair_table = 0.02 * rng.standard_normal((1 << case.code_bits, 2), dtype=np.float32)
        pairs = mx.array(pair_table).astype(case.dtype)
        num_codes = shape[1] * case.hidden_size // case.group_size
        codes = rng.integers(0, 1 << case.code_bits, (num_slots, num_codes)).astype(np.uint32)
        rows, scales, biases = pair_code_bank_ref(codes, pairs, shape[1], case.code_bits)
        return w, PairCodeScales(rows, pairs), scales, biases


def resident_gate_up(
    x: mx.array,
    w: mx.array,
    scales: ExpertScales,
    slots: mx.array,
    *,
    group_size: int,
    bits: int,
) -> mx.array:
    """The resident pass: gate_up rows of the pairs whose slot is resident."""
    if isinstance(scales, PairCodeScales):
        return affine_gather_qmv_pair_codes(
            x,
            w,
            scales.codes,
            scales.pairs,
            slots,
            group_size=group_size,
            bits=bits,
            code_bits=scales.code_bits,
            skip_pending=True,
        )
    else:
        return affine_gather_qmv(
            x, w, scales.scales, scales.biases, slots, group_size=group_size, bits=bits
        )


@pytest.mark.parametrize("case", CASES, ids=str)
def test_matches_lookup_and_silu_and_mul(case: Case) -> None:
    rng = np.random.default_rng(case.hidden_size * 13 + case.tokens * 7 + case.num_absent)
    num_real_rows = case.tokens if case.num_real_rows is None else case.num_real_rows
    # Rows draw their experts from a few candidates, so that the pool stays small.
    candidates = rng.choice(case.num_experts, min(case.num_experts, case.top_k + 4), replace=False)
    topk_ids = np.stack(
        [rng.choice(candidates, case.top_k, replace=False) for _ in range(case.tokens)]
    ).astype(np.int32)
    real_experts = sorted({int(expert) for expert in topk_ids[:num_real_rows].reshape(-1)})
    first = int(topk_ids[0, 0])
    others = [expert for expert in real_experts if expert != first]
    absent = list(rng.choice(others, case.num_absent - case.is_first_absent, replace=False))
    absent += [first] if case.is_first_absent else []
    num_slots = len(candidates) + 4
    slot_order = rng.permutation(num_slots).tolist()
    # Each real expert's slot once resident: the resident ones' now, the absent ones' once read.
    final_slot = {expert: slot_order[i] for i, expert in enumerate(real_experts)}
    slot_of = np.full((NUM_LAYERS * case.num_experts,), -1, dtype=np.int32)
    layer_slots = slot_of[LAYER_ID * case.num_experts :]
    for expert, slot in final_slot.items():
        if expert not in absent:
            layer_slots[expert] = FIRST_SLOT + slot
    w, scale_source, scales, biases = gate_up_bank(case, num_slots, rng)
    x_shape = (case.tokens, 1, 1, case.hidden_size)
    x = mx.array(rng.standard_normal(x_shape, dtype=np.float32)).astype(case.dtype)
    shared_slot_of = mx.array(slot_of)
    mailbox = mx.zeros((_C.MAILBOX_WORDS,), dtype=mx.uint32)
    last_use = mx.zeros((FIRST_SLOT + num_slots,), dtype=mx.uint32)
    hit_stats = mx.zeros((NUM_LAYERS, 2), dtype=mx.uint32)
    mx.eval(w, scales, biases, x, shared_slot_of, mailbox, last_use, hit_stats)
    words = mailbox_words(mailbox)
    is_abandoned = case.answer == "abandoned"
    words[Word.SPIN_LIMIT] = TEST_SPIN_LIMIT if is_abandoned else _C.MAILBOX_SPIN_LIMIT
    slot_of_words = np.asarray(memoryview(shared_slot_of)).view(np.int32)
    if case.answer == "refused":
        words[Word.SHUTDOWN] = 1
    is_posted = case.num_absent > 0 and case.answer != "refused"

    def answer() -> None:
        wait_for_post(words, 1)
        if case.is_late:
            time.sleep(ANSWER_DELAY_S)
        # slot_of as the reads leave it, also where the answer does not let the GPU read it.
        for expert in absent:
            slot_of_words[LAYER_ID * case.num_experts + expert] = FIRST_SLOT + final_slot[expert]
        if case.answer == "done":
            answer_request(words, 1, State.DONE)
        elif case.answer == "failed":
            answer_request(words, 1, State.FAILED)
        else:
            time.sleep(WRITE_GAP_S)
            words[Word.SHUTDOWN] = 1

    lookup = moe_lookup_issue(
        mx.array(topk_ids),
        mx.array([num_real_rows, 1], dtype=mx.int32),
        shared_slot_of,
        mailbox,
        last_use,
        hit_stats,
        layer_id=LAYER_ID,
        slot_base=FIRST_SLOT,
    )
    if not case.is_late:
        mx.eval(lookup.slots, lookup.issued, lookup.mailbox)
        if is_posted and not is_abandoned:
            answer()
    gate_up = resident_gate_up(
        x, w, scale_source, lookup.slots, group_size=case.group_size, bits=case.bits
    )
    act, resolved_slots, waited_mailbox = affine_swiglu_pending(
        x,
        w,
        scale_source,
        gate_up,
        lookup.slots,
        lookup.issued,
        shared_slot_of,
        lookup.mailbox,
        x,
        layer_id=LAYER_ID,
        slot_base=FIRST_SLOT,
        num_experts=case.num_experts,
        group_size=case.group_size,
        bits=case.bits,
    )
    if case.is_late:
        cpu = threading.Thread(target=answer)
        started = time.monotonic()
        mx.async_eval(act, resolved_slots, lookup.issued)
        cpu.start()
        mx.eval(act, resolved_slots, lookup.issued)
        assert time.monotonic() - started >= ANSWER_DELAY_S, "the kernel did not wait"
        cpu.join()
    else:
        mx.eval(act, resolved_slots, lookup.issued)

    issued = np.array(lookup.issued)
    expected_issued = (1, len(absent)) if is_posted else (0, 0)
    assert (issued[Issued.SEQUENCE], issued[Issued.NUM_MISSES]) == expected_issued
    is_answered = is_posted and case.answer == "done"
    resolved = {
        expert: slot for expert, slot in final_slot.items() if expert not in absent or is_answered
    }
    fallback = resolved.get(first, 0) if num_real_rows > 0 else 0
    expected_slots = np.array(
        [
            [
                resolved.get(int(expert), fallback) if row < num_real_rows else fallback
                for expert in topk_ids[row]
            ]
            for row in range(case.tokens)
        ],
        dtype=np.int32,
    )
    assert np.array(resolved_slots).tolist() == expected_slots.tolist()
    expected = silu_and_mul(
        mx.gather_qmm(
            x,
            w,
            scales,
            biases,
            rhs_indices=mx.array(expected_slots),
            transpose=True,
            group_size=case.group_size,
            bits=case.bits,
            mode="affine",
        )
    )
    assert act.shape == expected.shape and act.dtype == expected.dtype
    mismatches = (act != expected).sum().item()
    assert mismatches == 0, f"{mismatches} of {act.size} values differ"
    expected_state = {"done": State.DONE, "failed": State.FAILED}.get(case.answer, State.POSTED)
    assert words[Word.STATE] == (expected_state if is_posted else State.IDLE)
    assert words[Word.ABANDONED] == is_abandoned
    if is_abandoned:
        again = moe_lookup_issue(
            mx.array(topk_ids),
            mx.array([num_real_rows, 2], dtype=mx.int32),
            shared_slot_of,
            waited_mailbox,
            lookup.last_use,
            lookup.hit_stats,
            layer_id=LAYER_ID,
            slot_base=FIRST_SLOT,
        )
        mx.eval(again.issued)
        assert np.array(again.issued)[Issued.NUM_MISSES] == 0 and words[Word.SEQUENCE] == 1


@dataclass(frozen=True)
class ChainCase:
    hidden_size: int = 256
    intermediate_size: int = 128
    bits: int = 4
    group_size: int = 64
    top_k: int = 4
    num_experts: int = 32
    is_pair_codes: bool = False
    is_replayed: bool = False

    def __str__(self) -> str:
        return (
            f"H{self.hidden_size}-I{self.intermediate_size}-b{self.bits}-g{self.group_size}"
            f"-k{self.top_k}-E{self.num_experts}"
            f"-{'pair_codes' if self.is_pair_codes else 'plain'}"
            f"-{'replay' if self.is_replayed else 'eager'}"
        )


CHAIN_LAYERS = 3
CHAIN_ROWS = 4
CHAIN_STEPS = 60
# Qwen3.8-Flash-Next's proportions: top 10 of more than 256 experts (posted ids past one byte),
# group 32, gate_up on the fast impl and down off it.
QWEN38_CHAIN = ChainCase(
    hidden_size=512, group_size=32, top_k=10, num_experts=320, is_pair_codes=True
)
CHAIN_CASES = [
    ChainCase(),
    ChainCase(is_pair_codes=True),
    ChainCase(bits=8),
    ChainCase(hidden_size=128, intermediate_size=64),
    ChainCase(is_replayed=True),
    ChainCase(is_pair_codes=True, is_replayed=True),
    QWEN38_CHAIN,
    replace(QWEN38_CHAIN, is_replayed=True),
]


@dataclass(frozen=True)
class LayerInputs:
    """One MoE layer's routing and its shared expert's output and gate logit."""

    topk_ids: mx.array
    weights: mx.array
    shared: mx.array
    shared_gate: mx.array

    def arrays(self) -> list[mx.array]:
        return [self.topk_ids, self.weights, self.shared, self.shared_gate]

    def after(self, out: mx.array | None) -> LayerInputs:
        """These inputs with the ids waiting for out, as a router's wait for the layer before."""
        if out is None:
            return self
        else:
            dependency = mx.isnan(out[:1, :1]).astype(mx.int32) * 0
            return LayerInputs(
                topk_ids=self.topk_ids + dependency,
                weights=self.weights,
                shared=self.shared,
                shared_gate=self.shared_gate,
            )


def split_layer(
    cache: OffloadMoeCache,
    method: AffineMoEMethod,
    layer_id: int,
    x: mx.array,
    inputs: LayerInputs,
    moe_step: mx.array,
) -> tuple[mx.array, mx.array]:
    """The layer's output [T, H] and slots as OffloadMoELayer runs a split decode layer: the
    issued lookup, then apply_pending with the shared expert's output and gate logit."""
    assert supports_pending(method, x, inputs.weights)
    cache.moe_step = moe_step
    lookup = cache.issue_lookup(layer_id, inputs.topk_ids)
    outputs = apply_pending(
        method,
        cache.layer_banks(layer_id),
        x,
        inputs.weights,
        lookup,
        lambda _: (inputs.shared, inputs.shared_gate),
    )
    cache.end_issued_lookup(outputs.mailbox)
    return outputs.out, outputs.slots


def reference_layer(
    method: AffineMoEMethod,
    banks: dict[str, mx.array],
    x: mx.array,
    inputs: LayerInputs,
    slots: mx.array,
) -> mx.array:
    """The waiting lookup's decode layer at the slots: apply_slots, then the shared expert's add."""
    routed = method.apply_slots(banks, x, inputs.weights, inputs.topk_ids, slots)
    return sigmoid_gated_add(routed, inputs.shared, inputs.shared_gate)


def affine_method(spec: CheckpointSpec, top_k: int) -> AffineMoEMethod:
    cfg = MoEConfig(
        num_experts=spec.num_experts,
        top_k=top_k,
        hidden=spec.hidden_size,
        intermediate=spec.intermediate_size,
        schemes=spec.expert_schemes,
        activation="silu",
        dtype=mx.bfloat16,
        strategy="offload",
    )
    method = moe_method(cfg)
    assert isinstance(method, AffineMoEMethod)
    return method


def random_layer_inputs(
    rng: np.random.Generator, num_layers: int, num_rows: int, top_k: int, spec: CheckpointSpec
) -> list[LayerInputs]:
    def normal(*shape: int) -> mx.array:
        return mx.array(rng.standard_normal(shape, dtype=np.float32)).astype(mx.bfloat16)

    layers = []
    for _ in range(num_layers):
        ids = np.argsort(rng.random((num_rows, spec.num_experts)), axis=1)[:, :top_k]
        weights = mx.array(rng.random((num_rows, top_k), dtype=np.float32)).astype(mx.bfloat16)
        layers.append(
            LayerInputs(
                topk_ids=mx.array(ids.astype(np.int32)),
                weights=weights,
                shared=normal(num_rows, spec.hidden_size),
                shared_gate=normal(num_rows, 1),
            )
        )
    return layers


@pytest.mark.parametrize("case", CHAIN_CASES, ids=str)
def test_chain_matches_the_waiting_lookup(tmp_path: Path, case: ChainCase) -> None:
    spec = CheckpointSpec(
        num_layers=CHAIN_LAYERS,
        num_experts=case.num_experts,
        hidden_size=case.hidden_size,
        intermediate_size=case.intermediate_size,
        scheme=AffineScheme(case.bits, case.group_size),
    )
    caches = [
        build_cache(
            tmp_path / name,
            spec,
            # Each layer's pool holds a step's CHAIN_ROWS * top_k experts and a few more.
            cache_size=CHAIN_LAYERS * (CHAIN_ROWS * case.top_k + 4),
            prefill_overlap=False,
            is_packed=case.is_pair_codes,
            with_pair_codes=case.is_pair_codes,
        )[0]
        for name in ("waiting", "split")
    ]
    waiting, split = caches
    for cache in caches:
        cache.fill_initial()
    has_codes = "gate_up_proj.codes" in split.layer_banks(0)
    assert has_codes == case.is_pair_codes, "the pack's rows hold no pair codes"
    method = affine_method(spec, case.top_k)

    def split_chain(x: mx.array, layers: list[LayerInputs], moe_step: mx.array) -> mx.array:
        """The split path's summary: each layer's output bits, then its slots, as int32."""
        out = None
        summary = []
        for layer_id, inputs in enumerate(layers):
            out, slots = split_layer(split, method, layer_id, x, inputs.after(out), moe_step)
            summary += [out.view(mx.uint16).astype(mx.int32).reshape(-1), slots.reshape(-1)]
            x = out
        return mx.concatenate(summary)

    graph = None
    if case.is_replayed:
        moe_step = mx.zeros((2,), dtype=mx.int32)
        x = mx.zeros((CHAIN_ROWS, spec.hidden_size), dtype=mx.bfloat16)
        layers = random_layer_inputs(
            np.random.default_rng(0), CHAIN_LAYERS, CHAIN_ROWS, case.top_k, spec
        )
        step_inputs = [moe_step, x] + [array for inputs in layers for array in inputs.arrays()]
        mx.eval(step_inputs)
        pools = split.table_arrays()
        summary = split_chain(x, layers, moe_step)
        graph = capture(
            [summary, *split.table_arrays()], step_inputs, pools, split.graph_externals()
        )

    rng = np.random.default_rng(case.hidden_size + case.bits)
    for step in range(1, CHAIN_STEPS + 1):
        num_real_rows = int(rng.integers(0, CHAIN_ROWS + 1))
        moe_step = mx.array([num_real_rows, step], dtype=mx.int32)
        x = mx.array(rng.standard_normal((CHAIN_ROWS, spec.hidden_size), dtype=np.float32))
        x = x.astype(mx.bfloat16)
        layers = random_layer_inputs(rng, CHAIN_LAYERS, CHAIN_ROWS, case.top_k, spec)
        step_inputs = [moe_step, x] + [array for inputs in layers for array in inputs.arrays()]
        mx.eval(step_inputs)
        for cache in caches:
            cache.arm()
        expected = []
        layer_x = x
        for layer_id, inputs in enumerate(layers):
            slots, _ = waiting.lookup(layer_id, inputs.topk_ids, moe_step, _C.MoeLookupKind.DECODE)
            out = reference_layer(method, waiting.layer_banks(layer_id), layer_x, inputs, slots)
            expected += [out.view(mx.uint16).astype(mx.int32).reshape(-1), slots.reshape(-1)]
            layer_x = out
        expected_summary = mx.concatenate(expected)
        mx.eval(expected_summary, waiting.table_arrays())
        if graph is None:
            summary = split_chain(x, layers, moe_step)
        else:
            summary, pools = replay(graph, step_inputs, split.table_arrays())
            split.set_table_arrays(pools)
        mx.eval(summary, split.table_arrays())
        for cache in caches:
            cache.check_healthy()
        mismatches = (summary != expected_summary).sum().item()
        assert mismatches == 0, f"step {step}: {mismatches} output bits or slots differ"

    assert np.array(split.slot_of).tolist() == np.array(waiting.slot_of).tolist()
    assert np.array(split.last_use).tolist() == np.array(waiting.last_use).tolist()
    waiting_stats, split_stats = waiting.stats(), split.stats()
    assert split_stats.lookups[0] == waiting_stats.lookups[0] + (1 if case.is_replayed else 0)
    assert split_stats.lookup_hits == waiting_stats.lookup_hits
    assert split_stats.misses == waiting_stats.misses and sum(split_stats.misses) > 0
    words = mailbox_words(split.mailbox)
    assert words[Word.SEQUENCE] == words[Word.ACKNOWLEDGED] > 0
    assert words[Word.STATE] == State.DONE and words[Word.ERROR] == _C.MailboxError.NONE


EDGE_SPEC = CheckpointSpec(num_layers=2, num_experts=8, hidden_size=128, intermediate_size=64)
EDGE_TOP_K = 2
# After fill_initial each layer's pool of 4 slots holds its experts 0-3, expert e in slot 3 - e:
# layer 0's request misses 5 and 6, layer 1's 4 and 7; the first pairs' experts are resident.
EDGE_IDS = [[[0, 5], [1, 6]], [[1, 4], [7, 0]]]


def edge_cache(
    tmp_path: Path, *, run_coordinator: bool = True, stall_timeout_s: float = 600.0
) -> OffloadMoeCache:
    cache, _ = build_cache(
        tmp_path / "checkpoint",
        EDGE_SPEC,
        cache_size=8,
        prefill_overlap=False,
        stall_timeout_s=stall_timeout_s,
        run_coordinator=run_coordinator,
    )
    cache.fill_initial()
    return cache


def edge_layers(rng: np.random.Generator) -> list[LayerInputs]:
    layers = random_layer_inputs(rng, len(EDGE_IDS), 2, EDGE_TOP_K, EDGE_SPEC)
    return [
        LayerInputs(
            topk_ids=mx.array(ids, dtype=mx.int32),
            weights=inputs.weights,
            shared=inputs.shared,
            shared_gate=inputs.shared_gate,
        )
        for ids, inputs in zip(EDGE_IDS, layers)
    ]


def run_edge_chain(
    cache: OffloadMoeCache, layers: list[LayerInputs]
) -> tuple[mx.array, list[mx.array], list[mx.array]]:
    """The split chain over the layers, unevaluated: its input, each layer's output and slots."""
    method = affine_method(EDGE_SPEC, EDGE_TOP_K)
    x = mx.array(np.random.default_rng(1).standard_normal((2, 128), dtype=np.float32))
    x = x.astype(mx.bfloat16)
    mx.eval(x)
    moe_step = mx.array([2, 1], dtype=mx.int32)
    outs: list[mx.array] = []
    slots: list[mx.array] = []
    layer_x = x
    for layer_id, inputs in enumerate(layers):
        out, layer_slots = split_layer(
            cache,
            method,
            layer_id,
            layer_x,
            inputs.after(outs[-1] if outs else None),
            moe_step,
        )
        outs.append(out)
        slots.append(layer_slots)
        layer_x = out
    return x, outs, slots


def check_edge_outputs(
    cache: OffloadMoeCache,
    layers: list[LayerInputs],
    x: mx.array,
    outs: list[mx.array],
    slots: list[mx.array],
) -> None:
    """Every layer's output equals the waiting lookup's ops at its resolved slots, on the same
    banks."""
    method = affine_method(EDGE_SPEC, EDGE_TOP_K)
    layer_x = x
    for layer_id, inputs in enumerate(layers):
        expected = reference_layer(
            method, cache.layer_banks(layer_id), layer_x, inputs, slots[layer_id]
        )
        assert mx.array_equal(outs[layer_id], expected).item(), layer_id
        layer_x = outs[layer_id]


def test_waiters_read_rows_the_cpu_writes_while_they_wait(tmp_path: Path) -> None:
    """A coordinator in the test answers each request late, reading the misses into their slots
    (ExpertService.ensure_experts) while the waiters spin; they must read the new rows."""
    cache = edge_cache(tmp_path, run_coordinator=False)
    words = mailbox_words(cache.mailbox)
    layers = edge_layers(np.random.default_rng(3))
    x, outs, slots = run_edge_chain(cache, layers)
    answered_slots: list[list[int]] = []

    def coordinate() -> None:
        for sequence in range(1, len(layers) + 1):
            wait_for_post(words, sequence)
            time.sleep(ANSWER_DELAY_S)
            posted = words[Word.IDS : Word.IDS + 4].view(np.uint16)[: words[Word.NUM_IDS]]
            answered_slots.append(
                cache.ensure_experts(
                    int(words[Word.LAYER]), posted.astype(int).tolist(), int(words[Word.STEP])
                )
            )
            answer_request(words, sequence, State.DONE)

    started = time.monotonic()
    mx.async_eval(outs, slots, cache.table_arrays())
    coordinator = threading.Thread(target=coordinate)
    coordinator.start()
    mx.eval(outs, slots, cache.table_arrays())
    assert time.monotonic() - started >= len(layers) * ANSWER_DELAY_S, "the kernels did not wait"
    coordinator.join()
    assert len(answered_slots) == len(layers)
    slot_of = np.array(cache.slot_of).reshape(len(layers), -1)
    for layer_id, ids in enumerate(EDGE_IDS):
        first_slot = cache.slot_layout.pools[cache.layer_pools[layer_id]].first_slot
        expected = (slot_of[layer_id][np.array(ids)] - first_slot).tolist()
        assert np.array(slots[layer_id]).tolist() == expected, layer_id
    check_edge_outputs(cache, layers, x, outs, slots)


def fallback_slots(cache: OffloadMoeCache) -> list[list[list[int]]]:
    """moe_lookup's slots for EDGE_IDS when no request is answered: the resident experts', and
    for the misses the first pair's."""
    slot_of = np.array(cache.slot_of).reshape(len(EDGE_IDS), -1)
    expected = []
    for layer_id, ids in enumerate(EDGE_IDS):
        first_slot = cache.slot_layout.pools[cache.layer_pools[layer_id]].first_slot
        layer_slots = slot_of[layer_id] - first_slot
        fallback = int(layer_slots[ids[0][0]])
        expected.append(
            [[int(layer_slots[e]) if layer_slots[e] >= 0 else fallback for e in row] for row in ids]
        )
    return expected


def test_watchdog_releases_the_waiters(tmp_path: Path) -> None:
    cache = edge_cache(tmp_path, run_coordinator=False, stall_timeout_s=0.5)
    expected_slots = fallback_slots(cache)
    # One layer: each unanswered request takes the watchdog a few seconds.
    layers = edge_layers(np.random.default_rng(4))[:1]
    x, outs, slots = run_edge_chain(cache, layers)
    mx.eval(outs, slots, cache.table_arrays())
    words = mailbox_words(cache.mailbox)
    assert words[Word.ACKNOWLEDGED] == words[Word.SEQUENCE] == 1
    assert words[Word.STATE] == State.FAILED and words[Word.ERROR] == _C.MailboxError.TIMEOUT
    assert np.array(slots[0]).tolist() == expected_slots[0]
    check_edge_outputs(cache, layers, x, outs, slots)
    with pytest.raises(RuntimeError, match="unanswered"):
        cache.check_healthy()


def test_shutdown_releases_the_waiters(tmp_path: Path) -> None:
    cache = edge_cache(tmp_path, run_coordinator=False)
    expected_slots = fallback_slots(cache)
    layers = edge_layers(np.random.default_rng(5))
    words = mailbox_words(cache.mailbox)
    x, outs, slots = run_edge_chain(cache, layers)
    mx.async_eval(outs, slots, cache.table_arrays())
    wait_for_post(words, 1)
    cache.shutdown()
    mx.eval(outs, slots, cache.table_arrays())
    # The second layer posts nothing once the shutdown word is set.
    assert words[Word.SEQUENCE] == 1 and words[Word.STATE] == State.SHUTDOWN
    assert [np.array(layer_slots).tolist() for layer_slots in slots] == expected_slots
    check_edge_outputs(cache, layers, x, outs, slots)
    with pytest.raises(RuntimeError, match="shut down"):
        cache.check_healthy()


def test_failed_read_falls_back(tmp_path: Path) -> None:
    cache = edge_cache(tmp_path)
    expected_slots = fallback_slots(cache)
    # Layer 1's experts lie in the second file: its request fails, layer 0's is answered.
    last_file = tmp_path / "checkpoint" / "model-00001.safetensors"
    os.truncate(last_file, last_file.stat().st_size // 2)
    layers = edge_layers(np.random.default_rng(6))
    x, outs, slots = run_edge_chain(cache, layers)
    cache.arm()
    mx.eval(outs, slots, cache.table_arrays())
    words = mailbox_words(cache.mailbox)
    assert words[Word.ACKNOWLEDGED] == words[Word.SEQUENCE] == 2
    assert words[Word.STATE] == State.FAILED and words[Word.ERROR] == _C.MailboxError.IO
    assert np.array(slots[1]).tolist() == expected_slots[1]
    check_edge_outputs(cache, layers, x, outs, slots)
    with pytest.raises(RuntimeError, match="unexpected end of file"):
        cache.check_healthy()


# The shared expert's stand-in: shared = x, a zero gate logit, and the mailbox's first four words
# (State, Sequence, Acknowledged, Shutdown) as it runs, read coherently: the CPU writes them.
SHARED_PROBE = metal_kernel(
    name="shared_expert_probe",
    input_names=["x", "mailbox"],
    output_names=["shared", "gate", "seen"],
    source="""
const uint i = thread_position_in_grid.x;
shared[i] = x[i];
if (i == 0) {
  gate[0] = static_cast<T>(0);
  auto words = reinterpret_cast<volatile coherent(system) device const uint*>(mailbox);
  metal::atomic_thread_fence(
      metal::mem_flags::mem_device, metal::memory_order_seq_cst, metal::thread_scope_system);
  for (int w = 0; w < 4; w++) {
    seen[w] = words[w];
  }
}
""",
    header="""
#pragma METAL internals : enable
#ifndef __METAL_MEMORY_SCOPE_SYSTEM__
#define __METAL_MEMORY_SCOPE_SYSTEM__ 3
#endif
namespace metal {
constexpr constant metal::thread_scope thread_scope_system =
    static_cast<thread_scope>(__METAL_MEMORY_SCOPE_SYSTEM__);
}
""",
)


def test_shared_expert_runs_during_the_read(tmp_path: Path) -> None:
    """The coordinator in the test answers ANSWER_DELAY_S after the post. As in a model, x is
    unevaluated and the router's ids depend on it; the shared expert's stand-in is a chain of a
    probe, two ops and a probe, each probe recording the mailbox: both must find the request
    posted and unanswered, so the read overlaps the whole shared expert."""
    cache = edge_cache(tmp_path, run_coordinator=False)
    words = mailbox_words(cache.mailbox)
    method = affine_method(EDGE_SPEC, EDGE_TOP_K)
    inputs = edge_layers(np.random.default_rng(7))[0]
    x0 = mx.array(np.random.default_rng(1).standard_normal((2, 128), dtype=np.float32))
    x0 = x0.astype(mx.bfloat16)
    mx.eval(x0)
    x = -(-x0)
    topk_ids = inputs.topk_ids + mx.isnan(x[:1, :1]).astype(mx.int32) * 0
    # The mailbox before the issue: the probes read the words the lookup posts into, but no array
    # of the issue's orders them.
    mailbox = cache.mailbox
    seen: list[mx.array] = []

    def probe(y: mx.array) -> tuple[mx.array, mx.array]:
        shared, gate, words_seen = SHARED_PROBE(
            inputs=[y, mailbox],
            template=[("T", y.dtype)],
            grid=(y.size, 1, 1),
            threadgroup=(min(256, y.size), 1, 1),
            output_shapes=[y.shape, (y.shape[0], 1), (4,)],
            output_dtypes=[y.dtype, y.dtype, mx.uint32],
        )
        seen.append(words_seen)
        return shared, gate

    def shared_expert(shared_x: mx.array) -> tuple[mx.array, mx.array]:
        first, _ = probe(shared_x)
        return probe(-(-first))

    cache.moe_step = mx.array([2, 1], dtype=mx.int32)
    lookup = cache.issue_lookup(0, topk_ids)
    outputs = apply_pending(method, cache.layer_banks(0), x, inputs.weights, lookup, shared_expert)
    cache.end_issued_lookup(outputs.mailbox)

    def coordinate() -> None:
        wait_for_post(words, 1)
        time.sleep(ANSWER_DELAY_S)
        posted = words[Word.IDS : Word.IDS + 4].view(np.uint16)[: words[Word.NUM_IDS]]
        cache.ensure_experts(0, posted.astype(int).tolist(), 1)
        answer_request(words, 1, State.DONE)

    mx.async_eval(outputs.out, seen, cache.table_arrays())
    coordinator = threading.Thread(target=coordinate)
    coordinator.start()
    mx.eval(outputs.out, seen, cache.table_arrays())
    coordinator.join()
    for words_seen in seen:
        assert np.array(words_seen).tolist() == [State.POSTED, 1, 0, 0]
    probed = replace(inputs, shared=x0, shared_gate=mx.zeros((2, 1), dtype=x0.dtype))
    expected = reference_layer(method, cache.layer_banks(0), x0, probed, outputs.slots)
    assert mx.array_equal(outputs.out, expected).item()
