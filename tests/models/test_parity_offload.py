"""Numeric parity of MoE offload against resident experts on a tiny Qwen3.5 MoE model.

One checkpoint (affine 4-bit experts, 16 per layer over four layers, top-4) is served with
every expert resident and from an offload cache of 52 slots, which misses and evicts at
most steps: the fp32 greedy tokens must agree and the logits of the first prompt's last
prefill chunk must match. The cache is one pool with the prefill buffers over its first 32
slots unless a case says otherwise: a decode then fits three rows, and a looked-up prefill
of four tokens or more runs as two steps (a step holds three layers' 16 experts). Per-layer
pools under the buffers are runs of 13 (three rows, three tokens per looked-up forward);
beside dedicated buffers runs of 5 (one row, one token); one pool beside dedicated buffers
is 20 slots, a step per layer. Cases: decode replayed and eager; copy-miss and overlap
prefills (the overlap threshold knob lowered so tiny prompts take it), the chunks of one
prompt on both sides of the threshold, a prompt chunked through overlap prefills; two
requests decoding together, and three; prefill overlap off, with no buffers; layer 0's up
in 8 bits (its gate and up apart when resident, every layer's under offload) on a replayed
decode and an overlap prefill; the last layer's experts in 8 bits, one pool per layout or
per layer. Every other format the experts load in serves the same overlap, copy-miss and
decode forwards: unquantized, MLX's float modes, the Hugging Face layout (one tensor per
expert, or each layer's experts in two banks) and the ModelOpt and compressed-tensors
exports, NVFP4 or FP8 (whose unit scales the checkpoint does not store) with per-expert
output scales. A victim cache of a few entries, affine and NVFP4, serves the same forwards
from the experts the slots evict. So does an expert pack (ftm pack): of an affine checkpoint
(16-bit pair codes for its fp32 scales), of one whose coarse scales take narrower codes, of an
export (its parts beside unit scales and output scales) and of a Hugging Face layout one, every
expert read a whole record at a time; a stale pack and FTMETAL_MOE_PACK=0 read the checkpoint.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten
from mlx_lm.models import qwen3_5 as reference

from ftmetal.cli import run_pack
from ftmetal.core import SamplingParams
from ftmetal.engine.config import EngineConfig
from ftmetal.layers.quantization.config import FP_BLOCK_SHAPES
from ftmetal.models.config import MoeStrategy
from ftmetal.moe.pack import PACK_FILE
from ftmetal.moe.slot_pools import MoePoolGrouping, PrefillBufferPlacement
from ftmetal.scheduler.scheduler import Scheduler
from tiny_model import (
    EXPERTS,
    LINEAR_ATTENTION,
    TINY_TEXT_CONFIG,
    build_reference,
    hf_layout_weights,
    save_checkpoint,
    save_export,
    save_hf_checkpoint,
)

NUM_EXPERTS = 16
OFFLOAD_TEXT_CONFIG = {**TINY_TEXT_CONFIG, "num_experts": NUM_EXPERTS, "num_experts_per_tok": 4}
# Group 32: the tiny widths (64 and 96) are not multiples of 64.
EXPERT_GROUP_SIZE = 32
EXPERT_BITS = 4
CACHE_SIZE = 52
STEPS = 10
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
# Absolute and relative: the paths differ in summation order only (a weighted sum against
# moe_combine, the sorted tile kernel against the per-row matvec).
LOGITS_TOLERANCE = 1e-4
# Layer 0's up projection, by its MLX layout name.
FIRST_UP = r"layers\.0\.mlp\.switch_mlp\.up_proj$"

# How the checkpoint stores its experts: MLX affine (in up_bits and last_layer_bits) or a float
# mode, unquantized in the MLX or the Hugging Face layout (one tensor per expert, or stacked in
# two banks), or an export's NVFP4 or FP8 (in the Hugging Face layout, or under MLX names), or
# NVFP4 with layer 0's up in FP8.
ExpertFormat = Literal[
    "affine",
    "mxfp4",
    "nvfp4",
    "mxfp8",
    "unquantized",
    "hf-unquantized",
    "hf-unquantized-stacked",
    "modelopt",
    "modelopt-mlx-names",
    "modelopt-up-fp8",
    "compressed-tensors",
    "compressed-tensors-fp8",
]


@dataclass(frozen=True)
class Case:
    name: str
    prompt_lens: tuple[int, ...]
    max_extend_tokens: int = 64
    overlap_min_tokens: int = 128
    command_graph: bool = True
    prefill_overlap: bool = True
    cache_size: int = CACHE_SIZE
    grouping: MoePoolGrouping = "unified"
    buffer_placement: PrefillBufferPlacement = "overlay"
    # The rows a decode may take: the requests that run at once.
    max_decode_rows: int = 3
    # Whether some prefill reads layers ahead into the prefill buffers.
    reads_buffers: bool = False
    # Bits of layer 0's up projection.
    up_bits: int = EXPERT_BITS
    last_layer_bits: int = EXPERT_BITS
    expert_format: ExpertFormat = "affine"
    # The offload's victim cache (0: none).
    victim_cache_gib: float = 0
    # Offload reads an expert pack of the checkpoint (ftm pack), and whether it holds pair codes;
    # coarse: 256-wide experts whose scales and biases take 32 levels each (narrow codes).
    is_packed: bool = False
    has_pair_codes: bool = False
    is_coarse: bool = False


CASES = [
    Case("replayed-decode-copy-miss-prefill", (9,)),
    Case("eager-decode", (9,), command_graph=False),
    Case("overlap-prefill", (40,), overlap_min_tokens=16, reads_buffers=True),
    # 64 tokens through the buffers, then 36 looked up.
    Case("chunks-across-the-threshold", (100,), overlap_min_tokens=48, reads_buffers=True),
    # 64, 64 and 22 tokens through the buffers.
    Case("chunked-overlap-prefill", (150,), overlap_min_tokens=16, reads_buffers=True),
    Case("two-requests", (13, 29)),
    Case("three-requests", (9, 17, 26)),
    Case("overlap-off", (40,), prefill_overlap=False, cache_size=32, max_decode_rows=2),
    Case("up-8-bit-replayed-decode", (9,), up_bits=8),
    Case("up-8-bit-overlap-prefill", (40,), overlap_min_tokens=16, reads_buffers=True, up_bits=8),
    Case(
        "per-layer-dedicated-chunks-across-the-threshold",
        (100, 9),
        overlap_min_tokens=48,
        grouping="per-layer",
        buffer_placement="dedicated",
        max_decode_rows=1,
        reads_buffers=True,
    ),
    Case(
        "per-layer-overlaid-chunked-overlap-prefill",
        (150, 9),
        overlap_min_tokens=16,
        grouping="per-layer",
        max_decode_rows=3,
        reads_buffers=True,
    ),
    Case(
        "unified-dedicated-chunks-across-the-threshold",
        (100,),
        overlap_min_tokens=48,
        buffer_placement="dedicated",
        max_decode_rows=1,
        reads_buffers=True,
    ),
    # Pools of 13 (the last layer) and 39 (the others): three rows of every layer fit.
    Case(
        "mixed-layouts-overlap-prefill",
        (40, 9),
        overlap_min_tokens=16,
        reads_buffers=True,
        last_layer_bits=8,
    ),
    Case(
        "mixed-layouts-per-layer",
        (100, 9),
        overlap_min_tokens=48,
        grouping="per-layer",
        reads_buffers=True,
        up_bits=8,
        last_layer_bits=8,
    ),
    # The other formats: 64 tokens through the buffers, 36 looked up, both requests decoding.
    Case("mxfp4", (100, 9), overlap_min_tokens=48, reads_buffers=True, expert_format="mxfp4"),
    Case("nvfp4", (100, 9), overlap_min_tokens=48, reads_buffers=True, expert_format="nvfp4"),
    Case("mxfp8", (100, 9), overlap_min_tokens=48, reads_buffers=True, expert_format="mxfp8"),
    Case(
        "unquantized",
        (100, 9),
        overlap_min_tokens=48,
        reads_buffers=True,
        expert_format="unquantized",
    ),
    Case(
        "hf-unquantized",
        (100, 9),
        overlap_min_tokens=48,
        reads_buffers=True,
        expert_format="hf-unquantized",
    ),
    Case(
        "hf-unquantized-stacked",
        (100, 9),
        overlap_min_tokens=48,
        reads_buffers=True,
        expert_format="hf-unquantized-stacked",
    ),
    Case("modelopt", (100, 9), overlap_min_tokens=48, reads_buffers=True, expert_format="modelopt"),
    Case(
        "modelopt-mlx-names",
        (100, 9),
        overlap_min_tokens=48,
        reads_buffers=True,
        expert_format="modelopt-mlx-names",
    ),
    Case(
        "modelopt-up-fp8",
        (100, 9),
        overlap_min_tokens=48,
        reads_buffers=True,
        expert_format="modelopt-up-fp8",
    ),
    Case(
        "compressed-tensors",
        (100, 9),
        overlap_min_tokens=48,
        reads_buffers=True,
        expert_format="compressed-tensors",
    ),
    Case(
        "compressed-tensors-fp8",
        (100, 9),
        overlap_min_tokens=48,
        reads_buffers=True,
        expert_format="compressed-tensors-fp8",
    ),
    # Six affine (eight NVFP4) victim cache entries for the 12 experts outside the slots.
    Case(
        "victim-cache",
        (100, 9),
        overlap_min_tokens=48,
        reads_buffers=True,
        victim_cache_gib=8e-5,
    ),
    Case(
        "nvfp4-victim-cache",
        (100, 9),
        overlap_min_tokens=48,
        reads_buffers=True,
        expert_format="nvfp4",
        victim_cache_gib=8e-5,
    ),
    # From an expert pack: affine, whose fp32 scales and biases take 16-bit pair codes; an
    # export's parts beside its unit scales (not stored) and output scales (resident), gate and up
    # apart; one tensor per expert.
    Case(
        "packed",
        (100, 9),
        overlap_min_tokens=48,
        reads_buffers=True,
        is_packed=True,
        has_pair_codes=True,
    ),
    Case(
        "modelopt-up-fp8-packed",
        (100, 9),
        overlap_min_tokens=48,
        reads_buffers=True,
        expert_format="modelopt-up-fp8",
        is_packed=True,
    ),
    Case(
        "hf-unquantized-packed",
        (100, 9),
        overlap_min_tokens=48,
        reads_buffers=True,
        expert_format="hf-unquantized",
        is_packed=True,
    ),
    Case(
        "pair-codes-packed",
        (100, 9),
        overlap_min_tokens=48,
        reads_buffers=True,
        is_packed=True,
        has_pair_codes=True,
        is_coarse=True,
    ),
]
# The routed experts' width under pair codes, and the levels their scales and biases take.
CODED_EXPERT_WIDTH = 256
CODED_LEVELS = 32


def is_expert_projection(module_path: str, module: nn.Module) -> bool:
    return "switch_mlp" in module_path and hasattr(module, "to_quantized")


def write_checkpoint(path: Path, case: Case) -> None:
    """The reference model with its experts in the case's format: affine quantized alone,
    layer 0's up in up_bits and the last layer's in last_layer_bits, or another."""
    text_config = (
        {**OFFLOAD_TEXT_CONFIG, "moe_intermediate_size": CODED_EXPERT_WIDTH}
        if case.is_coarse
        else OFFLOAD_TEXT_CONFIG
    )
    model = build_reference(seed=11, text_config=text_config)
    match case.expert_format:
        case "affine":
            write_affine_checkpoint(path, model, case.up_bits, case.last_layer_bits, case.is_coarse)
        case "mxfp4" | "nvfp4" | "mxfp8" as mode:
            bits, group_size = FP_BLOCK_SHAPES[mode]
            nn.quantize(model, group_size, bits, mode=mode, class_predicate=is_expert_projection)
            mx.eval(model.parameters())
            quantization = {"group_size": group_size, "bits": bits, "mode": mode}
            save_checkpoint(model, path, quantization=quantization)
        case "unquantized":
            save_checkpoint(model, path, quantization=None)
        case "hf-unquantized" | "hf-unquantized-stacked" as layout:
            weights = dict(tree_flatten(model.parameters()))
            is_stacked = layout == "hf-unquantized-stacked"
            hf_weights = hf_layout_weights(
                weights, "model.language_model.", has_stacked_experts=is_stacked
            )
            save_hf_checkpoint(hf_weights, model, path, "Qwen3_5MoeForConditionalGeneration")
        # The export quantizes the same reference model itself.
        case "modelopt" | "compressed-tensors" as dialect:
            save_export(dialect, 11, path, mx.float32, text_config=OFFLOAD_TEXT_CONFIG)
        case "modelopt-mlx-names":
            save_export(
                "modelopt",
                11,
                path,
                mx.float32,
                text_config=OFFLOAD_TEXT_CONFIG,
                is_hf_layout=False,
            )
        case "modelopt-up-fp8":
            save_export(
                "modelopt",
                11,
                path,
                mx.float32,
                text_config=OFFLOAD_TEXT_CONFIG,
                formats=(FIRST_UP, EXPERTS),
            )
        case "compressed-tensors-fp8":
            save_export(
                "compressed-tensors",
                11,
                path,
                mx.float32,
                text_config=OFFLOAD_TEXT_CONFIG,
                formats=(EXPERTS, LINEAR_ATTENTION),
            )


def write_affine_checkpoint(
    path: Path, model: reference.Model, up_bits: int, last_layer_bits: int, is_coarse: bool
) -> None:
    """The model's experts affine quantized, with is_coarse each projection's scales and biases
    rounded to CODED_LEVELS levels (few enough pairs for pair codes)."""
    num_layers = OFFLOAD_TEXT_CONFIG["num_hidden_layers"]
    assert isinstance(num_layers, int)
    last_layer = f"language_model.model.layers.{num_layers - 1}.mlp.switch_mlp"
    first_up = "language_model.model.layers.0.mlp.switch_mlp.up_proj"
    up = {"group_size": EXPERT_GROUP_SIZE, "bits": up_bits}
    last = {"group_size": EXPERT_GROUP_SIZE, "bits": last_layer_bits}

    def predicate(module_path: str, module: nn.Module) -> bool | dict[str, int]:
        if not is_expert_projection(module_path, module):
            return False
        elif module_path.startswith(last_layer):
            return last
        else:
            return up if module_path == first_up else True

    nn.quantize(model, group_size=EXPERT_GROUP_SIZE, bits=EXPERT_BITS, class_predicate=predicate)
    if is_coarse:
        for module_path, module in model.named_modules():
            if "switch_mlp" in module_path and hasattr(module, "biases"):
                for name in ("scales", "biases"):
                    values = getattr(module, name)
                    low, high = values.min(), values.max()
                    step = (high - low) / (CODED_LEVELS - 1)
                    setattr(module, name, low + mx.round((values - low) / step) * step)
    mx.eval(model.parameters())
    quantization: dict[str, object] = {
        "group_size": EXPERT_GROUP_SIZE,
        "bits": EXPERT_BITS,
        "mode": "affine",
    }
    quantization[first_up] = up
    quantization.update({f"{last_layer}.{projection}": last for projection in PROJECTIONS})
    save_checkpoint(model, path, quantization=quantization)


def make_scheduler(path: Path, case: Case, strategy: MoeStrategy) -> Scheduler:
    config = EngineConfig(
        model_path=str(path),
        dtype=mx.float32,
        max_running_req=max(2, len(case.prompt_lens)),
        page_size=4,
        num_page_override=128,
        max_extend_tokens_override=case.max_extend_tokens,
        command_graph=case.command_graph,
        moe_strategy=strategy,
        moe_cache_size=case.cache_size if strategy == "offload" else 0,
        moe_pool=case.grouping,
        moe_prefill_overlap=case.prefill_overlap,
        moe_prefill_buffer=case.buffer_placement,
        moe_victim_cache_gib=case.victim_cache_gib if strategy == "offload" else 0,
    )
    return Scheduler(config, eos_token_ids={127})


def serve(
    path: Path, case: Case, strategy: MoeStrategy, prompts: list[list[int]]
) -> tuple[mx.array, list[list[int]], Scheduler]:
    """The logits of the first prompt's last prefill chunk, every prompt's greedy tokens
    decoded together, and the scheduler that decoded them."""
    scheduler = make_scheduler(path, case, strategy)
    scheduler.add_request(0, np.array(prompts[0], dtype=np.int32), SamplingParams(max_tokens=1))
    while True:
        forward_input = scheduler.schedule_next_batch()
        assert forward_input is not None
        if forward_input.batch.reqs[0].can_decode:
            logits = scheduler.engine.model.forward(forward_input.batch)
            mx.eval(logits)
            break
        scheduler.process_last_data(scheduler.forward(forward_input))

    scheduler = make_scheduler(path, case, strategy)
    for uid, prompt in enumerate(prompts):
        params = SamplingParams(temperature=0.0, max_tokens=STEPS, ignore_eos=True)
        scheduler.add_request(uid, np.array(prompt, dtype=np.int32), params)
    events = scheduler.run_until_idle()
    scheduler.cache_manager.check_integrity()
    tokens = [[event.token for event in events if event.uid == uid] for uid in range(len(prompts))]
    return logits, tokens, scheduler


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_offload_matches_resident_experts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: Case
) -> None:
    monkeypatch.setenv("FTMETAL_MOE_PREFILL_OVERLAP_MIN_TOKENS", str(case.overlap_min_tokens))
    write_checkpoint(tmp_path, case)
    if case.is_packed:
        assert run_pack(["--model", str(tmp_path)]) == 0
    rng = np.random.default_rng(len(case.name))
    prompts = [[int(t) for t in rng.integers(1, 120, size=n)] for n in case.prompt_lens]

    resident_logits, resident_tokens, _ = serve(tmp_path, case, "fused", prompts)
    logits, tokens, scheduler = serve(tmp_path, case, "offload", prompts)
    assert tokens == resident_tokens
    assert mx.allclose(logits, resident_logits, atol=LOGITS_TOLERANCE, rtol=LOGITS_TOLERANCE)

    cache = scheduler.engine.moe_offload_cache
    assert cache is not None
    assert (scheduler.engine.expert_pack is not None) == case.is_packed
    assert (cache.pairs is not None) == case.has_pair_codes
    # Dedicated buffers take two layers of the slots out of the pools.
    dedicated = 2 * NUM_EXPERTS if case.buffer_placement == "dedicated" else 0
    assert cache.cache_size == case.cache_size - dedicated
    # One layer's gate and up apart puts every layer's apart.
    is_apart = case.up_bits != EXPERT_BITS or case.expert_format == "modelopt-up-fp8"
    assert all(
        ("gate_up_proj.weight" not in pool.layout) == is_apart for pool in cache.slot_layout.pools
    )
    assert cache.slot_layout.max_decode_rows == case.max_decode_rows
    max_running = max(2, len(case.prompt_lens))
    assert len(scheduler.table_manager.free_slots) == min(max_running, case.max_decode_rows)
    stats = cache.stats()
    assert sum(stats.misses) > 0
    # A miss is an expert read or a victim cache hit; so is a prefill buffer's absent expert,
    # which is no miss.
    copied = sum(stats.victim_hits) + sum(stats.victim_buffer_hits)
    assert (sum(stats.experts_read) + copied > sum(stats.misses)) == case.reads_buffers
    assert (copied > 0) == (case.victim_cache_gib > 0)
    if case.is_packed:
        # Every read a run of whole records: the records' bytes alone, FILL parts or not.
        with (tmp_path / PACK_FILE).open("rb") as f:
            header = json.loads(f.read(struct.unpack("<Q", f.read(8))[0]))
        for layer_id, layer in enumerate(header["layers"]):
            assert stats.bytes_read[layer_id] == stats.experts_read[layer_id] * layer["stride"]
    cache.check_healthy()


@pytest.mark.parametrize("fallback", ["stale", "ignored"])
def test_offload_reads_the_checkpoint_past_its_pack(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    fallback: Literal["stale", "ignored"],
) -> None:
    """A pack of another checkpoint (one file grown since) is logged and passed over;
    FTMETAL_MOE_PACK=0 passes over a good one."""
    case = Case("pack-fallback", (9,))
    write_checkpoint(tmp_path, case)
    assert run_pack(["--model", str(tmp_path)]) == 0
    if fallback == "stale":
        with next(tmp_path.glob("*.safetensors")).open("ab") as f:
            f.write(b"\0")
    else:
        monkeypatch.setenv("FTMETAL_MOE_PACK", "0")
    scheduler = make_scheduler(tmp_path, case, "offload")
    cache = scheduler.engine.moe_offload_cache
    assert cache is not None and cache.pairs is None
    assert scheduler.engine.expert_pack is None
    is_warned = any("reading the experts from the checkpoint" in r.message for r in caplog.records)
    assert is_warned == (fallback == "stale")


def test_offload_refuses_layers_in_different_banks(tmp_path: Path) -> None:
    """The last layer's experts in bf16 beside affine ones elsewhere: one set of banks holds
    either, not both."""
    model = build_reference(seed=11, text_config=OFFLOAD_TEXT_CONFIG)
    num_layers = OFFLOAD_TEXT_CONFIG["num_hidden_layers"]
    assert isinstance(num_layers, int)
    last_layer = f"language_model.model.layers.{num_layers - 1}."

    def predicate(module_path: str, module: nn.Module) -> bool:
        return is_expert_projection(module_path, module) and not module_path.startswith(last_layer)

    nn.quantize(model, EXPERT_GROUP_SIZE, EXPERT_BITS, class_predicate=predicate)
    mx.eval(model.parameters())
    quantization = {"group_size": EXPERT_GROUP_SIZE, "bits": EXPERT_BITS, "mode": "affine"}
    save_checkpoint(model, tmp_path, quantization=quantization)
    make_scheduler(tmp_path, Case("resident", (9,)), "fused")
    with pytest.raises(ValueError, match="one set of banks"):
        make_scheduler(tmp_path, Case("offload", (9,)), "offload")
