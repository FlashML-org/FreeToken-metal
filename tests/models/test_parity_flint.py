"""Numeric parity of the paged engine against a plain reference on a tiny Flint.

The reference (tiny_flint_model.Model: contiguous caches for the full layers, rotating ones for
the sliding layers) and ftmetal (both pools, both attention backends, the scheduler with overlap
and decode graphs, the SWA radix cache) run the same random weights: prefill logits must match and
greedy decode must agree token for token, with the window crossed in prefill, in decode and at
chunk seams, batched, resumed from the prefix cache, quantized as flint-int4 stores it and stored
as the conversions that nest the text model under language_model. do. Under MoE offload, whose
layer indices start past the dense layers and whose shared expert has no gate: replayed decode with
and without split lookup, eager decode, an overlap prefill, two requests, an expert pack and a
conversion's experts read from their stored names. The nvfp4 conversion, each linear under a
global scale, must serve like the reference on its dequantized weights, fused and offloaded.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import mlx.core as mx
import numpy as np
import pytest

from ftmetal.attention.metadata import AttnType
from ftmetal.cli import run_pack
from ftmetal.core import SamplingParams
from ftmetal.engine.config import EngineConfig
from ftmetal.kvcache.mha_pool import MHAKVCache
from ftmetal.kvcache.swa_pool import SWAKVCache
from ftmetal.layers import moe as moe_layers
from ftmetal.layers.quantization.moe import AffineMoEMethod, SharedExpertForward
from ftmetal.layers.quantization.moe_pending import PendingOutputs
from ftmetal.models.config import MoeStrategy
from ftmetal.models.flint.config import parse_config
from ftmetal.models.flint.weight import mlx_layout
from ftmetal.models.register import create_model, read_checkpoint
from ftmetal.moe.offload_cache import IssuedLookup
from ftmetal.scheduler.scheduler import Scheduler
from tiny_flint_model import (
    EOS_TOKEN_ID,
    ROUTER_BITS,
    TINY_FLINT_CONFIG,
    WINDOW,
    build_reference,
    quantize,
    quantize_nvfp4,
    reference_greedy,
    save_checkpoint,
    save_conversion,
)

LOGITS_TOLERANCE = 1e-4
NUM_MOE_LAYERS = 4
# 16 experts over the four MoE layers, top 4; 52 slots miss and evict at most steps.
OFFLOAD_CONFIG = {**TINY_FLINT_CONFIG, "num_experts": 16, "num_experts_per_tok": 4}
OFFLOAD_CACHE_SIZE = 52
OFFLOAD_STEPS = 10


def make_scheduler(
    path: Path,
    attention_backend: str = "metal",
    page_size: int = 1,
    max_extend_tokens: int = 64,
    max_running_req: int = 2,
    command_graph: bool = True,
    strategy: MoeStrategy = "fused",
    split_lookup: bool = True,
) -> Scheduler:
    config = EngineConfig(
        model_path=str(path),
        dtype=mx.float32,
        max_running_req=max_running_req,
        page_size=page_size,
        attention_backend=attention_backend,
        num_page_override=128,
        max_extend_tokens_override=max_extend_tokens,
        command_graph=command_graph,
        moe_strategy=strategy,
        moe_cache_size=OFFLOAD_CACHE_SIZE if strategy == "offload" else 0,
        moe_split_lookup=split_lookup,
    )
    return Scheduler(config, eos_token_ids={EOS_TOKEN_ID})


def random_prompt(seed: int, length: int) -> list[int]:
    rng = np.random.default_rng(seed)
    return [int(token) for token in rng.integers(1, EOS_TOKEN_ID, size=length)]


def generate(scheduler: Scheduler, prompts: list[list[int]], steps: int) -> list[list[int]]:
    params = SamplingParams(temperature=0.0, max_tokens=steps, ignore_eos=True)
    for uid, prompt in enumerate(prompts):
        scheduler.add_request(uid, np.array(prompt, dtype=np.int32), params)
    events = scheduler.run_until_idle()
    scheduler.cache_manager.check_integrity()
    return [[event.token for event in events if event.uid == uid] for uid in range(len(prompts))]


@dataclass(frozen=True)
class GreedyCase:
    name: str
    attention_backend: Literal["mlx", "metal"]
    page_size: int
    prompt_len: int
    steps: int
    max_extend_tokens: int = 64
    is_quantized: bool = False
    command_graph: bool = True
    # Stored as a conversion under language_model. (save_conversion); quantized, its router too.
    is_conversion: bool = False


GREEDY_CASES = [
    GreedyCase("mlx-inside-the-window", "mlx", 1, 4, 3),
    GreedyCase("metal-inside-the-window", "metal", 1, 4, 3),
    GreedyCase("mlx-across-the-window-in-prefill", "mlx", 1, 23, 6),
    GreedyCase("metal-across-the-window-in-prefill", "metal", 1, 23, 6),
    GreedyCase("metal-across-the-window-in-decode", "metal", 4, 3, 20),
    GreedyCase("metal-eager-decode", "metal", 1, 3, 20, command_graph=False),
    GreedyCase("mlx-chunk-seams", "mlx", 1, 29, 6, max_extend_tokens=12),
    GreedyCase("metal-chunk-seams", "metal", 4, 29, 6, max_extend_tokens=12),
    GreedyCase("mlx-flint-int4", "mlx", 1, 19, 16, is_quantized=True),
    GreedyCase("metal-flint-int4", "metal", 1, 19, 16, is_quantized=True),
    GreedyCase("mlx-conversion", "mlx", 1, 19, 16, is_quantized=True, is_conversion=True),
    GreedyCase("metal-conversion", "metal", 1, 19, 16, is_quantized=True, is_conversion=True),
    GreedyCase("metal-bf16-conversion", "metal", 1, 19, 16, is_conversion=True),
]


@pytest.mark.parametrize("case", GREEDY_CASES, ids=lambda case: case.name)
def test_greedy_decode_matches_reference(tmp_path: Path, case: GreedyCase) -> None:
    model = build_reference(seed=11)
    if case.is_quantized:
        quantize(model, router_bits=ROUTER_BITS if case.is_conversion else None)
    if case.is_conversion:
        save_conversion(model, tmp_path)
    else:
        save_checkpoint(model, tmp_path, case.is_quantized)
    prompt = random_prompt(case.prompt_len, case.prompt_len)
    _, expected = reference_greedy(model, prompt, case.steps)
    assert len(set(expected)) > 1 or case.steps < 6, "a token stream too flat to tell paths apart"
    scheduler = make_scheduler(
        tmp_path,
        case.attention_backend,
        case.page_size,
        case.max_extend_tokens,
        command_graph=case.command_graph,
    )
    assert generate(scheduler, [prompt], case.steps) == [expected]


@pytest.mark.parametrize("attention_backend", ["mlx", "metal"])
def test_two_requests_batched_match_single(tmp_path: Path, attention_backend: str) -> None:
    model = build_reference(seed=12)
    save_checkpoint(model, tmp_path, is_quantized=False)
    prompts = [random_prompt(3, 11), random_prompt(4, 5)]
    expected = [reference_greedy(model, prompt, 12)[1] for prompt in prompts]
    assert generate(make_scheduler(tmp_path, attention_backend), prompts, 12) == expected


@pytest.mark.parametrize("attention_backend", ["mlx", "metal"])
def test_prefill_logits_match_reference(tmp_path: Path, attention_backend: str) -> None:
    model = build_reference(seed=14)
    save_checkpoint(model, tmp_path, is_quantized=False)
    prompt = random_prompt(6, 21)
    expected_logits, _ = reference_greedy(model, prompt, 1)
    scheduler = make_scheduler(tmp_path, attention_backend)
    scheduler.add_request(0, np.array(prompt, dtype=np.int32), SamplingParams(max_tokens=1))
    forward_input = scheduler.schedule_next_batch()
    assert forward_input is not None
    logits = scheduler.engine.model.forward(forward_input.batch)
    assert mx.allclose(logits[0], expected_logits[-1], atol=LOGITS_TOLERANCE, rtol=LOGITS_TOLERANCE)


def test_sinks_change_the_logits(tmp_path: Path) -> None:
    """The parity above would pass with sinks dropped on both sides: the kernels' are compared to
    the reference's, whose sinks must move the logits."""
    model = build_reference(seed=15)
    prompt = random_prompt(7, 21)
    with_sinks, _ = reference_greedy(model, prompt, 1)
    for layer in model.model.layers:
        layer.self_attn.softmax_offset = mx.zeros_like(layer.self_attn.softmax_offset)
    without_sinks, _ = reference_greedy(model, prompt, 1)
    assert not mx.allclose(with_sinks, without_sinks, atol=1e-2, rtol=1e-2)


def test_a_prompt_resumes_its_cached_prefix(tmp_path: Path) -> None:
    model = build_reference(seed=17)
    save_checkpoint(model, tmp_path, is_quantized=False)
    prompt = random_prompt(8, 29)
    _, expected = reference_greedy(model, prompt, 6)
    scheduler = make_scheduler(tmp_path, max_extend_tokens=12)
    assert generate(scheduler, [prompt], 6) == [expected]
    params = SamplingParams(temperature=0.0, max_tokens=6, ignore_eos=True)
    scheduler.add_request(1, np.array(prompt, dtype=np.int32), params)
    forward_input = scheduler.schedule_next_batch()
    assert forward_input is not None
    assert forward_input.batch.reqs[0].cached_len == len(prompt) - 1
    data = scheduler.forward(forward_input)
    while scheduler.runnable or data is not None:
        data = scheduler.overlap_loop(data)
    scheduler.cache_manager.check_integrity()
    assert [event.token for event in scheduler.take_results()] == expected


def test_engine_builds_both_pools_and_the_dense_prefix(tmp_path: Path) -> None:
    save_checkpoint(build_reference(seed=16), tmp_path, is_quantized=False)
    engine = make_scheduler(tmp_path).engine
    full_pool, swa_pool = engine.pools[AttnType.FULL], engine.pools[AttnType.SWA]
    assert isinstance(full_pool, MHAKVCache) and isinstance(swa_pool, SWAKVCache)
    assert full_pool.layer_ids == (0, 1, 4) and swa_pool.layer_ids == (2, 3, 5)
    assert engine.model_config.num_moe_layers == NUM_MOE_LAYERS
    assert swa_pool.sliding_window == WINDOW


@dataclass(frozen=True)
class OffloadCase:
    name: str
    prompt_lens: tuple[int, ...]
    overlap_min_tokens: int = 128
    command_graph: bool = True
    split_lookup: bool = True
    is_packed: bool = False
    is_conversion: bool = False


OFFLOAD_CASES = [
    OffloadCase("replayed-decode", (9,)),
    OffloadCase("eager-decode", (9,), command_graph=False),
    OffloadCase("replayed-decode-split-off", (9, 17), split_lookup=False),
    OffloadCase("overlap-prefill", (40,), overlap_min_tokens=16),
    OffloadCase("chunks-across-the-threshold", (100,), overlap_min_tokens=48),
    OffloadCase("two-requests", (13, 29)),
    OffloadCase("expert-pack", (9,), is_packed=True),
    OffloadCase("conversion", (9,), is_conversion=True),
]


@pytest.mark.parametrize("case", OFFLOAD_CASES, ids=lambda case: case.name)
def test_offload_matches_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: OffloadCase
) -> None:
    monkeypatch.setenv("FTMETAL_MOE_PREFILL_OVERLAP_MIN_TOKENS", str(case.overlap_min_tokens))
    model = build_reference(seed=18, config=OFFLOAD_CONFIG)
    if case.is_conversion:
        quantize(model, router_bits=ROUTER_BITS)
        save_conversion(model, tmp_path, config=OFFLOAD_CONFIG)
    else:
        quantize(model)
        save_checkpoint(model, tmp_path, is_quantized=True, config=OFFLOAD_CONFIG)
    if case.is_packed:
        assert run_pack(["--model", str(tmp_path)]) == 0
    prompts = [random_prompt(n, n) for n in case.prompt_lens]
    expected = [reference_greedy(model, prompt, OFFLOAD_STEPS)[1] for prompt in prompts]
    pending_layers: list[int] = []
    apply_pending = moe_layers.apply_pending

    def recorded_apply_pending(
        method: AffineMoEMethod,
        banks: Mapping[str, mx.array],
        x: mx.array,
        topk_weights: mx.array,
        lookup: IssuedLookup,
        shared_expert: SharedExpertForward | None,
    ) -> PendingOutputs:
        pending_layers.append(lookup.layer_id)
        return apply_pending(method, banks, x, topk_weights, lookup, shared_expert)

    monkeypatch.setattr(moe_layers, "apply_pending", recorded_apply_pending)
    scheduler = make_scheduler(
        tmp_path,
        page_size=4,
        max_running_req=max(2, len(prompts)),
        command_graph=case.command_graph,
        strategy="offload",
        split_lookup=case.split_lookup,
    )
    assert generate(scheduler, prompts, OFFLOAD_STEPS) == expected
    cache = scheduler.engine.moe_offload_cache
    assert cache is not None and cache.num_layers == NUM_MOE_LAYERS
    assert (scheduler.engine.expert_pack is not None) == case.is_packed
    assert set(pending_layers) == (set(range(NUM_MOE_LAYERS)) if case.split_lookup else set())
    assert sum(cache.stats().misses) > 0
    cache.check_healthy()


@pytest.mark.parametrize("strategy", ["fused", "offload"])
def test_nvfp4_conversion_matches_its_dequantized_weights(
    tmp_path: Path, strategy: MoeStrategy
) -> None:
    model = build_reference(seed=19, config=OFFLOAD_CONFIG)
    nvfp4_tensors = quantize_nvfp4(model)
    save_conversion(model, tmp_path, config=OFFLOAD_CONFIG, nvfp4_tensors=nvfp4_tensors)
    prompt = random_prompt(10, 19)
    _, expected = reference_greedy(model, prompt, OFFLOAD_STEPS)
    scheduler = make_scheduler(tmp_path, page_size=4, strategy=strategy)
    assert generate(scheduler, [prompt], OFFLOAD_STEPS) == [expected]


# A module the nvfp4 conversion stores with a global scale, declared otherwise under its own name.
SCALED_MODULE = "model.layers.2.self_attn.q_proj"
NOT_NVFP4_ENTRIES: list[dict[str, object]] = [
    {"group_size": 64, "bits": 4, "mode": "affine"},
    {"group_size": 32, "bits": 8, "mode": "mxfp8"},
]


@pytest.mark.parametrize("entry", NOT_NVFP4_ENTRIES, ids=lambda entry: str(entry["mode"]))
def test_a_global_scale_on_a_module_not_declared_nvfp4_is_refused(
    tmp_path: Path, entry: dict[str, object]
) -> None:
    model = build_reference(seed=20)
    save_conversion(model, tmp_path, nvfp4_tensors=quantize_nvfp4(model))
    config = json.loads((tmp_path / "config.json").read_text())
    config["quantization"][f"language_model.{SCALED_MODULE}"] = entry
    (tmp_path / "config.json").write_text(json.dumps(config))
    engine_config = EngineConfig(model_path=str(tmp_path), dtype=mx.float32)
    with pytest.raises(ValueError, match=f"{SCALED_MODULE}: a global scale"):
        create_model(engine_config.architecture, engine_config.model_config, mx.float32)


MTP_NORM = "language_model.mtp.layers.0.input_layernorm.weight"
NESTED_TOWER_NORM = "language_model.vision_tower.blocks.0.norm1.weight"
TOWER_NORM = "vision_tower.blocks.0.norm1.weight"
# (a checkpoint tensor name, its name in the MLX layout; None: left out)
LAYOUT_NAME_CASES: list[tuple[str, str | None]] = [
    ("model.layers.0.mlp.down_proj.weight", "model.layers.0.mlp.down_proj.weight"),
    ("language_model.model.layers.2.mlp.gate.scales", "model.layers.2.mlp.gate.scales"),
    ("language_model.lm_head.weight", "lm_head.weight"),
    ("language_model.model.layers.2.mlp.expert_bias", None),
    (MTP_NORM, MTP_NORM),
    (NESTED_TOWER_NORM, NESTED_TOWER_NORM),
    (TOWER_NORM, TOWER_NORM),
]


@pytest.mark.parametrize(("name", "expected"), LAYOUT_NAME_CASES)
def test_layout_takes_only_the_text_model_out_from_under_language_model(
    name: str, expected: str | None
) -> None:
    """An MTP head or a tower beside the text model keeps its name, so it fails the strict load
    instead of passing for a module of the text model."""
    state, _ = mlx_layout({name: mx.zeros((2, 2))})
    assert list(state) == ([] if expected is None else [expected])


def test_layout_refuses_a_nonzero_router_bias() -> None:
    bias = "language_model.model.layers.2.mlp.expert_bias"
    with pytest.raises(ValueError, match=f"{bias}: a nonzero router expert bias"):
        mlx_layout({bias: mx.array([0.0, 0.5])})


def test_layout_refuses_a_tensor_stored_under_both_names() -> None:
    """Either copy would otherwise replace the other unseen."""
    state = {"model.norm.weight": mx.zeros((2,)), "language_model.model.norm.weight": mx.ones((2,))}
    with pytest.raises(ValueError, match="model.norm.weight: the checkpoint stores it both"):
        mlx_layout(state)


def test_the_hugging_face_layout_is_refused(tmp_path: Path) -> None:
    """The release's bf16 layout, and an export in it, fail by name, not at the strict load."""
    tensor = {"model.layers.2.mlp.experts.0.up_proj.weight": mx.zeros((2, 2))}
    mx.save_safetensors(str(tmp_path / "model.safetensors"), tensor)
    (tmp_path / "config.json").write_text(
        json.dumps({**TINY_FLINT_CONFIG, "quantization_config": {"quant_method": "modelopt"}})
    )
    with pytest.raises(ValueError, match="MLX layout checkpoints only"):
        read_checkpoint("FlintForCausalLM", str(tmp_path))


# (config overrides, the refusal)
REFUSED_CONFIG_CASES: list[tuple[dict[str, object], str]] = [
    ({"layer_types": ["full_attention"]}, "one full_attention or sliding_attention"),
    ({"no_rope_layer_indices": [0, 1]}, "rope on its sliding layers"),
    ({"mlp_only_layers": [1]}, "leading run"),
    ({"decoder_sparse_step": 2}, "MoE block in every layer"),
    ({"router_score_function": "softmax"}, "sigmoid router"),
    ({"scoring_func": "softmax"}, "sigmoid router"),
    ({"router_expert_bias": True}, "expert bias"),
    ({"softmax_type": "default"}, "attention sinks"),
]


@pytest.mark.parametrize(("overrides", "error"), REFUSED_CONFIG_CASES)
def test_config_refuses_what_it_does_not_serve(overrides: dict[str, object], error: str) -> None:
    with pytest.raises(ValueError, match=error):
        parse_config({**TINY_FLINT_CONFIG, **overrides})
