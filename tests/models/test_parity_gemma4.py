"""Numeric parity of the paged engine against mlx-lm on tiny Gemma4 models.

The reference (mlx-lm, a contiguous cache for the global layers and a rotating one for
the window layers) and ftmetal (the FULL pool, the window pool behind its full-to-window
mapping, both attention backends, the scheduler with overlap and decode graphs, the SWA
radix cache) run the same random weights: prefill logits must match and greedy decode must
agree token for token, with the window crossed in prefill, in decode and at chunk seams,
and on prompts that resume a cached prefix. Dense (31B shape) and MoE (26B shape)
variants: 2:1 window:global layers, k_eq_v global layers with proportional rope, logit
softcap, per-expert-scale router; fp32 throughout.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, NamedTuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten
from mlx_lm.models import gemma4 as reference

from ftmetal.attention.full.common import FullAttnBackend
from ftmetal.attention.metadata import AttnType
from ftmetal.core import SamplingParams
from ftmetal.engine.config import EngineConfig
from ftmetal.kvcache.mha_pool import MHAKVCache
from ftmetal.kvcache.swa_pool import SWAKVCache
from ftmetal.scheduler.scheduler import Scheduler

VOCAB = 128
WINDOW = 8
NUM_LAYERS = 6
EOS = VOCAB - 1
TEXT_CONFIG: dict[str, object] = {
    "model_type": "gemma4_text",
    "hidden_size": 64,
    "num_hidden_layers": NUM_LAYERS,
    "intermediate_size": 96,
    "num_attention_heads": 4,
    "head_dim": 64,
    "global_head_dim": 128,
    "num_key_value_heads": 2,
    "num_global_key_value_heads": 1,
    "attention_k_eq_v": True,
    "sliding_window": WINDOW,
    "layer_types": [
        "full_attention" if (i + 1) % 3 == 0 else "sliding_attention" for i in range(NUM_LAYERS)
    ],
    "rms_norm_eps": 1e-6,
    "vocab_size": VOCAB,
    "vocab_size_per_layer_input": VOCAB,
    "tie_word_embeddings": True,
    "final_logit_softcapping": 30.0,
    "max_position_embeddings": 512,
    "rope_parameters": {
        "full_attention": {
            "partial_rotary_factor": 0.25,
            "rope_theta": 1000000.0,
            "rope_type": "proportional",
        },
        "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"},
    },
    # The mlx-lm defaults that are off at the 26B / 31B shapes.
    "num_kv_shared_layers": 0,
    "hidden_size_per_layer_input": 0,
    "use_double_wide_mlp": False,
    "enable_moe_block": False,
}
MOE_CONFIG: dict[str, object] = {
    "enable_moe_block": True,
    "num_experts": 8,
    "top_k_experts": 2,
    "moe_intermediate_size": 32,
}
POST_NORM_SCALE = 8.0
# The QAT checkpoint's recipe: 4-bit by default, the dense MLP and the router at 8 bits.
QUANT_GROUP_SIZE = 32
EIGHT_BIT_MODULES = ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj", "router.proj")


def text_config(moe: bool) -> dict[str, object]:
    return {**TEXT_CONFIG, **(MOE_CONFIG if moe else {})}


def build_reference(seed: int, moe: bool) -> reference.Model:
    mx.random.seed(seed)
    model = reference.Model(
        reference.ModelArgs(model_type="gemma4", text_config=text_config(moe), vocab_size=VOCAB)
    )
    # Random init leaves the norms, scalars and router scales at one: perturb them all. The
    # post norms set each branch's scale against the sqrt(hidden)-scaled embedding; at one
    # the residual stream repeats the last token.
    params: dict[str, mx.array] = {}
    for key, value in tree_flatten(model.parameters()):
        assert isinstance(value, mx.array)
        if key.endswith(("norm.weight", "layer_scalar", "router.scale", "per_expert_scale")):
            value = value + 0.1 * mx.random.normal(value.shape)
        if ".post_" in key:
            value = POST_NORM_SCALE * value
        params[key] = value
    model.load_weights(list(params.items()))
    model.set_dtype(mx.float32)
    mx.eval(model.parameters())
    return model


def quantize(model: reference.Model) -> dict[str, object]:
    """Quantize the reference as the QAT recipe does; returns the config's quantization map."""
    overrides: dict[str, object] = {}

    def predicate(path: str, module: nn.Module) -> bool | dict[str, int]:
        if not hasattr(module, "to_quantized"):
            return False
        if path.endswith(EIGHT_BIT_MODULES):
            overrides[path] = {"group_size": QUANT_GROUP_SIZE, "bits": 8}
            return {"group_size": QUANT_GROUP_SIZE, "bits": 8}
        return True

    nn.quantize(model, group_size=QUANT_GROUP_SIZE, bits=4, class_predicate=predicate)
    mx.eval(model.parameters())
    return {"group_size": QUANT_GROUP_SIZE, "bits": 4, "mode": "affine", **overrides}


def save_checkpoint(
    model: reference.Model, path: Path, moe: bool, quantization: dict[str, object] | None
) -> None:
    """The MLX layout of the multimodal checkpoint: the text model under language_model."""
    mx.save_safetensors(str(path / "model.safetensors"), dict(tree_flatten(model.parameters())))
    config: dict[str, object] = {
        "architectures": ["Gemma4ForConditionalGeneration"],
        "model_type": "gemma4",
        "text_config": text_config(moe),
        "eos_token_id": [EOS],
    }
    if quantization is not None:
        config["quantization"] = quantization
    (path / "config.json").write_text(json.dumps(config))


def reference_greedy(
    model: reference.Model, prompt: list[int], steps: int
) -> tuple[mx.array, list[int]]:
    cache = model.make_cache()
    logits = model(mx.array(prompt)[None], cache=cache)
    prefill_logits = logits[0]
    token = int(mx.argmax(logits[0, -1]))
    tokens = [token]
    for _ in range(steps - 1):
        logits = model(mx.array([[token]]), cache=cache)
        token = int(mx.argmax(logits[0, -1]))
        tokens.append(token)
    return prefill_logits, tokens


def make_scheduler(
    path: Path,
    page_size: int,
    attention_backend: str,
    max_extend_tokens: int = 64,
    cache_type: Literal["naive", "radix"] = "radix",
    num_pages: int = 64,
) -> Scheduler:
    config = EngineConfig(
        model_path=str(path),
        dtype=mx.float32,
        max_running_req=2,
        page_size=page_size,
        attention_backend=attention_backend,
        num_page_override=num_pages,
        max_extend_tokens=max_extend_tokens,
        cache_type=cache_type,
    )
    return Scheduler(config, eos_token_ids={EOS})


def random_prompt(seed: int, length: int) -> list[int]:
    return [int(t) for t in np.random.default_rng(seed).integers(1, EOS, size=length)]


def generate(scheduler: Scheduler, prompts: list[list[int]], steps: int) -> list[list[int]]:
    params = SamplingParams(temperature=0.0, max_tokens=steps, ignore_eos=True)
    for uid, prompt in enumerate(prompts):
        scheduler.add_request(uid, np.array(prompt, dtype=np.int32), params)
    events = scheduler.run_until_idle()
    scheduler.cache_manager.check_integrity()
    return [[event.token for event in events if event.uid == uid] for uid in range(len(prompts))]


BACKENDS_AND_PAGES = [("mlx", 1), ("metal", 1), ("metal", 4)]
# (prompt seed, prompt length, decode steps): inside the window, across it in prefill,
# across it in decode.
GREEDY_CASES = [(0, 4, 6), (1, 23, 6), (2, 3, 20)]


@pytest.mark.parametrize("moe", [False, True], ids=["dense", "moe"])
@pytest.mark.parametrize(("attention_backend", "page_size"), BACKENDS_AND_PAGES)
def test_greedy_decode_matches_mlx_lm(
    tmp_path: Path, attention_backend: str, page_size: int, moe: bool
) -> None:
    model = build_reference(seed=11, moe=moe)
    save_checkpoint(model, tmp_path, moe, quantization=None)
    scheduler = make_scheduler(tmp_path, page_size, attention_backend)
    for seed, length, steps in GREEDY_CASES:
        prompt = random_prompt(seed, length)
        _, expected = reference_greedy(model, prompt, steps)
        assert generate(scheduler, [prompt], steps) == [expected], (seed, length)


@pytest.mark.parametrize("moe", [False, True], ids=["dense", "moe"])
@pytest.mark.parametrize("attention_backend", ["mlx", "metal"])
def test_two_requests_batched_match_single(
    tmp_path: Path, attention_backend: str, moe: bool
) -> None:
    model = build_reference(seed=12, moe=moe)
    save_checkpoint(model, tmp_path, moe, quantization=None)
    prompts = [random_prompt(3, 11), random_prompt(4, 5)]
    steps = 12
    expected = [reference_greedy(model, prompt, steps)[1] for prompt in prompts]
    scheduler = make_scheduler(tmp_path, 1, attention_backend)
    assert generate(scheduler, prompts, steps) == expected


@pytest.mark.parametrize("moe", [False, True], ids=["dense", "moe"])
@pytest.mark.parametrize(("attention_backend", "page_size"), BACKENDS_AND_PAGES)
def test_chunked_prefill_across_the_window_matches_mlx_lm(
    tmp_path: Path, attention_backend: str, page_size: int, moe: bool
) -> None:
    model = build_reference(seed=13, moe=moe)
    save_checkpoint(model, tmp_path, moe, quantization=None)
    prompt = random_prompt(5, 29)
    steps = 6
    _, expected = reference_greedy(model, prompt, steps)
    # Chunks of 12 put window starts inside chunks and windows across chunk seams.
    scheduler = make_scheduler(tmp_path, page_size, attention_backend, max_extend_tokens=12)
    assert generate(scheduler, [prompt], steps) == [expected]


@pytest.mark.parametrize("moe", [False, True], ids=["dense", "moe"])
@pytest.mark.parametrize("attention_backend", ["mlx", "metal"])
def test_prefill_logits_match_mlx_lm(tmp_path: Path, attention_backend: str, moe: bool) -> None:
    model = build_reference(seed=14, moe=moe)
    save_checkpoint(model, tmp_path, moe, quantization=None)
    prompt = random_prompt(6, 21)
    expected_logits, _ = reference_greedy(model, prompt, 1)
    scheduler = make_scheduler(tmp_path, 1, attention_backend)
    scheduler.add_request(0, np.array(prompt, dtype=np.int32), SamplingParams(max_tokens=1))
    forward_input = scheduler.schedule_next_batch()
    assert forward_input is not None
    logits = scheduler.engine.model.forward(forward_input.batch)
    assert mx.allclose(logits[0], expected_logits[-1], atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("attention_backend", ["mlx", "metal"])
def test_quantized_greedy_decode_matches_mlx_lm(tmp_path: Path, attention_backend: str) -> None:
    model = build_reference(seed=15, moe=True)
    quantization = quantize(model)
    save_checkpoint(model, tmp_path, moe=True, quantization=quantization)
    prompt = random_prompt(7, 19)
    steps = 16
    _, expected = reference_greedy(model, prompt, steps)
    scheduler = make_scheduler(tmp_path, 1, attention_backend)
    assert generate(scheduler, [prompt], steps) == [expected]


def test_engine_builds_a_pool_and_a_backend_per_attention_type(tmp_path: Path) -> None:
    save_checkpoint(build_reference(seed=16, moe=True), tmp_path, moe=True, quantization=None)
    engine = make_scheduler(tmp_path, 4, "metal").engine
    full_pool, swa_pool = engine.pools[AttnType.FULL], engine.pools[AttnType.SWA]
    assert set(engine.pools) == {AttnType.FULL, AttnType.SWA}
    assert type(full_pool) is MHAKVCache and isinstance(swa_pool, SWAKVCache)
    assert swa_pool.layer_ids == (0, 1, 3, 4) and full_pool.layer_ids == (2, 5)
    assert full_pool.k_cache(2).shape[1:] == (1, 128) and swa_pool.k_cache(0).shape[1:] == (2, 64)
    full_backend, swa_backend = (
        engine.attn_backends[AttnType.FULL],
        engine.attn_backends[AttnType.SWA],
    )
    assert isinstance(full_backend, FullAttnBackend) and isinstance(swa_backend, FullAttnBackend)
    assert (full_backend.kvcache, swa_backend.kvcache) == (full_pool, swa_pool)
    assert (full_backend.sliding_window, swa_backend.sliding_window) == (None, WINDOW)
    assert full_backend.scale == swa_backend.scale == 1.0
    assert engine.graph_runner.graphs, "decode graphs captured over both pools"


def run_alone(
    scheduler: Scheduler, uid: int, prompt: list[int], steps: int
) -> tuple[int, list[int]]:
    """One prompt to completion; returns the prefix its admission found cached and its tokens."""
    params = SamplingParams(temperature=0.0, max_tokens=steps, ignore_eos=True)
    scheduler.add_request(uid, np.array(prompt, dtype=np.int32), params)
    forward_input = scheduler.schedule_next_batch()
    assert forward_input is not None
    cached_len = forward_input.batch.reqs[0].cached_len
    data = scheduler.forward(forward_input)
    while scheduler.runnable or data is not None:
        data = scheduler.overlap_loop(data)
    scheduler.cache_manager.check_integrity()
    return cached_len, [event.token for event in scheduler.take_results()]


def hit_prefill_logits(scheduler: Scheduler, prompt: list[int]) -> tuple[int, mx.array]:
    """Admit prompt and run its prefill forward alone; returns the prefix its admission found
    cached and the last position's logits. Leaves the scheduler mid-request."""
    scheduler.add_request(99, np.array(prompt, dtype=np.int32), SamplingParams(max_tokens=1))
    forward_input = scheduler.schedule_next_batch()
    assert forward_input is not None
    req = forward_input.batch.reqs[0]
    assert req.device_len == len(prompt), "the rest of the prompt must fit one chunk"
    return req.cached_len, scheduler.engine.model.forward(forward_input.batch)[0]


class PrefixCase(NamedTuple):
    label: str
    attention_backend: str
    page_size: int
    max_extend_tokens: int
    # Served alone, one after another; then the last one again, whose prefill on the tree's
    # prefix is held to mlx-lm's logits.
    prompts: list[list[int]]
    # The prefix each admission finds cached.
    cached_lens: list[int]
    cache_type: Literal["naive", "radix"] = "radix"
    num_pages: int = 64


HEAD = random_prompt(8, 29)
PREFIX_CASES = [
    PrefixCase(
        "the same one-chunk prompt resumes all but its last token",
        "mlx",
        1,
        64,
        [HEAD[:23], HEAD[:23]],
        [0, 22],
    ),
    PrefixCase(
        "a chunked prompt resumes from the live window of its last chunk",
        "metal",
        1,
        12,
        [HEAD, HEAD],
        [0, 28],
    ),
    PrefixCase(
        "a shared head of a one-chunk prompt resumes at a page boundary",
        "metal",
        4,
        64,
        [HEAD[:23], HEAD[:17] + random_prompt(9, 10)],
        [0, 16],
    ),
    PrefixCase(
        "a head behind the window is recomputed, revived and served to the next prompt",
        "metal",
        1,
        12,
        [
            HEAD,
            HEAD[:10] + random_prompt(10, 5),
            HEAD[:10] + random_prompt(10, 5) + random_prompt(11, 8),
        ],
        [0, 0, 15],
    ),
    PrefixCase(
        "naive resumes nothing",
        "metal",
        4,
        64,
        [HEAD, HEAD],
        [0, 0],
        cache_type="naive",
    ),
]


@pytest.mark.parametrize("moe", [False, True], ids=["dense", "moe"])
def test_prompts_resuming_a_cached_prefix_match_mlx_lm(tmp_path: Path, moe: bool) -> None:
    model = build_reference(seed=17, moe=moe)
    save_checkpoint(model, tmp_path, moe, quantization=None)
    steps = 6
    for case in PREFIX_CASES:
        scheduler = make_scheduler(
            tmp_path,
            case.page_size,
            case.attention_backend,
            case.max_extend_tokens,
            case.cache_type,
            case.num_pages,
        )
        assert scheduler.cache_manager.cache_type == (
            "swa_radix" if case.cache_type == "radix" else "naive"
        )
        for uid, (prompt, cached_len) in enumerate(zip(case.prompts, case.cached_lens)):
            expected = reference_greedy(model, prompt, steps)[1]
            assert run_alone(scheduler, uid, prompt, steps) == (cached_len, expected), case.label
        hit_len, logits = hit_prefill_logits(scheduler, case.prompts[-1])
        assert (hit_len > 0) == (case.cache_type == "radix"), case.label
        expected_logits = reference_greedy(model, case.prompts[-1], 1)[0][-1]
        assert mx.allclose(logits, expected_logits, atol=1e-4, rtol=1e-4), case.label


def test_next_turn_resumes_past_the_decoded_tokens(tmp_path: Path) -> None:
    model = build_reference(seed=18, moe=True)
    save_checkpoint(model, tmp_path, moe=True, quantization=None)
    scheduler = make_scheduler(tmp_path, 1, "metal", max_extend_tokens=12)
    first = random_prompt(12, 23)
    _, tokens = run_alone(scheduler, 0, first, 6)
    assert tokens == reference_greedy(model, first, 6)[1]
    second = first + tokens[:5] + random_prompt(13, 7)
    expected = reference_greedy(model, second, 6)[1]
    assert run_alone(scheduler, 1, second, 6) == (28, expected)
