"""Numeric parity of the paged engine against mlx-lm on tiny Gemma4 models.

The reference (mlx-lm, a contiguous cache for the global layers and a rotating one for
the window layers) and ftmetal (the FULL pool, the window pool behind its full-to-window
mapping, both attention backends, the scheduler with overlap and decode graphs, the SWA
radix cache) run the same random weights: prefill logits must match and greedy decode must
agree token for token, with the window crossed in prefill, in decode and at chunk seams,
and on prompts that resume a cached prefix. Dense (31B shape) and MoE (26B shape)
variants: 2:1 window:global layers, k_eq_v global layers with proportional rope, logit
softcap, per-expert-scale router; fp32 throughout. The same weights in the Hugging Face
layout (multimodal or text-only, the experts in the bf16 release's two banks or one tensor
per expert, an untied lm_head) give the MLX layout's prefill logits bit for bit; a ModelOpt
or compressed-tensors export (the real exports' recipes and FP8 beside NVFP4, written by
tiny_model.quantize_export) serves in fp32 like mlx-lm on its dequantized weights, and in
bf16 as the same tensors under the MLX layout's names, bit for bit. The MoE variant served from
an offload cache that misses and evicts every step decodes the resident experts' greedy tokens
and its last prefill chunk's logits match theirs, in every format above and from an expert pack:
its decode replayed, on the pending kernels (split lookup) when affine, one and two
requests, looked up or read through the prefill buffers, with a victim cache; a prompt resuming
another's cached head on the slots it left does too, on every forward.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import replace
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
from ftmetal.cli import run_pack
from ftmetal.core import SamplingParams
from ftmetal.engine.config import EngineConfig
from ftmetal.engine.sample import BatchSamplingArgs
from ftmetal.kvcache.mha_pool import MHAKVCache
from ftmetal.kvcache.swa_pool import SWAKVCache
from ftmetal.layers import moe as moe_layers
from ftmetal.layers.quantization.moe import AffineMoEMethod, SharedExpertForward
from ftmetal.layers.quantization.moe_pending import PendingOutputs
from ftmetal.models.gemma4.weight import CONDITIONAL_GENERATION, mlx_layout
from ftmetal.models.loader import ExpertSlices
from ftmetal.moe.offload_cache import IssuedLookup, OffloadMoeCache, ResidentFirst
from ftmetal.scheduler.scheduler import Scheduler
from tiny_model import EXPORT_CONFIGS, quantize_export

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
# The recipe in each mode, (group size, bits by default, the 8-bit modules' entry): a float
# mode keeps the 8-bit modules in mxfp8 through entries that name it.
QUANT_RECIPES: dict[str, tuple[int, int, dict[str, object]]] = {
    "affine": (QUANT_GROUP_SIZE, 4, {"group_size": QUANT_GROUP_SIZE, "bits": 8}),
    "mxfp4": (32, 4, {"group_size": 32, "bits": 8, "mode": "mxfp8"}),
    "nvfp4": (16, 4, {"group_size": 32, "bits": 8, "mode": "mxfp8"}),
}


def text_config(moe: bool, tie_word_embeddings: bool = True) -> dict[str, object]:
    return {
        **TEXT_CONFIG,
        **(MOE_CONFIG if moe else {}),
        "tie_word_embeddings": tie_word_embeddings,
    }


def build_reference(seed: int, moe: bool, tie_word_embeddings: bool = True) -> reference.Model:
    mx.random.seed(seed)
    model = reference.Model(
        reference.ModelArgs(
            model_type="gemma4",
            text_config=text_config(moe, tie_word_embeddings),
            vocab_size=VOCAB,
        )
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


def quantize(
    model: reference.Model,
    mode: str = "affine",
    eight_bit_modules: tuple[str, ...] = EIGHT_BIT_MODULES,
) -> dict[str, object]:
    """Quantize the reference as the QAT recipe does in mode; returns the config's
    quantization map."""
    group_size, bits, eight_bit = QUANT_RECIPES[mode]
    overrides: dict[str, object] = {}

    def predicate(path: str, module: nn.Module) -> bool | dict[str, object]:
        if not hasattr(module, "to_quantized"):
            return False
        if path.endswith(eight_bit_modules):
            overrides[path] = eight_bit
            return dict(eight_bit)
        return True

    nn.quantize(model, group_size, bits, mode=mode, class_predicate=predicate)
    mx.eval(model.parameters())
    return {"group_size": group_size, "bits": bits, "mode": mode, **overrides}


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
        # mlx-lm writes the section twice, the second time as quantization_config.
        config["quantization"] = quantization
        config["quantization_config"] = quantization
    (path / "config.json").write_text(json.dumps(config))


def hf_layout_weights(
    weights: dict[str, mx.array], architecture: str, *, has_expert_banks: bool
) -> dict[str, mx.array]:
    """MLX layout weights (the reference's and a tower's, or an export's tensors under their
    module names) as a Hugging Face checkpoint of architecture stores them: a multimodal one
    nests the text model under model.language_model. and the towers under model., lm_head
    stays at the top; each layer's experts in a gate_up and a down bank (the bf16 release) or
    one tensor per expert (an export)."""
    text_prefix = "model.language_model." if architecture == CONDITIONAL_GENERATION else "model."
    out: dict[str, mx.array] = {}
    for mlx_key, value in weights.items():
        if mlx_key.startswith("language_model.model."):
            key = text_prefix + mlx_key.removeprefix("language_model.model.")
        elif mlx_key.startswith("language_model."):
            key = mlx_key.removeprefix("language_model.")
        else:
            key = f"model.{mlx_key}"
        parent, _, leaf = key.partition(".experts.switch_glu.")
        if leaf and has_expert_banks:
            if leaf == "gate_proj.weight":
                up = weights[mlx_key.replace(".gate_proj.", ".up_proj.")]
                out[f"{parent}.experts.gate_up_proj"] = mx.concatenate([value, up], axis=-2)
            elif leaf == "down_proj.weight":
                out[f"{parent}.experts.down_proj"] = value
            else:
                assert leaf == "up_proj.weight", f"{mlx_key}: banked experts are bf16"
        elif leaf:
            for expert_id in range(value.shape[0]):
                out[f"{parent}.experts.{expert_id}.{leaf}"] = value[expert_id]
        else:
            out[key] = value
    return out


def save_weights(
    weights: dict[str, mx.array],
    model: reference.Model,
    path: Path,
    architecture: str,
    quantization_config: dict[str, object] | None = None,
) -> None:
    """weights, in either layout, with the config of model under architecture."""
    path.mkdir()
    mx.save_safetensors(str(path / "model.safetensors"), weights)
    config: dict[str, object] = {
        "architectures": [architecture],
        "model_type": "gemma4",
        "text_config": model.args.text_config,
        "eos_token_id": [EOS],
    }
    if quantization_config is not None:
        config["quantization_config"] = quantization_config
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
    dtype: mx.Dtype = mx.float32,
) -> Scheduler:
    config = EngineConfig(
        model_path=str(path),
        dtype=dtype,
        max_running_req=2,
        page_size=page_size,
        attention_backend=attention_backend,
        num_page_override=64,
        max_extend_tokens_override=max_extend_tokens,
        cache_type=cache_type,
    )
    return Scheduler(config, eos_token_ids={EOS})


def random_prompt(seed: int, length: int) -> list[int]:
    return [int(t) for t in np.random.default_rng(seed).integers(1, EOS, size=length)]


def prefill_logits(scheduler: Scheduler, prompt: list[int]) -> mx.array:
    """The last position's logits of a one-chunk prompt's prefill; leaves the scheduler
    mid-request."""
    scheduler.add_request(0, np.array(prompt, dtype=np.int32), SamplingParams(max_tokens=1))
    forward_input = scheduler.schedule_next_batch()
    assert forward_input is not None
    return scheduler.engine.model.forward(forward_input.batch)[0]


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
    logits = prefill_logits(make_scheduler(tmp_path, 1, attention_backend), prompt)
    assert mx.allclose(logits, expected_logits[-1], atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("attention_backend", ["mlx", "metal"])
@pytest.mark.parametrize("mode", list(QUANT_RECIPES))
def test_quantized_greedy_decode_matches_mlx_lm(
    tmp_path: Path, attention_backend: str, mode: str
) -> None:
    model = build_reference(seed=15, moe=True)
    quantization = quantize(model, mode)
    save_checkpoint(model, tmp_path, moe=True, quantization=quantization)
    prompt = random_prompt(7, 19)
    steps = 16
    _, expected = reference_greedy(model, prompt, steps)
    scheduler = make_scheduler(tmp_path, 1, attention_backend)
    assert generate(scheduler, [prompt], steps) == [expected]


class HfLayout(NamedTuple):
    architecture: str
    moe: bool
    has_expert_banks: bool
    tie_word_embeddings: bool = True
    # The checkpoint stores lm_head.weight though it is tied to the embeddings.
    stores_tied_lm_head: bool = False


HF_LAYOUTS = {
    "bf16-release": HfLayout(CONDITIONAL_GENERATION, True, True),
    "per-expert": HfLayout(CONDITIONAL_GENERATION, True, False),
    "text-only": HfLayout("Gemma4ForCausalLM", True, True),
    "dense-untied": HfLayout(CONDITIONAL_GENERATION, False, False, tie_word_embeddings=False),
    "dense-tied-head-stored": HfLayout(
        CONDITIONAL_GENERATION, False, False, stores_tied_lm_head=True
    ),
    "text-only-tied-head-stored": HfLayout(
        "Gemma4ForCausalLM", True, True, stores_tied_lm_head=True
    ),
}


@pytest.mark.parametrize("layout", list(HF_LAYOUTS))
def test_hugging_face_layout_matches_the_mlx_layout(tmp_path: Path, layout: str) -> None:
    case = HF_LAYOUTS[layout]
    model = build_reference(seed=20, moe=case.moe, tie_word_embeddings=case.tie_word_embeddings)
    weights = dict(tree_flatten(model.parameters()))
    save_weights(weights, model, tmp_path / "mlx", CONDITIONAL_GENERATION)
    hf_weights = hf_layout_weights(
        weights, case.architecture, has_expert_banks=case.has_expert_banks
    )
    if case.stores_tied_lm_head:
        hf_weights["lm_head.weight"] = weights["language_model.model.embed_tokens.weight"]
    save_weights(hf_weights, model, tmp_path / "hf", case.architecture)
    prompt = random_prompt(20, 21)
    steps = 12
    expected_logits = prefill_logits(make_scheduler(tmp_path / "mlx", 1, "metal"), prompt)
    logits = prefill_logits(make_scheduler(tmp_path / "hf", 1, "metal"), prompt)
    assert mx.array_equal(logits, expected_logits)
    _, expected = reference_greedy(model, prompt, steps)
    assert generate(make_scheduler(tmp_path / "hf", 1, "metal"), [prompt], steps) == [expected]


def test_mlx_layout_stacks_experts_in_numeric_order() -> None:
    """Twelve experts, shuffled: text order (10 before 2) or insertion order would show."""
    num_experts = 12
    layer = "model.language_model.layers.0.experts"
    state = {
        f"{layer}.{expert}.gate_proj.weight": mx.full((2, 4), expert, dtype=mx.float32)
        for expert in np.random.default_rng(0).permutation(num_experts).tolist()
    }
    out, expert_bytes = mlx_layout(state)
    bank = "language_model.model.layers.0.experts.switch_glu.gate_proj.weight"
    assert np.array(out[bank])[:, 0, 0].tolist() == list(range(num_experts))
    slices = expert_bytes[bank]
    assert isinstance(slices, ExpertSlices)
    assert slices.names == tuple(
        f"{layer}.{expert}.gate_proj.weight" for expert in range(num_experts)
    )


def test_mlx_layout_on_experts_without_a_tensor() -> None:
    """An export may omit some experts' input_scale, which nothing reads; not their weights."""
    layer = "model.language_model.layers.0.experts"
    state = {
        key: mx.ones((2, 4), dtype=mx.float32)
        for expert in range(3)
        for key in (f"{layer}.{expert}.gate_proj.weight", f"{layer}.{expert}.gate_proj.input_scale")
    }
    state.pop(f"{layer}.1.gate_proj.input_scale")
    out, _ = mlx_layout(state)
    switch_glu = "language_model.model.layers.0.experts.switch_glu.gate_proj"
    assert f"{switch_glu}.weight" in out and f"{switch_glu}.input_scale" not in out
    state.pop(f"{layer}.1.gate_proj.weight")
    with pytest.raises(ValueError, match=rf"{switch_glu}\.weight: .* experts \[1\]"):
        mlx_layout(state)


class ExportCase(NamedTuple):
    dialect: str
    moe: bool
    # The modules (MLX layout names) stored FP8 and NVFP4.
    formats: tuple[str, str]
    quantization_config: dict[str, object]


EXPERT_BANKS = r"\.experts\.switch_glu\.(gate|up|down)_proj$"
DENSE_MLP = r"\.mlp\.(gate|up|down)_proj$"
ATTENTION = r"\.self_attn\.[qkvo]_proj$"
# No module name is empty: an export without FP8 modules.
NO_MODULES = r"^$"
# The real exports' config: everything it does not exclude is NVFP4.
MODELOPT_NVFP4: dict[str, object] = {"quant_method": "modelopt", "quant_algo": "NVFP4"}
EXPORT_CASES = {
    # Gemma-4-26B-A4B-NVFP4: the experts NVFP4, attention, the dense MLP and the router bf16.
    "modelopt-26b": ExportCase("modelopt", True, (NO_MODULES, EXPERT_BANKS), MODELOPT_NVFP4),
    # Gemma-4-31B-IT-NVFP4: the MLP NVFP4, attention bf16.
    "modelopt-31b": ExportCase("modelopt", False, (NO_MODULES, DENSE_MLP), MODELOPT_NVFP4),
    # FP8 attention (a k_eq_v layer's q and k fused) and dense MLP beside NVFP4 experts.
    "modelopt-fp8": ExportCase(
        "modelopt", True, (f"{ATTENTION}|{DENSE_MLP}", EXPERT_BANKS), EXPORT_CONFIGS["modelopt"]
    ),
    "compressed-tensors": ExportCase(
        "compressed-tensors",
        True,
        (ATTENTION, f"{EXPERT_BANKS}|{DENSE_MLP}"),
        EXPORT_CONFIGS["compressed-tensors"],
    ),
    # A 26B export NVFP4 throughout, so a k_eq_v layer's fused qk_proj too.
    "compressed-tensors-26b": ExportCase(
        "compressed-tensors",
        True,
        (NO_MODULES, f"{ATTENTION}|{EXPERT_BANKS}|{DENSE_MLP}"),
        EXPORT_CONFIGS["compressed-tensors"],
    ),
}
EXPORT_PARITY_CASES = [
    ("modelopt-26b", "metal"),
    ("modelopt-26b", "mlx"),
    ("modelopt-31b", "metal"),
    ("modelopt-fp8", "metal"),
    ("compressed-tensors", "mlx"),
    ("compressed-tensors-26b", "metal"),
    ("compressed-tensors-26b", "mlx"),
]


def save_export(
    model: reference.Model,
    case: ExportCase,
    path: Path,
    dtype: mx.Dtype,
    seed: int,
    *,
    is_hf_layout: bool = True,
) -> dict[str, mx.array]:
    """Write the reference as the case's export, its other tensors in dtype, in the Hugging Face
    layout or, without is_hf_layout, under the MLX layout's names; returns the dequantized
    weights by reference key."""
    stored, dequantized = quantize_export(
        dict(tree_flatten(model.parameters())), case.dialect, case.formats, dtype, seed
    )
    if is_hf_layout:
        weights = hf_layout_weights(stored, CONDITIONAL_GENERATION, has_expert_banks=False)
    else:
        weights = stored
    save_weights(weights, model, path, CONDITIONAL_GENERATION, case.quantization_config)
    return dequantized


@pytest.mark.parametrize(("export", "attention_backend"), EXPORT_PARITY_CASES)
def test_export_matches_its_dequantized_weights(
    tmp_path: Path, export: str, attention_backend: str
) -> None:
    case = EXPORT_CASES[export]
    model = build_reference(seed=21, moe=case.moe)
    model.load_weights(list(save_export(model, case, tmp_path / "export", mx.float32, 21).items()))
    prompt = random_prompt(21, 21)
    steps = 12
    expected_logits, expected = reference_greedy(model, prompt, steps)
    scheduler = make_scheduler(tmp_path / "export", 1, attention_backend)
    logits = prefill_logits(scheduler, prompt)
    assert mx.allclose(logits, expected_logits[-1], atol=1e-4, rtol=1e-4)
    scheduler = make_scheduler(tmp_path / "export", 1, attention_backend)
    assert generate(scheduler, [prompt], steps) == [expected]


@pytest.mark.parametrize("export", list(EXPORT_CASES))
def test_export_serves_in_bf16(tmp_path: Path, export: str) -> None:
    """The output scales are float32, the forward around them stays in bf16, and the Hugging
    Face layout gives the logits of the same tensors under the MLX layout's names bit for bit.
    The tiny model's bf16 logits stray too far from fp32 (up to 1.2 for a plain bf16
    checkpoint) for a tolerance to catch a quantization error, so the numerics are the fp32
    test's."""
    case = EXPORT_CASES[export]
    model = build_reference(seed=22, moe=case.moe)
    save_export(model, case, tmp_path / "hf", mx.bfloat16, 22)
    save_export(model, case, tmp_path / "mlx", mx.bfloat16, 22, is_hf_layout=False)
    prompt = random_prompt(22, 21)
    expected_logits = prefill_logits(
        make_scheduler(tmp_path / "mlx", 1, "metal", dtype=mx.bfloat16), prompt
    )
    logits = prefill_logits(make_scheduler(tmp_path / "hf", 1, "metal", dtype=mx.bfloat16), prompt)
    assert logits.dtype == mx.bfloat16
    assert mx.array_equal(logits, expected_logits)


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


def test_too_few_windows_fail_at_startup_and_naive_ignores_the_count(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    save_checkpoint(build_reference(seed=19, moe=False), tmp_path, moe=False, quantization=None)
    config = EngineConfig(
        model_path=str(tmp_path),
        dtype=mx.float32,
        max_running_req=2,
        num_page_override=64,
        max_extend_tokens_override=64,
        swa_windows=24,
    )
    # Two running requests and two chunks hold 2 x (8 + 16 + 8 + 3) + 2 x 64 = 198 window
    # slots: 25 windows of 8.
    with pytest.raises(ValueError, match="--swa-windows 25 or more"):
        Scheduler(config, eos_token_ids={EOS})
    with caplog.at_level(logging.WARNING, logger="ftmetal.scheduler.scheduler"):
        scheduler = Scheduler(replace(config, cache_type="naive"), eos_token_ids={EOS})
    assert "--swa-windows" in caplog.text
    pool = scheduler.engine.ctx.swa_pool
    assert pool is not None and pool.swa_num_tokens == 3 * 96  # 3 x align32(8 + 64 + 1)


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
        "a head within a prompt's last window and gap resumes at a page boundary",
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


# How an offload case's checkpoint stores its experts: an MLX mode (QUANT_RECIPES), unquantized in
# the MLX or the Hugging Face layout (two banks, or one tensor per expert), or an export
# (EXPORT_CASES).
class OffloadCase(NamedTuple):
    name: str
    prompt_lens: tuple[int, ...]
    expert_format: str = "affine"
    # Prompts of this many tokens or more prefill through the prefill buffers.
    overlap_min_tokens: int = 128
    # An affine decode issues its lookups (moe_split_lookup).
    split_lookup: bool = True
    # The offload's victim cache (0: none).
    victim_cache_gib: float = 0
    # Offload reads an expert pack of the checkpoint (ftm pack), whose affine scales are pair codes.
    is_packed: bool = False


# 30 slots of the 48 experts: two decode rows, the prefill buffers over the first 16. Chunks of
# 64: a 100-token prompt reads its first chunk through the buffers and looks the second up.
OFFLOAD_CACHE_SIZE = 30
OFFLOAD_STEPS = 10
OFFLOAD_CASES = [
    OffloadCase("replayed-decode-copy-miss-prefill", (9,)),
    OffloadCase("replayed-decode-split-off", (9,), split_lookup=False),
    OffloadCase("chunks-across-the-threshold", (100, 9), overlap_min_tokens=48),
    OffloadCase("chunked-overlap-prefill", (150,), overlap_min_tokens=16),
    OffloadCase("unquantized", (100, 9), "unquantized", overlap_min_tokens=48),
    OffloadCase("mxfp4", (100, 9), "mxfp4", overlap_min_tokens=48),
    OffloadCase("nvfp4", (100, 9), "nvfp4", overlap_min_tokens=48),
    OffloadCase("hf-banks", (100, 9), "hf-banks", overlap_min_tokens=48),
    OffloadCase("hf-per-expert", (100, 9), "hf-per-expert", overlap_min_tokens=48),
    OffloadCase("modelopt-26b", (100, 9), "modelopt-26b", overlap_min_tokens=48),
    OffloadCase("modelopt-fp8", (100, 9), "modelopt-fp8", overlap_min_tokens=48),
    OffloadCase("compressed-tensors", (100, 9), "compressed-tensors", overlap_min_tokens=48),
    OffloadCase("victim-cache", (100, 9), overlap_min_tokens=48, victim_cache_gib=8e-5),
    OffloadCase("packed", (100, 9), overlap_min_tokens=48, is_packed=True),
    OffloadCase("one-layer-up-8-bit", (100, 9), "affine-up-8-bit", overlap_min_tokens=48),
    OffloadCase("modelopt-26b-packed", (100, 9), "modelopt-26b", 48, is_packed=True),
]


def write_offload_checkpoint(model: reference.Model, path: Path, expert_format: str) -> None:
    if expert_format == "unquantized":
        path.mkdir()
        save_checkpoint(model, path, moe=True, quantization=None)
    elif expert_format in QUANT_RECIPES:
        path.mkdir()
        save_checkpoint(model, path, moe=True, quantization=quantize(model, expert_format))
    elif expert_format == "affine-up-8-bit":
        path.mkdir()
        modules = (*EIGHT_BIT_MODULES, "layers.2.experts.switch_glu.up_proj")
        quantization = quantize(model, "affine", modules)
        save_checkpoint(model, path, moe=True, quantization=quantization)
    elif expert_format in EXPORT_CASES:
        save_export(model, EXPORT_CASES[expert_format], path, mx.float32, seed=21)
    else:
        weights = hf_layout_weights(
            dict(tree_flatten(model.parameters())),
            CONDITIONAL_GENERATION,
            has_expert_banks=expert_format == "hf-banks",
        )
        save_weights(weights, model, path, CONDITIONAL_GENERATION)


def make_offload_scheduler(path: Path, case: OffloadCase, offload: bool) -> Scheduler:
    config = EngineConfig(
        model_path=str(path),
        dtype=mx.float32,
        max_running_req=2,
        page_size=4,
        num_page_override=128,
        max_extend_tokens_override=64,
        moe_strategy="offload" if offload else "fused",
        moe_cache_size=OFFLOAD_CACHE_SIZE if offload else 0,
        moe_split_lookup=case.split_lookup,
        moe_victim_cache_gib=case.victim_cache_gib if offload else 0,
    )
    return Scheduler(config, eos_token_ids={EOS})


def serve_offload(
    path: Path, case: OffloadCase, offload: bool, prompts: list[list[int]]
) -> tuple[mx.array, list[list[int]], Scheduler]:
    """The logits of the first prompt's last prefill chunk, every prompt's greedy tokens
    decoded together, and the scheduler that decoded them."""
    scheduler = make_offload_scheduler(path, case, offload)
    scheduler.add_request(0, np.array(prompts[0], dtype=np.int32), SamplingParams(max_tokens=1))
    while True:
        forward_input = scheduler.schedule_next_batch()
        assert forward_input is not None
        if forward_input.batch.reqs[0].can_decode:
            logits = scheduler.engine.model.forward(forward_input.batch)
            mx.eval(logits)
            break
        scheduler.process_last_data(scheduler.forward(forward_input))
    scheduler = make_offload_scheduler(path, case, offload)
    return logits, generate(scheduler, prompts, OFFLOAD_STEPS), scheduler


@pytest.mark.parametrize("case", OFFLOAD_CASES, ids=lambda case: case.name)
def test_offload_matches_resident_experts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: OffloadCase
) -> None:
    monkeypatch.setenv("FTMETAL_MOE_PREFILL_OVERLAP_MIN_TOKENS", str(case.overlap_min_tokens))
    path = tmp_path / "checkpoint"
    write_offload_checkpoint(build_reference(seed=20, moe=True), path, case.expert_format)
    if case.is_packed:
        assert run_pack(["--model", str(path)]) == 0
    rng = np.random.default_rng(len(case.name))
    prompts = [[int(t) for t in rng.integers(1, EOS, size=n)] for n in case.prompt_lens]
    resident_logits, resident_tokens, _ = serve_offload(path, case, False, prompts)
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
    logits, tokens, scheduler = serve_offload(path, case, True, prompts)
    assert tokens == resident_tokens
    assert mx.allclose(logits, resident_logits, atol=1e-4, rtol=1e-4)
    cache = scheduler.engine.moe_offload_cache
    assert cache is not None and cache.cache_size == OFFLOAD_CACHE_SIZE
    assert (scheduler.engine.expert_pack is not None) == case.is_packed
    assert (cache.pairs is not None) == (case.is_packed and case.expert_format == "affine")
    stats = cache.stats()
    assert sum(stats.misses) > 0
    # A miss is an expert read or a victim cache hit; so is a prefill buffer's absent expert,
    # which is no miss.
    copied = sum(stats.victim_hits) + sum(stats.victim_buffer_hits)
    reads_buffers = max(case.prompt_lens) >= case.overlap_min_tokens
    assert (sum(stats.experts_read) + copied > sum(stats.misses)) == reads_buffers
    assert (copied > 0) == (case.victim_cache_gib > 0)
    # One layer's gate and up apart puts every layer's apart.
    is_apart = case.expert_format == "affine-up-8-bit"
    assert all(
        ("gate_up_proj.weight" not in pool.layout) == is_apart for pool in cache.slot_layout.pools
    )
    # Affine decodes run the pending kernels under gelu_tanh too.
    is_split = case.split_lookup and case.expert_format == "affine"
    assert set(pending_layers) == (set(range(NUM_LAYERS)) if is_split else set())
    cache.check_healthy()


# Each format's second prompt resumes the first's head on the slots the first left: no cold start.
WARM_PREFIX_FORMATS = ["affine", "mxfp4", "nvfp4", "unquantized", "hf-per-expert", "modelopt-26b"]


@pytest.mark.parametrize("expert_format", WARM_PREFIX_FORMATS)
def test_offload_resuming_a_cached_prefix_matches_resident_experts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, expert_format: str
) -> None:
    """A prompt served after another, on the cached head of its KV and the experts already in
    slots (and so resident first, through the prefill buffers), gives every forward's logits as
    the resident experts do, within fp32's rounding: the paths differ, the values must not."""
    monkeypatch.setenv("FTMETAL_MOE_PREFILL_OVERLAP_MIN_TOKENS", "48")
    path = tmp_path / "checkpoint"
    write_offload_checkpoint(build_reference(seed=22, moe=True), path, expert_format)
    first = random_prompt(23, 23)
    second = first[:17] + random_prompt(24, 60)
    splits = 0
    resident_first = OffloadMoeCache.resident_first

    def counting(
        cache: OffloadMoeCache, layer_id: int
    ) -> tuple[ResidentFirst, dict[str, mx.array]] | None:
        nonlocal splits
        first_pass = resident_first(cache, layer_id)
        splits += first_pass is not None
        return first_pass

    monkeypatch.setattr(OffloadMoeCache, "resident_first", counting)

    def serve(offload: bool) -> tuple[int, list[int], list[mx.array], Scheduler]:
        scheduler = make_offload_scheduler(path, OffloadCase("warm", ()), offload)
        run_alone(scheduler, 0, first, OFFLOAD_STEPS)
        seen: list[mx.array] = []
        sample = scheduler.engine.sampler.sample

        def recording_sample(logits: mx.array, args: BatchSamplingArgs) -> mx.array:
            seen.append(logits)
            return sample(logits, args)

        scheduler.engine.sampler.sample = recording_sample
        cached_len, tokens = run_alone(scheduler, 1, second, OFFLOAD_STEPS)
        return cached_len, tokens, seen, scheduler

    resident_cached, resident_tokens, resident_seen, _ = serve(False)
    cached, tokens, seen, scheduler = serve(True)
    assert cached == resident_cached == 16
    assert splits > 0, "the second prompt's prefill ran its slots' experts first"
    assert tokens == resident_tokens
    assert len(seen) == len(resident_seen)
    for forward, (got, expected) in enumerate(zip(seen, resident_seen, strict=True)):
        assert mx.allclose(got, expected, atol=1e-4, rtol=1e-4), f"forward {forward}"
    cache = scheduler.engine.moe_offload_cache
    assert cache is not None
    cache.check_healthy()
