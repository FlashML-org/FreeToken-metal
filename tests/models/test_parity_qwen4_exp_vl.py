"""Image prompts through the paged engine on a tiny Qwen3.8-Flash-Next, against the plain
reference forward (tiny_qwen4_model.reference_logits with ReferenceVision).

The checkpoint stores the tower as the MLX conversion does (vision_tower., a channels-last
patch conv) and the reference runs the same tower channels first, every parameter random, so
the soft tokens agree only when the conversion's layout is read right. Every forward's logits
must match the reference's, not only the greedy tokens, which the image tokens' rope and the
block keys reach only weakly on a tiny random model: whether the prefill is one chunk or its
chunks cut images and indexer blocks (a block's 3-axis start position then comes from the
pool), whether decode replays CommandGraphs, and when a decode step closes a block that
starts on image tokens. The image prefix comes from the radix cache, image rows hash in the
PLE as the placeholder token, and a text prompt decodes as the text model with the tower on
or disabled.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten, tree_map

from ftmetal.core import Batch, SamplingParams
from ftmetal.engine.config import EngineConfig
from ftmetal.engine.engine import ForwardOutput
from ftmetal.engine.sample import BatchSamplingArgs
from ftmetal.mm import restore_placeholder
from ftmetal.mm.config import EncoderKind, MultimodalConfig
from ftmetal.mm.processor import MMResult
from ftmetal.models.loader import read_model_config
from ftmetal.models.qwen3_vl.vision import Qwen3VLVisionModel
from ftmetal.scheduler.scheduler import Scheduler
from tiny_qwen4_model import (
    EOS_TOKEN_ID,
    TINY_QWEN4_CONFIG,
    ReferenceVision,
    random_weights,
    reference_greedy,
    reference_logits,
    save_checkpoint,
)
from tiny_vl_model import (
    QWEN_PREPROCESSOR_CONFIG,
    TINY_IMAGE_BUDGET,
    TINY_IMAGE_TOKEN_ID,
    TINY_VISION_CONFIG,
    build_tower,
    image_prompt,
    noise_png,
)

STEPS = 10
# A 16-token 8x8-patch image, then an 8-token 8x4 one.
IMAGE_SIZES = ((128, 128), (128, 64))
# Two QSA layers: a forward's first stores the rope positions and reads the block starts back,
# the second reuses them.
VL_CONFIG: dict[str, object] = {
    **TINY_QWEN4_CONFIG,
    "layer_types": ["linear_attention", "linear_attention", "full_attention", "full_attention"],
}


def build_checkpoint(seed: int, path: Path) -> tuple[dict[str, mx.array], Qwen3VLVisionModel]:
    """A tiny random model saved under path with its tower under vision_tower., the patch
    conv channels last; returns the text weights and the tower, channels first."""
    weights = random_weights(seed, VL_CONFIG)
    save_checkpoint(weights, path, VL_CONFIG)
    tower = build_tower(seed)
    # The tower's zero-initialized parameters (its merger's output among them) made random.
    tower.update(tree_map(lambda value: 0.1 * mx.random.normal(value.shape), tower.parameters()))
    tensors = {
        f"vision_tower.{key}": value
        for key, value in tree_flatten(tower.parameters())
        if isinstance(value, mx.array)
    }
    patch = "vision_tower.patch_embed.proj.weight"
    tensors[patch] = tensors[patch].transpose(0, 2, 3, 4, 1)
    mx.save_safetensors(str(path / "visual.safetensors"), tensors)
    config = read_model_config(str(path))
    config["vision_config"] = TINY_VISION_CONFIG
    config["image_token_id"] = TINY_IMAGE_TOKEN_ID
    (path / "config.json").write_text(json.dumps(config))
    (path / "preprocessor_config.json").write_text(json.dumps(QWEN_PREPROCESSOR_CONFIG))
    return weights, tower


def make_prompt(path: Path, seed: int, text_lengths: tuple[int, ...]) -> MMResult:
    """Random text runs with an image between each two (a run may be empty); eos ids inside
    the first run restart the n-gram hash."""
    rng = np.random.default_rng(seed)
    texts = [[int(t) for t in rng.integers(1, 120, size=n)] for n in text_lengths]
    if len(texts[0]) > 3:
        texts[0][3] = EOS_TOKEN_ID
    images = [noise_png(*size, seed + i) for i, size in enumerate(IMAGE_SIZES)]
    return image_prompt(path, texts, images[: len(texts) - 1])


def reference_vision(
    tower: Qwen3VLVisionModel, prompt: MMResult, extra_tokens: int = 0
) -> ReferenceVision:
    """The prompt's image rows and soft tokens, and its mrope rows then extra_tokens more."""
    assert prompt.mrope_positions is not None
    rows = [
        row for item in prompt.mm_items for start, end in item.offsets for row in range(start, end)
    ]
    embeds = []
    for item in prompt.mm_items:
        assert item.feature is not None and item.grid_thw is not None
        embeds.append(tower(mx.array(item.feature).view(mx.bfloat16), [item.grid_thw]))
    prompt_len = len(prompt.input_ids)
    tail = np.arange(prompt_len, prompt_len + extra_tokens, dtype=np.int32) + prompt.mrope_delta
    positions = np.concatenate([prompt.mrope_positions, np.tile(tail, (3, 1))], axis=1)
    return ReferenceVision(positions=positions, rows=rows, embeds=mx.concatenate(embeds))


def reference_vl_greedy(
    weights: dict[str, mx.array], tower: Qwen3VLVisionModel, prompt: MMResult, steps: int
) -> list[int]:
    """Greedy decode rerunning the whole sequence per token: the image rows hold the
    placeholder and the tower's soft tokens, and the rope positions are the prompt's mrope rows,
    then the sequence index plus delta past it."""
    sequence = restore_placeholder(prompt.input_ids.astype(np.int64), TINY_IMAGE_TOKEN_ID).tolist()
    tokens: list[int] = []
    for step in range(steps):
        vision = reference_vision(tower, prompt, step)
        logits = reference_logits(weights, sequence, VL_CONFIG, vision)
        tokens.append(int(mx.argmax(logits[-1])))
        sequence.append(tokens[-1])
    return tokens


def make_scheduler(
    path: Path,
    page_size: int,
    attention_backend: str,
    max_extend_tokens: int,
    command_graph: bool,
    mm: MultimodalConfig = TINY_IMAGE_BUDGET,
) -> Scheduler:
    config = EngineConfig(
        model_path=str(path),
        dtype=mx.float32,
        max_running_req=2,
        page_size=page_size,
        attention_backend=attention_backend,
        num_page_override=512,
        max_extend_tokens_override=max_extend_tokens,
        command_graph=command_graph,
        mm=mm,
    )
    return Scheduler(config, eos_token_ids={127})


def run(scheduler: Scheduler, uid: int, prompt: MMResult) -> list[int]:
    params = SamplingParams(temperature=0.0, max_tokens=STEPS, ignore_eos=True)
    scheduler.add_request(
        uid,
        prompt.input_ids,
        params,
        None,
        prompt.mm_items,
        prompt.mrope_positions,
        prompt.mrope_delta,
    )
    tokens = [event.token for event in scheduler.run_until_idle() if event.uid == uid]
    scheduler.cache_manager.check_integrity()
    return tokens


def check_every_step(
    weights: dict[str, mx.array],
    tower: Qwen3VLVisionModel,
    prompt: MMResult,
    scheduler: Scheduler,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Greedy decode of prompt; every forward's logits as the sampler gets them (each prefill
    chunk's last row, each decode step's) against the reference's at the same sequence, so a
    block key that changes no greedy token still shows."""
    # Before the run: a forward frees the features it encodes.
    vision = reference_vision(tower, prompt, STEPS)
    engine = scheduler.engine
    lengths: list[int] = []
    seen: list[mx.array] = []
    forward_batch, sample = engine.forward_batch, engine.sampler.sample

    def recording_forward_batch(batch: Batch, args: BatchSamplingArgs) -> ForwardOutput:
        lengths.append(batch.reqs[0].device_len)
        return forward_batch(batch, args)

    def recording_sample(logits: mx.array, args: BatchSamplingArgs) -> mx.array:
        seen.append(logits[0])
        return sample(logits, args)

    monkeypatch.setattr(engine, "forward_batch", recording_forward_batch)
    monkeypatch.setattr(engine.sampler, "sample", recording_sample)
    tokens = run(scheduler, 0, prompt)
    sequence = restore_placeholder(prompt.input_ids.astype(np.int64), TINY_IMAGE_TOKEN_ID).tolist()
    sequence += tokens
    prompt_len = len(prompt.input_ids)

    def reference_at(length: int) -> mx.array:
        """The reference's logits [length, vocab] of the sequence's first length tokens."""
        at = dataclasses.replace(vision, positions=vision.positions[:, :length])
        return reference_logits(weights, sequence[:length], VL_CONFIG, at)

    prompt_logits = reference_at(prompt_len)
    for length, logits in zip(lengths, seen, strict=True):
        if length <= prompt_len:
            expected = prompt_logits[length - 1]
        else:
            expected = reference_at(length)[-1]
        assert mx.allclose(logits, expected, atol=1e-4, rtol=1e-4), length


# (attention backend, page size, prefill chunk, CommandGraph, text run lengths), 7 + 16 + 9 +
# 8 + 5 = 45 tokens unless noted, so the indexer selects: one chunk; 14-token chunks, which
# cut the first image and the block at 12, whose start (an image token's position) is read back
# from the pool and whose key changes later selections; eager decode; a prompt ending on the
# second image at 40, whose block decode closes at 43.
STEP_CASES = [
    ("mlx", 4, 256, False, (7, 9, 5)),
    ("metal", 1, 14, True, (7, 9, 5)),
    ("metal", 4, 256, False, (7, 9, 5)),
    ("metal", 4, 256, True, (7, 10, 0)),
]


@pytest.mark.parametrize(
    ("attention_backend", "page_size", "max_extend_tokens", "command_graph", "text_lengths"),
    STEP_CASES,
)
def test_image_prompt_matches_reference_at_every_step(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    attention_backend: str,
    page_size: int,
    max_extend_tokens: int,
    command_graph: bool,
    text_lengths: tuple[int, ...],
) -> None:
    weights, tower = build_checkpoint(seed=5, path=tmp_path)
    prompt = make_prompt(tmp_path, 5, text_lengths)
    scheduler = make_scheduler(
        tmp_path, page_size, attention_backend, max_extend_tokens, command_graph
    )
    check_every_step(weights, tower, prompt, scheduler, monkeypatch)


def test_image_prefix_from_the_radix_cache(tmp_path: Path) -> None:
    """The second identical request (20 + 16 + 26 + 8 + 20 tokens) resumes from the GDN
    snapshot at 64, inside the second image [62, 70), with the cached pages' keys, index keys
    and rope positions: only the second image is encoded again."""
    weights, tower = build_checkpoint(seed=6, path=tmp_path)
    expected = reference_vl_greedy(weights, tower, make_prompt(tmp_path, 6, (20, 26, 20)), STEPS)
    scheduler = make_scheduler(tmp_path, 4, "metal", 256, command_graph=True)
    plans = []
    for uid in range(2):
        prompt = make_prompt(tmp_path, 6, (20, 26, 20))
        params = SamplingParams(temperature=0.0, max_tokens=STEPS, ignore_eos=True)
        scheduler.add_request(
            uid,
            prompt.input_ids,
            params,
            None,
            prompt.mm_items,
            prompt.mrope_positions,
            prompt.mrope_delta,
        )
        forward_input = scheduler.schedule_next_batch()
        assert forward_input is not None
        batch = forward_input.batch
        plans.append((batch.reqs[0].cached_len, len(batch.mm_encoder_jobs)))
        data = scheduler.forward(forward_input)
        while scheduler.runnable or data is not None:
            data = scheduler.overlap_loop(data)
        assert [event.token for event in scheduler.take_results()] == expected
    assert plans == [(0, 2), (64, 1)]
    scheduler.cache_manager.check_integrity()


@pytest.mark.parametrize("has_vision", [True, False])
def test_text_prompt_matches_text_reference(tmp_path: Path, has_vision: bool) -> None:
    """A text prompt decodes as the text model: on the vision model every rope axis takes the
    sequence index; with the tower disabled the model is built without it, on the 1-D rope."""
    weights, _ = build_checkpoint(seed=7, path=tmp_path)
    prompt = make_prompt(tmp_path, 7, (45,))
    _, expected = reference_greedy(weights, prompt.input_ids.tolist(), STEPS, VL_CONFIG)
    disabled: frozenset[EncoderKind] = frozenset() if has_vision else frozenset({"vision"})
    mm = MultimodalConfig(image_min_tokens=4, disabled_encoders=disabled)
    scheduler = make_scheduler(tmp_path, 4, "metal", 256, command_graph=True, mm=mm)
    assert scheduler.engine.model_config.is_mrope == has_vision
    assert ("visual" in scheduler.engine.model) == has_vision
    assert run(scheduler, 0, prompt) == expected
