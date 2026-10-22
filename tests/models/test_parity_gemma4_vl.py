"""Image prompts through the paged engine on a tiny Gemma4-shaped VL model, against an unpaged
reference forward: mlx-lm's Gemma4 text model with the image rows in its input embeddings
(unscaled, after the sqrt(hidden) scaling of the text rows) and, on the window layers, each
image's tokens attending to each other in both directions, the window still applying; the
global layers stay causal.

Greedy decode must agree token for token on both attention backends whether the prefill is
one chunk or chunked around the images, whether the image prefix comes from the radix cache
(a match ending inside an image) or is encoded again, whether decode replays CommandGraphs
and whether a request waits a pass for its image; a text-only prompt on the vision-serving
model decodes as mlx-lm. The images are 16 tokens (longer than the window of 8) and 10. The
soft tokens come from the engine's own encode, so nothing here depends on the tower's
numerics (tests/models/test_gemma4_vision.py holds those).
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten
from mlx_lm.models import gemma4 as reference
from test_parity_gemma4 import EOS, WINDOW, build_reference, reference_greedy, text_config

from ftmetal.core import SamplingParams
from ftmetal.engine.config import EngineConfig
from ftmetal.mm.config import ENCODER_KINDS, EncoderKind, MultimodalConfig
from ftmetal.mm.item import MMItem
from ftmetal.mm.processor import MMResult
from ftmetal.mm.processors.gemma4 import Gemma4MMProcessor
from ftmetal.models.blocks import SupportsMultimodal
from ftmetal.models.loader import read_model_config
from ftmetal.scheduler.scheduler import Scheduler
from tiny_vl_model import (
    GEMMA4_IMAGE_PROCESSOR_CONFIG,
    TINY_GEMMA4_VISION_CONFIG,
    gemma4_tower_weights,
    noise_png,
)

STEPS = 8
BOI_TOKEN_ID, IMAGE_TOKEN_ID, EOI_TOKEN_ID = 124, 125, 126
# A square image pools to 16 soft tokens, a 2:1 one to 10.
MAX_SOFT_TOKENS = 16
SQUARE, WIDE = (64, 64), (128, 64)


def build_checkpoint(seed: int, path: Path) -> reference.Model:
    """The text model in the MLX layout plus a random tower; returns the text reference."""
    model = build_reference(seed, moe=False)
    weights = dict(tree_flatten(model.parameters()))
    hidden_size = text_config(moe=False)["hidden_size"]
    assert isinstance(hidden_size, int)
    weights |= gemma4_tower_weights(TINY_GEMMA4_VISION_CONFIG, hidden_size, seed)
    mx.save_safetensors(str(path / "model.safetensors"), weights)
    config = {
        "architectures": ["Gemma4ForConditionalGeneration"],
        "model_type": "gemma4",
        "text_config": {**text_config(moe=False), "use_bidirectional_attention": "vision"},
        "vision_config": TINY_GEMMA4_VISION_CONFIG,
        "boi_token_id": BOI_TOKEN_ID,
        "image_token_id": IMAGE_TOKEN_ID,
        "eoi_token_id": EOI_TOKEN_ID,
        "eos_token_id": [EOS],
    }
    (path / "config.json").write_text(json.dumps(config))
    image_processor = {**GEMMA4_IMAGE_PROCESSOR_CONFIG, "max_soft_tokens": MAX_SOFT_TOKENS}
    (path / "processor_config.json").write_text(json.dumps({"image_processor": image_processor}))
    return model


def image_prompt(
    path: Path, texts: list[list[int]], sizes: list[tuple[int, int]], seed: int
) -> MMResult:
    """texts with an image between each two, fresh items every call (a forward frees them)."""
    assert len(texts) == len(sizes) + 1
    ids = list(texts[0])
    for text in texts[1:]:
        ids += [IMAGE_TOKEN_ID, *text]
    images = [noise_png(width, height, seed + i) for i, (width, height) in enumerate(sizes)]
    processor = Gemma4MMProcessor(read_model_config(str(path)), str(path), MultimodalConfig())
    return processor.apply(np.array(ids, dtype=np.int32), images)


def random_texts(seed: int, lengths: tuple[int, ...]) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    return [[int(t) for t in rng.integers(1, 120, size=n)] for n in lengths]


def reference_vl_greedy(
    model: reference.Model, scheduler: Scheduler, prompt: MMResult, bidirectional: bool = True
) -> list[int]:
    """Greedy decode rerunning the whole sequence per token, the engine's soft tokens in the
    image rows."""
    text_model = model.language_model.model
    served = scheduler.engine.model
    assert isinstance(served, SupportsMultimodal)
    encoded = [served.encode(item) for item in prompt.mm_items]
    rows = mx.array(
        [row for item in prompt.mm_items for lo, hi in item.offsets for row in range(lo, hi)]
    )
    sequence = prompt.input_ids.tolist()
    vocab_size = text_model.embed_tokens.weight.shape[0]
    tokens: list[int] = []
    embed_scale = text_model.embed_scale
    for _ in range(STEPS):
        length = len(sequence)
        image_ids = np.full(length, -1)
        for index, item in enumerate(prompt.mm_items):
            for lo, hi in item.offsets:
                image_ids[lo:hi] = index if bidirectional else -1
        same_image = mx.array((image_ids[:, None] == image_ids[None]) & (image_ids[:, None] >= 0))
        positions = mx.arange(length)
        causal = positions[None] <= positions[:, None]
        sliding = (causal | same_image) & (positions[None] > positions[:, None] - WINDOW)
        masks = [
            causal if layer.layer_type == "full_attention" else sliding
            for layer in text_model.layers
        ]
        ids = mx.array(sequence)
        x = text_model.embed_tokens(mx.minimum(ids, vocab_size - 1)) * embed_scale
        if encoded:
            x[rows] = mx.concatenate(encoded)
        text_model.embed_scale = 1.0
        text_model._make_masks = lambda h, cache: masks
        try:
            logits = model(ids[None], input_embeddings=x[None])
        finally:
            text_model.embed_scale = embed_scale
            del text_model._make_masks
        tokens.append(int(mx.argmax(logits[0, -1])))
        sequence.append(tokens[-1])
    return tokens


def make_scheduler(
    path: Path,
    attention_backend: str,
    max_extend_tokens: int = 256,
    command_graph: bool = False,
    mm: MultimodalConfig | None = None,
    max_running_req: int = 2,
) -> Scheduler:
    config = EngineConfig(
        model_path=str(path),
        dtype=mx.float32,
        max_running_req=max_running_req,
        page_size=4,
        attention_backend=attention_backend,
        num_page_override=128,
        max_extend_tokens=max_extend_tokens,
        command_graph=command_graph,
        mm=MultimodalConfig() if mm is None else mm,
    )
    return Scheduler(config, eos_token_ids={EOS})


def add_prompt(scheduler: Scheduler, uid: int, prompt: MMResult) -> None:
    params = SamplingParams(temperature=0.0, max_tokens=STEPS, ignore_eos=True)
    scheduler.add_request(uid, prompt.input_ids, params, None, prompt.mm_items)


def run_in_steps(
    scheduler: Scheduler,
) -> tuple[list[list[tuple[int, int, int]]], dict[int, list[int]]]:
    """Run to idle a step at a time: every prefill batch's (uid, cached_len, device_len) rows
    and each request's tokens, after checking the caches came out clean: no encoder rows, no
    table row and no locked prefix left."""
    chunks: list[list[tuple[int, int, int]]] = []
    while scheduler.runnable:
        forward_input = scheduler.schedule_next_batch()
        assert forward_input is not None
        batch = forward_input.batch
        if batch.is_prefill:
            chunks.append([(req.uid, req.cached_len, req.device_len) for req in batch.reqs])
        scheduler.process_last_data(scheduler.forward(forward_input))
    tokens: dict[int, list[int]] = {}
    for event in scheduler.take_results():
        tokens.setdefault(event.uid, []).append(event.token)
    assert scheduler.engine.encoder_cache is not None
    assert scheduler.engine.encoder_cache.entries == {}
    assert scheduler.table_manager.available_size == scheduler.config.max_running_req
    prefix_cache, _ = scheduler.cache_manager.swa_parts()
    assert (prefix_cache.full_protected, prefix_cache.swa_protected) == (0, 0)
    scheduler.cache_manager.check_integrity()
    return chunks, tokens


def item_spans(items: list[MMItem]) -> list[list[tuple[int, int]]]:
    return [item.offsets for item in items]


# 7 text tokens, boi, a 16-token image on [8, 24), eoi, 9 text, boi, a 10-token image on
# [35, 45), eoi, 5 text.
TEXT_LENGTHS = (7, 9, 5)
# (prefill chunk budget, attention backend): 20 would end the first chunk inside the first
# image, so it ends at 8 and each image falls whole into a later chunk.
CHUNK_CASES = [(256, "mlx"), (256, "metal"), (20, "mlx"), (20, "metal")]
CHUNKS = {256: [(0, 51)], 20: [(0, 8), (8, 28), (28, 48), (48, 51)]}


@pytest.mark.parametrize(("max_extend_tokens", "attention_backend"), CHUNK_CASES)
def test_image_prompt_matches_the_reference_however_it_is_chunked(
    tmp_path: Path, max_extend_tokens: int, attention_backend: str
) -> None:
    model = build_checkpoint(0, tmp_path)
    texts = random_texts(0, TEXT_LENGTHS)
    prompt = image_prompt(tmp_path, texts, [SQUARE, WIDE], seed=0)
    assert item_spans(prompt.mm_items) == [[(8, 24)], [(35, 45)]]
    scheduler = make_scheduler(tmp_path, attention_backend, max_extend_tokens)
    expected = reference_vl_greedy(model, scheduler, prompt)
    # The bidirectional window layers matter: causal attention over the images decodes otherwise.
    causal = image_prompt(tmp_path, texts, [SQUARE, WIDE], seed=0)
    assert reference_vl_greedy(model, scheduler, causal, bidirectional=False) != expected

    add_prompt(scheduler, 0, image_prompt(tmp_path, texts, [SQUARE, WIDE], seed=0))
    chunks, tokens = run_in_steps(scheduler)
    assert chunks == [[(0, *chunk)] for chunk in CHUNKS[max_extend_tokens]]
    assert tokens == {0: expected}


@pytest.mark.parametrize("attention_backend", ["mlx", "metal"])
def test_image_prefix_from_the_radix_cache_matches_a_fresh_encode(
    tmp_path: Path, attention_backend: str
) -> None:
    """The second prompt shares the first one's text and image and differs right after the
    image's eoi: its match ends inside the image (page size 4), the image is encoded again
    for its last rows, and its tokens are those of a fresh prefill."""
    model = build_checkpoint(1, tmp_path)
    head, first_tail, second_tail = random_texts(1, (8, 6, 6))
    prompts = [
        image_prompt(tmp_path, [head, tail], [SQUARE], seed=10)
        for tail in (first_tail, second_tail)
    ]
    assert item_spans(prompts[0].mm_items) == [[(9, 25)]]
    scheduler = make_scheduler(tmp_path, attention_backend)
    expected = [reference_vl_greedy(model, scheduler, prompt) for prompt in prompts]
    cached_lens = []
    for uid, tail in enumerate((first_tail, second_tail)):
        add_prompt(scheduler, uid, image_prompt(tmp_path, [head, tail], [SQUARE], seed=10))
        chunks, tokens = run_in_steps(scheduler)
        cached_lens.append(chunks[0][0][1])
        assert tokens == {uid: expected[uid]}
    assert cached_lens[0] == 0 and 9 < cached_lens[1] < 25


def concurrent_prompts(path: Path) -> list[MMResult]:
    """Two image prompts (one image in both) and a text prompt."""
    return [
        image_prompt(path, random_texts(2, TEXT_LENGTHS), [SQUARE, WIDE], seed=20),
        image_prompt(path, random_texts(3, (4, 11)), [SQUARE], seed=20),
        image_prompt(path, random_texts(4, (13,)), [], seed=0),
    ]


def test_concurrent_prompts_replay_decode_as_the_reference(tmp_path: Path) -> None:
    model = build_checkpoint(2, tmp_path)
    scheduler = make_scheduler(tmp_path, "metal", command_graph=True, max_running_req=4)
    assert scheduler.engine.graph_runner.graphs
    prompts = concurrent_prompts(tmp_path)
    expected = [reference_vl_greedy(model, scheduler, prompt) for prompt in prompts]
    for uid, prompt in enumerate(concurrent_prompts(tmp_path)):
        add_prompt(scheduler, uid, prompt)
    assert run_in_steps(scheduler)[1] == dict(enumerate(expected))


def test_image_waits_for_a_pass_of_its_own(tmp_path: Path) -> None:
    """A prompt resuming at its image's first token (a cached text prompt ends there) with
    less of the pass's budget left than the image is deferred, its admission undone, rather
    than cut; the next pass prefills the image whole."""
    model = build_checkpoint(3, tmp_path)
    head, tail, text = random_texts(3, (7, 3, 10))
    prompt = image_prompt(tmp_path, [head, tail], [SQUARE], seed=30)
    assert item_spans(prompt.mm_items) == [[(8, 24)]]
    scheduler = make_scheduler(tmp_path, "mlx", max_extend_tokens=20)
    expected = reference_vl_greedy(model, scheduler, prompt)
    params = SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True)
    scheduler.add_request(0, np.array([*head, BOI_TOKEN_ID], dtype=np.int32), params)
    scheduler.run_until_idle()
    params = SamplingParams(temperature=0.0, max_tokens=STEPS, ignore_eos=True)
    scheduler.add_request(1, np.array(text, dtype=np.int32), params)
    add_prompt(scheduler, 2, image_prompt(tmp_path, [head, tail], [SQUARE], seed=30))
    chunks, tokens = run_in_steps(scheduler)
    assert chunks == [[(1, 0, 10)], [(2, 8, 28)]]
    assert tokens[2] == expected


@pytest.mark.parametrize("disabled_encoders", [frozenset(), frozenset(ENCODER_KINDS)])
def test_text_prompt_decodes_as_mlx_lm(
    tmp_path: Path, disabled_encoders: frozenset[EncoderKind]
) -> None:
    """A text prompt on the vision-serving engine, and on the same checkpoint served text
    only, decodes as mlx-lm's text model."""
    model = build_checkpoint(4, tmp_path)
    prompt = random_texts(5, (37,))[0]
    _, expected = reference_greedy(model, prompt, STEPS)
    mm = dataclasses.replace(MultimodalConfig(), disabled_encoders=disabled_encoders)
    scheduler = make_scheduler(tmp_path, "metal", mm=mm)
    assert (scheduler.engine.encoder_cache is None) == bool(disabled_encoders)
    params = SamplingParams(temperature=0.0, max_tokens=STEPS, ignore_eos=True)
    assert scheduler.generate(np.array(prompt, dtype=np.int32), params) == expected
