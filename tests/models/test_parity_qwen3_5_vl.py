"""Image prompts through the paged engine on a tiny Qwen3.6-shaped VL model, against an unpaged
reference forward of the same text model and tower (tiny_vl_model.reference_vl_greedy).

Greedy decode must agree token for token whether the prefill is one chunk or cut inside an
image span, whether the image prefix comes from the radix cache or is encoded again, and
whether decode replays CommandGraphs; a text-only prompt on a vision-serving model decodes as
the same model without vision; the tower loads from the Hugging Face layout and from the MLX
conversion's. The tower is whatever models/qwen3_vl/vision.py builds, so nothing here depends
on its numerics.
"""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from ftmetal.command_graph import _C, check_same_graph
from ftmetal.core import Batch, SamplingParams
from ftmetal.engine.config import EngineConfig
from ftmetal.engine.graph import GraphRunner
from ftmetal.mm.config import ENCODER_KINDS, EncoderKind, MultimodalConfig
from ftmetal.mm.processor import MMResult
from ftmetal.scheduler.scheduler import Scheduler
from tiny_model import reference_greedy
from tiny_vl_model import (
    TINY_IMAGE_BUDGET,
    build_vl_checkpoint,
    image_prompt,
    noise_png,
    reference_vl_greedy,
)

STEPS = 10


def make_scheduler(
    path: Path,
    max_extend_tokens: int = 256,
    attention_backend: str = "mlx",
    command_graph: bool = False,
    mm: MultimodalConfig = TINY_IMAGE_BUDGET,
    max_running_req: int = 2,
) -> Scheduler:
    config = EngineConfig(
        model_path=str(path),
        dtype=mx.float32,
        max_running_req=max_running_req,
        page_size=4,
        attention_backend=attention_backend,
        num_page_override=256,
        max_extend_tokens_override=max_extend_tokens,
        command_graph=command_graph,
        mm=mm,
    )
    return Scheduler(config, eos_token_ids={127})


def add_prompt(scheduler: Scheduler, uid: int, prompt: MMResult, steps: int = STEPS) -> None:
    params = SamplingParams(temperature=0.0, max_tokens=steps, ignore_eos=True)
    scheduler.add_request(
        uid,
        prompt.input_ids,
        params,
        None,
        prompt.mm_items,
        prompt.mrope_positions,
        prompt.mrope_delta,
    )


def tokens_of(scheduler: Scheduler, num_requests: int) -> list[list[int]]:
    """Run to idle; each request's tokens, after checking the caches came out clean."""
    events = scheduler.run_until_idle()
    assert scheduler.engine.encoder_cache is not None
    assert scheduler.engine.encoder_cache.entries == {}
    scheduler.cache_manager.check_integrity()
    return [[event.token for event in events if event.uid == uid] for uid in range(num_requests)]


def random_texts(seed: int, lengths: tuple[int, ...]) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    return [[int(t) for t in rng.integers(1, 120, size=n)] for n in lengths]


# 7 text tokens, a 16-token image on [7, 23), 9 text, an 8-token 2x4 image on [32, 40), 5 text.
TEXT_LENGTHS = (7, 9, 5)
IMAGE_SIZES = ((128, 128), (128, 64))

# (prefill chunk budget, attention backend): 16 cuts the first image at 16 and ends a chunk
# where the second begins.
CHUNK_CASES: list[tuple[int, str]] = [(256, "mlx"), (16, "mlx"), (16, "metal")]


@pytest.mark.parametrize(("max_extend_tokens", "attention_backend"), CHUNK_CASES)
def test_image_prompt_matches_the_reference_however_it_is_chunked(
    tmp_path: Path, max_extend_tokens: int, attention_backend: str
) -> None:
    model, tower = build_vl_checkpoint(0, tmp_path)
    texts = random_texts(0, TEXT_LENGTHS)
    images = [noise_png(width, height, seed) for seed, (width, height) in enumerate(IMAGE_SIZES)]
    prompt = image_prompt(tmp_path, texts, images)
    assert [item.offsets for item in prompt.mm_items] == [[(7, 23)], [(32, 40)]]
    expected = reference_vl_greedy(model, tower, image_prompt(tmp_path, texts, images), STEPS)
    # The 3-axis positions matter: the sequence index alone decodes otherwise.
    one_axis = dataclasses.replace(
        image_prompt(tmp_path, texts, images),
        mrope_positions=np.tile(np.arange(len(prompt.input_ids), dtype=np.int32), (3, 1)),
        mrope_delta=0,
    )
    assert reference_vl_greedy(model, tower, one_axis, STEPS) != expected

    scheduler = make_scheduler(tmp_path, max_extend_tokens, attention_backend)
    add_prompt(scheduler, 0, prompt)
    assert tokens_of(scheduler, 1) == [expected]


def test_image_prefix_from_the_radix_cache_matches_a_fresh_encode(tmp_path: Path) -> None:
    """The second identical request resumes from the GDN snapshot at 64, inside the second
    image [62, 70): the first image is not encoded again, the second is, for its last 6 rows."""
    model, tower = build_vl_checkpoint(1, tmp_path)
    texts = random_texts(1, (20, 26, 20))
    images = [
        noise_png(width, height, 10 + seed) for seed, (width, height) in enumerate(IMAGE_SIZES)
    ]
    expected = reference_vl_greedy(model, tower, image_prompt(tmp_path, texts, images), STEPS)

    scheduler = make_scheduler(tmp_path)
    plans = []
    for uid in range(2):
        add_prompt(scheduler, uid, image_prompt(tmp_path, texts, images))
        forward_input = scheduler.schedule_next_batch()
        assert forward_input is not None
        batch = forward_input.batch
        plans.append((batch.reqs[0].cached_len, len(batch.mm_encoder_jobs), batch.mm_gather_plan))
        data = scheduler.forward(forward_input)
        while scheduler.runnable or data is not None:
            data = scheduler.overlap_loop(data)
        assert [event.token for event in scheduler.take_results()] == expected
    (_, first_jobs, _), (cached_len, second_jobs, second_plan) = plans
    assert first_jobs == 2 and (cached_len, second_jobs) == (64, 1)
    assert [(row.row_lo, row.row_hi, row.position) for row in second_plan] == [(2, 8, 0)]
    assert scheduler.engine.encoder_cache is not None
    assert scheduler.engine.encoder_cache.entries == {}
    scheduler.cache_manager.check_integrity()


def concurrent_prompts(path: Path) -> list[MMResult]:
    """Two image prompts (one image twice) and a text prompt."""
    images = [
        noise_png(width, height, 20 + seed) for seed, (width, height) in enumerate(IMAGE_SIZES)
    ]
    return [
        image_prompt(path, random_texts(2, TEXT_LENGTHS), images),
        image_prompt(path, random_texts(3, (4, 11)), images[:1]),
        image_prompt(path, random_texts(4, (13,)), []),
    ]


def test_decode_replays_the_same_tokens_as_eager_decode(tmp_path: Path) -> None:
    model, tower = build_vl_checkpoint(2, tmp_path)
    expected = [
        reference_vl_greedy(model, tower, prompt, STEPS) for prompt in concurrent_prompts(tmp_path)
    ]
    for command_graph in (False, True):
        scheduler = make_scheduler(
            tmp_path, attention_backend="metal", command_graph=command_graph, max_running_req=4
        )
        assert bool(scheduler.engine.graph_runner.graphs) == command_graph
        for uid, prompt in enumerate(concurrent_prompts(tmp_path)):
            add_prompt(scheduler, uid, prompt)
        assert tokens_of(scheduler, 3) == expected


def test_replayed_mrope_step_captures_the_same_graph(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The double-capture check of a real padded decode step (three requests, two of them past
    images, in bucket 4) against the graph captured on dummy rows. Two captures of the step
    whose mrope positions differ record the same graph; left out of the step inputs, the
    positions are caught as a constant."""
    build_vl_checkpoint(3, tmp_path)
    scheduler = make_scheduler(
        tmp_path, attention_backend="metal", command_graph=True, max_running_req=4
    )
    for uid, prompt in enumerate(concurrent_prompts(tmp_path)):
        add_prompt(scheduler, uid, prompt, steps=16)
    while scheduler.prefill_manager.runnable:
        forward_input = scheduler.schedule_next_batch()
        assert forward_input is not None
        scheduler.process_last_data(scheduler.forward(forward_input))
    forward_input = scheduler.schedule_next_batch()
    assert forward_input is not None
    batch = forward_input.batch
    assert batch.is_decode and (batch.size, batch.padded_size) == (3, 4)
    assert batch.mrope_positions is not None and batch.mrope_positions.shape == (3, 4)
    runner = scheduler.engine.graph_runner
    for backend in runner.attn_backends.values():
        backend.prepare_for_replay(batch)
    check_same_graph(runner.graphs[batch.padded_size], runner.capture(batch))

    positions = np.array(batch.mrope_positions)

    def capture_shifted(shift: int) -> _C.Graph:
        batch.mrope_positions = mx.array(positions + shift)
        return runner.capture(batch)

    check_same_graph(capture_shifted(0), capture_shifted(1))
    real_step_inputs = GraphRunner.step_inputs

    def without_mrope(runner: GraphRunner, batch: Batch) -> list[mx.array]:
        arrays = real_step_inputs(runner, batch)
        return [array for array in arrays if array is not batch.mrope_positions]

    monkeypatch.setattr(GraphRunner, "step_inputs", without_mrope)
    with pytest.raises(RuntimeError, match="constant leaf"):
        check_same_graph(capture_shifted(0), capture_shifted(1))


# (the encoder kinds left unbuilt, whether the vision tower is built, the startup log line
# about the unbuilt tower or None)
TEXT_PROMPT_CASES: list[tuple[frozenset[EncoderKind], bool, str | None]] = [
    (frozenset(), True, None),
    (frozenset({"audio"}), True, None),
    (frozenset(ENCODER_KINDS), False, "Multimodal disabled: --text-model-only"),
    (frozenset({"vision"}), False, "Multimodal disabled: --mm-disable"),
]


def test_text_prompt_decodes_as_the_model_without_vision(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Three equal mrope rows are the 1-D rope: a vision-serving engine decodes a text prompt
    as the same checkpoint served text only, and as mlx-lm."""
    model, _ = build_vl_checkpoint(4, tmp_path)
    prompt = random_texts(5, (37,))[0]
    _, expected = reference_greedy(model, prompt, STEPS)
    params = SamplingParams(temperature=0.0, max_tokens=STEPS, ignore_eos=True)
    for disabled_encoders, has_tower, log_line in TEXT_PROMPT_CASES:
        caplog.clear()
        mm = dataclasses.replace(TINY_IMAGE_BUDGET, disabled_encoders=disabled_encoders)
        with caplog.at_level(logging.INFO, logger="ftmetal.engine.engine"):
            scheduler = make_scheduler(tmp_path, mm=mm)
        assert scheduler.engine.model_config.is_mrope is has_tower
        assert (scheduler.engine.encoder_cache is not None) is has_tower
        disabled_records = [
            record for record in caplog.records if "Multimodal disabled" in record.getMessage()
        ]
        assert [record.getMessage() for record in disabled_records] == (
            [] if log_line is None else [log_line]
        )
        # A tower left unbuilt on purpose is no warning.
        assert all(record.levelno == logging.INFO for record in disabled_records)
        tokens = scheduler.generate(np.array(prompt, dtype=np.int32), params)
        assert tokens == expected, disabled_encoders


def test_tower_in_the_mlx_layout_matches_the_reference(tmp_path: Path) -> None:
    """A tower stored as the MLX conversion does (vision_tower., the patch conv kernel channels
    last) encodes and decodes as the reference, which runs it channels first."""
    model, tower = build_vl_checkpoint(7, tmp_path)
    # A random patch kernel: the zero-initialized one reads the same in either layout.
    kernel = tower.patch_embed.proj.weight
    tower.patch_embed.proj.update({"weight": 0.1 * mx.random.normal(kernel.shape)})
    tensors = {
        f"vision_tower.{key}": value
        for key, value in tree_flatten(tower.parameters())
        if isinstance(value, mx.array)
    }
    patch = "vision_tower.patch_embed.proj.weight"
    tensors[patch] = tensors[patch].moveaxis(1, 4)
    mx.save_safetensors(str(tmp_path / "visual.safetensors"), tensors)
    texts = random_texts(7, TEXT_LENGTHS)
    images = [
        noise_png(width, height, 40 + seed) for seed, (width, height) in enumerate(IMAGE_SIZES)
    ]
    expected = reference_vl_greedy(model, tower, image_prompt(tmp_path, texts, images), STEPS)

    scheduler = make_scheduler(tmp_path)
    for item in image_prompt(tmp_path, texts, images).mm_items:
        assert mx.array_equal(scheduler.engine.model.encode(item), tower.encode(item))
    add_prompt(scheduler, 0, image_prompt(tmp_path, texts, images))
    assert tokens_of(scheduler, 1) == [expected]


def test_checkpoint_without_tower_weights_warns_and_serves_text_only(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A checkpoint whose config has a vision section but whose weights hold no tower, with no
    processor config, loads as its text model, with a warning naming the prefixes read."""
    model, _ = build_vl_checkpoint(6, tmp_path)
    (tmp_path / "visual.safetensors").unlink()
    (tmp_path / "preprocessor_config.json").unlink()
    prompt = random_texts(5, (37,))[0]
    _, expected = reference_greedy(model, prompt, STEPS)
    with caplog.at_level(logging.INFO, logger="ftmetal.engine.engine"):
        scheduler = make_scheduler(tmp_path)
    assert scheduler.engine.encoder_cache is None and not scheduler.engine.model_config.is_mrope
    warning = (
        "Multimodal disabled: config.json has vision_config but the checkpoint has no vision "
        "encoder weights under model.visual., vision_tower.; serving text only"
    )
    disabled_records = [
        (record.levelno, record.getMessage())
        for record in caplog.records
        if "Multimodal disabled" in record.getMessage()
    ]
    assert disabled_records == [(logging.WARNING, warning)]
    params = SamplingParams(temperature=0.0, max_tokens=STEPS, ignore_eos=True)
    assert scheduler.generate(np.array(prompt, dtype=np.int32), params) == expected


def test_abort_mid_prefill_releases_the_encoder_rows(tmp_path: Path) -> None:
    build_vl_checkpoint(5, tmp_path)
    images = [
        noise_png(width, height, 30 + seed) for seed, (width, height) in enumerate(IMAGE_SIZES)
    ]
    scheduler = make_scheduler(tmp_path, max_extend_tokens=16)
    add_prompt(scheduler, 0, image_prompt(tmp_path, random_texts(6, TEXT_LENGTHS), images))
    encoder_cache = scheduler.engine.encoder_cache
    assert encoder_cache is not None
    # The first chunk encodes the first image and gathers 9 of its 16 rows.
    data = scheduler.overlap_loop(None)
    assert data is not None
    assert {len(entry.remaining) for entry in encoder_cache.entries.values()} == {1}
    scheduler.abort_request(0)
    scheduler.overlap_loop(data)
    assert encoder_cache.entries == {} and not scheduler.runnable
    scheduler.cache_manager.check_integrity()
