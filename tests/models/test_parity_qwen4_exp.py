"""Numeric parity of the paged engine against a plain reference on a tiny Qwen3.8-Flash-Next.

The reference (tiny_qwen4_model.reference_logits: whole-sequence fp32, no caches) and
ftmetal (paged QSA K/V and index keys, the GDN state pool with the PLE slot states, the
scheduler, overlap, decode replayed from CommandGraphs) run the same random weights: prefill
logits must match and greedy decode must agree token for token, with prompts short enough
for every query to attend every key and long enough for the block indexer to select,
chunked, batched, and resumed from the prefix cache. The prompts hold eos ids, which restart
the n-gram hash.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from ftmetal.core import SamplingParams
from ftmetal.engine.config import EngineConfig
from ftmetal.models.qwen4_exp.model import MAX_PLE_ROWS, Qwen4ExpForCausalLM
from ftmetal.scheduler.scheduler import Scheduler
from tiny_qwen4_model import (
    EOS_TOKEN_ID,
    TINY_QWEN4_WIDE_CONFIG,
    quantize_weights,
    random_weights,
    reference_greedy,
    save_checkpoint,
)

STEPS = 12


def make_scheduler(
    path: Path,
    page_size: int,
    attention_backend: str,
    max_extend_tokens: int = 64,
    num_pages: int = 512,
    command_graph: bool = True,
    max_running_req: int = 2,
) -> Scheduler:
    config = EngineConfig(
        model_path=str(path),
        dtype=mx.float32,
        max_running_req=max_running_req,
        page_size=page_size,
        attention_backend=attention_backend,
        num_page_override=num_pages,
        max_extend_tokens_override=max_extend_tokens,
        command_graph=command_graph,
    )
    return Scheduler(config, eos_token_ids={127})


def make_prompt(seed: int, length: int) -> list[int]:
    prompt = [int(t) for t in np.random.default_rng(seed).integers(1, 120, size=length)]
    for position in (3, length // 2, length // 2 + 1):
        if position < length:
            prompt[position] = EOS_TOKEN_ID
    return prompt


def greedy(scheduler: Scheduler, prompt: list[int], steps: int = STEPS) -> list[int]:
    params = SamplingParams(temperature=0.0, max_tokens=steps, ignore_eos=True)
    return scheduler.generate(np.array(prompt, dtype=np.int32), params)


def run_all(scheduler: Scheduler, prompts: list[list[int]]) -> list[list[int]]:
    """Every prompt admitted at once, greedy to STEPS tokens each."""
    for uid, prompt in enumerate(prompts):
        params = SamplingParams(temperature=0.0, max_tokens=STEPS, ignore_eos=True)
        scheduler.add_request(uid, np.array(prompt, dtype=np.int32), params)
    events = scheduler.run_until_idle()
    return [[e.token for e in events if e.uid == uid] for uid in range(len(prompts))]


# (attention backend, prompt length, max_extend_tokens, page size, CommandGraph): every query
# attends every key (the budget is 16 keys); the indexer selects from the prefill on; chunked
# prefills whose chunks cut blocks; a last chunk of one token (a prefill on the decode path);
# eager decode; 64-token pages and chunks past QUERY_CHUNK queries that start mid-block.
GREEDY_CASES = [
    ("mlx", 12, 64, 1, False),
    ("mlx", 40, 64, 4, False),
    ("mlx", 130, 9, 1, False),
    ("metal", 12, 64, 1, True),
    ("metal", 40, 64, 4, True),
    ("metal", 40, 64, 1, True),
    ("metal", 130, 64, 4, True),
    ("metal", 130, 9, 1, True),
    ("metal", 129, 64, 1, True),
    ("metal", 40, 64, 4, False),
    ("metal", 300, 512, 16, True),
    ("metal", 600, 290, 64, True),
]


@pytest.mark.parametrize(
    ("attention_backend", "prompt_len", "max_extend_tokens", "page_size", "command_graph"),
    GREEDY_CASES,
)
def test_greedy_decode_matches_reference(
    tmp_path: Path,
    attention_backend: str,
    prompt_len: int,
    max_extend_tokens: int,
    page_size: int,
    command_graph: bool,
) -> None:
    weights = random_weights(seed=prompt_len)
    save_checkpoint(weights, tmp_path)
    prompt = make_prompt(prompt_len, prompt_len)
    _, expected = reference_greedy(weights, prompt, STEPS)
    scheduler = make_scheduler(
        tmp_path,
        page_size,
        attention_backend,
        max_extend_tokens=max_extend_tokens,
        num_pages=1024,
        command_graph=command_graph,
    )
    assert greedy(scheduler, prompt) == expected
    scheduler.cache_manager.check_integrity()


# (attention backend, prompt length): dense and selecting prefills.
PREFILL_CASES = [("mlx", 12), ("mlx", 70), ("metal", 12), ("metal", 70)]


@pytest.mark.parametrize(("attention_backend", "prompt_len"), PREFILL_CASES)
def test_prefill_logits_match_reference(
    tmp_path: Path, attention_backend: str, prompt_len: int
) -> None:
    weights = random_weights(seed=prompt_len)
    save_checkpoint(weights, tmp_path)
    prompt = make_prompt(prompt_len, prompt_len)
    expected_logits, _ = reference_greedy(weights, prompt, 1)
    scheduler = make_scheduler(tmp_path, 4, attention_backend, max_extend_tokens=128)
    scheduler.add_request(0, np.array(prompt, dtype=np.int32), SamplingParams(max_tokens=1))
    forward_input = scheduler.schedule_next_batch()
    assert forward_input is not None
    logits = scheduler.engine.model.forward(forward_input.batch)
    assert mx.allclose(logits[0], expected_logits[-1], atol=1e-4, rtol=1e-4)


# (attention backend, prompt lengths): two requests decoded together; one prefill batch of a
# request with no complete block beside one that selects.
BATCHED_CASES = [("mlx", (13, 45)), ("metal", (13, 45)), ("metal", (2, 130))]


@pytest.mark.parametrize(("attention_backend", "prompt_lens"), BATCHED_CASES)
def test_two_requests_batched_match_single(
    tmp_path: Path, attention_backend: str, prompt_lens: tuple[int, int]
) -> None:
    weights = random_weights(seed=3)
    save_checkpoint(weights, tmp_path)
    prompts = [make_prompt(3 + i, length) for i, length in enumerate(prompt_lens)]
    expected = [reference_greedy(weights, prompt, STEPS)[1] for prompt in prompts]
    scheduler = make_scheduler(tmp_path, 4, attention_backend, max_extend_tokens=256)
    assert run_all(scheduler, prompts) == expected
    scheduler.cache_manager.check_integrity()


# (attention backend, page size, the prefix the second turn resumes at least): the snapshot
# inside the first prompt's chunk; at page size 1 the first turn's finished state, with the
# PLE context, conv window and block keys its decode steps left.
NEXT_TURN_CASES = [("mlx", 4, 128), ("metal", 4, 128), ("metal", 1, 150 + STEPS - 1)]


@pytest.mark.parametrize(("attention_backend", "page_size", "min_cached_len"), NEXT_TURN_CASES)
def test_next_turn_resumes_from_the_prefix_cache(
    tmp_path: Path, attention_backend: str, page_size: int, min_cached_len: int
) -> None:
    """A second turn extending the first conversation resumes from its cached prefix: the
    GDN snapshot with the PLE context and conv window, and the QSA keys and index keys of the
    shared pages."""
    weights = random_weights(seed=9)
    save_checkpoint(weights, tmp_path)
    first = make_prompt(9, 150)
    scheduler = make_scheduler(tmp_path, page_size, attention_backend, max_extend_tokens=256)
    reply = greedy(scheduler, first)
    assert reply == reference_greedy(weights, first, STEPS)[1]
    second = first + reply + make_prompt(10, 30)
    params = SamplingParams(temperature=0.0, max_tokens=STEPS, ignore_eos=True)
    scheduler.add_request(1, np.array(second, dtype=np.int32), params)
    forward_input = scheduler.schedule_next_batch()
    assert forward_input is not None
    assert forward_input.batch.reqs[0].cached_len >= min_cached_len
    data = scheduler.forward(forward_input)
    while scheduler.runnable or data is not None:
        data = scheduler.overlap_loop(data)
    tokens = [event.token for event in scheduler.take_results()]
    assert tokens == reference_greedy(weights, second, STEPS)[1]
    scheduler.cache_manager.check_integrity()


@pytest.mark.parametrize("attention_backend", ["mlx", "metal"])
def test_quantized_greedy_decode_matches_reference(tmp_path: Path, attention_backend: str) -> None:
    """The real checkpoint's recipe, at widths the one-token kernels take: the hyper-connections'
    fused down and inject GEMV with its silu, the up projection fused with the mix; the PLE
    table read as quantized rows."""
    config = TINY_QWEN4_WIDE_CONFIG
    stored, dequantized, quantization = quantize_weights(random_weights(seed=11, config=config))
    save_checkpoint(stored, tmp_path, config, quantization)
    prompt = make_prompt(11, 40)
    _, expected = reference_greedy(dequantized, prompt, STEPS, config)
    scheduler = make_scheduler(tmp_path, 4, attention_backend)
    model = scheduler.engine.model
    assert isinstance(model, Qwen4ExpForCausalLM)
    layer = model.model.layers[0]
    assert "down_inject" in layer.attn_hyper_connection
    assert greedy(scheduler, prompt) == expected


def test_decode_graphs_stop_at_one_lookup_post(tmp_path: Path) -> None:
    """A decode batch past what one PLE post takes looks its rows up on the host, which a
    CommandGraph cannot replay: the bucket ladder stops there."""
    save_checkpoint(random_weights(seed=12), tmp_path)
    scheduler = make_scheduler(tmp_path, 4, "metal", max_running_req=MAX_PLE_ROWS + 6)
    assert scheduler.engine.graph_runner.max_graph_bs == MAX_PLE_ROWS
