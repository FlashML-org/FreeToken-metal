"""Prompts for bench_serving: real datasets at their natural length, or random tokens."""

from __future__ import annotations

import heapq
import json
import random
import sys
import urllib.request
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

DATASETS = ("random", "aime", "gsm8k", "sharegpt")
MATH_INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."
# Pinned revisions, so that a dataset name always means the same samples.
AIME_REPO = "math-ai/aime25"
AIME_REVISION = "563bb8404243c5f09de6ec262f2db674fe5bce9b"
GSM8K_URL = (
    "https://raw.githubusercontent.com/openai/grade-school-math/"
    "3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/data/test.jsonl"
)
SHAREGPT_REPO = "anon8231489123/ShareGPT_Vicuna_unfiltered"
SHAREGPT_REVISION = "192ab2185289094fc556ec8ce5ce1e8e587154ca"
SHAREGPT_FILE = "ShareGPT_V3_unfiltered_cleaned_split.json"
SHAREGPT_ROLES = {"human": "user", "gpt": "assistant"}
# Chat-template tokens per message, added to the content when estimating a prompt's length.
TEMPLATE_TOKENS_PER_MESSAGE = 5
# Characters per token of English text: the length estimate without a tokenizer, and the filter
# that keeps tokenization to the candidates nearest a ctx-len.
CHARS_PER_TOKEN = 4.0
MIN_CANDIDATES_PER_CTX_LEN = 400
# Two prompts whose first turns open with the same characters share a cached prefix (ShareGPT
# repeats conversations under new ids), so at most one of them is chosen: about 64 tokens.
PREFIX_KEY_CHARS = 256

Message = dict[str, str]


@dataclass(kw_only=True)
class Sample:
    sample_id: str
    messages: list[Message]
    estimated_tokens: int


@dataclass(kw_only=True)
class Selection:
    """The measured samples per ctx-len, and the warm-up samples, run first and not reported."""

    measured: dict[int, list[Sample]]
    warmups: list[Sample]


def read_jsonl(path: Path) -> Iterator[Mapping[str, object]]:
    with path.open(encoding="utf-8") as lines:
        for line in lines:
            if line.strip():
                row = json.loads(line)
                assert isinstance(row, dict)
                yield row


def write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as lines:
        for row in rows:
            lines.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def aime_conversations() -> Iterator[tuple[str, list[Message]]]:
    path = hf_hub_download(AIME_REPO, "test.jsonl", repo_type="dataset", revision=AIME_REVISION)
    for row in read_jsonl(Path(path)):
        prompt = f"{row['problem']}\n{MATH_INSTRUCTION}"
        yield f"aime25-{row['id']}", [{"role": "user", "content": prompt}]


def gsm8k_conversations(cache_dir: Path) -> Iterator[tuple[str, list[Message]]]:
    path = cache_dir / "gsm8k_test.jsonl"
    if not path.exists():
        with urllib.request.urlopen(GSM8K_URL, timeout=120) as response:
            path.write_bytes(response.read())
    for index, row in enumerate(read_jsonl(path)):
        prompt = f"{row['question']}\n{MATH_INSTRUCTION}"
        yield f"gsm8k-{index}", [{"role": "user", "content": prompt}]


def sharegpt_messages(turns: object) -> list[Message]:
    """The turns up to the last user turn, while they alternate from a user turn."""
    messages: list[Message] = []
    if not isinstance(turns, list):
        return messages
    for turn in turns:
        if not isinstance(turn, dict):
            break
        role = SHAREGPT_ROLES.get(str(turn.get("from")))
        content = turn.get("value")
        expected = "user" if len(messages) % 2 == 0 else "assistant"
        if role != expected or not isinstance(content, str) or not content.strip():
            break
        messages.append({"role": expected, "content": content})
    if messages and messages[-1]["role"] == "assistant":
        messages.pop()
    return messages


def sharegpt_conversations(cache_dir: Path) -> Iterator[tuple[str, list[Message]]]:
    path = cache_dir / f"sharegpt_v3_{SHAREGPT_REVISION[:12]}.jsonl"
    if not path.exists():
        print("converting ShareGPT once (a few GB of memory) ...", flush=True)
        source = hf_hub_download(
            SHAREGPT_REPO, SHAREGPT_FILE, repo_type="dataset", revision=SHAREGPT_REVISION
        )
        entries = json.loads(Path(source).read_text(encoding="utf-8"))
        assert isinstance(entries, list)
        write_jsonl(
            path,
            (
                {"id": str(entry.get("id")), "messages": messages}
                for entry in entries
                if (messages := sharegpt_messages(entry.get("conversations")))
            ),
        )
        del entries
    for row in read_jsonl(path):
        messages = row["messages"]
        assert isinstance(messages, list)
        yield f"sharegpt-{row['id']}", messages


def load_conversations(dataset: str, cache_dir: Path) -> Iterator[tuple[str, list[Message]]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    if dataset == "aime":
        return aime_conversations()
    elif dataset == "gsm8k":
        return gsm8k_conversations(cache_dir)
    else:
        assert dataset == "sharegpt", dataset
        return sharegpt_conversations(cache_dir)


def load_tokenizer(name: str) -> Tokenizer:
    """A model directory, a tokenizer.json, or a Hugging Face repo id."""
    path = Path(name).expanduser()
    if path.is_dir():
        return Tokenizer.from_file(str(path / "tokenizer.json"))
    elif path.is_file():
        return Tokenizer.from_file(str(path))
    else:
        return Tokenizer.from_file(hf_hub_download(name, "tokenizer.json"))


def estimate_tokens(messages: list[Message], tokenizer: Tokenizer | None) -> int:
    if tokenizer is None:
        content = sum(round(len(message["content"]) / CHARS_PER_TOKEN) for message in messages)
    else:
        contents = [message["content"] for message in messages]
        encodings = tokenizer.encode_batch(contents, add_special_tokens=False)
        content = sum(len(encoding.ids) for encoding in encodings)
    return content + TEMPLATE_TOKENS_PER_MESSAGE * len(messages)


def warn(message: str) -> None:
    print(f"WARNING: {message}", file=sys.stderr)


def check_selection(ctx_len: int, chosen: list[Sample], count: int, tolerance: float) -> None:
    lengths = sorted(sample.estimated_tokens for sample in chosen)
    outside = [length for length in lengths if abs(length - ctx_len) > tolerance * ctx_len]
    if len(chosen) < count:
        warn(f"ctx-len {ctx_len}: only {len(chosen)} of {count} samples are left")
    if outside:
        warn(
            f"ctx-len {ctx_len}: {len(outside)} of {len(chosen)} samples are beyond "
            f"{tolerance:.0%} of the target ({lengths} tokens); they run at their natural length, "
            f"not padded. --seed cannot change samples beyond the tolerance: restart the server "
            f"before a rerun, or it reuses their cached prefixes"
        )


def select_samples(
    conversations: Iterator[tuple[str, list[Message]]],
    ctx_lens: list[int],
    count: int,
    warmup: int,
    tolerance: float,
    tokenizer: Tokenizer | None,
    seed: int,
) -> Selection:
    """count samples per ctx-len and warmup more at the shortest, no two opening alike.

    Each ctx-len takes the samples of natural length closest to it, the seed choosing among
    those within tolerance. One pass keeps, per ctx-len, the candidates nearest it by
    characters, so a large dataset is neither held in memory nor tokenized whole. The longest
    ctx-len chooses first: long samples are the scarce ones.
    """
    capacity = max(MIN_CANDIDATES_PER_CTX_LEN, 4 * (count + warmup))
    nearest: dict[int, list[tuple[float, int, str, list[Message]]]] = {n: [] for n in ctx_lens}
    for order, (sample_id, messages) in enumerate(conversations):
        tokens_by_chars = sum(len(message["content"]) for message in messages) / CHARS_PER_TOKEN
        for ctx_len, heap in nearest.items():
            entry = (-abs(tokens_by_chars - ctx_len), order, sample_id, messages)
            if len(heap) < capacity:
                heapq.heappush(heap, entry)
            elif entry > heap[0]:
                heapq.heapreplace(heap, entry)
    used_prefixes: set[str] = set()
    measured: dict[int, list[Sample]] = {}
    warmups: list[Sample] = []
    for ctx_len in sorted(ctx_lens, reverse=True):
        wanted = count + (warmup if ctx_len == min(ctx_lens) else 0)
        candidates = [
            Sample(
                sample_id=sample_id,
                messages=messages,
                estimated_tokens=estimate_tokens(messages, tokenizer),
            )
            for _, _, sample_id, messages in nearest[ctx_len]
        ]
        candidates.sort(
            key=lambda sample: (abs(sample.estimated_tokens - ctx_len), sample.sample_id)
        )
        # The seed picks among the samples within tolerance, so a rerun on the same server
        # avoids the prefixes a previous run cached; the rest follow by distance.
        within = [
            sample
            for sample in candidates
            if abs(sample.estimated_tokens - ctx_len) <= tolerance * ctx_len
        ]
        random.Random(f"{seed}-{ctx_len}").shuffle(within)
        chosen: list[Sample] = []
        for sample in within + candidates[len(within) :]:
            prefix = sample.messages[0]["content"][:PREFIX_KEY_CHARS]
            if len(chosen) < wanted and prefix not in used_prefixes:
                used_prefixes.add(prefix)
                chosen.append(sample)
        measured[ctx_len], extra = chosen[:count], chosen[count:]
        warmups += extra
        check_selection(ctx_len, measured[ctx_len], count, tolerance)
    return Selection(measured=measured, warmups=warmups)


def random_samples(
    ctx_lens: list[int], count: int, warmup: int, tokenizer: Tokenizer, seed: int
) -> Selection:
    """count prompts of random tokens per ctx-len, and warmup more at the shortest."""
    rng = random.Random(seed)
    special = set(tokenizer.get_added_tokens_decoder())
    vocabulary = [token for token in range(tokenizer.get_vocab_size()) if token not in special]

    def draw(ctx_len: int, sample_id: str) -> Sample:
        target = max(1, ctx_len - TEMPLATE_TOKENS_PER_MESSAGE)
        text = tokenizer.decode([rng.choice(vocabulary) for _ in range(target)])
        ids = tokenizer.encode(text, add_special_tokens=False).ids
        if len(ids) > target:
            text = tokenizer.decode(ids[:target])
        messages = [{"role": "user", "content": text}]
        return Sample(
            sample_id=sample_id,
            messages=messages,
            estimated_tokens=estimate_tokens(messages, tokenizer),
        )

    measured = {
        ctx_len: [draw(ctx_len, f"random-{ctx_len}-{index}") for index in range(count)]
        for ctx_len in ctx_lens
    }
    warmups = [draw(min(ctx_lens), f"random-warmup-{index}") for index in range(warmup)]
    return Selection(measured=measured, warmups=warmups)
