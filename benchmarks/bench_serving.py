"""API serving benchmark: prefill and decode tokens per second of an OpenAI-compatible server.

    uv run python -m benchmarks.bench_serving --base-url http://127.0.0.1:1919 \\
        --dataset sharegpt --tokenizer /path/to/model [--ctx-lens 2048,8192,32768] \\
        [--num-prompts 4] [--max-tokens 256] [--label "ftm serve --num-token 65536"] \\
        [--output results.jsonl]
    uv run python -m benchmarks.bench_serving --dataset sharegpt --tokenizer /path/to/model \\
        --prepare-only

It starts no engine: the server under test runs on its own (ftm serve, mlx_lm.server,
llama-server, Ollama, LM Studio, ...), and every engine is measured the same way, through
streaming chat completions (POST /v1/chat/completions with usage), one request at a time:

    prefill tok/s = prompt tokens / time to the first generated token (content or reasoning)
    decode tok/s  = (completion tokens - 1) / (time of the last token - time of the first)

so the decode rate never includes the prefill. Token counts come from the server's usage; when a
server sends none, the tokenizer's estimate and one token per streamed chunk stand in, marked
in the report. A request whose stream reports an error or ends without a finish reason stops
the run; the requests finished before it are already in --output.

Datasets (--dataset): random (random tokens of the target length; needs --tokenizer), aime
(AIME 2025), gsm8k (GSM8K test), sharegpt (ShareGPT V3 conversations up to a user turn), at
pinned revisions. A real dataset runs the samples whose natural length is closest to each
ctx-len, the seed choosing among those within --tolerance, and warns about the rest: samples are
never padded, cut or concatenated. No two prompts open with the same text, so none hits a prefix
another one cached. A rerun on the same server needs another --seed, or a restarted server when
samples lie beyond the tolerance (the seed cannot change those). --prepare-only downloads and
converts the datasets without sending requests (ShareGPT's 642 MB JSON takes a few GB of memory
once): run it before starting the server under test.

Random tokens route MoE experts unlike real text and give an unrealistically low expert-cache
miss rate: never benchmark MoE offload with --dataset random.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TextIO

from benchmarks.serving_datasets import (
    DATASETS,
    Sample,
    Selection,
    load_conversations,
    load_tokenizer,
    random_samples,
    select_samples,
    warn,
)

# Requests to the server under test bypass any HTTP proxy, which would add a hop and may buffer
# the event stream; dataset downloads keep the environment's proxy.
SERVER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
# A prompt-cache hit below this share of the prompt is a shared chat-template prefix: not flagged.
CACHE_HIT_SHARE = 0.01


@dataclass(kw_only=True)
class RequestOptions:
    base_url: str
    api_key: str
    model: str
    max_tokens: int
    temperature: float
    extra_body: dict[str, object]
    timeout_s: float


@dataclass(kw_only=True)
class RequestResult:
    ctx_len: int
    sample_id: str
    estimated_prompt_tokens: int
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int
    ttft_s: float
    decode_s: float
    total_s: float
    finish_reason: str
    usage_reported: bool

    @property
    def prefill_tokens_per_second(self) -> float:
        return self.prompt_tokens / self.ttft_s

    @property
    def decode_tokens_per_second(self) -> float | None:
        if self.completion_tokens < 2 or self.decode_s <= 0:
            return None
        else:
            return (self.completion_tokens - 1) / self.decode_s


def server_events(response: Iterator[bytes], sample_id: str) -> Iterator[Mapping[str, object]]:
    """The JSON data events of a server-sent event stream, up to [DONE]; errors raise."""
    data_lines: list[str] = []
    for raw_line in response:
        line = raw_line.decode("utf-8").rstrip("\r\n")
        if line.startswith("data:"):
            data_lines.append(line[len("data:") :].strip())
            continue
        if line or not data_lines:
            continue
        data, data_lines = "\n".join(data_lines), []
        if data == "[DONE]":
            return
        event = json.loads(data)
        if not isinstance(event, dict):
            continue
        if event.get("error"):
            raise RuntimeError(f"{sample_id}: the server reported {event['error']}")
        yield event


def has_token(delta: object) -> bool:
    if not isinstance(delta, dict):
        return False
    else:
        return any(delta.get(key) for key in ("content", "reasoning_content", "reasoning"))


def send_request(options: RequestOptions, ctx_len: int, sample: Sample) -> RequestResult:
    payload: dict[str, object] = {
        "model": options.model,
        "messages": sample.messages,
        "max_tokens": options.max_tokens,
        "temperature": options.temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
        **options.extra_body,
    }
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {options.api_key}"}
    request = urllib.request.Request(
        f"{options.base_url}/v1/chat/completions", json.dumps(payload).encode(), headers
    )
    first_token_s: float | None = None
    last_token_s = 0.0
    chunks = 0
    usage: Mapping[str, object] = {}
    finish_reason = ""
    started = time.perf_counter()
    try:
        response = SERVER.open(request, timeout=options.timeout_s)
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", "replace")[:500]
        raise RuntimeError(f"{sample.sample_id}: HTTP {error.code}: {body}") from error
    with response:
        for event in server_events(response, sample.sample_id):
            event_usage = event.get("usage")
            if isinstance(event_usage, dict):
                usage = event_usage
            choices = event.get("choices")
            for choice in choices if isinstance(choices, list) else []:
                if not isinstance(choice, dict):
                    continue
                if has_token(choice.get("delta")):
                    last_token_s = time.perf_counter()
                    if first_token_s is None:
                        first_token_s = last_token_s
                    chunks += 1
                finish_reason = str(choice.get("finish_reason") or finish_reason)
    total_s = time.perf_counter() - started
    if first_token_s is None:
        raise RuntimeError(f"{sample.sample_id}: the server streamed no token")
    if not finish_reason:
        raise RuntimeError(f"{sample.sample_id}: the stream ended without a finish reason")
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    details = usage.get("prompt_tokens_details")
    cached_tokens = details.get("cached_tokens") if isinstance(details, dict) else None
    return RequestResult(
        ctx_len=ctx_len,
        sample_id=sample.sample_id,
        estimated_prompt_tokens=sample.estimated_tokens,
        prompt_tokens=prompt_tokens if isinstance(prompt_tokens, int) else sample.estimated_tokens,
        completion_tokens=completion_tokens if isinstance(completion_tokens, int) else chunks,
        cached_tokens=cached_tokens if isinstance(cached_tokens, int) else 0,
        ttft_s=first_token_s - started,
        decode_s=last_token_s - first_token_s,
        total_s=total_s,
        finish_reason=finish_reason,
        usage_reported=isinstance(prompt_tokens, int) and isinstance(completion_tokens, int),
    )


def served_model(base_url: str, api_key: str) -> str:
    request = urllib.request.Request(
        f"{base_url}/v1/models", headers={"Authorization": f"Bearer {api_key}"}
    )
    with SERVER.open(request, timeout=30) as response:
        body = json.loads(response.read())
    models = body.get("data") if isinstance(body, dict) else None
    if not isinstance(models, list) or not models or not isinstance(models[0], dict):
        raise RuntimeError(f"{base_url}/v1/models lists no model; pass --model")
    return str(models[0]["id"])


def median_text(values: list[float], digits: int) -> str:
    return f"{statistics.median(values):.{digits}f}" if values else "-"


def report(
    ctx_len: int, results: list[RequestResult], tolerance: float, has_tokenizer: bool
) -> str:
    prompts = [result.prompt_tokens for result in results]
    decode_rates = [
        rate for result in results if (rate := result.decode_tokens_per_second) is not None
    ]
    notes: list[str] = []
    if not all(result.usage_reported for result in results):
        notes.append("estimated counts")
    largest_hit = max(results, key=lambda result: result.cached_tokens / result.prompt_tokens)
    if largest_hit.cached_tokens > CACHE_HIT_SHARE * largest_hit.prompt_tokens:
        notes.append(f"prefix-cache hit of {largest_hit.cached_tokens} tokens")
    mismatched = [
        result
        for result in results
        if has_tokenizer
        and result.usage_reported
        and abs(result.prompt_tokens - result.estimated_prompt_tokens)
        > tolerance * result.estimated_prompt_tokens
    ]
    if mismatched:
        notes.append(
            f"the server's prompt length differs from the sample's for {len(mismatched)} of "
            f"{len(results)}: truncated?"
        )
    prefill_rates = [result.prefill_tokens_per_second for result in results]
    completions = [float(result.completion_tokens) for result in results]
    return (
        f"{ctx_len:>8}  {len(results):>2}  {min(prompts):>6}-{max(prompts):<6}  "
        f"{median_text(prefill_rates, 1):>13}  {median_text(decode_rates, 1):>12}  "
        f"{median_text([result.ttft_s for result in results], 2):>7}  "
        f"{median_text(completions, 0):>10}" + (f"  ({'; '.join(notes)})" if notes else "")
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="API serving benchmark: prefill and decode tokens per second"
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:1919")
    parser.add_argument(
        "--model", default=None, help="default: the first of /v1/models; pass it on Ollama"
    )
    parser.add_argument("--api-key", default="none")
    parser.add_argument("--dataset", choices=DATASETS, default="sharegpt")
    parser.add_argument("--ctx-lens", default="2048,8192,32768")
    parser.add_argument("--num-prompts", type=int, default=4, help="measured requests per ctx-len")
    parser.add_argument("--warmup", type=int, default=1, help="unmeasured requests first")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--extra-body", type=json.loads, default={}, help="JSON object merged into each request"
    )
    parser.add_argument("--tokenizer", default=None, help="model dir, tokenizer.json or HF repo")
    parser.add_argument("--tolerance", type=float, default=0.25)
    parser.add_argument(
        "--seed", type=int, default=0, help="which in-tolerance samples, or random tokens, run"
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=Path.home() / ".cache" / "ftmetal" / "bench_serving"
    )
    parser.add_argument("--timeout", type=float, default=1800.0, help="socket read timeout, s")
    parser.add_argument("--label", default="", help="server settings, recorded with the results")
    parser.add_argument("--output", type=Path, default=None, help="JSON lines: run, requests")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.num_prompts < 1 or args.warmup < 0:
        parser.error("--num-prompts must be at least 1 and --warmup at least 0")
    if not isinstance(args.extra_body, dict):
        parser.error("--extra-body must be a JSON object")
    if args.dataset == "random" and args.tokenizer is None:
        parser.error("--dataset random needs --tokenizer")
    return args


def select(args: argparse.Namespace, ctx_lens: list[int], has_tokenizer: bool) -> Selection:
    tokenizer = load_tokenizer(args.tokenizer) if has_tokenizer else None
    if args.dataset == "random":
        assert tokenizer is not None
        warn(
            "random tokens give an unrealistically low MoE expert-cache miss rate; never use "
            "them to benchmark MoE offload"
        )
        return random_samples(ctx_lens, args.num_prompts, args.warmup, tokenizer, args.seed)
    if tokenizer is None:
        warn("no --tokenizer: sample lengths are estimated from characters")
    conversations = load_conversations(args.dataset, args.cache_dir)
    return select_samples(
        conversations, ctx_lens, args.num_prompts, args.warmup, args.tolerance, tokenizer, args.seed
    )


def write_record(output: TextIO | None, record: Mapping[str, object]) -> None:
    if output is not None:
        output.write(json.dumps(record, ensure_ascii=False) + "\n")
        output.flush()


def main() -> int:
    args = parse_args()
    base_url = args.base_url.rstrip("/").removesuffix("/v1")
    ctx_lens = sorted(set(int(value) for value in args.ctx_lens.split(",")))
    has_tokenizer = args.tokenizer is not None
    selection = select(args, ctx_lens, has_tokenizer)
    if args.prepare_only:
        print(f"{args.dataset}: samples ready")
        return 0
    for ctx_len in ctx_lens:
        if not selection.measured[ctx_len]:
            warn(f"ctx-len {ctx_len}: no sample left, skipped")
    options = RequestOptions(
        base_url=base_url,
        api_key=args.api_key,
        model=args.model or served_model(base_url, args.api_key),
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        extra_body=args.extra_body,
        timeout_s=args.timeout,
    )
    output = None if args.output is None else args.output.open("w", encoding="utf-8")
    write_record(
        output,
        {
            "type": "run",
            "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "base_url": base_url,
            "model": options.model,
            "dataset": args.dataset,
            "ctx_lens": ctx_lens,
            "num_prompts": args.num_prompts,
            "warmup": args.warmup,
            "seed": args.seed,
            "tolerance": args.tolerance,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "extra_body": args.extra_body,
            "tokenizer": args.tokenizer,
            "label": args.label,
        },
    )
    print(f"{base_url}  model {options.model}  dataset {args.dataset}  label {args.label!r}")
    print(
        f"max_tokens {args.max_tokens}, temperature {args.temperature}, extra body "
        f"{json.dumps(args.extra_body)}; one request at a time; medians"
    )
    for sample in selection.warmups:
        send_request(options, min(ctx_lens), sample)
    print("ctx-len   n  prompt tokens  prefill tok/s  decode tok/s  TTFT s  out tokens", flush=True)
    for ctx_len in ctx_lens:
        results = []
        for sample in selection.measured[ctx_len]:
            result = send_request(options, ctx_len, sample)
            write_record(output, {"type": "request", **asdict(result)})
            results.append(result)
        if results:
            print(report(ctx_len, results, args.tolerance, has_tokenizer), flush=True)
    if output is not None:
        output.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
