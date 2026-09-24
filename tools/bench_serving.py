"""HTTP serving benchmark: TTFT and decode tok/s against an OpenAI-compatible server.

    uv run python tools/bench_serving.py --base-url http://127.0.0.1:1919 \\
        [--prompt-tokens 2048] [--max-tokens 256] [--concurrency 1] [--rounds 3]

Each round sends `concurrency` streaming chat completions at once and reports the
per-request time to first token and the decode rate after it, plus the aggregate
generated tokens per second of the round. Standard library only.
"""

from __future__ import annotations

import argparse
import http.client
import json
import random
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from urllib.parse import urlparse

FILLER = (
    "the quick brown fox jumps over lazy dog while seventeen engineers review distributed "
    "systems throughput latency memory bandwidth cache coherence protocol design tradeoffs "
    "across heterogeneous accelerators"
).split()


@dataclass
class RequestResult:
    ttft_s: float
    total_s: float
    completion_tokens: int

    @property
    def decode_tokens_per_second(self) -> float:
        decode_s = self.total_s - self.ttft_s
        return (self.completion_tokens - 1) / decode_s if decode_s > 0 else 0.0


def make_prompt(num_tokens: int, seed: int) -> str:
    """About num_tokens of English filler (roughly 1.3 words per token)."""
    rng = random.Random(seed)
    return " ".join(rng.choice(FILLER) for _ in range(max(8, int(num_tokens * 0.75))))


def one_request(base_url: str, model: str, prompt: str, max_tokens: int) -> RequestResult:
    url = urlparse(base_url)
    connection = http.client.HTTPConnection(url.hostname, url.port or 80, timeout=600.0)
    body = json.dumps(
        {
            "model": model,
            "stream": True,
            "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        }
    )
    started = time.perf_counter()
    connection.request(
        "POST", "/v1/chat/completions", body=body, headers={"Content-Type": "application/json"}
    )
    response = connection.getresponse()
    if response.status != 200:
        raise RuntimeError(f"HTTP {response.status}: {response.read()[:200]!r}")
    ttft = None
    completion_tokens = 0
    chunks_with_content = 0
    for raw_line in response:
        line = raw_line.decode("utf-8").strip()
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        payload = json.loads(line[len("data: ") :])
        usage = payload.get("usage")
        if usage:
            completion_tokens = int(usage.get("completion_tokens", 0))
        choices = payload.get("choices") or []
        if choices and (choices[0].get("delta") or {}).get("content"):
            chunks_with_content += 1
            if ttft is None:
                ttft = time.perf_counter() - started
    total = time.perf_counter() - started
    connection.close()
    if completion_tokens == 0:
        completion_tokens = chunks_with_content
    return RequestResult(
        ttft_s=ttft if ttft is not None else total,
        total_s=total,
        completion_tokens=completion_tokens,
    )


def served_model(base_url: str) -> str:
    url = urlparse(base_url)
    connection = http.client.HTTPConnection(url.hostname, url.port or 80, timeout=30.0)
    connection.request("GET", "/v1/models")
    response = connection.getresponse()
    models = json.loads(response.read())["data"]
    connection.close()
    return str(models[0]["id"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:1919")
    parser.add_argument("--model", default=None)
    parser.add_argument("--prompt-tokens", type=int, default=2048)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()
    model = args.model or served_model(args.base_url)
    print(
        f"model {model}, prompt ~{args.prompt_tokens} tokens, "
        f"max_tokens {args.max_tokens}, c={args.concurrency}"
    )
    one_request(args.base_url, model, make_prompt(64, seed=999), 8)  # warm the server
    for round_index in range(args.rounds):
        prompts = [
            make_prompt(args.prompt_tokens, seed=round_index * 100 + i)
            for i in range(args.concurrency)
        ]
        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            results = list(
                pool.map(lambda p: one_request(args.base_url, model, p, args.max_tokens), prompts)
            )
        wall = time.perf_counter() - started
        generated = sum(r.completion_tokens for r in results)
        ttft_ms = statistics.median(r.ttft_s for r in results) * 1000
        decode = statistics.median(r.decode_tokens_per_second for r in results)
        print(
            f"round {round_index}: ttft median {ttft_ms:.0f} ms, decode per request median "
            f"{decode:.1f} tok/s, aggregate {generated / wall:.1f} tok/s "
            f"({generated} tokens in {wall:.1f} s)"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
