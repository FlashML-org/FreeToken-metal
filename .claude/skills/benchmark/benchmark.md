# Benchmarking

How to measure a performance change in ftmetal, and the rules every reported number follows. The
benchmarks live in `benchmarks/`; run them from the repository root with `uv run python -m ...`.

## 1. Two kinds of bench

| kind | where | answers |
|---|---|---|
| kernel micro bench | `benchmarks/kernel/<same tree as python/ftmetal/kernel>/bench_<name>.py` | is this kernel faster than what it replaces, and how close is it to the hardware |
| end-to-end bench | `benchmarks/e2e/` (in process), `benchmarks/bench_serving.py` (over HTTP) | does the gain reach a decode step, a prefill, a served request |

A performance change usually needs both:

- After changing a kernel, run its micro bench first to confirm the gain at the kernel, then an
  end-to-end bench to confirm it reaches the model. A small kernel change stops at the micro bench,
  one end-to-end run at 8k and the correctness check (§5); a real optimization reports every depth.
- Some kernels synchronize with the CPU: `moe_lookup`, `affine_glu_pending` and `ple_lookup` wait on
  a CPU service through a mailbox. A micro bench measures only their kernels: bench them as part of
  the chain they belong to (`bench_affine_glu_pending.py` times a whole offload MoE layer with the
  mailbox shut, so no CPU answers), and measure the service's cost end to end.
- Small kernels (norms, rope, routing, gathers) cost a few microseconds each, and launch overhead
  dominates them alone. Bench them together, as the module they serve (a layer's decode tail, say),
  or rely on the end-to-end decode rate.

## 2. Rules for every measurement

- **The GPU is exclusive.** Nothing else runs on it: no pytest, other benches, servers or model
  loads. Check before starting
  (`pgrep -fl 'ftm serve|pytest|bench_|aime0|mlx_lm|llama-server|ollama|LM Studio'` prints
  nothing), and rerun a row that something else disturbed.
- **One model-loading process at a time**, and check free memory before loading one.
- **Depths.** Report prefill and decode at 2k, 8k and 32k of context; add others on top.
- **State the setup** with every table: the machine, the MLX version, the commit, the model and its
  quantization, the KV capacity the server got (`--num-token`; without it, the startup line `KV
  cache: N pages x P tokens`: everything the budget leaves on a resident model, 65536 tokens before
  an offload cache), whether experts are resident or offloaded (and the cache size),
  `max_running_req`, and anything switched off (`FTMETAL_COMMAND_GRAPH=0`, ...).
- **Real workloads.** Prompts are real samples at their natural length: no filler, repetition,
  truncation or random tokens when the result depends on the data. **Never benchmark MoE offload
  with random prompts**: random tokens route experts unlike text and give an unrealistically low
  expert-cache miss rate. Use real data (ShareGPT, AIME, GSM8K, real long documents).
- **Comparing engines.** One harness (`bench_serving.py`), API servers only, every engine in the
  same session with fresh runs; never mix in an archived row or derive one metric from another.
- **Aggregates.** Micro benches report the best of their repeats (interference only slows a run).
  End-to-end and serving numbers report medians over several requests, warm-up excluded. Record
  negative results too.

## 3. Kernel micro benches

```
uv run python -m benchmarks.kernel.attention.mha.bench_mha_decode --lengths 1024,8192,32768
uv run python -m benchmarks.kernel.sample.bench_sample_tokens
```

- One bench per kernel wrapper that has a performance target, at the shapes of the models it
  serves (Qwen3.6, Gemma4, ...). Layout and naming follow `docs/dev/kernel-layout.md`.
- Timing: `benchmarks/timing.py` (`best_seconds_per_call`) chains N calls through their output, so
  they run one after another on the GPU as layers do, evaluates the run once, and reports the best
  mean per call over the repeats; one call before timing compiles the kernels. The time includes
  building the calls' lazy graph on the host, which matters for kernels of a few microseconds.
- Report four things: microseconds per call; the path the kernel replaces, timed the same way (the
  MLX op or composition of MLX ops, or the previous version when optimizing our own kernel); the
  fraction of the measured ceiling (DRAM bandwidth for decode kernels, matrix throughput for
  prefill GEMMs, each measured once per machine with a large streaming copy and a large bf16
  matmul); and the worst relative error against the reference (`kernel/ref.py`).
- To see why a kernel is slow (per-kernel GPU time, occupancy, counters), profile it with the
  tools in `tools/profiling/`, as `.claude/skills/metal-profiling/profiling.md` describes.

## 4. End-to-end benches

**In process** (`benchmarks/e2e/`): `bench_decode_depth.py` (decode rate at given depths) and
`bench_prefill.py` (prefill rate) drive the `Scheduler` directly, without HTTP:

```
uv run python -m benchmarks.e2e.bench_decode_depth /path/to/model [--depths 2048,8192,32768]
uv run python -m benchmarks.e2e.bench_prefill /path/to/model [--depths 2048,8192,32768]
```

They cap the page table at the longest depth (a server's spans the model's maximum position or
the pool), which changes flash-decode's partitions, so do not compare their numbers with a
server's. Both fill their prompts with random tokens: fine for a resident model, not for MoE
offload, whose prefill and decode depend on the routing.

**Over HTTP** (`benchmarks/bench_serving.py`): measures any server with an OpenAI chat completions
endpoint (`ftm serve`, `mlx_lm.server`, `llama-server`, Ollama, LM Studio, ...) the same way. It
starts no engine; start the server under test yourself, alone on the machine.

```
uv run python -m benchmarks.bench_serving --dataset sharegpt --tokenizer /path/to/model \
    --prepare-only
uv run python -m benchmarks.bench_serving --base-url http://127.0.0.1:1919 --dataset sharegpt \
    --tokenizer /path/to/model --ctx-lens 2048,8192,32768 --num-prompts 4 --max-tokens 256 \
    --label "ftm serve --num-token 65536, experts resident" --output results.jsonl
```

- It sends streaming chat completions one at a time. Prefill tok/s is prompt tokens over the time
  to the first generated token (content or reasoning); decode tok/s is completion tokens minus one
  over the time from the first token to the last, so decode never includes the prefill.
- Datasets (pinned revisions): `random` (never for MoE offload), `aime`, `gsm8k`, `sharegpt`. A real
  dataset runs, for each of `--ctx-lens`, the samples whose natural length is closest to it, the
  seed choosing among those within `--tolerance`, and prints a WARNING about samples beyond it;
  AIME and GSM8K are short, and ShareGPT rarely reaches 32k. Pass the served model's `--tokenizer`:
  lengths are then counted (content tokens plus a few per message) instead of guessed from
  characters, and the report flags a server whose prompt length differs, as one that truncated
  the prompt would. The prompt lengths reported are the server's own.
- Run `--prepare-only` before starting the server: the first ShareGPT conversion needs a few GB of
  memory. During a run the client stays under half a GB.
- No two prompts open with the same text, so none hits another's cached prefix. A second run on
  the same server instance needs another `--seed`, or a restarted server when the WARNING says
  samples lie beyond the tolerance (the seed cannot change those). Start `ftm serve` with
  `--enable-cache-report` so the report can flag prefix-cache hits; other servers report them in
  `usage` when they support it. Hits under 1 % of the prompt (a shared chat template) are ignored.
- A stream that reports an error or ends without a finish reason stops the run, with the
  server's message; the requests finished before it are already in `--output`. Requests to the
  server bypass HTTP proxies.
- `--extra-body` passes engine-specific fields, e.g. `'{"ignore_eos": true}'` for a fixed output
  length where the server supports it. `--label` records the server's settings in the output.
- Engines need their context set to cover `--ctx-lens`: Ollama through a Modelfile `num_ctx`
  (it wins over `OLLAMA_CONTEXT_LENGTH`), `llama-server` through `-c`. Give `mlx_lm.server`
  `--prompt-cache-size 1` at very long contexts.
- Each 32k request prefills its whole prompt (about 40 s for Qwen3.6-35B-A3B 4-bit under `ftm serve`
  on an M4 Pro), so keep `--num-prompts` small there. Prefill beyond 32k need not be measured.

## 5. Correctness with every performance change

A faster kernel that changes results is a regression. Run the tests that cover it
(`uv run pytest tests/ -q`) and the model check: `uv run python tools/aime0_check.py /path/to/model`
must answer AIME25 #0 with 70.
