# Profiling Metal work

How to find where GPU time goes in ftmetal and why a kernel is slow, from the command line. The
tools live in `tools/profiling/`; run the scripts from the repository root with
`uv run python tools/profiling/<script>.py`. Measuring whether something got faster is the
benchmark skill's job (`.claude/skills/benchmark/benchmark.md`); this file is for when a number is
bad and the cause is unknown.

## 1. Setup

- `sh tools/profiling/build.sh` builds `libmtlstat.dylib` and `pso_info` into
  `~/.cache/ftmetal/profiling/` with the Command Line Tools' `cc`. Run it before first use, after a
  macOS update and after editing `mtlstat.m` or `pso_info.m`.
- `xctrace`, the offline Metal compiler and opening a `.gputrace` need a full Xcode with its Metal
  toolchain; nothing else does. If `xcode-select -p` points at the Command Line Tools, set
  `DEVELOPER_DIR=/path/to/Xcode.app/Contents/Developer` in the shell rather than switching the
  system default.
- The benchmark skill's exclusive-GPU and one-model rules apply (`benchmark.md` §2). Outputs go
  outside the repository; the examples use `OUT=/tmp/ftmetal-profile` (`mkdir -p $OUT`).

## 2. Which tool

| question | tool |
|---|---|
| which kernel takes the GPU time of a step, a bench or a request; did a change move it | `kernel_table.py` (§3) |
| where the GPU idles: host stalls, host/GPU overlap, CPU-to-GPU latency, other processes | `xctrace_intervals.py` (§3) |
| dispatches, commits and GPU busy % of one process, a live server included | mtlstat (§5) |
| how close a kernel is to the hardware limit | `kernel_bench.py` (§4) |
| why a kernel is slow: the limiting unit, occupancy, bandwidth | `xctrace_counters.py` (§4) |
| per-dispatch time, per-line shader cost | `gputrace_capture.py`, then Xcode (§4) |
| threadgroup cap, SIMD width, threadgroup memory; does the MSL compile, what did the compiler make of it | `pso_info`, `xcrun metal` (§4) |
| GPU busy of the whole machine, nothing restarted | `ioreg` (§5) |

## 3. Workflow A: which kernel or phase is slow

1. Make the workload a command that exits normally: call `mx.synchronize()` before exiting
   (`os._exit` skips mtlstat's final flush).
2. Per-kernel table. `kernel_table.py` runs the command under mtlstat with
   `MLX_MAX_OPS_PER_BUFFER=0`, so every MLX primitive commits its own command buffer and a buffer's
   GPU interval is that primitive's time:

   ```bash
   uv run python tools/profiling/kernel_table.py --json $OUT/before.json -- \
       uv run python -m benchmarks.kernel.attention.mha.bench_mha_decode --lengths 2048
   uv run python tools/profiling/kernel_table.py --diff $OUT/before.json $OUT/after.json
   ```

   - `total_us` is the GPU-clock time of that kernel's command buffers, each with about 1.9 us of
     fixed cost (`--overhead-us 1.9` subtracts it). One kernel per buffer also loses the overlap
     of independent kernels and adds a commit per primitive on the host: trust the GPU columns.
     busy/wall spans the whole run, setup and first-use compiles included, so it shows host
     starvation only for a steady-state command run with `--no-isolate` (MLX's own buffers; rows
     become kernel groups). For a step's busy %, take mtlstat deltas over a steady window (§5).
   - `(empty)` rows are buffers without a dispatch, some taking GPU time and some none: under MLX
     the end-of-eval, in-flight-throttle (§6) and `mx.synchronize` commits. A row naming two
     kernels is one primitive encoding two dispatches: split it by ablation.
   - `mx.fast.metal_kernel` kernels appear as `custom_kernel_<name>_<hash>_<dtypes>`, so template
     and dtype variants are already apart; give an experimental copy of the source its own `name=`.
     With several Metal processes (`uv` is one), the table reads the largest log.
   - The workload must follow `--` itself: a command started through a SIP-protected binary
     (`sh -c`, `/usr/bin/env`, `/usr/bin/time`, the system Python) loses `DYLD_INSERT_LIBRARIES`
     and logs nothing.
3. Under CommandGraph replay a decode step commits two command buffers: the replayed graph, one
   MLX primitive and so one row naming every kernel, and eager sampling. Profile the step's kernels
   with `FTMETAL_COMMAND_GRAPH=0`: the same kernels and GPU work, except the copies a capture adds
   and the padding of a batch to its bucket (`docs/dev/command-graph.md` §1-3).
4. Timeline, when the question is where the GPU idles:

   ```bash
   uv run python tools/profiling/xctrace_intervals.py record --out $OUT/x.trace --time-limit 5 -- \
       .venv/bin/python workload.py
   uv run python tools/profiling/xctrace_intervals.py summary $OUT/x.trace \
       --process python --gaps 10
   uv run python tools/profiling/xctrace_intervals.py export $OUT/x.trace \
       --table metal-application-intervals --tsv $OUT/app.tsv
   ```

   xctrace records the host-side tables only for the process it launches, so launch the
   interpreter, not `uv run` (the script refuses `uv`). It records one interval per encoder (about
   one per command buffer under MLX; per-dispatch times need a `.gputrace`) and slows GPU work by
   about 8 %. The limit (seconds) or Ctrl-C ends the recording, saves it and kills the launched
   process, so give the workload a loop that outlasts the limit. It shows where a gap is, not why;
   take busy % from an untraced mtlstat run. One recording at a time: a second fails with
   `_lockKPerf: could not lock kperf` and the first loses data from then on. `tables` lists a
   trace's tables (`--help` names the useful ones; `time-profile` holds CPU samples).

## 4. Workflow B: why a kernel is slow

1. Isolate the call as a `--setup`/`--expr` pair, or a short script that warms up (§6).
2. Time it and place it on the roofline:

   ```bash
   uv run python tools/profiling/kernel_bench.py \
       --setup "a = mx.random.normal((4096, 4096)).astype(mx.bfloat16)" --expr "a @ a" \
       --bytes $((3 * 4096 * 4096 * 2)) --flops $((2 * 4096 ** 3)) --gpu
   ```

   Wall time (best round, fenced by `mx.synchronize`) blocks on every `mx.eval`, so each call pays
   a full submit-to-completion round trip, about 100 us here: it overstates any kernel under a few
   milliseconds and is not what a pipelined step pays. `--gpu` adds the GPU-clock time from
   command-buffer timestamps: the kernel itself, and what goes on the roofline. The script first
   runs a matmul loop for `--warmup-s` (1 s), because an idle GPU clocks down (§7); an input small
   enough to stay in the GPU's caches can read above the DRAM ceiling. `--time-us X`
   places an existing measurement; `--peak-bw` (GB/s) and `--peak-tf` (TF/s) override the ceilings.
   Above 85 % of the relevant ceiling, stop: the kernel is done. To time a kernel the way a layer
   calls it, chained, use its micro bench under `benchmarks/kernel/` (`benchmark.md` §3).
3. Ablation ladder, each step a copy of the kernel timed as in step 2: drop the global loads, drop
   the MMA or ALU work, drop the barriers, vary the threadgroup size and threadgroup memory, add
   `[[max_total_threads_per_threadgroup(N)]]` (if the time moves, registers are the limit).
4. Hardware counters, headless. Metal System Trace records counter profile 0, on M4 Pro only "RT
   Unit Active"; `make_counter_template.py` copies the template with profile 13, about 85 counters
   (occupancy, every limiter and utilization, L1 residency and bandwidth, GPU read and write
   bandwidth). Profile 14 is ray tracing. Any other profile, or a second counters instance added
   with `--instrument 'Metal GPU Counters'`, is rejected and empties every counter table.

   ```bash
   uv run python tools/profiling/make_counter_template.py --profile 13 \
       --out $OUT/mst13.tracetemplate
   uv run python tools/profiling/xctrace_intervals.py record --template $OUT/mst13.tracetemplate \
       --out $OUT/c.trace --time-limit 2 -- .venv/bin/python loop.py
   uv run python tools/profiling/xctrace_counters.py $OUT/c.trace --process python
   ```

   The counters are global to the GPU: run the kernel alone, in a loop (`--process` drops the
   samples outside its GPU intervals). Record 1-3 s: profile 13 writes about 0.15 GB of trace and
   0.55 GB of export XML per second, and the export takes minutes. With `--shader-profiler 1`,
   profile 13 also records a per-kernel Shader Timeline (`metal-shader-profiler-intervals`), but
   per-line samples stay empty: those need Xcode.

   The highest *Limiter* names the bound unit: F32/F16/Integer for the ALU, L1 Cache for loads,
   stores and threadgroup memory, Last Level Cache for DRAM-side traffic (compare GPU Read and
   Write Bandwidth with the ceiling, §7); the others are Control Flow, MMU and Instruction
   Throughput. A utilization below its limiter means the unit stalls. Low Kernel Occupancy with
   high L1 Register and Stack Residency is register pressure; otherwise check, in order (tech talk
   111374): Compute Shader Launch Limiter, threadgroup memory, Occupancy Manager Target, L1
   Eviction Rate, L1 residency by memory type.
5. GPU trace, for per-dispatch time and per-line shader cost:

   ```bash
   uv run python tools/profiling/gputrace_capture.py --out $OUT/k.gputrace \
       --setup "..." --expr "..."
   open -a Xcode $OUT/k.gputrace
   ```

   `--call module:function` captures what a function returns instead (`PYTHONPATH=.` from the
   repository root). From other code: set `MTL_CAPTURE_ENABLED=1` before the Metal device exists
   and wrap one warmed-up evaluation in `mx.metal.start_capture(path)` / `mx.metal.stop_capture()`.
   Every MLX capture is at least about 190 MB: it embeds `mlx.metallib`. In Xcode's Metal
   debugger: Summary, then Performance (per-encoder and per-dispatch GPU time and counters;
   Control-click a column header for median, maximum and total columns, filter to find a
   counter), select a dispatch, Pipeline Statistics, Shader Profiler on the function (cost per
   line: ALU, memory, control flow, synchronization), Shader Cost Graph. A trace replays only on
   the same GPU and OS version.
6. Compiler view. A C++ primitive's `.metal` is a whole library, usually with `{{KEY}}` markers its
   `.cpp` fills; a `metal_kernel`'s holds only the function body (`docs/dev/kernel-layout.md`, "The
   files of one kernel") and compiles only through MLX: call the kernel with `verbose=True` to print
   the source MLX compiles. Substitute every key first (`grep -o '{{[A-Z_]*}}' k.metal | sort -u`;
   MSL spells bf16 `bfloat`) and pass the include roots:

   ```bash
   I="-I python/ftmetal/kernel -I $(echo .venv/lib/python3.*/site-packages/mlx/include)"
   ~/.cache/ftmetal/profiling/pso_info k.metal [kernel] $I [--sub KEY=VALUE ...]
   xcrun -sdk macosx metal -std=metal4.0 -fno-fast-math $I -c k.metal -o k.air  # as MLX: safe math
   xcrun -sdk macosx metal -std=metal4.0 -fno-fast-math $I -S -emit-llvm k.metal -o k.ll  # IR
   xcrun metal-nm k.air           # kernels, threadgroup statics, intrinsics used
   xcrun metal-objdump -d k.air   # IR; addrspace(3) is threadgroup traffic
   ```

   MLX compiles every runtime source with safe math (Metal 4.0 on macOS 26); the compiler's
   default is fast math, which changes the code (`exp` becomes `fast_exp`). `pso_info` compiles
   the same way and prints `maxTotalThreadsPerThreadgroup` (below 1024 and below any
   `[[max_total_threads_per_threadgroup(N)]]` the kernel declares: the compiler capped the
   threadgroup for registers), `threadExecutionWidth` and `staticThreadgroupMemoryLength`. It sets
   no function constants, so it lists the kernels that need them (the gather and sampling
   libraries) without building them. Native AGX instructions, register counts and spill reports
   are not available offline.

## 5. mtlstat and GPU busy

`libmtlstat.dylib` is a `DYLD_INSERT_LIBRARIES` library for any Metal process (MLX, PyTorch MPS,
llama.cpp) that reads each command buffer's GPU start and end time on completion.

| variable | meaning |
|---|---|
| `MTLSTAT_DIR` | where it writes (default `/tmp`); if it cannot, it prints one error and hooks nothing |
| `MTLSTAT_CBLOG=1` | also log every command buffer |
| `MTLSTAT_FLUSH_MS` | flush interval (default 20); it also flushes at exit |
| `MTLSTAT_VERBOSE=1` | print each process's pid and the number of hooked methods at load |

`mtlstat.<pid>.tsv` appends the cumulative counters at every flush: `t_host_s dispatches commits
cb_done gpu_busy_s gpu_sum_s` (busy is the union of the buffers' GPU intervals, sum their plain
sum, on the clock of `t_host_s`). `cblog.<pid>.tsv` has one line per command buffer:
`gpu_start_s gpu_end_s n_dispatch kernels`. Dispatches are not ops: MLX fuses some (an add folded
into a norm is one dispatch), so compare counts between two builds, not against a list of ops.

For a live server, read the deltas of the server's `mtlstat.<pid>.tsv` over a request window
(under `uv run` the dylib also loads into `uv`, whose file stays at zero):

```bash
mkdir -p /tmp/mtlstat
DYLD_INSERT_LIBRARIES=~/.cache/ftmetal/profiling/libmtlstat.dylib MTLSTAT_DIR=/tmp/mtlstat \
    MTLSTAT_VERBOSE=1 uv run ftm serve ...
```

Machine-wide GPU busy without sudo (`powermetrics` needs root; there is no unprivileged power or
clock reading):

```bash
ioreg -r -d 1 -c IOAccelerator | grep -o '"Device Utilization %"=[0-9]*'
```

## 6. How MLX turns ops into GPU work

- `mx.eval` walks the lazy graph and encodes each primitive's dispatches into the stream's open
  command buffer through one concurrent compute encoder, with barriers only between dependent
  dispatches: independent kernels overlap, dependent ones serialize.
- MLX commits the buffer when it holds too many dispatches or binds too many elements
  (`docs/dev/command-graph.md` §1 has the limits; `MLX_MAX_OPS_PER_BUFFER=0` commits every
  primitive).
- `mx.eval` also throttles: once more than 10 command buffers are in flight it commits the open
  buffers, even empty ones, and waits for one to finish. A profiler shows that wait as host time in
  `mx.eval`; it is GPU back-pressure.
- `mx.synchronize()` commits and waits for the calling thread's stream (MLX 0.32 keeps one per
  thread): the fence for wall-clock timing, called on the thread that ran the work. `mx.eval(x)`
  returns once `x` is computed, so evaluating the last result is a fence too.
- Every `mx.fast.metal_kernel` and each configuration of ftmetal's C++ libraries is compiled from
  source at first use (tens to hundreds of ms); MLX's own kernels ship in `mlx.metallib` but still
  build their pipelines at first use. Warm up before timing or capturing.
- `MTL_SHADER_VALIDATION=1` and `MTL_DEBUG_LAYER=1` check shaders and API use at a large cost:
  correctness only, never with timing.

## 7. M4 Pro numbers (20-core GPU)

- DRAM 254-260 GB/s streaming (273 nominal); bf16 simdgroup-matrix peak 7.8 TF/s (MLX's 4096³
  GEMM reaches 7.6); machine balance about 30 FLOP/byte. `kernel_bench.py` defaults to 257 GB/s
  and 7.8 TF/s.
- An idle GPU clocks down: a 4 MB multiply read 35.7 us of GPU time cold and 9.5 us after a second
  of matmuls. Warm the GPU before timing small kernels.
- A command buffer costs about 1.9 us of fixed GPU time on a warm GPU (6.4 cold), a trivial
  dependent dispatch inside one about 1.3 us, a typical small decode kernel 4-6 us: fusing small
  kernels pays.
- How many threadgroups run at once depends on the kernel's threads, threadgroup memory and
  registers (about 48 for a 128-thread flash-decode kernel with 20 KB of threadgroup memory); a
  grid just past a multiple of it leaves a tail (flash decode at 2k ran faster on 24 split-K
  partitions than on 64). Occupancy drops past about 27-29 KB of threadgroup memory per
  threadgroup.
- Above the machine balance a fused kernel can lose to its parts: attention prefill at head dim 512
  reached 4.4 TF/s fused, 6.4 as separate GEMMs. Put both paths on the roofline before fusing.

## 8. References

- Metal debugger: https://developer.apple.com/documentation/xcode/metal-debugger (capture,
  replay, performance, counters, shader profiler and shader cost graph pages hang off it). Apple
  also documents a `gpudebug` CLI for `.gputrace` files, which Xcode 26.6 does not ship.
- GPU counters and counter sample buffers:
  https://developer.apple.com/documentation/metal/gpu-counters-and-counter-sample-buffers
- `MTLComputePipelineState`:
  https://developer.apple.com/documentation/metal/mtlcomputepipelinestate
- Metal Shading Language Specification:
  https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf
- WWDC20 10603 "Optimize Metal apps and games with GPU counters": occupancy, limiters, bandwidth.
- Tech talk 111374 "Discover new Metal profiling tools for M3 and A17 Pro": occupancy triage and the
  counters profile 13 records.
- Tech talk 111375 "Explore GPU advancements in M3 and A17 Pro": dynamic register allocation and
  why occupancy follows L1 residency.
- `man xctrace` (with `DEVELOPER_DIR` set) documents the CLI; the Apple developer pages render with
  JavaScript, so fetch `https://developer.apple.com/tutorials/data/documentation/<path>.json` to
  read one as text.

Verified on an M4 Pro with macOS 26.5.2, Xcode 26.6 and mlx 0.32.3. When a tool behaves
differently on a newer macOS, Xcode or MLX, update this file and say since when.
