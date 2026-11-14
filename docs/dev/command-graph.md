# CommandGraph

How `command_graph/` and `engine/graph.py` capture a decode step once per batch-size bucket and
replay it as one MLX primitive, and what code that runs inside a decode step must do to stay
replayable.

## 1. What it is and why

MLX runs a step on the host one primitive at a time. The Python forward builds thousands of lazy
ops, `eval` walks them, and each primitive's `eval_gpu` picks a kernel, fills its parameters,
allocates its outputs and tracks hazards. MLX also commits the command buffer whenever it holds more
than a set number of dispatches or input elements (`CommandEncoder::needs_commit` in MLX 0.32's
`mlx/backend/metal/device.cpp`). On an M4 Pro (`applegpu_g16s`; the limits follow the last letter of
the GPU architecture name) that is more than 50 dispatches, or input buffers holding more than 50 Mi
elements; `MLX_MAX_OPS_PER_BUFFER` and `MLX_MAX_MB_PER_BUFFER` override them. A MoE decode step,
whose weights fill the second limit, is split into about 85 command buffers. At small batch sizes
this host work takes longer than the GPU work. `mx.compile` removes the Python trace, not the
per-primitive encoding.

CommandGraph records the Metal dispatches of one decode step and replays them as one `GraphReplay`
primitive, which re-issues the recorded encoder calls with this step's buffers. A replay runs no
Python forward, no graph walk, no per-primitive logic, no allocation besides the logits, and no
intermediate commit. It
plays the role of FreeToken's CUDA graphs on the unmodified mlx wheel, with three differences: the
host still encodes every dispatch; step inputs and pools are rebound, not copied into static
buffers; and scratch memory goes into a shared arena that is checked by poisoning (§4), because the
recorder sees neither host writes nor allocator lifetimes. Prefill and sampling stay eager.

Measured on an Apple M4 Pro (20-core GPU, 48 GB) with MLX 0.32.3: Qwen3.6-35B-A3B in MLX affine
4-bit (group 64), experts resident, `metal` attention, page size 1, KV pools from the default budget
(page table 262144 tokens wide, no `--max-seq-len`), `max_running_req` 1. Four 2048-token
natural-text prompts (one document each), 256 greedy tokens per request, the first 8 steps dropped,
so the context runs from about 2.06k to 2.30k tokens. Run in process through the `Scheduler` (like
`benchmarks/e2e/bench_decode_depth.py`, but with these prompts and the default page table), not
over HTTP; medians of four requests. Command buffers and GPU time come from separate runs that
intercept Metal's command buffers, at the same rates; `tools/command_graph_m1.py` reports the same
per-step counts.

| per decode step | CommandGraph on | off (`FTMETAL_COMMAND_GRAPH=0`) |
|---|---|---|
| decode rate | 103.2 tok/s (9.69 ms) | 88.8 tok/s (11.26 ms) |
| host launch (`Scheduler.forward`: forward or replay, sampling, `async_eval`) | 1.2–2.8 ms | 10.9–11.0 ms |
| command buffers committed | 2 | 85–86 |
| GPU busy | 9.6 ms, 98–100 % of wall time | 9.5–9.6 ms, 83–85 % of wall time |

The GPU does the same work either way. Eager decode is bound by the host and replayed decode by the
GPU: +16 % at this depth. Only this depth and batch size 1 were measured. GPU time grows with the
context and host time does not, so the gain should shrink at longer contexts.

## 2. Capture

`command_graph.capture(outputs, step_inputs, pools, externals)` calls `begin_capture`, then
`mx.eval(*outputs)`, then `end_capture`. A failed eval calls `_C.abort_capture` and re-raises.

| argument | contents | at capture | at replay |
|---|---|---|---|
| `outputs` | the lazy logits, then one updated array per pool, in pool order | not evaluated | — |
| `step_inputs` | per-step arrays: token ids, positions, `out_loc`, attention metadata, `moe_step` | evaluated, row contiguous, one buffer each | new arrays of the same shape and dtype |
| `pools` | arrays the step updates in place: KV, GDN state, offload lookup tables | evaluated, one buffer each | passed in, returned aliased |
| `externals` | long-lived buffers: model parameters, offload banks, lookup buffers | evaluated | the captured pointer, not kept alive |

**Hooks.** The first `begin_capture` or `begin_count` of a process installs Objective-C hooks on the
concrete compute-encoder and command-buffer classes of a probe made on MLX's device. They are never
removed. Each hook calls the original method; it records only while a session is open on the calling
thread, so the `eval` must run on the thread that opened the capture. Hooking the probe's concrete
class also covers the wrapper classes of `MTL_DEBUG_LAYER`.

**`begin_capture`** requires that the logits are not evaluated yet and are computed on the GPU, and
that no other capture or count is open on the thread. It then:

- Holds every array of the lazy graph until the graph's blocks are placed (§4). So nothing is
  donated during a capture: MLX normally reuses an input's buffer for an output when nothing else
  holds the input, and here something always does. A capture therefore records copies that eager
  decode avoids by donation (for example same-dtype copies), and an MLX slice update cannot write a
  pool in place.
- Sets MLX's cache limit to 0 and clears the cache, so every scratch buffer is a fresh allocation.
  The memory peak during the captures can exceed the steady state; the startup log line
  `memory after graphs ... peak` shows it.

**What is recorded**, per dispatch: what was set since the previous dispatch (the pipeline, the
`setBuffer` bindings, a copy of each `setBytes` payload, threadgroup memory), the dispatch kind
(threads or threadgroups) with its grid and group sizes, and whether a memory barrier or an encoder
boundary came before it. Replay re-issues these calls in the same order, so like the original it
relies on Metal keeping earlier bindings. MLX's `end_encoding` waits on the fences of earlier
encoders, calls `updateFence:` and then moves `temporaries_` (arrays registered with
`add_temporary`) into the command buffer's completion handler; the recorder reads `temporaries_`
inside its `updateFence:` hook, when the list is complete. `commit`, `encodeSignalEvent:value:`,
`addCompletedHandler:` and `waitForFence:` are only counted. The first failure is kept, the eval
completes, and `end_capture` raises it as `CommandGraph capture failed: ...`.

**Classification.** `end_capture` gives every bound `MTLBuffer` a role. The first role that claims a
buffer wins, and a view takes the role of the buffer it views (a slice of a pool binds as the pool
at an offset).

| role | buffer | a replay binds |
|---|---|---|
| step input, pool | declared by the caller | the replay's array, at the recorded offset relative to it |
| external | declared by the caller | the captured buffer and offset |
| logits | `outputs[0]` | the replay's fresh logits |
| constant | any other evaluated leaf of the graph | the captured buffer, held by the graph; its bytes are kept for the double-capture check |
| intermediate | an unevaluated array of the graph | a block (§4) |
| temporary | an array registered with `add_temporary` | a block, or pinned (§4) |

A buffer that fits no role fails the capture with `end_capture: N bindings are neither whitelisted,
graph arrays nor MLX temporaries`, followed by up to 8 of the bindings. `end_capture` also requires
that the logits are row contiguous, own their buffer and alias no step input, pool or external; that
no two step inputs or pools share a buffer; that every updated pool aliases its pool; that no
dispatch binds a step input, pool or the logits outside the array; and that the capture evaluated
its outputs and recorded at least one dispatch. §7 lists the messages.

**Double-capture check.** `check_same_graph(a, b)` compares two captures of one bucket made with
different step inputs. Per dispatch it compares the pipeline, the kind, the grid and group sizes,
the `setBytes` payloads and the threadgroup memory; per binding, the slot index, the role, the id
(the parameter index, or the order of first appearance within the role) and the offset, plus the
buffer pointer for externals; and the bytes of every constant. A constant whose bytes differ is
reported as an undeclared step input. If the dispatch counts differ it reports only that; otherwise
it lists up to 40 differences by dispatch and kernel name. Steel `GEMMParams` payloads are compared
over 68 of their 72 bytes, because the padding holds the caller's stack. Only tests and tools run
this check.

## 3. Replay

`replay(graph, step_inputs, pools)` checks the counts, shapes and dtypes against the capture; a
graph whose blocks are placed also needs its arena (`graph_replay: the graph's blocks are placed;
bind an arena first`). It builds one lazy `GraphReplay` node:

- Inputs: each step input through `mx::contiguous` (the buffer is shared, not copied, when the array
  is row contiguous and its buffer at most 16 KiB larger than the array), each pool, and the arena.
- Outputs: the logits in a fresh, uninitialized buffer, then each pool aliased to its input
  (`copy_shared_buffer`). Callers rebind the pools to the returned arrays.

`GraphReplay::eval_gpu` encodes into MLX's own concurrent compute encoder:

1. One barrier against earlier work (including a previous replay on the shared arena), and
   `set_input_array` / `register_output_array` for every input, output and the arena, so that MLX's
   fences see them.
2. For each recorded dispatch: a buffer-scope barrier where the capture had a barrier or an encoder
   boundary, the pipeline where the capture set one, the bindings by role (§2), the `setBytes`
   payloads, the threadgroup memory, and the dispatch.
3. A completion handler that holds the `Graph`, because MLX's command buffers do not retain what
   they bind.

A decode graph exceeds MLX's per-buffer op limit, so MLX commits right after it: the replay and the
ops before it fill one command buffer, and sampling a second (the 2 per step of §1). There is no
CPU path (`GraphReplay: GPU only`). Externals are not retained by the graph: a
replay binds the captured pointer, so freeing or reallocating a weight after startup leaves a
dangling binding.

## 4. Arena

A block is one intermediate or temporary `MTLBuffer`, at its full length. `GraphArena` (`arena.py`)
moves the blocks of every graph into one shared buffer:

1. **Poison test** (`add`, after each capture). Replay the capture batch, fill every temporary with
   0xFF and replay again. Only if the logits change, test each temporary alone and pin the ones that
   change them (warning `N temporaries stay pinned`). Some temporaries are written by the host while
   encoding, such as the scalar behind a `metal_kernel`'s `init_value`; a replay never redoes that
   write. Intermediates are not poisoned here; the bind check below covers them.
2. **Plan** (`plan_arena`). A recorded barrier or encoder boundary starts a new segment. Dispatches
   within one segment may run concurrently, so a block lives from the segment of its first binding
   to the segment of its last, and two blocks share bytes only when those ranges are disjoint.
   Blocks are placed largest first, each at its lowest free offset, 256-byte aligned. `lower_bound`
   is the most bytes live in any one segment.
3. **Place** (`place_blocks`). Rebind each block to its arena offset and release the original
   buffer. Only a temporary may stay pinned (offset -1); an intermediate without an offset is an
   error.
4. **Bind** (`bind`, after all captures). One `uint8` arena as large as the largest plan; each graph
   is bound to it, the arena is filled with 0xFF, and each capture batch's replay must give its
   clean logits bit for bit, or the startup fails.

Constants, pinned temporaries and pipelines stay pinned. `GraphArena.nbytes`, the arena plus every
graph's pinned bytes, must fit `graph_reserve_bytes(max_bs)` = 4 MiB + 1.25 MiB per row of the top
bucket (`engine/cache_budget.py`), which the memory budget sets aside before the paged pools. The
4 MiB and 1.25 MiB were rounded up from Qwen3.6-35B-A3B at a 256k-token page table: 1.05 MiB at
batch size 1 and 8.30 MiB at 8, nearly all flash-decode partials. Flash-decode's partition count
follows the page-table width up to 64 partitions (64k keys), so a 256k table is already at the cap:
about 1 MiB per row for 16 query heads of head dim 256 in fp32. A model with more query heads ×
head dim per row can exceed the reserve at larger batch sizes (Gemma4's head-dim-512 global layers
need about 2 MiB per row). Per-replay logits and step inputs,
and the peak during capture, are outside it. The checks compare logits only, on the capture batch: a
read-before-write that shows only in pool contents is not caught.

## 5. How the engine uses it

- **Switch.** On by default. `--disable-command-graph` or `FTMETAL_COMMAND_GRAPH=0`
  turns it off; any other value than `0` or `1` raises.
  The attention backend must declare `BackendInfo(can_capture=True)`. Only `metal` does; with `mlx`,
  decode runs eagerly and the engine logs a warning.
- **Buckets.** `determine_graph_bs(max_running_req)` gives 1, 2, 4, the multiples of 8 below
  `max_running_req`, then `max_running_req` itself: the default 1 gives `[1]`, 6 gives `[1, 2, 4,
  6]`. A model that serves step lookups (`ServesStepLookups`: a host service looks rows up for each
  decode step, such as Qwen3.8-Flash-Next's PLE table rows) cuts the list at its `max_lookup_rows`.
  There is no separate cap like FreeToken's `--cuda-graph-max-bs`.
- **Capture.** Buckets are captured largest first, on rows of the engine's dummy request: its own
  page-table row, every slot on the dummy page, position 0. Padded rows of real steps use GDN's
  `PADDING_SLOT`; capture rows use scratch slots borrowed from the state pool (§6). Before a
  capture: each backend's `init_capture_graph` once, then `prepare_metadata` and
  `prepare_for_capture` per bucket; after the last, `release_capture`.
- **Step inputs**, in order: `input_ids` (lazy: the previous step's tokens, gathered on the device),
  `positions`, `out_loc`, an mrope model's `[3, bs]` positions, each backend's
  `metadata.step_inputs()`, then `moe_step`. **Pools**: every pool's arrays, then the offload
  cache's `mailbox`, `last_use` and `hit_stats`. **Externals**: the parameters, the offload banks,
  `slot_of`, the pair tables and `lookup_buffers()`.
- **Each step.** The scheduler pads a decode batch to the smallest bucket that holds it
  (`pad_batch`) and builds metadata over the padded rows. `forward_batch` replays when `can_replay`
  holds (a decode of at most the top bucket): each backend's `prepare_for_replay`, then `replay`,
  which the step's `async_eval` runs. Padded rows are dropped from the logits (`logits[:size]`).
- **Eager decode.** Graphs off, a backend that cannot capture, or a decode larger than the top
  bucket (which only a step-lookup model above `max_lookup_rows` reaches); such a batch is not
  padded. Prefill is always eager. With graphs on, an offload decode always replays: the scheduler
  admits no more requests than the cache holds in one decode step (`max_decode_rows`, warning `MoE
  offload: N requests run at once`).
- **Startup.** A capture error, a reserve overrun or a mismatch on the poisoned arena raises out of
  `Engine.__init__`. The log has one line per bucket (dispatches, temporaries, poison failures,
  arena against its lower bound, pinned MiB), a summary against the reserve, and `memory after
  graphs`.

## 6. Requirements for code inside a decode step

Two rules cover every kind of code. Per-step values arrive only through declared step inputs or
pools, because pipelines, grids, `setBytes` payloads and constants are frozen at capture
(`check_same_graph`, in the tests). And the step is deterministic and writes scratch before it reads
it, because startup compares replays on a 0xFF arena.

**`mx.fast.metal_kernel`**
- No per-step value in `template=`, in a `{{NAME}}` marker or in the grid: they choose the pipeline
  and the dispatch size, and a difference shows as `pipeline differs` or `grid ... vs ...`.
- No per-step Python scalar in `inputs=`: MLX turns it into a constant, reported as an undeclared
  step input.
- An `init_value` creates a temporary the host writes while encoding; the poison test pins it, which
  costs reserve.

**C++ primitive**
- Encode on `mx::metal::get_command_encoder(stream())` of the step's GPU stream, where temporaries
  are collected (`updateFence: on an encoder that is not MLX's encoder of the capture stream`).
- Use only: the pipeline, `set_input_array` / `set_output_array` / `set_buffer`, `set_bytes`,
  threadgroup memory, `dispatch_threads` / `dispatch_threadgroups` and `barrier()`. A replay
  re-issues nothing else, and a call that is neither recorded nor forbidden silently vanishes at
  replay. The forbidden calls fail the capture (`-[Class selector] is not replayable`):
  `setBuffers:offsets:withRange:`, `setBufferOffset:atIndex:`, the `attributeStride:` variants of
  `setBuffer` and `setBytes`, `useResource(s)`, `useHeap(s)`, `memoryBarrierWithResources:count:`,
  both `executeCommandsInBuffer:` calls, indirect `dispatchThreadgroups`, `setTexture:atIndex:`,
  `setSamplerState:atIndex:`, blit encoders, `encodeWaitForEvent:value:`, and compute encoders made
  without a dispatch type or from a descriptor. A serial compute encoder fails with `a serial
  compute encoder (replay encodes into a concurrent one)`, a dispatch without a pipeline with `a
  dispatch without a compute pipeline state`.
- No host side effect the step relies on: host writes into buffers in `eval_gpu`, completion
  handlers and events run at capture only.
- Per-step values live in arrays, not in `setBytes`: `moe_lookup` reads its real row count from
  `moe_step[0]` (`moe/moe_lookup.metal`). Keep parameter structs free of padding, or zero them
  (`Params params{}` in practice, a `memset` for certain), so two captures cannot differ in bytes
  no kernel reads.
- Scratch comes from `add_temporary` or from an output; anything else cannot be classified.
- Pools are updated in place with `copy_shared_buffer` from the pool input, never through
  `mx::contiguous`, which may copy (`end_capture: updated pool k does not alias pool k`).

**Model layer**
- No `item`, `tolist`, `np.array` or `mx.eval` of step data in the decode forward. The forward is
  built before the capture opens, so such a result, and any branch on it, is frozen as a constant.
- No op off the default GPU stream (`begin_capture: <primitive> runs off the capture's GPU stream`).
- A new per-step array joins `GraphRunner.step_inputs` or a metadata record's `step_inputs()`,
  evaluated, in its own buffer. `test_replayed_mrope_step_captures_the_same_graph` leaves one out
  and expects the double-capture check to report the constant. Python ints that shape kernels stay
  constant per bucket.
- Per-step state lives in a pool and is written with `kernel/kvcache/pool_rows` (`scatter_rows`,
  `kv_write`, ...): an MLX slice update writes in place only by donation, which a capture disables.
- Weights in `model.parameters()` are externals and must not be replaced after startup (§3). Other
  long-lived arrays, such as RoPE tables, become constants.
- The logits are row contiguous, own their buffer and are fully written.

**Attention backend**
- Declare `can_capture=True` only when decode runs at fixed shapes and no per-step host value shapes
  a kernel. The `mlx` backend slices by `seq_lens_host[0]`, so it cannot capture.
- Hooks (`attention/base.py`): `init_capture_graph` fixes the metadata widths and borrows what the
  capture batches need; `prepare_for_replay` builds the metadata arrays at the bucket's shapes;
  `prepare_for_capture` may give the dummy rows private state; `reset_capture_state` resets it
  before each check replay; `release_capture` returns what was borrowed.
- Full attention binds whole page-table rows (sliding-window attention: window-wide rows), and
  `init_capture_graph` asserts that `max_seq_len` is the table width (`a graph spans whole table
  rows`). Flash-decode's `max_pages` payload comes from the table width and its partition count from
  `max_seqlen_k` and the batch size, so both are pinned per bucket. Entries past a request's keys
  are stale but in bounds and never read (`test_replay_metadata_matches_reference_and_eager`).
- Every metadata array is in the record's `step_inputs()`; `*_host` values and `max_seqlen_*` are
  frozen.
- Dummy rows that would race on read-modify-write state get their own. `LinearAttnBackend` borrows
  one scratch GDN slot per row of the largest bucket from the state pool until `release_capture`, so
  the pool must have them free at startup, and zeroes them before every check replay.

**Pools** are row contiguous, one buffer each, of the capture's shape and dtype, and updated in
place. A replay checks only shape and dtype, so a pool may be a new array each step; at capture each
pool needs its own buffer (`end_capture: pool k shares its buffer with a step input or another pool
(or has none)`), and at replay it must be row contiguous (`GraphReplay: pool k is not row
contiguous`).

**Padded rows** run the dummy request and are dropped. They must not change the real rows' results
(`test_padded_eager_decode_matches_unpadded`). A kernel with effects beyond its own rows counts the
real rows from a step input, not from the padded size: `moe_lookup` takes them from `moe_step[0]`
(`tests/kernel/moe/test_moe_lookup.py`, replayed against eager in
`tests/models/test_parity_offload.py`).

**MoE offload and mailbox kernels**

| buffer | role | why |
|---|---|---|
| expert banks, `slot_of`, `lookup_buffers()` | externals | long-lived, written in place by a CPU service |
| pair tables | externals | long-lived and read only |
| `mailbox`, `last_use`, `hit_stats` | pools | updated in place and returned aliased, so each step's lookups are ordered after the previous step's |
| `moe_step` = (real rows, step), int32 [2] | the last step input | a new array each forward, read at run time |

- Captures post nothing: `prepare_for_capture` begins a lookup of 0 real rows.
- No slot a step reads may be rewritten within that step, so the scheduler caps concurrent requests
  at the cache's `max_decode_rows` and `begin_lookups` raises above it (§5).
- Before a forward, the engine arms the offload cache (except in overlap mode, where a long prefill
  reads each layer's experts ahead; `--disable-moe-prefill-overlap` turns it off) and, on decode
  batches, a step-lookup model's lookups; the scheduler checks them once the tokens are on the host.
  The kernels wait inside the replay's command buffer while the CPU service fills the externals
  (replayed chains: `tests/kernel/moe/test_moe_lookup.py`,
  `tests/kernel/moe/affine/test_affine_glu_pending.py`,
  `tests/kernel/embedding/test_ple_lookup.py`).

## 7. Debugging, tests and scripts

```
uv run pytest tests/command_graph tests/engine/test_graph_runner.py -q   # no checkpoint needed
uv run pytest tests/ -q -k "replay or captur or graph"                 # every related test
FTMETAL_TEST_MODEL_PATH=/path/to/model uv run pytest tests/models/test_real_checkpoint.py -q
uv run python tools/command_graph_m1.py /path/to/Qwen3.6-35B-A3B \
    [--context 2048] [--steps 18] [--timing-steps 64] [--attention-backend metal] [--page-size 1]
uv sync --reinstall-package freetoken-metal                            # after editing csrc/
```

| test | covers |
|---|---|
| `tests/command_graph/test_command_graph.py` | a synthetic step (`TinyStep`): replay equals eager with fresh logits; the double capture accepts new inputs and catches a Python int and a `setBytes` change; an unclassified buffer; arena plans (50 random tapes, a residual stream); placement and the poisoned replay; the poison test finding an `init_value` |
| `tests/engine/test_graph_runner.py` | a tiny hybrid model in a real engine: the ladder; replay equals eager and the reference across bucket switches and page sizes; one tight shared arena; reserve and capture failures fail the startup; a backend that cannot capture; padding; the double capture of a real padded step |
| `tests/attention/test_full_attn_metal.py` | replay-width metadata gives the per-length results (`test_replay_metadata_matches_reference_and_eager`, and its sliding-window twin) |
| `tests/kernel/moe/test_moe_lookup.py`, `tests/kernel/moe/affine/test_affine_glu_pending.py`, `tests/kernel/embedding/test_ple_lookup.py` | capture and replay of the mailbox chains |
| `tests/models/test_parity_*.py` | graphs on and off per model family; `qwen3_5_vl` double-captures an mrope step; `offload` and `flint` replay offload decode against eager; `qwen4_exp` checks that the ladder stops at one lookup post (`test_decode_graphs_stop_at_one_lookup_post`) |
| `tests/models/test_real_checkpoint.py`, `tests/server/test_server.py`, `tests/engine/test_cache_budget.py` | replay on a real checkpoint, the flag over the environment variable, the `graphs` line of the budget |

`tools/command_graph_m1.py` is the check on a real checkpoint. It runs batch size 1 eagerly against
the runner's replay (logits and tokens must match bit for bit), runs the double-capture check,
prints graph and arena statistics, and counts per-step dispatches, command buffers and host time
(`_C.begin_count` / `end_count`, with a busy wait). It exits non-zero unless the double capture
passes, the logits are bit-exact and the timing run's tokens match. It needs a hybrid GDN model and
bypasses `forward_batch`, so it covers neither MoE offload nor step lookups.

| symptom | cause |
|---|---|
| `begin_capture: ...` | an argument breaks the contract of §2, or an op runs off the capture's GPU stream |
| `begin_capture: the logits are already evaluated` | something evaluated the step before the capture |
| `CommandGraph capture failed: -[Class selector] is not replayable` | a forbidden call (§6); no kernel is named, so bisect with a `TinyStep`-style capture |
| `end_capture: N bindings are neither whitelisted, graph arrays nor MLX temporaries` | scratch not registered with `add_temporary`, or an unrelated array evaluated with the step |
| `end_capture: the logits must own their buffer` | an intermediate or temporary lives in the logits buffer |
| `end_capture: dispatch d (...) binds its <role> buffer outside the array` | a kernel binds past the end of a step input, pool or the logits |
| `end_capture: the capture recorded no dispatch` | the outputs were already computed, or computed on another thread |
| `a replay on the poisoned arena differs` | a block read before it is written, or a non-deterministic kernel |
| replay differs from eager | compare with `FTMETAL_COMMAND_GRAPH=0`, then double-capture a real step |

To double-capture a real step, as `test_padded_decode_step_captures_the_replayed_graph` does, run
each backend's `prepare_for_replay(batch)`, then `check_same_graph(runner.graphs[batch.padded_size],
runner.capture(batch))`. That capture runs the step and writes the pools.

## 8. The MLX dependency

The recorder relies on MLX internals: private members of `CommandEncoder` and `Device`, the order in
which `end_encoding` updates its fence and hands over its temporaries, the layout of steel
`GEMMParams`, and the Metal classes MLX creates. So mlx is pinned; read
[mlx-internals.md](mlx-internals.md) before upgrading it.
