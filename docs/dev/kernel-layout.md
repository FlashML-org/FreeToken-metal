# Kernel layout

Custom kernels live in `python/ftmetal/kernel/`, grouped by what they compute. Tests and benchmarks
mirror the same tree under `tests/kernel/` and `benchmarks/kernel/`. Read this before adding,
moving or renaming a kernel. If a rule here gets in your way, change this file in the same PR.

## Layout

```
python/ftmetal/kernel/
  _ext.py          # loads .metal sources, inlines their #includes, inits the primitives' libraries
  custom.py        # typed mx.fast.metal_kernel
  ref.py           # pure-MLX references shared by several tests
  bindings.cpp     # the _C extension module
  CMakeLists.txt
  common/          # headers used by more than one directory
  attention/
    mha/           # MHA, GQA, MQA
    gdn/           # Gated DeltaNet
    qsa/           # QSA sparse attention
    ...            # one directory per attention type, e.g. dsa/
  moe/             # routing, sorting, slot lookup, expert GEMMs
    affine/  fp/  unquantized/
  gemm/            # matmuls of dense layers
    affine/
  norm/            # RMS norm variants
  rope/            # position encoding
  hc/              # hyper-connection
  sample/          # sampling
  kvcache/         # row reads and writes of the KV and state pools
  embedding/       # embedding lookup
  service/         # CPU-side runtime services, not kernels, built into the same _C
    expert/        # ExpertService, expert_lru, expert_mailbox, victim_cache
    ple/           # PleService, ple_row_store
```

A directory with a `.py` file is a Python package; a format directory that holds only `.metal`
files is not. Don't create a directory until it has files.

## Where a kernel goes

Pick the directory by what the kernel computes, and open a new one only when nothing fits. Most
directories are flat. Only `attention/` has subdirectories per attention type (a new one gets its
own, e.g. `attention/dsa/`), and only `moe/` and `gemm/` have them per quantization format.

Cases that come up:

- Every kernel MoE uses lives in `moe/`, the expert GEMMs (`gather_sorted`) included. `gemm/` is
  for dense layers only.
- A fused kernel lives with the module it was written for. `hc_up_mix` contains an affine GEMV, but
  it serves hyper-connection, so it is in `hc/`.
- A variant that a general kernel adds for one model type stays in the general kernel. QSA's
  `flash_prefill_qsa` is in `attention/mha/mha_prefill`, and `qsa/` holds only kernels unique to
  QSA. Likewise `hc_norm` is a grouped (1 + w) RMS norm, so it is in `norm/offset_rms_norm`.
- A new member of an existing family shares its directory, told apart by a prefix: DeepSeek's mHC
  would be `hc/mhc_*`.

### Quantization formats

`moe/` and `gemm/` split by how the weights are dequantized: `affine/` (MLX affine, 2–8 bit),
`fp/` (mxfp4, nvfp4, mxfp8), `unquantized/` (weights in the input's dtype), and later `lut/` (LUT
quantization). Bit width and dtype are template parameters, not directories.

A format directory holds only that format's code. Code that dispatches across formats sits one
level up: `moe/gather_sorted.{py,h,cpp}` picks its Metal library from `moe/affine/`, `moe/fp/` or
`moe/unquantized/`. A kernel that supports a single format lives entirely in that format's
directory (`gemm/affine/affine_qmv_rows.py`). Format code that both `moe/` and `gemm/` use goes in
`common/quant/<format>.h`.

Files in a format directory are named the way MLX names its kernels, not after the wrapper:
quantized ones start with the format (`affine_`, `fp_`, `lut_`) and unquantized ones have no prefix
(`gather_mm_sorted`), as in MLX's `affine_gather_qmm_rhs`, `fp_gather_qmm_rhs` and
`gather_mm_rhs`. Tests and benches follow the wrapper, so `test_gather_sorted.py` covers all three
formats.

## The files of one kernel

A kernel here means the Metal functions one Python wrapper exposes. Its files sit in one directory
and share the wrapper's name, e.g. `norm/gated_norm.py` and `norm/gated_norm.metal`:

| file | contents |
|---|---|
| `<name>.py` | the wrapper: shape checks, grid, template parameters, the call |
| `<name>.metal` | the function body of its one `metal_kernel`, or a C++ primitive's whole Metal library |
| `<name>_<kernel>.metal` | one function body per file when the wrapper has several `metal_kernel`s |
| `<name>.h` | a Metal header passed as `header=` to this wrapper's `metal_kernel`s |
| `<name>.h`, `<name>.cpp` | a C++ primitive, only when one is needed (see [C++ primitives](#c-primitives)) |
| `<name>_shared.h` | constants and structs a primitive's C++ and Metal share |
| `tests/kernel/<dir>/test_<name>.py` | the test |
| `benchmarks/kernel/<dir>/bench_<name>.py` | the bench |

The test compares against a pure-MLX reference named `<function>_ref`. Put it in the test file, or
in `kernel/ref.py` when several tests use it. Tests of `service/` follow the same tree:
`tests/kernel/service/expert/test_expert_lru.py`.

## Naming

File names are unique across the tree and make sense outside their directory:
`attention/mha/mha_decode.py`, not `attention/mha/decode.py`; `kvcache/pool_rows.py`, not
`kvcache/rows.py`. Directories only group. There are three reasons. `tests/` has no
`__init__.py`, so pytest needs test file names that are unique across all of `tests/`; the sampling
kernel is `sample_tokens` because `tests/engine/test_sample.py` already exists. Metal library names
and `_C.init_<name>_library` must be unique. And a file that moves keeps its test, bench and
library names.

Name what is computed, not a code name: `mha_decode`, not `fdecode`. Domain and type names follow
vLLM and SGLang (`mha`, `gdn`, `moe_topk`, `moe_align`, `indexer`, `mrope`, `kvcache`,
`embedding`). Hyper-connection is `hc`, the abbreviation the Qwen3.8 (`hc_count`) and DeepSeek V4
(`hc_mult`) configs share. A kernel adapted from MLX keeps MLX's name (`gather_qmm`, `gather_mm`),
so it stays easy to diff against upstream when MLX is upgraded.

## Metal sources

All Metal code lives in `.metal` files, never in Python strings. Kernels are compiled at run time,
so there is no include path: `_ext.py` inlines every `#include` before compiling, taking `mlx/...`
from the installed mlx wheel and everything else from `kernel/`. Write includes relative to
`kernel/` (`#include "common/mailbox.h"`). Paths in comments are written the same way
(`kvcache/pool_rows.h`); a file in the same directory can be named alone.

A `metal_kernel` body file holds the function body and nothing else, and the kernel's `name` is the
file name without `.metal`. Variants generated from one body get a suffix after that name
(`gdn_prefill_recurrence_mask`). Helper functions come in through `header=`. The wrapper loads the
body with `metal_source()` and the header with `metal_kernel_header()`, which leaves out the
`utils.h` the `metal_kernel` preamble already has:

```python
SOURCE = metal_source(Path(__file__).with_name("gated_norm.metal"))

KERNEL = metal_kernel(
    name="gated_norm",
    input_names=["x", "weight", "z", "eps"],
    output_names=["out"],
    source=SOURCE,
    header=metal_kernel_header(KERNEL_DIR / "common" / "norm.h"),
)
```

Types and integer constants are `template` parameters. Code fragments that change with the
configuration are `{{NAME}}` markers, which the wrapper (or a primitive's `.cpp`) substitutes
before compiling.

clang-format (`.clang-format`) formats `.h`, `.cpp` and `.mm` files but not `.metal` ones: a
`metal_kernel` body is not a translation unit, and `{{NAME}}` markers would not survive it. Keep
`.metal` files in the same style by hand.

A `.h` with a `.cpp` of the same name is a C++ primitive. A `.h` without one is a Metal header
(`<name>.h`) or a header C++ and Metal share (`<name>_shared.h`). Headers used across directories
go in `common/`, whichever language they are for: `norm.h` (Metal), `mailbox.h` (both),
`metal_primitive.h` (C++).

### Definitions C++ and Metal share

Structs, constants and status codes that both languages use are written once, in a header both
include. One kernel's go in `<name>_shared.h` next to it (`sample/sample_tokens_shared.h`); a
protocol between kernels and a service goes in `common/` (`common/mailbox.h`). Inside, `#ifdef
__METAL_VERSION__` tells the two apart: Metal program-scope constants are `constant constexpr`, the
C++ ones go in `namespace ftmetal`, and macros defined for this are `#undef`ed at the end of the
file.

### Ported code

A ported or adapted kernel's module docstring names the source file and what changed. A `.metal`
or `.h` file adapted from MLX or mlx-lm, function-body files included, starts with the source's
version, file and line numbers and its copyright and license notice, as
`moe/affine/affine_gather_qmm_sorted.metal` does. The MIT license text they refer to is
`kernel/LICENSE-MLX`.

## C++ primitives

Use `mx.fast.metal_kernel` unless it can't do the job. You need a C++ primitive when:

- the output is written in place into an existing buffer;
- several dispatches share a scratch buffer;
- the kernel talks to a CPU-side service through a mailbox.

Besides its own `.h`, `.cpp` and `.metal`, a primitive takes one entry in each of four files.
For `kvcache/pool_rows`:

| file | entry |
|---|---|
| `bindings.cpp` | `#include "kvcache/pool_rows.h"`, `init_pool_rows_library` and the ops |
| `_C.pyi` | their stubs |
| `CMakeLists.txt` | `kvcache/pool_rows.cpp` in the sources |
| `_ext.py` | `_C.init_pool_rows_library(metal_source(...))` |

`service/` is built into the same `_C`, one subdirectory per service. The protocol between a
service and its kernels lives in `common/`.

## Adding a kernel

Using `norm/gated_norm` as the example:

1. Pick the directory and name as described above. It is an RMS norm variant, so it goes in
   `norm/` as `gated_norm`.
2. Write `norm/gated_norm.metal` and `norm/gated_norm.py`. A kernel that needs a primitive also
   gets `<name>.h` and `<name>.cpp` and the four entries above.
3. Write the reference and `tests/kernel/norm/test_gated_norm.py`.
4. If the kernel has a performance target, add `benchmarks/kernel/norm/bench_gated_norm.py`
   ([benchmark.md](../../.claude/skills/benchmark/benchmark.md) says how to measure).
5. Run `uv run pytest tests/kernel/norm -q`. After changing a `.h`, `.cpp` or `.mm` file, run
   `uv sync --reinstall-package freetoken-metal` first.

## Why this shape

The tree is split by kernel type, as vLLM's `csrc/{attention,moe,quantization}/` and SGLang's
`kernels/ops/<group>/` are. MLX and vllm-metal split by language first because they precompile a
metallib, or have several backends for one op. We have one backend, Metal, compiled at run time,
and a kernel's files always change together. The per-format directories follow TensorRT-LLM's
`cutlass_kernels/{fp4_gemm,fp8_rowwise_gemm,int8_gemm}/`, and one `.metal` file per body follows
vllm-metal's `kernels_v2/`.
