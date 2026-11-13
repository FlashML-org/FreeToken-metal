# kernel/ layout

Applies to the custom kernels under `python/ftmetal/kernel/`, and to their tests and benches.
Adding, moving or renaming a kernel follows this document; to change a rule, change this document
first.

## 1. Directory tree

```
python/ftmetal/kernel/
  _ext.py  custom.py  ref.py  bindings.cpp  CMakeLists.txt  LICENSE-MLX
  common/              headers shared across domains
  attention/
    mha/               MHA, GQA, MQA
    qsa/               QSA sparse attention
    gdn/               Gated DeltaNet
  moe/                 MoE: routing, sorting, slot lookup, expert GEMMs
    affine/  fp/  unquantized/
  gemm/                matrix multiplies of dense layers
    affine/
  norm/                RMS norm variants
  hc/                  hyper-connection
  rope/                position encoding
  sample/              sampling
  kvcache/             row reads and writes of the KV and state pools
  embedding/           embedding lookup
  service/             CPU-side runtime services, not kernels
    expert/              ExpertService, expert_lru, expert_mailbox, victim_cache
    ple/                 PleService, ple_row_store
tests/kernel/          the same tree as kernel/
benchmarks/kernel/     the same tree as kernel/
```

A subdirectory with a `.py` file is a Python package; a format directory that holds only `.metal`
files is not. Create a directory only once it has files.

## 2. The files of one kernel

`<name>` is the name of the set of kernels one Python wrapper exposes. Its files all live in one
directory and share that name:

| file | contents |
|---|---|
| `<name>.py` | Python wrapper: shape checks, grid, template parameters, the call |
| `<name>.metal` | the function body when there is one `metal_kernel`, or the whole Metal library of a C++ primitive |
| `<name>_<kernel>.metal` | one function body per file when the wrapper has several `metal_kernel`s |
| `<name>.h` (no `.cpp` of the same name) | a header used only by this wrapper's `metal_kernel`s, passed as `header=` |
| `<name>.h`, `<name>.cpp` | a C++ primitive, only when one is needed (see §6) |
| `<name>_shared.h` | constants and structs shared by a primitive's C++ and Metal (see §6.3) |
| `tests/kernel/<directory>/test_<name>.py` | the test |
| `benchmarks/kernel/<directory>/bench_<name>.py` | the bench |

`_ext.py` hands a C++ primitive's Metal library to `_C.init_<name>_library` to load. Tests compare
against a pure-MLX reference named `<function>_ref`: one shared by several tests lives in
`kernel/ref.py`, one used by a single test lives in that test file. Tests of `service/` follow the
same tree, e.g. `tests/kernel/service/expert/test_expert_lru.py`.

## 3. Which directory

1. Pick the domain directory by kernel type; open a new one only when no domain fits.
2. A domain is flat by default. Only two places have subdirectories per variant:
   - `attention/` splits by attention type: `mha/`, `qsa/`, `gdn/`. A new attention type gets a new
     directory, e.g. `attention/dsa/`.
   - `moe/` and `gemm/` split by quantization format, see §4.
3. Every kernel MoE uses goes in `moe/`, including the expert GEMMs (e.g. `gather_sorted`). `gemm/`
   holds only the matrix multiplies of dense layers.
4. A fused kernel goes with the module it serves. `hc_up_mix` contains an affine GEMV, but it was
   written for hyper-connection, so it lives in `hc/`.
5. A variant a general kernel adds for one type stays in the general kernel. QSA's
   `flash_prefill_qsa` lives in `attention/mha/mha_prefill`, and `qsa/` holds only kernels unique to
   QSA. The `hc_norm` of the hyper-connection injection branch is a variant of the grouped (1 + w)
   RMS norm, so it lives in `norm/offset_rms_norm`, while `hc/` holds `hc_mix` and `hc_up_mix`.
6. A new variant of the same family goes in the same directory, told apart by a file-name prefix.
   For example, DeepSeek's mHC goes in `hc/` with files prefixed `mhc_`.

## 4. Quantization format directories

- Split by how weights are dequantized: `affine/` (MLX affine, 2–8 bit), `fp/` (mxfp4, nvfp4,
  mxfp8), `unquantized/` (weights in the input's dtype), and later `lut/` (LUT quantization). Bit
  width and dtype are template parameters, not directories.
- A format directory holds only that format's code. Wrappers and primitives that dispatch across
  formats sit one level up: `moe/gather_sorted.{py,h,cpp}` picks the Metal library in `moe/affine/`,
  `moe/fp/` or `moe/unquantized/` by format. A kernel that supports one format only goes wholly into
  that format's directory, e.g. `gemm/affine/affine_qmv_rows.py`.
- File names of quantized kernels start with the format (`affine_`, `fp_`, `lut_`); unquantized ones
  carry no prefix (`gather_mm_sorted`), matching MLX's kernel names (`affine_gather_qmm_rhs`,
  `fp_gather_qmm_rhs`, `gather_mm_rhs`). Files in a format directory are named by this rule, not by
  the `<name>` of §2.
- Format code that both `moe/` and `gemm/` use goes in `common/quant/<format>.h`.
- Tests and benches follow the wrapper, not the format: `test_gather_sorted` covers all three
  formats.

## 5. Naming

- File names are unique across the tree and say what they are outside their directory:
  `attention/mha/mha_decode.py`, not `attention/mha/decode.py`; `kvcache/pool_rows.py`, not
  `kvcache/rows.py`. Directories only group. Because:
  - `tests/` has no `__init__.py`, and pytest's default import mode needs test file names to be
    unique across all of `tests/`. A generic name collides with a test elsewhere: the sampling
    kernel is therefore `sample_tokens`, not `sample`, or `test_sample.py` would clash with
    `tests/engine/test_sample.py`;
  - Metal library names and `_C.init_<name>_library` must be unique;
  - a file that changes directory keeps its test, bench and library names.
- A name says what is computed, not a code name: `mha_decode`, not `fdecode`.
- Domain and type names match vLLM and SGLang: `mha`, `gdn`, `moe_topk`, `moe_align`, `indexer`,
  `mrope`, `kvcache`, `embedding`. Hyper-connection uses `hc`, the abbreviation the Qwen3.8
  (`hc_count`) and DeepSeek V4 (`hc_mult`) configs share.
- A kernel adapted from MLX keeps MLX's name (`gather_qmm`, `gather_mm`), so it stays easy to
  compare with upstream when MLX is upgraded.

## 6. Sources

1. All Metal code lives in `.metal` files, never in Python strings.
   - One `metal_kernel` function body per file, and the file holds only the body; the
     `metal_kernel`'s `name` equals the file name without its extension. Variants generated from one
     body by substituting `{{...}}` markers get a suffix after the name, e.g.
     `gdn_prefill_recurrence_mask`. Helper functions come in through `header=`.
   - Wrappers read bodies and Metal libraries with `_ext.py`'s `metal_source()`, and `header=`
     headers with `metal_kernel_header()` (which skips the `utils.h` the `metal_kernel` preamble
     already has).
   - Code fragments that change with the configuration use `{{NAME}}` markers, which the wrapper
     substitutes before compiling; types and integer constants use the `metal_kernel`'s `template`
     parameter.
   - clang-format (`.clang-format`) formats `.h`, `.cpp` and `.mm` files, not `.metal` ones: a
     `metal_kernel` body is not a translation unit, and `{{NAME}}` markers do not survive it.
     Write `.metal` files in the same style by hand.
2. Headers:
   - A C++ primitive's `<name>.h` and `<name>.cpp` come in pairs. A `.h` in a kernel directory
     without a `.cpp` of the same name is a Metal header (`<name>.h`) or a header C++ and Metal
     share (`<name>_shared.h`).
   - Headers shared across domains live in `common/`: Metal ones (`norm.h`), ones both languages
     share (`mailbox.h`), and the C++ headers several primitives share (`metal_primitive.h`).
   - Every `#include` is written relative to `kernel/`, e.g. `#include "common/mailbox.h"`. Metal
     compiled at run time has no include path, so `_ext.py` inlines headers before compiling:
     `mlx/...` comes from the mlx wheel, everything else from `kernel/`.
   - A comment that mentions a file in another directory writes its path relative to `kernel/`
     (`kvcache/pool_rows.h`); a file in the same directory may be named alone.
3. Shared definitions are written once: structs, constants and status codes that both C++ and Metal
   use are defined in one header that both include. One that belongs to a single kernel goes in
   `<name>_shared.h` in its directory (`sample/sample_tokens_shared.h`); the protocol between
   kernels and a service goes in `common/` (`common/mailbox.h`). The header tells the two languages
   apart with `#ifdef __METAL_VERSION__`: Metal program-scope constants are `constant constexpr`,
   and the C++ ones go in `namespace ftmetal`; macros defined for this are `#undef`ed at the end of
   the file.
4. Write a C++ primitive only when a `metal_kernel` cannot do the job:
   - the output must be written in place into an existing buffer;
   - several dispatches share a scratch buffer;
   - the kernel talks to a CPU-side service (mailbox).

   A primitive is registered in `bindings.cpp` and declared in `_C.pyi`, its `.cpp` is added to
   `CMakeLists.txt`, and `_ext.py` loads its Metal library.
5. A ported or adapted kernel's module docstring names the source file and the changes. A `.metal`
   or `.h` file adapted from MLX or mlx-lm (including the function-body files of a `metal_kernel`)
   starts with the source's version, file and line numbers, and its copyright and license notice.
   The MIT license text they refer to is `kernel/LICENSE-MLX`.
6. `service/` is not kernels: the CPU-side runtime services get one subdirectory per service and are
   built into the same `_C`. The protocol definitions between a service and its kernels live in
   `common/`.

## 7. Adding a kernel

1. Pick the directory by §3 and §4, and the name by §5.
2. Write `<name>.metal` and `<name>.py`. When a primitive is needed, also write `<name>.h` and
   `<name>.cpp`, and add one entry each to `bindings.cpp`, `_C.pyi`, `CMakeLists.txt` and `_ext.py`.
3. Write the reference (in `ref.py` when several tests share it) and
   `tests/kernel/<directory>/test_<name>.py`.
4. A kernel with a performance target gets `benchmarks/kernel/<directory>/bench_<name>.py` (how to
   benchmark: [benchmark.md](../../.claude/skills/benchmark/benchmark.md)).

## 8. Rationale

- Directories by type: vLLM `csrc/{attention,moe,quantization}/`, SGLang `kernels/ops/<domain>/`.
  MLX and vllm-metal split by language first because they precompile a metallib, or have several
  backends for one op; we have one backend, Metal, compiled at run time, and a kernel's files always
  change together.
- Quantization format directories: TensorRT-LLM
  `cutlass_kernels/{fp4_gemm,fp8_rowwise_gemm,int8_gemm}/`, vLLM `csrc/quantization/<method>/`.
- One directory per sparse attention: SGLang `layers/attention/{nsa,dsa}/`.
- Shared definitions written once: llama.cpp `ggml-metal-impl.h`, PyTorch MPS `kernels/<Op>.h`.
- One `.metal` file per `metal_kernel` body: vllm-metal `kernels_v2/`, e.g.
  `gdn_recurrent_decode.metal`.
- File names inside a directory still carry a prefix: MLX
  `steel/gemm/kernels/steel_gemm_fused.metal`.
