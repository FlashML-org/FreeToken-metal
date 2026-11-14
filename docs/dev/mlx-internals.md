# MLX internals

ftmetal runs on the unmodified mlx wheel, but its C++ extensions and Metal sources are built
against MLX internals, so mlx is pinned to one version and an upgrade is a change of its own.
Read this before upgrading mlx or changing a C++ primitive or `command_graph/`.

## 1. What depends on MLX

| part | relies on |
|---|---|
| `kernel/_C` (the C++ primitives) | `mx::Primitive`, `metal::Device`, `get_command_encoder` and `CommandEncoder`, the allocator; compiled against the wheel's headers and linked to its `libmlx`, so any mlx change needs a rebuild |
| Metal sources | `_ext.py` inlines MLX kernel headers from the installed wheel before compiling at run time: `utils.h`, `quantized.h`, `quantized_utils.h`, `fp_quantized.h`, `steel/gemm/gemm.h`, `steel_gemm_gather.h`. Kernels adapted from MLX or mlx-lm name the version, file and lines they come from ([kernel-layout.md](kernel-layout.md#ported-code)) |
| `command_graph/_C` | private members read through member pointers (`CommandEncoder::temporaries_` and `encoder_`, `Device::library_kernels_` and `kernel_mtx_`); what `end_encoding` does with the fence and the temporaries; the layout of the setBytes payloads it copies (steel `GEMMParams` and its tail padding); and the Objective-C runtime, to replace methods of Metal's concrete encoder and command-buffer classes |
| Metal language version | MLX compiles run-time sources at Metal 3.1 on macOS 14, 3.2 on 15 and 4.0 on 26. The mailbox kernels need 3.2 (`thread_scope`), hence the macOS 15 floor. The `mpp` prefill path (MetalPerformancePrimitives `matmul2d`) needs 4.0; it is picked only on macOS 26.2 and later GPUs with matrix units, or with `FTMETAL_FPREFILL_MMA=mpp` |

## 2. The pin

The mlx version is written in three places that move together: `dependencies` and
`[build-system] requires` in `pyproject.toml`, and `REQUIRED_MLX_VERSION` in `kernel/_ext.py`, which
refuses to load against another version. `nanobind` in `[build-system] requires` must produce the
same nanobind ABI as mlx's own bindings, or arrays cannot cross between the extensions. Check it
after a build: `strings` on mlx's `core` module and on both `_C` extensions shows the same
`nanobind_..._v<N>` tag.

## 3. Upgrading mlx

1. Read the release notes and the diff of `mlx/backend/metal/device.{h,cpp}`, of the inlined
   kernel headers, of `steel/gemm/params.h` (`command_graph.mm` hardcodes the 72 and 68 bytes of
   `GEMMParams`), and of the files the adapted kernels cite.
2. Move the three pins (and `nanobind` with mlx's), then `uv sync --reinstall-package freetoken-metal`.
3. A compile error in `command_graph.mm` means a private member changed: re-read how it is used
   before renaming anything.
4. Update the version and line numbers in the adapted kernels' headers and the version named in
   `command_graph.mm`'s comments, and port the upstream fixes that apply.
5. Run `uv run pytest tests/ -q` (the model parity tests compare against mlx-lm;
   `tests/command_graph/` and `tests/engine/test_graph_runner.py` cover capture and replay). On a
   real checkpoint, run `tests/models/test_real_checkpoint.py`, `tools/command_graph_m1.py` (eager
   against replay bit for bit, and the double-capture check) and `tools/aime0_check.py`, then the
   decode benchmarks with `FTMETAL_COMMAND_GRAPH=1` and `0`.

## 4. CommandGraph

`command_graph/` captures the Metal dispatches of one decode step per batch-size bucket and replays
them as one `GraphReplay` primitive. How it works, and what code that runs inside a decode step must
do to stay replayable, is in [command-graph.md](command-graph.md).
