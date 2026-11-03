# kernel/ 目录规范

适用于 `python/ftmetal/kernel/` 下的自定义 kernel，以及它们的测试和 bench。新增、移动、改名
kernel 都按本文做；要改规则，先改本文。

## 1. 目录树

```
python/ftmetal/kernel/
  _ext.py  custom.py  ref.py  bindings.cpp  CMakeLists.txt
  common/              跨领域共用的头文件
  attention/
    mha/               MHA、GQA、MQA
    qsa/               QSA 稀疏注意力
    gdn/               Gated DeltaNet
  moe/                 MoE：路由、排序、槽位查找、专家 GEMM
    affine/  fp/  unquantized/
  gemm/                稠密层的矩阵乘
    affine/
  norm/                RMS norm 的各种变体
  hc/                  hyper-connection
  rope/                位置编码
  sample/              采样
  kvcache/             KV 池和状态池的行读写
  embedding/           embedding 查表
  service/             CPU 端运行时服务，不是 kernel
    expert/              ExpertService、expert_lru、expert_mailbox、victim_cache
    ple/                 PleService、ple_row_store
tests/kernel/          与 kernel/ 相同的目录结构
tools/bench_<名>.py
```

有 `.py` 的子目录是 Python 包；只放 `.metal` 的格式目录不是。目录里有文件才建。

## 2. 一个 kernel 的文件

`<名>` 是一个 Python 包装对外提供的那组 kernel 的名字。它的文件都放在同一个目录，用同一个名字：

| 文件 | 内容 |
|---|---|
| `<名>.py` | Python 包装：形状检查、grid、模板参数、调用 |
| `<名>.metal` | 只有一个 `metal_kernel` 时的函数体；或 C++ primitive 的整个 Metal 库 |
| `<名>_<kernel>.metal` | 包装有多个 `metal_kernel` 时，每个函数体一个文件 |
| `<名>.h`（没有同名 `.cpp`） | 只给这个包装的 `metal_kernel` 用的头文件，作为 `header=` 传入 |
| `<名>.h`、`<名>.cpp` | C++ primitive，只在需要时才有（见 §6） |
| `<名>_shared.h` | primitive 的 C++ 和 Metal 共用的常量、结构体（见 §6.3） |
| `tests/kernel/<目录>/test_<名>.py` | 测试 |
| `tools/bench_<名>.py` | bench |

C++ primitive 的 Metal 库由 `_ext.py` 交给 `_C.init_<名>_library` 加载。测试对照纯 MLX 的参考实现，
命名为 `<函数>_ref`：多个测试共用的放在 `kernel/ref.py`，只有一个测试用的写在那个测试文件里。
`service/` 的测试同样按目录放，比如 `tests/kernel/service/expert/test_expert_lru.py`。

## 3. 放在哪个目录

1. 按 kernel 类型选领域目录；没有合适的领域时再开新目录。
2. 领域内默认平铺。按变体开子目录的只有两处：
   - `attention/` 按注意力类型分：`mha/`、`qsa/`、`gdn/`。新的注意力类型开新目录，比如 `attention/dsa/`。
   - `moe/` 和 `gemm/` 按量化格式分，见 §4。
3. MoE 用的 kernel 都放 `moe/`，包括专家 GEMM（比如 `gather_sorted`）。`gemm/` 只放稠密层的矩阵乘。
4. 融合 kernel 跟着它服务的模块走。`hc_up_mix` 内含 affine GEMV，但它是为 hyper-connection 写的，
   放在 `hc/`。
5. 通用 kernel 为某种类型加的变体，留在通用 kernel 里。QSA 用的 `flash_prefill_qsa` 在
   `attention/mha/mha_prefill`，`qsa/` 只放 QSA 独有的 kernel；hyper-connection 注入分支的
   `hc_norm` 是分组 (1 + w) RMS norm 的一个变体，在 `norm/offset_rms_norm`，`hc/` 放 `hc_mix`
   和 `hc_up_mix`。
6. 同一族的新变体用文件名前缀区分，放进同一个目录。比如 DeepSeek 的 mHC 放 `hc/`，文件用 `mhc_` 前缀。

## 4. 量化格式目录

- 按反量化方式分：`affine/`（MLX affine，2–8 bit）、`fp/`（mxfp4、nvfp4、mxfp8）、`unquantized/`
  （权重与输入同 dtype），以后加 `lut/`（LUT 量化）。位宽和 dtype 是模板参数，不分目录。
- 格式目录只放该格式的代码。跨格式分派的包装和 primitive 放在上一层：`moe/gather_sorted.{py,h,cpp}`
  按格式选用 `moe/affine/`、`moe/fp/`、`moe/unquantized/` 里的 Metal 库。只支持一种格式的 kernel
  整个放进格式目录，比如 `gemm/affine/affine_qmv_rows.py`。
- 量化 kernel 的文件名以格式开头（`affine_`、`fp_`、`lut_`），不量化的不加（`gather_mm_sorted`），
  与 MLX 的 kernel 名一致（`affine_gather_qmm_rhs`、`fp_gather_qmm_rhs`、`gather_mm_rhs`）。格式目录里
  的文件按这条命名，不受 §2 的 `<名>` 约束。
- `moe/` 和 `gemm/` 都要用的格式代码放 `common/quant/<格式>.h`。
- 测试和 bench 跟着包装走，不跟着格式走：`test_gather_sorted` 覆盖三种格式。

## 5. 命名

- 文件名全局唯一，离开目录也看得出是什么：`attention/mha/mha_decode.py`，不写成
  `attention/mha/decode.py`；`kvcache/pool_rows.py`，不写成 `kvcache/rows.py`。目录只负责分组。原因：
  - `tests/` 没有 `__init__.py`，pytest 默认的导入模式要求整个 `tests/` 里的测试文件名唯一。
    名字太泛就会和别处的测试撞名：采样 kernel 因此叫 `sample_tokens`，不叫 `sample`，否则
    `test_sample.py` 会和 `tests/engine/test_sample.py` 冲突；
  - Metal 库名和 `_C.init_<名>_library` 必须唯一；
  - 文件换目录时，测试、bench、库名都不用改。
- 名字说明算的是什么，不用代号：`mha_decode`，不是 `fdecode`。
- 领域和类型名与 vLLM、SGLang 一致：`mha`、`gdn`、`moe_topk`、`moe_align`、`indexer`、`mrope`、
  `kvcache`、`embedding`。hyper-connection 用 Qwen3.8（`hc_count`）和 DeepSeek V4（`hc_mult`）
  config 共同的缩写 `hc`。
- 从 MLX 改来的 kernel 保留 MLX 的名字（`gather_qmm`、`gather_mm`），升级 MLX 时方便对照上游。

## 6. 源码

1. Metal 代码都写在 `.metal` 文件里，不写在 Python 字符串里。
   - 一个 `metal_kernel` 函数体一个文件，文件里只有函数体；`metal_kernel` 的 `name` 与文件名
     （去掉扩展名）相同。同一个函数体替换 `{{...}}` 记号生成的变体，在名字后加后缀区分，比如
     `gdn_prefill_recurrence_mask`。辅助函数通过 `header=` 传入。
   - 包装用 `_ext.py` 的 `metal_source()` 读函数体和 Metal 库，用 `metal_kernel_header()` 读
     `header=` 的头文件（它跳过 `metal_kernel` 前导部分已有的 `utils.h`）。
   - 随配置变化的代码片段用 `{{NAME}}` 记号，由包装在编译前替换；类型和整数常量用
     `metal_kernel` 的 `template` 参数。
2. 头文件：
   - C++ primitive 的 `<名>.h` 与 `<名>.cpp` 成对出现。kernel 目录里没有同名 `.cpp` 的 `.h`，
     是 Metal 头文件（`<名>.h`）或 C++ 和 Metal 共用的头文件（`<名>_shared.h`）。
   - 跨领域共用的头文件放 `common/`：Metal 的（`norm.h`）、两种语言共用的（`mailbox.h`），以及
     各个 primitive 共用的 C++ 头文件（`metal_primitive.h`）。
   - `#include` 一律写相对 `kernel/` 的路径，比如 `#include "common/mailbox.h"`。运行时编译 Metal
     没有 include 路径，`_ext.py` 在编译前内联头文件：`mlx/...` 取自 mlx wheel，其余取自 `kernel/`。
   - 注释里提到别的目录的文件时写相对 `kernel/` 的路径（`kvcache/pool_rows.h`），同目录的可以只写文件名。
3. 共用定义只写一份：C++ 和 Metal 都用到的结构体、常量和状态码只在一个头文件里定义，两边都
   include。只属于一个 kernel 的放它目录里的 `<名>_shared.h`（`sample/sample_tokens_shared.h`），
   kernel 和服务之间的协议放 `common/`（`common/mailbox.h`）。头文件用 `#ifdef __METAL_VERSION__`
   区分两种语言：Metal 的程序级常量要写 `constant constexpr`，C++ 的放进 `namespace ftmetal`；
   为此定义的宏在文件末尾 `#undef`。
4. 只有 `metal_kernel` 做不到时才写 C++ primitive：
   - 输出要原地写进已有的 buffer；
   - 多次 dispatch 共享临时 buffer；
   - 要和 CPU 端服务通信（mailbox）。

   primitive 在 `bindings.cpp` 注册并写进 `_C.pyi`，`.cpp` 加进 `CMakeLists.txt`，Metal 库由 `_ext.py`
   加载。
5. 移植或改编来的 kernel，模块 docstring 写明来源文件和改动。从 MLX 或 mlx-lm 改来的 `.metal`、`.h`
   文件（包括 `metal_kernel` 的函数体文件）开头写明来源的版本、文件和行号，以及版权和许可声明。
6. `service/` 不是 kernel：CPU 端运行时服务按服务分子目录，编进同一个 `_C`。服务和 kernel 之间的
   协议定义放在 `common/`。

## 7. 新增 kernel 的步骤

1. 按 §3、§4 选目录，按 §5 起名。
2. 写 `<名>.metal` 和 `<名>.py`。需要 primitive 时再写 `<名>.h`、`<名>.cpp`，并在 `bindings.cpp`、
   `_C.pyi`、`CMakeLists.txt`、`_ext.py` 各加一处。
3. 写参考实现（多个测试共用的放 `ref.py`）和 `tests/kernel/<目录>/test_<名>.py`。
4. 有性能要求的 kernel 写 `tools/bench_<名>.py`。

## 8. 依据

- 按类型分目录：vLLM `csrc/{attention,moe,quantization}/`，SGLang `kernels/ops/<领域>/`。MLX 和
  vllm-metal 先按语言分，是因为它们预编译 metallib，或同一个 op 有多个后端；我们只有 Metal 一个
  后端、运行时编译，一个 kernel 的几个文件总是一起改。
- 量化格式目录：TensorRT-LLM `cutlass_kernels/{fp4_gemm,fp8_rowwise_gemm,int8_gemm}/`，vLLM
  `csrc/quantization/<方法>/`。
- 稀疏注意力每种一个目录：SGLang `layers/attention/{nsa,dsa}/`。
- 共用定义只写一份：llama.cpp `ggml-metal-impl.h`，PyTorch MPS `kernels/<Op>.h`。
- 每个 `metal_kernel` 函数体一个 `.metal` 文件：vllm-metal `kernels_v2/`，比如
  `gdn_recurrent_decode.metal`。
- 目录里的文件名仍带前缀：MLX `steel/gemm/kernels/steel_gemm_fused.metal`。
