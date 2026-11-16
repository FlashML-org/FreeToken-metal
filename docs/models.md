# Supported models

`ftm serve --model DIR` serves a checkpoint directory on local disk. Download one first:

```bash
uvx hf download mlx-community/Qwen3.6-35B-A3B-4bit --local-dir ~/models/Qwen3.6-35B-A3B-4bit
ftm serve --model ~/models/Qwen3.6-35B-A3B-4bit
```

## Checkpoints

| Model | Checkpoints (download size) | Image input |
| --- | --- | --- |
| Qwen3.6-35B-A3B | mlx-community: [INT4](https://huggingface.co/mlx-community/Qwen3.6-35B-A3B-4bit) (20 GB), [INT6](https://huggingface.co/mlx-community/Qwen3.6-35B-A3B-6bit) (29 GB), [INT8](https://huggingface.co/mlx-community/Qwen3.6-35B-A3B-8bit) (38 GB), [BF16](https://huggingface.co/mlx-community/Qwen3.6-35B-A3B-bf16) (70 GB) | yes |
| Qwen3.5-122B-A10B | mlx-community: [INT4](https://huggingface.co/mlx-community/Qwen3.5-122B-A10B-4bit) (70 GB), [INT6](https://huggingface.co/mlx-community/Qwen3.5-122B-A10B-6bit) (100 GB), [INT8](https://huggingface.co/mlx-community/Qwen3.5-122B-A10B-8bit) (131 GB), [BF16](https://huggingface.co/mlx-community/Qwen3.5-122B-A10B-bf16) (245 GB) | yes |
| Qwen3.5-397B-A17B | mlx-community: [INT4](https://huggingface.co/mlx-community/Qwen3.5-397B-A17B-4bit) (224 GB), [INT6](https://huggingface.co/mlx-community/Qwen3.5-397B-A17B-6bit) (323 GB), [INT8](https://huggingface.co/mlx-community/Qwen3.5-397B-A17B-8bit) (422 GB) | yes |
| Qwen3.8-Flash-Next | mlx-community: [INT4](https://huggingface.co/mlx-community/Qwen3.8-Flash-Next-4bit) (112 GB) | yes |
| Gemma 4 26B-A4B | mlx-community: [INT4](https://huggingface.co/mlx-community/gemma-4-26b-a4b-it-4bit) (15 GB), [INT6](https://huggingface.co/mlx-community/gemma-4-26b-a4b-it-6bit) (22 GB), [INT8](https://huggingface.co/mlx-community/gemma-4-26b-a4b-it-8bit) (28 GB); Google: [BF16](https://huggingface.co/google/gemma-4-26B-A4B-it) (52 GB) | yes |
| Flint | RecursiveSI: [INT4](https://huggingface.co/RecursiveSI/flint-int4) (23 GB); mlx-community: [INT6](https://huggingface.co/mlx-community/flint-oQ6e) (33 GB), [INT8](https://huggingface.co/mlx-community/flint-oQ8e) (42 GB) | no |
| Ornith-1.5-35B-A3B | ornith-ai: [INT4](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit) (20 GB), [INT6](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-MLX-6bit) (28 GB), [INT8](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-MLX-8bit) (37 GB), [BF16](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-MLX) (69 GB) | no |
| Ornith-1.5-122B | ornith-ai: [INT4](https://huggingface.co/ornith-ai/Ornith-1.5-122B-MLX-4bit) (70 GB), [INT8](https://huggingface.co/ornith-ai/Ornith-1.5-122B-MLX-8bit) (131 GB), [BF16](https://huggingface.co/ornith-ai/Ornith-1.5-122B-MLX) (245 GB) | yes |

A model whose experts do not fit in memory serves them from the SSD ([MoE offload](#moe-offload)).

## Other checkpoints

Other checkpoints of these families work too. The family comes from `architectures` in the
checkpoint's `config.json`:

| Family | `architectures` | Formats |
| --- | --- | --- |
| Qwen3.6 / Qwen3.5 MoE (Ornith-1.5 too) | `Qwen3_5MoeForConditionalGeneration`, `Qwen3_5MoeForCausalLM` | MLX, Hugging Face, exports |
| Gemma 4 (26B-A4B MoE, 31B dense) | `Gemma4ForConditionalGeneration`, `Gemma4ForCausalLM` | MLX, Hugging Face, exports |
| Qwen3.8-Flash-Next | `Qwen4ExpForConditionalGeneration`, `Qwen4ExpForCausalLM` | MLX |
| Flint | `FlintForCausalLM` | MLX |

- MLX: `mlx_lm.convert` and the like, in bf16, affine quantization including mixed per-layer
  bits, or the float block modes `mxfp4`, `nvfp4` and `mxfp8`.
- Hugging Face: the original bf16 release.
- Exports: ModelOpt and compressed-tensors NVFP4 and per-tensor FP8, such as
  [nvidia/Qwen3.6-35B-A3B-NVFP4](https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4).

Image input needs a `ForConditionalGeneration` checkpoint. GGUF is not served yet.

## Image input

Images come in OpenAI `image_url`, Anthropic `image` and Responses `input_image` content blocks.
`--text-model-only` serves a vision checkpoint as text only and frees the vision tower's memory.

## MoE offload

When a MoE model's experts do not fit in memory, the server keeps them in the checkpoint on disk
and caches the experts in use. `--moe-strategy` chooses: `auto` (the default) offloads only when
the experts do not fit, `fused` keeps them all in memory, `offload` always offloads.

By default the KV cache gets its context first (`--num-token`, else 64K tokens) and the expert
cache gets the rest of the memory; `--moe-cache-size` or `--moe-cache-rate` sets the expert cache
directly. Keep the checkpoint on the internal SSD: every cache miss is a read from it.

### Expert pack

```bash
ftm pack --model DIR
```

writes `DIR/experts.packed`, each expert as one record, which the offload cache then reads instead
of the checkpoint: one read per expert, and for affine checkpoints smaller experts, so more of them
fit in the cache. Run it once per checkpoint.
