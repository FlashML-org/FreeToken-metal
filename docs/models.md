# Supported models

FreeToken-metal loads local checkpoint directories. The model family comes from `architectures`
in the checkpoint's `config.json`:

| Model | `architectures` | Image input |
|---|---|---|
| Qwen3.6 / Qwen3.5 MoE (e.g. Qwen3.6-35B-A3B) | `Qwen3_5MoeForConditionalGeneration`, `Qwen3_5MoeForCausalLM` | `ForConditionalGeneration` only |
| Gemma 4 (26B-A4B MoE, 31B dense) | `Gemma4ForConditionalGeneration`, `Gemma4ForCausalLM` | `ForConditionalGeneration` only |
| Qwen3.8-Flash-Next | `Qwen4ExpForConditionalGeneration`, `Qwen4ExpForCausalLM` | `ForConditionalGeneration` only |
| Flint | `FlintForCausalLM` | no |

## Checkpoint formats

MLX checkpoints (`mlx_lm.convert` and the like): bf16, affine quantization including mixed
per-layer bits, and the float block modes `mxfp4`, `nvfp4` and `mxfp8`.

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
