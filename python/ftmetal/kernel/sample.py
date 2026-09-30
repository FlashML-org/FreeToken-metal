"""Token sampling kernels: softmax(logits / T), top-k and top-p filtering, inverse-CDF draw.

Port of FreeToken kernel/triton/sampling.py (top_k_top_p_sampling_from_probs and its
single-filter forms). Semantics are FreeToken's: top-k keeps every token whose probability
is at least the k-th largest, boundary ties included; top-p then keeps every token at least
the largest probability whose descending mass reaches p times the mass top-k kept; the
token is the first, in vocab order, whose cumulative kept mass exceeds u * total.
Divergences: the radix select runs on the logits' bit pattern (16-bit keys for bf16 / fp16,
32-bit for fp32), not on fp32 probabilities, so the probabilities are never written and a
16-bit row needs two rounds per filter instead of four; the kept set is the same except that
logits whose fp32 probabilities round to a tie stay distinct. The row's G threadgroups never
wait on one another (a Metal grid is not co-resident): each round is one dispatch whose
threadgroups each re-select the previous round's bin from the per-chunk histograms it left,
so all sums run in a fixed order and a given u always draws the same token. The uniform
comes from random bits drawn with MLX's key sequence, so mx.random.seed governs it.
"""

from __future__ import annotations

import mlx.core as mx

from ftmetal.kernel._ext import _C

# Threadgroups across the batch, four per GPU core on the M4 Pro (20 cores), and no chunk
# under MIN_CHUNK logits, below which a threadgroup's fixed cost (re-selecting the previous
# round's bin) outweighs its share of the row. Measured on the M4 Pro (tools/bench_sample.py):
# G = 15-25 at bs 1 and 10 at bs 8 are fastest for 152k and 248k vocabularies.
TARGET_THREADGROUPS = 80
MIN_CHUNK = 10240
# The draw scans the per-chunk masses in one threadgroup (metal/sample.metal MAX_GROUPS).
MAX_GROUPS = 256


def plan_groups(batch_size: int, vocab_size: int) -> int:
    """Threadgroups per row: enough to fill the GPU at small batch, no chunk below MIN_CHUNK."""
    by_batch = -(-TARGET_THREADGROUPS // batch_size)
    by_chunk = -(-vocab_size // MIN_CHUNK)
    return max(1, min(by_batch, by_chunk, MAX_GROUPS))


def sample_tokens(
    logits: mx.array,
    temperatures: mx.array,
    top_k: mx.array | None,
    top_p: mx.array | None,
    random_bits: mx.array | None = None,
) -> mx.array:
    """One token per row of logits [B, V] (bf16, fp16 or fp32).

    temperatures [B] fp32 > 0; top_k [B] int32, off at 0 or >= V; top_p [B] fp32, off at
    >= 1; None turns a filter off for the whole batch and skips its rounds. random_bits
    [B] uint32 give the draw's uniform u = (bits >> 8) / 2**24; by default they come from
    MLX's global key sequence, as mx.random draws do. Returns int32 [B].
    """
    batch_size, vocab_size = logits.shape
    num_groups = plan_groups(batch_size, vocab_size)
    return _C.sample_tokens(logits, temperatures, top_k, top_p, num_groups, random_bits)


__all__ = ["plan_groups", "sample_tokens"]
