"""Token sampling on device: greedy, or temperature with top-k / top-p.

Port of FreeToken engine/sample.py. Divergences: the sampling kernels are MLX ops
(sorted-probability masks and mx.random.categorical) instead of flashinfer.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx

from ftmetal.core import Batch

MIN_TEMPERATURE = 1e-6
MIN_TOP_P = 1e-6


@dataclass
class BatchSamplingArgs:
    temperatures: mx.array | None
    top_k: mx.array | None = None
    top_p: mx.array | None = None


def apply_top_k(logits: mx.array, top_k: mx.array) -> mx.array:
    """Mask every logit below the row's top_k-th largest; top_k is int32 [rows]."""
    ranks = mx.argsort(mx.argsort(-logits, axis=-1), axis=-1)
    return mx.where(ranks < top_k[:, None], logits, -mx.inf)


def apply_top_p(logits: mx.array, top_p: mx.array) -> mx.array:
    """Keep the smallest prefix of descending probabilities whose mass reaches top_p."""
    order = mx.argsort(-logits, axis=-1)
    sorted_probs = mx.softmax(mx.take_along_axis(logits, order, axis=-1), axis=-1)
    cumulative = mx.cumsum(sorted_probs, axis=-1)
    keep_sorted = (cumulative - sorted_probs) < top_p[:, None]
    keep = mx.put_along_axis(mx.zeros_like(keep_sorted), order, keep_sorted, axis=-1)
    return mx.where(keep, logits, -mx.inf)


@dataclass
class Sampler:
    vocab_size: int

    def prepare(self, batch: Batch) -> BatchSamplingArgs:
        params = [req.sampling_params for req in batch.reqs]
        if all(p.is_greedy for p in params):
            return BatchSamplingArgs(temperatures=None)
        temperatures = [max(0.0 if p.is_greedy else p.temperature, MIN_TEMPERATURE) for p in params]
        top_ks = [p.top_k if p.top_k >= 1 else self.vocab_size for p in params]
        top_ps = [min(max(p.top_p, MIN_TOP_P), 1.0) for p in params]
        args = BatchSamplingArgs(temperatures=mx.array(temperatures, dtype=mx.float32))
        if any(k != self.vocab_size for k in top_ks):
            args.top_k = mx.array(top_ks, dtype=mx.int32)
        if any(p < 1.0 for p in top_ps):
            args.top_p = mx.array(top_ps, dtype=mx.float32)
        return args

    def sample(self, logits: mx.array, args: BatchSamplingArgs) -> mx.array:
        """logits [rows, vocab] -> int32 [rows]."""
        if args.temperatures is None:
            return mx.argmax(logits, axis=-1).astype(mx.int32)
        scaled = logits.astype(mx.float32) / args.temperatures[:, None]
        if args.top_k is not None:
            scaled = apply_top_k(scaled, args.top_k)
        if args.top_p is not None:
            scaled = apply_top_p(scaled, args.top_p)
        return mx.random.categorical(scaled, axis=-1).astype(mx.int32)


__all__ = ["BatchSamplingArgs", "Sampler", "apply_top_k", "apply_top_p"]
