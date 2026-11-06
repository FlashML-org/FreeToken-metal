"""Qwen3.8-Flash-Next's zero-centred RMSNorm and its hyper-connections over the residual
streams.

Port of FreeToken models/qwen4_exp/hc.py (GroupedPlusOneRMSNorm, GatedResidual).
Divergences: the hyper-connection returns its branch input and the branch's raw inject
logits, and injects the branch before it (the attention's or the MLP's, pending) as it
norms the streams (kernel/norm/offset_rms_norm.py), so the decoder layer hands a pending
(branch, logits) on; the down and inject projections are fused once loaded (FreeToken fuses
them at load too), and a decode token runs its up projection fused with the mix.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from ftmetal.kernel.gemm.affine.affine_qmv_rows import qmv_rows
from ftmetal.kernel.hc.hc_mix import hc_mix
from ftmetal.kernel.hc.hc_up_mix import hc_up_mix, supports_hc_up_mix
from ftmetal.kernel.norm.offset_rms_norm import hc_norm, offset_rms_norm
from ftmetal.layers.base import BaseOP
from ftmetal.layers.linear import Linear, LinearMerged
from ftmetal.layers.quantization.config import QuantScheme
from ftmetal.layers.quantization.linear import AffineLinearMethod

# The branch an attention or MLP sublayer adds back to the streams: its output [T, H] and its
# raw inject logits [T, hc_count].
PendingBranch = tuple[mx.array, mx.array]


class CenteredRMSNorm(BaseOP):
    """rmsnorm(x) * (1 + w) in fp32, w stored centred at zero; with group_size, each
    group_size slice normed apart."""

    def __init__(
        self, size: int, eps: float, *, group_size: int | None = None, dtype: mx.Dtype
    ) -> None:
        super().__init__()
        assert group_size is None or size % group_size == 0
        self.eps = eps
        self.group_size = group_size
        self.weight = mx.zeros((size,), dtype=dtype)

    def forward(self, x: mx.array) -> mx.array:
        return offset_rms_norm(x, self.weight, self.eps, self.group_size)


def inject(hyper: mx.array, branch: mx.array, logits: mx.array) -> mx.array:
    """hyper [T, hc * H] plus branch [T, H] into each stream s, weighted 2 * sigmoid(logits[:, s]
    / hc), in fp32 (hc_norm's inject, without the norm)."""
    num_tokens, hc_count = logits.shape
    gates = 2.0 * mx.sigmoid(logits.astype(mx.float32) / hc_count)
    streams = hyper.astype(mx.float32).reshape(num_tokens, hc_count, -1)
    added = streams + branch.astype(mx.float32)[:, None, :] * gates[:, :, None]
    return added.reshape(hyper.shape).astype(hyper.dtype)


class HyperConnection(BaseOP):
    """Mixes the hc_count residual streams [T, hc_count * H] into one sublayer input [T, H];
    with inject, also the raw logits [T, hc_count] that weigh the sublayer's output back in.
    Once loaded, an affine down and inject of one scheme are one projection (fuse_down_inject),
    so a decode token runs both, with the down's silu, in one GEMV."""

    def __init__(
        self,
        hc_count: int,
        hidden_size: int,
        lowrank: int,
        eps: float,
        *,
        down_scheme: QuantScheme | None,
        up_scheme: QuantScheme | None,
        inject_scheme: QuantScheme | None,
        with_inject: bool,
        dtype: mx.Dtype,
    ) -> None:
        super().__init__()
        self.hc_count = hc_count
        self.lowrank = lowrank
        width = hc_count * hidden_size
        self.hc_norm = CenteredRMSNorm(width, eps, group_size=hidden_size, dtype=dtype)
        self.input_mix_weight_down = Linear(width, lowrank, scheme=down_scheme, dtype=dtype)
        self.input_mix_weight_up = Linear(lowrank, width, scheme=up_scheme, dtype=dtype)
        if with_inject:
            self.block_inject_weight = Linear(width, hc_count, scheme=inject_scheme, dtype=dtype)

    def fuse_down_inject(self) -> None:
        """Replace the down and inject projections, when both are affine in one scheme, by one
        of their rows, then zero rows up to a whole qmv_rows tile."""
        fused = LinearMerged.fuse_rows([self.input_mix_weight_down, self.block_inject_weight])
        if fused is None:
            return
        self.down_inject = fused
        self.pop("input_mix_weight_down")
        self.pop("block_inject_weight")

    def down_and_logits(self, normed: mx.array) -> tuple[mx.array, mx.array | None]:
        """(silu(down(normed) / hc) in normed's dtype [T, lowrank], the raw inject logits)."""
        lowrank, hc_count = self.lowrank, self.hc_count
        if "down_inject" in self:
            fused = self.down_inject
            method = fused.quant_method
            if isinstance(method, AffineLinearMethod) and method.uses_qmv_rows(fused, normed):
                # The decode token's down rows activated as qmv_rows stores them.
                scheme = method.scheme
                projected = qmv_rows(
                    normed,
                    fused["weight"],
                    fused["scales"],
                    fused["biases"],
                    group_size=scheme.group_size,
                    bits=scheme.bits,
                    hc_epilogue=(lowrank, hc_count),
                )
                return projected[:, :lowrank], projected[:, lowrank : lowrank + hc_count]
            else:
                down, logits = fused.forward_split(normed)
        else:
            down = self.input_mix_weight_down.forward(normed)
            if "block_inject_weight" in self:
                logits = self.block_inject_weight.forward(normed)
            else:
                logits = None
        activated = nn.silu(down.astype(mx.float32) / hc_count).astype(normed.dtype)
        return activated, logits

    def mix(self, normed: mx.array, activated: mx.array) -> mx.array:
        """The mean over the streams of sigmoid(up(activated)) * normed, in fp32."""
        up = self.input_mix_weight_up
        method = up.quant_method
        if isinstance(method, AffineLinearMethod):
            scheme = method.scheme
            if supports_hc_up_mix(
                activated,
                up["weight"],
                up["scales"],
                normed,
                self.hc_count,
                scheme.group_size,
                scheme.bits,
            ):
                return hc_up_mix(
                    activated,
                    up["weight"],
                    up["scales"],
                    up["biases"],
                    normed,
                    self.hc_count,
                    group_size=scheme.group_size,
                    bits=scheme.bits,
                )
        return hc_mix(up.forward(activated), normed, self.hc_count)

    def norm(self, hyper: mx.array, pending: PendingBranch | None) -> tuple[mx.array, mx.array]:
        """(normed streams, streams): hyper with the pending branch injected."""
        branch, logits = pending if pending is not None else (None, None)
        norm = self.hc_norm
        return hc_norm(hyper, norm.weight, norm.eps, self.hc_count, branch, logits)

    def forward(
        self, hyper: mx.array, pending: PendingBranch | None
    ) -> tuple[mx.array, mx.array, mx.array]:
        """(the streams with the pending branch injected, the sublayer input [T, H], its raw
        inject logits [T, hc_count])."""
        normed, hyper = self.norm(hyper, pending)
        activated, logits = self.down_and_logits(normed)
        assert logits is not None
        return hyper, self.mix(normed, activated), logits

    def forward_final(self, hyper: mx.array, pending: PendingBranch | None) -> mx.array:
        """The final mixer's output, the LM head's input."""
        normed, _ = self.norm(hyper, pending)
        activated, _ = self.down_and_logits(normed)
        return self.mix(normed, activated)


__all__ = ["CenteredRMSNorm", "HyperConnection", "PendingBranch", "inject"]
