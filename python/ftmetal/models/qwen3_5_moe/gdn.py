"""Gated DeltaNet layer over the per-request state pool.

Port of FreeToken models/qwen3_5_moe/gdn.py. Divergences: the conv is mx.conv1d over
the slot's conv state plus this step's inputs, the recurrence is the Metal gated
delta kernel, and both new states are scattered into the pool in place. Decode runs
the batch in one launch; prefill runs one launch per request (varlen).
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from ftmetal.context import get_global_ctx
from ftmetal.core import Batch
from ftmetal.kernel.gdn import compute_gate_decay, gated_delta_recurrence
from ftmetal.kernel.inplace import scatter_rows
from ftmetal.layers.base import BaseOP
from ftmetal.layers.linear import Linear, LinearMerged
from ftmetal.layers.norm import GatedRMSNorm
from ftmetal.layers.quantization.config import AffineScheme
from ftmetal.models.config import LinearGatedDeltaGroupConfig

# The kernel's fixed l2-norm epsilon for q and k.
QK_NORM_EPS = 1e-6


class Qwen3_5GatedDeltaNet(BaseOP):
    def __init__(
        self,
        hidden_size: int,
        group: LinearGatedDeltaGroupConfig,
        rms_norm_eps: float,
        layer_id: int,
        *,
        qkvz_scheme: AffineScheme | None,
        ba_scheme: AffineScheme | None,
        out_scheme: AffineScheme | None,
        dtype: mx.Dtype,
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.num_k_heads = group.num_key_heads
        self.num_v_heads = group.num_value_heads
        self.head_k_dim = group.key_head_dim
        self.head_v_dim = group.value_head_dim
        self.key_dim = self.num_k_heads * self.head_k_dim
        self.value_dim = self.num_v_heads * self.head_v_dim
        self.conv_dim = 2 * self.key_dim + self.value_dim
        self.conv_kernel_size = group.conv_kernel_dim
        self.in_proj_qkvz = LinearMerged(
            hidden_size, [self.conv_dim, self.value_dim], scheme=qkvz_scheme, dtype=dtype
        )
        self.in_proj_ba = LinearMerged(
            hidden_size, [self.num_v_heads, self.num_v_heads], scheme=ba_scheme, dtype=dtype
        )
        self.conv1d = DepthwiseConv1d(self.conv_dim, self.conv_kernel_size, dtype)
        self.dt_bias = mx.ones((self.num_v_heads,), dtype=dtype)
        self.A_log = mx.zeros((self.num_v_heads,), dtype=dtype)
        self.norm = GatedRMSNorm(self.head_v_dim, eps=rms_norm_eps, dtype=dtype)
        self.out_proj = Linear(self.value_dim, hidden_size, scheme=out_scheme, dtype=dtype)

    def recurrence_inputs(
        self, conv_out: mx.array, a: mx.array, b: mx.array
    ) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array]:
        """conv_out [B, T, conv_dim]; a, b [B, T, num_v_heads] -> q, k, v, g, beta."""
        batch_size, seq_len, _ = conv_out.shape
        q, k, v = mx.split(conv_out, [self.key_dim, 2 * self.key_dim], axis=-1)
        q = q.reshape(batch_size, seq_len, self.num_k_heads, self.head_k_dim)
        k = k.reshape(batch_size, seq_len, self.num_k_heads, self.head_k_dim)
        v = v.reshape(batch_size, seq_len, self.num_v_heads, self.head_v_dim)
        inv_scale = self.head_k_dim**-0.5
        q = (inv_scale**2) * mx.fast.rms_norm(q, None, QK_NORM_EPS)
        k = inv_scale * mx.fast.rms_norm(k, None, QK_NORM_EPS)
        g = compute_gate_decay(self.A_log, a, self.dt_bias)
        beta = mx.sigmoid(b)
        return q, k, v, g, beta

    def forward(self, hidden_states: mx.array, batch: Batch) -> mx.array:
        pool = get_global_ctx().linear_state_pool
        assert pool is not None and batch.fla_metadata is not None
        li = pool.local_index(self.layer_id)
        num_tokens = hidden_states.shape[0]
        conv_in, z = self.in_proj_qkvz.forward_split(hidden_states)
        b, a = self.in_proj_ba.forward_split(hidden_states)
        slots = batch.fla_metadata.cache_indices
        n_keep = self.conv_kernel_size - 1

        if batch.is_decode:
            conv_state = pool.conv_states[li][slots]
            conv_input = mx.concatenate([conv_state, conv_in[:, None, :]], axis=1)
            conv_out = nn.silu(self.conv1d.forward(conv_input))
            q, k, v, g, beta = self.recurrence_inputs(conv_out, a[:, None, :], b[:, None, :])
            state = pool.recurrent_states[li][slots]
            y, new_state = gated_delta_recurrence(q, k, v, g, beta, state)
            pool.conv_states[li] = scatter_rows(pool.conv_states[li], conv_input[:, 1:], slots)
            pool.recurrent_states[li] = scatter_rows(pool.recurrent_states[li], new_state, slots)
            core_out = y.reshape(num_tokens, self.num_v_heads, self.head_v_dim)
        else:
            outputs = []
            start = 0
            for i, seq_len in enumerate(batch.fla_metadata.seq_lens_host):
                slot = slots[i : i + 1]
                conv_state = pool.conv_states[li][slot]
                conv_input = mx.concatenate(
                    [conv_state, conv_in[None, start : start + seq_len]], axis=1
                )
                conv_out = nn.silu(self.conv1d.forward(conv_input))
                q, k, v, g, beta = self.recurrence_inputs(
                    conv_out, a[None, start : start + seq_len], b[None, start : start + seq_len]
                )
                state = pool.recurrent_states[li][slot]
                y, new_state = gated_delta_recurrence(q, k, v, g, beta, state)
                pool.conv_states[li] = scatter_rows(
                    pool.conv_states[li], conv_input[:, -n_keep:], slot
                )
                pool.recurrent_states[li] = scatter_rows(pool.recurrent_states[li], new_state, slot)
                outputs.append(y[0])
                start += seq_len
            core_out = mx.concatenate(outputs) if len(outputs) > 1 else outputs[0]

        z = z.reshape(num_tokens, self.num_v_heads, self.head_v_dim)
        out = self.norm.forward(core_out, z).reshape(num_tokens, self.value_dim)
        return self.out_proj.forward(out)


class DepthwiseConv1d(BaseOP):
    """Causal depthwise conv over [B, L, conv_dim] with the checkpoint's [conv_dim, K, 1] weight."""

    def __init__(self, conv_dim: int, kernel_size: int, dtype: mx.Dtype) -> None:
        super().__init__()
        self.conv_dim = conv_dim
        self.weight = mx.zeros((conv_dim, kernel_size, 1), dtype=dtype)

    def forward(self, x: mx.array) -> mx.array:
        return mx.conv1d(x, self.weight, groups=self.conv_dim)


__all__ = ["DepthwiseConv1d", "Qwen3_5GatedDeltaNet"]
