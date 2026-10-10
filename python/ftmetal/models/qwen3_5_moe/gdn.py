"""Gated DeltaNet layer over the per-request state pool.

Port of FreeToken models/qwen3_5_moe/gdn.py. Divergences: decode is one fused
launch per layer (conv, gating, q/k norm, recurrence) that updates the conv ring and
the recurrent state in place; prefill runs one request at a time (varlen): one fused
launch for conv over the ring's last taps plus the chunk, gating and q/k norm, the
chunked gated delta rule (the per-token recurrence below CHUNKED_MIN_TOKENS), and the
new states scattered into the pool. A prefill request with a hybrid-radix track
boundary runs the recurrence in two parts and writes the state and conv ring at the
boundary into its track slot.
"""

from __future__ import annotations

import mlx.core as mx

from ftmetal.attention.metadata import AttnType, LinearAttnMetadata
from ftmetal.context import get_global_ctx
from ftmetal.core import Batch
from ftmetal.kernel.gdn import gated_delta_recurrence, gdn_decode_step, gdn_prefill_inputs
from ftmetal.kernel.gdn_chunk import gated_delta_chunked
from ftmetal.kernel.inplace import scatter_rows
from ftmetal.kvcache.linear_state_pool import LinearStatePool
from ftmetal.layers.base import BaseOP
from ftmetal.layers.linear import Linear, LinearMerged
from ftmetal.layers.norm import GatedRMSNorm
from ftmetal.layers.quantization.config import AffineScheme
from ftmetal.models.config import LinearGatedDeltaGroupConfig
from ftmetal.models.qwen3_5_moe.weight import IN_PROJ_BA_PARTS, IN_PROJ_QKVZ_PARTS

# Shortest prefill run chunked; the per-token kernel is faster below it (crossover at
# 52-56 tokens on the M4 Pro at the Qwen3.6 shapes, calls serialized as layers are).
CHUNKED_MIN_TOKENS = 56


class Qwen3_5GatedDeltaNet(BaseOP):
    def __init__(
        self,
        hidden_size: int,
        group: LinearGatedDeltaGroupConfig,
        rms_norm_eps: float,
        layer_id: int,
        *,
        qkvz_schemes: list[AffineScheme | None],
        ba_schemes: list[AffineScheme | None],
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
            hidden_size,
            [self.conv_dim, self.value_dim],
            schemes=qkvz_schemes,
            part_names=IN_PROJ_QKVZ_PARTS,
            dtype=dtype,
        )
        self.in_proj_ba = LinearMerged(
            hidden_size,
            [self.num_v_heads, self.num_v_heads],
            schemes=ba_schemes,
            part_names=IN_PROJ_BA_PARTS,
            dtype=dtype,
        )
        self.conv1d = DepthwiseConv1d(self.conv_dim, self.conv_kernel_size, dtype)
        self.dt_bias = mx.ones((self.num_v_heads,), dtype=dtype)
        self.A_log = mx.zeros((self.num_v_heads,), dtype=dtype)
        self.norm = GatedRMSNorm(self.head_v_dim, eps=rms_norm_eps, dtype=dtype)
        self.out_proj = Linear(self.value_dim, hidden_size, scheme=out_scheme, dtype=dtype)

    @staticmethod
    def recurrence(
        q: mx.array, k: mx.array, v: mx.array, g: mx.array, beta: mx.array, state: mx.array
    ) -> tuple[mx.array, mx.array]:
        """The gated delta rule over one request's chunk: chunked from CHUNKED_MIN_TOKENS."""
        if q.shape[1] >= CHUNKED_MIN_TOKENS:
            return gated_delta_chunked(q, k, v, g, beta, state)
        return gated_delta_recurrence(q, k, v, g, beta, state)

    def recurrence_with_snapshot(
        self,
        pool: LinearStatePool,
        li: int,
        inputs: tuple[mx.array, mx.array, mx.array, mx.array, mx.array],
        state: mx.array,
        taps: mx.array,
        chunk: mx.array,
        first: int,
        split: int,
        track_slot: mx.array,
    ) -> tuple[mx.array, mx.array]:
        """The chunk's recurrence from state, split after split tokens; returns y and the end state.

        The state and the conv ring after the first split tokens (positions first ..
        first + split - 1) go to track_slot. taps [kernel - 1, conv_dim] are the conv
        inputs before the chunk, chunk [T, conv_dim] the chunk's own.
        """
        kernel = self.conv_kernel_size
        assert split >= 1
        q, k, v, g, beta = inputs
        y_head, split_state = self.recurrence(
            q[:, :split], k[:, :split], v[:, :split], g[:, :split], beta[:, :split], state
        )
        y_tail, end_state = self.recurrence(
            q[:, split:], k[:, split:], v[:, split:], g[:, split:], beta[:, split:], split_state
        )
        # The ring holds position p at cell p mod kernel; the last kernel rows of recent are
        # positions first + split - kernel .. first + split - 1.
        recent = mx.concatenate([taps, chunk[max(0, split - kernel) : split]])[-kernel:]
        ring = recent[mx.array([(cell - first - split) % kernel for cell in range(kernel)])]
        pool.conv_states[li] = scatter_rows(pool.conv_states[li], ring[None], track_slot)
        pool.recurrent_states[li] = scatter_rows(pool.recurrent_states[li], split_state, track_slot)
        return mx.concatenate([y_head, y_tail], axis=1), end_state

    def forward(self, hidden_states: mx.array, batch: Batch) -> mx.array:
        pool = get_global_ctx().linear_state_pool
        metadata = batch.attn_metadata[AttnType.LINEAR]
        assert pool is not None and isinstance(metadata, LinearAttnMetadata)
        li = pool.local_index(self.layer_id)
        num_tokens = hidden_states.shape[0]
        conv_in, z = self.in_proj_qkvz.forward_split(hidden_states)
        b, a = self.in_proj_ba.forward_split(hidden_states)
        slots = metadata.cache_indices
        kernel = self.conv_kernel_size

        if batch.is_decode:
            pool.conv_states[li], pool.recurrent_states[li], core_out = gdn_decode_step(
                pool.conv_states[li],
                pool.recurrent_states[li],
                conv_in,
                self.conv1d.weight.reshape(self.conv_dim, kernel),
                a,
                b,
                self.A_log,
                self.dt_bias,
                slots,
                batch.positions,
            )
        else:
            conv_weight = self.conv1d.weight.reshape(self.conv_dim, kernel)
            outputs = []
            start = 0
            for i, seq_len in enumerate(metadata.seq_lens_host):
                slot = slots[i : i + 1]
                first = batch.reqs[i].cached_len
                last = first + seq_len - 1
                ring = pool.conv_states[li][slot][0]
                # The kernel - 1 inputs before this chunk, oldest first, by position mod kernel.
                tap_cells = [(first - kernel + 1 + j) % kernel for j in range(kernel - 1)]
                taps = ring[mx.array(tap_cells)]
                chunk = conv_in[start : start + seq_len]
                inputs = gdn_prefill_inputs(
                    chunk,
                    taps,
                    conv_weight,
                    a[start : start + seq_len],
                    b[start : start + seq_len],
                    self.A_log,
                    self.dt_bias,
                    self.num_k_heads,
                )
                state = pool.recurrent_states[li][slot]
                if metadata.track_slots_host[i] < 0:
                    y, new_state = self.recurrence(*inputs, state)
                else:
                    y, new_state = self.recurrence_with_snapshot(
                        pool,
                        li,
                        inputs,
                        state,
                        taps,
                        chunk,
                        first,
                        metadata.track_seqlens_host[i] - first,
                        metadata.track_slots[i : i + 1],
                    )
                written = min(kernel, seq_len)
                cells = mx.array([(last - j) % kernel for j in range(written)])
                ring[cells] = chunk[seq_len - 1 - mx.arange(written)]
                pool.conv_states[li] = scatter_rows(pool.conv_states[li], ring[None], slot)
                pool.recurrent_states[li] = scatter_rows(pool.recurrent_states[li], new_state, slot)
                outputs.append(y[0])
                start += seq_len
            core_out = mx.concatenate(outputs) if len(outputs) > 1 else outputs[0]

        z = z.reshape(num_tokens, self.num_v_heads, self.head_v_dim)
        out = self.norm.forward(core_out, z).reshape(num_tokens, self.value_dim)
        return self.out_proj.forward(out)


class DepthwiseConv1d(BaseOP):
    """Holds the checkpoint's causal depthwise conv weight [conv_dim, K, 1] (key conv1d.weight);
    the fused kernels read it directly, as in FreeToken."""

    def __init__(self, conv_dim: int, kernel_size: int, dtype: mx.Dtype) -> None:
        super().__init__()
        self.weight = mx.zeros((conv_dim, kernel_size, 1), dtype=dtype)

    def forward(self) -> mx.array:
        return self.weight


__all__ = ["DepthwiseConv1d", "Qwen3_5GatedDeltaNet"]
