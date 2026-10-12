"""Routed experts: resident banks over mx.gather_qmm / mx.gather_mm, or an offload cache's.

Port of FreeToken layers/moe.py (MoELayer, OffloadMoELayer). Divergences: the experts live
in one fused gate_up_proj and one down_proj bank per layer in bf16 or affine quantization
(the arrays of expert_bank_layout), gated by silu or gelu_tanh (FreeToken's activation
argument); a checkpoint that stores gate and up in different schemes keeps them as two
banks under gate_up_proj, and down may differ from both; routing is computed by the model
and handed to forward; from SORT_MIN_PAIRS pairs up the pairs are expert-sorted by the
moe_align kernel and un-sorted by moe_combine. Resident and offload layers share glu and
differ only in where a projection gathers its experts' weights from.
OffloadMoELayer serves affine experts only, from the slot banks of the context's
offload cache: a decode or copy-miss forward gathers the slots its lookup returns, unsorted
(below four rows per expert the per-row matvec is the fast path); an overlap prefill sorts
its pairs and reads each expert at its slot map entry (gather_qmm_sorted), its prefill
buffers and the order of its reads and layers kept by the host, where FreeToken waits on
CUDA events.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from typing import Literal, Protocol

import mlx.core as mx
import mlx.nn as nn

from ftmetal.context import get_global_ctx
from ftmetal.kernel.gather_qmm_sorted import gather_qmm_sorted
from ftmetal.kernel.moe_align import moe_align, moe_combine
from ftmetal.layers.activation import gelu_tanh_and_mul, silu_and_mul
from ftmetal.layers.base import BaseOP
from ftmetal.layers.quantization.config import AffineScheme, BankSpec, expert_bank_layout
from ftmetal.moe.offload_cache import OffloadMoeCache

# From this many (token, expert) pairs the expert-sorted path is at least as fast as the
# unsorted one; measured on the M4 Pro at Qwen3.6-35B-A3B shapes (256 experts, top-8, 4-bit).
SORT_MIN_PAIRS = 32


class ExpertBank(BaseOP):
    """One [num_experts, output, input] projection shared by every expert of a layer."""

    # The roles of expert_bank_layout: weight always, scales and biases when affine.
    weight: mx.array
    scales: mx.array
    biases: mx.array

    def __init__(
        self,
        num_experts: int,
        input_size: int,
        output_size: int,
        *,
        scheme: AffineScheme | None,
        dtype: mx.Dtype,
    ) -> None:
        super().__init__()
        self.scheme = scheme
        for role, spec in expert_bank_layout(input_size, output_size, scheme, dtype).items():
            setattr(self, role, mx.zeros((num_experts, *spec.shape), dtype=spec.dtype))

    def forward(self, x: mx.array, indices: mx.array, sorted_indices: bool) -> mx.array:
        if self.scheme is None:
            return mx.gather_mm(
                x, self.weight.swapaxes(-1, -2), rhs_indices=indices, sorted_indices=sorted_indices
            )
        return mx.gather_qmm(
            x,
            self.weight,
            self.scales,
            self.biases,
            rhs_indices=indices,
            transpose=True,
            group_size=self.scheme.group_size,
            bits=self.scheme.bits,
            mode="affine",
            sorted_indices=sorted_indices,
        )


class ExpertBankParts(nn.Module):
    """The banks of gate_up_proj when the checkpoint stores gate and up in different schemes,
    under the parts' names (the loader leaves them apart)."""

    def __init__(self, gate_proj: ExpertBank, up_proj: ExpertBank) -> None:
        super().__init__()
        self.gate_proj = gate_proj
        self.up_proj = up_proj


@dataclass(frozen=True)
class ExpertSchemes:
    """The scheme of each expert projection as the checkpoint stores it."""

    gate: AffineScheme | None
    up: AffineScheme | None
    down: AffineScheme | None

    @classmethod
    def uniform(cls, scheme: AffineScheme | None) -> ExpertSchemes:
        return cls(scheme, scheme, scheme)

    def projections(
        self, hidden_size: int, intermediate_size: int
    ) -> dict[str, tuple[int, int, AffineScheme | None]]:
        """Each bank projection by its key under the layer, gate|up first: (input, output,
        scheme); gate and up are one projection when they share a scheme."""
        if self.gate == self.up:
            gate_up = {"gate_up_proj": (hidden_size, 2 * intermediate_size, self.gate)}
        else:
            gate_up = {
                "gate_up_proj.gate_proj": (hidden_size, intermediate_size, self.gate),
                "gate_up_proj.up_proj": (hidden_size, intermediate_size, self.up),
            }
        return {**gate_up, "down_proj": (intermediate_size, hidden_size, self.down)}


class ExpertProjection(Protocol):
    """One bank projection (a key of ExpertSchemes.projections) of x for a forward's experts."""

    def __call__(self, projection: str, x: mx.array, /) -> mx.array: ...


def moe_layout(
    hidden_size: int, intermediate_size: int, schemes: ExpertSchemes, dtype: mx.Dtype
) -> dict[str, BankSpec]:
    """The expert arrays of one MoELayer, by their key under the layer, in bank order."""
    return {
        f"{projection}.{role}": spec
        for projection, (input_size, output_size, scheme) in schemes.projections(
            hidden_size, intermediate_size
        ).items()
        for role, spec in expert_bank_layout(input_size, output_size, scheme, dtype).items()
    }


class MoELayer(BaseOP):
    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        *,
        schemes: ExpertSchemes,
        activation: Literal["silu", "gelu_tanh"] = "silu",
        dtype: mx.Dtype = mx.bfloat16,
        allocate_experts: bool = True,
    ) -> None:
        """Without allocate_experts the layer has no banks (an offload subclass's)."""
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.activation = activation
        self.schemes = schemes
        self.bank_dtype = dtype
        projections = schemes.projections(hidden_size, intermediate_size)
        self.gate_up_projections = tuple(key for key in projections if key != "down_proj")
        if allocate_experts:
            banks = {
                key: ExpertBank(num_experts, input_size, output_size, scheme=scheme, dtype=dtype)
                for key, (input_size, output_size, scheme) in projections.items()
            }
            if "gate_up_proj" in banks:
                self.gate_up_proj = banks["gate_up_proj"]
            else:
                self.gate_up_proj = ExpertBankParts(
                    banks["gate_up_proj.gate_proj"], banks["gate_up_proj.up_proj"]
                )
            self.down_proj = banks["down_proj"]

    def layout(self) -> dict[str, BankSpec]:
        """The layer's expert arrays (an offload cache's bank rows), by key under the layer."""
        return moe_layout(self.hidden_size, self.intermediate_size, self.schemes, self.bank_dtype)

    def bank(self, projection: str) -> ExpertBank:
        bank = reduce(getattr, projection.split("."), self)
        assert isinstance(bank, ExpertBank), f"{projection} is not an expert bank"
        return bank

    def act_and_mul(self, gate_up: mx.array) -> mx.array:
        if self.activation == "silu":
            return silu_and_mul(gate_up)
        else:
            return gelu_tanh_and_mul(gate_up)

    def glu(self, x: mx.array, project: ExpertProjection) -> mx.array:
        """down(act(gate x) * up x), each projection through project."""
        parts = [project(projection, x) for projection in self.gate_up_projections]
        gate_up = parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=-1)
        return project("down_proj", self.act_and_mul(gate_up))

    def forward(
        self, hidden_states: mx.array, topk_weights: mx.array, topk_ids: mx.array
    ) -> mx.array:
        """Expert compute for a routing decision (FreeToken's routed_forward): hidden [T, H],
        weights / ids [T, top_k]; the model computes the routing."""
        if topk_ids.size < SORT_MIN_PAIRS:
            x = mx.expand_dims(hidden_states, (-2, -3))
            y = self.glu(x, lambda projection, h: self.bank(projection).forward(h, topk_ids, False))
            return (y.squeeze(-2) * topk_weights[..., None]).sum(axis=-2)
        else:
            alignment = moe_align(topk_ids, self.num_experts)
            x = mx.expand_dims(hidden_states, -2)[alignment.sorted_token_indices]
            expert_ids = alignment.sorted_expert_ids
            y = self.glu(
                x, lambda projection, h: self.bank(projection).forward(h, expert_ids, True)
            )
            return moe_combine(y, alignment.pair_positions, topk_weights)


class OffloadMoELayer(MoELayer):
    def __init__(
        self,
        moe_layer_id: int,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        *,
        schemes: ExpertSchemes,
        activation: Literal["silu", "gelu_tanh"] = "silu",
        dtype: mx.Dtype = mx.bfloat16,
    ) -> None:
        """moe_layer_id: the layer's index among the model's MoE layers, the cache's layer."""
        if None in (schemes.gate, schemes.up, schemes.down):
            raise ValueError("MoE offload serves affine-quantized experts only; these are bf16")
        super().__init__(
            num_experts,
            top_k,
            hidden_size,
            intermediate_size,
            schemes=schemes,
            activation=activation,
            dtype=dtype,
            allocate_experts=False,
        )
        self.moe_layer_id = moe_layer_id

    def affine_scheme(self, projection: str) -> AffineScheme:
        scheme = self.schemes.projections(self.hidden_size, self.intermediate_size)[projection][2]
        assert scheme is not None, "offload experts are affine quantized"
        return scheme

    def forward(
        self, hidden_states: mx.array, topk_weights: mx.array, topk_ids: mx.array
    ) -> mx.array:
        cache = get_global_ctx().moe_offload_cache
        assert cache is not None, "an offload MoE layer runs on the engine's offload cache"
        if cache.mode == "overlap":
            return self.prefill_forward(cache, hidden_states, topk_weights, topk_ids)
        else:
            slots = cache.lookup_slots(self.moe_layer_id, topk_ids, hidden_states)
            x = mx.expand_dims(hidden_states, (-2, -3))
            y = self.glu(x, lambda projection, h: self.gather(cache, projection, h, slots))
            return (y.squeeze(-2) * topk_weights[..., None]).sum(axis=-2)

    def prefill_forward(
        self,
        cache: OffloadMoeCache,
        hidden_states: mx.array,
        topk_weights: mx.array,
        topk_ids: mx.array,
    ) -> mx.array:
        """An overlap prefill layer: the layer's input runs while its prefill buffer is read,
        its MoE runs as a command buffer of its own once the buffer is in."""
        if self.moe_layer_id == 0:
            cache.begin_prefill_forward()
        mx.async_eval(hidden_states, topk_weights, topk_ids)
        slot_map = cache.prefill_slot_map(self.moe_layer_id)
        alignment = moe_align(topk_ids, self.num_experts)
        x = mx.expand_dims(hidden_states, -2)[alignment.sorted_token_indices]
        offsets = alignment.expert_offsets
        y = self.glu(
            x, lambda projection, h: self.gather_sorted(cache, projection, h, offsets, slot_map)
        )
        out = moe_combine(y, alignment.pair_positions, topk_weights)
        mx.async_eval(out)
        cache.end_prefill_layer(self.moe_layer_id, out)
        return out

    def gather(
        self, cache: OffloadMoeCache, projection: str, x: mx.array, slots: mx.array
    ) -> mx.array:
        """x [T, 1, 1, K] through the projection of the experts at slots [T, top_k]."""
        scheme = self.affine_scheme(projection)
        return mx.gather_qmm(
            x,
            cache.banks[f"{projection}.weight"],
            cache.banks[f"{projection}.scales"],
            cache.banks[f"{projection}.biases"],
            rhs_indices=slots,
            transpose=True,
            group_size=scheme.group_size,
            bits=scheme.bits,
            mode="affine",
        )

    def gather_sorted(
        self,
        cache: OffloadMoeCache,
        projection: str,
        x: mx.array,
        expert_offsets: mx.array,
        slot_map: mx.array,
    ) -> mx.array:
        """x [P, 1, K], its pairs sorted by expert, through each expert's projection at its
        slot map entry."""
        scheme = self.affine_scheme(projection)
        return gather_qmm_sorted(
            x,
            cache.banks[f"{projection}.weight"],
            cache.banks[f"{projection}.scales"],
            cache.banks[f"{projection}.biases"],
            expert_offsets,
            slot_map,
            group_size=scheme.group_size,
            bits=scheme.bits,
        )


__all__ = [
    "SORT_MIN_PAIRS",
    "ExpertBank",
    "ExpertBankParts",
    "ExpertProjection",
    "ExpertSchemes",
    "MoELayer",
    "OffloadMoELayer",
    "moe_layout",
]
