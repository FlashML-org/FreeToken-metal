"""MoE methods: a layer's expert banks for its schemes, and its expert forward.

Port of FreeToken layers/quantization/moe/ (MoEConfig, MoEMethod and its unquantized method,
with their kernel's layout and apply). Divergences: moe_method picks the method instead of the
(kind, layer kind) registry, with no kernel table, pack or finalize step, since the banks load
straight from the checkpoint's MLX tensors; gate, up and down may each have their own scheme,
so a method gathers each projection by its scheme, and gate and up are two banks when theirs
differ; the forward is MLX's gather ops around the moe_align / moe_combine kernels instead of
one fused kernel; one layer's quantized projections are of one kind (affine or float block).
A method reads its experts from resident banks (apply), from an offload pool's slot rows, keyed
like the layout (apply_slots), and from an offload cache's raw byte banks at each expert's byte
offsets (apply_sorted, on the expert-sorted kernels); a resident array (an export's
output_scale, FreeToken's resident BankSpec) stays the layer's and is indexed by expert in all
three. An offload pool's affine rows may hold pair codes (an expert pack's, moe/pack.py) in place
of their scales and biases, which the pair-code kernels look up in the layer's pair table of
2^code_bits rows. An affine offload decode may instead run on its issued lookup
(layers/quantization/moe_pending.py).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal, NamedTuple, Protocol

import mlx.core as mx
import numpy as np
from mlx.utils import tree_unflatten

from ftmetal.kernel.moe.affine.affine_gather_qmv_pair_codes import affine_gather_qmv_pair_codes
from ftmetal.kernel.moe.gather_sorted import (
    FLOAT_INPUT_TILE,
    gather_mm_sorted,
    gather_qmm_sorted,
    gather_qmm_sorted_pair_codes,
)
from ftmetal.kernel.moe.moe_align import ExpertAlignment, moe_align, moe_combine
from ftmetal.layers.activation import gelu_tanh_and_mul, silu_and_mul
from ftmetal.layers.base import BaseOP
from ftmetal.layers.quantization.config import AffineScheme, BankSpec, FpBlockScheme, QuantScheme
from ftmetal.layers.quantization.linear import scale_output

# From this many (token, expert) pairs the expert-sorted path is at least as fast as the
# unsorted one; measured on the M4 Pro at Qwen3.6-35B-A3B shapes (256 experts, top-8, 4-bit).
SORT_MIN_PAIRS = 32


@dataclass(frozen=True)
class ExpertSchemes:
    """The scheme of each expert projection as the checkpoint stores it."""

    gate: QuantScheme | None
    up: QuantScheme | None
    down: QuantScheme | None
    # Gate and up as two banks even in one scheme.
    is_gate_up_apart: bool = field(default=False, kw_only=True)

    @classmethod
    def uniform(cls, scheme: QuantScheme | None) -> ExpertSchemes:
        return cls(scheme, scheme, scheme)

    def projections(
        self, hidden_size: int, intermediate_size: int
    ) -> dict[str, tuple[int, int, QuantScheme | None]]:
        """Each bank projection by its key under the layer, gate|up first: (input, output,
        scheme); gate and up are one projection when they share a scheme and are not kept
        apart."""
        if self.gate == self.up and not self.is_gate_up_apart:
            gate_up = {"gate_up_proj": (hidden_size, 2 * intermediate_size, self.gate)}
        else:
            gate_up = {
                "gate_up_proj.gate_proj": (hidden_size, intermediate_size, self.gate),
                "gate_up_proj.up_proj": (hidden_size, intermediate_size, self.up),
            }
        return {**gate_up, "down_proj": (intermediate_size, hidden_size, self.down)}


@dataclass(frozen=True, kw_only=True)
class MoEConfig:
    """What a MoE method sizes and computes by: the layer's shape, activation and float dtype,
    each projection's scheme, and where its banks live. top_k is FreeToken's field; the pending
    kernels' eligibility reads it (moe_pending.py)."""

    num_experts: int
    top_k: int
    hidden: int
    intermediate: int
    schemes: ExpertSchemes
    activation: Literal["silu", "gelu_tanh"]
    dtype: mx.Dtype
    # resident: the layer's own banks; offload: an offload cache's slot rows, beside the
    # layer's resident arrays.
    strategy: Literal["resident", "offload"]


class ExpertProjection(Protocol):
    """One bank projection (a key of ExpertSchemes.projections) of x for a forward's experts."""

    def __call__(self, projection: str, x: mx.array, /) -> mx.array: ...


class SortedSplit(NamedTuple):
    """apply_sorted in two passes (apply_sorted_first, apply_sorted_rest): the pairs sorted
    with the experts of order's head first, the head's glu output (the first num_rows sorted
    pairs), and what the rest needs."""

    alignment: ExpertAlignment
    sorted_x: mx.array
    # int32 [E]: the experts in the sort's order, the head's num_experts first.
    order: mx.array
    num_experts: int
    num_rows: int
    head: mx.array


class SharedExpertForward(Protocol):
    """A shared expert's output and gate logit [T, 1] for x [T, H]; the MoE output adds the
    output gated by the logit's sigmoid, or as it is when the expert has no gate (None)."""

    def __call__(self, x: mx.array, /) -> tuple[mx.array, mx.array | None]: ...


class MoEMethod(ABC):
    def __init__(self, cfg: MoEConfig) -> None:
        self.cfg = cfg
        self.projections = cfg.schemes.projections(cfg.hidden, cfg.intermediate)

    @abstractmethod
    def projection_layout(self, projection: str) -> dict[str, BankSpec]:
        """The arrays of one bank projection, per expert, by role."""

    @abstractmethod
    def gather(
        self,
        projection: str,
        banks: Mapping[str, mx.array],
        x: mx.array,
        indices: mx.array,
        sorted_indices: bool,
    ) -> mx.array:
        """x through the projection of the experts (or slots) at indices; banks: the layer's
        arrays keyed as layout, the expert or slot axis prepended."""

    @abstractmethod
    def gather_sorted(
        self,
        projection: str,
        banks: Mapping[str, mx.array],
        x: mx.array,
        expert_offsets: mx.array,
        expert_bytes: mx.array,
    ) -> mx.array:
        """x, its rows sorted by expert (moe_align's expert_offsets), through each expert's
        projection read at its byte offsets expert_bytes [E, arrays] into raw byte banks, one
        column per bank array of the projection in layout order."""

    def scale_parts(
        self, projection: str, banks: Mapping[str, mx.array], y: mx.array, expert_ids: mx.array
    ) -> mx.array:
        """A projection's output y for the experts at expert_ids, times any per-expert scale of
        its parts; down's scales the routing weights instead (routing_weights)."""
        return y

    def routing_weights(
        self, banks: Mapping[str, mx.array], topk_weights: mx.array, topk_ids: mx.array
    ) -> mx.array:
        """topk_weights times any per-expert scale of down's output (down is linear)."""
        return topk_weights

    def layout(self) -> dict[str, BankSpec]:
        """The layer's expert arrays by key under the layer: the banks (an offload cache's bank
        rows) and its resident arrays."""
        return {
            f"{projection}.{role}": spec
            for projection in self.projections
            for role, spec in self.projection_layout(projection).items()
        }

    def create_weights(self, layer: BaseOP) -> None:
        """Set the layer's banks, or an offload layer's resident arrays alone: placeholders the
        loader replaces."""
        banks = [
            (key, mx.zeros((self.cfg.num_experts, *spec.shape), dtype=spec.dtype))
            for key, spec in self.layout().items()
            if self.cfg.strategy == "resident" or spec.resident
        ]
        if banks:
            for name, value in tree_unflatten(banks).items():
                setattr(layer, name, value)

    def glu(self, x: mx.array, project: ExpertProjection) -> mx.array:
        """down(act(gate x) * up x), each projection through project."""
        parts = [
            project(projection, x) for projection in self.projections if projection != "down_proj"
        ]
        gate_up = parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=-1)
        act_and_mul = silu_and_mul if self.cfg.activation == "silu" else gelu_tanh_and_mul
        return project("down_proj", act_and_mul(gate_up))

    def apply(
        self,
        banks: Mapping[str, mx.array],
        x: mx.array,
        topk_weights: mx.array,
        topk_ids: mx.array,
    ) -> mx.array:
        """The experts' output for a routing decision over resident banks: x [T, H],
        topk_weights / topk_ids [T, top_k]; from SORT_MIN_PAIRS pairs up the pairs run
        expert-sorted."""
        if topk_ids.size < SORT_MIN_PAIRS:
            return self.apply_slots(banks, x, topk_weights, topk_ids, topk_ids)
        else:
            alignment = moe_align(topk_ids, self.cfg.num_experts)
            expert_ids = alignment.sorted_expert_ids

            def project(projection: str, h: mx.array) -> mx.array:
                y = self.gather(projection, banks, h, expert_ids, True)
                return self.scale_parts(projection, banks, y, expert_ids)

            return self.sorted_forward(banks, x, topk_weights, topk_ids, alignment, project)

    def apply_slots(
        self,
        banks: Mapping[str, mx.array],
        x: mx.array,
        topk_weights: mx.array,
        topk_ids: mx.array,
        slots: mx.array,
    ) -> mx.array:
        """apply with each pair gathered where it is, unsorted; slots [T, top_k]: each pair's
        row of the banks (its expert in resident banks, its slot in an offload pool's)."""

        def project(projection: str, h: mx.array) -> mx.array:
            y = self.gather(projection, banks, h, slots, False)
            return self.scale_parts(projection, banks, y, topk_ids)

        y = self.glu(mx.expand_dims(x, (-2, -3)), project)
        weights = self.routing_weights(banks, topk_weights, topk_ids)
        return (y.squeeze(-2) * weights[..., None]).sum(axis=-2)

    def apply_sorted(
        self,
        banks: Mapping[str, mx.array],
        x: mx.array,
        topk_weights: mx.array,
        topk_ids: mx.array,
        expert_bytes: Mapping[str, mx.array],
    ) -> mx.array:
        """apply over an offload cache's raw byte banks (beside the layer's resident arrays):
        the pairs sorted by expert, each expert's projection read at its byte offsets
        expert_bytes[projection] [E, arrays]."""
        # The kernels read the raw banks' float arrays (scales, biases, bf16 weights) in x's
        # dtype.
        assert x.dtype == self.cfg.dtype, f"x is {x.dtype}, the banks {self.cfg.dtype}"
        alignment = moe_align(topk_ids, self.cfg.num_experts)
        offsets, expert_ids = alignment.expert_offsets, alignment.sorted_expert_ids

        def project(projection: str, h: mx.array) -> mx.array:
            y = self.gather_sorted(projection, banks, h, offsets, expert_bytes[projection])
            return self.scale_parts(projection, banks, y, expert_ids)

        return self.sorted_forward(banks, x, topk_weights, topk_ids, alignment, project)

    def apply_sorted_first(
        self,
        banks: Mapping[str, mx.array],
        x: mx.array,
        topk_ids: mx.array,
        order: np.ndarray,
        num_experts: int,
        num_rows: int,
        expert_bytes: Mapping[str, mx.array],
    ) -> SortedSplit:
        """apply_sorted's first pass: the pairs of the experts order[:num_experts] (the
        routing's first num_rows pairs once sorted in order, every expert once in it) through
        glu, for apply_sorted_rest to finish. Only where each pair is computed changes: every
        pair's row is the same kernel's over the same expert, the combine the same."""
        assert x.dtype == self.cfg.dtype, f"x is {x.dtype}, the banks {self.cfg.dtype}"
        rank = np.empty(self.cfg.num_experts, dtype=np.int32)
        rank[order] = np.arange(self.cfg.num_experts, dtype=np.int32)
        alignment = moe_align(mx.take(mx.array(rank), topk_ids), self.cfg.num_experts)
        sorted_x = mx.expand_dims(x, -2)[alignment.sorted_token_indices]
        split = SortedSplit(
            alignment, sorted_x, mx.array(order.astype(np.int32)), num_experts, num_rows, sorted_x
        )
        head = self.glu(
            sorted_x[:num_rows], self.part_projection(banks, split, expert_bytes, is_head=True)
        )
        return split._replace(head=head)

    def apply_sorted_rest(
        self,
        banks: Mapping[str, mx.array],
        split: SortedSplit,
        topk_weights: mx.array,
        topk_ids: mx.array,
        expert_bytes: Mapping[str, mx.array],
    ) -> mx.array:
        """apply_sorted's second pass after apply_sorted_first: the other pairs through glu,
        then every pair combined by its routing weight."""
        rest = self.glu(
            split.sorted_x[split.num_rows :],
            self.part_projection(banks, split, expert_bytes, is_head=False),
        )
        weights = self.routing_weights(banks, topk_weights, topk_ids)
        out = mx.concatenate([split.head, rest])
        return moe_combine(out, split.alignment.pair_positions, weights)

    def part_projection(
        self,
        banks: Mapping[str, mx.array],
        split: SortedSplit,
        expert_bytes: Mapping[str, mx.array],
        is_head: bool,
    ) -> ExpertProjection:
        """The sorted projection of the split's head experts' pairs, or its other ones."""
        num_experts, num_rows = split.num_experts, split.num_rows
        experts = slice(0, num_experts) if is_head else slice(num_experts, self.cfg.num_experts)
        rows = slice(0, num_rows) if is_head else slice(num_rows, None)
        offsets = split.alignment.expert_offsets[experts.start : experts.stop + 1]
        offsets = offsets if is_head else offsets - num_rows
        part_experts = split.order[experts]
        expert_ids = mx.take(split.order, split.alignment.sorted_expert_ids[rows])

        def project(projection: str, h: mx.array) -> mx.array:
            part_bytes = mx.take(expert_bytes[projection], part_experts, axis=0)
            y = self.gather_sorted(projection, banks, h, offsets, part_bytes)
            return self.scale_parts(projection, banks, y, expert_ids)

        return project

    def sorted_forward(
        self,
        banks: Mapping[str, mx.array],
        x: mx.array,
        topk_weights: mx.array,
        topk_ids: mx.array,
        alignment: ExpertAlignment,
        project: ExpertProjection,
    ) -> mx.array:
        """The pairs of x sorted by expert through glu(project), combined by their routing
        weights."""
        sorted_x = mx.expand_dims(x, -2)[alignment.sorted_token_indices]
        weights = self.routing_weights(banks, topk_weights, topk_ids)
        return moe_combine(self.glu(sorted_x, project), alignment.pair_positions, weights)


class UnquantizedMoEMethod(MoEMethod):
    def projection_layout(self, projection: str) -> dict[str, BankSpec]:
        input_size, output_size, _ = self.projections[projection]
        return {"weight": BankSpec((output_size, input_size), self.cfg.dtype)}

    def gather(
        self,
        projection: str,
        banks: Mapping[str, mx.array],
        x: mx.array,
        indices: mx.array,
        sorted_indices: bool,
    ) -> mx.array:
        return mx.gather_mm(
            x,
            banks[f"{projection}.weight"].swapaxes(-1, -2),
            rhs_indices=indices,
            sorted_indices=sorted_indices,
        )

    def gather_sorted(
        self,
        projection: str,
        banks: Mapping[str, mx.array],
        x: mx.array,
        expert_offsets: mx.array,
        expert_bytes: mx.array,
    ) -> mx.array:
        _, output_size, _ = self.projections[projection]
        weight = banks[f"{projection}.weight"]
        return gather_mm_sorted(x, weight, expert_offsets, expert_bytes, n=output_size)


class AffineMoEMethod(UnquantizedMoEMethod):
    """Affine-quantized experts; a projection the checkpoint keeps in bf16 gathers as the
    unquantized method's."""

    def projection_layout(self, projection: str) -> dict[str, BankSpec]:
        input_size, output_size, scheme = self.projections[projection]
        if scheme is None:
            return super().projection_layout(projection)
        else:
            assert isinstance(scheme, AffineScheme), f"{projection} is not affine quantized"
            return scheme.layout(input_size, output_size, self.cfg.dtype)

    def gather(
        self,
        projection: str,
        banks: Mapping[str, mx.array],
        x: mx.array,
        indices: mx.array,
        sorted_indices: bool,
    ) -> mx.array:
        scheme = self.projections[projection][2]
        if scheme is None:
            return super().gather(projection, banks, x, indices, sorted_indices)
        else:
            assert isinstance(scheme, AffineScheme), f"{projection} is not affine quantized"
            if f"{projection}.codes" in banks:
                assert not sorted_indices, "pair codes are an offload pool's rows, gathered by slot"
                pairs = banks["pairs"]
                return affine_gather_qmv_pair_codes(
                    x,
                    banks[f"{projection}.weight"],
                    banks[f"{projection}.codes"],
                    pairs,
                    indices,
                    group_size=scheme.group_size,
                    bits=scheme.bits,
                    code_bits=pairs.shape[0].bit_length() - 1,
                )
            else:
                return mx.gather_qmm(
                    x,
                    banks[f"{projection}.weight"],
                    banks[f"{projection}.scales"],
                    banks[f"{projection}.biases"],
                    rhs_indices=indices,
                    transpose=True,
                    group_size=scheme.group_size,
                    bits=scheme.bits,
                    mode="affine",
                    sorted_indices=sorted_indices,
                )

    def gather_sorted(
        self,
        projection: str,
        banks: Mapping[str, mx.array],
        x: mx.array,
        expert_offsets: mx.array,
        expert_bytes: mx.array,
    ) -> mx.array:
        _, output_size, scheme = self.projections[projection]
        if scheme is None:
            return super().gather_sorted(projection, banks, x, expert_offsets, expert_bytes)
        else:
            assert isinstance(scheme, AffineScheme), f"{projection} is not affine quantized"
            if f"{projection}.codes" in banks:
                pairs = banks["pairs"]
                return gather_qmm_sorted_pair_codes(
                    x,
                    banks[f"{projection}.weight"],
                    banks[f"{projection}.codes"],
                    pairs,
                    expert_offsets,
                    expert_bytes,
                    n=output_size,
                    group_size=scheme.group_size,
                    bits=scheme.bits,
                    code_bits=pairs.shape[0].bit_length() - 1,
                )
            else:
                return gather_qmm_sorted(
                    x,
                    banks[f"{projection}.weight"],
                    banks[f"{projection}.scales"],
                    banks[f"{projection}.biases"],
                    expert_offsets,
                    expert_bytes,
                    n=output_size,
                    group_size=scheme.group_size,
                    bits=scheme.bits,
                    mode="affine",
                )


class FpBlockMoEMethod(UnquantizedMoEMethod):
    """Experts in an MLX float block mode; a projection the checkpoint keeps in bf16 gathers as
    the unquantized method's. An output_scale is one float32 per expert and fused part,
    [parts, 1]: gate's and up's apply to their outputs, down's to each pair's routing weight."""

    def projection_layout(self, projection: str) -> dict[str, BankSpec]:
        input_size, output_size, scheme = self.projections[projection]
        if scheme is None:
            return super().projection_layout(projection)
        else:
            assert isinstance(scheme, FpBlockScheme), f"{projection} is not float block quantized"
            layout = scheme.layout(input_size, output_size, self.cfg.dtype)
            if scheme.has_output_scale:
                parts = 2 if projection == "gate_up_proj" else 1
                layout["output_scale"] = BankSpec((parts, 1), mx.float32, scheme, resident=True)
            return layout

    def gather(
        self,
        projection: str,
        banks: Mapping[str, mx.array],
        x: mx.array,
        indices: mx.array,
        sorted_indices: bool,
    ) -> mx.array:
        scheme = self.projections[projection][2]
        if scheme is None:
            return super().gather(projection, banks, x, indices, sorted_indices)
        else:
            assert isinstance(scheme, FpBlockScheme), f"{projection} is not float block quantized"
            return mx.gather_qmm(
                x,
                banks[f"{projection}.weight"],
                banks[f"{projection}.scales"],
                rhs_indices=indices,
                transpose=True,
                group_size=scheme.group_size,
                bits=scheme.bits,
                mode=scheme.mode,
                sorted_indices=sorted_indices,
            )

    def gather_sorted(
        self,
        projection: str,
        banks: Mapping[str, mx.array],
        x: mx.array,
        expert_offsets: mx.array,
        expert_bytes: mx.array,
    ) -> mx.array:
        _, output_size, scheme = self.projections[projection]
        if scheme is None:
            return super().gather_sorted(projection, banks, x, expert_offsets, expert_bytes)
        else:
            assert isinstance(scheme, FpBlockScheme), f"{projection} is not float block quantized"
            return gather_qmm_sorted(
                x,
                banks[f"{projection}.weight"],
                banks[f"{projection}.scales"],
                None,
                expert_offsets,
                expert_bytes,
                n=output_size,
                group_size=scheme.group_size,
                bits=scheme.bits,
                mode=scheme.mode,
            )

    def scale_parts(
        self, projection: str, banks: Mapping[str, mx.array], y: mx.array, expert_ids: mx.array
    ) -> mx.array:
        scheme = self.projections[projection][2]
        if (
            not isinstance(scheme, FpBlockScheme)
            or not scheme.has_output_scale
            or projection == "down_proj"
        ):
            return y
        else:
            # y [..., 1, out] cut into its parts, each times its expert's scale.
            part_scales = banks[f"{projection}.output_scale"][expert_ids]
            parts = y.reshape(*y.shape[:-2], part_scales.shape[-2], -1)
            return scale_output(parts, part_scales).reshape(y.shape)

    def routing_weights(
        self, banks: Mapping[str, mx.array], topk_weights: mx.array, topk_ids: mx.array
    ) -> mx.array:
        down = self.projections["down_proj"][2]
        if isinstance(down, FpBlockScheme) and down.has_output_scale:
            down_scales = banks["down_proj.output_scale"][topk_ids][..., 0, 0]
            return (topk_weights * down_scales).astype(topk_weights.dtype)
        else:
            return topk_weights


def moe_method(cfg: MoEConfig) -> MoEMethod:
    projections = cfg.schemes.projections(cfg.hidden, cfg.intermediate)
    kinds = {
        projection: type(scheme)
        for projection, (_, _, scheme) in projections.items()
        if scheme is not None
    }
    untiled = [
        f"{projection} ({input_size})"
        for projection, (input_size, _, scheme) in projections.items()
        if isinstance(scheme, FpBlockScheme) and input_size % FLOAT_INPUT_TILE != 0
    ]
    if len(set(kinds.values())) > 1:
        mixed = ", ".join(f"{projection} {kind.__name__}" for projection, kind in kinds.items())
        raise ValueError(f"a MoE layer's quantized experts are of one kind, not {mixed}")
    elif cfg.strategy == "offload" and untiled:
        raise ValueError(
            f"MoE offload reads float block experts {FLOAT_INPUT_TILE} inputs at a time, not "
            f"{', '.join(untiled)} (--moe-strategy fused keeps them resident)"
        )
    elif not kinds:
        return UnquantizedMoEMethod(cfg)
    elif AffineScheme in kinds.values():
        return AffineMoEMethod(cfg)
    else:
        return FpBlockMoEMethod(cfg)


__all__ = [
    "SORT_MIN_PAIRS",
    "AffineMoEMethod",
    "ExpertProjection",
    "ExpertSchemes",
    "FpBlockMoEMethod",
    "MoEConfig",
    "MoEMethod",
    "SharedExpertForward",
    "UnquantizedMoEMethod",
    "moe_method",
]
