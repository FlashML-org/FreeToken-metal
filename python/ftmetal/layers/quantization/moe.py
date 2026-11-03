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
three.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal, Protocol

import mlx.core as mx
from mlx.utils import tree_unflatten

from ftmetal.kernel.moe.gather_sorted import FLOAT_INPUT_TILE, gather_mm_sorted, gather_qmm_sorted
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
    each projection's scheme, and where its banks live. top_k is FreeToken's field, read by
    no method yet."""

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
    "UnquantizedMoEMethod",
    "moe_method",
]
