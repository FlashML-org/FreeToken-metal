"""MoE methods: a layer's expert banks for its schemes, and its expert forward.

Port of FreeToken layers/quantization/moe/ (MoEConfig, MoEMethod and its unquantized method,
with their kernel's layout and apply). Divergences: moe_method picks the method instead of the
(kind, layer kind) registry, with no kernel table, pack or finalize step, since the banks load
straight from the checkpoint's MLX tensors; gate, up and down may each have their own scheme,
so a method gathers each projection by its scheme, and gate and up are two banks when theirs
differ; the forward is MLX's gather ops around the moe_align / moe_combine kernels instead of
one fused kernel; one layer's quantized projections are of one kind (affine or float block);
an offload cache's slot rows arrive as banks keyed like the layout, and the affine method also
runs the overlap prefill's expert-sorted gather_qmm_sorted.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal, Protocol

import mlx.core as mx
from mlx.utils import tree_unflatten

from ftmetal.kernel.gather_qmm_sorted import gather_qmm_sorted
from ftmetal.kernel.moe_align import moe_align, moe_combine
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
    # resident: the layer's own banks; offload: an offload cache's slot rows.
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

    def layout(self) -> dict[str, BankSpec]:
        """The layer's expert arrays (an offload cache's bank rows), by key under the layer."""
        return {
            f"{projection}.{role}": spec
            for projection in self.projections
            for role, spec in self.projection_layout(projection).items()
        }

    def create_weights(self, layer: BaseOP) -> None:
        """Set the layer's resident banks: placeholders the loader replaces."""
        banks = [
            (key, mx.zeros((self.cfg.num_experts, *spec.shape), dtype=spec.dtype))
            for key, spec in self.layout().items()
        ]
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
        """The experts' output for a routing decision: x [T, H], topk_weights / topk_ids
        [T, top_k]; from SORT_MIN_PAIRS pairs up the pairs run expert-sorted."""
        if topk_ids.size < SORT_MIN_PAIRS:
            return self.apply_unsorted(banks, x, topk_weights, topk_ids)
        else:
            alignment = moe_align(topk_ids, self.cfg.num_experts)
            sorted_x = mx.expand_dims(x, -2)[alignment.sorted_token_indices]
            expert_ids = alignment.sorted_expert_ids
            y = self.glu(
                sorted_x,
                lambda projection, h: self.gather(projection, banks, h, expert_ids, True),
            )
            return moe_combine(y, alignment.pair_positions, topk_weights)

    def apply_unsorted(
        self,
        banks: Mapping[str, mx.array],
        x: mx.array,
        topk_weights: mx.array,
        indices: mx.array,
    ) -> mx.array:
        """apply with each pair gathered where it is; indices [T, top_k]: the experts, or an
        offload pool's slots."""
        y = self.glu(
            mx.expand_dims(x, (-2, -3)),
            lambda projection, h: self.gather(projection, banks, h, indices, False),
        )
        return (y.squeeze(-2) * topk_weights[..., None]).sum(axis=-2)


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

    def apply_sorted(
        self,
        banks: Mapping[str, mx.array],
        x: mx.array,
        topk_weights: mx.array,
        topk_ids: mx.array,
        expert_bytes: Mapping[str, mx.array],
    ) -> mx.array:
        """apply over raw byte banks (an offload prefill buffer): the pairs sorted by expert,
        each expert's projection read at its byte offsets expert_bytes[projection] [E, 3]."""
        # The kernel reads the raw banks' scales and biases in x's dtype.
        assert x.dtype == self.cfg.dtype, f"x is {x.dtype}, the banks {self.cfg.dtype}"
        alignment = moe_align(topk_ids, self.cfg.num_experts)
        sorted_x = mx.expand_dims(x, -2)[alignment.sorted_token_indices]

        def project(projection: str, h: mx.array) -> mx.array:
            _, output_size, scheme = self.projections[projection]
            assert isinstance(scheme, AffineScheme), "offload experts are affine quantized"
            return gather_qmm_sorted(
                h,
                banks[f"{projection}.weight"],
                banks[f"{projection}.scales"],
                banks[f"{projection}.biases"],
                alignment.expert_offsets,
                expert_bytes[projection],
                n=output_size,
                group_size=scheme.group_size,
                bits=scheme.bits,
            )

        return moe_combine(self.glu(sorted_x, project), alignment.pair_positions, topk_weights)


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
                layout["output_scale"] = BankSpec((parts, 1), mx.float32, scheme)
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
            y = mx.gather_qmm(
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
            if not scheme.has_output_scale or projection == "down_proj":
                return y
            else:
                # y [..., 1, out] cut into its parts, each times its expert's scale.
                part_scales = banks[f"{projection}.output_scale"][indices]
                parts = y.reshape(*y.shape[:-2], part_scales.shape[-2], -1)
                return scale_output(parts, part_scales).reshape(y.shape)

    def apply(
        self,
        banks: Mapping[str, mx.array],
        x: mx.array,
        topk_weights: mx.array,
        topk_ids: mx.array,
    ) -> mx.array:
        """apply, with down's output scale on each pair's routing weight (down is linear); only
        here, as apply_unsorted runs under it."""
        down = self.projections["down_proj"][2]
        if isinstance(down, FpBlockScheme) and down.has_output_scale:
            down_scales = banks["down_proj.output_scale"][topk_ids][..., 0, 0]
            topk_weights = (topk_weights * down_scales).astype(topk_weights.dtype)
        return super().apply(banks, x, topk_weights, topk_ids)


def can_offload(scheme: QuantScheme | None) -> bool:
    """Whether an offload cache serves an expert projection in scheme: only the affine method
    has an expert-sorted kernel for the overlap prefill."""
    return isinstance(scheme, AffineScheme)


def moe_method(cfg: MoEConfig) -> MoEMethod:
    projections = cfg.schemes.projections(cfg.hidden, cfg.intermediate)
    kinds = {
        projection: type(scheme)
        for projection, (_, _, scheme) in projections.items()
        if scheme is not None
    }
    unsortable = [
        projection for projection, (_, _, scheme) in projections.items() if not can_offload(scheme)
    ]
    if len(set(kinds.values())) > 1:
        mixed = ", ".join(f"{projection} {kind.__name__}" for projection, kind in kinds.items())
        raise ValueError(f"a MoE layer's quantized experts are of one kind, not {mixed}")
    elif cfg.strategy == "offload" and unsortable:
        refused = ", ".join(unsortable)
        raise ValueError(f"MoE offload serves affine-quantized experts only, not {refused}")
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
    "can_offload",
    "moe_method",
]
