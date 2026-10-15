"""Synthetic MoE checkpoints, the offload cache over one, and a reference LRU, for the
offload tests.

A synthetic checkpoint stacks each expert projection as MLX does, [experts, rows, ...]
(affine quantized by mx.quantize, or bf16), in Qwen3.6 or Gemma4 names, over two files
behind odd-sized filler tensors so that experts start off 4 KiB boundaries, with one
layer's projections split across the files. An identifying checkpoint (affine only) fills
every packed weight word of an expert's projection with expert_code(layer, projection,
expert) instead, so a bank row read anywhere names the expert it holds.

ReferenceLru is the offload simulation's token-granular LRU per pool: per slot, the expert
it holds and the step it was last used. A call first marks its hits used at this step, then
gives each miss the slot of its layer's pool least recently used before this step: a free
slot first, then the lowest flat expert id.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import mlx.core as mx
import numpy as np

from ftmetal.layers.moe import ExpertSchemes, align_gate_up, moe_layout
from ftmetal.layers.quantization.config import AffineScheme, BankSpec
from ftmetal.models.loader import FuseRule, WeightRules
from ftmetal.moe.offload_cache import STALL_TIMEOUT_S, OffloadMoeCache
from ftmetal.moe.slot_pools import MoePoolGrouping, PrefillBufferPlacement, plan_slot_layout
from ftmetal.moe.source import ExpertSource

Naming = Literal["qwen", "gemma4"]
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
# What a cache built here tells a forward's prepare (which these caches never see) and the
# slot split (each layer gets at least this many): the lookups take any top-k.
TOP_K = 2
PREFILL_OVERLAP_MIN_TOKENS = 128


@dataclass(frozen=True, kw_only=True)
class CheckpointSpec:
    num_layers: int
    num_experts: int
    hidden_size: int
    intermediate_size: int
    naming: Naming = "qwen"
    scheme: AffineScheme | None = AffineScheme(4, 64)
    # Bits of the last layer's experts, if not the scheme's.
    last_layer_bits: int | None = None
    # Bits of every layer's up projection, if not the scheme's: gate and up load apart.
    up_bits: int | None = None
    is_identifying: bool = False

    @property
    def expert_schemes(self) -> ExpertSchemes:
        if self.up_bits is None or self.scheme is None:
            return ExpertSchemes.uniform(self.scheme)
        up = AffineScheme(self.up_bits, self.scheme.group_size)
        return ExpertSchemes(self.scheme, up, self.scheme)


def weight_rules(naming: Naming) -> WeightRules:
    """The text rules of the model family, for the expert banks."""
    gate_up = FuseRule("gate_up_proj", ("gate_proj", "up_proj"))
    if naming == "qwen":
        return WeightRules(
            renames=(("language_model.model.", "model."),),
            segments=(("switch_mlp", "experts"),),
            fuses=(gate_up,),
            expert_segment="experts",
        )
    else:
        return WeightRules(
            segments=(("experts.switch_glu", "experts"),),
            fuses=(gate_up,),
            expert_segment="experts",
        )


def layer_layouts(spec: CheckpointSpec) -> list[dict[str, BankSpec]]:
    """Each layer's bank arrays per expert as the offload cache holds them, gate and up in one
    scheme."""
    layouts = []
    for layer_id in range(spec.num_layers):
        schemes = spec.expert_schemes
        if layer_id == spec.num_layers - 1 and spec.last_layer_bits is not None:
            assert spec.scheme is not None, "a last layer in other bits is affine"
            schemes = ExpertSchemes.uniform(
                AffineScheme(spec.last_layer_bits, spec.scheme.group_size)
            )
        if schemes.gate is not None:
            schemes = align_gate_up(schemes)
        layouts.append(moe_layout(spec.hidden_size, spec.intermediate_size, schemes, mx.bfloat16))
    return layouts


def checkpoint_rules(spec: CheckpointSpec) -> WeightRules:
    """The rules the engine derives for the checkpoint: gate and up apart when they differ."""
    rules = weight_rules(spec.naming)
    schemes = spec.expert_schemes
    if schemes.gate == schemes.up:
        return rules
    experts = "model.layers.{}.mlp.experts" if spec.naming == "qwen" else "model.layers.{}.experts"
    unfused = frozenset(
        f"{experts.format(layer_id)}.gate_up_proj" for layer_id in range(spec.num_layers)
    )
    return replace(rules, unfused=unfused)


def expert_code(layer_id: int, projection: int, expert: int) -> int:
    """The word an identifying checkpoint fills the packed weight of an expert's projection
    (index into PROJECTIONS) with."""
    return (layer_id + 1) << 16 | projection << 12 | expert


def write_checkpoint(directory: Path, spec: CheckpointSpec) -> None:
    rng = np.random.default_rng(spec.num_layers * 101 + spec.hidden_size)
    files: list[dict[str, mx.array]] = [
        {"filler.a": mx.zeros((1001,), dtype=mx.uint8)},
        {"filler.b": mx.zeros((333,), dtype=mx.uint8), "filler.c": mx.zeros((77,), mx.uint16)},
    ]
    split_layer = spec.num_layers // 2
    for layer_id in range(spec.num_layers):
        if spec.naming == "qwen":
            module = f"language_model.model.layers.{layer_id}.mlp.switch_mlp"
        else:
            module = f"model.layers.{layer_id}.experts.switch_glu"
        for projection_index, projection in enumerate(PROJECTIONS):
            if projection == "down_proj":
                shape = (spec.num_experts, spec.hidden_size, spec.intermediate_size)
            else:
                shape = (spec.num_experts, spec.intermediate_size, spec.hidden_size)
            if layer_id < split_layer or (layer_id == split_layer and projection == "gate_proj"):
                file = files[0]
            else:
                file = files[1]
            if spec.is_identifying:
                assert spec.scheme is not None, "an identifying checkpoint is affine quantized"
                codes = [
                    expert_code(layer_id, projection_index, e) for e in range(spec.num_experts)
                ]
                packed_shape = (*shape[:2], shape[2] * spec.scheme.bits // 32)
                packed = np.broadcast_to(np.array(codes, np.uint32)[:, None, None], packed_shape)
                group_shape = (*shape[:2], shape[2] // spec.scheme.group_size)
                file[f"{module}.{projection}.weight"] = mx.array(np.ascontiguousarray(packed))
                file[f"{module}.{projection}.scales"] = mx.zeros(group_shape, dtype=mx.bfloat16)
                file[f"{module}.{projection}.biases"] = mx.zeros(group_shape, dtype=mx.bfloat16)
                continue
            weight = mx.array(0.02 * rng.standard_normal(shape, dtype=np.float32), mx.bfloat16)
            if spec.scheme is None:
                file[f"{module}.{projection}.weight"] = weight
            else:
                bits = spec.scheme.bits
                if layer_id == spec.num_layers - 1 and spec.last_layer_bits is not None:
                    bits = spec.last_layer_bits
                elif projection == "up_proj" and spec.up_bits is not None:
                    bits = spec.up_bits
                packed, scales, biases = mx.quantize(weight, spec.scheme.group_size, bits)
                file[f"{module}.{projection}.weight"] = packed
                file[f"{module}.{projection}.scales"] = scales
                file[f"{module}.{projection}.biases"] = biases
    directory.mkdir()
    for i, file in enumerate(files):
        mx.save_safetensors(str(directory / f"model-{i:05d}.safetensors"), file)


def build_cache(
    directory: Path,
    spec: CheckpointSpec,
    *,
    cache_size: int,
    prefill_overlap: bool,
    grouping: MoePoolGrouping = "per-layer",
    buffer_placement: PrefillBufferPlacement = "overlay",
    stall_timeout_s: float = STALL_TIMEOUT_S,
    run_coordinator: bool = True,
) -> tuple[OffloadMoeCache, ExpertSource]:
    """The cache over a new checkpoint of spec written to directory, and its source."""
    write_checkpoint(directory, spec)
    source = ExpertSource(str(directory), checkpoint_rules(spec))
    slot_layout = plan_slot_layout(
        cache_size,
        layer_layouts(spec),
        spec.num_experts,
        TOP_K,
        grouping=grouping,
        counts=None,
        prefill_overlap=prefill_overlap,
        buffer_placement=buffer_placement,
    )
    cache = OffloadMoeCache(
        slot_layout=slot_layout,
        source=source,
        prefill_overlap=prefill_overlap,
        prefill_overlap_min_tokens=PREFILL_OVERLAP_MIN_TOKENS,
        stall_timeout_s=stall_timeout_s,
        run_coordinator=run_coordinator,
    )
    return cache, source


class ReferenceLru:
    """SlotLru's victim choice, slot by slot: each layer's pool is the slot range
    pool_bounds[layer_pools[layer]:][:2]."""

    def __init__(
        self,
        num_experts: int,
        layer_pools: list[int],
        pool_bounds: list[int],
        fill: tuple[int, ...],
    ) -> None:
        self.num_experts = num_experts
        self.layer_pools = layer_pools
        self.pool_bounds = pool_bounds
        self.id_of_slot = [-1] * pool_bounds[-1]
        self.last_use = [0] * pool_bounds[-1]
        next_slot = pool_bounds[:-1]
        for flat_id in fill:
            pool = layer_pools[flat_id // num_experts]
            self.id_of_slot[next_slot[pool]] = flat_id
            next_slot[pool] += 1

    def pool_slots(self, layer_id: int) -> range:
        pool = self.layer_pools[layer_id]
        return range(self.pool_bounds[pool], self.pool_bounds[pool + 1])

    def resolve(
        self, layer_id: int, expert_ids: list[int], step: int
    ) -> tuple[list[int], list[tuple[int, int]]]:
        """The slot of each of the layer's distinct experts, and the (expert, slot) loads."""
        flat_ids = [layer_id * self.num_experts + expert for expert in expert_ids]
        slot_of = {flat_id: slot for slot, flat_id in enumerate(self.id_of_slot) if flat_id >= 0}
        for flat_id in flat_ids:
            if flat_id in slot_of:
                self.last_use[slot_of[flat_id]] = step
        loads: list[tuple[int, int]] = []
        for expert, flat_id in zip(expert_ids, flat_ids):
            if flat_id not in slot_of:
                victim = min(
                    (slot for slot in self.pool_slots(layer_id) if self.last_use[slot] < step),
                    key=lambda slot: (self.last_use[slot], self.id_of_slot[slot], slot),
                )
                slot_of.pop(self.id_of_slot[victim], None)
                self.id_of_slot[victim] = flat_id
                self.last_use[victim] = step
                slot_of[flat_id] = victim
                loads.append((expert, victim))
        return [slot_of[flat_id] for flat_id in flat_ids], loads
