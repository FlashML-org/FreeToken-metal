"""ExpertSource: where each routed-expert bank row of a model lies in its checkpoint.

No FreeToken donor: FreeToken copies expert rows from host banks it loaded, while here
the slot cache reads them from the checkpoint files. The checkpoint in the MLX layout stacks
each expert projection as [experts, rows, ...], and expert e of such a tensor is one byte
range of a checkpoint file: its e-th slice as stored, or wherever the rewrite into the MLX
layout found it (ExpertBytes: one tensor per expert, a slice of a fused tensor), or bytes of
one value the checkpoint does not store. A fused bank row (gate_up = gate then up) is its
parts' ranges back to back, and a layout that keeps the parts apart takes each as a bank of
its own. Only the safetensors headers are read.
"""

from __future__ import annotations

import dataclasses
import json
import math
import re
import struct
from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from ftmetal.kernel._ext import _C
from ftmetal.layers.quantization.config import BankSpec
from ftmetal.models.loader import (
    ExpertBytes,
    ExpertFill,
    WeightRules,
    stacked_slices,
    weight_files,
)

LAYER_INDEX = re.compile(r"(?:^|\.)layers\.(\d+)\.")


@dataclass(frozen=True, kw_only=True)
class StoredTensor:
    """A tensor of the checkpoint files: nbytes bytes at offset of file file_index."""

    file_index: int
    offset: int
    nbytes: int


@dataclass(frozen=True, kw_only=True)
class ExpertPart:
    """One stacked expert tensor of the MLX layout, [experts, ...] of dtype, and where each
    expert's bytes start: locations int64 [experts, 2], (file index, file offset), or
    (FILL_FILE, the byte) for a tensor the checkpoint does not store."""

    shape: tuple[int, ...]
    dtype: mx.Dtype
    locations: np.ndarray

    @property
    def expert_bytes(self) -> int:
        return math.prod(self.shape[1:]) * self.dtype.size


class ExpertSource:
    """The checkpoint parts of every routed-expert bank of a model, per MoE layer."""

    def __init__(
        self,
        model_path: str,
        rules: WeightRules,
        state: dict[str, mx.array],
        expert_bytes: dict[str, ExpertBytes],
        resident_arrays: frozenset[str] = frozenset(),
    ) -> None:
        """state: the lazily mapped checkpoint in the MLX layout and expert_bytes, its rewrite's
        record (a Checkpoint's); resident_arrays: the array names (output_scale) the model's
        layers hold themselves, which no bank takes."""
        if rules.expert_segment is None:
            raise ValueError(f"{model_path}: the model's weight rules name no expert banks")
        # Every part under its fused role and index: segment_table maps layout banks onto them.
        rules = dataclasses.replace(rules, unfused=frozenset())
        self.part_index = rules.part_index
        self.expert_infix = f".{rules.expert_segment}."
        self.files = weight_files(model_path)
        stored: dict[str, StoredTensor] = {}
        for file_index, file in enumerate(self.files):
            with open(file, "rb") as f:
                header_bytes = struct.unpack("<Q", f.read(8))[0]
                header = json.loads(f.read(header_bytes))
            assert isinstance(header, dict), f"{file}: the safetensors header is not an object"
            data_start = 8 + header_bytes
            for name, entry in header.items():
                if name == "__metadata__":
                    continue
                offsets = entry.get("data_offsets") if isinstance(entry, dict) else None
                if not (
                    isinstance(offsets, list)
                    and len(offsets) == 2
                    and all(isinstance(offset, int) for offset in offsets)
                ):
                    raise ValueError(f"{file}: {name} has no tensor entry: {entry}")
                stored[name] = StoredTensor(
                    file_index=file_index,
                    offset=data_start + offsets[0],
                    nbytes=offsets[1] - offsets[0],
                )
        # model key prefix -> bank role -> part index -> part, and each role's part count
        found: dict[str, dict[str, dict[int, ExpertPart]]] = {}
        num_parts: dict[str, int] = {}
        for key, tensor in state.items():
            renamed = rules.rename(key)
            if (
                renamed is None
                or not rules.is_expert(renamed)
                or renamed.rpartition(".")[2] in resident_arrays
            ):
                continue
            model_key, index, count = rules.fused_key(renamed)
            if tensor.ndim < 2:
                raise ValueError(f"{key}: {tensor.shape} is not a stacked expert tensor")
            num_experts = tensor.shape[0]
            locations = np.empty((num_experts, 2), dtype=np.int64)
            slices = expert_bytes[key] if key in expert_bytes else stacked_slices(key, tensor)
            if isinstance(slices, ExpertFill):
                locations[:] = (_C.FILL_FILE, slices.value)
            elif slices.nbytes * num_experts != tensor.nbytes:
                raise ValueError(f"{key}: {slices.nbytes} bytes per expert make no {tensor.shape}")
            else:
                for e, (name, start) in enumerate(zip(slices.names, slices.offsets, strict=True)):
                    source = stored.get(name)
                    if source is None or start + slices.nbytes > source.nbytes:
                        raise ValueError(
                            f"{key}: expert {e} lies in {name}, which the checkpoint does not "
                            "store whole"
                        )
                    locations[e] = (source.file_index, source.offset + start)
            part = ExpertPart(shape=tuple(tensor.shape), dtype=tensor.dtype, locations=locations)
            prefix, _, role = model_key.partition(self.expert_infix)
            found.setdefault(prefix, {}).setdefault(role, {})[index] = part
            num_parts[model_key] = count
        if not found:
            raise ValueError(f"{model_path}: no tensor maps to a {self.expert_infix} expert bank")
        self.layer_prefixes = sorted(found, key=layer_index)
        # Per MoE layer: bank role -> its parts in fuse order.
        self.parts: list[dict[str, list[ExpertPart]]] = []
        for prefix in self.layer_prefixes:
            roles: dict[str, list[ExpertPart]] = {}
            for role, by_index in found[prefix].items():
                if len(by_index) != num_parts[f"{prefix}{self.expert_infix}{role}"]:
                    raise ValueError(f"{prefix}{self.expert_infix}{role}: incomplete fused parts")
                roles[role] = [by_index[i] for i in range(len(by_index))]
            self.parts.append(roles)
        expert_counts = {
            part.shape[0] for roles in self.parts for parts in roles.values() for part in parts
        }
        if len(expert_counts) != 1:
            raise ValueError(f"{model_path}: expert tensors stack {sorted(expert_counts)} experts")
        self.num_experts = expert_counts.pop()

    @property
    def num_layers(self) -> int:
        return len(self.layer_prefixes)

    def segment_table(self, layouts: list[dict[str, BankSpec]]) -> tuple[np.ndarray, np.ndarray]:
        """Each layout bank's checkpoint parts per MoE layer, a fused role's parts back to back
        or one part (<fused>.<part>.<array>) kept apart: segments int64 [num_layers,
        num_segments, 3], (bank index, byte offset in the bank row, bytes per expert), and
        locations int64 [num_layers, num_segments, num_experts, 2], each part's
        ExpertPart.locations."""
        segment_rows: list[list[tuple[int, int, int]]] = []
        location_rows: list[list[np.ndarray]] = []
        for prefix, roles, layout in zip(self.layer_prefixes, self.parts, layouts, strict=True):
            segments: list[tuple[int, int, int]] = []
            locations: list[np.ndarray] = []
            taken: list[tuple[str, int]] = []
            for bank_index, (role, spec) in enumerate(layout.items()):
                if role in roles:
                    parts = [(role, index) for index in range(len(roles[role]))]
                else:
                    module, _, array = role.rpartition(".")
                    fused, _, part_name = module.rpartition(".")
                    hit = self.part_index.get(part_name)
                    fused_role = f"{fused}.{array}"
                    if (
                        hit is None
                        or hit[0].fused != fused
                        or len(roles.get(fused_role, [])) != len(hit[0].parts)
                    ):
                        raise ValueError(f"{prefix}{self.expert_infix}{role}: no checkpoint part")
                    parts = [(fused_role, hit[1])]
                row_offset = 0
                for fused_role, index in parts:
                    part = roles[fused_role][index]
                    if part.dtype != spec.dtype or part.shape[2:] != spec.shape[1:]:
                        raise ValueError(
                            f"{prefix}{self.expert_infix}{role}: checkpoint part "
                            f"{part.shape} {part.dtype} does not fit a {spec.shape} "
                            f"{spec.dtype} bank row"
                        )
                    segments.append((bank_index, row_offset, part.expert_bytes))
                    locations.append(part.locations)
                    row_offset += part.expert_bytes
                    taken.append((fused_role, index))
                if row_offset != spec.nbytes:
                    raise ValueError(
                        f"{prefix}{self.expert_infix}{role}: checkpoint parts "
                        f"{[roles[fused_role][index].shape for fused_role, index in parts]} do "
                        f"not make a {spec.shape} {spec.dtype} bank row"
                    )
            every_part = [
                (role, index) for role, parts in roles.items() for index in range(len(parts))
            ]
            if sorted(taken) != sorted(every_part):
                raise ValueError(
                    f"{prefix}{self.expert_infix}: banks {sorted(layout)} do not take each "
                    f"checkpoint part of {sorted(roles)} once"
                )
            segment_rows.append(segments)
            location_rows.append(locations)
        if len({len(segments) for segments in segment_rows}) != 1:
            raise ValueError("MoE layers split their expert banks into different checkpoint parts")
        return np.array(segment_rows, dtype=np.int64), np.array(location_rows, dtype=np.int64)


def layer_index(prefix: str) -> int:
    """The decoder layer a model key prefix belongs to."""
    match = LAYER_INDEX.search(prefix + ".")
    if match is None:
        raise ValueError(f"expert bank {prefix} is not under a layers.<index> module")
    return int(match.group(1))


__all__ = ["ExpertPart", "ExpertSource", "StoredTensor"]
