"""ExpertSource: where each routed-expert bank row of a model lies in its checkpoint.

No FreeToken donor: FreeToken copies expert rows from host banks it loaded, while here
the slot cache reads them from the checkpoint files. MLX checkpoints stack each expert
projection as [experts, rows, ...], so expert e of a part is one contiguous byte range;
a fused bank row (gate_up = gate then up) is its parts' ranges back to back, also when the
checkpoint stores gate and up apart: a 4-bit part of an 8-bit bank row is widened as it is
read. Only the safetensors headers are read.
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

from ftmetal.layers.quantization.config import BankSpec
from ftmetal.models.loader import WeightRules, weight_files

SAFETENSORS_DTYPES: dict[str, mx.Dtype] = {
    "BOOL": mx.bool_,
    "U8": mx.uint8,
    "I8": mx.int8,
    "U16": mx.uint16,
    "I16": mx.int16,
    "U32": mx.uint32,
    "I32": mx.int32,
    "U64": mx.uint64,
    "I64": mx.int64,
    "F16": mx.float16,
    "BF16": mx.bfloat16,
    "F32": mx.float32,
}
LAYER_INDEX = re.compile(r"(?:^|\.)layers\.(\d+)\.")


@dataclass(frozen=True, kw_only=True)
class CheckpointTensor:
    """One stacked expert tensor: [experts, ...] of dtype at offset of file file_index."""

    file_index: int
    offset: int
    nbytes: int
    shape: tuple[int, ...]
    dtype: mx.Dtype


class ExpertSource:
    """The checkpoint parts of every routed-expert bank of a model, per MoE layer."""

    def __init__(self, model_path: str, rules: WeightRules) -> None:
        if rules.expert_segment is None:
            raise ValueError(f"{model_path}: the model's weight rules name no expert banks")
        # The cache fuses every layer's gate and up (align_gate_up): their parts in one row.
        rules = dataclasses.replace(rules, unfused=frozenset())
        self.expert_infix = f".{rules.expert_segment}."
        self.files = weight_files(model_path)
        # model key prefix -> bank role -> part index -> tensor, and each role's part count
        found: dict[str, dict[str, dict[int, CheckpointTensor]]] = {}
        num_parts: dict[str, int] = {}
        for file_index, file in enumerate(self.files):
            with open(file, "rb") as f:
                header_bytes = struct.unpack("<Q", f.read(8))[0]
                header = json.loads(f.read(header_bytes))
            assert isinstance(header, dict), f"{file}: the safetensors header is not an object"
            data_start = 8 + header_bytes
            for name, entry in header.items():
                key = rules.rename(name)
                if key is None or not rules.is_expert(key):
                    continue
                model_key, index, count = rules.fused_key(key)
                if not isinstance(entry, dict):
                    raise ValueError(f"{file}: {name} has no tensor entry")
                dtype_name, shape, offsets = (
                    entry.get("dtype"),
                    entry.get("shape"),
                    entry.get("data_offsets"),
                )
                if not (
                    isinstance(dtype_name, str)
                    and dtype_name in SAFETENSORS_DTYPES
                    and isinstance(shape, list)
                    and len(shape) >= 2
                    and all(isinstance(size, int) for size in shape)
                    and isinstance(offsets, list)
                    and len(offsets) == 2
                    and all(isinstance(offset, int) for offset in offsets)
                ):
                    raise ValueError(f"{file}: {name} is not a stacked expert tensor: {entry}")
                tensor = CheckpointTensor(
                    file_index=file_index,
                    offset=data_start + offsets[0],
                    nbytes=offsets[1] - offsets[0],
                    shape=tuple(shape),
                    dtype=SAFETENSORS_DTYPES[dtype_name],
                )
                if tensor.nbytes != math.prod(tensor.shape) * tensor.dtype.size:
                    raise ValueError(f"{file}: {name} holds {tensor.nbytes} bytes for {shape}")
                prefix, _, role = model_key.partition(self.expert_infix)
                found.setdefault(prefix, {}).setdefault(role, {})[index] = tensor
                num_parts[model_key] = count
        if not found:
            raise ValueError(f"{model_path}: no tensor maps to a {self.expert_infix} expert bank")
        self.layer_prefixes = sorted(found, key=layer_index)
        # Per MoE layer: bank role -> its parts in fuse order.
        self.parts: list[dict[str, list[CheckpointTensor]]] = []
        for prefix in self.layer_prefixes:
            roles: dict[str, list[CheckpointTensor]] = {}
            for role, by_index in found[prefix].items():
                if len(by_index) != num_parts[f"{prefix}{self.expert_infix}{role}"]:
                    raise ValueError(f"{prefix}{self.expert_infix}{role}: incomplete fused parts")
                roles[role] = [by_index[i] for i in range(len(by_index))]
            self.parts.append(roles)
        expert_counts = {
            tensor.shape[0] for roles in self.parts for parts in roles.values() for tensor in parts
        }
        if len(expert_counts) != 1:
            raise ValueError(f"{model_path}: expert tensors stack {sorted(expert_counts)} experts")
        self.num_experts = expert_counts.pop()

    @property
    def num_layers(self) -> int:
        return len(self.layer_prefixes)

    def segment_table(self, layouts: list[dict[str, BankSpec]]) -> np.ndarray:
        """int64 [num_layers, num_segments, 6]: one row per checkpoint part of each bank of
        each layer's layout, banks in layout order: (file index, file offset of expert 0,
        bytes per expert, bank index, byte offset in the bank row, 1 when the part's 4-bit
        codes widen to the bank's 8 bits, else 0)."""
        table: list[list[tuple[int, int, int, int, int, int]]] = []
        for prefix, roles, layout in zip(self.layer_prefixes, self.parts, layouts, strict=True):
            if set(roles) != set(layout):
                raise ValueError(
                    f"{prefix}{self.expert_infix}: banks {sorted(roles)} are not the layout's "
                    f"{sorted(layout)}"
                )
            segments: list[tuple[int, int, int, int, int, int]] = []
            for bank_index, (role, spec) in enumerate(layout.items()):
                parts = roles[role]
                row_offset = 0
                for part in parts:
                    # align_gate_up leaves 4-bit codes under an 8-bit packed weight row.
                    widens = spec.dtype == mx.uint32 and part.shape[2] != spec.shape[1]
                    if (
                        part.dtype != spec.dtype
                        or (widens and 2 * part.shape[2] != spec.shape[1])
                        or (not widens and part.shape[2:] != spec.shape[1:])
                    ):
                        raise ValueError(
                            f"{prefix}{self.expert_infix}{role}: checkpoint part "
                            f"{part.shape} {part.dtype} does not fit a {spec.shape} {spec.dtype} "
                            f"bank row"
                        )
                    expert_bytes = part.nbytes // self.num_experts
                    segments.append(
                        (
                            part.file_index,
                            part.offset,
                            expert_bytes,
                            bank_index,
                            row_offset,
                            int(widens),
                        )
                    )
                    row_offset += expert_bytes * (2 if widens else 1)
                if row_offset != spec.nbytes:
                    raise ValueError(
                        f"{prefix}{self.expert_infix}{role}: checkpoint parts "
                        f"{[(part.shape, part.dtype) for part in parts]} do not make a "
                        f"{spec.shape} {spec.dtype} bank row"
                    )
            table.append(segments)
        if len({len(segments) for segments in table}) != 1:
            raise ValueError("MoE layers split their expert banks into different checkpoint parts")
        return np.array(table, dtype=np.int64)


def layer_index(prefix: str) -> int:
    """The decoder layer a model key prefix belongs to."""
    match = LAYER_INDEX.search(prefix + ".")
    if match is None:
        raise ValueError(f"expert bank {prefix} is not under a layers.<index> module")
    return int(match.group(1))


__all__ = ["CheckpointTensor", "ExpertSource"]
