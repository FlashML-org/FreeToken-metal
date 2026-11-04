"""The expert pack: a model's routed experts laid out for the offload cache's reads, one
record per expert of each MoE layer, written by ftm pack into the model directory.

No FreeToken donor: FreeToken holds its experts in host banks. A record is the expert's
stored checkpoint parts back to back, so a miss is one read (a run of consecutive experts one
read) instead of one per part; records are PACK_ALIGNMENT apart, so their uncached reads go
straight to the device. The file holds an 8-byte little-endian header length, the JSON header,
padding to PACK_ALIGNMENT, each layer's records, then with pair codes each layer's pair table.
The offload cache reads a pack only when its header names this checkpoint (each safetensors
file's name, bytes, modification time and header digest) and every part the source reads, and
one random expert of each layer equals the checkpoint's bytes.

With pair codes (moe/pair_codes.py), when every MoE layer's experts are affine, a record holds
in place of each projection part's scales and biases its codes into the layer's table of
pairs, and the offload cache's rows hold the codes too (coded_layout). The whole model or none,
in one width: every layer shares the offload banks' roles.
"""

from __future__ import annotations

import fcntl
import hashlib
import logging
import math
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import mlx.core as mx
import numpy as np
from pydantic import BaseModel, Field, NonNegativeInt, PositiveInt, ValidationError

from ftmetal.kernel._ext import _C
from ftmetal.kernel.moe.affine.affine_gather_qmv_pair_codes import (
    MAX_PAIR_CODE_BITS,
    pair_code_row_bytes,
)
from ftmetal.models.loader import weight_files
from ftmetal.moe.pair_codes import (
    PAIR_BITS,
    code_width,
    coded_projections,
    decode_codes,
    encode_codes,
    pair_keys,
    pair_values,
)
from ftmetal.moe.source import ExpertPart, ExpertSource

logger = logging.getLogger(__name__)

PACK_FILE = "experts.packed"
PACK_FORMAT = "ftmetal-experts-1"
PACK_ALIGNMENT = 4096
# The records one write carries, at most (one record at least).
WRITE_CHUNK_BYTES = 256 << 20
# The longest header a pack is read with: a few KiB per MoE layer.
MAX_HEADER_BYTES = 64 << 20


class CheckpointFile(BaseModel):
    file: str
    nbytes: NonNegativeInt
    mtime_ns: int
    header_sha256: str


class PackPart(BaseModel):
    role: str
    index: NonNegativeInt
    dtype: str
    shape: list[PositiveInt]
    # Into the record.
    offset: NonNegativeInt


class PackPairs(BaseModel):
    # The table [count, 2] of (scale, bias) in dtype, from the end of the header's padding.
    offset: NonNegativeInt
    count: PositiveInt
    dtype: str


class PackLayer(BaseModel):
    prefix: str
    # Expert 0's record, from the end of the header's padding.
    offset: NonNegativeInt
    stride: PositiveInt
    parts: list[PackPart]
    # With pair codes.
    pairs: PackPairs | None


class PackHeader(BaseModel):
    format: str
    checkpoint: list[CheckpointFile]
    num_experts: PositiveInt
    layers: list[PackLayer]
    # The pair codes' width, None without.
    code_bits: Annotated[int, Field(ge=1, le=MAX_PAIR_CODE_BITS)] | None


class StalePack(ValueError):
    """The model's pack does not hold this checkpoint's experts."""


@dataclass(frozen=True, kw_only=True)
class ExpertPack:
    """A pack the source reads: its path and, with pair codes, their width and each layer's
    pair table [2^code_bits, 2] (zero past the pairs, so a code past them reads zeros)."""

    path: Path
    code_bits: int | None
    pairs: list[mx.array] | None


def checkpoint_fingerprint(model_path: str) -> list[CheckpointFile]:
    """Each safetensors file of the checkpoint: its name, bytes, modification time and header
    digest."""
    files = []
    for file in weight_files(model_path):
        with open(file, "rb") as f:
            header_length = struct.unpack("<Q", f.read(8))[0]
            digest = hashlib.sha256(f.read(header_length)).hexdigest()
        status = os.stat(file)
        files.append(
            CheckpointFile(
                file=os.path.basename(file),
                nbytes=status.st_size,
                mtime_ns=status.st_mtime_ns,
                header_sha256=digest,
            )
        )
    return files


def stored_parts(roles: dict[str, list[ExpertPart]]) -> list[tuple[str, int, ExpertPart]]:
    """The (role, index, part) of each part of a layer the checkpoint stores, in record order."""
    return [
        (role, index, part)
        for role, parts in roles.items()
        for index, part in enumerate(parts)
        if part.locations[0, 0] != _C.FILL_FILE
    ]


def read_part(descriptors: list[int], part: ExpertPart, first: int, last: int) -> np.ndarray:
    """Experts [first, last) of a stored part, uint8 [experts, expert bytes]: a run of experts
    stored back to back in one file is one read."""
    expert_bytes = part.expert_bytes
    experts = np.empty((last - first, expert_bytes), dtype=np.uint8)
    expert = first
    while expert < last:
        file_index, offset = (int(location) for location in part.locations[expert])
        run = 1
        while (
            expert + run < last
            and part.locations[expert + run, 0] == file_index
            and part.locations[expert + run, 1] == offset + run * expert_bytes
        ):
            run += 1
        rows = experts[expert - first : expert - first + run]
        if os.preadv(descriptors[file_index], [rows], offset) != rows.nbytes:
            raise OSError(f"file {file_index} ends inside experts {expert}-{expert + run - 1}")
        expert += run
    return experts


def align(nbytes: int) -> int:
    return -(-nbytes // PACK_ALIGNMENT) * PACK_ALIGNMENT


def pair_tables(
    source: ExpertSource, descriptors: list[int]
) -> tuple[list[np.ndarray], int] | None:
    """Each layer's sorted (scale, bias) keys and the codes' width when the whole model takes pair
    codes, else None (logged)."""
    layer_projections = [coded_projections(roles) for roles in source.parts]
    coded = [projections for projections in layer_projections if projections is not None]
    if len(coded) != source.num_layers:
        logger.info("no pair codes: not every MoE layer's experts are affine")
        return None
    else:
        tables = []
        part_codes = set()
        for roles, projections in zip(source.parts, coded, strict=True):
            keys = np.empty(0, dtype=PAIR_BITS[roles[f"{projections[0]}.scales"][0].dtype][1])
            for projection in projections:
                for scales, biases in zip(
                    roles[f"{projection}.scales"], roles[f"{projection}.biases"], strict=True
                ):
                    part_codes.add(math.prod(scales.shape[1:]))
                    part_keys = pair_keys(
                        read_part(descriptors, scales, 0, source.num_experts),
                        read_part(descriptors, biases, 0, source.num_experts),
                        scales.dtype,
                    )
                    keys = np.union1d(keys, part_keys)
            tables.append(keys)
        most_pairs = max(keys.size for keys in tables)
        code_bits = code_width(most_pairs, part_codes)
        if code_bits is None:
            logger.info(
                f"no pair codes: {most_pairs} (scale, bias) pairs in a layer, projection parts of "
                f"{sorted(part_codes)} groups"
            )
            return None
        else:
            return tables, code_bits


def write_pack(model_path: str, source: ExpertSource, *, with_pair_codes: bool = True) -> Path:
    """Write the source's stored expert parts to model_path's pack, with pair codes when
    with_pair_codes and the model takes them, first as a .partial file renamed once complete
    (and removed when the write fails); its path."""
    path = Path(model_path) / PACK_FILE
    partial = path.with_name(f"{PACK_FILE}.partial")
    descriptors = [os.open(file, os.O_RDONLY) for file in source.files]
    try:
        # The pack is read uncached, and it and the checkpoint would evict everything else.
        for descriptor in descriptors:
            fcntl.fcntl(descriptor, fcntl.F_NOCACHE, 1)
        coding = pair_tables(source, descriptors) if with_pair_codes else None
        tables, code_bits = (None, None) if coding is None else coding
        layers = []
        offset = 0
        for prefix, roles in zip(source.layer_prefixes, source.parts, strict=True):
            coded = set(coded_projections(roles) or ()) if code_bits is not None else set()
            parts = []
            record_bytes = 0
            for role, index, part in stored_parts(roles):
                projection, _, array = role.rpartition(".")
                if projection in coded and array == "biases":
                    continue
                elif projection in coded and array == "scales" and code_bits is not None:
                    nbytes = pair_code_row_bytes(math.prod(part.shape[1:]), code_bits)
                    pack_part = PackPart(
                        role=f"{projection}.codes",
                        index=index,
                        dtype=str(mx.uint8),
                        shape=[source.num_experts, nbytes],
                        offset=record_bytes,
                    )
                else:
                    nbytes = part.expert_bytes
                    pack_part = PackPart(
                        role=role,
                        index=index,
                        dtype=str(part.dtype),
                        shape=list(part.shape),
                        offset=record_bytes,
                    )
                parts.append(pack_part)
                record_bytes += nbytes
            stride = align(record_bytes)
            layers.append(
                PackLayer(prefix=prefix, offset=offset, stride=stride, parts=parts, pairs=None)
            )
            offset += source.num_experts * stride
        dtypes = []
        if tables is not None:
            for layer, keys, roles in zip(layers, tables, source.parts, strict=True):
                projections = coded_projections(roles)
                assert projections is not None, "pair_tables took every layer"
                dtype = roles[f"{projections[0]}.scales"][0].dtype
                layer.pairs = PackPairs(offset=offset, count=keys.size, dtype=str(dtype))
                offset += align(keys.size * 2 * np.dtype(PAIR_BITS[dtype][0]).itemsize)
                dtypes.append(dtype)
        header = PackHeader(
            format=PACK_FORMAT,
            checkpoint=checkpoint_fingerprint(model_path),
            num_experts=source.num_experts,
            layers=layers,
            code_bits=code_bits,
        ).model_dump_json()
        header_bytes = header.encode()
        with open(partial, "wb") as f:
            fcntl.fcntl(f.fileno(), fcntl.F_NOCACHE, 1)
            f.write(struct.pack("<Q", len(header_bytes)) + header_bytes)
            f.write(bytes(align(8 + len(header_bytes)) - 8 - len(header_bytes)))
            for layer_id, (layer, roles) in enumerate(zip(layers, source.parts, strict=True)):
                experts_per_write = max(1, WRITE_CHUNK_BYTES // layer.stride)
                for first in range(0, source.num_experts, experts_per_write):
                    last = min(first + experts_per_write, source.num_experts)
                    records = np.zeros((last - first, layer.stride), dtype=np.uint8)
                    for pack_part in layer.parts:
                        projection, _, array = pack_part.role.rpartition(".")
                        if array == "codes" and tables is not None and code_bits is not None:
                            scales = roles[f"{projection}.scales"][pack_part.index]
                            biases = roles[f"{projection}.biases"][pack_part.index]
                            keys = pair_keys(
                                read_part(descriptors, scales, first, last),
                                read_part(descriptors, biases, first, last),
                                scales.dtype,
                            )
                            codes = np.searchsorted(tables[layer_id], keys).astype(np.uint16)
                            part_rows = encode_codes(codes, code_bits)
                        else:
                            part = roles[pack_part.role][pack_part.index]
                            part_rows = read_part(descriptors, part, first, last)
                        end = pack_part.offset + part_rows.shape[1]
                        records[:, pack_part.offset : end] = part_rows
                    f.write(records.data)
                logger.info(f"packed MoE layer {layer_id + 1}/{len(layers)}")
            for keys, dtype in zip(tables or [], dtypes, strict=True):
                table_bytes = pair_values(keys, dtype).tobytes()
                f.write(table_bytes + bytes(align(len(table_bytes)) - len(table_bytes)))
        os.replace(partial, path)
    except BaseException:
        # A failed write of tens of GiB leaves nothing behind; the error goes on up.
        partial.unlink(missing_ok=True)
        raise
    finally:
        for descriptor in descriptors:
            os.close(descriptor)
    return path


def use_pack(model_path: str, source: ExpertSource) -> ExpertPack | None:
    """Point the source's stored parts at the records of model_path's pack (with pair codes, an
    affine projection's codes in place of its scales and biases); the pack, or None without one.
    Raises StalePack, the source untouched, when the pack does not hold them."""
    path = Path(model_path) / PACK_FILE
    if path.is_file():
        return open_pack(path, model_path, source)
    else:
        return None


def open_pack(path: Path, model_path: str, source: ExpertSource) -> ExpertPack:
    """use_pack on a pack file: validate it against the checkpoint, then relocate the source."""
    pack_bytes = path.stat().st_size
    with open(path, "rb") as f:
        length_bytes = f.read(8)
        header_length = struct.unpack("<Q", length_bytes)[0] if len(length_bytes) == 8 else 0
        if not 0 < header_length <= min(pack_bytes - 8, MAX_HEADER_BYTES):
            raise StalePack(f"{path}: no pack header")
        header_json = f.read(header_length)
    try:
        header = PackHeader.model_validate_json(header_json)
    except ValidationError as error:
        raise StalePack(f"{path}: no pack header ({error.error_count()} errors)") from error
    has_pairs = {layer.pairs is not None for layer in header.layers}
    data_start = align(8 + header_length)
    records_end = data_start + max(
        (layer.offset + header.num_experts * layer.stride for layer in header.layers), default=0
    )
    if header.format != PACK_FORMAT:
        raise StalePack(f"{path}: format {header.format}, not {PACK_FORMAT}")
    elif header.checkpoint != checkpoint_fingerprint(model_path):
        raise StalePack(f"{path}: written from another checkpoint")
    elif (
        header.num_experts != source.num_experts
        or [layer.prefix for layer in header.layers] != source.layer_prefixes
    ):
        raise StalePack(f"{path}: holds other MoE layers or experts")
    elif len(has_pairs) != 1 or (header.code_bits is not None) != (True in has_pairs):
        raise StalePack(f"{path}: pair codes in some layers only")
    elif records_end > pack_bytes:
        raise StalePack(f"{path}: cut short at {pack_bytes} bytes, not {records_end}")
    code_bits = header.code_bits
    pack_index = len(source.files)
    experts = np.arange(source.num_experts, dtype=np.int64)
    parts: list[dict[str, list[ExpertPart]]] = []
    # Per layer: each part read from the pack and the checkpoint parts it stands for.
    checks: list[list[tuple[ExpertPart, list[ExpertPart]]]] = []
    tables: list[tuple[np.ndarray, mx.Dtype]] = []
    for layer, roles in zip(header.layers, source.parts, strict=True):
        pack_parts = {(part.role, part.index): part for part in layer.parts}
        projections = coded_projections(roles) if layer.pairs is not None else []
        if projections is None:
            raise StalePack(f"{path}: pair codes for {layer.prefix}, whose experts take none")
        coded = set(projections)
        layer_parts: dict[str, list[ExpertPart]] = {}
        layer_checks = []
        for role, role_parts in roles.items():
            projection, _, array = role.rpartition(".")
            if projection in coded and array == "biases":
                continue
            for index, part in enumerate(role_parts):
                if part.locations[0, 0] == _C.FILL_FILE:
                    layer_parts.setdefault(role, []).append(part)
                    continue
                elif projection in coded and array == "scales" and code_bits is not None:
                    pack_role = f"{projection}.codes"
                    nbytes = pair_code_row_bytes(math.prod(part.shape[1:]), code_bits)
                    shape = (source.num_experts, nbytes)
                    dtype = mx.uint8
                    checkpoint_parts = [part, roles[f"{projection}.biases"][index]]
                else:
                    pack_role, shape, dtype, checkpoint_parts = role, part.shape, part.dtype, [part]
                    nbytes = part.expert_bytes
                pack_part = pack_parts.get((pack_role, index))
                if (
                    pack_part is None
                    or pack_part.dtype != str(dtype)
                    or tuple(pack_part.shape) != shape
                ):
                    raise StalePack(f"{path}: {layer.prefix} {pack_role}[{index}] is not in it")
                elif pack_part.offset + nbytes > layer.stride:
                    raise StalePack(
                        f"{path}: {layer.prefix} {pack_role}[{index}] leaves its record"
                    )
                offsets = data_start + layer.offset + experts * layer.stride + pack_part.offset
                packed = ExpertPart(
                    shape=shape,
                    dtype=dtype,
                    locations=np.stack([np.full_like(offsets, pack_index), offsets], axis=1),
                )
                layer_parts.setdefault(pack_role, []).append(packed)
                layer_checks.append((packed, checkpoint_parts))
        if layer.pairs is not None and code_bits is not None:
            dtype = roles[f"{projections[0]}.scales"][0].dtype
            value_dtype = PAIR_BITS[dtype][0]
            table_nbytes = layer.pairs.count * 2 * np.dtype(value_dtype).itemsize
            if layer.pairs.dtype != str(dtype) or layer.pairs.count > 1 << code_bits:
                raise StalePack(f"{path}: {layer.prefix}'s pair table is not {dtype} pairs")
            elif data_start + layer.pairs.offset < records_end:
                raise StalePack(f"{path}: {layer.prefix}'s pair table overlaps the records")
            with open(path, "rb") as f:
                f.seek(data_start + layer.pairs.offset)
                table_bytes = f.read(table_nbytes)
            if len(table_bytes) != table_nbytes:
                raise StalePack(f"{path}: cut short in {layer.prefix}'s pair table")
            values = np.frombuffer(table_bytes, dtype=value_dtype).reshape(-1, 2)
            keys = pair_keys(
                np.ascontiguousarray(values[:, 0]).view(np.uint8),
                np.ascontiguousarray(values[:, 1]).view(np.uint8),
                dtype,
            )
            tables.append((keys, dtype))
        parts.append(layer_parts)
        checks.append(layer_checks)
    generator = np.random.default_rng()
    descriptors = [os.open(file, os.O_RDONLY) for file in [*source.files, str(path)]]
    try:
        for layer_id, (layer, layer_checks) in enumerate(zip(header.layers, checks, strict=True)):
            expert = int(generator.integers(source.num_experts))
            for packed, checkpoint_parts in layer_checks:
                stored = read_part(descriptors, packed, expert, expert + 1)
                if len(checkpoint_parts) == 1:
                    is_equal = (
                        read_part(descriptors, checkpoint_parts[0], expert, expert + 1) == stored
                    ).all()
                else:
                    assert code_bits is not None, "codes stand for scales and biases"
                    scales, biases = checkpoint_parts
                    codes = decode_codes(stored, math.prod(scales.shape[1:]), code_bits)
                    keys = pair_keys(
                        read_part(descriptors, scales, expert, expert + 1),
                        read_part(descriptors, biases, expert, expert + 1),
                        scales.dtype,
                    )
                    table = tables[layer_id][0]
                    is_equal = (codes < table.size).all() and (
                        table[np.minimum(codes, table.size - 1)] == keys
                    ).all()
                if not is_equal:
                    raise StalePack(
                        f"{path}: expert {expert} of {layer.prefix} differs from the checkpoint"
                    )
    finally:
        for descriptor in descriptors:
            os.close(descriptor)
    source.relocate(str(path), parts)
    if code_bits is not None:
        pairs = []
        for keys, dtype in tables:
            padded = np.zeros((1 << code_bits, 2), dtype=PAIR_BITS[dtype][0])
            padded[: keys.size] = pair_values(keys, dtype)
            pairs.append(mx.array(padded).view(dtype))
    else:
        pairs = None
    return ExpertPack(path=path, code_bits=code_bits, pairs=pairs)


__all__ = ["PACK_FILE", "ExpertPack", "StalePack", "use_pack", "write_pack"]
