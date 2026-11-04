"""use_pack on offload_harness checkpoints.

A pack relocates every stored part into its records. A pack that does not hold this
checkpoint's experts raises StalePack and leaves the source as it was: a header that is not one
(a headerless file too), another format, a checkpoint file changed since, a header naming other
experts, a part missing or in another dtype, codes too wide, pair tables in some layers only or
in another dtype, a pack cut short in its records or its pair tables, and records whose weights
or (weights intact) codes differ from the checkpoint, in every expert of a layer so that the
random spot check finds them. No pack is no pack, not an error. Pair codes replace each affine
projection's scales and biases with codes in the fewest bits that index every layer's pairs and
make each part whole 64-byte lines (SPEC's parts of 192 codes take 8 or 16 bits); none for bf16
experts or a pack written without them. The offload rows they make are test_offload_cache's.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Literal

import mlx.core as mx
import numpy as np
import pytest

from ftmetal.moe.pack import PACK_ALIGNMENT, StalePack, use_pack, write_pack
from ftmetal.moe.source import ExpertSource
from offload_harness import CheckpointSpec, checkpoint_rules, read_state, write_checkpoint

SPEC = CheckpointSpec(num_layers=2, num_experts=4, hidden_size=192, intermediate_size=64)
CODED_SPEC = CheckpointSpec(num_layers=2, num_experts=4, hidden_size=256, intermediate_size=128)
BF16_SPEC = CheckpointSpec(
    num_layers=2, num_experts=4, hidden_size=192, intermediate_size=64, scheme=None
)
LINE_BITS = 512
Breakage = Literal[
    "header",
    "headerless",
    "format",
    "checkpoint",
    "experts",
    "part dtype",
    "missing part",
    "code bits",
    "pairs in one layer",
    "table dtype",
    "length",
    "table",
    "records",
    "codes",
]
# How each breakage is reported, of a pack of SPEC without pair codes or of CODED_SPEC with them.
STALE_PACKS: list[tuple[Breakage, CheckpointSpec, str]] = [
    ("header", SPEC, "no pack header"),
    ("headerless", SPEC, "no pack header"),
    ("format", SPEC, "format ftmetal-experts-9"),
    ("checkpoint", SPEC, "another checkpoint"),
    ("experts", SPEC, "other MoE layers or experts"),
    ("part dtype", SPEC, "is not in it"),
    ("missing part", SPEC, "is not in it"),
    ("code bits", CODED_SPEC, "no pack header"),
    ("pairs in one layer", CODED_SPEC, "pair codes in some layers only"),
    ("table dtype", CODED_SPEC, "pair table is not"),
    ("length", SPEC, "cut short"),
    ("table", CODED_SPEC, "cut short in"),
    ("records", SPEC, "differs from the checkpoint"),
    ("records", CODED_SPEC, "differs from the checkpoint"),
    ("codes", CODED_SPEC, "differs from the checkpoint"),
]
# Each spec packed with and without pair codes where it matters: the width SPEC's whole lines
# raise, bf16 experts (no scales), a pack without codes.
PAIR_CODE_CASES: list[tuple[CheckpointSpec, bool]] = [
    (CODED_SPEC, True),
    (CODED_SPEC, False),
    (SPEC, True),
    (BF16_SPEC, True),
]


def checkpoint_source(directory: Path, spec: CheckpointSpec = SPEC) -> ExpertSource:
    state, expert_bytes = read_state(directory, spec)
    return ExpertSource(str(directory), checkpoint_rules(spec), state, expert_bytes)


def packed_checkpoint(directory: Path, spec: CheckpointSpec = SPEC) -> Path:
    """A checkpoint of spec in directory and its pack, with pair codes for CODED_SPEC; the
    pack's path."""
    write_checkpoint(directory, spec)
    source = checkpoint_source(directory, spec)
    return write_pack(str(directory), source, with_pair_codes=spec is CODED_SPEC)


def part_locations(source: ExpertSource) -> list[np.ndarray]:
    return [
        part.locations.copy()
        for roles in source.parts
        for parts in roles.values()
        for part in parts
    ]


def test_pack_relocates_every_stored_part(tmp_path: Path) -> None:
    directory = tmp_path / "checkpoint"
    path = packed_checkpoint(directory)
    source = checkpoint_source(directory)
    checkpoint = part_locations(source)
    pack = use_pack(str(directory), source)
    assert pack is not None and pack.path == path and pack.pairs is None
    assert source.files[-1] == str(path)
    pack_bytes = np.fromfile(path, dtype=np.uint8)
    parts = [part for roles in source.parts for parts in roles.values() for part in parts]
    for part, locations in zip(parts, checkpoint, strict=True):
        assert (part.locations[:, 0] == len(source.files) - 1).all()
        for expert, (file_index, offset) in enumerate(locations):
            stored = np.fromfile(
                source.files[file_index],
                dtype=np.uint8,
                count=part.expert_bytes,
                offset=int(offset),
            )
            start = int(part.locations[expert, 1])
            assert (pack_bytes[start : start + part.expert_bytes] == stored).all()


def fewest_code_bits(directory: Path, spec: CheckpointSpec) -> int:
    """The pair codes' width a pack of the checkpoint takes, from its state: the fewest bits that
    index the most (scale, bias) pairs a layer takes and make each part whole 64-byte lines."""
    state = read_state(directory, spec)[0]
    most_pairs = 0
    part_codes = set()
    for layer_id in range(spec.num_layers):
        pairs = set()
        for projection in ("gate_proj", "up_proj", "down_proj"):
            name = f"language_model.model.layers.{layer_id}.mlp.switch_mlp.{projection}"
            scales = state[f"{name}.scales"]
            part_codes.add(scales[0].size)
            scale_bits = np.array(scales.view(mx.uint16)).reshape(-1).tolist()
            bias_bits = np.array(state[f"{name}.biases"].view(mx.uint16)).reshape(-1).tolist()
            pairs |= set(zip(scale_bits, bias_bits, strict=True))
        most_pairs = max(most_pairs, len(pairs))
    fewest = max(1, (most_pairs - 1).bit_length())
    return next(
        code_bits
        for code_bits in range(fewest, 17)
        if all(num_codes * code_bits % LINE_BITS == 0 for num_codes in part_codes)
    )


@pytest.mark.parametrize(
    ("spec", "with_pair_codes"),
    PAIR_CODE_CASES,
    ids=[
        f"H{spec.hidden_size}-{'affine' if spec.scheme else 'bf16'}-{'codes' if codes else 'plain'}"
        for spec, codes in PAIR_CODE_CASES
    ],
)
def test_pair_codes_replace_scales_and_biases(
    tmp_path: Path, spec: CheckpointSpec, with_pair_codes: bool
) -> None:
    directory = tmp_path / "checkpoint"
    write_checkpoint(directory, spec)
    write_pack(str(directory), checkpoint_source(directory, spec), with_pair_codes=with_pair_codes)
    source = checkpoint_source(directory, spec)
    pack = use_pack(str(directory), source)
    assert pack is not None
    is_coded = with_pair_codes and spec.scheme is not None
    if spec.scheme is None:
        arrays = ["weight"]
    elif is_coded:
        arrays = ["codes", "weight"]
    else:
        arrays = ["biases", "scales", "weight"]
    roles = sorted(
        f"{projection}.{array}" for projection in ("down_proj", "gate_up_proj") for array in arrays
    )
    assert all(sorted(layer) == roles for layer in source.parts)
    if is_coded:
        code_bits = pack.code_bits
        assert code_bits is not None and code_bits == fewest_code_bits(directory, spec)
        assert pack.pairs is not None and len(pack.pairs) == spec.num_layers
        assert all(pairs.shape == (1 << code_bits, 2) for pairs in pack.pairs)
    else:
        assert pack.code_bits is None and pack.pairs is None


def test_no_pack_is_none(tmp_path: Path) -> None:
    directory = tmp_path / "checkpoint"
    write_checkpoint(directory, SPEC)
    source = checkpoint_source(directory)
    files = list(source.files)
    assert use_pack(str(directory), source) is None
    assert source.files == files


def pack_header(pack_bytes: bytearray) -> dict[str, object]:
    header_length = struct.unpack("<Q", pack_bytes[:8])[0]
    header = json.loads(bytes(pack_bytes[8 : 8 + header_length]))
    assert isinstance(header, dict)
    return header


def data_start(pack_bytes: bytearray) -> int:
    """Where the records start: after the header, at the next PACK_ALIGNMENT boundary."""
    header_length = struct.unpack("<Q", pack_bytes[:8])[0]
    return -(-(8 + header_length) // PACK_ALIGNMENT) * PACK_ALIGNMENT


def rewrite_header(pack_bytes: bytearray, header: dict[str, object]) -> None:
    """Write header over the pack's, the records where they were (the JSON padded with spaces
    to the old length when shorter)."""
    old_length = struct.unpack("<Q", pack_bytes[:8])[0]
    start = data_start(pack_bytes)
    edited = json.dumps(header, separators=(",", ":")).encode().ljust(old_length)
    assert 8 + len(edited) <= start, "the header outgrows its padding"
    pack_bytes[:start] = struct.pack("<Q", len(edited)) + edited + bytes(start - 8 - len(edited))


def break_pack(breakage: Breakage, pack: Path, checkpoint: Path) -> None:
    pack_bytes = bytearray(pack.read_bytes())
    header = pack_header(pack_bytes)
    layers = header["layers"]
    assert isinstance(layers, list)
    if breakage == "header":
        pack_bytes[:] = struct.pack("<Q", 16) + b"{}"
    elif breakage == "headerless":
        # A file of raw records, as the prototype pack was: a length past the file.
        pack_bytes[:8] = b"\xff" * 8
    elif breakage == "format":
        rewrite_header(pack_bytes, {**header, "format": "ftmetal-experts-9"})
    elif breakage == "checkpoint":
        with next(checkpoint.glob("*.safetensors")).open("ab") as f:
            f.write(b"\0")
    elif breakage == "experts":
        num_experts = header["num_experts"]
        assert isinstance(num_experts, int)
        rewrite_header(pack_bytes, {**header, "num_experts": num_experts + 1})
    elif breakage == "part dtype":
        layers[0]["parts"][0]["dtype"] = "mlx.core.float32"
        rewrite_header(pack_bytes, header)
    elif breakage == "missing part":
        layers[0]["parts"].pop()
        rewrite_header(pack_bytes, header)
    elif breakage == "code bits":
        rewrite_header(pack_bytes, {**header, "code_bits": 17})
    elif breakage == "pairs in one layer":
        layers[-1]["pairs"] = None
        rewrite_header(pack_bytes, header)
    elif breakage == "table dtype":
        layers[0]["pairs"]["dtype"] = "mlx.core.float32"
        rewrite_header(pack_bytes, header)
    elif breakage == "length":
        pack_bytes.pop()
    elif breakage == "table":
        # Past the last table's padding, into its pairs.
        del pack_bytes[-(PACK_ALIGNMENT + 1) :]
    elif breakage == "records":
        # Every byte of the last layer's records, the second half of the records.
        start = data_start(pack_bytes)
        last = start + (len(pack_bytes) - start) // 2
        flipped = 255 - np.frombuffer(bytes(pack_bytes[last:]), dtype=np.uint8)
        pack_bytes[last:] = flipped.tobytes()
    else:
        # One code bit of each expert's first codes part in layer 0, its weights intact.
        layer = layers[0]
        codes_part = next(part for part in layer["parts"] if part["role"].endswith(".codes"))
        first = data_start(pack_bytes) + layer["offset"] + codes_part["offset"]
        for expert in range(codes_part["shape"][0]):
            pack_bytes[first + expert * layer["stride"]] ^= 1
    pack.write_bytes(bytes(pack_bytes))


@pytest.mark.parametrize(
    ("breakage", "spec", "reason"),
    STALE_PACKS,
    ids=[
        f"{breakage}-{'coded' if spec is CODED_SPEC else 'plain'}"
        for breakage, spec, _ in STALE_PACKS
    ],
)
def test_stale_pack_is_refused(
    tmp_path: Path, breakage: Breakage, spec: CheckpointSpec, reason: str
) -> None:
    directory = tmp_path / "checkpoint"
    path = packed_checkpoint(directory, spec)
    break_pack(breakage, path, directory)
    source = checkpoint_source(directory, spec)
    files = list(source.files)
    locations = part_locations(source)
    with pytest.raises(StalePack, match=reason):
        use_pack(str(directory), source)
    assert source.files == files
    after = part_locations(source)
    assert all((now == before).all() for now, before in zip(after, locations, strict=True))
