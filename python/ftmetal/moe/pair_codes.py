"""Pair codes: an affine expert projection's scales and biases as one code per group into its
MoE layer's sorted table of distinct (scale, bias) pairs, for the expert pack (moe/pack.py) and
the offload rows the pair-code kernels (kernel/moe/affine/affine_gather_qmv_pair_codes.py) read.

No FreeToken donor. A pair is the scale's bits above the bias's (pair_keys); codes are code_bits
wide, least significant bit first, row-major over each scales part's [rows, groups]
(encode_codes), the fewest bits that index every layer's table and make each scales part whole
64-byte lines, at most MAX_PAIR_CODE_BITS (code_width).
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np

from ftmetal.kernel._ext import _C
from ftmetal.kernel.moe.affine.affine_gather_qmv_pair_codes import (
    MAX_PAIR_CODE_BITS,
    pair_code_row_bytes,
)
from ftmetal.layers.quantization.config import BankSpec
from ftmetal.moe.source import ExpertPart

# The dtype of a scale's or bias's bits, and of a (scale, bias) key, per scales dtype.
PAIR_BITS: dict[mx.Dtype, tuple[type[np.unsignedinteger], type[np.unsignedinteger]]] = {
    mx.bfloat16: (np.uint16, np.uint32),
    mx.float16: (np.uint16, np.uint32),
    mx.float32: (np.uint32, np.uint64),
}


def coded_projections(roles: dict[str, list[ExpertPart]]) -> list[str] | None:
    """The projections of a MoE layer whose scales and biases parts pair codes replace, or None
    when a scales part has no biases or is not stored, or the layer's scales take two dtypes."""
    projections = [role.rpartition(".")[0] for role in roles if role.endswith(".scales")]
    pairs = [
        (scales, biases)
        for projection in projections
        if f"{projection}.biases" in roles
        for scales, biases in zip(
            roles[f"{projection}.scales"], roles[f"{projection}.biases"], strict=True
        )
    ]
    is_coded = (
        bool(projections)
        and all(f"{projection}.biases" in roles for projection in projections)
        and len({scales.dtype for scales, _ in pairs}) == 1
        and all(
            scales.dtype in PAIR_BITS
            and (biases.dtype, biases.shape) == (scales.dtype, scales.shape)
            and _C.FILL_FILE not in (scales.locations[0, 0], biases.locations[0, 0])
            for scales, biases in pairs
        )
    )
    return projections if is_coded else None


def pair_keys(scales: np.ndarray, biases: np.ndarray, dtype: mx.Dtype) -> np.ndarray:
    """Each group's (scale, bias) bits as one key, scale high: [experts, groups] of the parts'
    expert bytes uint8 [experts, bytes] (or of their values' bits, 1-D)."""
    value_dtype, key_dtype = PAIR_BITS[dtype]
    shift = 8 * np.dtype(value_dtype).itemsize
    return scales.view(value_dtype).astype(key_dtype) << shift | biases.view(value_dtype).astype(
        key_dtype
    )


def pair_values(keys: np.ndarray, dtype: mx.Dtype) -> np.ndarray:
    """The (scale, bias) bits [pairs, 2] of keys."""
    value_dtype, _ = PAIR_BITS[dtype]
    shift = 8 * np.dtype(value_dtype).itemsize
    return np.stack([keys >> shift, keys & ((1 << shift) - 1)], axis=1).astype(value_dtype)


def encode_codes(codes: np.ndarray, code_bits: int) -> np.ndarray:
    """[experts, codes] codes as code_bits-bit pair-code bytes [experts, codes * code_bits / 8],
    least significant bit first."""
    bits = codes[..., None] >> np.arange(code_bits, dtype=codes.dtype) & 1
    return np.packbits(bits.astype(np.uint8).reshape(codes.shape[0], -1), axis=1, bitorder="little")


def decode_codes(rows: np.ndarray, num_codes: int, code_bits: int) -> np.ndarray:
    """The num_codes code_bits-bit codes int64 [experts, num_codes] of pair-code bytes
    [experts, bytes]."""
    bits = np.unpackbits(rows, axis=1, count=num_codes * code_bits, bitorder="little")
    weights = 1 << np.arange(code_bits, dtype=np.int64)
    return bits.reshape(rows.shape[0], num_codes, code_bits).astype(np.int64) @ weights


def code_width(most_pairs: int, part_codes: set[int]) -> int | None:
    """The fewest bits that index most_pairs pairs and make each part's codes (part_codes, the
    codes per expert of each scales part) whole 64-byte lines, or None above MAX_PAIR_CODE_BITS."""
    fewest_bits = max(1, (most_pairs - 1).bit_length())
    return next(
        (
            code_bits
            for code_bits in range(fewest_bits, MAX_PAIR_CODE_BITS + 1)
            if all(
                8 * pair_code_row_bytes(num_codes, code_bits) == num_codes * code_bits
                for num_codes in part_codes
            )
        ),
        None,
    )


def coded_layout(layout: dict[str, BankSpec], code_bits: int) -> dict[str, BankSpec]:
    """An offload layout whose affine projections hold code_bits-bit pair codes: each one's
    weight, then its codes in place of its scales and biases."""
    coded = {}
    for role, spec in layout.items():
        projection, _, array = role.rpartition(".")
        if array == "scales" and f"{projection}.biases" in layout:
            num_codes = math.prod(spec.shape)
            coded[f"{projection}.codes"] = BankSpec(
                (pair_code_row_bytes(num_codes, code_bits),), mx.uint8, spec.quant
            )
        elif array != "biases" or f"{projection}.scales" not in layout:
            coded[role] = spec
    return coded


__all__ = [
    "PAIR_BITS",
    "code_width",
    "coded_layout",
    "coded_projections",
    "decode_codes",
    "encode_codes",
    "pair_keys",
    "pair_values",
]
