"""Loads the C++ extension against the pinned MLX, hands it the Metal libraries, and reads the
Metal sources of the metal_kernel wrappers."""

import re
from importlib.metadata import version
from pathlib import Path

REQUIRED_MLX_VERSION = "0.32.3"
KERNEL_DIR = Path(__file__).resolve().parent

if version("mlx") != REQUIRED_MLX_VERSION:
    raise ImportError(
        f"ftmetal.kernel is built against mlx=={REQUIRED_MLX_VERSION}, found {version('mlx')}"
    )

# mlx.core must be imported first: _C resolves libmlx symbols from the loaded module.
import mlx.core  # noqa: E402  (import order is the loading contract)

from ftmetal.kernel import _C  # noqa: E402

# Runtime Metal compilation has no include path, so a source gets the headers it includes
# inlined: mlx/... from the mlx wheel, any other path from this package (relative to kernel/).
MLX_INCLUDE_DIR = Path(mlx.core.__file__).parent / "include"
INCLUDE_PATTERN = re.compile(r'^#include "([^"]+)"$|^#pragma once$', re.MULTILINE)


def inline_headers(source: str, included: set[str]) -> str:
    """source with each #include "..." replaced by that header, the first time only."""

    def inline(match: re.Match[str]) -> str:
        header = match.group(1)
        if header is None or header in included:
            return ""
        else:
            included.add(header)
            root = MLX_INCLUDE_DIR if header.startswith("mlx/") else KERNEL_DIR
            return inline_headers((root / header).read_text(), included)

    return INCLUDE_PATTERN.sub(inline, source)


def metal_source(path: Path) -> str:
    """A Metal library or a metal_kernel body, its includes inlined."""
    return inline_headers(path.read_text(), set())


def metal_kernel_header(path: Path) -> str:
    """A metal_kernel header, its includes inlined less utils.h, which the preamble holds."""
    included: set[str] = set()
    inline_headers('#include "mlx/backend/metal/kernels/utils.h"\n', included)
    return inline_headers(path.read_text(), included)


_C.init_pool_rows_library(metal_source(KERNEL_DIR / "kvcache" / "pool_rows.metal"))
_C.init_gdn_decode_library(metal_source(KERNEL_DIR / "attention" / "gdn" / "gdn_decode.metal"))
_C.init_mha_decode_library(metal_source(KERNEL_DIR / "attention" / "mha" / "mha_decode.metal"))
_C.init_mha_prefill_library(metal_source(KERNEL_DIR / "attention" / "mha" / "mha_prefill.metal"))
_C.init_gather_sorted_libraries(
    metal_source(KERNEL_DIR / "moe" / "affine" / "affine_gather_qmm_sorted.metal"),
    metal_source(KERNEL_DIR / "moe" / "affine" / "affine_gather_qmm_sorted_pair_codes.metal"),
    metal_source(KERNEL_DIR / "moe" / "fp" / "fp_gather_qmm_sorted.metal"),
    metal_source(KERNEL_DIR / "moe" / "unquantized" / "gather_mm_sorted.metal"),
)
_C.init_sample_tokens_library(metal_source(KERNEL_DIR / "sample" / "sample_tokens.metal"))
_C.init_moe_lookup_library(metal_source(KERNEL_DIR / "moe" / "moe_lookup.metal"))
_C.init_affine_swiglu_pending_library(
    metal_source(KERNEL_DIR / "moe" / "affine" / "affine_swiglu_pending.metal")
)
_C.init_ple_lookup_library(metal_source(KERNEL_DIR / "embedding" / "ple_lookup.metal"))

__all__ = ["KERNEL_DIR", "_C", "metal_kernel_header", "metal_source"]
