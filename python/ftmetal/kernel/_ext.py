"""Loads the C++ extension against the pinned MLX and hands it the Metal sources."""

import re
from importlib.metadata import version
from pathlib import Path

REQUIRED_MLX_VERSION = "0.32.3"
METAL_DIR = Path(__file__).resolve().parent / "metal"

if version("mlx") != REQUIRED_MLX_VERSION:
    raise ImportError(
        f"ftmetal.kernel is built against mlx=={REQUIRED_MLX_VERSION}, found {version('mlx')}"
    )

# mlx.core must be imported first: _C resolves libmlx symbols from the loaded module.
import mlx.core  # noqa: E402  (import order is the loading contract)

from ftmetal.kernel import _C  # noqa: E402

# The Metal kernel headers the mlx wheel ships; runtime Metal compilation has no include
# path, so a source that builds on them gets them inlined.
MLX_INCLUDE_DIR = Path(mlx.core.__file__).parent / "include"
MLX_INCLUDE_PATTERN = re.compile(r'^#include "(mlx/[^"]+)"$|^#pragma once$', re.MULTILINE)


def inline_mlx_headers(source: str, included: set[str]) -> str:
    """source with each #include "mlx/..." replaced by that header, the first time only."""

    def inline(match: re.Match[str]) -> str:
        header = match.group(1)
        if header is None or header in included:
            return ""
        else:
            included.add(header)
            return inline_mlx_headers((MLX_INCLUDE_DIR / header).read_text(), included)

    return MLX_INCLUDE_PATTERN.sub(inline, source)


_C.init_library((METAL_DIR / "rows.metal").read_text())
_C.init_gdn_step_library((METAL_DIR / "gdn_step.metal").read_text())
_C.init_fdecode_library((METAL_DIR / "fdecode.metal").read_text())
_C.init_fprefill_library((METAL_DIR / "fprefill.metal").read_text())
_C.init_gather_qmm_sorted_library(
    inline_mlx_headers((METAL_DIR / "gather_qmm_sorted.metal").read_text(), set())
)
_C.init_sample_library((METAL_DIR / "sample.metal").read_text())
_C.init_moe_lookup_library((METAL_DIR / "moe_lookup.metal").read_text())

__all__ = ["_C"]
