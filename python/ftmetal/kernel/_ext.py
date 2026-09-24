"""Loads the C++ extension against the pinned MLX and hands it the Metal sources."""

from pathlib import Path

import mlx.core as mx

REQUIRED_MLX_VERSION = "0.32.1"
METAL_DIR = Path(__file__).resolve().parent / "metal"

if mx.__version__ != REQUIRED_MLX_VERSION:
    raise ImportError(
        f"ftmetal.kernel is built against mlx=={REQUIRED_MLX_VERSION}, found {mx.__version__}"
    )

# mlx.core must be imported first: _C resolves libmlx symbols from the loaded module.
from ftmetal.kernel import _C  # noqa: E402  (import order is the loading contract)

_C.init_library((METAL_DIR / "rows.metal").read_text())
_C.init_gdn_step_library((METAL_DIR / "gdn_step.metal").read_text())

__all__ = ["_C"]
