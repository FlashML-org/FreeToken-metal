"""Loads the C++ extension against the pinned MLX and hands it the Metal sources."""

from importlib.metadata import version
from pathlib import Path

REQUIRED_MLX_VERSION = "0.32.1"
METAL_DIR = Path(__file__).resolve().parent / "metal"

if version("mlx") != REQUIRED_MLX_VERSION:
    raise ImportError(
        f"ftmetal.kernel is built against mlx=={REQUIRED_MLX_VERSION}, found {version('mlx')}"
    )

# mlx.core must be imported first: _C resolves libmlx symbols from the loaded module.
import mlx.core  # noqa: E402, F401  (import order is the loading contract)

from ftmetal.kernel import _C  # noqa: E402

_C.init_library((METAL_DIR / "rows.metal").read_text())
_C.init_gdn_step_library((METAL_DIR / "gdn_step.metal").read_text())
_C.init_fdecode_library((METAL_DIR / "fdecode.metal").read_text())
_C.init_fprefill_library((METAL_DIR / "fprefill.metal").read_text())
_C.init_sample_library((METAL_DIR / "sample.metal").read_text())

__all__ = ["_C"]
