"""Environment knobs, read as FTMETAL_<NAME>.

Port of FreeToken env.py (FREETOKEN_<NAME> there), reduced to a lookup.
"""

from __future__ import annotations

import os

ENV_PREFIX = "FTMETAL_"


def read_env(name: str) -> str | None:
    """FTMETAL_<name>, or None when it is unset or empty."""
    return os.environ.get(ENV_PREFIX + name) or None


__all__ = ["ENV_PREFIX", "read_env"]
