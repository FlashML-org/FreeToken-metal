"""Environment knobs, read as FTMETAL_<NAME> with a FREETOKEN_<NAME> fallback.

Port of FreeToken env.py, reduced to a lookup: the fallback lets both engines share
one shell environment.
"""

from __future__ import annotations

import os

ENV_PREFIXES = ("FTMETAL_", "FREETOKEN_")


def read_env(name: str) -> str | None:
    """The first non-empty FTMETAL_<name> or FREETOKEN_<name>, else None."""
    for prefix in ENV_PREFIXES:
        value = os.environ.get(prefix + name)
        if value:
            return value
    return None


__all__ = ["ENV_PREFIXES", "read_env"]
