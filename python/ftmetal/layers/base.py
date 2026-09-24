"""BaseOP: the module base every layer and model derives from.

Port of FreeToken layers/base.py. Divergences: BaseOP is an mlx.nn.Module, so
parameters(), update() and tree_flatten replace state_dict / load_state_dict and
OPList is a plain list; there is no base forward signature, each op declares its
own. Public array, list, dict and tuple attributes become module state, so runtime
data never goes on an op and such helper values stay underscored.
"""

from __future__ import annotations

import mlx.nn as nn


class BaseOP(nn.Module):
    """A layer or model; forward has the op's own signature."""


__all__ = ["BaseOP"]
