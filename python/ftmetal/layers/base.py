"""BaseOP: the module base every layer and model derives from.

Port of FreeToken layers/base.py. Divergences: BaseOP is an mlx.nn.Module, so
parameters(), update() and tree_flatten replace state_dict / load_state_dict and
OPList is a plain list. Public array, list, dict and tuple attributes become module
state, so runtime data never goes on an op and such helper values stay underscored.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

import mlx.nn as nn


class BaseOP(nn.Module):
    # The one Any signature in the package: the donor's, so any op signature overrides it.
    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Run the op; the signature is the op's own."""


__all__ = ["BaseOP"]
