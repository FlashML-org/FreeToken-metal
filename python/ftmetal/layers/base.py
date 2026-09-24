"""BaseOP: the module base every layer and model derives from.

Port of FreeToken layers/base.py. Divergences: BaseOP is an mlx.nn.Module, so
parameters(), update() and tree_flatten replace state_dict / load_state_dict and
OPList is a plain list. Public array, list, dict and tuple attributes become module
state, so runtime data never goes on an op and such helper values stay underscored.
"""

from __future__ import annotations

from abc import abstractmethod

import mlx.nn as nn


class BaseOP(nn.Module):
    @abstractmethod
    def forward(self, *args, **kwargs):  # noqa: ANN002, ANN003, ANN201  (each op fixes its own signature)
        """Run the op; the signature is the op's own."""


__all__ = ["BaseOP"]
