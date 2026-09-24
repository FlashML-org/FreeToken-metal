"""Checkpoint reading and the rename / fuse rules that map it onto a model.

Port of FreeToken models/loader.py and models/weight.py. Divergences: the source is
an MLX checkpoint (safetensors read lazily by mx.load), experts arrive pre-stacked, and
the fused state is applied with nn.Module.update after a strict key and shape check.
"""

from __future__ import annotations

import glob
import json
import logging
import os
from dataclasses import dataclass, field

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten

from ftmetal.layers.base import BaseOP

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FuseRule:
    """Leaves of one module fused into one weight: parts in order, concatenated on the output axis.

    The output axis is -2 for [out, in] weights and [experts, out, in] banks alike (and
    for their scales and biases); a 1-D bias concatenates on its only axis.
    """

    fused: str
    parts: tuple[str, ...]


@dataclass(frozen=True)
class WeightRules:
    # Checkpoint key prefix -> model key prefix, first match wins.
    renames: tuple[tuple[str, str], ...] = ()
    # Inner ".segment." -> ".segment." renames applied after the prefix rename.
    segments: tuple[tuple[str, str], ...] = ()
    # Checkpoint key prefixes to drop entirely.
    drops: tuple[str, ...] = ()
    fuses: tuple[FuseRule, ...] = ()
    # Model-key regex -> callable is deliberately absent: a shape fix is a fuse or a rename.
    part_index: dict[str, tuple[FuseRule, int]] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        index: dict[str, tuple[FuseRule, int]] = {}
        for rule in self.fuses:
            for i, part in enumerate(rule.parts):
                index[part] = (rule, i)
        object.__setattr__(self, "part_index", index)

    def rename(self, key: str) -> str | None:
        for prefix in self.drops:
            if key.startswith(prefix):
                return None
        for source, target in self.renames:
            if key.startswith(source):
                key = target + key[len(source) :]
                break
        for source, target in self.segments:
            key = key.replace(f".{source}.", f".{target}.")
        return key


def read_model_config(model_path: str) -> dict[str, object]:
    with open(os.path.join(model_path, "config.json"), encoding="utf-8") as f:
        return json.load(f)


def weight_files(model_path: str) -> list[str]:
    files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"no safetensors files under {model_path}")
    return files


def load_checkpoint(model_path: str) -> dict[str, mx.array]:
    """Every tensor of the checkpoint under its checkpoint name, lazily mapped."""
    state: dict[str, mx.array] = {}
    for file in weight_files(model_path):
        state.update(mx.load(file))
    return state


def apply_rules(state: dict[str, mx.array], rules: WeightRules) -> dict[str, mx.array]:
    """Rename every key and fuse the parts the rules name; the result is lazy."""
    out: dict[str, mx.array] = {}
    pending: dict[str, dict[int, mx.array]] = {}
    for key, tensor in state.items():
        name = rules.rename(key)
        if name is None:
            continue
        module, _, suffix = name.rpartition(".")
        parent, _, leaf = module.rpartition(".")
        hit = rules.part_index.get(leaf)
        if hit is None:
            out[name] = tensor
            continue
        rule, index = hit
        fused_key = f"{parent}.{rule.fused}.{suffix}" if parent else f"{rule.fused}.{suffix}"
        parts = pending.setdefault(fused_key, {})
        parts[index] = tensor
        if len(parts) == len(rule.parts):
            ordered = [parts[i] for i in range(len(rule.parts))]
            axis = -2 if ordered[0].ndim > 1 else 0
            out[fused_key] = mx.concatenate(ordered, axis=axis)
            pending.pop(fused_key)
    if pending:
        raise ValueError(f"incomplete fused weights: {sorted(pending)}")
    return out


def load_state(model: BaseOP, state: dict[str, mx.array]) -> None:
    """Strictly replace every parameter of model with state (keys, shapes, dtypes must match)."""
    expected = dict(tree_flatten(model.parameters()))
    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    if missing or unexpected:
        raise ValueError(
            f"checkpoint does not match the model: missing {missing[:8]}, "
            f"unexpected {unexpected[:8]}"
        )
    for key, placeholder in expected.items():
        tensor = state[key]
        if tensor.shape != placeholder.shape or tensor.dtype != placeholder.dtype:
            raise ValueError(
                f"{key}: checkpoint {tensor.shape} {tensor.dtype} vs model "
                f"{placeholder.shape} {placeholder.dtype}"
            )
    model.update(tree_unflatten(list(state.items())))


def load_weights(model: BaseOP, model_path: str, rules: WeightRules) -> None:
    state = apply_rules(load_checkpoint(model_path), rules)
    load_state(model, state)
    mx.eval(model.parameters())
    logger.info(f"loaded weights from {model_path}")


__all__ = [
    "FuseRule",
    "WeightRules",
    "apply_rules",
    "load_checkpoint",
    "load_state",
    "load_weights",
    "read_model_config",
    "weight_files",
]
