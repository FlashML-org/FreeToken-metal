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
import struct
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
        config = json.load(f)
    assert isinstance(config, dict), "config.json must hold an object"
    return config


def config_dict(config: dict[str, object], key: str) -> dict[str, object] | None:
    value = config.get(key)
    if value is None:
        return None
    assert isinstance(value, dict), f"config {key!r} must be an object"
    return value


def config_optional_int(config: dict[str, object], key: str) -> int | None:
    value = config.get(key)
    if value is None:
        return None
    assert isinstance(value, int) and not isinstance(value, bool), f"config {key!r} must be an int"
    return value


def config_int(config: dict[str, object], key: str, default: int | None = None) -> int:
    value = config_optional_int(config, key)
    if value is None:
        if default is None:
            raise KeyError(f"config {key!r} is missing")
        return default
    return value


def config_optional_float(config: dict[str, object], key: str) -> float | None:
    value = config.get(key)
    if value is None:
        return None
    assert isinstance(value, (int, float)) and not isinstance(value, bool), (
        f"config {key!r} must be a number"
    )
    return float(value)


def config_float(config: dict[str, object], key: str, default: float | None = None) -> float:
    value = config_optional_float(config, key)
    if value is None:
        if default is None:
            raise KeyError(f"config {key!r} is missing")
        return default
    return value


def config_bool(config: dict[str, object], key: str, default: bool) -> bool:
    value = config.get(key)
    if value is None:
        return default
    assert isinstance(value, bool), f"config {key!r} must be a bool"
    return value


def config_optional_str(config: dict[str, object], key: str) -> str | None:
    value = config.get(key)
    if value is None:
        return None
    assert isinstance(value, str), f"config {key!r} must be a string"
    return value


def config_str(config: dict[str, object], key: str, default: str) -> str:
    value = config_optional_str(config, key)
    return default if value is None else value


def config_str_list(config: dict[str, object], key: str) -> list[str] | None:
    value = config.get(key)
    if value is None:
        return None
    assert isinstance(value, list) and all(isinstance(item, str) for item in value), (
        f"config {key!r} must be a list of strings"
    )
    return [str(item) for item in value]


def weight_files(model_path: str) -> list[str]:
    files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"no safetensors files under {model_path}")
    return files


def checkpoint_tensor_names(model_path: str) -> list[str]:
    names: list[str] = []
    for file in weight_files(model_path):
        with open(file, "rb") as f:
            header_len = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(header_len))
        names.extend(name for name in header if name != "__metadata__")
    return names


def load_checkpoint(model_path: str) -> dict[str, mx.array]:
    """Every tensor of the checkpoint under its checkpoint name, lazily mapped."""
    state: dict[str, mx.array] = {}
    for file in weight_files(model_path):
        loaded = mx.load(file)
        assert isinstance(loaded, dict), f"{file}: not a safetensors dictionary"
        state.update(loaded)
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
    "checkpoint_tensor_names",
    "config_bool",
    "config_dict",
    "config_float",
    "config_int",
    "config_optional_float",
    "config_optional_int",
    "config_optional_str",
    "config_str",
    "config_str_list",
    "load_checkpoint",
    "load_state",
    "load_weights",
    "read_model_config",
    "weight_files",
]
