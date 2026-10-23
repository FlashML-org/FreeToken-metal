"""Checkpoint reading and the rename / fuse rules that map it onto a model.

Port of FreeToken models/loader.py and models/weight.py. Divergences: the source is
an MLX checkpoint (safetensors read lazily by mx.load), experts arrive pre-stacked, the
fused state is applied with nn.Module.update after a strict key and shape check, and a
fused module whose parts the checkpoint quantizes differently loads its parts apart.
"""

from __future__ import annotations

import glob
import json
import logging
import os
import struct
from dataclasses import dataclass, field
from typing import NamedTuple

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten

from ftmetal.layers.base import BaseOP
from ftmetal.layers.quantization.config import QuantConfig, QuantScheme

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
    # The model-key segment whose leaves are routed-expert banks ("experts" matches
    # model.layers.3.mlp.experts.gate_up_proj.weight); None: the model has none.
    expert_segment: str | None = None
    # Fused modules (model keys) whose parts load apart, each as <fused>.<part>.<suffix>.
    unfused: frozenset[str] = frozenset()
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

    def fused_key(self, name: str) -> tuple[str, int, int]:
        """The model key a renamed checkpoint key loads into, its part index and its rule's
        part count; a key no fuse rule names, or a part of an unfused module, loads whole as
        part 0 of 1."""
        module, _, suffix = name.rpartition(".")
        parent, _, leaf = module.rpartition(".")
        hit = self.part_index.get(leaf)
        if hit is None:
            return name, 0, 1
        rule, index = hit
        fused = f"{parent}.{rule.fused}" if parent else rule.fused
        if fused in self.unfused:
            return f"{fused}.{leaf}.{suffix}", 0, 1
        return f"{fused}.{suffix}", index, len(rule.parts)

    def is_expert(self, model_key: str) -> bool:
        """Whether a model key is a leaf of a routed-expert bank."""
        return self.expert_segment is not None and f".{self.expert_segment}." in model_key


def mixed_fuses(rules: WeightRules, quant: QuantConfig | None) -> frozenset[str]:
    """The fused modules (model keys) whose parts the checkpoint stores in different schemes,
    bf16 parts among quantized ones included: rules.unfused for that checkpoint."""
    if quant is None:
        return frozenset()
    parts: dict[str, tuple[FuseRule, dict[int, QuantScheme | None]]] = {}
    for module in quant.quantized_modules.union(quant.overrides):
        key = rules.rename(f"{module}.weight")
        if key is None:
            continue
        parent, _, leaf = key.removesuffix(".weight").rpartition(".")
        hit = rules.part_index.get(leaf)
        if hit is not None:
            rule, index = hit
            fused = f"{parent}.{rule.fused}" if parent else rule.fused
            parts.setdefault(fused, (rule, {}))[1][index] = quant.scheme_for(module)
    return frozenset(
        fused
        for fused, (rule, schemes) in parts.items()
        if len(schemes) < len(rule.parts) or len(set(schemes.values())) > 1
    )


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


def checkpoint_headers(model_path: str) -> dict[str, object]:
    """Every tensor entry of the checkpoint's safetensors headers, by checkpoint name."""
    entries: dict[str, object] = {}
    for file in weight_files(model_path):
        with open(file, "rb") as f:
            header_len = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(header_len))
        assert isinstance(header, dict), f"{file}: the safetensors header is not an object"
        entries.update((name, entry) for name, entry in header.items() if name != "__metadata__")
    return entries


def checkpoint_tensor_names(model_path: str) -> list[str]:
    return list(checkpoint_headers(model_path))


class CheckpointBytes(NamedTuple):
    """What the checkpoint's weights a model loads under rules take, from the headers alone."""

    non_expert: int
    experts: int


def checkpoint_bytes(model_path: str, rules: WeightRules) -> CheckpointBytes:
    non_expert = experts = 0
    for name, entry in checkpoint_headers(model_path).items():
        key = rules.rename(name)
        if key is None:
            continue
        offsets = entry.get("data_offsets") if isinstance(entry, dict) else None
        if not (
            isinstance(offsets, list)
            and len(offsets) == 2
            and all(isinstance(offset, int) for offset in offsets)
        ):
            raise ValueError(f"{model_path}: {name} has no data offsets: {entry}")
        if rules.is_expert(rules.fused_key(key)[0]):
            experts += offsets[1] - offsets[0]
        else:
            non_expert += offsets[1] - offsets[0]
    return CheckpointBytes(non_expert, experts)


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
        fused_key, index, num_parts = rules.fused_key(name)
        if num_parts == 1:
            out[fused_key] = tensor
            continue
        parts = pending.setdefault(fused_key, {})
        parts[index] = tensor
        if len(parts) == num_parts:
            ordered = [parts[i] for i in range(num_parts)]
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


def load_weights(
    model: BaseOP, model_path: str, rules: WeightRules, *, skip_experts: bool = False
) -> None:
    """skip_experts leaves the routed-expert banks unread: an offload cache reads them."""
    state = apply_rules(load_checkpoint(model_path), rules)
    if skip_experts:
        state = {key: tensor for key, tensor in state.items() if not rules.is_expert(key)}
    load_state(model, state)
    mx.eval(model.parameters())
    logger.info(f"loaded weights from {model_path}")


__all__ = [
    "CheckpointBytes",
    "FuseRule",
    "WeightRules",
    "apply_rules",
    "checkpoint_bytes",
    "checkpoint_headers",
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
    "mixed_fuses",
    "read_model_config",
    "weight_files",
]
