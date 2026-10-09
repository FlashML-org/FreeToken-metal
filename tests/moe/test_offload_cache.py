"""OffloadMoeCache and ExpertSource on synthetic checkpoints, and on the real one.

A synthetic checkpoint stacks each expert projection as MLX does, [experts, rows, ...]
(affine quantized by mx.quantize, or bf16), in Qwen3.6 or Gemma4 names, over two files
behind odd-sized filler tensors so that experts start off 4 KiB boundaries, with one
layer's projections split across the files. Every bank row the cache holds must equal,
byte for byte, the expert's row of apply_rules(load_checkpoint(...)): after the initial
fill, after each ensure_experts call of a hand-written script (each returned slot, and
every slot slot_of maps), and at every slot of a prefetched prefill layer's slot map,
whose resident experts keep their slots. Large experts make one run of reads longer than
a staging buffer. Errors: a layer in another format, the window protocol, a file cut
short under the cache. The real-checkpoint case needs FTMETAL_TEST_MODEL_PATH.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from ftmetal.core import host_ints
from ftmetal.engine.config import EngineConfig
from ftmetal.layers.moe import MoELayer, moe_layout
from ftmetal.layers.quantization.config import AffineScheme
from ftmetal.models.loader import FuseRule, WeightRules, apply_rules, load_checkpoint
from ftmetal.models.qwen3_5_moe.model import fused_scheme_of
from ftmetal.models.qwen3_5_moe.weight import GATE_UP_PARTS
from ftmetal.models.register import model_weight_rules
from ftmetal.moe.offload_cache import OffloadMoeCache
from ftmetal.moe.source import ExpertSource

MODEL_PATH = os.environ.get("FTMETAL_TEST_MODEL_PATH")
Naming = Literal["qwen", "gemma4"]
# One decode step: the (layer, distinct expert ids) calls of ensure_experts, in order.
Step = tuple[tuple[int, tuple[int, ...]], ...]
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


@dataclass(frozen=True)
class Prefill:
    """An overlapped prefill of these layers, in order: two prefetched ahead."""

    layers: tuple[int, ...]


@dataclass(frozen=True)
class Case:
    num_layers: int
    num_experts: int
    hidden_size: int
    intermediate_size: int
    cache_size: int
    prefill_overlap: bool
    script: tuple[Step | Prefill, ...] = ()
    naming: Naming = "qwen"
    scheme: AffineScheme | None = AffineScheme(4, 64)
    # Bits of the last layer's experts, if not the scheme's.
    last_layer_bits: int | None = None

    def __str__(self) -> str:
        scheme = "bf16" if self.scheme is None else f"q{self.scheme.bits}g{self.scheme.group_size}"
        return (
            f"{self.naming}-L{self.num_layers}-E{self.num_experts}-H{self.hidden_size}"
            f"-I{self.intermediate_size}-S{self.cache_size}"
            f"{'-overlap' if self.prefill_overlap else ''}-{scheme}"
        )


CASES = [
    Case(
        3,
        8,
        192,
        64,
        12,
        False,
        script=(
            ((0, (0, 5, 6)), (1, (7,)), (2, (1, 2))),
            ((0, (5, 7, 3)), (1, (0, 1, 2, 3)), (2, (6,))),
            ((1, (4, 5)), (2, (0, 7, 6))),
            ((0, (1, 2, 4)), (2, (3, 4, 5))),
        ),
    ),
    Case(
        4,
        4,
        128,
        128,
        14,
        True,
        script=(
            ((0, (1, 2)), (1, (3,)), (3, (0, 3))),
            Prefill((0, 1, 2, 3)),
            ((0, (0, 3)), (1, (0, 1, 2)), (2, (2,)), (3, (1,))),
            Prefill((1, 2)),
            ((2, (0, 1, 2, 3)),),
        ),
        naming="gemma4",
    ),
    Case(2, 4, 192, 64, 8, True, script=(Prefill((0, 1)), ((1, (2, 3)),)), scheme=None),
    # 2 MiB per expert and projection: a run of four experts outgrows the staging buffer.
    Case(1, 4, 2048, 2048, 4, False, script=(((0, (0, 1, 2, 3)),),)),
]


def weight_rules(naming: Naming) -> WeightRules:
    """The text rules of the model family, for the expert banks."""
    gate_up = FuseRule("gate_up_proj", ("gate_proj", "up_proj"))
    if naming == "qwen":
        return WeightRules(
            renames=(("language_model.model.", "model."),),
            segments=(("switch_mlp", "experts"),),
            fuses=(gate_up,),
            expert_segment="experts",
        )
    else:
        return WeightRules(
            segments=(("experts.switch_glu", "experts"),),
            fuses=(gate_up,),
            expert_segment="experts",
        )


def write_checkpoint(directory: Path, case: Case) -> None:
    rng = np.random.default_rng(case.num_layers * 101 + case.hidden_size)
    files: list[dict[str, mx.array]] = [
        {"filler.a": mx.zeros((1001,), dtype=mx.uint8)},
        {"filler.b": mx.zeros((333,), dtype=mx.uint8), "filler.c": mx.zeros((77,), mx.uint16)},
    ]
    split_layer = case.num_layers // 2
    for layer_id in range(case.num_layers):
        if case.naming == "qwen":
            module = f"language_model.model.layers.{layer_id}.mlp.switch_mlp"
        else:
            module = f"model.layers.{layer_id}.experts.switch_glu"
        for projection in PROJECTIONS:
            if projection == "down_proj":
                shape = (case.num_experts, case.hidden_size, case.intermediate_size)
            else:
                shape = (case.num_experts, case.intermediate_size, case.hidden_size)
            weight = mx.array(0.02 * rng.standard_normal(shape, dtype=np.float32), mx.bfloat16)
            if layer_id < split_layer or (layer_id == split_layer and projection == "gate_proj"):
                file = files[0]
            else:
                file = files[1]
            if case.scheme is None:
                file[f"{module}.{projection}.weight"] = weight
            else:
                bits = case.scheme.bits
                if layer_id == case.num_layers - 1 and case.last_layer_bits is not None:
                    bits = case.last_layer_bits
                packed, scales, biases = mx.quantize(weight, case.scheme.group_size, bits)
                file[f"{module}.{projection}.weight"] = packed
                file[f"{module}.{projection}.scales"] = scales
                file[f"{module}.{projection}.biases"] = biases
    directory.mkdir()
    for i, file in enumerate(files):
        mx.save_safetensors(str(directory / f"model-{i:05d}.safetensors"), file)


def build(directory: Path, case: Case) -> tuple[OffloadMoeCache, dict[str, mx.array], ExpertSource]:
    """The cache over the case's checkpoint, the fused reference state and the source."""
    write_checkpoint(directory, case)
    rules = weight_rules(case.naming)
    source = ExpertSource(str(directory), rules)
    cache = OffloadMoeCache(
        num_layers=case.num_layers,
        num_experts=case.num_experts,
        cache_size=case.cache_size,
        prefill_overlap=case.prefill_overlap,
        layout=moe_layout(case.hidden_size, case.intermediate_size, case.scheme, mx.bfloat16),
        source=source,
    )
    return cache, apply_rules(load_checkpoint(str(directory)), rules), source


def resident_rows(cache: OffloadMoeCache) -> dict[tuple[int, int], int]:
    """(layer, expert) -> slot of every expert slot_of maps; no two share a slot."""
    slot_of = np.array(cache.slot_of).reshape(cache.num_layers, cache.num_experts)
    rows = {
        (int(layer), int(expert)): int(slot_of[layer, expert])
        for layer, expert in zip(*np.nonzero(slot_of >= 0))
    }
    assert len(set(rows.values())) == len(rows)
    assert all(0 <= slot < cache.cache_size for slot in rows.values())
    return rows


def check_rows(
    cache: OffloadMoeCache,
    reference: dict[str, mx.array],
    prefixes: list[str],
    rows: dict[tuple[int, int], int],
) -> None:
    """Each (layer, expert) -> slot of rows holds the expert's checkpoint row in every bank."""
    for role, bank in cache.bank_views().items():
        for (layer_id, expert), slot in rows.items():
            expected = reference[f"{prefixes[layer_id]}.experts.{role}"][expert]
            assert mx.array_equal(bank[slot], expected), (role, layer_id, expert, slot)


@pytest.mark.parametrize("case", CASES, ids=str)
def test_rows_match_checkpoint(tmp_path: Path, case: Case) -> None:
    cache, reference, source = build(tmp_path / "checkpoint", case)
    prefixes = source.layer_prefixes
    assert (source.segment_table(cache.layout)[..., 1] % 4096 != 0).any(), "all experts aligned"
    cache.fill_initial()
    rows = resident_rows(cache)
    assert len(rows) == case.cache_size - cache.window_slots
    check_rows(cache, reference, prefixes, rows)
    for step, entry in enumerate(case.script, start=1):
        if isinstance(entry, Prefill):
            cache.begin_prefill()
            resident = resident_rows(cache)
            assert all(slot >= cache.window_slots for slot in resident.values())
            for layer_id in entry.layers[:2]:
                cache.prefetch_prefill_layer(layer_id)
            for i, layer_id in enumerate(entry.layers):
                slot_map = host_ints(cache.prefetch_prefill_layer(layer_id))
                cache.wait_prefill_layer(layer_id)
                window = (layer_id % 2) * cache.num_experts
                for expert, slot in enumerate(slot_map):
                    assert slot == resident.get((layer_id, expert), window + expert)
                layer_rows = {(layer_id, expert): slot for expert, slot in enumerate(slot_map)}
                check_rows(cache, reference, prefixes, layer_rows)
                if i + 2 < len(entry.layers):
                    cache.prefetch_prefill_layer(entry.layers[i + 2])
        else:
            for layer_id, expert_ids in entry:
                slots = cache.ensure_experts(layer_id, list(expert_ids), step)
                rows = resident_rows(cache)
                assert slots == [rows[(layer_id, expert)] for expert in expert_ids]
                check_rows(cache, reference, prefixes, rows)
    stats = cache.stats()
    calls = [call for entry in case.script if not isinstance(entry, Prefill) for call in entry]
    assert sum(stats.hits) + sum(stats.misses) == sum(len(expert_ids) for _, expert_ids in calls)


def test_source_rejects_a_layer_in_another_format(tmp_path: Path) -> None:
    case = Case(3, 4, 128, 64, 4, False, last_layer_bits=8)
    write_checkpoint(tmp_path / "checkpoint", case)
    source = ExpertSource(str(tmp_path / "checkpoint"), weight_rules(case.naming))
    layout = moe_layout(case.hidden_size, case.intermediate_size, case.scheme, mx.bfloat16)
    with pytest.raises(ValueError, match="model.layers.2.mlp.experts.gate_up_proj.weight"):
        source.segment_table(layout)


def test_window_protocol(tmp_path: Path) -> None:
    case = Case(4, 4, 128, 64, 10, True)
    cache, _, _ = build(tmp_path / "checkpoint", case)
    cache.fill_initial()
    cache.begin_prefill()
    cache.prefetch_prefill_layer(0)
    with pytest.raises(RuntimeError, match="still being read"):
        cache.ensure_experts(1, [0], 1)
    with pytest.raises(RuntimeError, match="never waited"):
        cache.prefetch_prefill_layer(2)
    with pytest.raises(RuntimeError, match="not prefetched"):
        cache.wait_prefill_layer(1)
    cache.wait_prefill_layer(0)
    cache.prefetch_prefill_layer(2)
    cache.begin_prefill()
    assert cache.ensure_experts(1, [3], 1) == [0]


def test_read_failure_poisons_the_cache(tmp_path: Path) -> None:
    case = Case(2, 4, 128, 64, 4, False)
    cache, _, _ = build(tmp_path / "checkpoint", case)
    last_file = tmp_path / "checkpoint" / "model-00001.safetensors"
    os.truncate(last_file, last_file.stat().st_size // 2)
    with pytest.raises(RuntimeError, match="model-00001.safetensors at .*: unexpected end of file"):
        cache.ensure_experts(1, [0, 1, 2, 3], 1)
    with pytest.raises(RuntimeError, match="failed earlier"):
        cache.ensure_experts(0, [0], 2)


def test_expert_bank_matches_moe_layout() -> None:
    affine = {
        "gate_up_proj.weight": ((8, 128, 16), mx.uint32),
        "gate_up_proj.scales": ((8, 128, 2), mx.bfloat16),
        "gate_up_proj.biases": ((8, 128, 2), mx.bfloat16),
        "down_proj.weight": ((8, 128, 8), mx.uint32),
        "down_proj.scales": ((8, 128, 1), mx.bfloat16),
        "down_proj.biases": ((8, 128, 1), mx.bfloat16),
    }
    bf16 = {
        "gate_up_proj.weight": ((8, 128, 128), mx.bfloat16),
        "down_proj.weight": ((8, 128, 64), mx.bfloat16),
    }
    for scheme, expected in [(AffineScheme(4, 64), affine), (None, bf16)]:
        layer = MoELayer(8, 2, 128, 64, scheme=scheme, dtype=mx.bfloat16)
        parameters = {
            key: (value.shape, value.dtype)
            for key, value in dict(tree_flatten(layer.parameters())).items()
        }
        assert parameters == expected
        layout = moe_layout(128, 64, scheme, mx.bfloat16)
        assert list(layout) == list(expected)
        assert {role: ((8, *spec.shape), spec.dtype) for role, spec in layout.items()} == expected


@pytest.mark.skipif(MODEL_PATH is None, reason="FTMETAL_TEST_MODEL_PATH is not set")
def test_real_checkpoint_rows() -> None:
    assert MODEL_PATH is not None
    config = EngineConfig(model_path=MODEL_PATH)
    model_config = config.model_config
    rules = model_weight_rules(config.architecture, model_config)
    source = ExpertSource(MODEL_PATH, rules)
    scheme = fused_scheme_of(model_config, "model.layers.0.mlp.experts", GATE_UP_PARTS)
    layout = moe_layout(
        model_config.hidden_size, model_config.moe_intermediate_size, scheme, mx.bfloat16
    )
    cache = OffloadMoeCache(
        num_layers=source.num_layers,
        num_experts=source.num_experts,
        cache_size=source.num_experts,
        prefill_overlap=False,
        layout=layout,
        source=source,
    )
    reference = apply_rules(load_checkpoint(MODEL_PATH), rules)
    last = source.num_layers - 1
    for layer_id, expert_ids in [(0, [0, 1, 2, source.num_experts - 1]), (last, [17, 18, 100])]:
        slots = cache.ensure_experts(layer_id, expert_ids, 1)
        check_rows(
            cache,
            reference,
            source.layer_prefixes,
            {(layer_id, expert): slot for expert, slot in zip(expert_ids, slots)},
        )
