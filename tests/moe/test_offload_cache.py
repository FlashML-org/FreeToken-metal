"""OffloadMoeCache and ExpertSource on synthetic checkpoints, and on the real one.

The synthetic checkpoints are offload_harness's. Every bank row the cache holds must
equal, byte for byte, the expert's row of apply_rules(load_checkpoint(...)): after the
initial fill, after each ensure_experts call of a hand-written script (each returned slot,
and every slot slot_of maps), and at every slot of a prefetched prefill layer's slot map,
whose resident experts keep their slots. Large experts make one run of reads longer than
a staging buffer; gate and up in different bits load as two banks; one pool for every layer
and prefill buffers over the pools (which drop the experts under them as a prefill begins)
take the same checks. Errors: a layer in another format, the prefill buffer protocol, a
file cut short under the cache. The real-checkpoint case needs FTMETAL_TEST_MODEL_PATH.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from ftmetal.core import host_ints
from ftmetal.engine.config import EngineConfig
from ftmetal.layers.moe import ExpertSchemes, MoELayer, moe_layout
from ftmetal.layers.quantization.config import AffineScheme
from ftmetal.models.loader import apply_rules, load_checkpoint
from ftmetal.models.qwen3_5_moe.model import schemes_of
from ftmetal.models.qwen3_5_moe.weight import GATE_UP_PARTS
from ftmetal.models.register import model_weight_rules
from ftmetal.moe.offload_cache import OffloadMoeCache
from ftmetal.moe.slot_pools import MoePoolGrouping, PrefillBufferPlacement
from ftmetal.moe.source import ExpertSource
from offload_harness import (
    CheckpointSpec,
    Naming,
    build_cache,
    checkpoint_rules,
    weight_rules,
    write_checkpoint,
)

MODEL_PATH = os.environ.get("FTMETAL_TEST_MODEL_PATH")
# One decode step: the (layer, distinct expert ids) calls of ensure_experts, in order.
Step = tuple[tuple[int, tuple[int, ...]], ...]


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
    # Bits of every layer's up projection, if not the scheme's.
    up_bits: int | None = None
    grouping: MoePoolGrouping = "per-layer"
    buffer_placement: PrefillBufferPlacement = "dedicated"

    @property
    def spec(self) -> CheckpointSpec:
        return CheckpointSpec(
            num_layers=self.num_layers,
            num_experts=self.num_experts,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            naming=self.naming,
            scheme=self.scheme,
            last_layer_bits=self.last_layer_bits,
            up_bits=self.up_bits,
        )

    def __str__(self) -> str:
        scheme = "bf16" if self.scheme is None else f"q{self.scheme.bits}g{self.scheme.group_size}"
        if self.up_bits is not None:
            scheme += f"-up{self.up_bits}"
        overlap = f"-{self.buffer_placement}" if self.prefill_overlap else ""
        return (
            f"{self.naming}-L{self.num_layers}-E{self.num_experts}-H{self.hidden_size}"
            f"-I{self.intermediate_size}-S{self.cache_size}-{self.grouping}{overlap}-{scheme}"
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
            ((2, (0, 1, 3)),),
        ),
        naming="gemma4",
    ),
    Case(2, 4, 192, 64, 8, True, script=(Prefill((0, 1)), ((1, (2, 3)),)), scheme=None),
    # 2 MiB per expert and projection: a run of four experts outgrows the staging buffer.
    Case(1, 4, 2048, 2048, 4, False, script=(((0, (0, 1, 2, 3)),),)),
    # Up in 8 bits: gate and up are two banks, read and placed apart.
    Case(
        3,
        8,
        192,
        64,
        12,
        False,
        script=(((0, (0, 5, 6)), (1, (7,)), (2, (1, 2))), ((0, (5, 7, 3)), (2, (6, 0)))),
        up_bits=8,
    ),
    Case(
        3,
        4,
        128,
        64,
        11,
        True,
        script=(((1, (2,)),), Prefill((0, 1, 2)), ((1, (0, 3)),)),
        naming="gemma4",
        up_bits=8,
    ),
    # One pool of 12 for every layer, the buffers over its first 8 slots.
    Case(
        4,
        4,
        128,
        64,
        12,
        True,
        script=(
            ((0, (1, 2)), (1, (3,)), (3, (0, 3))),
            ((0, (0, 3)), (1, (0, 1, 2)), (2, (2,)), (3, (1,))),
            Prefill((0, 1, 2, 3)),
            ((1, (3, 1)), (2, (0, 1, 2, 3))),
            Prefill((1, 2)),
            ((2, (0, 1)),),
        ),
        grouping="unified",
        buffer_placement="overlay",
    ),
    # Runs of 6 per layer, the buffers over layers 0 and 1 and the start of layer 2's.
    Case(
        3,
        8,
        192,
        64,
        18,
        True,
        script=(
            ((0, (0, 5, 6)), (1, (7,)), (2, (1, 2, 6, 7))),
            Prefill((0, 1, 2)),
            ((0, (5, 7, 3)), (2, (6, 0))),
        ),
        grouping="per-layer",
        buffer_placement="overlay",
    ),
]


def build(directory: Path, case: Case) -> tuple[OffloadMoeCache, dict[str, mx.array], ExpertSource]:
    """The cache over the case's checkpoint, the fused reference state and the source."""
    cache, source = build_cache(
        directory,
        case.spec,
        cache_size=case.cache_size,
        prefill_overlap=case.prefill_overlap,
        grouping=case.grouping,
        buffer_placement=case.buffer_placement,
    )
    reference = apply_rules(load_checkpoint(str(directory)), checkpoint_rules(case.spec))
    return cache, reference, source


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
    assert len(rows) == sum(pool.num_slots for pool in cache.slot_layout.pools)
    check_rows(cache, reference, prefixes, rows)
    calls_with_misses = 0
    for step, entry in enumerate(case.script, start=1):
        if isinstance(entry, Prefill):
            cache.begin_prefill()
            resident = resident_rows(cache)
            assert all(slot >= 2 * cache.buffer_size for slot in resident.values())
            for layer_id in entry.layers[:2]:
                cache.prefetch_prefill_layer(layer_id)
            for i, layer_id in enumerate(entry.layers):
                slot_map = host_ints(cache.prefetch_prefill_layer(layer_id))
                cache.wait_prefill_layer(layer_id)
                buffer_slot = (layer_id % 2) * cache.buffer_size
                for expert, slot in enumerate(slot_map):
                    if (layer_id, expert) in resident:
                        assert slot == resident[(layer_id, expert)]
                    else:
                        assert slot == buffer_slot
                        buffer_slot += 1
                layer_rows = {(layer_id, expert): slot for expert, slot in enumerate(slot_map)}
                check_rows(cache, reference, prefixes, layer_rows)
                if i + 2 < len(entry.layers):
                    cache.prefetch_prefill_layer(entry.layers[i + 2])
        else:
            for layer_id, expert_ids in entry:
                misses_before = sum(cache.stats().misses)
                slots = cache.ensure_experts(layer_id, list(expert_ids), step)
                calls_with_misses += sum(cache.stats().misses) > misses_before
                rows = resident_rows(cache)
                assert slots == [rows[(layer_id, expert)] for expert in expert_ids]
                check_rows(cache, reference, prefixes, rows)
    stats = cache.stats()
    calls = [call for entry in case.script if not isinstance(entry, Prefill) for call in entry]
    assert sum(stats.hits) + sum(stats.misses) == sum(len(expert_ids) for _, expert_ids in calls)
    assert sum(stats.requests) == calls_with_misses


def test_source_rejects_a_layer_in_another_format(tmp_path: Path) -> None:
    case = Case(3, 4, 128, 64, 4, False, last_layer_bits=8)
    write_checkpoint(tmp_path / "checkpoint", case.spec)
    source = ExpertSource(str(tmp_path / "checkpoint"), weight_rules(case.naming))
    layout = moe_layout(
        case.hidden_size, case.intermediate_size, ExpertSchemes.uniform(case.scheme), mx.bfloat16
    )
    with pytest.raises(ValueError, match="model.layers.2.mlp.experts.gate_up_proj.weight"):
        source.segment_table(layout)


def test_prefill_buffer_protocol(tmp_path: Path) -> None:
    case = Case(4, 4, 128, 64, 14, True)
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
    # Buffers [0, 2); layer 1's pool [5, 8) holds experts 0-2, expert 0 (lowest id) goes.
    assert cache.ensure_experts(1, [3], 1) == [5]


def test_read_failure_poisons_the_cache(tmp_path: Path) -> None:
    case = Case(2, 4, 128, 64, 4, False)
    cache, _, _ = build(tmp_path / "checkpoint", case)
    last_file = tmp_path / "checkpoint" / "model-00001.safetensors"
    os.truncate(last_file, last_file.stat().st_size // 2)
    with pytest.raises(RuntimeError, match="model-00001.safetensors at .*: unexpected end of file"):
        cache.ensure_experts(1, [2, 3], 1)
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
    # Gate 4-bit, up 8-bit, down 4-bit: gate and up are two banks.
    split = {
        "gate_up_proj.gate_proj.weight": ((8, 64, 16), mx.uint32),
        "gate_up_proj.gate_proj.scales": ((8, 64, 2), mx.bfloat16),
        "gate_up_proj.gate_proj.biases": ((8, 64, 2), mx.bfloat16),
        "gate_up_proj.up_proj.weight": ((8, 64, 32), mx.uint32),
        "gate_up_proj.up_proj.scales": ((8, 64, 2), mx.bfloat16),
        "gate_up_proj.up_proj.biases": ((8, 64, 2), mx.bfloat16),
        "down_proj.weight": ((8, 128, 8), mx.uint32),
        "down_proj.scales": ((8, 128, 1), mx.bfloat16),
        "down_proj.biases": ((8, 128, 1), mx.bfloat16),
    }
    four, eight = AffineScheme(4, 64), AffineScheme(8, 64)
    for schemes, expected in [
        (ExpertSchemes.uniform(four), affine),
        (ExpertSchemes.uniform(None), bf16),
        (ExpertSchemes(four, eight, four), split),
    ]:
        layer = MoELayer(8, 2, 128, 64, schemes=schemes, dtype=mx.bfloat16)
        parameters = {
            key: (value.shape, value.dtype)
            for key, value in dict(tree_flatten(layer.parameters())).items()
        }
        assert parameters == expected
        layout = moe_layout(128, 64, schemes, mx.bfloat16)
        assert list(layout) == list(expected)
        assert {role: ((8, *spec.shape), spec.dtype) for role, spec in layout.items()} == expected


@pytest.mark.skipif(MODEL_PATH is None, reason="FTMETAL_TEST_MODEL_PATH is not set")
def test_real_checkpoint_rows() -> None:
    assert MODEL_PATH is not None
    config = EngineConfig(model_path=MODEL_PATH)
    model_config = config.model_config
    rules = model_weight_rules(config.architecture, model_config)
    source = ExpertSource(MODEL_PATH, rules)
    schemes = ExpertSchemes(
        *schemes_of(model_config, "model.layers.0.mlp.experts", (*GATE_UP_PARTS, "down_proj"))
    )
    layout = moe_layout(
        model_config.hidden_size, model_config.moe_intermediate_size, schemes, mx.bfloat16
    )
    cache = OffloadMoeCache(
        num_layers=source.num_layers,
        num_experts=source.num_experts,
        top_k=model_config.num_experts_per_tok,
        cache_size=source.num_layers * model_config.num_experts_per_tok,
        prefill_overlap=False,
        prefill_overlap_min_tokens=128,
        layout=layout,
        source=source,
        grouping="unified",
        buffer_placement="overlay",
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
