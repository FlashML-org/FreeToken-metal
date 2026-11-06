"""OffloadMoeCache and ExpertSource on synthetic checkpoints, and on the real one.

The synthetic checkpoints are offload_harness's. Every bank row the cache holds must
equal, byte for byte, the expert's row of apply_rules(load_checkpoint(...)): after the
initial fill, after each ensure_experts call of a hand-written script (each returned slot,
and every slot slot_of maps), and at every slot of a prefetched prefill layer's slot map,
whose resident experts keep their slots. Large experts make one run of reads longer than
a staging buffer; gate and up in different bits load as two banks in every layer; one pool
for every layer and prefill buffers over the pools (which drop the experts under them as a
prefill begins) take the same checks, and so do some of the cases reading every stored part
from an expert pack of the checkpoint. With a victim cache the same scripts must also leave
no expert both in a slot and in the victim cache, and every miss or prefetched absent expert
must be read or copied from it. The victim cache on its own: an evicted expert asked for again
is copied back, not read, and its entries drop their oldest when full; purged entries are
read again; an overlap prefill copies the experts it holds into the buffer and keeps them,
and stores the experts it drops from the slots under the buffers, the most recently used
first. Errors: a layer in another format, the prefill buffer protocol, a file cut short under
the cache. The real-checkpoint case needs FTMETAL_TEST_MODEL_PATH.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, replace
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from ftmetal.engine.config import EngineConfig
from ftmetal.kernel._ext import _C
from ftmetal.layers.moe import MoELayer, OffloadMoELayer
from ftmetal.layers.quantization.config import AffineScheme, FpBlockScheme
from ftmetal.layers.quantization.moe import ExpertSchemes
from ftmetal.models.loader import apply_rules, load_checkpoint
from ftmetal.models.qwen3_5_moe.model import schemes_of
from ftmetal.models.qwen3_5_moe.weight import GATE_UP_PARTS
from ftmetal.models.register import model_weight_rules, read_checkpoint
from ftmetal.moe.offload_cache import OffloadMoeCache
from ftmetal.moe.slot_pools import (
    MoeCachePolicy,
    MoePoolGrouping,
    PrefillBufferPlacement,
    VictimMemoryKind,
    bank_layouts,
    plan_slot_layout,
)
from ftmetal.moe.source import ExpertSource
from offload_harness import (
    CheckpointSpec,
    Naming,
    build_cache,
    checkpoint_rules,
    entry_bytes,
    expected_code_rows,
    moe_layout,
    read_state,
    weight_rules,
    write_checkpoint,
)

MODEL_PATH = os.environ.get("FTMETAL_TEST_MODEL_PATH")
# One decode step: the (layer, distinct expert ids) calls of ensure_experts, in order.
Step = tuple[tuple[int, tuple[int, ...]], ...]


@dataclass(frozen=True)
class Prefill:
    """An overlapped prefill of these layers, in order: two prefetched ahead; with routed, each
    layer's routed experts, the reads of the others dropped where they have not begun."""

    layers: tuple[int, ...]
    routed: tuple[tuple[int, ...], ...] | None = None


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
    buffer_placement: PrefillBufferPlacement = "overlay"
    # One tensor per expert, in the Hugging Face layout.
    is_per_expert: bool = False
    # The victim cache's budget in entries of the widest layer (0: none).
    victim_cache_entries: int = 0
    # Read from an expert pack of the checkpoint, with pair codes.
    is_packed: bool = False
    has_pair_codes: bool = False
    # Every pool's spare slots for deferred victim stores (None: the cache's default, none in
    # pools this small), the victim cache's memory, the eviction order, the SSD keepalive.
    deferred_store_spares: int | None = None
    victim_memory: VictimMemoryKind = "purgeable"
    cache_policy: MoeCachePolicy = "lru"
    ssd_keepalive_us: int = 0

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
            is_per_expert=self.is_per_expert,
        )

    def __str__(self) -> str:
        scheme = "bf16" if self.scheme is None else f"q{self.scheme.bits}g{self.scheme.group_size}"
        if self.up_bits is not None:
            scheme += f"-up{self.up_bits}"
        overlap = f"-{self.buffer_placement}" if self.prefill_overlap else ""
        per_expert = "-per-expert" if self.is_per_expert else ""
        victims = f"-V{self.victim_cache_entries}" if self.victim_cache_entries else ""
        packed = "-coded" if self.has_pair_codes else "-packed" if self.is_packed else ""
        routed = any(isinstance(e, Prefill) and e.routed is not None for e in self.script)
        spares = "" if self.deferred_store_spares is None else f"-D{self.deferred_store_spares}"
        memory = "-reserved" if self.victim_memory == "reserved" else ""
        policy = "-lrfu" if self.cache_policy == "lrfu" else ""
        keepalive = f"-K{self.ssd_keepalive_us}" if self.ssd_keepalive_us else ""
        routed_suffix = "-routed" if routed else ""
        return (
            f"{self.naming}-L{self.num_layers}-E{self.num_experts}-H{self.hidden_size}"
            f"-I{self.intermediate_size}-S{self.cache_size}-{self.grouping}{overlap}-{scheme}"
            f"{per_expert}{victims}{packed}{spares}{memory}{policy}{keepalive}{routed_suffix}"
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
    # The last layer in 8 bits: one pool per layout, the 8-bit one first under the buffers.
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
            ((0, (5, 7, 3)), (2, (6, 0)), (1, (3, 4))),
        ),
        last_layer_bits=8,
        grouping="unified",
    ),
    # Up in 8 bits, and the last layer all in 8, its gate and up apart like the others'.
    Case(
        3,
        8,
        192,
        64,
        18,
        True,
        script=(((0, (0, 5)), (2, (1, 2))), Prefill((0, 1, 2)), ((2, (6, 0)), (1, (3,)))),
        up_bits=8,
        last_layer_bits=8,
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
    # One tensor per expert: each expert's part is a read of its own (or a run, where the
    # file happens to store the experts back to back), in slots and in the prefill buffers.
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
        is_per_expert=True,
    ),
    Case(
        2,
        4,
        192,
        64,
        8,
        True,
        script=(Prefill((0, 1)), ((1, (2, 3)),)),
        scheme=None,
        is_per_expert=True,
    ),
    Case(1, 4, 2048, 2048, 4, False, script=(((0, (0, 1, 2, 3)),),), is_per_expert=True),
    # A victim cache under the scripts above: decode misses, overlap prefills, bf16, gate and
    # up apart, one pool for every layer, pools of two layouts.
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
        victim_cache_entries=6,
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
        victim_cache_entries=8,
    ),
    # Every slot under the buffers: the prefill stores each pool's two newest experts.
    Case(
        3,
        8,
        192,
        64,
        16,
        True,
        script=(((0, (6, 7)), (2, (5, 6, 7))), Prefill((0, 1, 2)), ((0, (0, 1)), (2, (0, 1)))),
        scheme=None,
        victim_cache_entries=8,
    ),
    Case(
        3,
        8,
        192,
        64,
        12,
        False,
        script=(((0, (0, 5, 6)), (1, (7,)), (2, (1, 2))), ((0, (5, 7, 1)), (2, (6, 0)))),
        up_bits=8,
        victim_cache_entries=6,
    ),
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
        victim_cache_entries=4,
    ),
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
            ((0, (5, 7, 3)), (2, (6, 0)), (1, (3, 4))),
        ),
        last_layer_bits=8,
        grouping="unified",
        victim_cache_entries=6,
    ),
]


# The same from an expert pack: decode misses, overlap prefills, bf16, an expert of 6.75 MiB (one
# record a read), gate and up apart, pools of two layouts (two record strides), one tensor per
# expert, a victim cache.
CASES += [
    replace(case, is_packed=True)
    for case in (CASES[0], CASES[1], CASES[2], CASES[3], CASES[4], CASES[7], CASES[10], CASES[13])
]
# Pair codes: hidden 256 and intermediate 128 in groups of 64 (512 codes per projection part,
# whole lines in any width; 4 experts, a few thousand pairs), on decode misses, overlap prefills
# in Gemma4 names, gate and up apart, pools of two layouts with the buffers over them, a victim
# cache; hidden 192 (192 codes per part: whole lines in 8 or 16 bits, so 16 for its pairs); and
# the 6.75 MiB expert.
CODED_SCRIPT: tuple[Step | Prefill, ...] = (
    ((0, (0, 1)), (1, (3,)), (2, (1, 2))),
    ((0, (2, 3)), (1, (0, 1, 2)), (2, (3, 0))),
    ((1, (3, 1)), (2, (1, 2))),
)
CASES += [
    Case(3, 4, 256, 128, 8, False, script=CODED_SCRIPT, is_packed=True, has_pair_codes=True),
    replace(CASES[1], hidden_size=256, is_packed=True, has_pair_codes=True),
    replace(CASES[5], hidden_size=256, intermediate_size=128, is_packed=True, has_pair_codes=True),
    Case(
        3,
        4,
        256,
        128,
        10,
        True,
        script=(((0, (0, 1)), (2, (1, 2, 3))), Prefill((0, 1, 2)), ((0, (2, 3)), (2, (0, 1)))),
        last_layer_bits=8,
        grouping="unified",
        is_packed=True,
        has_pair_codes=True,
    ),
    Case(
        3,
        4,
        256,
        128,
        8,
        False,
        script=CODED_SCRIPT,
        victim_cache_entries=3,
        is_packed=True,
        has_pair_codes=True,
    ),
    replace(CASES[0], is_packed=True, has_pair_codes=True),
    replace(CASES[3], is_packed=True, has_pair_codes=True),
]

# Route-drop: each prefill layer routes some of its experts, the reads of the others that have
# not begun are dropped (a pack's records only); with a victim cache's copies too.
ROUTED_SCRIPT = (
    ((0, (0, 1)), (1, (2,)), (2, (0, 3)), (3, (1,))),
    Prefill((0, 1, 2, 3), routed=((0,), (1, 2), (), (3,))),
    ((0, (2, 3)), (1, (0, 1)), (2, (1,)), (3, (0, 2))),
    Prefill((1, 2), routed=((0, 1, 2, 3), (2,))),
    ((1, (3,)), (2, (0, 1))),
)
ROUTED_CASE = Case(
    4, 4, 128, 64, 12, True, script=ROUTED_SCRIPT, grouping="unified", victim_cache_entries=4
)
CASES += [
    ROUTED_CASE,
    replace(ROUTED_CASE, is_packed=True),
    replace(ROUTED_CASE, is_packed=True, has_pair_codes=True),
    replace(ROUTED_CASE, victim_cache_entries=0, is_packed=True),
]


def decode_steps(seed: int, num_steps: int) -> tuple[Step, ...]:
    """num_steps decode steps of 4 layers, each 2 of its 4 experts."""
    rng = np.random.default_rng(seed)
    return tuple(
        tuple(
            (layer_id, tuple(int(e) for e in rng.choice(4, 2, replace=False)))
            for layer_id in range(4)
        )
        for _ in range(num_steps)
    )


# Deferred victim stores: the overlap prefills free the slots under their buffers, the spares
# come from them, and the decode's victim hits that evict a resident take one, the resident
# stored after the answer. The same with reserved victim memory, and under LRFU.
DEFERRED_SCRIPT = (
    *decode_steps(0, 6),
    Prefill((0, 1, 2, 3)),
    *decode_steps(1, 30),
    Prefill((1, 2)),
    *decode_steps(2, 20),
)
DEFERRED_CASE = Case(
    4,
    4,
    128,
    64,
    12,
    True,
    script=DEFERRED_SCRIPT,
    grouping="unified",
    victim_cache_entries=4,
    deferred_store_spares=2,
)
CASES += [
    DEFERRED_CASE,
    replace(DEFERRED_CASE, victim_memory="reserved"),
    replace(DEFERRED_CASE, cache_policy="lrfu"),
    replace(CASES[3], cache_policy="lrfu"),
    replace(CASES[0], ssd_keepalive_us=50),
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
        victim_cache_bytes=case.victim_cache_entries * entry_bytes(case.spec),
        is_packed=case.is_packed,
        with_pair_codes=case.has_pair_codes,
        victim_memory=case.victim_memory,
        cache_policy=case.cache_policy,
        deferred_store_spares=case.deferred_store_spares,
        ssd_keepalive_us=case.ssd_keepalive_us,
    )
    reference = apply_rules(read_state(directory, case.spec)[0], checkpoint_rules(case.spec))
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


def expected_rows(
    cache: OffloadMoeCache,
    reference: dict[str, mx.array],
    prefixes: list[str],
    layers: list[int] | None = None,
) -> dict[int, dict[str, np.ndarray]]:
    """Per layer (of layers, all by default), role -> uint8 [E, row bytes]: each expert's bank
    row as the cache holds it."""
    rows = {}
    for layer_id in range(cache.num_layers) if layers is None else layers:
        layout = cache.slot_layout.pools[cache.layer_pools[layer_id]].layout
        experts = f"{prefixes[layer_id]}.experts"
        layer_rows = {}
        for role in layout:
            projection, _, array = role.rpartition(".")
            if array == "codes":
                assert cache.pairs is not None, "code rows come with pair tables"
                layer_rows[role] = expected_code_rows(
                    reference[f"{experts}.{projection}.scales"],
                    reference[f"{experts}.{projection}.biases"],
                    cache.pairs[layer_id],
                )
            else:
                bank = reference[f"{experts}.{role}"]
                layer_rows[role] = np.array(bank.view(mx.uint8)).reshape(bank.shape[0], -1)
        rows[layer_id] = layer_rows
    return rows


def check_exclusive(cache: OffloadMoeCache) -> None:
    """No expert is both in a slot and in the victim cache."""
    resident = {layer_id * cache.num_experts + expert for layer_id, expert in resident_rows(cache)}
    assert resident.isdisjoint(cache.service.victim_cache_ids())


def slot_offsets(cache: OffloadMoeCache, layer_id: int, slot: int) -> np.ndarray:
    """The byte offset of the layer's row at slot in every bank."""
    return cache.destinations[layer_id, :, 0] + slot * cache.destinations[layer_id, :, 1]


def check_rows(
    cache: OffloadMoeCache,
    expected: dict[int, dict[str, np.ndarray]],
    located: dict[tuple[int, int], np.ndarray],
) -> None:
    """Each (layer, expert) -> its byte offsets in every bank holds the expert's row there."""
    banks = {role: np.array(bank).reshape(-1) for role, bank in cache.banks.items()}
    for (layer_id, expert), offsets in located.items():
        for (role, rows), offset in zip(expected[layer_id].items(), offsets, strict=True):
            row = banks[role][offset : offset + rows.shape[1]]
            assert (row == rows[expert]).all(), (role, layer_id, expert, offset)


@pytest.mark.parametrize("case", CASES, ids=str)
def test_rows_match_checkpoint(tmp_path: Path, case: Case) -> None:
    cache, reference, source = build(tmp_path / "checkpoint", case)
    expected = expected_rows(cache, reference, source.layer_prefixes)
    layouts = [cache.slot_layout.pools[pool].layout for pool in cache.layer_pools]
    locations = source.segment_table(layouts)[1]
    assert (cache.pairs is not None) == case.has_pair_codes
    if case.is_packed:
        stored = locations[..., 0] != _C.FILL_FILE
        assert (locations[..., 0][stored] == len(source.files) - 1).all(), "not all from the pack"
    else:
        assert (locations[..., 1] % 4096 != 0).any(), "all aligned"
    cache.fill_initial()
    rows = resident_rows(cache)
    assert len(rows) == sum(pool.num_slots for pool in cache.slot_layout.pools)

    def located(rows: dict[tuple[int, int], int]) -> dict[tuple[int, int], np.ndarray]:
        return {key: slot_offsets(cache, key[0], slot) for key, slot in rows.items()}

    check_rows(cache, expected, located(rows))
    pools = cache.slot_layout.pools
    calls_with_misses = 0
    # The absent experts of the prefetched layers.
    buffered = 0
    for step, entry in enumerate(case.script, start=1):
        if isinstance(entry, Prefill):
            cache.begin_prefill()
            check_exclusive(cache)
            resident = resident_rows(cache)
            for (layer_id, _), slot in resident.items():
                pool = pools[cache.layer_pools[layer_id]]
                assert slot - pool.first_slot >= pool.volatile_slots, "a slot under the buffers"
            for layer_id in entry.layers[:2]:
                cache.prefetch_prefill_layer(layer_id)
            for i, layer_id in enumerate(entry.layers):
                expert_bytes = cache.prefetch_prefill_layer(layer_id)
                routed = range(cache.num_experts) if entry.routed is None else entry.routed[i]
                if entry.routed is not None:
                    cache.service.route_prefill_layer(layer_id, list(routed))
                cache.wait_prefill_layer(layer_id)
                offsets = np.concatenate([np.array(v) for v in expert_bytes.values()], axis=1)
                buffer_row = 0
                for expert in range(cache.num_experts):
                    if (layer_id, expert) in resident:
                        expected_offsets = slot_offsets(
                            cache, layer_id, resident[(layer_id, expert)]
                        )
                    else:
                        expected_offsets = (layer_id % 2) * cache.buffer_bytes + buffer_row * (
                            cache.destinations[layer_id, :, 1]
                        )
                        buffer_row += 1
                    assert (offsets[expert] == expected_offsets).all(), (layer_id, expert)
                buffered += buffer_row
                # A dropped expert's row is never filled: the routed and the resident ones are.
                wanted = set(routed) | {e for (layer, e) in resident if layer == layer_id}
                check_rows(
                    cache, expected, {(layer_id, expert): offsets[expert] for expert in wanted}
                )
                if i + 2 < len(entry.layers):
                    cache.prefetch_prefill_layer(entry.layers[i + 2])
        else:
            for layer_id, expert_ids in entry:
                misses_before = sum(cache.stats().misses)
                slots = cache.ensure_experts(layer_id, list(expert_ids), step)
                calls_with_misses += sum(cache.stats().misses) > misses_before
                rows = resident_rows(cache)
                assert slots == [rows[(layer_id, expert)] for expert in expert_ids]
                check_rows(cache, expected, located(rows))
                check_exclusive(cache)
    stats = cache.stats()
    calls = [call for entry in case.script if not isinstance(entry, Prefill) for call in entry]
    assert sum(stats.hits) + sum(stats.misses) == sum(len(expert_ids) for _, expert_ids in calls)
    assert sum(stats.requests) == calls_with_misses
    if case.is_packed:
        # Whole records, read in runs and never around: the records' bytes alone.
        for layer_id, roles in enumerate(source.parts):
            part = next(iter(roles.values()))[0]
            stride = int(part.locations[1, 1] - part.locations[0, 1])
            assert stats.bytes_read[layer_id] == stats.experts_read[layer_id] * stride, layer_id
    copied = sum(stats.victim_hits) + sum(stats.victim_buffer_hits)
    dropped = sum(stats.experts_dropped)
    assert sum(stats.experts_read) + copied + dropped == sum(stats.misses) + buffered
    # Only the experts a prefill routes elsewhere are dropped; without records only the victim
    # cache's copies.
    unrouted = sum(
        len(set(range(case.num_experts)) - set(routed))
        for entry in case.script
        if isinstance(entry, Prefill) and entry.routed is not None
        for routed in entry.routed
    )
    assert dropped <= unrouted
    assert case.is_packed or case.victim_cache_entries > 0 or dropped == 0
    assert (copied > 0) == (case.victim_cache_entries > 0)
    assert (sum(stats.deferred_stores) > 0) == bool(case.deferred_store_spares)


def check_resident_rows(cache: OffloadMoeCache, expected: dict[int, dict[str, np.ndarray]]) -> None:
    """Every slot slot_of maps holds its expert's row."""
    rows = resident_rows(cache)
    check_rows(
        cache, expected, {key: slot_offsets(cache, key[0], slot) for key, slot in rows.items()}
    )


# One layer of 8 experts, two slots and two victim cache entries: each call's experts, then the
# flat ids the victim cache holds and the experts read from the checkpoint. The fill put
# experts 1 and 0 in slots 0 and 1.
VICTIM_LRU_SCRIPT: list[tuple[tuple[int, ...], tuple[int, ...], int]] = [
    ((2,), (0,), 1),
    ((3,), (0, 1), 1),
    # A third store drops the oldest.
    ((4,), (1, 2), 1),
    ((0,), (2, 3), 1),
    # 2 comes back from the victim cache and leaves it; storing 4 drops 3.
    ((2,), (4,), 0),
    ((4, 0), (2,), 0),
]


@pytest.mark.parametrize("memory", ["purgeable", "reserved"])
def test_victim_cache_serves_evicted_experts(tmp_path: Path, memory: VictimMemoryKind) -> None:
    cache, reference, source = build(
        tmp_path / "checkpoint",
        Case(1, 8, 128, 64, 2, False, victim_cache_entries=2, victim_memory=memory),
    )
    expected = expected_rows(cache, reference, source.layer_prefixes)
    assert cache.stats().victim_entries == 2
    cache.fill_initial()
    for step, (expert_ids, cached, num_read) in enumerate(VICTIM_LRU_SCRIPT, start=1):
        read_before = sum(cache.stats().experts_read)
        cache.ensure_experts(0, list(expert_ids), step)
        check_resident_rows(cache, expected)
        assert cache.service.victim_cache_ids() == list(cached), step
        assert sum(cache.stats().experts_read) - read_before == num_read, step
    stats = cache.stats()
    assert (sum(stats.misses), sum(stats.victim_hits), sum(stats.victim_stores)) == (6, 2, 6)
    assert stats.victim_copy_ns > 0


def test_purged_victim_entries_are_read_again(tmp_path: Path) -> None:
    cache, reference, source = build(
        tmp_path / "checkpoint", Case(1, 8, 128, 64, 2, False, victim_cache_entries=2)
    )
    expected = expected_rows(cache, reference, source.layer_prefixes)
    cache.fill_initial()
    cache.ensure_experts(0, [2], 1)
    cache.ensure_experts(0, [3], 2)
    assert cache.service.victim_cache_ids() == [0, 1]
    cache.service.purge_victim_cache()
    # 0 and 1 come back from the checkpoint; 2, stored after the purge, from the victim cache.
    for step, expert in [(3, 0), (4, 1), (5, 2)]:
        cache.ensure_experts(0, [expert], step)
        check_resident_rows(cache, expected)
    stats = cache.stats()
    assert (stats.victim_drops, stats.victim_hits, stats.experts_read) == ([2], [1], [4])


def test_spares_give_way_to_a_step_that_needs_every_slot(tmp_path: Path) -> None:
    """A pool of 12 slots, 2 of them spares once a lookup claims them: a copy-miss step whose
    three layers take every slot (the scheduler counts them all) gets the spares back."""
    case = Case(
        4,
        4,
        128,
        64,
        12,
        True,
        grouping="unified",
        victim_cache_entries=4,
        deferred_store_spares=2,
    )
    cache, reference, source = build(tmp_path / "checkpoint", case)
    expected = expected_rows(cache, reference, source.layer_prefixes)
    cache.fill_initial()
    cache.begin_prefill()
    absent = [e for e in range(4) if (0, e) not in resident_rows(cache)]
    cache.ensure_experts(0, absent[:1], 1)
    assert cache.slot_layout.step_starts(4) == [3]
    for layer_id in range(3):
        cache.ensure_experts(layer_id, [0, 1, 2, 3], 2)
    check_resident_rows(cache, expected)
    cache.check_healthy()


def test_begin_prefill_stores_the_experts_it_drops(tmp_path: Path) -> None:
    # Pools of 4 slots, all under the buffers, and two victim cache entries each.
    cache, reference, source = build(
        tmp_path / "checkpoint", Case(4, 8, 128, 64, 16, True, victim_cache_entries=8)
    )
    expected = expected_rows(cache, reference, source.layer_prefixes)
    assert all(pool.volatile_slots == pool.num_slots for pool in cache.slot_layout.pools)
    cache.fill_initial()
    for layer_id in range(cache.num_layers):
        cache.ensure_experts(layer_id, [0, 1], 1)
    cache.begin_prefill()
    # Each pool keeps its most recently used experts.
    assert resident_rows(cache) == {}
    assert cache.service.victim_cache_ids() == [0, 1, 8, 9, 16, 17, 24, 25]
    cache.ensure_experts(2, [1, 5], 2)
    check_resident_rows(cache, expected)
    stats = cache.stats()
    assert (stats.experts_read[2], stats.victim_hits[2], sum(stats.victim_stores)) == (1, 1, 8)
    assert cache.service.victim_cache_ids() == [0, 1, 8, 9, 16, 24, 25]


# (victim cache entries, the routed experts of layer 0): with every slot under the buffers, the
# layer's experts are all absent; its reads paused, routing drops every unrouted one exactly, the
# victim cache's copies (of the experts begin_prefill stored) as its record reads.
ROUTE_DROP_CASES: list[tuple[int, tuple[int, ...]]] = [
    (0, (1, 6)),
    (0, tuple(range(8))),
    (0, ()),
    (16, (0, 5, 7)),
]


@pytest.mark.parametrize(("victim_entries", "routed"), ROUTE_DROP_CASES)
def test_routing_drops_the_unrouted_reads(
    tmp_path: Path, victim_entries: int, routed: tuple[int, ...]
) -> None:
    case = Case(4, 8, 128, 64, 16, True, victim_cache_entries=victim_entries, is_packed=True)
    cache, reference, source = build(tmp_path / "checkpoint", case)
    expected = expected_rows(cache, reference, source.layer_prefixes)
    cache.fill_initial()
    cache.begin_prefill()
    assert resident_rows(cache) == {}
    cache.service.pause_reads(True)
    expert_bytes = cache.prefetch_prefill_layer(0)
    dropped = cache.service.route_prefill_layer(0, list(routed))
    cache.service.pause_reads(False)
    cache.wait_prefill_layer(0)
    assert dropped == cache.num_experts - len(routed)
    offsets = np.concatenate([np.array(v) for v in expert_bytes.values()], axis=1)
    check_rows(cache, expected, {(0, expert): offsets[expert] for expert in routed})
    stats = cache.stats()
    assert stats.experts_dropped[0] == dropped
    assert stats.experts_read[0] + stats.victim_buffer_hits[0] == len(routed)
    # The dropped copies' entries are back in the victim cache, unpinned: begin_prefill again.
    cache.begin_prefill()


def test_overlap_prefill_copies_cached_experts(tmp_path: Path) -> None:
    cache, reference, source = build(
        tmp_path / "checkpoint", Case(4, 8, 128, 64, 16, True, victim_cache_entries=16)
    )
    expected = expected_rows(cache, reference, source.layer_prefixes)
    cache.fill_initial()
    cache.begin_prefill()
    # Every slot lies under the buffers: each layer's experts 0-3 move to the victim cache.
    cached = [layer_id * 8 + expert for layer_id in range(4) for expert in range(4)]
    assert cache.service.victim_cache_ids() == cached
    expert_bytes = {layer_id: cache.prefetch_prefill_layer(layer_id) for layer_id in (0, 1)}
    for layer_id in (0, 1):
        cache.wait_prefill_layer(layer_id)
        offsets = np.concatenate([np.array(v) for v in expert_bytes[layer_id].values()], axis=1)
        check_rows(
            cache,
            expected,
            {(layer_id, expert): offsets[expert] for expert in range(cache.num_experts)},
        )
    stats = cache.stats()
    assert (stats.experts_read[:2], stats.victim_buffer_hits[:2]) == ([4, 4], [4, 4])
    assert sum(stats.victim_hits) == 0
    assert cache.service.victim_cache_ids() == cached


def test_source_rejects_a_layer_in_another_format(tmp_path: Path) -> None:
    case = Case(3, 4, 128, 64, 4, False, last_layer_bits=8)
    directory = tmp_path / "checkpoint"
    write_checkpoint(directory, case.spec)
    state = load_checkpoint(str(directory))
    source = ExpertSource(str(directory), weight_rules(case.naming), state, {})
    layout = moe_layout(
        case.hidden_size, case.intermediate_size, ExpertSchemes.uniform(case.scheme)
    )
    with pytest.raises(ValueError, match="model.layers.2.mlp.experts.gate_up_proj.weight"):
        source.segment_table([layout] * case.num_layers)


def test_prefill_buffer_protocol(tmp_path: Path) -> None:
    case = Case(4, 8, 128, 64, 28, True, buffer_placement="dedicated")
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
    # Layer 1's pool [3, 6) holds experts 2, 1, 0: expert 0 (lowest id) goes.
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
    # Gate and up in one scheme, kept apart: two 4-bit banks.
    apart = {
        **{
            f"gate_up_proj.{part}.{array}": (shape, dtype)
            for part in ("gate_proj", "up_proj")
            for array, (shape, dtype) in [
                ("weight", ((8, 64, 16), mx.uint32)),
                ("scales", ((8, 64, 2), mx.bfloat16)),
                ("biases", ((8, 64, 2), mx.bfloat16)),
            ]
        },
        **{key: value for key, value in affine.items() if key.startswith("down_proj")},
    }
    four, eight = AffineScheme(4, 64), AffineScheme(8, 64)
    for schemes, expected in [
        (ExpertSchemes.uniform(four), affine),
        (ExpertSchemes.uniform(None), bf16),
        (ExpertSchemes(four, eight, four), split),
        (ExpertSchemes(four, four, four, is_gate_up_apart=True), apart),
    ]:
        layer = MoELayer(8, 2, 128, 64, schemes=schemes, dtype=mx.bfloat16)
        parameters = {
            key: (value.shape, value.dtype)
            for key, value in dict(tree_flatten(layer.parameters())).items()
        }
        assert parameters == expected
        layout = moe_layout(128, 64, schemes)
        assert list(layout) == list(expected)
        assert {role: ((8, *spec.shape), spec.dtype) for role, spec in layout.items()} == expected


def test_offload_layer_holds_its_resident_arrays_alone() -> None:
    four, nvfp4 = AffineScheme(4, 64), FpBlockScheme("nvfp4")
    exported = FpBlockScheme("nvfp4", has_output_scale=True)
    for schemes, expected in [
        (ExpertSchemes.uniform(None), {}),
        (ExpertSchemes(None, four, four), {}),
        (ExpertSchemes.uniform(nvfp4), {}),
        (
            ExpertSchemes.uniform(exported),
            {
                "gate_up_proj.output_scale": ((8, 2, 1), mx.float32),
                "down_proj.output_scale": ((8, 1, 1), mx.float32),
            },
        ),
        (
            ExpertSchemes(exported, exported, exported, is_gate_up_apart=True),
            {
                "gate_up_proj.gate_proj.output_scale": ((8, 1, 1), mx.float32),
                "gate_up_proj.up_proj.output_scale": ((8, 1, 1), mx.float32),
                "down_proj.output_scale": ((8, 1, 1), mx.float32),
            },
        ),
    ]:
        layer = OffloadMoELayer(0, 8, 2, 128, 64, schemes=schemes)
        parameters = {
            key: (value.shape, value.dtype)
            for key, value in dict(tree_flatten(layer.parameters())).items()
        }
        assert parameters == expected


def test_offload_refuses_float_inputs_off_its_tile() -> None:
    # nvfp4 takes 80 inputs (five groups of 16), the offload kernel whole tiles of 32.
    schemes = ExpertSchemes.uniform(FpBlockScheme("nvfp4"))
    MoELayer(8, 2, 80, 64, schemes=schemes)
    with pytest.raises(ValueError, match=re.escape("not gate_up_proj (80) (--moe-strategy")):
        OffloadMoELayer(0, 8, 2, 80, 64, schemes=schemes)


def test_moe_layer_refuses_mixed_quant_kinds() -> None:
    four, nvfp4 = AffineScheme(4, 64), FpBlockScheme("nvfp4")
    for schemes, mixed in [
        (ExpertSchemes(four, four, nvfp4), "gate_up_proj AffineScheme, down_proj FpBlockScheme"),
        (
            ExpertSchemes(nvfp4, four, None),
            "gate_up_proj.gate_proj FpBlockScheme, gate_up_proj.up_proj AffineScheme",
        ),
    ]:
        message = f"quantized experts are of one kind, not {mixed}"
        with pytest.raises(ValueError, match=re.escape(message) + "$"):
            MoELayer(8, 2, 128, 64, schemes=schemes)


@pytest.mark.skipif(MODEL_PATH is None, reason="FTMETAL_TEST_MODEL_PATH is not set")
def test_real_checkpoint_rows() -> None:
    assert MODEL_PATH is not None
    config = EngineConfig(model_path=MODEL_PATH)
    model_config = config.model_config
    rules = model_weight_rules(config.architecture, model_config)
    checkpoint = read_checkpoint(config.architecture, MODEL_PATH)
    # The engine's offload layout: gate and up apart in every layer once one keeps them apart.
    is_apart = any(rules.is_expert(module) for module in rules.unfused)
    method_layouts = [
        moe_layout(
            model_config.hidden_size,
            model_config.moe_intermediate_size,
            ExpertSchemes(
                *schemes_of(
                    model_config,
                    f"model.layers.{layer_id}.mlp.experts",
                    (*GATE_UP_PARTS, "down_proj"),
                ),
                is_gate_up_apart=is_apart,
            ),
        )
        for layer_id in range(model_config.num_moe_layers)
    ]
    layouts, resident = bank_layouts(method_layouts)
    source = ExpertSource(MODEL_PATH, rules, checkpoint.state, checkpoint.expert_bytes, resident)
    top_k = model_config.num_experts_per_tok
    cache = OffloadMoeCache(
        slot_layout=plan_slot_layout(
            source.num_layers * top_k,
            layouts,
            source.num_experts,
            top_k,
            grouping="unified",
            counts=None,
            prefill_overlap=False,
            buffer_placement="overlay",
        ),
        source=source,
        prefill_overlap=False,
        prefill_overlap_min_tokens=128,
        victim_cache_bytes=0,
        pairs=None,
        split_lookup=False,
    )
    if is_apart:
        apart = {f"{prefix}.experts.gate_up_proj" for prefix in source.layer_prefixes}
        rules = replace(rules, unfused=rules.unfused | apart)
    reference = apply_rules(checkpoint.state, rules)
    checked = [(0, [0, 1, 2, source.num_experts - 1])]
    checked += [(layer_id, [17, 18, 100]) for layer_id in range(1, source.num_layers)]
    for layer_id, expert_ids in checked:
        # One layer's experts at a time: the whole checkpoint is tens of GiB.
        expected = expected_rows(cache, reference, source.layer_prefixes, [layer_id])
        slots = cache.ensure_experts(layer_id, expert_ids, 1)
        check_rows(
            cache,
            expected,
            {
                (layer_id, expert): slot_offsets(cache, layer_id, slot)
                for expert, slot in zip(expert_ids, slots)
            },
        )
