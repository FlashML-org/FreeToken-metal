"""Engine configuration and the model config it derives from the checkpoint.

Port of FreeToken engine/config.py (EngineConfig), reduced to the knobs this engine
has: one process, one GPU, checkpoints in the MLX layout or rewritten into it. Divergence:
served_modalities reads the encoders from mm.processor.active_encoders (no active_encoders
property) and is empty for a checkpoint without config.json; model_config drops the unbuilt
encoder sections from the raw config dictionary; the MoE strategies are auto, fused and
offload (no host-bank kinds), and no deprecated moe_backend field (only the flag);
memory_ratio defaults to None, the working set less a headroom; the three cache size fields
are checked exclusive here, not only as flags; moe_victim_cache_gib sizes the offload's host
victim cache, which FreeToken does not have.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Literal

import mlx.core as mx

from ftmetal.env import read_env
from ftmetal.mm.config import ENCODER_SECTIONS, Modality, MultimodalConfig
from ftmetal.mm.processor import active_encoders
from ftmetal.models.config import ModelConfig
from ftmetal.models.loader import config_str_list, read_model_config
from ftmetal.models.register import parse_model_config, read_checkpoint
from ftmetal.moe.slot_pools import MoePoolGrouping, MoePoolSlots, PrefillBufferPlacement

# Prefix reuse: radix (the hybrid radix on a model with linear attention) or naive (none).
CacheTypeChoice = Literal["naive", "radix"]
# How routed experts are served: fused keeps them resident, offload reads them into a slot
# cache, auto picks fused when every expert fits the memory budget.
MoeStrategyChoice = Literal["auto", "fused", "offload"]
# From this many prefill tokens on, an offload prefill reads each layer's absent experts ahead
# of it instead of looking them up: at T * top-8 >= 4 * 256 experts every expert averages
# four rows, where the expert-sorted tile kernel beats the per-row matvec in every expert
# format (1.5-2.8x at Qwen3.6's shapes on the M4 Pro).
DEFAULT_MOE_PREFILL_OVERLAP_MIN_TOKENS = 128


def command_graph_from_env() -> bool:
    """FTMETAL_COMMAND_GRAPH: 1 (default) replays decode steps from captured graphs, 0 not."""
    value = read_env("COMMAND_GRAPH")
    if value not in (None, "0", "1"):
        raise ValueError(f"FTMETAL_COMMAND_GRAPH must be 0 or 1, got {value!r}")
    return value != "0"


def moe_prefill_buffer_from_env() -> PrefillBufferPlacement:
    """FTMETAL_MOE_PREFILL_BUFFER: overlay (default) lays the prefill buffers over the first
    pools' slots, dedicated gives them slots of their own."""
    value = read_env("MOE_PREFILL_BUFFER")
    if value is None or value == "overlay":
        return "overlay"
    elif value == "dedicated":
        return "dedicated"
    else:
        raise ValueError(f"FTMETAL_MOE_PREFILL_BUFFER must be overlay or dedicated, got {value!r}")


def moe_pack_from_env() -> bool:
    """FTMETAL_MOE_PACK: 1 (default) reads offloaded experts from the model's expert pack when it
    holds them, 0 from the checkpoint."""
    value = read_env("MOE_PACK")
    if value not in (None, "0", "1"):
        raise ValueError(f"FTMETAL_MOE_PACK must be 0 or 1, got {value!r}")
    return value != "0"


def moe_split_lookup_from_env() -> bool:
    """FTMETAL_MOE_SPLIT_LOOKUP: 1 (default) runs an affine offload decode's resident experts and
    shared expert while its absent experts are read, 0 after."""
    value = read_env("MOE_SPLIT_LOOKUP")
    if value not in (None, "0", "1"):
        raise ValueError(f"FTMETAL_MOE_SPLIT_LOOKUP must be 0 or 1, got {value!r}")
    return value != "0"


def moe_prefill_overlap_min_tokens_from_env() -> int:
    """FTMETAL_MOE_PREFILL_OVERLAP_MIN_TOKENS: the overlap prefill threshold, for tests."""
    value = read_env("MOE_PREFILL_OVERLAP_MIN_TOKENS")
    if value is None:
        return DEFAULT_MOE_PREFILL_OVERLAP_MIN_TOKENS
    elif value.isdigit() and int(value) > 0:
        return int(value)
    else:
        raise ValueError(
            f"FTMETAL_MOE_PREFILL_OVERLAP_MIN_TOKENS must be a positive integer, got {value!r}"
        )


@dataclass(frozen=True)
class EngineConfig:
    model_path: str
    dtype: mx.Dtype = mx.bfloat16
    max_running_req: int = 4
    page_size: int = 1
    # Name in attention.SUPPORTED_ATTENTION_BACKENDS: metal runs the paged flash-prefill and
    # flash-decode kernels, mlx runs MLX SDPA over gathered pages.
    attention_backend: str = "metal"
    max_seq_len_override: int | None = None
    num_page_override: int | None = None
    # Prefill token budget per batch, longer prompts chunked; None: the model's.
    max_extend_tokens_override: int | None = None
    # Decode steps between status log lines.
    decode_log_interval: int = 40
    cache_type: CacheTypeChoice = "radix"
    # GDN snapshot cache of the hybrid radix cache, in slots on top of the running requests'
    # own, at least one conversation's two; None: kvcache.linear_state_pool's default.
    gdn_snapshots: int | None = None
    # Window pool of the SWA radix cache, in sliding windows, at least what the running
    # requests and a prefill hold; None: kvcache.swa_pool's default, that floor and 8 more.
    swa_windows: int | None = None
    # Capture a decode graph per batch-size bucket at startup and replay decode from it.
    command_graph: bool = field(default_factory=command_graph_from_env)
    # Runtime knobs of the multimodal path; the architecture side lives in the checkpoint
    # config.
    mm: MultimodalConfig = field(default_factory=MultimodalConfig)
    moe_strategy: MoeStrategyChoice = "auto"
    # The offload cache's size in expert slots, or as a fraction of every expert, or auto (the
    # KV pools get kv_reserve_tokens first, the experts the rest); at most one is set, and an
    # offload with none is auto. 0 and None: not set.
    moe_cache_size: int = 0
    moe_cache_rate: float | None = None
    moe_cache_auto: bool = False
    # KV tokens auto keeps before the experts: auto's fused fit check and cache sizing.
    kv_reserve_tokens: int = 8192
    moe_cache_policy: Literal["lru"] = "lru"
    # How the offload cache groups the MoE layers into pools of slots, and splits its slots
    # among them: evenly per layer, or by a hot-expert table (moe_pool_profile, else the model
    # directory's, else the one shipped for the model).
    moe_pool: MoePoolGrouping = "unified"
    moe_pool_slots: MoePoolSlots = "equal"
    moe_pool_profile: str | None = None
    # Long prefills read each layer's absent experts ahead into two prefill buffers.
    moe_prefill_overlap: bool = True
    moe_prefill_buffer: PrefillBufferPlacement = field(default_factory=moe_prefill_buffer_from_env)
    moe_prefill_overlap_min_tokens: int = field(
        default_factory=moe_prefill_overlap_min_tokens_from_env
    )
    # Offloaded experts come from the expert pack (ftm pack) when the model directory has one.
    moe_pack: bool = field(default_factory=moe_pack_from_env)
    # An offload decode issues its lookups and computes the resident experts during the reads.
    moe_split_lookup: bool = field(default_factory=moe_split_lookup_from_env)
    # The offload cache's victim cache in host memory beyond Metal's working set, in GiB, or
    # auto (that memory less a host reserve); 0: none.
    moe_victim_cache_gib: float | Literal["auto"] = 0
    # The memory budget as a fraction of Metal's recommended working set; None: the working
    # set less its headroom. A fraction also emulates a smaller Mac.
    memory_ratio: float | None = None

    def __post_init__(self) -> None:
        cache_flags = [
            flag
            for flag, is_set in (
                ("moe_cache_size", self.moe_cache_size != 0),
                ("moe_cache_rate", self.moe_cache_rate is not None),
                ("moe_cache_auto", self.moe_cache_auto),
            )
            if is_set
        ]
        if len(cache_flags) > 1:
            raise ValueError(f"{' and '.join(cache_flags)} are exclusive: set at most one")
        if self.moe_cache_size < 0:
            raise ValueError(f"moe_cache_size must not be negative, got {self.moe_cache_size}")
        if self.moe_cache_rate is not None and not 0 <= self.moe_cache_rate <= 1:
            raise ValueError(f"moe_cache_rate must be in [0, 1], got {self.moe_cache_rate}")
        if self.kv_reserve_tokens < 0:
            raise ValueError(
                f"kv_reserve_tokens must not be negative, got {self.kv_reserve_tokens}"
            )
        if self.memory_ratio is not None and not 0 < self.memory_ratio <= 1:
            raise ValueError(f"memory_ratio must be in (0, 1], got {self.memory_ratio}")
        if self.moe_victim_cache_gib != "auto" and not (
            math.isfinite(self.moe_victim_cache_gib) and self.moe_victim_cache_gib >= 0
        ):
            raise ValueError(
                f"moe_victim_cache_gib must be auto or a finite non-negative number, got "
                f"{self.moe_victim_cache_gib}"
            )

    def moe_cache_size_flag(self, total_experts: int) -> int:
        """The offload cache size the flags give, rate resolved (FreeToken: ceil of the
        fraction of every expert); 0: auto."""
        if self.moe_cache_rate:
            return math.ceil(total_experts * self.moe_cache_rate)
        else:
            return self.moe_cache_size

    @cached_property
    def raw_config(self) -> dict[str, object]:
        return read_model_config(self.model_path)

    @cached_property
    def served_modalities(self) -> frozenset[Modality]:
        """Modalities this process accepts."""
        if not (Path(self.model_path) / "config.json").is_file():
            return frozenset()
        else:
            encoders = active_encoders(self.model_path, self.raw_config, self.mm)
            return frozenset(modality for encoder in encoders for modality in encoder.modalities)

    @cached_property
    def architecture(self) -> str:
        architectures = config_str_list(self.raw_config, "architectures")
        assert architectures, "config.json names no architecture"
        return architectures[0]

    @cached_property
    def model_config(self) -> ModelConfig:
        quant = read_checkpoint(self.architecture, self.model_path).quant
        # The parser sees no section of a tower this process does not build (for the vision
        # tower that also means 1-D rope).
        encoders = active_encoders(self.model_path, self.raw_config, self.mm)
        built = {encoder.config_key for encoder in encoders}
        raw_config = {
            key: value
            for key, value in self.raw_config.items()
            if key in built or key not in ENCODER_SECTIONS
        }
        return parse_model_config(self.architecture, raw_config, quant)

    @property
    def max_seq_len(self) -> int:
        if self.max_seq_len_override is not None:
            return self.max_seq_len_override
        return self.model_config.rotary_config.max_position

    @property
    def max_extend_tokens(self) -> int:
        if self.max_extend_tokens_override is not None:
            return self.max_extend_tokens_override
        return self.model_config.max_extend_tokens

    @property
    def model_name(self) -> str:
        return os.path.basename(os.path.normpath(self.model_path))


__all__ = [
    "DEFAULT_MOE_PREFILL_OVERLAP_MIN_TOKENS",
    "CacheTypeChoice",
    "EngineConfig",
    "MoeStrategyChoice",
    "command_graph_from_env",
    "moe_pack_from_env",
    "moe_prefill_buffer_from_env",
    "moe_prefill_overlap_min_tokens_from_env",
    "moe_split_lookup_from_env",
]
