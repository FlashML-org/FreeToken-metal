"""Type stub of the C++ extension built from kernel/csrc (see bindings.cpp)."""

import mlx.core as mx
import numpy as np
import numpy.typing as npt

Stream = mx.Stream | mx.Device | None

class SlotLru:
    def __init__(
        self,
        num_layers: int,
        num_experts: int,
        num_slots: int,
        window_slots: int,
        slot_of: mx.array,
        last_use: mx.array,
    ) -> None: ...
    def fill(self, flat_ids: list[int]) -> list[int]: ...
    def resolve(
        self, layer_id: int, expert_ids: list[int], step: int
    ) -> tuple[list[int], list[tuple[int, int]]]: ...
    def invalidate_windows(self) -> None: ...
    def prefill_slot_map(self, layer_id: int) -> tuple[list[int], list[tuple[int, int]]]: ...

class ExpertServiceStats:
    @property
    def hits(self) -> list[int]: ...
    @property
    def misses(self) -> list[int]: ...
    @property
    def experts_read(self) -> list[int]: ...
    @property
    def bytes_read(self) -> list[int]: ...
    @property
    def read_wait_ns(self) -> list[int]: ...

class ExpertService:
    def __init__(
        self,
        num_layers: int,
        num_experts: int,
        num_slots: int,
        window_slots: int,
        slot_of: mx.array,
        last_use: mx.array,
        banks: list[mx.array],
        files: list[str],
        segments: npt.NDArray[np.int64],
        io_threads: int,
    ) -> None: ...
    def fill_initial(self, flat_ids: list[int]) -> None: ...
    def ensure_experts(self, layer_id: int, expert_ids: list[int], step: int) -> list[int]: ...
    def begin_prefill(self) -> None: ...
    def prefetch_prefill_layer(self, layer_id: int) -> list[int]: ...
    def wait_prefill_layer(self, layer_id: int) -> None: ...
    def stats(self) -> ExpertServiceStats: ...

def init_library(source: str) -> None: ...
def init_gdn_step_library(source: str) -> None: ...
def init_fdecode_library(source: str) -> None: ...
def init_fprefill_library(source: str) -> None: ...
def init_gather_qmm_sorted_library(source: str) -> None: ...
def init_sample_library(source: str) -> None: ...
def scatter_rows(
    pool: mx.array, rows: mx.array, dst_ids: mx.array, *, stream: Stream = None
) -> mx.array: ...
def copy_rows(
    pool: mx.array, src_ids: mx.array, dst_ids: mx.array, *, stream: Stream = None
) -> mx.array: ...
def zero_rows(pool: mx.array, ids: mx.array, *, stream: Stream = None) -> mx.array: ...
def kv_write(
    k_cache: mx.array,
    v_cache: mx.array,
    k: mx.array,
    v: mx.array,
    out_loc: mx.array,
    *,
    stream: Stream = None,
) -> list[mx.array]: ...
def gdn_step(
    conv_pool: mx.array,
    state_pool: mx.array,
    conv_in: mx.array,
    conv_weight: mx.array,
    a: mx.array,
    b: mx.array,
    A_log: mx.array,
    dt_bias: mx.array,
    slots: mx.array,
    positions: mx.array,
    *,
    stream: Stream = None,
) -> list[mx.array]: ...
def flash_decode(
    q: mx.array,
    k_cache: mx.array,
    v_cache: mx.array,
    page_table: mx.array,
    seq_lens: mx.array,
    page_size: int,
    num_partitions: int,
    partition_tokens: int,
    min_partition_tokens: int,
    max_partitions: int,
    min_threadgroups: int,
    num_head_groups: int,
    num_simdgroups: int,
    scale: float,
    *,
    stream: Stream = None,
) -> mx.array: ...
def flash_prefill(
    q: mx.array,
    k_cache: mx.array,
    v_cache: mx.array,
    page_table: mx.array,
    tiles: mx.array,
    page_size: int,
    num_row_groups: int,
    head_split: int,
    block_k: int,
    kv_shared: bool,
    num_passes: int,
    num_partitions: int,
    sliding_window: int,
    mpp: bool,
    scale: float,
    *,
    stream: Stream = None,
) -> mx.array: ...
def gather_qmm_sorted(
    x: mx.array,
    w: mx.array,
    scales: mx.array,
    biases: mx.array,
    expert_offsets: mx.array,
    slot_of: mx.array,
    group_size: int,
    bits: int,
    *,
    stream: Stream = None,
) -> mx.array: ...
def sample_tokens(
    logits: mx.array,
    temperatures: mx.array,
    top_k: mx.array | None,
    top_p: mx.array | None,
    num_groups: int,
    random_bits: mx.array | None = None,
    *,
    stream: Stream = None,
) -> mx.array: ...
