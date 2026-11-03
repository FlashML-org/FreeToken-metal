"""Type stub of the C++ extension built from the kernel package (see bindings.cpp)."""

import enum

import mlx.core as mx
import numpy as np
import numpy.typing as npt

Stream = mx.Stream | mx.Device | None

MAILBOX_WORDS: int
EXPERT_STAGING_BYTES: int
FILL_FILE: int
QMM_TILE_K: int
PLE_MAILBOX_WORDS: int
MAX_PLE_ROWS: int

class MailboxWord(enum.IntEnum):
    STATE = 0
    SEQUENCE = 1
    ACKNOWLEDGED = 2
    SHUTDOWN = 3
    KIND = 32
    LAYER = 33
    STEP = 34
    NUM_ROWS = 35
    NUM_IDS = 36
    NUM_MISSES = 37
    IDS = 64
    NUM_LOADED = 320
    NUM_EVICTED = 321
    ERROR = 322
    CLAIM_TIME = 324
    DONE_TIME = 326

class MailboxState(enum.IntEnum):
    IDLE = 0
    POSTED = 1
    CLAIMED = 2
    DONE = 3
    FAILED = 4
    SHUTDOWN = 5

class MailboxError(enum.IntEnum):
    NONE = 0
    IO = 1
    TIMEOUT = 2
    PROTOCOL = 3
    SHUTDOWN = 4
    CAPACITY = 5

class MoeLookupKind(enum.IntEnum):
    DECODE = 1
    PREFILL_COPY_MISS = 2
    PROBE = 3

class PleMailboxWord(enum.IntEnum):
    STATE = 0
    SEQUENCE = 1
    ACKNOWLEDGED = 2
    SHUTDOWN = 3
    NUM_ROWS = 32
    ERROR = 33
    ROWS = 64

def moe_lookup(
    topk_ids: mx.array,
    moe_step: mx.array,
    slot_of: mx.array,
    mailbox: mx.array,
    last_use: mx.array,
    hit_stats: mx.array,
    layer_id: int,
    slot_base: int,
    kind: MoeLookupKind,
    *,
    stream: Stream = None,
) -> list[mx.array]: ...

class SlotLru:
    def __init__(
        self,
        num_layers: int,
        num_experts: int,
        buffer_size: int,
        layer_pools: list[int],
        pool_bounds: list[int],
        pool_volatile: list[int],
        slot_of: mx.array,
        last_use: mx.array,
    ) -> None: ...
    @property
    def num_slots(self) -> int: ...
    def fill(self, flat_ids: list[int]) -> list[int]: ...
    def resolve(
        self, layer_id: int, expert_ids: list[int], step: int
    ) -> tuple[list[int], list[tuple[int, int]]]: ...
    def invalidate_prefill_buffers(self) -> list[tuple[int, int]]: ...
    def prefill_slot_map(self, layer_id: int) -> tuple[list[int], list[tuple[int, int]]]: ...

class ExpertServiceStats:
    @property
    def hits(self) -> list[int]: ...
    @property
    def misses(self) -> list[int]: ...
    @property
    def requests(self) -> list[int]: ...
    @property
    def experts_read(self) -> list[int]: ...
    @property
    def bytes_read(self) -> list[int]: ...
    @property
    def read_wait_ns(self) -> list[int]: ...
    @property
    def serve_ns(self) -> list[int]: ...
    @property
    def lookups(self) -> list[int]: ...
    @property
    def lookup_hits(self) -> list[int]: ...
    @property
    def victim_hits(self) -> list[int]: ...
    @property
    def victim_buffer_hits(self) -> list[int]: ...
    @property
    def victim_stores(self) -> list[int]: ...
    @property
    def victim_drops(self) -> list[int]: ...
    @property
    def victim_copy_ns(self) -> int: ...
    @property
    def victim_entries(self) -> int: ...
    @property
    def victim_bytes(self) -> int: ...

class ExpertService:
    def __init__(
        self,
        num_layers: int,
        num_experts: int,
        buffer_size: int,
        layer_pools: list[int],
        pool_bounds: list[int],
        pool_volatile: list[int],
        slot_of: mx.array,
        last_use: mx.array,
        mailbox: mx.array,
        hit_stats: mx.array,
        banks: list[mx.array],
        files: list[str],
        segments: npt.NDArray[np.int64],
        locations: npt.NDArray[np.int64],
        destinations: npt.NDArray[np.int64],
        buffer_bytes: list[int],
        victim_cache_bytes: int,
        io_threads: int,
        stall_timeout_s: float,
        run_coordinator: bool,
    ) -> None: ...
    def fill_initial(self, flat_ids: list[int]) -> None: ...
    def ensure_experts(self, layer_id: int, expert_ids: list[int], step: int) -> list[int]: ...
    def begin_prefill(self) -> None: ...
    def prefetch_prefill_layer(self, layer_id: int) -> list[int]: ...
    def wait_prefill_layer(self, layer_id: int) -> None: ...
    def stats(self) -> ExpertServiceStats: ...
    def arm(self) -> None: ...
    def check_healthy(self) -> None: ...
    def shutdown(self) -> None: ...
    def purge_victim_cache(self) -> None: ...
    def victim_cache_ids(self) -> list[int]: ...

class PleRowStore:
    def __init__(
        self,
        files: list[str],
        parts: npt.NDArray[np.int64],
        shard_starts: list[int],
        row_bytes: list[int],
        cache_rows: int,
        io_threads: int,
    ) -> None: ...
    def gather(
        self,
        ids: npt.NDArray[np.int64],
        weight: npt.NDArray[np.uint8],
        scales: npt.NDArray[np.uint8],
        biases: npt.NDArray[np.uint8],
    ) -> None: ...
    @property
    def num_rows(self) -> int: ...
    @property
    def row_bytes(self) -> list[int]: ...
    @property
    def hits(self) -> int: ...
    @property
    def misses(self) -> int: ...
    @property
    def bytes_read(self) -> int: ...

class PleService:
    def __init__(
        self,
        mailbox: mx.array,
        staging: list[mx.array],
        store: PleRowStore,
        multipliers: list[int],
        head_sizes: list[int],
        head_offsets: list[int],
        eos_token_id: int,
        stall_timeout_s: float,
        run_coordinator: bool,
    ) -> None: ...
    @property
    def num_heads(self) -> int: ...
    def ngram_ids(self, triples: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]: ...
    def arm(self) -> None: ...
    def check_healthy(self) -> None: ...
    def shutdown(self) -> None: ...

def init_pool_rows_library(source: str) -> None: ...
def init_gdn_decode_library(source: str) -> None: ...
def init_mha_decode_library(source: str) -> None: ...
def init_mha_prefill_library(source: str) -> None: ...
def init_gather_sorted_libraries(
    affine_source: str, fp_source: str, unquantized_source: str
) -> None: ...
def init_sample_tokens_library(source: str) -> None: ...
def init_moe_lookup_library(source: str) -> None: ...
def init_ple_lookup_library(source: str) -> None: ...
def ple_post(
    input_ids: mx.array,
    slots: mx.array,
    context: mx.array,
    mailbox: mx.array,
    *,
    stream: Stream = None,
) -> list[mx.array]: ...
def ple_wait(
    ticket: mx.array,
    mailbox: mx.array,
    staging: list[mx.array],
    num_rows: int,
    after: list[mx.array] = ...,
    *,
    stream: Stream = None,
) -> list[mx.array]: ...
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
def gdn_decode(
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
    block_ends: mx.array | None = None,
    *,
    stream: Stream = None,
) -> mx.array: ...
def flash_prefill_qsa(
    q: mx.array,
    k_cache: mx.array,
    v_cache: mx.array,
    page_table: mx.array,
    tiles: mx.array,
    qsa_blocks: mx.array,
    qsa_counts: mx.array,
    qsa_bits: mx.array,
    qsa_block_size: int,
    page_size: int,
    num_row_groups: int,
    head_split: int,
    block_k: int,
    kv_shared: bool,
    num_passes: int,
    num_partitions: int,
    mpp: bool,
    scale: float,
    *,
    stream: Stream = None,
) -> mx.array: ...
def gather_qmm_sorted(
    x: mx.array,
    w: mx.array,
    scales: mx.array,
    biases: mx.array | None,
    expert_offsets: mx.array,
    expert_bytes: mx.array,
    n: int,
    group_size: int,
    bits: int,
    mode: str,
    *,
    stream: Stream = None,
) -> mx.array: ...
def gather_mm_sorted(
    x: mx.array,
    w: mx.array,
    expert_offsets: mx.array,
    expert_bytes: mx.array,
    n: int,
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
