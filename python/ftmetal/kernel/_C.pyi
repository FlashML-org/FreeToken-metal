"""Type stub of the C++ extension built from kernel/csrc (see bindings.cpp)."""

import mlx.core as mx

Stream = mx.Stream | mx.Device | None

def init_library(source: str) -> None: ...
def init_gdn_step_library(source: str) -> None: ...
def init_fdecode_library(source: str) -> None: ...
def init_fprefill_library(source: str) -> None: ...
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
    num_partitions: int,
    sliding_window: int,
    scale: float,
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
