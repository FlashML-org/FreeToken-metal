"""The global Context: engine-lifetime objects the layers read at depth.

Port of FreeToken core.py (Context, set_global_ctx, get_global_ctx), in its own
module because core.py cannot import the backends. Divergences: the attention
backends are one per AttnType; the batch is not here (it is passed to
BaseLLMModel.forward) and neither is the page table (the CacheManager owns it).
"""

from __future__ import annotations

from dataclasses import dataclass

from ftmetal.attention.base import AttnType, BaseAttnBackend
from ftmetal.kvcache.base import BaseKVCachePool
from ftmetal.kvcache.linear_state_pool import LinearStatePool


@dataclass
class Context:
    page_size: int
    attn_backends: dict[AttnType, BaseAttnBackend]
    kv_cache: BaseKVCachePool
    linear_state_pool: LinearStatePool | None = None


GLOBAL_CTX: Context | None = None


def set_global_ctx(ctx: Context) -> None:
    """One engine per process; a new engine replaces the previous context (tests build several)."""
    global GLOBAL_CTX
    GLOBAL_CTX = ctx


def get_global_ctx() -> Context:
    assert GLOBAL_CTX is not None, "global context is not set"
    return GLOBAL_CTX


__all__ = ["Context", "get_global_ctx", "set_global_ctx"]
