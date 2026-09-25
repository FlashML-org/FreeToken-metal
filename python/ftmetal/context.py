"""The global Context: engine-lifetime objects the layers read at depth.

Port of FreeToken core.py (Context, set_global_ctx, get_global_ctx), in its own
module because core.py cannot import the backends. Divergences: the pools and the
attention backends are one per AttnType (kv_cache and linear_state_pool are views of
the pools); the batch is not here (it is passed to BaseLLMModel.forward) and neither
is the page table (the CacheManager owns it).
"""

from __future__ import annotations

from dataclasses import dataclass

from ftmetal.attention.base import AttnType, BaseAttnBackend
from ftmetal.kvcache.base import BaseKVCachePool, BasePool
from ftmetal.kvcache.linear_state_pool import LinearStatePool


@dataclass
class Context:
    page_size: int
    pools: dict[AttnType, BasePool]
    attn_backends: dict[AttnType, BaseAttnBackend]

    @property
    def kv_cache(self) -> BaseKVCachePool:
        pool = self.pools[AttnType.FULL]
        assert isinstance(pool, BaseKVCachePool)
        return pool

    @property
    def linear_state_pool(self) -> LinearStatePool | None:
        pool = self.pools.get(AttnType.LINEAR)
        assert pool is None or isinstance(pool, LinearStatePool)
        return pool


GLOBAL_CTX: Context | None = None


def set_global_ctx(ctx: Context) -> None:
    """One engine per process; a new engine replaces the previous context (tests build several)."""
    global GLOBAL_CTX
    GLOBAL_CTX = ctx


def get_global_ctx() -> Context:
    assert GLOBAL_CTX is not None, "global context is not set"
    return GLOBAL_CTX


__all__ = ["Context", "get_global_ctx", "set_global_ctx"]
