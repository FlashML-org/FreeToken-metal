"""Attention backends, one package per attention type, and the backend registry.

Port of FreeToken attention/__init__.py. Divergences: creators take the group's spec,
its pool and the page table (no global context at construction); LINEAR always gets
LinearAttnBackend, which only builds metadata (FreeToken's backend-driven type); there
is no prefill,decode pair and no auto resolution: two kernel sets on one machine, chosen
by name, each serving the paged types (FULL, SWA, QSA) with one instance per type.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np

from ftmetal.attention.metadata import AttnType
from ftmetal.utils import Registry

# The package initialises before core (which imports attention.metadata), and before
# models.config and kvcache (which import it too), so the backend base class, the spec and
# the pool are type-only imports here and creators import lazily.
if TYPE_CHECKING:
    from ftmetal.attention.base import BaseAttnBackend
    from ftmetal.kvcache.base import BasePool
    from ftmetal.models.config import KVCacheGroupSpec, ModelConfig


class BackendCreator(Protocol):
    def __call__(
        self,
        config: ModelConfig,
        spec: KVCacheGroupSpec,
        pool: BasePool,
        page_table: np.ndarray,
        page_size: int,
    ) -> BaseAttnBackend: ...


@dataclass(frozen=True)
class BackendInfo:
    """Declarative capability entry for one backend; registration never imports kernels."""

    supported_types: frozenset[AttnType]
    # Allowed page sizes (None: any); a mismatch is a config error, not a coercion.
    page_sizes: tuple[int, ...] | None = None
    # Decode runs at the fixed shapes a CommandGraph replays (no per-step host value
    # shapes the kernels), so the engine may capture it.
    can_capture: bool = False


SUPPORTED_ATTENTION_BACKENDS = Registry[BackendCreator]("Attention Backend")


@SUPPORTED_ATTENTION_BACKENDS.register(
    "mlx", BackendInfo(supported_types=frozenset({AttnType.FULL, AttnType.SWA, AttnType.QSA}))
)
def create_mlx_backend(
    config: ModelConfig,
    spec: KVCacheGroupSpec,
    pool: BasePool,
    page_table: np.ndarray,
    page_size: int,
) -> BaseAttnBackend:
    from ftmetal.attention.full.mlx import FullAttnMLXBackend
    from ftmetal.attention.qsa.mlx import QSAMLXBackend

    if spec.attn_type is AttnType.QSA:
        return QSAMLXBackend(config, spec, pool, page_table, page_size)
    else:
        return FullAttnMLXBackend(config, spec, pool, page_table, page_size)


@SUPPORTED_ATTENTION_BACKENDS.register(
    "metal",
    BackendInfo(
        supported_types=frozenset({AttnType.FULL, AttnType.SWA, AttnType.QSA}),
        page_sizes=(1, 2, 4, 8, 16, 32, 64),
        can_capture=True,
    ),
)
def create_metal_backend(
    config: ModelConfig,
    spec: KVCacheGroupSpec,
    pool: BasePool,
    page_table: np.ndarray,
    page_size: int,
) -> BaseAttnBackend:
    from ftmetal.attention.full.metal import FullAttnMetalBackend
    from ftmetal.attention.qsa.metal import QSAMetalBackend

    if spec.attn_type is AttnType.QSA:
        return QSAMetalBackend(config, spec, pool, page_table, page_size)
    else:
        return FullAttnMetalBackend(config, spec, pool, page_table, page_size)


def attention_backend_info(name: str) -> BackendInfo:
    info = SUPPORTED_ATTENTION_BACKENDS.info(name)
    assert isinstance(info, BackendInfo)
    return info


def validate_attn_backend(backend: str) -> str:
    SUPPORTED_ATTENTION_BACKENDS.assert_supported(backend)
    return backend


def check_attention_backend(
    backend: str, page_size: int, required: frozenset[AttnType]
) -> BackendInfo:
    """Config-time type x page-size capability check; the page-size rule is the backend's
    (its kernels'), never the engine's or the cache's, so it is declared in BackendInfo."""
    validate_attn_backend(backend)
    info = attention_backend_info(backend)
    missing = required - info.supported_types
    if missing:
        raise ValueError(
            f"attention backend {backend!r} does not serve {sorted(t.value for t in missing)}"
        )
    if info.page_sizes is not None and page_size not in info.page_sizes:
        raise ValueError(
            f"attention backend {backend!r} supports page sizes {info.page_sizes}, got {page_size}"
        )
    return info


def create_attention_backend(
    backend: str,
    config: ModelConfig,
    spec: KVCacheGroupSpec,
    pool: BasePool,
    page_table: np.ndarray,
    page_size: int,
) -> BaseAttnBackend:
    """The backend for spec's type: LINEAR's metadata builder, else the named backend after
    the capability check."""
    if spec.attn_type is AttnType.LINEAR:
        from ftmetal.attention.linear import LinearAttnBackend

        return LinearAttnBackend(pool, page_size)
    else:
        check_attention_backend(backend, page_size, frozenset({spec.attn_type}))
        return SUPPORTED_ATTENTION_BACKENDS[backend](config, spec, pool, page_table, page_size)


__all__ = [
    "BackendCreator",
    "BackendInfo",
    "SUPPORTED_ATTENTION_BACKENDS",
    "attention_backend_info",
    "check_attention_backend",
    "create_attention_backend",
    "validate_attn_backend",
]
