"""Engine: the model, its pools and one forward step.

Port of FreeToken engine/engine.py. Divergences: no CUDA graphs, streams or TP; the
page table is a host numpy array the scheduler owns through the CacheManager; one
attention backend per attention type in the global Context; the forward returns
lazy sampled tokens and the caller issues the step's async_eval.
"""

from __future__ import annotations

import logging
from typing import NamedTuple

import mlx.core as mx
import numpy as np

from ftmetal.attention.base import AttnType, BaseAttnBackend
from ftmetal.attention.full.metal import FullAttnMetalBackend
from ftmetal.attention.full.mlx import FullAttnMLXBackend
from ftmetal.attention.linear import LinearAttnBackend
from ftmetal.context import Context, set_global_ctx
from ftmetal.core import Batch
from ftmetal.engine.cache_budget import (
    activation_reserve_bytes,
    memory_budget_bytes,
    plan_budget,
)
from ftmetal.engine.config import EngineConfig
from ftmetal.engine.sample import BatchSamplingArgs, Sampler
from ftmetal.kvcache.linear_state_pool import (
    LinearStatePool,
    linear_pool_num_slots,
    linear_state_bytes_per_req,
)
from ftmetal.kvcache.mha_pool import MHAKVCache
from ftmetal.models.loader import load_weights
from ftmetal.models.register import create_model, model_weight_rules

logger = logging.getLogger(__name__)

# Page-table rows are padded to this many tokens.
PAGE_TABLE_ALIGN = 128


class ForwardOutput(NamedTuple):
    next_tokens: mx.array


class Engine:
    def __init__(self, config: EngineConfig) -> None:
        self.config = config
        model_config = config.model_config
        self.model_config = model_config
        self.dtype = config.dtype

        self.model = create_model(config.architecture, model_config, config.dtype)
        load_weights(
            self.model, config.model_path, model_weight_rules(config.architecture, model_config)
        )
        self.weights_bytes = mx.get_active_memory()
        logger.info(f"weights resident: {self.weights_bytes / 2**30:.2f} GiB")

        budget = memory_budget_bytes(mx.device_info()["max_recommended_working_set_size"])
        linear_group = model_config.linear_attention_group()
        num_slots = linear_pool_num_slots(config.max_running_req)
        linear_state_bytes = 0
        if linear_group is not None:
            linear_state_bytes = num_slots * linear_state_bytes_per_req(linear_group, config.dtype)
        activations = activation_reserve_bytes(config.max_extend_tokens, model_config.num_qo_heads)
        report = plan_budget(budget, self.weights_bytes, linear_state_bytes, activations)
        for line in report.lines():
            logger.info(line)

        kv_specs = model_config.kv_cache_group_specs()
        assert len(kv_specs) == 1, "one paged-KV group per model"
        self.num_pages = MHAKVCache.solve_num_pages(
            model_config,
            config.page_size,
            config.dtype,
            report.kv_available,
            config.num_page_override,
        )
        # + 1: the dummy page padded page-table entries point at.
        self.kv_cache = MHAKVCache(
            num_kv_heads=kv_specs[0].num_kv_heads,
            num_layers=model_config.num_layers,
            head_dim=kv_specs[0].head_dim,
            num_pages=self.num_pages + 1,
            page_size=config.page_size,
            dtype=config.dtype,
            layer_ids=kv_specs[0].layer_ids,
        )
        self.linear_state_pool: LinearStatePool | None = None
        if linear_group is not None:
            self.linear_state_pool = LinearStatePool(linear_group, num_slots, config.dtype)

        num_tokens = self.num_pages * config.page_size
        self.max_seq_len = min(config.max_seq_len, num_tokens)
        width = -(-self.max_seq_len // PAGE_TABLE_ALIGN) * PAGE_TABLE_ALIGN
        self.page_table = np.zeros((config.max_running_req, width), dtype=np.int32)

        full_attn_backend = {"mlx": FullAttnMLXBackend, "metal": FullAttnMetalBackend}[
            config.attention_backend
        ]
        self.attn_backends: dict[AttnType, BaseAttnBackend] = {
            AttnType.FULL: full_attn_backend(
                model_config, self.kv_cache, self.page_table, config.page_size
            )
        }
        if linear_group is not None:
            self.attn_backends[AttnType.LINEAR] = LinearAttnBackend()
        self.sampler = Sampler(model_config.vocab_size)
        set_global_ctx(
            Context(
                page_size=config.page_size,
                attn_backends=self.attn_backends,
                kv_cache=self.kv_cache,
                linear_state_pool=self.linear_state_pool,
            )
        )
        mx.set_wired_limit(budget)
        logger.info(f"memory after init: {mx.get_active_memory() / 2**30:.2f} GiB")

    def prepare_metadata(self, batch: Batch) -> None:
        for backend in self.attn_backends.values():
            backend.prepare_metadata(batch)

    def forward_batch(self, batch: Batch, args: BatchSamplingArgs) -> ForwardOutput:
        logits = self.model.forward(batch)
        for req in batch.reqs:
            req.complete_one()
        return ForwardOutput(self.sampler.sample(logits, args))

    def pool_arrays(self) -> list[mx.array]:
        arrays = self.kv_cache.arrays()
        if self.linear_state_pool is not None:
            arrays += self.linear_state_pool.arrays()
        return arrays


__all__ = ["PAGE_TABLE_ALIGN", "Engine", "ForwardOutput"]
