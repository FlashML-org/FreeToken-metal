"""Engine: the model, its pools and one forward step.

Port of FreeToken engine/engine.py. Divergences: CommandGraph replays instead of CUDA
graphs (engine/graph.py), no streams or TP; the page table is a host numpy array the
scheduler owns through the CacheManager; one pool and one attention backend per
attention type, created from the model's group specs, in the global Context (FreeToken
builds one KV family and special-cases the GDN state pool); the forward returns lazy
sampled tokens and the caller issues the step's async_eval.
"""

from __future__ import annotations

import logging
from typing import NamedTuple

import mlx.core as mx
import numpy as np

from ftmetal.attention import (
    attention_backend_info,
    check_attention_backend,
    create_attention_backend,
)
from ftmetal.attention.base import AttnType, BaseAttnBackend
from ftmetal.context import Context, set_global_ctx
from ftmetal.core import Batch, Req, SamplingParams
from ftmetal.engine.cache_budget import (
    GIB,
    activation_reserve_bytes,
    graph_reserve_bytes,
    memory_budget_bytes,
    plan_budget,
    solve_num_pages,
)
from ftmetal.engine.config import EngineConfig
from ftmetal.engine.graph import GraphRunner, determine_graph_bs
from ftmetal.engine.sample import BatchSamplingArgs, Sampler
from ftmetal.kvcache import create_pools, pool_arrays, pool_costs
from ftmetal.models.loader import load_weights
from ftmetal.models.register import create_model, model_weight_rules

logger = logging.getLogger(__name__)

# Page-table rows are padded to this many tokens.
PAGE_TABLE_ALIGN = 128


def page_table_width(max_seq_len: int, page_size: int) -> int:
    """Columns of the page table (FreeToken's _page_table_width): allocate_paged writes whole
    trailing pages, so the row must reach align_ceil(max_seq_len, page_size), then pad."""
    tokens = -(-max_seq_len // page_size) * page_size
    return -(-tokens // PAGE_TABLE_ALIGN) * PAGE_TABLE_ALIGN


class ForwardOutput(NamedTuple):
    next_tokens: mx.array


class Engine:
    def __init__(self, config: EngineConfig) -> None:
        self.config = config
        model_config = config.model_config
        self.model_config = model_config
        self.dtype = config.dtype
        # Fail on a page size the backend's kernels reject before any pool is allocated.
        check_attention_backend(config.attention_backend, config.page_size)

        # Delta, not the total: an earlier engine's pools may still be alive in the process.
        resident_before = mx.get_active_memory()
        self.model = create_model(config.architecture, model_config, config.dtype)
        load_weights(
            self.model, config.model_path, model_weight_rules(config.architecture, model_config)
        )
        self.weights_bytes = mx.get_active_memory() - resident_before
        logger.info(f"weights resident: {self.weights_bytes / 2**30:.2f} GiB")

        can_capture = attention_backend_info(config.attention_backend).can_capture
        if config.command_graph and not can_capture:
            logger.warning(
                f"CommandGraph needs a capturable attention backend, not "
                f"{config.attention_backend!r}: decode runs eagerly"
            )
        use_graphs = config.command_graph and can_capture
        graph_bs_list = determine_graph_bs(config.max_running_req) if use_graphs else []
        graph_reserve = graph_reserve_bytes(max(graph_bs_list)) if graph_bs_list else 0

        working_set = mx.device_info()["max_recommended_working_set_size"]
        assert isinstance(working_set, int)
        budget = memory_budget_bytes(working_set)
        specs = model_config.kv_cache_group_specs()
        activations = activation_reserve_bytes(config.max_extend_tokens, model_config.num_qo_heads)
        costs = pool_costs(specs, config)
        report = plan_budget(budget, self.weights_bytes, activations, graph_reserve, costs)
        for line in report.lines():
            logger.info(line)
        self.num_pages = solve_num_pages(report, config.page_size, config.num_page_override)
        self.pools = create_pools(specs, config, self.num_pages)

        num_tokens = self.num_pages * config.page_size
        self.max_seq_len = min(config.max_seq_len, num_tokens)
        self.aligned_max_seq_len = page_table_width(self.max_seq_len, config.page_size)
        # + 1: the dummy request's row, every entry on the dummy page.
        self.page_table = np.zeros(
            (config.max_running_req + 1, self.aligned_max_seq_len), dtype=np.int32
        )
        self.dummy_req = Req(
            input_ids=np.zeros(1, dtype=np.int32),
            table_idx=config.max_running_req,
            cached_len=0,
            output_len=1,
            uid=-1,
            sampling_params=SamplingParams(),
        )
        self.page_table[self.dummy_req.table_idx] = num_tokens

        self.attn_backends: dict[AttnType, BaseAttnBackend] = {
            spec.attn_type: create_attention_backend(
                config.attention_backend,
                model_config,
                spec,
                self.pools[spec.attn_type],
                self.page_table,
                config.page_size,
            )
            for spec in specs
        }
        self.sampler = Sampler(model_config.vocab_size)
        self.ctx = Context(
            page_size=config.page_size, pools=self.pools, attn_backends=self.attn_backends
        )
        set_global_ctx(self.ctx)
        mx.set_wired_limit(budget)
        logger.info(f"memory after init: {mx.get_active_memory() / GIB:.2f} GiB")

        # Padded decode rows run on the dummy page and the GDN padding slot.
        linear_state_pool = self.ctx.linear_state_pool
        if linear_state_pool is not None:
            self.dummy_req.linear_slot_idx = linear_state_pool.padding_slot
        self.graph_runner = GraphRunner(
            self.model,
            self.attn_backends,
            self.pools,
            self.dummy_req,
            graph_bs_list,
            self.aligned_max_seq_len,
            graph_reserve,
        )
        logger.info(
            f"memory after graphs: {mx.get_active_memory() / GIB:.2f} GiB, "
            f"peak {mx.get_peak_memory() / GIB:.2f} GiB"
        )

    def prepare_metadata(self, batch: Batch) -> None:
        for backend in self.attn_backends.values():
            backend.prepare_metadata(batch)

    def forward_batch(self, batch: Batch, args: BatchSamplingArgs) -> ForwardOutput:
        if self.graph_runner.can_replay(batch):
            logits = self.graph_runner.replay(batch)
        else:
            logits = self.model.forward(batch)[: batch.size]
        for req in batch.reqs:
            req.complete_one()
        return ForwardOutput(self.sampler.sample(logits, args))

    def pool_arrays(self) -> list[mx.array]:
        return pool_arrays(self.pools)


__all__ = ["PAGE_TABLE_ALIGN", "Engine", "ForwardOutput", "page_table_width"]
