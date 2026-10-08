"""Engine: the model, its pools and one forward step.

Port of FreeToken engine/engine.py. Divergences: CommandGraph replays instead of CUDA
graphs (engine/graph.py), no streams or TP; the page table is a host numpy array the
scheduler owns through the CacheManager; one pool and one attention backend per
attention type, created from the model's group specs, in the global Context (FreeToken
builds one KV family and special-cases the GDN state pool); the forward returns lazy
sampled tokens and the caller issues the step's async_eval; the encoder cache holds
mx.arrays in unified memory.
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
    encoder_reserve_bytes,
    graph_reserve_bytes,
    memory_budget_bytes,
    plan_budget,
    solve_num_pages,
)
from ftmetal.engine.config import EngineConfig
from ftmetal.engine.graph import GraphRunner, determine_graph_bs
from ftmetal.engine.sample import BatchSamplingArgs, Sampler
from ftmetal.kvcache import create_pools, pool_arrays, pool_costs
from ftmetal.mm.config import ENCODER_SECTIONS
from ftmetal.mm.encoder_cache import EncoderCache
from ftmetal.mm.processor import MMProcessor, get_mm_processor
from ftmetal.models.blocks import SupportsMultimodal
from ftmetal.models.loader import load_weights
from ftmetal.models.register import MODEL_REGISTRY, create_model, model_weight_rules

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
        # The image processor of a served vision tower: the warm-up's dummy image and the
        # largest image the encoder's reserve covers.
        self.mm_processor: MMProcessor | None = None
        self.encoder_cache: EncoderCache | None = None
        encoder_reserve = 0
        if model_config.vision_config is not None:
            self.mm_processor = get_mm_processor(config.model_path, config.mm)
            assert self.mm_processor is not None, "a built vision tower has a processor"
            self.encoder_cache = EncoderCache()
            encoder_reserve = encoder_reserve_bytes(
                model_config.vision_config, config.dtype, self.mm_processor.max_patches
            )
        elif any(config.raw_config.get(key) is not None for key in ENCODER_SECTIONS):
            configured = [
                encoder
                for encoder in MODEL_REGISTRY[config.architecture].encoders
                if config.raw_config.get(encoder.config_key) is not None
            ]
            if not configured:
                reason = "no encoder registered for this architecture"
            elif all(encoder.kind in config.mm.disabled_encoders for encoder in configured):
                reason = "--text-model-only" if config.mm.text_model_only else "--mm-disable"
            else:
                prefixes = ", ".join(encoder.weight_prefix for encoder in configured)
                reason = f"the checkpoint has no tower weights under {prefixes}"
            logger.info(f"Multimodal disabled: {reason}")

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
        report = plan_budget(
            budget, self.weights_bytes, activations, graph_reserve, costs, encoder_reserve
        )
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
            model_config.is_mrope,
        )
        logger.info(
            f"memory after graphs: {mx.get_active_memory() / GIB:.2f} GiB, "
            f"peak {mx.get_peak_memory() / GIB:.2f} GiB"
        )
        if self.mm_processor is not None:
            self.warmup_encoder(self.mm_processor)

    def warmup_encoder(self, mm_processor: MMProcessor) -> None:
        """Encode the processor's dummy items once, so the first image pays no kernel
        compilation."""
        assert isinstance(self.model, SupportsMultimodal)
        mx.eval(*[self.model.encode(item) for item in mm_processor.dummy_items()])

    def prepare_metadata(self, batch: Batch) -> None:
        for backend in self.attn_backends.values():
            backend.prepare_metadata(batch)

    def run_mm_encoder(self, batch: Batch) -> None:
        """Encode the chunk's cache-miss images and gather its embedding rows into
        batch.mm_embeds, right before the LM forward."""
        cache = self.encoder_cache
        assert cache is not None and isinstance(self.model, SupportsMultimodal)
        # A job whose rows are not gathered this chunk would be encoded and freed unread.
        planned = {row.item_hash for row in batch.mm_gather_plan}
        orphans = [item.hash for item in batch.mm_encoder_jobs if item.hash not in planned]
        assert not orphans, f"encoder jobs without gather rows: {orphans}"
        for item in batch.mm_encoder_jobs:
            if not cache.has(item.hash):
                cache.put(item.hash, self.model.encode(item))
            # Cached now: the feature buffer is read no more.
            item.feature = None
        parts: list[mx.array] = []
        for row in batch.mm_gather_plan:
            parts.append(cache.get_slice(row.item_hash, row.row_lo, row.row_hi))
            cache.consume(row.item_hash, row.uid, row.row_hi - row.row_lo)
        batch.mm_embeds = mx.concatenate([part.astype(self.dtype) for part in parts])

    def forward_batch(self, batch: Batch, args: BatchSamplingArgs) -> ForwardOutput:
        if batch.mm_gather_plan:
            self.run_mm_encoder(batch)
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
