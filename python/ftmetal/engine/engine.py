"""Engine: the model, its pools and one forward step.

Port of FreeToken engine/engine.py. Divergences: CommandGraph replays instead of CUDA
graphs (engine/graph.py), no streams or TP; the page table is a host numpy array the
scheduler owns through the CacheManager; one pool and one attention backend per
attention type, created from the model's group specs, in the global Context (FreeToken
builds one KV family and special-cases the GDN state pool); the forward returns lazy
sampled tokens and the caller issues the step's async_eval; the encoder cache holds
mx.arrays in unified memory. MoE offload: the offload cache reads its experts from the
checkpoint (no host banks, so no --expert-load or CPU and hybrid strategies); auto keeps
the experts resident when they fit (resolve_moe_strategy); the strategy and the cache size
are settled before the model is built, and the size from the measured weights after it;
the cache's health is checked once per forward by the caller, after the forward's tokens
land (check_healthy), not right after its launch.
"""

from __future__ import annotations

import dataclasses
import logging
import time
from pathlib import Path
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
    IN_FLIGHT_STEPS,
    OFFLOAD_IN_FLIGHT_STEPS,
    BudgetReport,
    activation_reserve_bytes,
    choose_moe_strategy,
    encoder_reserve_bytes,
    graph_reserve_bytes,
    kv_reserve_pages,
    memory_budget_bytes,
    plan_budget,
    resolve_moe_cache_size,
    solve_num_pages,
)
from ftmetal.engine.config import EngineConfig
from ftmetal.engine.graph import GraphRunner, determine_graph_bs
from ftmetal.engine.sample import BatchSamplingArgs, Sampler
from ftmetal.kvcache import create_pools, pool_arrays, pool_costs
from ftmetal.kvcache.base import PoolCost
from ftmetal.layers.moe import OffloadMoELayer
from ftmetal.mm.config import ENCODER_SECTIONS
from ftmetal.mm.encoder_cache import EncoderCache
from ftmetal.mm.processor import MMProcessor, get_mm_processor
from ftmetal.models.blocks import BaseLLMModel, SupportsMultimodal
from ftmetal.models.config import MoeStrategy
from ftmetal.models.loader import WeightRules, checkpoint_bytes, load_weights
from ftmetal.models.register import MODEL_REGISTRY, create_model, model_weight_rules
from ftmetal.moe.offload_cache import STAGING_BYTES, OffloadMoeCache
from ftmetal.moe.slot_pools import (
    HOT_EXPERTS_FILE,
    SlotLayout,
    find_expert_counts,
    plan_slot_layout,
    read_expert_counts,
)
from ftmetal.moe.source import ExpertSource

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


def resolve_moe_strategy(
    config: EngineConfig,
    rules: WeightRules,
    *,
    budget: int,
    graphs: int,
    costs: dict[AttnType, PoolCost],
    encoder: int,
) -> MoeStrategy:
    """FreeToken's MoE strategy resolution: an explicit strategy stands; auto offloads when a
    cache size flag is given, else keeps the experts resident when the checkpoint's weights
    leave the paged pools kv_reserve_tokens (choose_moe_strategy)."""
    model_config = config.model_config
    has_cache_flag = (
        config.moe_cache_size_flag(model_config.num_moe_layers * model_config.num_experts) > 0
        or config.moe_cache_auto
    )
    if not model_config.is_moe and config.moe_strategy == "offload":
        raise ValueError(f"--moe-strategy offload: {config.model_path} routes no experts")
    elif not model_config.is_moe:
        return "fused"
    elif config.moe_strategy == "fused":
        if has_cache_flag:
            logger.warning(
                "--moe-strategy fused keeps the experts resident: the MoE cache size flags are "
                "ignored (--moe-strategy offload serves the experts from a slot cache)"
            )
        return "fused"
    elif config.moe_strategy == "offload" or has_cache_flag:
        return "offload"
    else:
        checkpoint = checkpoint_bytes(config.model_path, rules)
        resident = BudgetReport(
            budget=budget,
            weights=checkpoint.non_expert + checkpoint.experts,
            activations=activation_reserve_bytes(
                config.max_extend_tokens, model_config.num_qo_heads, IN_FLIGHT_STEPS
            ),
            graphs=graphs,
            costs=costs,
            encoder=encoder,
        )
        strategy = choose_moe_strategy(
            resident, kv_reserve_pages(config.kv_reserve_tokens, config.page_size)
        )
        logger.info(
            f"Auto-selected MoE strategy: {strategy} ({checkpoint.experts / GIB:.2f} GiB of "
            f"experts beside {config.kv_reserve_tokens} tokens of KV; "
            f"{resident.paged_bytes / GIB:.2f} GiB left for the KV pools if resident)"
        )
        return strategy


def offload_layers(model: BaseLLMModel) -> list[OffloadMoELayer]:
    """The model's offload MoE layers in MoE layer order."""
    layers = sorted(
        (module for module in model.modules() if isinstance(module, OffloadMoELayer)),
        key=lambda layer: layer.moe_layer_id,
    )
    assert [layer.moe_layer_id for layer in layers] == list(range(len(layers))), (
        "the MoE layer indices run from 0"
    )
    return layers


class Engine:
    def __init__(self, config: EngineConfig) -> None:
        self.config = config
        self.dtype = config.dtype
        # Fail on a page size the backend's kernels reject before any pool is allocated.
        check_attention_backend(config.attention_backend, config.page_size)

        # The image processor of a served vision tower: the warm-up's dummy image and the
        # largest image the encoder's reserve covers.
        self.mm_processor: MMProcessor | None = None
        self.encoder_cache: EncoderCache | None = None
        encoder_reserve = 0
        vision_config = config.model_config.vision_config
        if vision_config is not None:
            self.mm_processor = get_mm_processor(config.model_path, config.mm)
            assert self.mm_processor is not None, "a built vision tower has a processor"
            self.encoder_cache = EncoderCache()
            feature = self.mm_processor.dummy_items()[0].feature
            assert feature is not None
            vision_encoder = next(
                encoder
                for encoder in MODEL_REGISTRY[config.architecture].encoders
                if encoder.kind == "vision"
            )
            encoder_reserve = encoder_reserve_bytes(
                vision_encoder,
                vision_config,
                config.dtype,
                self.mm_processor.max_patches,
                feature[0].nbytes,
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
        budget = memory_budget_bytes(working_set, config.memory_ratio)
        specs = config.model_config.kv_cache_group_specs()
        costs = pool_costs(specs, config)
        rules = model_weight_rules(config.architecture, config.model_config)
        moe_strategy = resolve_moe_strategy(
            config,
            rules,
            budget=budget,
            graphs=graph_reserve,
            costs=costs,
            encoder=encoder_reserve,
        )
        is_offload = moe_strategy == "offload"
        # The offload cache's layers share one set of banks.
        is_expert_gate_up_apart = is_offload and any(
            rules.is_expert(module) for module in rules.unfused
        )
        model_config = dataclasses.replace(
            config.model_config,
            moe_strategy=moe_strategy,
            is_expert_gate_up_apart=is_expert_gate_up_apart,
        )
        self.model_config = model_config
        # Prefill overlap reads layer L + 1 into one prefill buffer while layer L reads the other.
        prefill_overlap = (
            is_offload and config.moe_prefill_overlap and model_config.num_moe_layers >= 2
        )
        in_flight_steps = OFFLOAD_IN_FLIGHT_STEPS if prefill_overlap else IN_FLIGHT_STEPS
        activations = activation_reserve_bytes(
            config.max_extend_tokens, model_config.num_qo_heads, in_flight_steps
        )

        # Delta, not the total: an earlier engine's pools may still be alive in the process.
        resident_before = mx.get_active_memory()
        self.model = create_model(config.architecture, model_config, config.dtype)
        load_weights(self.model, config.model_path, rules, skip_experts=is_offload)
        # Under offload the experts are the cache's, not weights.
        self.weights_bytes = mx.get_active_memory() - resident_before
        logger.info(f"weights resident: {self.weights_bytes / 2**30:.2f} GiB")

        moe_layers: list[OffloadMoELayer] = []
        expert_source: ExpertSource | None = None
        expert_counts: np.ndarray | None = None
        slot_layout: SlotLayout | None = None
        cache_size = 0
        experts_bytes = 0
        if is_offload:
            moe_layers = offload_layers(self.model)
            layouts = [layer.layout() for layer in moe_layers]
            expert_source = ExpertSource(config.model_path, rules)
            if config.moe_pool_slots == "profile":
                if config.moe_pool_profile is not None:
                    table_path = Path(config.moe_pool_profile)
                else:
                    table_path = find_expert_counts(
                        config.model_path,
                        model_config.model_type,
                        len(moe_layers),
                        model_config.num_experts,
                        model_config.hidden_size,
                    )
                if table_path is None:
                    raise ValueError(
                        f"--moe-pool-slots profile: no hot-expert table for this model (no "
                        f"{HOT_EXPERTS_FILE} in its directory, none shipped); pass "
                        f"profile:<path> or equal"
                    )
                expert_counts = read_expert_counts(
                    table_path, len(moe_layers), model_config.num_experts
                )
                logger.info(f"MoE offload: pool slots from the hot-expert table {table_path}")
            total_experts = len(moe_layers) * model_config.num_experts
            cache_size = resolve_moe_cache_size(
                config.moe_cache_size_flag(total_experts),
                report=BudgetReport(
                    budget=budget,
                    weights=self.weights_bytes,
                    activations=activations,
                    graphs=graph_reserve,
                    costs=costs,
                    encoder=encoder_reserve,
                ),
                reserve_pages=kv_reserve_pages(config.kv_reserve_tokens, config.page_size),
                # The widest row of each bank: the banks hold at most this much per slot.
                slot_bytes=sum(
                    max(layout[role].nbytes for layout in layouts) for role in layouts[0]
                ),
                staging_bytes=STAGING_BYTES,
                num_layers=len(moe_layers),
                num_experts=model_config.num_experts,
                top_k=model_config.num_experts_per_tok,
                total_experts=total_experts,
                prefill_overlap=prefill_overlap,
                buffer_placement=config.moe_prefill_buffer,
            )
            slot_layout = plan_slot_layout(
                cache_size,
                layouts,
                model_config.num_experts,
                model_config.num_experts_per_tok,
                grouping=config.moe_pool,
                counts=expert_counts,
                prefill_overlap=prefill_overlap,
                buffer_placement=config.moe_prefill_buffer,
            )
            experts_bytes = sum(slot_layout.bank_bytes.values()) + STAGING_BYTES
            overlap_note = (
                f"prefills of {config.moe_prefill_overlap_min_tokens} tokens or more read "
                f"each layer ahead"
                if prefill_overlap
                else "no prefill overlap"
            )
            logger.info(
                f"MoE offload: {cache_size} expert slots of {total_experts} "
                f"({cache_size / total_experts:.2f}), {config.moe_cache_policy}; {overlap_note}"
            )
        report = plan_budget(
            budget,
            self.weights_bytes,
            activations,
            graph_reserve,
            costs,
            encoder_reserve,
            experts_bytes,
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
        self.moe_offload_cache: OffloadMoeCache | None = None
        if expert_source is not None and slot_layout is not None:
            started = time.perf_counter()
            self.moe_offload_cache = OffloadMoeCache(
                slot_layout=slot_layout,
                source=expert_source,
                prefill_overlap=prefill_overlap,
                prefill_overlap_min_tokens=config.moe_prefill_overlap_min_tokens,
                expert_counts=expert_counts,
            )
            pool_slots = [pool.num_slots for pool in slot_layout.pools]
            logger.info(
                f"MoE offload: {config.moe_pool} pools x{len(pool_slots)} of {min(pool_slots)}-"
                f"{max(pool_slots)} slots ({config.moe_pool_slots}), prefill buffers of "
                f"{slot_layout.buffer_size} ({config.moe_prefill_buffer}), decode rows <= "
                f"{slot_layout.max_decode_rows}, looked-up prefill tokens <= "
                f"{slot_layout.max_lookup_tokens}"
            )
            self.moe_offload_cache.fill_initial()
            self.moe_offload_cache.probe()
            logger.info(
                f"MoE offload cache filled in {time.perf_counter() - started:.1f} s; the "
                f"coordinator answers"
            )
        self.ctx = Context(
            page_size=config.page_size,
            pools=self.pools,
            attn_backends=self.attn_backends,
            moe_offload_cache=self.moe_offload_cache,
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
            self.moe_offload_cache,
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
        if self.moe_offload_cache is not None:
            self.moe_offload_cache.prepare(batch)

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
        cache = self.moe_offload_cache
        if cache is not None and cache.mode != "overlap":
            cache.arm()
        if self.graph_runner.can_replay(batch):
            logits = self.graph_runner.replay(batch)
        else:
            logits = self.model.forward(batch)[: batch.size]
        for req in batch.reqs:
            req.complete_one()
        return ForwardOutput(self.sampler.sample(logits, args))

    def check_healthy(self) -> None:
        """Raise the offload cache's first failure (FreeToken raise_if_unhealthy): a forward
        whose lookups failed computed with stand-in experts, so its tokens must not be used."""
        if self.moe_offload_cache is not None:
            self.moe_offload_cache.check_healthy()

    def pool_arrays(self) -> list[mx.array]:
        """The arrays a step updates in place: the pools and the offload cache's tables."""
        tables = [] if self.moe_offload_cache is None else self.moe_offload_cache.table_arrays()
        return [*pool_arrays(self.pools), *tables]


__all__ = [
    "PAGE_TABLE_ALIGN",
    "Engine",
    "ForwardOutput",
    "offload_layers",
    "page_table_width",
    "resolve_moe_strategy",
]
