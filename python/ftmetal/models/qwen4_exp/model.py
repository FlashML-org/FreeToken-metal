"""Qwen3.8-Flash-Next (qwen4_exp): a hybrid GDN + QSA MoE decoder over hyper-connected
residual streams, with a PLE n-gram branch.

Port of FreeToken models/qwen4_exp/model.py, with the encode of its
Qwen4ExpForConditionalGeneration. Divergences: the batch flows through forward explicitly;
one class serves both architectures and builds the vision tower (channels-last patch conv,
the MLX conversion's) when the config has one; no MTP head; each Linear takes its scheme
from the checkpoint's quantization map at construction; a sublayer's output is injected into
the streams by the next hyper-connection as it norms them (a pending branch), the last
layer's by the final mixer, after the last rows are selected; once the weights load
(FinishesLoading) the PLE tables are opened from the checkpoint files and each
hyper-connection's down and inject projections fused; a decode token's projections run on
kernel/gemm/affine/affine_qmv_rows; the PLE layers' row service is armed before each decode step
and checked once its tokens land (ServesStepLookups).
"""

from __future__ import annotations

import mlx.core as mx

from ftmetal.context import get_global_ctx
from ftmetal.core import Batch
from ftmetal.kernel._ext import _C
from ftmetal.kvcache import pool_arrays
from ftmetal.layers.base import BaseOP
from ftmetal.layers.embedding import Embedding
from ftmetal.layers.gdn import IN_PROJ_BA_PARTS, IN_PROJ_QKVZ_PARTS, GatedDeltaNet
from ftmetal.layers.linear import Linear
from ftmetal.layers.projection import project
from ftmetal.layers.quantization.config import QuantScheme
from ftmetal.layers.quantization.moe import ExpertSchemes
from ftmetal.layers.shared_expert_moe import GATE_UP_PARTS, SharedExpertMoE
from ftmetal.mm.item import MMItem
from ftmetal.models.blocks import BaseLLMModel, embed_input_ids, select_last_rows
from ftmetal.models.config import ModelConfig
from ftmetal.models.qwen3_vl.vision import Qwen3VLVisionModel
from ftmetal.models.qwen4_exp.attention import Qwen4ExpAttention
from ftmetal.models.qwen4_exp.config import Qwen4ExpConfig
from ftmetal.models.qwen4_exp.norm import HyperConnection, PendingBranch, inject
from ftmetal.models.qwen4_exp.ple import Qwen4ExpPLE
from ftmetal.models.qwen4_exp.weight import QKV_PARTS, checkpoint_module

# Decode rows one step's PLE lookup posts; a larger batch looks its rows up on the host.
MAX_PLE_ROWS = _C.MAX_PLE_ROWS


def scheme_of(config: ModelConfig, model_key: str) -> QuantScheme | None:
    if config.quant is None:
        return None
    else:
        return config.quant.scheme_for(checkpoint_module(model_key))


def schemes_of(
    config: ModelConfig, model_prefix: str, parts: tuple[str, ...]
) -> list[QuantScheme | None]:
    """The scheme of each part of a fused module; they fuse only when all agree."""
    if config.quant is None:
        return [None] * len(parts)
    else:
        return config.quant.schemes_for(
            [checkpoint_module(f"{model_prefix}.{part}") for part in parts]
        )


def hyper_connection(
    config: Qwen4ExpConfig, prefix: str, *, with_inject: bool, dtype: mx.Dtype
) -> HyperConnection:
    return HyperConnection(
        config.hc_count,
        config.hidden_size,
        config.hc_lowrank,
        config.rms_norm_eps,
        down_scheme=scheme_of(config, f"{prefix}.input_mix_weight_down"),
        up_scheme=scheme_of(config, f"{prefix}.input_mix_weight_up"),
        inject_scheme=scheme_of(config, f"{prefix}.block_inject_weight"),
        with_inject=with_inject,
        dtype=dtype,
    )


class Qwen4ExpDecoderLayer(BaseOP):
    def __init__(
        self, config: Qwen4ExpConfig, layer_id: int, *, prefix: str, dtype: mx.Dtype
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.is_linear = config.is_linear_layer(layer_id)
        self.has_ple = layer_id in config.ple_layer_ids
        if self.is_linear:
            group = config.linear_attention_group()
            assert group is not None
            gdn_prefix = f"{prefix}.linear_attn"
            self.linear_attn = GatedDeltaNet(
                config.hidden_size,
                group,
                config.rms_norm_eps,
                layer_id,
                qkvz_schemes=schemes_of(config, gdn_prefix, IN_PROJ_QKVZ_PARTS),
                ba_schemes=schemes_of(config, gdn_prefix, IN_PROJ_BA_PARTS),
                out_scheme=scheme_of(config, f"{gdn_prefix}.out_proj"),
                dtype=dtype,
            )
        else:
            attn_prefix = f"{prefix}.self_attn"
            self.self_attn = Qwen4ExpAttention(
                config,
                layer_id,
                qkv_schemes=schemes_of(config, attn_prefix, QKV_PARTS),
                o_scheme=scheme_of(config, f"{attn_prefix}.o_proj"),
                index_scheme=scheme_of(config, f"{attn_prefix}.indexer.index_qk_proj"),
                dtype=dtype,
            )
        if self.has_ple:
            self.ple = Qwen4ExpPLE(
                config,
                layer_id,
                key_scheme=scheme_of(config, f"{prefix}.ple.key_proj"),
                value_scheme=scheme_of(config, f"{prefix}.ple.value_proj"),
                dtype=dtype,
            )
        mlp_prefix = f"{prefix}.mlp"
        self.mlp = SharedExpertMoE(
            config,
            layer_id,
            experts_schemes=ExpertSchemes(
                *schemes_of(config, f"{mlp_prefix}.experts", (*GATE_UP_PARTS, "down_proj")),
                is_gate_up_apart=config.is_expert_gate_up_apart,
            ),
            gate_scheme=scheme_of(config, f"{mlp_prefix}.gate"),
            shared_gate_up_schemes=schemes_of(config, f"{mlp_prefix}.shared_expert", GATE_UP_PARTS),
            shared_down_scheme=scheme_of(config, f"{mlp_prefix}.shared_expert.down_proj"),
            shared_gate_scheme=scheme_of(config, f"{mlp_prefix}.shared_expert_gate"),
            dtype=dtype,
        )
        self.attn_hyper_connection = hyper_connection(
            config, f"{prefix}.attn_hyper_connection", with_inject=True, dtype=dtype
        )
        self.mlp_hyper_connection = hyper_connection(
            config, f"{prefix}.mlp_hyper_connection", with_inject=True, dtype=dtype
        )

    def forward(
        self,
        hyper: mx.array,
        pending: PendingBranch | None,
        batch: Batch,
        ple_ticket: mx.array | None = None,
    ) -> tuple[mx.array, PendingBranch]:
        """hyper [T, hc_count * H]: the residual streams, the layer before's MLP branch pending
        (not yet injected); returns the streams and this layer's MLP branch, pending.
        ple_ticket: a decode step's PLE post for this layer (Qwen4ExpPLE.post_decode)."""
        if self.has_ple:
            if pending is not None:
                hyper = inject(hyper, *pending)
                pending = None
            hyper = hyper + self.ple.forward(hyper, batch, ple_ticket)
        hyper, mixed, logits = self.attn_hyper_connection.forward(hyper, pending)
        if self.is_linear:
            branch = self.linear_attn.forward(mixed, batch)
        else:
            branch = self.self_attn.forward(mixed, batch)
        hyper, mixed, logits = self.mlp_hyper_connection.forward(hyper, (branch, logits))
        return hyper, (self.mlp.forward(mixed), logits)


class Qwen4ExpModel(BaseOP):
    def __init__(self, config: Qwen4ExpConfig, *, dtype: mx.Dtype) -> None:
        super().__init__()
        self.hc_count = config.hc_count
        self.embed_tokens = Embedding(
            config.vocab_size,
            config.hidden_size,
            scheme=scheme_of(config, "model.embed_tokens"),
            dtype=dtype,
        )
        self.layers = [
            Qwen4ExpDecoderLayer(config, layer_id, prefix=f"model.layers.{layer_id}", dtype=dtype)
            for layer_id in range(config.num_layers)
        ]
        self.hyper_connection_mixer = hyper_connection(
            config, "model.hyper_connection_mixer", with_inject=False, dtype=dtype
        )

    def forward(self, batch: Batch) -> tuple[mx.array, PendingBranch]:
        """The residual streams after the last layer [T, hc_count * H], and its MLP branch,
        pending."""
        tickets: dict[int, mx.array] = {}
        if batch.is_decode and batch.input_ids.shape[0] <= MAX_PLE_ROWS:
            # The PLE lookups are posted first, so the service gathers the rows while the
            # layers before theirs run.
            input_ids = batch.input_ids
            for layer in self.layers:
                if layer.has_ple:
                    input_ids, tickets[layer.layer_id] = layer.ple.post_decode(batch, input_ids)
            embedded = self.embed_tokens.forward(input_ids)
        else:
            embedded = embed_input_ids(self.embed_tokens, batch)
        hyper = mx.tile(embedded, (1, self.hc_count))
        pending = None
        pools = get_global_ctx().pools
        for layer in self.layers:
            hyper, pending = layer.forward(hyper, pending, batch, tickets.get(layer.layer_id))
            if not batch.is_decode:
                # A pending pool write holds its inputs (a GDN layer's whole projection among
                # them) until the step ends: a prefill writes them layer by layer instead
                # (1.3 to 0.3 MB of activations per prefill token).
                mx.async_eval(hyper, *pending, *pool_arrays(pools))
        assert pending is not None
        return hyper, pending


class Qwen4ExpForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig, *, dtype: mx.Dtype = mx.bfloat16) -> None:
        super().__init__()
        assert isinstance(config, Qwen4ExpConfig)
        self.config = config
        self.model = Qwen4ExpModel(config, dtype=dtype)
        if not config.tie_word_embeddings:
            self.lm_head = Linear(
                config.hidden_size,
                config.vocab_size,
                scheme=scheme_of(config, "lm_head"),
                dtype=dtype,
            )
        if config.vision_config is not None:
            self.visual = Qwen3VLVisionModel(config.vision_config, channels_last=True)
            self.visual.set_dtype(dtype)

    def finish_loading(self, model_path: str) -> int:
        """Open each PLE layer's n-gram table in the checkpoint files and fuse each
        hyper-connection's down and inject projections and each MoE block's gates; returns the
        tables' row cache bytes."""
        cache_bytes = 0
        for layer in self.model.layers:
            if layer.has_ple:
                cache_bytes += layer.ple.open_table(model_path)
            layer.attn_hyper_connection.fuse_down_inject()
            layer.mlp_hyper_connection.fuse_down_inject()
            layer.mlp.fuse_gates()
        return cache_bytes

    @property
    def max_lookup_rows(self) -> int:
        return MAX_PLE_ROWS

    def lookup_buffers(self) -> list[mx.array]:
        return [
            array
            for layer in self.model.layers
            if layer.has_ple
            for array in (layer.ple.lookup.mailbox, *layer.ple.lookup.staging)
        ]

    def arm_lookups(self) -> None:
        for layer in self.model.layers:
            if layer.has_ple:
                layer.ple.lookup.service.arm()

    def check_lookups(self) -> None:
        for layer in self.model.layers:
            if layer.has_ple:
                layer.ple.lookup.service.check_healthy()

    def forward(self, batch: Batch) -> mx.array:
        hyper, (branch, logits) = self.model.forward(batch)
        pending = (select_last_rows(branch, batch), select_last_rows(logits, batch))
        hidden = self.model.hyper_connection_mixer.forward_final(
            select_last_rows(hyper, batch), pending
        )
        if self.config.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(hidden)
        else:
            return project(self.lm_head, hidden)

    def encode(self, item: MMItem) -> mx.array:
        """[item.num_tokens, out_hidden_size] soft tokens of one image."""
        return self.visual.encode(item)


__all__ = ["Qwen4ExpDecoderLayer", "Qwen4ExpForCausalLM", "Qwen4ExpModel"]
