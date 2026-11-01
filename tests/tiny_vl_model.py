"""Tiny multimodal checkpoints for the image pipeline.

write_qwen_vl_model and write_gemma4_model write Qwen3.6- and Gemma4-shaped config.json and
image processor configs without weights; write_qwen_vl_model also extends the tiny tokenizer
with Qwen's vision tokens and installs a chat template that renders one placeholder per image
part, as Qwen3.6's does. save_vl_checkpoint writes a servable tiny Qwen3.6-shaped model, the
mlx-lm text reference and a vision tower, and reference_vl_greedy is its unpaged greedy decode
of an image prompt: the tower's rows in the input embeddings and mlx-lm's rope swapped for the
3-axis one.
"""

from __future__ import annotations

import dataclasses
import io
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten
from mlx_lm.models import qwen3_5 as reference
from PIL import Image
from pydantic import JsonValue
from tokenizers import AddedToken, Tokenizer

from ftmetal.kernel.ref import mrope_ref
from ftmetal.layers.rotary import build_section_table
from ftmetal.mm.config import MultimodalConfig
from ftmetal.mm.processor import MMResult
from ftmetal.mm.processors.qwen_vl import QwenVLMMProcessor
from ftmetal.models.loader import read_model_config
from ftmetal.models.qwen3_vl.vision import Qwen3VLVisionModel
from ftmetal.models.register import MODEL_REGISTRY
from tiny_model import TINY_TEXT_CONFIG, build_reference, save_checkpoint

# The image processor configs the Qwen3.6-35B-A3B and Gemma-4-26B-A4B checkpoints ship.
QWEN_PREPROCESSOR_CONFIG: dict[str, JsonValue] = {
    "size": {"longest_edge": 16777216, "shortest_edge": 65536},
    "patch_size": 16,
    "temporal_patch_size": 2,
    "merge_size": 2,
    "image_mean": [0.5, 0.5, 0.5],
    "image_std": [0.5, 0.5, 0.5],
    "processor_class": "Qwen3VLProcessor",
    "image_processor_type": "Qwen2VLImageProcessorFast",
}
GEMMA4_IMAGE_PROCESSOR_CONFIG: dict[str, JsonValue] = {
    "do_convert_rgb": True,
    "do_normalize": False,
    "do_rescale": True,
    "do_resize": True,
    "image_mean": [0.0, 0.0, 0.0],
    "image_processor_type": "Gemma4ImageProcessor",
    "image_seq_length": 280,
    "image_std": [1.0, 1.0, 1.0],
    "max_soft_tokens": 280,
    "patch_size": 16,
    "pooling_kernel_size": 3,
    "resample": 3,
    "rescale_factor": 0.00392156862745098,
}
# Gemma4's image token ids.
GEMMA4_BOI_TOKEN_ID = 255999
GEMMA4_IMAGE_TOKEN_ID = 258880
GEMMA4_EOI_TOKEN_ID = 258882

QWEN_VL_IMAGE_TOKENS = ("<|vision_start|>", "<|image_pad|>", "<|vision_end|>")
QWEN_VL_CHAT_TEMPLATE = """\
{%- for message in messages %}
{{- '<|im_start|>' + message.role + '\\n' }}
{%- if message.content is string %}
{{- message.content }}
{%- elif message.content %}
{%- for part in message.content %}
{%- if part.type == 'image' %}
{{- '<|vision_start|><|image_pad|><|vision_end|>' }}
{%- else %}
{{- part.text }}
{%- endif %}
{%- endfor %}
{%- endif %}
{{- '<|im_end|>\\n' }}
{%- endfor %}
{%- if add_generation_prompt %}
{{- '<|im_start|>assistant\\n<think>\\n' }}
{%- endif %}
"""


def write_qwen_vl_model(model_dir: Path, has_vision: bool = True) -> int:
    """Turn the tiny checkpoint into a Qwen3.6-shaped multimodal one, without its vision_config
    unless has_vision; returns the image token id."""
    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    tokenizer.add_special_tokens(
        [AddedToken(token, special=True) for token in QWEN_VL_IMAGE_TOKENS]
    )
    tokenizer.save(str(model_dir / "tokenizer.json"))
    image_token_id = tokenizer.token_to_id("<|image_pad|>")
    assert image_token_id is not None
    (model_dir / "chat_template.jinja").write_text(QWEN_VL_CHAT_TEMPLATE)
    config: dict[str, JsonValue] = {
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "model_type": "qwen3_5_moe",
        "image_token_id": image_token_id,
        "text_config": {
            "vocab_size": 512,
            "rope_parameters": {
                "mrope_interleaved": True,
                "mrope_section": [11, 11, 10],
                "rope_theta": 10000000,
                "rope_type": "default",
            },
        },
    }
    if has_vision:
        config["vision_config"] = {
            "in_channels": 3,
            "patch_size": 16,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
        }
    (model_dir / "config.json").write_text(json.dumps(config))
    (model_dir / "preprocessor_config.json").write_text(json.dumps(QWEN_PREPROCESSOR_CONFIG))
    return image_token_id


def write_gemma4_model(model_dir: Path) -> None:
    """A Gemma4-shaped multimodal config.json and processor_config.json."""
    config = {
        "architectures": ["Gemma4ForConditionalGeneration"],
        "model_type": "gemma4",
        "boi_token_id": GEMMA4_BOI_TOKEN_ID,
        "image_token_id": GEMMA4_IMAGE_TOKEN_ID,
        "eoi_token_id": GEMMA4_EOI_TOKEN_ID,
        "vision_config": {"patch_size": 16, "pooling_kernel_size": 3},
        "text_config": {"vocab_size": 262144},
    }
    (model_dir / "config.json").write_text(json.dumps(config))
    processor_config = {"image_processor": GEMMA4_IMAGE_PROCESSOR_CONFIG}
    (model_dir / "processor_config.json").write_text(json.dumps(processor_config))


# Gemma4's vision_config keys at toy sizes: head_dim 32, a 16-row position table per axis.
TINY_GEMMA4_VISION_CONFIG: dict[str, object] = {
    "model_type": "gemma4_vision",
    "hidden_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 2,
    "num_key_value_heads": 2,
    "head_dim": 32,
    "intermediate_size": 96,
    "hidden_activation": "gelu_pytorch_tanh",
    "patch_size": 16,
    "pooling_kernel_size": 3,
    "position_embedding_size": 16,
    "rope_parameters": {"rope_theta": 100.0, "rope_type": "default"},
    "rms_norm_eps": 1e-6,
    "standardize": True,
    "use_clipped_linears": False,
}


def gemma4_tower_weights(
    vision_config: dict[str, object], text_hidden_size: int, seed: int
) -> dict[str, mx.array]:
    """fp32 random weights of a Gemma4 vision tower and its projection under the checkpoint's
    names: norms near one, fan-in scaled projections."""
    rng = np.random.default_rng(seed)
    sizes = {key: value for key, value in vision_config.items() if isinstance(value, int)}
    hidden, heads, head_dim = sizes["hidden_size"], sizes["num_attention_heads"], sizes["head_dim"]
    kv_heads, intermediate = sizes["num_key_value_heads"], sizes["intermediate_size"]
    patch, table = sizes["patch_size"], sizes["position_embedding_size"]

    def projection(rows: int, columns: int) -> np.ndarray:
        return rng.standard_normal((rows, columns)) / np.sqrt(columns)

    def norm(size: int) -> np.ndarray:
        return 1 + 0.1 * rng.standard_normal(size)

    tower = "vision_tower"
    weights = {
        f"{tower}.patch_embedder.input_proj.weight": projection(hidden, 3 * patch**2),
        f"{tower}.patch_embedder.position_embedding_table": rng.standard_normal((2, table, hidden)),
        f"{tower}.std_bias": 0.1 * rng.standard_normal(hidden),
        f"{tower}.std_scale": norm(hidden),
        "embed_vision.embedding_projection.weight": projection(text_hidden_size, hidden),
    }
    shapes = {
        "self_attn.q_proj": (heads * head_dim, hidden),
        "self_attn.k_proj": (kv_heads * head_dim, hidden),
        "self_attn.v_proj": (kv_heads * head_dim, hidden),
        "self_attn.o_proj": (hidden, heads * head_dim),
        "mlp.gate_proj": (intermediate, hidden),
        "mlp.up_proj": (intermediate, hidden),
        "mlp.down_proj": (hidden, intermediate),
    }
    for layer_id in range(sizes["num_hidden_layers"]):
        layer = f"{tower}.encoder.layers.{layer_id}"
        for name, (rows, columns) in shapes.items():
            weights[f"{layer}.{name}.linear.weight"] = projection(rows, columns)
        for name in ("q_norm", "k_norm"):
            weights[f"{layer}.self_attn.{name}.weight"] = norm(head_dim)
        for name in ("input", "post_attention", "pre_feedforward", "post_feedforward"):
            weights[f"{layer}.{name}_layernorm.weight"] = norm(hidden)
    return {name: mx.array(value.astype(np.float32)) for name, value in weights.items()}


# TINY_TEXT_CONFIG's rope (theta 10000, a quarter of the 64-wide heads) with Qwen3.6's
# interleaved 3-axis split of its 8 frequency slots.
TINY_ROPE_THETA = 10000.0
TINY_ROTARY_DIM = 16
TINY_MROPE_SECTION = (3, 3, 2)
TINY_VL_TEXT_CONFIG: dict[str, object] = {
    **TINY_TEXT_CONFIG,
    "rope_parameters": {
        "type": "default",
        "rope_theta": TINY_ROPE_THETA,
        "partial_rotary_factor": 0.25,
        "mrope_section": list(TINY_MROPE_SECTION),
        "mrope_interleaved": True,
    },
}
# Qwen3.6's vision_config keys at toy sizes; the tower's output is the text width.
TINY_VISION_CONFIG: dict[str, object] = {
    "depth": 2,
    "hidden_size": 64,
    "num_heads": 2,
    "intermediate_size": 128,
    "patch_size": 16,
    "temporal_patch_size": 2,
    "spatial_merge_size": 2,
    "num_position_embeddings": 16,
    "out_hidden_size": TINY_TEXT_CONFIG["hidden_size"],
    "in_channels": 3,
    "hidden_act": "gelu_pytorch_tanh",
    "deepstack_visual_indexes": [],
}
# Inside the tiny vocabulary, clear of the random prompts' 1..119 and the EOS 127.
TINY_IMAGE_TOKEN_ID = 125
# Images down to 4 tokens, so that a prompt holds several.
TINY_IMAGE_BUDGET = MultimodalConfig(image_min_tokens=4)


def build_tower(seed: int) -> Qwen3VLVisionModel:
    mx.random.seed(seed)
    tower = Qwen3VLVisionModel(TINY_VISION_CONFIG, channels_last=False)
    tower.set_dtype(mx.float32)
    mx.eval(tower.parameters())
    return tower


def save_vl_checkpoint(model: reference.Model, tower: Qwen3VLVisionModel, path: Path) -> None:
    """The text model in the MLX layout, the tower under the HF model.visual. prefix, and
    the checkpoint's config.json and preprocessor config."""
    save_checkpoint(model, path, quantization=None)
    weights = {
        f"model.visual.{key}": value
        for key, value in tree_flatten(tower.parameters())
        if isinstance(value, mx.array)
    }
    mx.save_safetensors(str(path / "visual.safetensors"), weights)
    config = read_model_config(str(path))
    config["vision_config"] = TINY_VISION_CONFIG
    config["image_token_id"] = TINY_IMAGE_TOKEN_ID
    (path / "config.json").write_text(json.dumps(config))
    (path / "preprocessor_config.json").write_text(json.dumps(QWEN_PREPROCESSOR_CONFIG))


def build_vl_checkpoint(seed: int, path: Path) -> tuple[reference.Model, Qwen3VLVisionModel]:
    """A tiny random Qwen3.6-shaped VL model saved under path, and its two halves."""
    model = build_reference(seed, text_config=TINY_VL_TEXT_CONFIG)
    tower = build_tower(seed)
    save_vl_checkpoint(model, tower, path)
    return model, tower


def noise_png(width: int, height: int, seed: int) -> bytes:
    pixels = np.random.default_rng(seed).integers(0, 256, (height, width, 3), dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format="PNG")
    return buffer.getvalue()


def image_prompt(path: Path, texts: list[list[int]], images: list[bytes]) -> MMResult:
    """texts with an image placeholder between each two, expanded by the checkpoint's
    processor; fresh items every call, since a forward frees the features it encodes."""
    assert len(texts) == len(images) + 1
    ids = list(texts[0])
    for text in texts[1:]:
        ids += [TINY_IMAGE_TOKEN_ID, *text]
    processor = QwenVLMMProcessor(read_model_config(str(path)), str(path), TINY_IMAGE_BUDGET)
    return processor.apply(np.array(ids, dtype=np.int32), images)


class ReferenceMRope(nn.RoPE):
    """mlx-lm's rope slot, rotating a whole sequence by its 3-axis positions."""

    def __init__(self, rotary_dim: int, base: float) -> None:
        super().__init__(rotary_dim, base=base)
        half = rotary_dim // 2
        self.rotary_dim = rotary_dim
        self.inv_freq = mx.array((base ** (-np.arange(half) / half)).astype(np.float32))
        self.section = mx.array(build_section_table(TINY_MROPE_SECTION, "interleaved"))
        self.positions = mx.zeros((3, 0), dtype=mx.int32)

    def __call__(self, x: mx.array, offset: int = 0) -> mx.array:
        """x is [1, heads, tokens, head_dim] of the whole sequence: offset is always 0."""
        assert offset == 0
        rows = x[0].transpose(1, 0, 2)
        rotated = mrope_ref(rows, self.positions, self.inv_freq, self.section, self.rotary_dim)
        return rotated.transpose(1, 0, 2)[None]


def reference_vl_greedy(
    model: reference.Model, tower: Qwen3VLVisionModel, prompt: MMResult, steps: int
) -> list[int]:
    """Greedy decode rerunning the whole sequence per token: the tower's rows replace the
    image pad ids' embeddings, and the full-attention layers rotate by the prompt's mrope rows
    and by the sequence index plus delta past it."""
    assert prompt.mrope_positions is not None
    text_model = model.language_model.model
    full_attentions = [layer.self_attn for layer in text_model.layers if not layer.is_linear]
    one_axis_ropes = [attention.rope for attention in full_attentions]
    rope = ReferenceMRope(TINY_ROTARY_DIM, TINY_ROPE_THETA)
    for attention in full_attentions:
        attention.rope = rope
    rows = [
        row for item in prompt.mm_items for start, end in item.offsets for row in range(start, end)
    ]
    embeds: list[mx.array] = []
    for item in prompt.mm_items:
        assert item.feature is not None and item.grid_thw is not None
        embeds.append(tower(mx.array(item.feature).view(mx.bfloat16), [item.grid_thw]))
    sequence = prompt.input_ids.tolist()
    prompt_len = len(sequence)
    vocab_size = text_model.embed_tokens.weight.shape[0]
    tokens: list[int] = []
    for _ in range(steps):
        tail = np.arange(prompt_len, len(sequence), dtype=np.int32) + prompt.mrope_delta
        positions = np.concatenate([prompt.mrope_positions, np.tile(tail, (3, 1))], axis=1)
        rope.positions = mx.array(positions)
        ids = mx.array(sequence)
        x = text_model.embed_tokens(mx.minimum(ids, vocab_size - 1))
        if embeds:
            x[mx.array(rows)] = mx.concatenate(embeds)
        logits = model(ids[None], input_embeddings=x[None])
        tokens.append(int(mx.argmax(logits[0, -1])))
        sequence.append(tokens[-1])
    for attention, one_axis_rope in zip(full_attentions, one_axis_ropes):
        attention.rope = one_axis_rope
    return tokens


def drop_vision_encoder(monkeypatch: pytest.MonkeyPatch, architecture: str) -> None:
    """Take architecture's encoders out of its registry entry, as for a family that registers
    its image processor before its tower: its checkpoints serve no images."""
    spec = dataclasses.replace(MODEL_REGISTRY[architecture], encoders=())
    monkeypatch.setitem(MODEL_REGISTRY, architecture, spec)


__all__ = [
    "GEMMA4_BOI_TOKEN_ID",
    "GEMMA4_EOI_TOKEN_ID",
    "GEMMA4_IMAGE_PROCESSOR_CONFIG",
    "GEMMA4_IMAGE_TOKEN_ID",
    "QWEN_PREPROCESSOR_CONFIG",
    "QWEN_VL_CHAT_TEMPLATE",
    "ReferenceMRope",
    "TINY_GEMMA4_VISION_CONFIG",
    "TINY_IMAGE_BUDGET",
    "TINY_IMAGE_TOKEN_ID",
    "TINY_MROPE_SECTION",
    "TINY_ROPE_THETA",
    "TINY_ROTARY_DIM",
    "TINY_VISION_CONFIG",
    "TINY_VL_TEXT_CONFIG",
    "build_tower",
    "build_vl_checkpoint",
    "drop_vision_encoder",
    "gemma4_tower_weights",
    "image_prompt",
    "noise_png",
    "reference_vl_greedy",
    "save_vl_checkpoint",
    "write_gemma4_model",
    "write_qwen_vl_model",
]
