"""A tiny checkpoint directory: byte-level tokenizer, chat template, generation config."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from tokenizers import AddedToken, Tokenizer, decoders, models, pre_tokenizers

# The shape of Qwen3.6's template: ChatML turns and a <think> block the generation
# prompt opens, or opens and closes when enable_thinking is false.
TINY_CHAT_TEMPLATE = """\
{%- if not messages %}{{- raise_exception('No messages provided.') }}{%- endif %}
{%- if tools %}
{{- '<|im_start|>system\\n<tools>' + (tools | tojson) + '</tools><|im_end|>\\n' }}
{%- endif %}
{%- for message in messages %}
{%- if message.role == 'system' and not loop.first %}
{{- raise_exception('System message must be at the beginning.') }}
{%- endif %}
{{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>\\n' }}
{%- endfor %}
{%- if add_generation_prompt %}
{{- '<|im_start|>assistant\\n' }}
{%- if enable_thinking is defined and enable_thinking is false %}
{{- '<think>\\n\\n</think>\\n\\n' }}
{%- else %}
{{- '<think>\\n' }}
{%- endif %}
{%- endif %}
"""


@pytest.fixture
def tiny_model_dir(tmp_path: Path) -> Path:
    """Every byte is one token (so multi-byte characters span tokens), plus ChatML specials."""
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    tokenizer = Tokenizer(models.BPE(vocab={char: i for i, char in enumerate(alphabet)}, merges=[]))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.add_special_tokens(
        [AddedToken("<|im_start|>", special=True), AddedToken("<|im_end|>", special=True)]
    )
    tokenizer.add_tokens([AddedToken("<think>"), AddedToken("</think>")])
    model_dir = tmp_path / "tiny-model"
    model_dir.mkdir()
    tokenizer.save(str(model_dir / "tokenizer.json"))
    (model_dir / "chat_template.jinja").write_text(TINY_CHAT_TEMPLATE)
    generation_config = {
        "eos_token_id": [tokenizer.token_to_id("<|im_end|>")],
        "temperature": 0.6,
        "top_k": 20,
        "top_p": 0.95,
    }
    (model_dir / "generation_config.json").write_text(json.dumps(generation_config))
    (model_dir / "tokenizer_config.json").write_text(json.dumps({"eos_token": "<|im_end|>"}))
    return model_dir
