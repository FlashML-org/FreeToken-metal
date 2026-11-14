"""Prompt rendering, raw prompts, the reusable prefix end, image expansion, incremental
detokenization and stop strings.

The tiny checkpoint covers the contract; with FTMETAL_TEST_MODEL_PATH set, the real
checkpoint's tokenizer.json and chat template are checked against transformers, which is
also how FreeToken encodes a raw prompt.
"""

from __future__ import annotations

import io
import json
import logging
import os
from pathlib import Path
from typing import Literal

import pytest
from jinja2 import TemplateError
from PIL import Image
from tokenizers import AddedToken, Tokenizer, processors
from transformers import AutoTokenizer

from ftmetal.core import SamplingParams
from ftmetal.inline_system import InlineSystemError, normalize_inline_system
from ftmetal.message.frontend import UserReply
from ftmetal.message.tokenizer import DetokenizeMsg, FinishReason, JsonValue, TokenizeMsg
from ftmetal.mm.config import MultimodalConfig
from ftmetal.mm.item import MMItem
from ftmetal.mm.processors.qwen_vl import QwenVLMMProcessor
from ftmetal.models.loader import read_model_config
from ftmetal.tokenizer import (
    DetokenizeManager,
    TokenizeManager,
    load_eos_token_ids,
    load_sampling_defaults,
)
from tiny_vl_model import write_qwen_vl_model

MODEL_PATH = os.environ.get("FTMETAL_TEST_MODEL_PATH")
MULTILINGUAL_TEXT = "Héllo, 世界! Ünïcödé 🚀🧪 naïve café\n\tдобрый день — ok."


def make_managers(model_path: str) -> tuple[TokenizeManager, DetokenizeManager]:
    tokenize_manager = TokenizeManager(model_path)
    eos_token_ids = load_eos_token_ids(model_path, tokenize_manager.tokenizer)
    return tokenize_manager, DetokenizeManager(tokenize_manager.tokenizer, eos_token_ids)


def decode_stream(
    manager: DetokenizeManager,
    token_ids: list[int],
    finish_reason: FinishReason = "length",
    stop_strs: list[str] | None = None,
) -> list[UserReply | None]:
    """Feed token_ids one message at a time, the last one finishing the request."""
    manager.add_request(0, stop_strs or [])
    return [
        manager.detokenize(
            DetokenizeMsg(
                uid=0,
                next_token=token,
                finished=i == len(token_ids) - 1,
                finish_reason=finish_reason if i == len(token_ids) - 1 else None,
            )
        )
        for i, token in enumerate(token_ids)
    ]


def outputs(replies: list[UserReply | None]) -> list[str]:
    return [reply.incremental_output for reply in replies if reply is not None]


def test_incremental_detokenize_rebuilds_multibyte_text(tiny_model_dir: Path) -> None:
    tokenize_manager, detokenize_manager = make_managers(str(tiny_model_dir))
    eos = tokenize_manager.tokenizer.token_to_id("<|im_end|>")
    assert eos is not None
    token_ids = tokenize_manager.tokenizer.encode(MULTILINGUAL_TEXT).ids
    replies = decode_stream(detokenize_manager, [*token_ids, eos], finish_reason="stop")
    chunks = outputs(replies)
    assert "".join(chunks) == MULTILINGUAL_TEXT
    assert all("�" not in chunk for chunk in chunks)
    assert (
        sum(reply.completion_tokens_delta for reply in replies if reply is not None)
        == len(token_ids) + 1
    )
    last = replies[-1]
    assert last is not None and last.finished and last.finish_reason == "stop"


def test_partial_character_is_held_until_complete(tiny_model_dir: Path) -> None:
    tokenize_manager, detokenize_manager = make_managers(str(tiny_model_dir))
    token_ids = tokenize_manager.tokenizer.encode("a🚀b").ids
    assert len(token_ids) == 6
    assert outputs(decode_stream(detokenize_manager, token_ids)) == ["a", "", "", "", "🚀", "b"]


def test_finish_flushes_a_truncated_character(tiny_model_dir: Path) -> None:
    tokenize_manager, detokenize_manager = make_managers(str(tiny_model_dir))
    token_ids = tokenize_manager.tokenizer.encode("a🚀").ids[:3]
    assert "".join(outputs(decode_stream(detokenize_manager, token_ids))) == "a�"


def test_stop_string_ends_the_reply_before_the_match(tiny_model_dir: Path) -> None:
    tokenize_manager, detokenize_manager = make_managers(str(tiny_model_dir))
    token_ids = tokenize_manager.tokenizer.encode("The answer is 42. STOP and more").ids
    replies = decode_stream(detokenize_manager, token_ids, stop_strs=["never", "STOP"])
    assert "".join(outputs(replies)) == "The answer is 42. "
    stop_index = len(tokenize_manager.tokenizer.encode("The answer is 42. STOP").ids) - 1
    stopped = replies[stop_index]
    assert stopped is not None and stopped.finished
    assert stopped.finish_reason == "stop" and stopped.matched_stop == "STOP"
    assert replies[stop_index + 1 :] == [None] * (len(token_ids) - stop_index - 1)


def test_partial_stop_is_held_back_then_released(tiny_model_dir: Path) -> None:
    tokenize_manager, detokenize_manager = make_managers(str(tiny_model_dir))
    token_ids = tokenize_manager.tokenizer.encode("xxabdab").ids
    chunks = outputs(decode_stream(detokenize_manager, token_ids, stop_strs=["abc"]))
    assert chunks == ["x", "x", "", "", "abd", "", "ab"]


def test_dropped_request_ignores_late_tokens(tiny_model_dir: Path) -> None:
    tokenize_manager, detokenize_manager = make_managers(str(tiny_model_dir))
    detokenize_manager.add_request(7, [])
    detokenize_manager.discard(7)
    late = DetokenizeMsg(
        uid=7, next_token=tokenize_manager.tokenizer.encode("a").ids[0], finished=False
    )
    assert detokenize_manager.detokenize(late) is None


def tokenize_msg(
    messages: list[dict[str, JsonValue]],
    enable_thinking: bool | None = None,
    tools: list[dict[str, JsonValue]] | None = None,
    inline_system_policy: str | None = None,
) -> TokenizeMsg:
    kwargs: dict[str, JsonValue] = {}
    if enable_thinking is not None:
        kwargs["enable_thinking"] = enable_thinking
    return TokenizeMsg(
        uid=0,
        text=messages,
        sampling_params=SamplingParams(),
        chat_template_kwargs=kwargs,
        tools=tools,
        inline_system_policy=inline_system_policy,
    )


# A template that renders a system turn wherever it is.
PLAIN_CHATML_TEMPLATE = (
    "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)
FOLDED = "<|im_start|>user\nWeather?\n\nUse metric units.<|im_end|>"
PRESERVED = "<|im_start|>user\nWeather?<|im_end|>\n<|im_start|>system\nUse metric units.<|im_end|>"
# (policy, whether the template renders a late system turn, its rendering): the tiny template
# refuses one.
INLINE_SYSTEM_CASES: list[tuple[str, bool, str]] = [
    ("auto", False, FOLDED),
    ("auto", True, PRESERVED),
    ("fold", True, FOLDED),
    ("preserve", True, PRESERVED),
]


@pytest.mark.parametrize(("policy", "renders_late_system", "rendered"), INLINE_SYSTEM_CASES)
def test_late_system_message_stays_after_the_cached_prefix(
    tiny_model_dir: Path, policy: str, renders_late_system: bool, rendered: str
) -> None:
    if renders_late_system:
        (tiny_model_dir / "chat_template.jinja").write_text(PLAIN_CHATML_TEMPLATE)
    manager = TokenizeManager(str(tiny_model_dir))
    messages: list[dict[str, JsonValue]] = [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello."},
        {"role": "user", "content": "Weather?"},
        {"role": "system", "content": "Use metric units."},
    ]
    prompt = manager.render_prompt(tokenize_msg(messages, inline_system_policy=policy))
    head = "<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n"
    assert prompt.startswith(head) and rendered in prompt


def test_late_system_message_waits_for_the_tool_results() -> None:
    call: dict[str, JsonValue] = {"role": "assistant", "content": "", "tool_calls": []}
    call["tool_calls"] = [{"id": "a"}, {"id": "b"}]
    messages: list[dict[str, JsonValue]] = [
        {"role": "user", "content": "Weather?"},
        call,
        {"role": "tool", "tool_call_id": "a", "content": "Sunny"},
        {"role": "system", "content": "Use metric units."},
        {"role": "tool", "tool_call_id": "b", "content": "20 C"},
    ]
    assert normalize_inline_system(messages, "fold")[3:] == [
        {"role": "tool", "tool_call_id": "b", "content": "20 C\n\nUse metric units."}
    ]
    interrupted = [*messages[:2], messages[3], {"role": "user", "content": "Go on."}]
    with pytest.raises(InlineSystemError, match="interrupts incomplete tool results"):
        normalize_inline_system(interrupted, "fold")


def test_chat_template_raises_and_renders_plain_json(tiny_model_dir: Path) -> None:
    manager = TokenizeManager(str(tiny_model_dir))
    out_of_order: list[dict[str, JsonValue]] = [
        {"role": "user", "content": "hi"},
        {"role": "system", "content": "late"},
    ]
    with pytest.raises(TemplateError, match="System message must be at the beginning"):
        manager.render_prompt(tokenize_msg(out_of_order))
    tools: list[dict[str, JsonValue]] = [{"name": "lookup", "description": "a <b> & c"}]
    prompt = manager.render_prompt(tokenize_msg([{"role": "user", "content": "hi"}], tools=tools))
    assert '"description": "a <b> & c"' in prompt


# (tokenizer.json post-processor, raw prompt, input ids): None keeps the ChatML tokenizer;
# a post-processor comes with a <bos> special token (id 260) and a Gemma tokenizer config.
RAW_PROMPT_CASES: list[tuple[str | None, str, list[int]]] = [
    (None, "hi", [71, 72]),
    (None, "<|im_start|>hi<think>", [256, 71, 72, 258]),
    # Gemma4's post-processor adds no BOS, so a raw prompt has one only where it is written.
    ("$A", "hi", [71, 72]),
    ("$A", "<bos>hi", [260, 71, 72]),
    ("<bos> $A", "hi", [260, 71, 72]),
]


@pytest.mark.parametrize(("post_processor", "prompt", "input_ids"), RAW_PROMPT_CASES)
def test_raw_prompt_skips_the_template_and_matches_transformers(
    tiny_model_dir: Path, post_processor: str | None, prompt: str, input_ids: list[int]
) -> None:
    if post_processor is not None:
        tokenizer = Tokenizer.from_file(str(tiny_model_dir / "tokenizer.json"))
        tokenizer.add_special_tokens([AddedToken("<bos>", special=True)])
        special_tokens = [("<bos>", 260)] if "<bos>" in post_processor else []
        tokenizer.post_processor = processors.TemplateProcessing(
            single=post_processor, special_tokens=special_tokens
        )
        tokenizer.save(str(tiny_model_dir / "tokenizer.json"))
        config = {
            "eos_token": "<|im_end|>",
            "bos_token": "<bos>",
            "tokenizer_class": "GemmaTokenizer",
        }
        (tiny_model_dir / "tokenizer_config.json").write_text(json.dumps(config))
    msg = TokenizeMsg(uid=0, text=prompt, sampling_params=SamplingParams())
    user_msg = TokenizeManager(str(tiny_model_dir)).tokenize(msg)
    assert user_msg.input_ids.tolist() == input_ids
    # A raw prompt has no generation suffix: a next prompt can share all of it.
    assert user_msg.reuse_len == len(input_ids)
    assert AutoTokenizer.from_pretrained(tiny_model_dir).encode(prompt) == input_ids


# The shape of Gemma4's template: <bos>, turns between <|turn> and <turn|>, and a generation
# prompt that opens the model turn and, unless enable_thinking is true, an empty thought.
GEMMA_STYLE_TEMPLATE = """\
{{- bos_token }}
{%- for message in messages %}
{%- set role = 'model' if message.role == 'assistant' else message.role %}
{{- '<|turn>' + role + '\\n' + message.content + '<turn|>\\n' }}
{%- endfor %}
{%- if add_generation_prompt %}
{{- '<|turn>model\\n' }}
{%- if not enable_thinking | default(false) %}
{{- '<|channel>thought\\n<channel|>' }}
{%- endif %}
{%- endif %}
"""
GEMMA_STYLE_SPECIAL_TOKENS = ("<bos>", "<|turn>", "<turn|>", "<|channel>", "<channel|>")
# Renders a tail only without the generation prompt, so the conversation's encoding is not a
# prefix of the prompt's.
UNSTABLE_TEMPLATE_TAIL = "{%- if not add_generation_prompt %}{{- 'x' }}{%- endif %}\n"
CONVERSATION: list[dict[str, JsonValue]] = [
    {"role": "user", "content": "Hi"},
    {"role": "assistant", "content": "Hello!"},
    {"role": "user", "content": "Again"},
]

# (chat template, enable_thinking, the generation suffix the reusable prefix leaves out)
REUSE_CASES: list[tuple[Literal["qwen", "gemma"], bool | None, str]] = [
    ("qwen", None, "<|im_start|>assistant\n<think>\n"),
    ("qwen", False, "<|im_start|>assistant\n<think>\n\n</think>\n\n"),
    ("gemma", None, "<|turn>model\n<|channel>thought\n<channel|>"),
    ("gemma", True, "<|turn>model\n"),
]


@pytest.mark.parametrize(("template", "enable_thinking", "suffix"), REUSE_CASES)
def test_conversation_reuse_ends_before_the_generation_suffix(
    tiny_model_dir: Path,
    template: Literal["qwen", "gemma"],
    enable_thinking: bool | None,
    suffix: str,
) -> None:
    if template == "gemma":
        tokenizer = Tokenizer.from_file(str(tiny_model_dir / "tokenizer.json"))
        tokenizer.add_special_tokens(
            [AddedToken(token, special=True) for token in GEMMA_STYLE_SPECIAL_TOKENS]
        )
        tokenizer.save(str(tiny_model_dir / "tokenizer.json"))
        (tiny_model_dir / "chat_template.jinja").write_text(GEMMA_STYLE_TEMPLATE)
        config = {"eos_token": "<turn|>", "bos_token": "<bos>"}
        (tiny_model_dir / "tokenizer_config.json").write_text(json.dumps(config))
    manager = TokenizeManager(str(tiny_model_dir))
    msg = tokenize_msg(CONVERSATION, enable_thinking)
    user_msg = manager.tokenize(msg)
    input_ids = user_msg.input_ids.tolist()
    unshared = manager.tokenizer.decode(input_ids[user_msg.reuse_len :], skip_special_tokens=False)
    assert unshared == suffix
    assert manager.render_prompt(msg) == manager.render_prompt(msg, False) + suffix


def test_unstable_template_reuses_the_whole_prompt_and_warns_once(
    tiny_model_dir: Path, caplog: pytest.LogCaptureFixture
) -> None:
    template_path = tiny_model_dir / "chat_template.jinja"
    template_path.write_text(template_path.read_text() + UNSTABLE_TEMPLATE_TAIL)
    manager = TokenizeManager(str(tiny_model_dir))
    with caplog.at_level(logging.WARNING, logger="ftmetal.tokenizer"):
        for _ in range(2):
            user_msg = manager.tokenize(tokenize_msg(CONVERSATION))
            assert user_msg.reuse_len == len(user_msg.input_ids)
    assert len(caplog.records) == 1


def test_image_turn_reuses_its_expanded_prefix(
    tiny_model_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_qwen_vl_model(tiny_model_dir)
    model_path = str(tiny_model_dir)
    processor = QwenVLMMProcessor(read_model_config(model_path), model_path, MultimodalConfig())
    processed: list[int] = []
    process = processor.process

    def counting_process(images: list[Image.Image]) -> list[MMItem]:
        processed.append(len(images))
        return process(images)

    monkeypatch.setattr(processor, "process", counting_process)
    manager = TokenizeManager(model_path, processor)
    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), (120, 30, 200)).save(buffer, format="PNG")
    first_turn: list[dict[str, JsonValue]] = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "What is it?"}]}
    ]
    next_turn: list[dict[str, JsonValue]] = [
        *first_turn,
        {"role": "assistant", "content": "A square."},
        {"role": "user", "content": "Which color?"},
    ]
    first, following = [
        manager.tokenize(
            TokenizeMsg(
                uid=0, text=messages, sampling_params=SamplingParams(), images=[buffer.getvalue()]
            )
        )
        for messages in (first_turn, next_turn)
    ]
    # Each request decodes and preprocesses its image once.
    assert processed == [1, 1]
    assert first.mm_items is not None and following.mm_items is not None
    ((start, end),) = first.mm_items[0].offsets
    assert end - start == 64 and end <= first.reuse_len
    assert following.mm_items[0].offsets == first.mm_items[0].offsets
    # The first turn's reusable prefix, its image span included, opens the next turn's prompt.
    first_ids = first.input_ids.tolist()
    assert following.input_ids.tolist()[: first.reuse_len] == first_ids[: first.reuse_len]
    for user_msg in (first, following):
        unshared = user_msg.input_ids.tolist()[user_msg.reuse_len :]
        decoded = manager.tokenizer.decode(unshared, skip_special_tokens=False)
        assert decoded == "<|im_start|>assistant\n<think>\n"


def test_images_need_an_image_processor(tiny_model_dir: Path) -> None:
    messages: list[dict[str, JsonValue]] = [{"role": "user", "content": "hi"}]
    msg = TokenizeMsg(uid=0, text=messages, sampling_params=SamplingParams(), images=[b""])
    with pytest.raises(ValueError, match="image input is not supported"):
        TokenizeManager(str(tiny_model_dir)).tokenize(msg)


def test_checkpoint_eos_and_sampling_defaults(tiny_model_dir: Path) -> None:
    manager = TokenizeManager(str(tiny_model_dir))
    eos = manager.tokenizer.token_to_id("<|im_end|>")
    assert eos is not None
    assert load_eos_token_ids(str(tiny_model_dir), manager.tokenizer) == {eos}
    defaults = load_sampling_defaults(str(tiny_model_dir))
    assert (defaults.temperature, defaults.top_k, defaults.top_p) == (0.6, 20, 0.95)
    (tiny_model_dir / "generation_config.json").write_text('{"do_sample": false}')
    assert load_sampling_defaults(str(tiny_model_dir)).is_greedy


def test_stop_ids_come_from_config_json_without_generation_config(tiny_model_dir: Path) -> None:
    manager = TokenizeManager(str(tiny_model_dir))
    eos = manager.tokenizer.token_to_id("<|im_end|>")
    assert eos is not None
    end_of_text = 7
    (tiny_model_dir / "config.json").write_text(json.dumps({"eos_token_id": [eos, end_of_text]}))
    assert load_eos_token_ids(str(tiny_model_dir), manager.tokenizer) == {eos}
    (tiny_model_dir / "generation_config.json").unlink()
    assert load_eos_token_ids(str(tiny_model_dir), manager.tokenizer) == {eos, end_of_text}


@pytest.mark.skipif(MODEL_PATH is None, reason="FTMETAL_TEST_MODEL_PATH is not set")
def test_real_tokenizer_matches_transformers() -> None:
    assert MODEL_PATH is not None
    tokenize_manager, detokenize_manager = make_managers(MODEL_PATH)
    reference = AutoTokenizer.from_pretrained(MODEL_PATH)

    token_ids = tokenize_manager.tokenizer.encode(MULTILINGUAL_TEXT).ids
    assert token_ids == reference.encode(MULTILINGUAL_TEXT, add_special_tokens=False)
    raw = TokenizeMsg(uid=0, text=MULTILINGUAL_TEXT, sampling_params=SamplingParams())
    assert tokenize_manager.tokenize(raw).input_ids.tolist() == reference.encode(MULTILINGUAL_TEXT)
    assert "".join(outputs(decode_stream(detokenize_manager, token_ids))) == MULTILINGUAL_TEXT

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Weather for a <city> & country",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
            },
        }
    ]
    conversations: list[list[dict[str, JsonValue]]] = [
        [{"role": "user", "content": "What is 2 + 3?"}],
        [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!", "reasoning_content": "Greet back."},
            {"role": "user", "content": MULTILINGUAL_TEXT},
        ],
    ]
    for messages in conversations:
        for enable_thinking in (None, True, False):
            for tool_list in (None, tools):
                msg = tokenize_msg(messages, enable_thinking, tool_list)
                expected = reference.apply_chat_template(
                    messages,
                    tools=tool_list,
                    tokenize=False,
                    add_generation_prompt=True,
                    **msg.chat_template_kwargs,
                )
                history = reference.apply_chat_template(
                    messages,
                    tools=tool_list,
                    tokenize=False,
                    add_generation_prompt=False,
                    **msg.chat_template_kwargs,
                )
                assert tokenize_manager.render_prompt(msg) == expected
                user_msg = tokenize_manager.tokenize(msg)
                assert user_msg.input_ids.tolist() == reference.encode(
                    expected, add_special_tokens=False
                )
                history_ids = reference.encode(history, add_special_tokens=False)
                assert user_msg.reuse_len == len(history_ids)
