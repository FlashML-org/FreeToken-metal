"""Prompt rendering, incremental detokenization and stop strings.

The tiny checkpoint covers the contract; with FTMETAL_TEST_MODEL_PATH set, the real
checkpoint's tokenizer.json and chat template are checked against transformers.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from jinja2 import TemplateError
from transformers import AutoTokenizer

from ftmetal.core import SamplingParams
from ftmetal.message.frontend import UserReply
from ftmetal.message.tokenizer import (
    ChatMessage,
    DetokenizeMsg,
    FinishReason,
    JsonValue,
    TokenizeMsg,
)
from ftmetal.tokenizer import (
    DetokenizeManager,
    TokenizeManager,
    load_eos_token_ids,
    load_sampling_defaults,
)

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


def test_eos_is_counted_but_not_decoded(tiny_model_dir: Path) -> None:
    tokenize_manager, detokenize_manager = make_managers(str(tiny_model_dir))
    eos = tokenize_manager.tokenizer.token_to_id("<|im_end|>")
    assert eos is not None
    replies = decode_stream(detokenize_manager, [*tokenize_manager.tokenizer.encode("hi").ids, eos])
    assert "".join(outputs(replies)) == "hi"
    assert len(replies) == 3


def test_dropped_request_ignores_late_tokens(tiny_model_dir: Path) -> None:
    tokenize_manager, detokenize_manager = make_managers(str(tiny_model_dir))
    detokenize_manager.add_request(7, [])
    detokenize_manager.discard(7)
    late = DetokenizeMsg(
        uid=7, next_token=tokenize_manager.tokenizer.encode("a").ids[0], finished=False
    )
    assert detokenize_manager.detokenize(late) is None


def tokenize_msg(
    messages: list[ChatMessage],
    enable_thinking: bool | None = None,
    tools: list[dict[str, JsonValue]] | None = None,
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
    )


def test_chat_template_opens_or_closes_the_think_block(tiny_model_dir: Path) -> None:
    manager = TokenizeManager(str(tiny_model_dir))
    messages: list[ChatMessage] = [{"role": "user", "content": "hi"}]
    thinking = manager.render_prompt(tokenize_msg(messages))
    assert thinking == "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n"
    assert manager.render_prompt(tokenize_msg(messages, enable_thinking=False)).endswith(
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
    input_ids = manager.tokenize(tokenize_msg(messages)).input_ids.tolist()
    special_ids = [manager.tokenizer.token_to_id(t) for t in ("<|im_start|>", "<think>")]
    assert input_ids[0] == special_ids[0] and input_ids[-2] == special_ids[1]


def test_chat_template_raises_and_renders_plain_json(tiny_model_dir: Path) -> None:
    manager = TokenizeManager(str(tiny_model_dir))
    out_of_order: list[ChatMessage] = [
        {"role": "user", "content": "hi"},
        {"role": "system", "content": "late"},
    ]
    with pytest.raises(TemplateError, match="System message must be at the beginning"):
        manager.render_prompt(tokenize_msg(out_of_order))
    tools: list[dict[str, JsonValue]] = [{"name": "lookup", "description": "a <b> & c"}]
    prompt = manager.render_prompt(tokenize_msg([{"role": "user", "content": "hi"}], tools=tools))
    assert '"description": "a <b> & c"' in prompt


def test_checkpoint_eos_and_sampling_defaults(tiny_model_dir: Path) -> None:
    manager = TokenizeManager(str(tiny_model_dir))
    eos = manager.tokenizer.token_to_id("<|im_end|>")
    assert eos is not None
    assert load_eos_token_ids(str(tiny_model_dir), manager.tokenizer) == {eos}
    defaults = load_sampling_defaults(str(tiny_model_dir))
    assert (defaults.temperature, defaults.top_k, defaults.top_p) == (0.6, 20, 0.95)
    (tiny_model_dir / "generation_config.json").write_text('{"do_sample": false}')
    assert load_sampling_defaults(str(tiny_model_dir)).is_greedy


@pytest.mark.skipif(MODEL_PATH is None, reason="FTMETAL_TEST_MODEL_PATH is not set")
def test_real_tokenizer_matches_transformers() -> None:
    assert MODEL_PATH is not None
    tokenize_manager, detokenize_manager = make_managers(MODEL_PATH)
    reference = AutoTokenizer.from_pretrained(MODEL_PATH)

    token_ids = tokenize_manager.tokenizer.encode(MULTILINGUAL_TEXT).ids
    assert token_ids == reference.encode(MULTILINGUAL_TEXT, add_special_tokens=False)
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
    conversations: list[list[ChatMessage]] = [
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
                assert tokenize_manager.render_prompt(msg) == expected
                assert tokenize_manager.tokenize(msg).input_ids.tolist() == reference.encode(
                    expected, add_special_tokens=False
                )
