"""Chat-template rendering and encoding, incremental detokenization and stop strings.

Port of FreeToken tokenizer/tokenize.py and tokenizer/detokenize.py. Divergences: the
tokenizers library and a jinja2 environment that mirrors apply_chat_template replace
transformers; stop strings are matched here over the decoded text (the scheduler knows
only EOS and max_tokens), so a match finishes the reply and the caller aborts the
request; a request's decode state is created when its prompt is tokenized, so tokens
that arrive after a stop, an abort or an error are dropped; text ending in a partial
UTF-8 character is held back whole, and flushed as is when the request finishes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import NoReturn

import numpy as np
from jinja2 import TemplateError
from jinja2.ext import loopcontrols
from jinja2.sandbox import ImmutableSandboxedEnvironment
from pydantic import JsonValue
from tokenizers import Tokenizer

from ftmetal.core import SamplingParams
from ftmetal.message.backend import UserMsg
from ftmetal.message.frontend import UserReply
from ftmetal.message.tokenizer import DetokenizeMsg, TokenizeMsg

TOKENIZER_FILE = "tokenizer.json"
TOKENIZER_CONFIG_FILE = "tokenizer_config.json"
CHAT_TEMPLATE_FILE = "chat_template.jinja"
GENERATION_CONFIG_FILE = "generation_config.json"
# The tokenizer_config.json entries apply_chat_template hands the template (Gemma's opens
# with bos_token).
TEMPLATE_SPECIAL_TOKENS = ("bos_token", "eos_token", "pad_token", "unk_token")
# What a byte-level decode yields for a UTF-8 sequence the last token left incomplete.
REPLACEMENT_CHARACTER = "�"


def read_json_object(path: Path) -> dict[str, object]:
    """A checkpoint's JSON config file; an empty object when the file is absent."""
    if not path.is_file():
        return {}
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


def load_eos_token_ids(model_path: str, tokenizer: Tokenizer) -> set[int]:
    """generation_config.json's eos_token_id (an id or a list) plus the tokenizer's eos token."""
    eos_token_ids: set[int] = set()
    generation_eos = read_json_object(Path(model_path) / GENERATION_CONFIG_FILE).get("eos_token_id")
    if isinstance(generation_eos, int):
        eos_token_ids.add(generation_eos)
    elif isinstance(generation_eos, list):
        eos_token_ids.update(token for token in generation_eos if isinstance(token, int))
    eos_token = read_json_object(Path(model_path) / TOKENIZER_CONFIG_FILE).get("eos_token")
    eos_token_id = tokenizer.token_to_id(eos_token) if isinstance(eos_token, str) else None
    if eos_token_id is not None:
        eos_token_ids.add(eos_token_id)
    if not eos_token_ids:
        raise ValueError(f"{model_path} names no EOS token")
    return eos_token_ids


def load_sampling_defaults(model_path: str) -> SamplingParams:
    """The checkpoint's recommended temperature, top_k and top_p; greedy when do_sample is off."""
    config = read_json_object(Path(model_path) / GENERATION_CONFIG_FILE)
    framework = SamplingParams()
    temperature = config.get("temperature")
    top_k = config.get("top_k")
    top_p = config.get("top_p")
    if config.get("do_sample") is False:
        return framework
    else:
        return SamplingParams(
            temperature=(
                float(temperature)
                if isinstance(temperature, int | float)
                else framework.temperature
            ),
            top_k=top_k if isinstance(top_k, int) else framework.top_k,
            top_p=float(top_p) if isinstance(top_p, int | float) else framework.top_p,
        )


def raise_exception(message: str) -> NoReturn:
    raise TemplateError(message)


def tojson(
    value: JsonValue,
    ensure_ascii: bool = False,
    indent: int | None = None,
    separators: tuple[str, str] | None = None,
    sort_keys: bool = False,
) -> str:
    # jinja's own tojson escapes <, > and & for HTML; chat templates expect plain JSON.
    return json.dumps(
        value, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys
    )


class TokenizeManager:
    def __init__(self, model_path: str) -> None:
        path = Path(model_path)
        self.tokenizer = Tokenizer.from_file(str(path / TOKENIZER_FILE))
        # The environment apply_chat_template renders with.
        environment = ImmutableSandboxedEnvironment(
            trim_blocks=True, lstrip_blocks=True, extensions=[loopcontrols]
        )
        environment.filters["tojson"] = tojson
        environment.globals["raise_exception"] = raise_exception
        self.chat_template = environment.from_string((path / CHAT_TEMPLATE_FILE).read_text())
        tokenizer_config = read_json_object(path / TOKENIZER_CONFIG_FILE)
        self.special_tokens: dict[str, str] = {}
        for name in TEMPLATE_SPECIAL_TOKENS:
            token = tokenizer_config.get(name)
            if isinstance(token, str):
                self.special_tokens[name] = token

    def render_prompt(self, msg: TokenizeMsg) -> str:
        return self.chat_template.render(
            messages=msg.text,
            tools=msg.tools,
            add_generation_prompt=True,
            **self.special_tokens,
            **msg.chat_template_kwargs,
        )

    def tokenize(self, msg: TokenizeMsg) -> UserMsg:
        # The template renders every special token, so encoding adds none.
        encoding = self.tokenizer.encode(self.render_prompt(msg), add_special_tokens=False)
        return UserMsg(
            uid=msg.uid,
            input_ids=np.array(encoding.ids, dtype=np.int32),
            sampling_params=msg.sampling_params,
        )


@dataclass
class DecodeStatus:
    stop_strs: list[str]
    decoded_ids: list[int] = field(default_factory=list)
    # decoded_ids[:read_offset] are decoded; decoded_ids[surr_offset:read_offset] is the
    # context the next tokens are decoded against.
    read_offset: int = 0
    surr_offset: int = 0
    # Decoded text not sent yet: a suffix that may still grow into a stop string.
    unsent_str: str = ""


def stop_prefix_holdback(text: str, stop_strs: list[str]) -> int:
    """Length of the longest suffix of text that is a proper prefix of a stop string."""
    holdback = 0
    for stop in stop_strs:
        for length in range(min(len(stop) - 1, len(text)), holdback, -1):
            if text.endswith(stop[:length]):
                holdback = length
                break
    return holdback


class DetokenizeManager:
    def __init__(self, tokenizer: Tokenizer, eos_token_ids: set[int]) -> None:
        self.tokenizer = tokenizer
        self.eos_token_ids = eos_token_ids
        self.decode_map: dict[int, DecodeStatus] = {}

    def add_request(self, uid: int, stop_strs: list[str]) -> None:
        self.decode_map[uid] = DecodeStatus(stop_strs=stop_strs)

    def discard(self, uid: int) -> None:
        self.decode_map.pop(uid, None)

    def detokenize(self, msg: DetokenizeMsg) -> UserReply | None:
        """The reply for one sampled token; None once the request stopped or was dropped."""
        status = self.decode_map.get(msg.uid)
        if status is None:
            return None
        if not (msg.finished and msg.next_token in self.eos_token_ids):
            status.decoded_ids.append(msg.next_token)
        surr_text = self.tokenizer.decode(
            status.decoded_ids[status.surr_offset : status.read_offset], skip_special_tokens=False
        )
        read_text = self.tokenizer.decode(
            status.decoded_ids[status.surr_offset :], skip_special_tokens=False
        )
        new_text = read_text[len(surr_text) :]
        if msg.finished or (new_text and not new_text.endswith(REPLACEMENT_CHARACTER)):
            status.unsent_str += new_text
            status.surr_offset = status.read_offset
            status.read_offset = len(status.decoded_ids)

        # Sent text held no stop string and no unsent prefix of one, so a match can only
        # start in the unsent text.
        text = status.unsent_str
        found = [(text.find(stop), stop) for stop in status.stop_strs]
        matches = [(position, stop) for position, stop in found if position >= 0]
        if matches:
            emit_end, matched_stop = min(matches, key=lambda match: match[0])
            reply = UserReply(
                uid=msg.uid,
                incremental_output=text[:emit_end],
                finished=True,
                completion_tokens_delta=1,
                finish_reason="stop",
                matched_stop=matched_stop,
            )
        elif msg.finished:
            reply = UserReply(
                uid=msg.uid,
                incremental_output=text,
                finished=True,
                completion_tokens_delta=1,
                finish_reason=msg.finish_reason,
            )
        else:
            emit_end = len(text) - stop_prefix_holdback(text, status.stop_strs)
            reply = UserReply(
                uid=msg.uid,
                incremental_output=text[:emit_end],
                finished=False,
                completion_tokens_delta=1,
            )
            status.unsent_str = text[emit_end:]
        if reply.finished:
            self.decode_map.pop(msg.uid)
        return reply


__all__ = [
    "TEMPLATE_SPECIAL_TOKENS",
    "DecodeStatus",
    "DetokenizeManager",
    "TokenizeManager",
    "load_eos_token_ids",
    "load_sampling_defaults",
    "raise_exception",
    "read_json_object",
    "stop_prefix_holdback",
    "tojson",
]
