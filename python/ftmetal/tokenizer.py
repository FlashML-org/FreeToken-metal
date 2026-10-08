"""Chat-template rendering and encoding, incremental detokenization and stop strings; a
conversation's reasoning_effort is quantized onto the vocabulary probed from its template.

Port of FreeToken tokenizer/tokenize.py and tokenizer/detokenize.py. Divergences: the
tokenizers library and a jinja2 environment that mirrors apply_chat_template replace
transformers; stop strings are matched here over the decoded text (the scheduler knows
only EOS and max_tokens), so a match finishes the reply and the caller aborts the
request; a request's decode state is created when its prompt is tokenized, so tokens
that arrive after a stop, an abort or an error are dropped; text ending in a partial
UTF-8 character is held back whole, and flushed as is when the request finishes; a
conversation's reusable prefix ends where its rendering without the generation prompt
does, when that encoding is a prefix of the prompt's (FreeToken has no such boundary),
moved past the expansion of the image placeholders it holds, so each image is processed
once; resolve_thinking_mode is FreeToken's, for the reasoning parser's force_reasoning.

A raw prompt gets the special tokens tokenizer.json's post-processor adds, which is what
transformers 5 adds for FreeToken: none for Qwen3.6, and none for Gemma4 either (its
post-processor has no <bos>), so a raw Gemma4 prompt starts with BOS only when the
client writes <bos> into it.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, NoReturn

import numpy as np
from jinja2 import TemplateError
from jinja2.ext import loopcontrols
from jinja2.sandbox import ImmutableSandboxedEnvironment
from pydantic import JsonValue
from tokenizers import Tokenizer

from ftmetal.core import SamplingParams
from ftmetal.effort import (
    EffortProfile,
    ThinkingProfile,
    probe_effort_profile,
    probe_thinking_profile,
    quantize_effort,
)
from ftmetal.message.backend import UserMsg
from ftmetal.message.frontend import UserReply
from ftmetal.message.tokenizer import DetokenizeMsg, TokenizeMsg
from ftmetal.mm.processor import MMProcessor, find_all

logger = logging.getLogger(__name__)

TOKENIZER_FILE = "tokenizer.json"
TOKENIZER_CONFIG_FILE = "tokenizer_config.json"
CHAT_TEMPLATE_FILE = "chat_template.jinja"
GENERATION_CONFIG_FILE = "generation_config.json"
# The tokenizer_config.json entries apply_chat_template hands the template (Gemma's opens
# with bos_token).
TEMPLATE_SPECIAL_TOKENS = ("bos_token", "eos_token", "pad_token", "unk_token")
# What a byte-level decode yields for a UTF-8 sequence the last token left incomplete.
REPLACEMENT_CHARACTER = "�"
# The conversation the reasoning-effort and thinking probes render.
EFFORT_PROBE_MESSAGES: list[dict[str, JsonValue]] = [{"role": "user", "content": "ping"}]


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


def resolve_thinking_mode(
    chat_template_kwargs: dict[str, JsonValue] | None, tools: list[dict[str, JsonValue]] | None
) -> Literal["thinking", "chat"]:
    """Whether a chat request thinks: when tools are offered or chat_template_kwargs asks."""
    ctk = chat_template_kwargs or {}
    if tools or ctk.get("enable_thinking") or ctk.get("thinking"):
        return "thinking"
    elif ctk.get("thinking_mode") == "thinking":
        return "thinking"
    else:
        return "chat"


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
    def __init__(self, model_path: str, mm_processor: MMProcessor | None = None) -> None:
        path = Path(model_path)
        self.tokenizer = Tokenizer.from_file(str(path / TOKENIZER_FILE))
        # None: the model takes no images.
        self.mm_processor = mm_processor
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
        self.has_warned_unstable_template = False
        # The template's effort and thinking controls, probed on first use.
        self.probed_effort_profile: EffortProfile | None = None
        self.probed_thinking_profile: ThinkingProfile | None = None
        self.effort_lock = threading.Lock()
        # (requested effort, the gear it was served as) pairs already logged.
        self.logged_effort_maps: set[tuple[str, str | None]] = set()

    def render_prompt(self, msg: TokenizeMsg, add_generation_prompt: bool = True) -> str:
        """The prompt text. A conversation's reasoning_effort is quantized first: every render
        path (worker, frontend check, count_tokens) must agree."""
        if isinstance(msg.text, str):
            return msg.text
        else:
            return self.render_template(
                msg.text,
                msg.tools,
                self.sanitize_effort(msg.chat_template_kwargs),
                add_generation_prompt,
            )

    def render_template(
        self,
        messages: list[dict[str, JsonValue]],
        tools: list[dict[str, JsonValue]] | None,
        chat_template_kwargs: dict[str, JsonValue],
        add_generation_prompt: bool = True,
    ) -> str:
        """Raw render, no effort sanitation: the probes need unsupported values to reach the
        template so that a rejection is observable."""
        if "reasoning_effort" in chat_template_kwargs:
            # Broadcast in every spelling templates read (muse-glimmer grades
            # reasoning_strength); one the caller spelled out wins.
            chat_template_kwargs = {
                "reasoning_strength": chat_template_kwargs["reasoning_effort"],
                **chat_template_kwargs,
            }
        return self.chat_template.render(
            messages=messages,
            tools=tools,
            add_generation_prompt=add_generation_prompt,
            **self.special_tokens,
            **chat_template_kwargs,
        )

    def effort_profile(self) -> EffortProfile:
        """The template's reasoning-effort vocabulary, probed once and cached."""
        with self.effort_lock:
            if self.probed_effort_profile is None:
                profile = probe_effort_profile(self.probe_render)
                supported = sorted(profile.supported) or "(none)"
                logger.info(
                    f"reasoning-effort profile: supported={supported} default={profile.default}"
                )
                self.probed_effort_profile = profile
            return self.probed_effort_profile

    def thinking_profile(self) -> ThinkingProfile:
        """The template's thinking toggles and effort vocabulary, probed once and cached."""
        efforts = self.effort_profile()
        with self.effort_lock:
            if self.probed_thinking_profile is None:
                self.probed_thinking_profile = probe_thinking_profile(self.probe_render, efforts)
            return self.probed_thinking_profile

    def probe_render(
        self, chat_template_kwargs: dict[str, JsonValue], tools: list[dict[str, JsonValue]] | None
    ) -> str:
        """The probe conversation rendered with one probe's kwargs and tools."""
        return self.render_template(EFFORT_PROBE_MESSAGES, tools, chat_template_kwargs)

    def sanitize_effort(self, chat_template_kwargs: dict[str, JsonValue]) -> dict[str, JsonValue]:
        """chat_template_kwargs with reasoning_effort quantized onto the template's vocabulary,
        or dropped when nothing in it is close enough."""
        if "reasoning_effort" not in chat_template_kwargs:
            return chat_template_kwargs
        raw = chat_template_kwargs["reasoning_effort"]
        mapped = quantize_effort(raw, self.effort_profile())
        if mapped == raw:
            return chat_template_kwargs
        # raw is client-controlled and may be unhashable (a JSON list or object).
        key = (raw if isinstance(raw, str) else repr(raw), mapped)
        if key not in self.logged_effort_maps:
            self.logged_effort_maps.add(key)
            served = mapped if mapped is not None else "the template default"
            logger.info(
                f"reasoning_effort {raw!r} is not supported by this checkpoint; using {served}"
            )
        sanitized = dict(chat_template_kwargs)
        if mapped is None:
            sanitized.pop("reasoning_effort")
        else:
            sanitized["reasoning_effort"] = mapped
        return sanitized

    def tokenize(self, msg: TokenizeMsg) -> UserMsg:
        if isinstance(msg.text, str):
            # Only a raw prompt's encoding adds special tokens: the template renders its own.
            input_ids = self.tokenizer.encode(msg.text, add_special_tokens=True).ids
            reuse_len = len(input_ids)
        else:
            # A next turn re-renders this conversation and goes on past it: what it shares
            # ends before the generation suffix.
            prompt_encoding, history_encoding = self.tokenizer.encode_batch(
                [self.render_prompt(msg), self.render_prompt(msg, add_generation_prompt=False)],
                add_special_tokens=False,
            )
            input_ids = prompt_encoding.ids
            history_ids = history_encoding.ids
            if input_ids[: len(history_ids)] == history_ids:
                reuse_len = len(history_ids)
            else:
                if not self.has_warned_unstable_template:
                    logger.warning(
                        "the chat template's generation suffix does not extend the "
                        "conversation's encoding: GDN snapshots track up to the prompt end"
                    )
                    self.has_warned_unstable_template = True
                reuse_len = len(input_ids)
        if not msg.images:
            return UserMsg(
                uid=msg.uid,
                input_ids=np.array(input_ids, dtype=np.int32),
                sampling_params=msg.sampling_params,
                reuse_len=reuse_len,
            )
        elif self.mm_processor is None:
            raise ValueError("image input is not supported for this model")
        else:
            processor = self.mm_processor
            mm = processor.apply(np.array(input_ids, dtype=np.int32), msg.images)
            # The reusable prefix holds the prompt's first images, whose placeholders expand
            # alike: it grows by their expansion.
            prefix_images = len(find_all(input_ids[:reuse_len], processor.placeholder))
            expansion = sum(
                len(processor.prompt_replacement(item).full) - len(processor.placeholder)
                for item in mm.mm_items[:prefix_images]
            )
            return UserMsg(
                uid=msg.uid,
                input_ids=mm.input_ids,
                sampling_params=msg.sampling_params,
                reuse_len=reuse_len + expansion,
                mm_items=mm.mm_items,
                mrope_positions=mm.mrope_positions,
                mrope_delta=mm.mrope_delta,
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
        finished = msg.finished or bool(matches)
        if matches:
            emit_end, matched_stop = min(matches, key=lambda match: match[0])
            finish_reason = "stop"
        elif msg.finished:
            emit_end, matched_stop, finish_reason = len(text), None, msg.finish_reason
        else:
            emit_end = len(text) - stop_prefix_holdback(text, status.stop_strs)
            matched_stop, finish_reason = None, None
        status.unsent_str = text[emit_end:]
        if finished:
            self.decode_map.pop(msg.uid)
        return UserReply(
            uid=msg.uid,
            incremental_output=text[:emit_end],
            finished=finished,
            completion_tokens_delta=1,
            finish_reason=finish_reason,
            matched_stop=matched_stop,
            kv_used_pages=msg.kv_used_pages,
            kv_total_pages=msg.kv_total_pages,
            mamba_used_slots=msg.mamba_used_slots,
            mamba_total_slots=msg.mamba_total_slots,
            swa_used_tokens=msg.swa_used_tokens,
            swa_total_tokens=msg.swa_total_tokens,
        )


__all__ = [
    "EFFORT_PROBE_MESSAGES",
    "TEMPLATE_SPECIAL_TOKENS",
    "DecodeStatus",
    "DetokenizeManager",
    "TokenizeManager",
    "load_eos_token_ids",
    "load_sampling_defaults",
    "raise_exception",
    "read_json_object",
    "resolve_thinking_mode",
    "stop_prefix_holdback",
    "tojson",
]
