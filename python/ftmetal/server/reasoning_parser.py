"""Splits a completion into reasoning (the <think> block) and content.

Port of FreeToken server/reasoning_parser.py (BaseReasoningParser, as its <think>
parser). Divergences: only the <think>/</think> pair, no tool-start marker ending the
reasoning, reasoning always streamed, and no one-shot parse (a whole completion is
parsed as one stream).
"""

from __future__ import annotations

from ftmetal.tokenizer import stop_prefix_holdback

THINK_START_TOKEN = "<think>"
THINK_END_TOKEN = "</think>"


class ThinkReasoningParser:
    def __init__(self, force_reasoning: bool) -> None:
        # force_reasoning: the prompt already opened the block, so the output starts inside it.
        self.in_reasoning = force_reasoning
        self.buffer = ""
        self.stripped_think_start = False

    def parse_stream_chunk(self, chunk: str) -> tuple[str, str]:
        """(reasoning, content) deltas of one streamed chunk."""
        self.buffer += chunk
        text = self.buffer
        if not self.stripped_think_start and THINK_START_TOKEN in text:
            text = text.replace(THINK_START_TOKEN, "", 1)
            self.stripped_think_start = True
            self.in_reasoning = True
        if self.in_reasoning and THINK_END_TOKEN in text:
            end = text.find(THINK_END_TOKEN)
            self.buffer = ""
            self.in_reasoning = False
            return text[:end].rstrip(), text[end + len(THINK_END_TOKEN) :]
        # A marker that may still come, split across chunks, is held back until the next
        # chunk completes or breaks it.
        markers = [THINK_END_TOKEN] if self.in_reasoning else []
        if not self.stripped_think_start:
            markers.append(THINK_START_TOKEN)
        safe_end = len(text) - stop_prefix_holdback(text, markers)
        safe = text[:safe_end]
        self.buffer = text[safe_end:]
        if self.in_reasoning:
            return safe, ""
        else:
            return "", safe

    def flush(self) -> tuple[str, str]:
        """(reasoning, content) still held back at the end of the stream."""
        text, self.buffer = self.buffer, ""
        if self.in_reasoning:
            return text, ""
        else:
            return "", text


__all__ = ["THINK_END_TOKEN", "THINK_START_TOKEN", "ThinkReasoningParser"]
