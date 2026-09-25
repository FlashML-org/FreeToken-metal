"""A deterministic clock for the radix trees.

tree_walk stamps every node it touches with one time.monotonic_ns() read, so nodes tie
within a walk by construction; a real clock also ties between walks (its resolution is
coarser than a walk), which makes LRU order assertions flaky. A strictly increasing
counter removes those ties.
"""

from __future__ import annotations

import itertools
import time

import pytest


@pytest.fixture(autouse=True)
def deterministic_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(time, "monotonic_ns", itertools.count(1_000_000_001).__next__)
