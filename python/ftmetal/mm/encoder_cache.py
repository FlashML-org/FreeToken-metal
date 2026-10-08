"""Encoder embedding cache keyed by content hash; an entry lives while any request still has
rows of it to gather.

Port of FreeToken mm/encoder_cache.py. Divergences: entries are mx.arrays in unified memory,
so there is no storage knob and no host copy; get_slice takes no device.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mlx.core as mx


@dataclass
class EncoderCacheEntry:
    embedding: mx.array | None = None
    # uid -> embedding rows the request still has to gather, summed over its occurrences of
    # the image.
    remaining: dict[int, int] = field(default_factory=dict)


class EncoderCache:
    def __init__(self) -> None:
        self.entries: dict[int, EncoderCacheEntry] = {}

    def register(self, item_hash: int, uid: int, rows: int) -> None:
        """Claim rows of the image for uid before any of its chunks run; a repeated image adds
        up."""
        if rows <= 0:
            return
        entry = self.entries.setdefault(item_hash, EncoderCacheEntry())
        entry.remaining[uid] = entry.remaining.get(uid, 0) + rows

    def has(self, item_hash: int) -> bool:
        entry = self.entries.get(item_hash)
        return entry is not None and entry.embedding is not None

    def put(self, item_hash: int, embedding: mx.array) -> None:
        entry = self.entries.get(item_hash)
        assert entry is not None, f"image {item_hash} encoded before any request registered it"
        if entry.embedding is None:
            entry.embedding = embedding

    def get_slice(self, item_hash: int, row_lo: int, row_hi: int) -> mx.array:
        embedding = self.entries[item_hash].embedding
        assert embedding is not None, f"image {item_hash} gathered before it was encoded"
        return embedding[row_lo:row_hi]

    def consume(self, item_hash: int, uid: int, rows: int) -> None:
        """Account rows uid gathered; the last row of the last holder frees the entry."""
        entry = self.entries[item_hash]
        left = entry.remaining[uid] - rows
        assert left >= 0, f"request {uid} gathered more rows of image {item_hash} than it claimed"
        if left:
            entry.remaining[uid] = left
        else:
            entry.remaining.pop(uid)
            if not entry.remaining:
                self.entries.pop(item_hash)

    def release(self, uid: int, hashes: list[int]) -> None:
        """Drop uid's claims whatever it gathered (abort); entries nobody else holds are freed."""
        for item_hash in hashes:
            entry = self.entries.get(item_hash)
            if entry is None:
                continue
            entry.remaining.pop(uid, None)
            if not entry.remaining:
                self.entries.pop(item_hash)

    def stats(self) -> tuple[int, int]:
        """(encoded entries, their bytes)."""
        stored = [entry.embedding for entry in self.entries.values() if entry.embedding is not None]
        return len(stored), sum(embedding.nbytes for embedding in stored)


__all__ = ["EncoderCache", "EncoderCacheEntry"]
