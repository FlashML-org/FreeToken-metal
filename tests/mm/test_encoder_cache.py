"""EncoderCache ownership: rows claimed at admission, freed when the last claim is gathered or
released."""

from __future__ import annotations

import mlx.core as mx

from ftmetal.mm.encoder_cache import EncoderCache


def test_entry_lives_until_every_claim_is_gathered() -> None:
    cache = EncoderCache()
    embedding = mx.arange(12, dtype=mx.float32).reshape(6, 2)
    cache.register(7, 1, 6)
    cache.register(7, 2, 6)
    # Claimed, not encoded yet.
    assert not cache.has(7)
    cache.put(7, embedding)
    assert cache.has(7)
    assert mx.array_equal(cache.get_slice(7, 2, 4), embedding[2:4]).item()
    cache.consume(7, 1, 4)
    cache.consume(7, 1, 2)
    # Request 2 still has rows to gather.
    assert cache.has(7) and cache.stats() == (1, embedding.nbytes)
    cache.consume(7, 2, 6)
    assert not cache.has(7) and cache.stats() == (0, 0)


def test_repeated_image_in_one_request_adds_up() -> None:
    cache = EncoderCache()
    cache.register(7, 1, 4)
    cache.register(7, 1, 4)
    cache.put(7, mx.ones((4, 2)))
    # The first occurrence is done, the second still holds the entry.
    cache.consume(7, 1, 4)
    assert cache.has(7)
    cache.consume(7, 1, 4)
    assert not cache.has(7)


def test_release_drops_a_claim_even_before_the_embedding_arrives() -> None:
    cache = EncoderCache()
    cache.register(5, 1, 3)
    cache.register(5, 2, 3)
    # An unknown hash is a no-op.
    cache.release(1, [5, 99])
    assert list(cache.entries) == [5]
    cache.release(2, [5])
    assert cache.entries == {} and cache.stats() == (0, 0)


def test_a_claim_of_no_rows_holds_nothing() -> None:
    cache = EncoderCache()
    cache.register(5, 1, 0)
    assert cache.entries == {}
