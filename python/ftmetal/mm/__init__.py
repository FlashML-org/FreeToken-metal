"""Multimodal input handling: the pad-id scheme, per-family processors and media fetch.

Port of FreeToken mm/__init__.py. Divergence: restore_placeholder works on numpy ids (the
host hashes them).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

# An image span's tokens become a content pseudo id above every real token id: the radix
# cache keys them by content. The model clamps them back into the vocab for the embedding
# lookup; those rows are overwritten anyway. shift + hash % 2**30 stays below 2**31, so the
# ids fit the int32 token arrays.
MM_PAD_SHIFT_VALUE = 1_000_000


def mm_pad_value(item_hash: int) -> int:
    return MM_PAD_SHIFT_VALUE + item_hash % (1 << 30)


def restore_placeholder(input_ids: npt.NDArray[np.int64], token_id: int) -> npt.NDArray[np.int64]:
    """input_ids with every content pad id put back to the family's placeholder token."""
    return np.where(input_ids >= MM_PAD_SHIFT_VALUE, token_id, input_ids)


def check_mm_pad_shift(vocab_size: int) -> None:
    if vocab_size > MM_PAD_SHIFT_VALUE:
        raise ValueError(
            f"vocab_size ({vocab_size}) exceeds MM_PAD_SHIFT_VALUE ({MM_PAD_SHIFT_VALUE}); "
            "image pseudo-ids would alias real token ids"
        )


__all__ = ["MM_PAD_SHIFT_VALUE", "check_mm_pad_shift", "mm_pad_value", "restore_placeholder"]
