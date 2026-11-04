"""PLE table rows of a decode step, looked up inside the step: ple_post and ple_wait on the
GPU, _C.PleService and _C.PleRowStore on the CPU.

ple_post, at the step's start, posts each batch row's token with its slot's n-gram context
(the two tokens before it) to the mailbox and advances the context; the service's
coordinator hashes the n-grams, gathers the table rows from its row store into the staging
buffers and answers; ple_wait, encoded after the arrays it is given (the PLE layer's input),
waits for the answer and copies the rows out. No host code reads the step's token ids, so a
decode step with lookups replays as a command graph. A post refused after a shutdown, a
failed or unanswered post, a wait the shutdown releases and one that gave up past the spin
limit return zero rows; the service records the first ones and the GPU the last (the abandoned
word), so check_healthy raises before the step's tokens are used.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import mlx.core as mx

from ftmetal.kernel._ext import _C


class PleRowParts(NamedTuple):
    # uint8 [rows, heads, row bytes] each: the packed weights, the scales, the biases.
    weight: mx.array
    scales: mx.array
    biases: mx.array


class PlePostOutputs(NamedTuple):
    # int32 [num_slots, 2]: the advanced context; it aliases the argument, callers rebind it.
    context: mx.array
    # int32 [num_rows]: a copy of input_ids; embedding it orders the post first.
    input_ids: mx.array
    # uint32 [1]: the post's sequence number, 0 when a shutdown refused it.
    ticket: mx.array


def ple_post(
    input_ids: mx.array, slots: mx.array, context: mx.array, mailbox: mx.array
) -> PlePostOutputs:
    """Post input_ids int32 [num_rows], num_rows <= MAX_PLE_ROWS, with the n-gram context of
    their slots int32 [num_rows] in context int32 [num_slots, 2] (before previous, previous).

    Padding rows may share a slot: every context is read before any is advanced. mailbox
    uint32 [PLE_MAILBOX_WORDS] is the service's.
    """
    context, input_ids, ticket = _C.ple_post(input_ids, slots, context, mailbox)
    return PlePostOutputs(context, input_ids, ticket)


def ple_wait(
    ticket: mx.array,
    mailbox: mx.array,
    staging: PleRowParts,
    num_rows: int,
    *,
    after: Sequence[mx.array] = (),
) -> PleRowParts:
    """The table rows of the first num_rows batch rows of ticket's post, uint8 [num_rows,
    heads, row bytes] per part, once the service answered; zeros if it did not.

    staging is the service's, uint8 [MAX_PLE_ROWS, heads, row bytes] per part. The wait is
    encoded after the arrays in after, so it spins only once the step reaches them.
    """
    weight, scales, biases = _C.ple_wait(ticket, mailbox, list(staging), num_rows, list(after))
    return PleRowParts(weight, scales, biases)


__all__ = ["PlePostOutputs", "PleRowParts", "ple_post", "ple_wait"]
