"""CommandGraph capture and replay on a small synthetic step: no checkpoint needed."""

import mlx.core as mx
import pytest

from ftmetal.kernel._ext import _C
from ftmetal.kernel.command_graph import begin_capture, check_same_graph, end_capture, replay
from ftmetal.kernel.inplace import scatter_rows

HIDDEN = 256
NUM_HEADS = 4
NUM_KV_HEADS = 2
HEAD_DIM = HIDDEN // NUM_HEADS
# Grouped-query attention over at least 4096 keys runs MLX's two-pass SDPA on every GPU,
# whose partials are MLX temporaries.
KV_LEN = 4096
OUTPUT = 32
POOL_ROWS = 8
SCALE = HEAD_DIM**-0.5


class TinyStep:
    """A decode-like step: projection, norm, rope, attention, projection, pool row write."""

    def __init__(self) -> None:
        keys = mx.random.split(mx.random.key(0), 4)
        self.w_in = (mx.random.normal((HIDDEN, HIDDEN), key=keys[0]) * 0.05).astype(mx.bfloat16)
        self.norm = mx.ones((HIDDEN,), dtype=mx.bfloat16)
        kv_shape = (1, NUM_KV_HEADS, KV_LEN, HEAD_DIM)
        self.k = mx.random.normal(kv_shape, key=keys[1]).astype(mx.bfloat16)
        self.v = mx.random.normal(kv_shape, key=keys[2]).astype(mx.bfloat16)
        self.w_out = (mx.random.normal((HIDDEN, OUTPUT), key=keys[3]) * 0.05).astype(mx.bfloat16)
        mx.eval(*self.weights())

    def weights(self) -> list[mx.array]:
        return [self.w_in, self.norm, self.k, self.v, self.w_out]

    def forward(
        self,
        x: mx.array,
        rows: mx.array,
        positions: mx.array,
        pool: mx.array,
        rope_offset: int | None = None,
        scale: float = SCALE,
    ) -> tuple[mx.array, mx.array]:
        """Logits and the updated pool; rope_offset replaces positions by a Python int."""
        h = mx.fast.rms_norm(x @ self.w_in, self.norm, 1e-6) * 2.0
        q = mx.fast.rope(
            h.reshape(1, NUM_HEADS, 1, HEAD_DIM),
            HEAD_DIM,
            traditional=False,
            base=10000.0,
            scale=1.0,
            offset=positions if rope_offset is None else rope_offset,
        )
        o = mx.fast.scaled_dot_product_attention(q, self.k, self.v, scale=scale)
        logits = o.reshape(1, HIDDEN) @ self.w_out
        return logits, scatter_rows(pool, logits, rows)


def step_inputs(seed: int) -> list[mx.array]:
    x = mx.random.normal((1, HIDDEN), key=mx.random.key(seed)).astype(mx.bfloat16)
    rows = mx.array([seed % POOL_ROWS], dtype=mx.int32)
    positions = mx.array([7 * seed], dtype=mx.int32)
    mx.eval(x, rows, positions)
    return [x, rows, positions]


def zero_pool() -> mx.array:
    pool = mx.zeros((POOL_ROWS, OUTPUT), dtype=mx.bfloat16)
    mx.eval(pool)
    return pool


def capture(
    step: TinyStep,
    inputs: list[mx.array],
    pool: mx.array,
    rope_offset: int | None = None,
    scale: float = SCALE,
) -> tuple[_C.Graph, mx.array]:
    """The graph of one step and the logits the capture computed."""
    x, rows, positions = inputs
    logits, updated = step.forward(x, rows, positions, pool, rope_offset, scale)
    begin_capture([logits, updated], inputs, [pool], step.weights())
    mx.eval(logits, updated)
    return end_capture(), logits


def test_replay_matches_eager_on_new_inputs() -> None:
    step = TinyStep()
    x, rows, positions = step_inputs(1)
    expected, _ = step.forward(x, rows, positions, zero_pool())
    graph, captured = capture(step, [x, rows, positions], zero_pool())
    assert mx.array_equal(captured, expected)
    assert len(graph.stats.temporaries) >= 3

    eager_pool, replay_pool = zero_pool(), zero_pool()
    history: list[tuple[mx.array, mx.array]] = []
    for seed in range(2, 8):
        x, rows, positions = step_inputs(seed)
        expected, eager_pool = step.forward(x, rows, positions, eager_pool)
        logits, (replay_pool,) = replay(graph, [x, rows, positions], [replay_pool])
        mx.eval(expected, eager_pool, logits, replay_pool)
        assert mx.array_equal(logits, expected)
        assert mx.array_equal(replay_pool, eager_pool)
        history.append((logits, expected))
    # Each replay writes fresh logits: earlier results are not overwritten.
    assert all(mx.array_equal(logits, expected) for logits, expected in history)


def test_double_capture_accepts_new_step_inputs() -> None:
    step = TinyStep()
    first, _ = capture(step, step_inputs(1), zero_pool())
    second, _ = capture(step, step_inputs(2), zero_pool())
    check_same_graph(first, second)


@pytest.mark.parametrize(
    ("rope_offsets", "scales", "message"),
    [
        ((3, 4), (SCALE, SCALE), "constant leaf"),
        ((None, None), (SCALE, 2 * SCALE), "setBytes payload"),
    ],
)
def test_double_capture_catches_per_step_values_passed_by_value(
    rope_offsets: tuple[int | None, int | None], scales: tuple[float, float], message: str
) -> None:
    step = TinyStep()
    inputs = step_inputs(1)
    first, _ = capture(step, inputs, zero_pool(), rope_offsets[0], scales[0])
    second, _ = capture(step, inputs, zero_pool(), rope_offsets[1], scales[1])
    with pytest.raises(RuntimeError, match=message):
        check_same_graph(first, second)


def test_capture_fails_on_a_buffer_outside_the_graph() -> None:
    step = TinyStep()
    pool = zero_pool()
    x, rows, positions = step_inputs(1)
    logits, updated = step.forward(x, rows, positions, pool)
    unrelated = x * 3
    begin_capture([logits, updated], [x, rows, positions], [pool], step.weights())
    mx.eval(logits, updated, unrelated)
    with pytest.raises(RuntimeError, match="neither whitelisted"):
        end_capture()
