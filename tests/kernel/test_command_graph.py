"""CommandGraph capture and replay on a small synthetic step, and the arena planner on
random tapes: no checkpoint needed."""

import random

import mlx.core as mx
import pytest

from ftmetal.kernel._ext import _C
from ftmetal.kernel.command_graph import (
    ARENA_ALIGNMENT,
    begin_capture,
    check_same_graph,
    end_capture,
    place_blocks,
    plan_arena,
    plan_graph_arena,
    replay,
)
from ftmetal.kernel.custom import metal_kernel
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


def random_tape(
    rng: random.Random,
) -> tuple[dict[int, int], list[list[int]], list[bool]]:
    """Blocks of random sizes, each bound by one to four dispatches of a random tape."""
    num_dispatches = rng.randint(1, 40)
    num_blocks = rng.randint(1, 30)
    block_sizes = {block: rng.randint(1, 3 * ARENA_ALIGNMENT) for block in range(num_blocks)}
    dispatch_blocks: list[list[int]] = [[] for _ in range(num_dispatches)]
    for block in block_sizes:
        for d in rng.sample(range(num_dispatches), min(num_dispatches, rng.randint(1, 4))):
            dispatch_blocks[d].append(block)
    separated = [rng.random() < 0.3 for _ in range(num_dispatches)]
    return block_sizes, dispatch_blocks, separated


@pytest.mark.parametrize("seed", range(200))
def test_plan_arena_never_shares_bytes_without_a_barrier_between(seed: int) -> None:
    rng = random.Random(seed)
    block_sizes, dispatch_blocks, separated = random_tape(rng)
    # A block the caller leaves out (pinned) is not placed even though the tape binds it.
    pinned = rng.choice(list(block_sizes))
    placed_sizes = {block: size for block, size in block_sizes.items() if block != pinned}
    plan = plan_arena(placed_sizes, dispatch_blocks, separated)

    uses = {
        block: [d for d, blocks in enumerate(dispatch_blocks) if block in blocks]
        for block in placed_sizes
    }
    assert plan.placements.keys() == {block for block in placed_sizes if uses[block]}
    for block, (offset, size) in plan.placements.items():
        assert offset % ARENA_ALIGNMENT == 0 and size % ARENA_ALIGNMENT == 0
        assert size >= placed_sizes[block] and offset + size <= plan.arena_bytes

    def barrier_between(before: int, after: int) -> bool:
        return any(separated[d] for d in range(before + 1, after + 1))

    for a, (offset_a, size_a) in plan.placements.items():
        for b, (offset_b, size_b) in plan.placements.items():
            if a < b and offset_a < offset_b + size_b and offset_b < offset_a + size_a:
                assert barrier_between(max(uses[a]), min(uses[b])) or barrier_between(
                    max(uses[b]), min(uses[a])
                ), f"blocks {a} and {b} share bytes while both may be live"
    assert plan.lower_bound <= plan.arena_bytes
    assert (plan.lower_bound > 0) == bool(plan.placements)


def test_plan_arena_ping_pongs_a_residual_stream() -> None:
    """x[i + 1] = x[i] + f(norm(x[i])), one barrier between dispatches: the stream takes
    two slots and the arena is the lower bound."""
    size = 4 * ARENA_ALIGNMENT
    num_layers = 6
    dispatch_blocks: list[list[int]] = []
    stream = [0]
    block_sizes = {0: size}
    for _ in range(num_layers):
        x = stream[-1]
        normed, update, out = range(len(block_sizes), len(block_sizes) + 3)
        block_sizes.update({normed: size, update: size, out: size})
        dispatch_blocks += [[x, normed], [normed, update], [x, update, out]]
        stream.append(out)
    plan = plan_arena(block_sizes, dispatch_blocks, [True] * len(dispatch_blocks))
    offsets = [plan.placements[block].offset for block in stream]
    assert len(set(offsets)) == 2
    assert all(offsets[i] == offsets[i + 2] != offsets[i + 1] for i in range(len(offsets) - 2))
    assert plan.arena_bytes == plan.lower_bound == 3 * size


def arena_for(graph: _C.Graph) -> mx.array:
    """Plan and place graph's blocks, bind them to a fresh arena filled with 0xFF."""
    plan = plan_graph_arena(graph, [])
    place_blocks(graph, plan)
    arena = mx.zeros((plan.arena_bytes,), dtype=mx.uint8)
    mx.eval(arena)
    _C.bind_arena(graph, arena)
    _C.poison_arena(graph)
    return arena


def test_arena_replay_matches_eager_and_releases_the_capture() -> None:
    step = TinyStep()
    graph, _ = capture(step, step_inputs(1), zero_pool())
    stats = graph.stats
    assert stats.pinned_bytes == stats.capture_pinned_bytes > 0
    assert len(graph.temporary_blocks) >= 3

    plan = plan_graph_arena(graph, [])
    place_blocks(graph, plan)
    assert graph.stats.arena_bytes <= plan.arena_bytes <= sum(graph.block_sizes) + ARENA_ALIGNMENT
    assert graph.stats.pinned_bytes < stats.capture_pinned_bytes
    assert graph.stats.num_pinned_temporaries == 0
    with pytest.raises(ValueError, match="bind an arena"):
        replay(graph, step_inputs(2), [zero_pool()])
    arena = mx.zeros((plan.arena_bytes,), dtype=mx.uint8)
    mx.eval(arena)
    _C.bind_arena(graph, arena)
    _C.poison_arena(graph)

    eager_pool, replay_pool = zero_pool(), zero_pool()
    for seed in range(2, 6):
        x, rows, positions = step_inputs(seed)
        expected, eager_pool = step.forward(x, rows, positions, eager_pool)
        logits, (replay_pool,) = replay(graph, [x, rows, positions], [replay_pool])
        mx.eval(expected, eager_pool, logits, replay_pool)
        assert mx.array_equal(logits, expected)
        assert mx.array_equal(replay_pool, eager_pool)


# Writes out[0] only: the other elements keep the init_value MLX fills them with, from a
# scalar temporary the host writes at eval time.
FIRST_ELEMENT_KERNEL = metal_kernel(
    name="first_element",
    input_names=["x"],
    output_names=["out"],
    source="if (thread_position_in_grid.x == 0) { out[0] = x[0]; }",
)


def test_poison_finds_a_host_initialized_temporary() -> None:
    """A temporary the replay reads before writing changes the logits when poisoned; pinned,
    it keeps them exact on an arena full of 0xFF."""
    x = mx.arange(8, dtype=mx.float32) + 1
    pool = mx.zeros((1, 8), dtype=mx.float32)
    rows = mx.array([0], dtype=mx.int32)
    mx.eval(x, pool, rows)

    def forward(pool: mx.array) -> tuple[mx.array, mx.array]:
        (first,) = FIRST_ELEMENT_KERNEL(
            inputs=[x * 2],
            grid=(32, 1, 1),
            threadgroup=(32, 1, 1),
            output_shapes=[(8,)],
            output_dtypes=[mx.float32],
            init_value=5.0,
        )
        logits = (first + x).reshape(1, 8)
        return logits, scatter_rows(pool, logits, rows)

    expected, _ = forward(pool)
    mx.eval(expected)
    logits, updated = forward(pool)
    begin_capture([logits, updated], [rows], [pool], [x])
    mx.eval(logits, updated)
    graph = end_capture()

    temporaries = graph.temporary_blocks
    changed = []
    for block in temporaries:
        _C.poison_blocks(graph, [block])
        poisoned, (pool,) = replay(graph, [rows], [pool])
        mx.eval(poisoned)
        _C.restore_blocks(graph)
        if not mx.array_equal(poisoned, expected):
            changed.append(block)
    assert len(changed) == 1 and graph.block_labels[changed[0]].startswith("temporary of ")

    place_blocks(graph, plan_graph_arena(graph, changed))
    assert graph.stats.num_pinned_temporaries == 1
    arena = mx.zeros((max(graph.stats.arena_bytes, ARENA_ALIGNMENT),), dtype=mx.uint8)
    mx.eval(arena)
    _C.bind_arena(graph, arena)
    _C.poison_arena(graph)
    logits, _ = replay(graph, [rows], [pool])
    assert mx.array_equal(logits, expected)
