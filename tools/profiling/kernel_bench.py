"""Single-kernel microbenchmark with roofline placement.

    uv run python tools/profiling/kernel_bench.py \\
        --setup "a = mx.random.normal((4096, 4096)).astype(mx.bfloat16)" \\
        --expr "a @ a" --bytes 100663296 --flops 137438953472 [--gpu]
    uv run python tools/profiling/kernel_bench.py --time-us 18000 --bytes 1e8 --flops 1.37e11

The second form only places a time on the roofline. Timing: --warmup-s seconds of a
2048x2048 matmul loop, because an idle GPU clocks down (a 9.4 us kernel read 36 us cold),
then --warmup evaluations, then --repeats rounds of --iters evaluations each; a round is
mx.synchronize() -> iters x mx.eval(expr) -> mx.synchronize(). Wall time is the best
round's mean us per evaluation: GPU time plus one blocking submit-to-completion round trip
per eval (about 100 us on the M4 Pro), so it overstates any kernel under a few ms.
--gpu runs the benchmark in a child process under the mtlstat dylib (--dylib, default
~/.cache/ftmetal/profiling/libmtlstat.dylib) with one kernel per command buffer, which
also slows its wall loop, and adds the GPU-clock time of the same rounds: the best round's
mean of its non-empty command buffers, in total and per kernel (trust this one for the
roofline). It includes each command buffer's fixed cost, about 1.9 us for one trivial
kernel on a warm M4 Pro. Inputs small enough to stay in the GPU's caches between
evaluations can read above the DRAM ceiling. mx is in scope for --setup / --expr; --setup
may be given several times. --peak-bw / --peak-tf override the M4 Pro's measured ceilings.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import CodeType

import mlx.core as mx
from kernel_table import (
    DEFAULT_DYLIB,
    EMPTY_BUFFER,
    KernelTable,
    mtlstat_environment,
    parse_cblog,
    shorten,
)

PEAK_BANDWIDTH_GB_S = 257.0  # measured DRAM streaming on the M4 Pro
PEAK_TFLOPS = 7.8  # measured bf16 simdgroup-matrix peak on the M4 Pro
WARM_MATMUL_SIZE = 2048
# Long enough for mtlstat's completion handlers and its 20 ms flush to land in the log.
FLUSH_WAIT_S = 0.3


def print_roofline(
    time_us: float,
    bytes_moved: float | None,
    flops: float | None,
    peak_bandwidth_gb_s: float,
    peak_tflops: float,
) -> None:
    """Achieved bandwidth and FLOP rate of one call against the ceilings, and its bound."""
    seconds = time_us * 1e-6
    peak_bandwidth = peak_bandwidth_gb_s * 1e9
    peak_flops = peak_tflops * 1e12
    balance = peak_flops / peak_bandwidth
    print(
        f"time {time_us:.2f} us; machine balance {balance:.1f} FLOP/byte "
        f"(peak {peak_tflops:.1f} TF/s, {peak_bandwidth_gb_s:.0f} GB/s)"
    )
    if bytes_moved:
        print(
            f"  bytes {bytes_moved / 1e6:.2f} MB -> {bytes_moved / seconds / 1e9:.1f} GB/s = "
            f"{100 * bytes_moved / seconds / peak_bandwidth:.1f}% of DRAM ceiling"
            f" (memory floor {bytes_moved / peak_bandwidth * 1e6:.2f} us)"
        )
    if flops:
        print(
            f"  flops {flops / 1e9:.2f} GFLOP -> {flops / seconds / 1e12:.2f} TF/s = "
            f"{100 * flops / seconds / peak_flops:.1f}% of MMA ceiling"
            f" (compute floor {flops / peak_flops * 1e6:.2f} us)"
        )
    if bytes_moved and flops:
        intensity = flops / bytes_moved
        regime = "compute-bound" if intensity > balance else "memory-bound"
        floor_us = max(bytes_moved / peak_bandwidth, flops / peak_flops) * 1e6
        print(
            f"  arithmetic intensity {intensity:.1f} FLOP/byte -> roofline says {regime}; "
            f"roofline floor {floor_us:.2f} us, "
            f"achieved {100 * floor_us / time_us:.1f}% of roofline"
        )


def evaluate(expression: CodeType, namespace: dict[str, object]) -> None:
    """Evaluate the expression and its result: an array, a tuple or list of them, or None."""
    outputs = eval(expression, namespace)
    if isinstance(outputs, (tuple, list)):
        mx.eval(*outputs)
    elif outputs is not None:
        mx.eval(outputs)


def benchmark(
    *,
    setup: list[str],
    expression_text: str,
    warmup_s: float,
    warmup: int,
    repeats: int,
    iterations: int,
    bytes_moved: float | None,
    flops: float | None,
    peak_bandwidth_gb_s: float,
    peak_tflops: float,
    stat_dir: Path | None,
) -> None:
    """Time the expression and place it on the roofline; with a stat dir (the process runs
    under mtlstat) also read the GPU time of its command buffers."""
    namespace: dict[str, object] = {"mx": mx}
    for statement in setup:
        exec(statement, namespace)
    expression = compile(expression_text, "<expr>", "eval")
    load = mx.random.normal((WARM_MATMUL_SIZE, WARM_MATMUL_SIZE))
    warm_until_s = time.perf_counter() + warmup_s
    while time.perf_counter() < warm_until_s:
        mx.eval(load @ load)
    for _ in range(warmup):
        evaluate(expression, namespace)
    mx.synchronize()
    # mach_absolute_time, the clock of the GPU timestamps, is CLOCK_UPTIME_RAW.
    rounds_s: list[tuple[float, float]] = []
    for _ in range(repeats):
        mx.synchronize()
        started_s = time.clock_gettime(time.CLOCK_UPTIME_RAW)
        for _ in range(iterations):
            evaluate(expression, namespace)
        mx.synchronize()
        rounds_s.append((started_s, time.clock_gettime(time.CLOCK_UPTIME_RAW)))
    wall_us = min(end - start for start, end in rounds_s) / iterations * 1e6
    isolation = "; one kernel per command buffer under mtlstat" if stat_dir else ""
    print(
        f"wall: {wall_us:.2f} us/eval (best round's mean, {repeats} rounds of "
        f"{iterations} evals{isolation})"
    )
    print_roofline(wall_us, bytes_moved, flops, peak_bandwidth_gb_s, peak_tflops)
    if stat_dir is None:
        return

    time.sleep(FLUSH_WAIT_S)
    cblog = stat_dir / f"cblog.{os.getpid()}.tsv"
    if not cblog.exists():
        sys.exit(
            f"no command buffer log in {stat_dir}: the mtlstat dylib did not load into "
            f"{sys.executable} (an interpreter under /usr/bin or /System drops "
            "DYLD_INSERT_LIBRARIES)"
        )
    best_round: KernelTable | None = None
    best_us = float("inf")
    for start_s, end_s in rounds_s:
        table = parse_cblog(cblog, 0.0, start_s, end_s)
        table.kernels.pop(EMPTY_BUFFER, None)
        round_us = sum(stats.total_us for stats in table.kernels.values()) / iterations
        if table.kernels and round_us < best_us:
            best_round, best_us = table, round_us
    if best_round is None:
        sys.exit(f"no command buffer with a dispatch ran inside the timed rounds ({cblog})")
    print(
        f"GPU clock: {best_us:.2f} us/eval (best round's mean of its non-empty command "
        "buffers, one kernel per buffer)"
    )
    for name, stats in sorted(best_round.kernels.items(), key=lambda row: -row[1].total_us):
        print(
            f"  {shorten(name, 70):<70} {stats.count / iterations:6.2f} launches/eval  "
            f"{stats.total_us / iterations:9.2f} us/eval  (min {stats.min_us:.2f})"
        )
    if bytes_moved or flops:
        print("roofline on GPU-clock time:")
        print_roofline(best_us, bytes_moved, flops, peak_bandwidth_gb_s, peak_tflops)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--setup", action="append", default=[], help="python statements run once first"
    )
    parser.add_argument("--expr", help="python expression returning the mx.array(s) to time")
    parser.add_argument(
        "--warmup-s",
        type=float,
        default=1.0,
        help="seconds of a matmul loop first, to raise the GPU clock (default 1; 0: cold GPU)",
    )
    parser.add_argument(
        "--warmup", type=int, default=3, help="evaluations before timing (default 3)"
    )
    parser.add_argument("--repeats", type=int, default=10, help="timed rounds (default 10)")
    parser.add_argument("--iters", type=int, default=20, help="evaluations per round (default 20)")
    parser.add_argument("--bytes", type=float, help="DRAM bytes one evaluation moves")
    parser.add_argument("--flops", type=float, help="FLOPs of one evaluation")
    parser.add_argument(
        "--peak-bw",
        type=float,
        default=PEAK_BANDWIDTH_GB_S,
        help=f"DRAM ceiling in GB/s (default {PEAK_BANDWIDTH_GB_S:g})",
    )
    parser.add_argument(
        "--peak-tf",
        type=float,
        default=PEAK_TFLOPS,
        help=f"MMA ceiling in TF/s (default {PEAK_TFLOPS:g})",
    )
    parser.add_argument(
        "--time-us", type=float, help="skip running; place this time on the roofline"
    )
    parser.add_argument(
        "--gpu", action="store_true", help="also measure GPU-clock time per kernel via mtlstat"
    )
    parser.add_argument("--dylib", default=str(DEFAULT_DYLIB), help="mtlstat dylib for --gpu")
    # Set on the run under mtlstat that --gpu starts.
    parser.add_argument("--mtlstat-dir", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not (args.time_us or args.expr):
        parser.error("--expr or --time-us")

    if args.time_us:
        print_roofline(args.time_us, args.bytes, args.flops, args.peak_bw, args.peak_tf)
    elif args.gpu and args.mtlstat_dir is None:
        stat_dir = Path(tempfile.mkdtemp(prefix="kernel_bench."))
        try:
            environment = mtlstat_environment(Path(args.dylib), stat_dir, isolate=True)
            command = [sys.executable, *sys.argv, "--mtlstat-dir", str(stat_dir)]
            exit_code = subprocess.call(command, env=environment)
        finally:
            shutil.rmtree(stat_dir)
        sys.exit(exit_code)
    else:
        benchmark(
            setup=args.setup,
            expression_text=args.expr,
            warmup_s=args.warmup_s,
            warmup=args.warmup,
            repeats=args.repeats,
            iterations=args.iters,
            bytes_moved=args.bytes,
            flops=args.flops,
            peak_bandwidth_gb_s=args.peak_bw,
            peak_tflops=args.peak_tf,
            stat_dir=Path(args.mtlstat_dir) if args.mtlstat_dir else None,
        )


if __name__ == "__main__":
    main()
