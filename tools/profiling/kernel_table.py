"""Per-kernel GPU time table for any Metal process (MLX, torch-MPS, llama.cpp), no Xcode.

Runs a command under the mtlstat dylib with one kernel per command buffer, then turns
the per-command-buffer log (cblog.<pid>.tsv: gpu_start, gpu_end, n_dispatch, kernels)
into: kernel, count, total GPU us, mean, min, share %.

    uv run python tools/profiling/kernel_table.py [options] -- \\
        uv run python -m benchmarks.kernel.attention.mha.bench_mha_decode --lengths 2048
    uv run python tools/profiling/kernel_table.py --parse cblog.1234.tsv   # re-print a saved log
    uv run python tools/profiling/kernel_table.py --diff before.json after.json   # two --json

Options:
    --stat-dir DIR    where mtlstat writes (default: a fresh temp dir, printed)
    --top N           rows to print (default 30)
    --overhead-us X   subtract X us from every command buffer (a buffer holding one
                      trivial kernel takes about 1.9 us on a warm M4 Pro; default 0)
    --json PATH       also write the table as JSON (input for --diff)
    --no-isolate      do not set MLX_MAX_OPS_PER_BUFFER=0 (buffers then hold several
                      kernels and rows become kernel *groups*, time is per group)
    --dylib PATH      mtlstat dylib (default ~/.cache/ftmetal/profiling/libmtlstat.dylib,
                      built by tools/profiling/build.sh)

Reading the table: "total" is the sum of GPU execution time of the command buffers that
held that kernel (GPUEndTime - GPUStartTime, the GPU's own clock), so it includes the
per-buffer fixed cost and excludes host gaps. "busy union" is the union of all buffers'
GPU intervals; "wall" is first start to last end, setup and first-use compiles included.
busy/wall measures host starvation only with --no-isolate on a command that is all steady
state, since one kernel per buffer adds host work of its own. Rows named "(empty)" are
command buffers with no compute dispatch (under MLX the end-of-eval, in-flight throttle
and mx.synchronize commits); some take GPU time, some none. Put the workload itself
after --: a SIP-protected launcher (/bin/sh, /usr/bin/env, the system python) drops
DYLD_INSERT_LIBRARIES, and then nothing is logged.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

DEFAULT_DYLIB = Path.home() / ".cache/ftmetal/profiling/libmtlstat.dylib"
EMPTY_BUFFER = "(empty)"
# System Integrity Protection: dyld drops DYLD_INSERT_LIBRARIES for binaries under these.
SIP_PROTECTED_PREFIXES = ("/bin/", "/sbin/", "/usr/bin/", "/usr/sbin/", "/usr/libexec/", "/System/")
SHEBANG_BYTES = 256


@dataclass(kw_only=True)
class KernelStats:
    """GPU time of the command buffers that held one kernel (or one group of kernels)."""

    count: int = 0
    total_us: float = 0.0
    min_us: float = math.inf
    max_us: float = 0.0


@dataclass(kw_only=True)
class KernelTable:
    """One command buffer log, summarised per kernel; its JSON form is what --diff reads."""

    file: str
    buffers: int
    empty_buffers: int
    wall_s: float
    busy_s: float
    kernels: dict[str, KernelStats] = field(default_factory=dict)


def mtlstat_environment(dylib: Path, stat_dir: Path, isolate: bool) -> dict[str, str]:
    """This process's environment plus what runs a child under mtlstat with the kernel log on.

    isolate puts one MLX primitive in each command buffer, which attributes buffer time to it.
    """
    if not dylib.exists():
        sys.exit(f"{dylib} not found: build it with sh tools/profiling/build.sh, or pass --dylib")
    environment = dict(
        os.environ,
        DYLD_INSERT_LIBRARIES=str(dylib),
        MTLSTAT_DIR=str(stat_dir),
        MTLSTAT_CBLOG="1",
    )
    if isolate:
        environment["MLX_MAX_OPS_PER_BUFFER"] = "0"
    return environment


def parse_cblog(
    path: Path, overhead_us: float, start_s: float = 0.0, end_s: float = math.inf
) -> KernelTable:
    """Per-kernel GPU time of the command buffers in a cblog that ran inside [start_s, end_s]."""
    kernels: dict[str, KernelStats] = {}
    intervals: list[tuple[float, float]] = []
    empty_buffers = 0
    with path.open() as cblog:
        for line in cblog:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 3:
                continue
            start, end, dispatches = float(fields[0]), float(fields[1]), int(fields[2])
            if end < start or start <= 0 or start < start_s or end > end_s:
                continue
            duration_us = max(0.0, (end - start) * 1e6 - overhead_us)
            intervals.append((start, end))
            if dispatches == 0:
                empty_buffers += 1
            name = fields[3] if dispatches > 0 and len(fields) > 3 else EMPTY_BUFFER
            stats = kernels.setdefault(name, KernelStats())
            stats.count += 1
            stats.total_us += duration_us
            stats.min_us = min(stats.min_us, duration_us)
            stats.max_us = max(stats.max_us, duration_us)
    intervals.sort()
    busy_s = 0.0
    last_end = -1.0
    for start, end in intervals:
        busy_s += max(0.0, end - max(start, last_end))
        last_end = max(last_end, end)
    wall_s = last_end - intervals[0][0] if intervals else 0.0
    return KernelTable(
        file=str(path),
        buffers=len(intervals),
        empty_buffers=empty_buffers,
        wall_s=wall_s,
        busy_s=busy_s,
        kernels=kernels,
    )


def shorten(name: str, width: int) -> str:
    """The name cut to width characters, marked with an ellipsis when cut."""
    return name if len(name) <= width else name[: width - 3] + "..."


def print_table(table: KernelTable, top: int) -> None:
    """The top kernels by total GPU time, with the log's buffer and busy totals."""
    total_us = sum(stats.total_us for stats in table.kernels.values()) or 1.0
    rows = sorted(table.kernels.items(), key=lambda row: -row[1].total_us)
    print(f"# {table.file}")
    print(
        f"# buffers {table.buffers} (empty {table.empty_buffers}), "
        f"GPU sum {total_us / 1e3:.2f} ms, busy union {table.busy_s * 1e3:.2f} ms, "
        f"wall {table.wall_s * 1e3:.2f} ms, "
        f"busy/wall {100 * table.busy_s / max(table.wall_s, 1e-12):.1f}%"
    )
    print(f"{'kernel':<72} {'count':>7} {'total_us':>11} {'mean_us':>9} {'min_us':>9} {'share':>6}")
    for name, stats in rows[:top]:
        print(
            f"{shorten(name, 72):<72} {stats.count:>7} {stats.total_us:>11.1f} "
            f"{stats.total_us / stats.count:>9.2f} {stats.min_us:>9.2f} "
            f"{100 * stats.total_us / total_us:>5.1f}%"
        )
    if len(rows) > top:
        rest_us = sum(stats.total_us for _, stats in rows[top:])
        more = f"... {len(rows) - top} more"
        print(
            f"{more:<72} {'':>7} {rest_us:>11.1f} {'':>9} {'':>9} {100 * rest_us / total_us:>5.1f}%"
        )


def load_kernels(path: Path) -> dict[str, KernelStats]:
    """The per-kernel stats of a table written by --json."""
    document = json.loads(path.read_text())
    return {name: KernelStats(**stats) for name, stats in document["kernels"].items()}


def print_diff(before_path: Path, after_path: Path, top: int) -> None:
    """Counts and total GPU time per kernel of two --json tables, largest kernels first."""
    before = load_kernels(before_path)
    after = load_kernels(after_path)
    missing = KernelStats()
    names = sorted(
        before.keys() | after.keys(),
        key=lambda name: -(after.get(name, missing).total_us + before.get(name, missing).total_us),
    )
    before_us = sum(stats.total_us for stats in before.values())
    after_us = sum(stats.total_us for stats in after.values())
    print(
        f"# A={before_path} GPU sum {before_us / 1e3:.2f} ms;  "
        f"B={after_path} GPU sum {after_us / 1e3:.2f} ms;  "
        f"delta {(after_us - before_us) / 1e3:+.2f} ms"
    )
    print(
        f"{'kernel':<64} {'cnt A':>6} {'cnt B':>6} {'tot A us':>10} {'tot B us':>10} "
        f"{'delta us':>10}"
    )
    for name in names[:top]:
        old, new = before.get(name, missing), after.get(name, missing)
        print(
            f"{shorten(name, 64):<64} {old.count:>6} {new.count:>6} {old.total_us:>10.1f} "
            f"{new.total_us:>10.1f} {new.total_us - old.total_us:>+10.1f}"
        )


def profile_command(
    command: list[str], stat_dir: Path, dylib: Path, isolate: bool, overhead_us: float
) -> KernelTable:
    """Run the command under mtlstat and parse the busiest process's command buffer log."""
    executable = shutil.which(command[0])
    if executable is None:
        launchers = []
    else:
        with Path(executable).open("rb") as program:
            first_line = program.readline(SHEBANG_BYTES)
        shebang = first_line[2:].split()[:1] if first_line.startswith(b"#!") else []
        launchers = [executable, os.path.realpath(executable), *(word.decode() for word in shebang)]
    protected = [launcher for launcher in launchers if launcher.startswith(SIP_PROTECTED_PREFIXES)]
    if protected:
        print(
            f"# warning: the command goes through {protected[0]}, which SIP protects: dyld "
            "drops DYLD_INSERT_LIBRARIES there, so nothing is logged. Put the workload itself "
            "after -- (uv run python ... works).",
            file=sys.stderr,
            flush=True,
        )
    stat_dir.mkdir(parents=True, exist_ok=True)
    before = set(stat_dir.glob("cblog.*.tsv"))
    print(f"# running under mtlstat (stat dir {stat_dir}): {' '.join(command)}", flush=True)
    exit_code = subprocess.call(command, env=mtlstat_environment(dylib, stat_dir, isolate))
    print(f"# exit {exit_code}", flush=True)
    logs = set(stat_dir.glob("cblog.*.tsv")) - before
    if not any(log.stat().st_size for log in logs):
        sys.exit(
            "no command buffer logged: the command committed no Metal work, or it reached the "
            "process that does through a SIP-protected binary (/bin/sh, /usr/bin/env, the system "
            "python), which drops DYLD_INSERT_LIBRARIES (MTLSTAT_VERBOSE=1 makes every process "
            "the dylib loads into print '[mtlstat] pid ...')"
        )
    return parse_cblog(max(logs, key=lambda log: log.stat().st_size), overhead_us)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--stat-dir")
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument("--overhead-us", type=float, default=0.0)
    parser.add_argument("--json")
    parser.add_argument("--no-isolate", action="store_true")
    parser.add_argument("--dylib", default=str(DEFAULT_DYLIB))
    parser.add_argument("--parse", metavar="CBLOG")
    parser.add_argument("--diff", nargs=2, metavar=("A.json", "B.json"))
    parser.add_argument("command", nargs=argparse.REMAINDER, help="-- command to profile")
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not (args.diff or args.parse or command):
        parser.error("give a command after --, or --parse / --diff")

    if args.diff:
        print_diff(Path(args.diff[0]), Path(args.diff[1]), args.top)
    else:
        if args.parse:
            table = parse_cblog(Path(args.parse), args.overhead_us)
        else:
            stat_dir = Path(args.stat_dir or tempfile.mkdtemp(prefix="mtlstat."))
            table = profile_command(
                command, stat_dir, Path(args.dylib), not args.no_isolate, args.overhead_us
            )
        print_table(table, args.top)
        if args.json:
            Path(args.json).write_text(json.dumps(asdict(table), indent=1))
            print(f"# wrote {args.json}")


if __name__ == "__main__":
    main()
