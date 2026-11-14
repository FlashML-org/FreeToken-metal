"""Headless GPU performance counters (occupancy, limiters, bandwidth) from an Instruments
trace recorded with the profile-13 template of make_counter_template.py.

    uv run python tools/profiling/xctrace_counters.py x.trace [--process python] \\
        [--from S --to S] [--all] [--tsv out.tsv]

Exports gpu-counter-info + gpu-counter-value from the trace (xctrace export; ~0.5 GB of
XML per recorded second, several minutes for a 4 s trace: record 1-3 s), streams the
rows and prints, per counter: mean, p50, p90, max over the kept samples. Samples are kept
if they fall inside [--from, --to] trace seconds, and, with --process, only inside that
process's GPU intervals (metal-gpu-intervals), so idle gaps between a microbenchmark's
kernels do not dilute the averages. Without --all only the counters that matter for
compute kernels are printed (occupancy, limiters, utilization, bandwidth, L1 residency).
Exported XML lands next to the trace and later runs reuse it; an interrupted or failed
export leaves none behind. Needs Xcode, as xctrace_intervals.py does.

How to read it (WWDC20 10603, tech talk 111374): the *limiter* with the highest % is the
unit the kernel is bound by (F32/F16/Integer = ALU, L1 Cache = loads/stores/threadgroup,
Last Level Cache = DRAM-side traffic, Control Flow, Instruction Throughput, MMU, Texture).
Utilization < limiter means the unit is stalled rather than busy. Kernel Occupancy is the
% of SIMD-group slots in use; low occupancy + high L1 Register/Stack Residency = register
pressure / spills. GPU Read/Write Bandwidth is in GB/s, against about 257 GB/s of measured
streaming bandwidth on the M4 Pro.
"""

from __future__ import annotations

import argparse
import bisect
import re
import sys
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import astuple, dataclass
from pathlib import Path

from xctrace_intervals import (
    GPU_INTERVALS_TABLE,
    export_table,
    iter_rows,
    merge_intervals,
    require_xcode,
)

ROW = re.compile(rb"<row>(.*?)</row>", re.S)
ELEMENT = re.compile(rb'<([\w-]+)((?: [\w-]+="[^"]*")*)\s*(?:/>|>([^<]*)</\1>)')
ATTRIBUTE_ID = re.compile(rb' id="(\d+)"')
ATTRIBUTE_REF = re.compile(rb' ref="(\d+)"')
ATTRIBUTE_FMT = re.compile(rb' fmt="([^"]*)"')
KEEP = re.compile(
    r"Occupancy|Limiter|Utilization|Bandwidth|Residency|SIMD Groups|Eviction|Miss Rate"
    r"|Threads Launched"
)
READ_CHUNK_BYTES = 1 << 24


@dataclass(frozen=True)
class Counter:
    """One hardware counter as gpu-counter-info describes it."""

    name: str
    unit: str
    description: str


@dataclass(frozen=True)
class CounterSummary:
    """Statistics of one counter's kept samples, in the column order of the TSV."""

    name: str
    unit: str
    samples: int
    mean: float
    p50: float
    p90: float
    maximum: float
    description: str


def export_cached(trace: Path, table: str) -> Path:
    """The table's XML next to the trace, exported unless an earlier run left it there."""
    xml_path = trace.with_suffix(f".{table}.xml")
    if xml_path.exists() and xml_path.stat().st_size > 0:
        print(f"# reusing {xml_path}", file=sys.stderr)
    else:
        print(f"# exporting {table} -> {xml_path}", file=sys.stderr, flush=True)
        export_table(trace, table, xml_path)
    return xml_path


def iter_flat_rows(xml_path: Path) -> Iterator[list[bytes]]:
    """Each row's values with id/ref resolved, streamed with regular expressions: the counter
    tables are gigabytes and flat (no nested elements)."""
    definitions: dict[bytes, bytes] = {}
    pending = b""
    with xml_path.open("rb") as xml_file:
        while chunk := xml_file.read(READ_CHUNK_BYTES):
            pending += chunk
            consumed = 0
            for row in ROW.finditer(pending):
                values: list[bytes] = []
                for element in ELEMENT.finditer(row.group(1)):
                    attributes = element.group(2)
                    reference = ATTRIBUTE_REF.search(attributes)
                    if reference:
                        values.append(definitions.get(reference.group(1), b""))
                    else:
                        label = ATTRIBUTE_FMT.search(attributes)
                        text = element.group(3) or (label.group(1) if label else b"")
                        identifier = ATTRIBUTE_ID.search(attributes)
                        if identifier:
                            definitions[identifier.group(1)] = text
                        values.append(text)
                yield values
                consumed = row.end()
            pending = pending[consumed:]


def gpu_busy_mask(trace: Path, process: str) -> list[tuple[int, int]]:
    """The merged GPU intervals, in trace nanoseconds, of processes whose label contains
    process."""
    spans: list[tuple[int, int]] = []
    for columns, cells in iter_rows(export_cached(trace, GPU_INTERVALS_TABLE)):
        row = dict(zip(columns, cells))
        if process in row["process"].label:
            start = int(row["start"].raw)
            spans.append((start, start + int(row["duration"].raw)))
    mask = merge_intervals(spans)
    if not mask:
        sys.exit(f"no GPU intervals for process containing {process!r}")
    busy_ms = sum(end - start for start, end in mask) / 1e6
    print(
        f"# masking to {len(mask)} GPU intervals of {process!r}, {busy_ms:.1f} ms busy",
        file=sys.stderr,
    )
    return mask


def percentile(sorted_values: list[float], fraction: float) -> float:
    """The value at that fraction of a sorted list."""
    return sorted_values[min(len(sorted_values) - 1, int(fraction * len(sorted_values)))]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("trace")
    parser.add_argument(
        "--process", help="substring of 'name (pid)'; keep samples inside its GPU intervals"
    )
    parser.add_argument("--from", dest="t_from", type=float, default=0.0, help="trace seconds")
    parser.add_argument("--to", dest="t_to", type=float, default=1e9)
    parser.add_argument("--all", action="store_true", help="print every counter")
    parser.add_argument("--tsv", help="write the full per-counter table")
    args = parser.parse_args()
    trace = Path(args.trace)

    require_xcode()
    counters: dict[bytes, Counter] = {}
    # Columns: timestamp, counter-id, name, max-value, accelerator-id, description,
    # group-index, type, ...
    for values in iter_flat_rows(export_cached(trace, "gpu-counter-info")):
        counters[values[1]] = Counter(
            name=values[2].decode(errors="replace").lstrip("0123456789."),
            unit=values[7].decode(),
            description=values[5].decode(errors="replace"),
        )
    if not counters:
        sys.exit(
            "no gpu-counter-info rows: the trace was not recorded with a counter profile that "
            "this GPU supports"
        )
    mask = gpu_busy_mask(trace, args.process) if args.process else []
    mask_starts = [start for start, _ in mask]

    samples: dict[bytes, list[float]] = defaultdict(list)
    seen = kept = 0
    from_ns, to_ns = int(args.t_from * 1e9), int(args.t_to * 1e9)
    for values in iter_flat_rows(export_cached(trace, "gpu-counter-value")):
        seen += 1
        timestamp_ns = int(values[0])
        if timestamp_ns < from_ns or timestamp_ns > to_ns:
            continue
        if mask:
            span = bisect.bisect_right(mask_starts, timestamp_ns) - 1
            if span < 0 or timestamp_ns >= mask[span][1]:
                continue
        kept += 1
        samples[values[1]].append(float(values[2]))
    print(f"# {seen} samples, {kept} kept, {len(samples)} counters")

    table: list[CounterSummary] = []
    for counter_id, values in samples.items():
        counter = counters.get(
            counter_id, Counter(name=counter_id.decode(), unit="", description="")
        )
        values.sort()
        table.append(
            CounterSummary(
                name=counter.name,
                unit=counter.unit,
                samples=len(values),
                mean=sum(values) / len(values),
                p50=percentile(values, 0.5),
                p90=percentile(values, 0.9),
                maximum=values[-1],
                description=counter.description,
            )
        )
    table.sort(key=lambda summary: summary.name)
    print(f"{'counter':<44} {'unit':<10} {'n':>7} {'mean':>10} {'p50':>10} {'p90':>10} {'max':>10}")
    for summary in table:
        if not args.all and not KEEP.search(summary.name):
            continue
        unit = "%" if summary.unit == "Percentage" else summary.unit[:10]
        print(
            f"{summary.name:<44} {unit:<10} {summary.samples:>7} {summary.mean:>10.2f} "
            f"{summary.p50:>10.2f} {summary.p90:>10.2f} {summary.maximum:>10.2f}"
        )
    if args.tsv:
        with Path(args.tsv).open("w") as tsv:
            tsv.write("counter\tunit\tn\tmean\tp50\tp90\tmax\tdescription\n")
            for summary in table:
                tsv.write("\t".join(str(value) for value in astuple(summary)) + "\n")
        print(f"# wrote {args.tsv}")


if __name__ == "__main__":
    main()
