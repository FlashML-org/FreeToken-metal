"""Instruments (xctrace) from the command line: record a Metal System Trace, export any
table as TSV with the id/ref compression resolved, and summarise encoder-level GPU
intervals (the timeline view: who ran on which GPU channel, busy %, biggest gaps).

    uv run python tools/profiling/xctrace_intervals.py record --out x.trace \\
        [--time-limit 5] [--template 'Metal System Trace'] \\
        (--attach PID | -- .venv/bin/python workload.py)
    uv run python tools/profiling/xctrace_intervals.py tables x.trace   # list the tables
    uv run python tools/profiling/xctrace_intervals.py export x.trace \\
        --table metal-gpu-intervals [--tsv out.tsv]
    uv run python tools/profiling/xctrace_intervals.py summary x.trace [--process python] \\
        [--gaps 10]

Needs Xcode: xctrace is not part of the Command Line Tools, so when xcode-select points at
them, set DEVELOPER_DIR to an Xcode install. record takes --time-limit in seconds (1.5 is
fine) and exits 0 when the recording completes, at the limit or on Ctrl-C; either kills the
launched command, so give that a loop longer than the limit. xctrace records host-side tables
(metal-application-intervals, time-profile) only for the process it launches, so launch
the interpreter itself, not uv run, which starts python as a child: record refuses uv.
A failed recording's partial trace is removed. export and summary keep the exported XML
next to the trace, as <trace>.<table>.xml.
Tables worth knowing: metal-gpu-intervals (GPU-side encoder intervals: start, duration,
channel Vertex/Fragment/Compute/Blit, process, encoder id), metal-application-intervals
(host-side encoding/commit per command buffer), metal-command-buffer-completed,
gpu-counter-value / metal-gpu-counter-intervals (hardware counters: the stock template
records only "RT Unit Active" on the M4 Pro; see make_counter_template.py).
"""

from __future__ import annotations

import argparse
import contextlib
import os
import shutil
import signal
import subprocess
import sys
import xml.etree.ElementTree as ElementTree
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple

GPU_INTERVALS_TABLE = "metal-gpu-intervals"
# xctrace record's exit status for a completed recording, ended by the time limit or Ctrl-C.
RECORDING_COMPLETED_EXIT = 54


class Cell(NamedTuple):
    """One exported value: its raw text (a number for times and counts) and its label."""

    raw: str
    label: str


def require_xcode() -> Path:
    """The developer directory xcrun resolves; exits when it has no xctrace (the Command Line
    Tools)."""
    found = subprocess.run(["xcrun", "--find", "xctrace"], capture_output=True, text=True)
    if found.returncode != 0:
        selected = subprocess.run(["xcode-select", "-p"], capture_output=True, text=True)
        active = os.environ.get("DEVELOPER_DIR") or selected.stdout.strip()
        sys.exit(
            f"xctrace is not in the active developer directory {active}: it ships with Xcode, "
            "not the Command Line Tools. Set DEVELOPER_DIR to an Xcode install, e.g. "
            "DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer"
        )
    return Path(found.stdout.strip()).parents[2]


def export_table(trace: Path, table: str, xml_path: Path) -> None:
    """Export one table of the trace's first run to an XML file, which appears only once the
    export is complete."""
    xpath = f'/trace-toc/run[@number="1"]/data/table[@schema="{table}"]'
    partial = xml_path.with_name(xml_path.name + ".partial")
    try:
        with partial.open("w") as xml_file:
            exported = subprocess.run(
                ["xcrun", "xctrace", "export", "--input", str(trace), "--xpath", xpath],
                stdout=xml_file,
                stderr=subprocess.PIPE,
                text=True,
            )
        if exported.returncode != 0:
            sys.exit(exported.stderr)
        partial.replace(xml_path)
    finally:
        partial.unlink(missing_ok=True)


def cell_of(element: ElementTree.Element) -> Cell:
    """The value an exported element carries."""
    return Cell((element.text or "").strip(), element.attrib.get("fmt", ""))


def iter_rows(xml_path: Path) -> Iterator[tuple[list[str], list[Cell]]]:
    """The schema's column mnemonics and each row's cells, with id/ref references resolved.

    A value carries id="N" the first time and later rows reference it with ref="N"; nested
    elements (formatted labels, the process's pid) define ids too.
    """
    definitions: dict[str, Cell] = {}
    columns: list[str] = []
    for _, element in ElementTree.iterparse(xml_path, events=("end",)):
        if element.tag == "schema":
            columns = [column.findtext("mnemonic", "") for column in element.findall("col")]
        elif element.tag == "row":
            cells: list[Cell] = []
            for child in element:
                if "ref" in child.attrib:
                    cells.append(definitions.get(child.attrib["ref"], Cell("", "")))
                else:
                    cells.append(cell_of(child))
                for nested in child.iter():
                    if "id" in nested.attrib:
                        definitions[nested.attrib["id"]] = cell_of(nested)
            yield columns, cells
            element.clear()


def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Sorted, disjoint intervals covering the same time; touching intervals merge."""
    merged: list[tuple[int, int]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def union_ns(intervals: list[tuple[int, int]]) -> int:
    """Time covered by at least one interval."""
    return sum(end - start for start, end in merge_intervals(intervals))


def record(
    output: Path, template: str, time_limit_s: float, attach_pid: int | None, command: list[str]
) -> None:
    """Record a trace of an attached process or a launched command; exits 0 when the recording
    completes, and removes the partial trace of a failed one."""
    if output.exists():
        sys.exit(f"{output} exists and xctrace does not overwrite a trace: remove it first")
    arguments = ["record", "--template", template, "--output", str(output)]
    arguments += ["--time-limit", f"{round(time_limit_s * 1000)}ms"]
    if attach_pid:
        arguments += ["--attach", str(attach_pid)]
    elif command:
        # xctrace does not search PATH.
        executable = shutil.which(command[0])
        if executable is None:
            sys.exit(f"{command[0]} not found on PATH")
        elif Path(executable).name in ("uv", "uvx"):
            sys.exit(
                "uv starts python as a child, and xctrace records the host-side tables "
                "(metal-application-intervals, time-profile) only for the process it launches: "
                "launch the interpreter itself, e.g. -- .venv/bin/python workload.py"
            )
        else:
            arguments += ["--launch", "--", executable, *command[1:]]
    else:
        sys.exit("give --attach PID or -- command")
    print("# xcrun xctrace", " ".join(arguments), flush=True)
    recording = subprocess.Popen(["xcrun", "xctrace", *arguments])
    # Ctrl-C also reaches xctrace, which then stops the recording and saves the trace.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    exit_code = recording.wait()
    if exit_code in (0, RECORDING_COMPLETED_EXIT):
        sys.exit(0)
    elif output.exists():
        shutil.rmtree(output)
        sys.exit(f"xctrace exited {exit_code}; removed the partial {output}")
    else:
        sys.exit(f"xctrace exited {exit_code}")


def list_tables(trace: Path) -> None:
    """Every table of the trace's table of contents, with its attributes."""
    toc = subprocess.run(
        ["xcrun", "xctrace", "export", "--input", str(trace), "--toc"],
        capture_output=True,
        text=True,
    )
    if toc.returncode != 0:
        sys.exit(toc.stderr)
    for table in ElementTree.fromstring(toc.stdout).iter("table"):
        attributes = {name: value for name, value in table.attrib.items() if name != "schema"}
        print(table.attrib.get("schema"), attributes)


def export(trace: Path, table: str, tsv: Path | None) -> None:
    """One table as TSV (numbers raw, everything else as its label), to a file or stdout."""
    xml_path = trace.with_suffix(f".{table}.xml")
    export_table(trace, table, xml_path)
    rows = 0
    with tsv.open("w") if tsv else contextlib.nullcontext(sys.stdout) as output:
        for columns, cells in iter_rows(xml_path):
            if rows == 0:
                output.write("\t".join(columns) + "\n")
            values = [
                cell.raw if cell.raw.replace(".", "").replace("-", "").isdigit() else cell.label
                for cell in cells
            ]
            output.write("\t".join(values) + "\n")
            rows += 1
    print(f"# {rows} rows ({table}); raw xml kept at {xml_path}", file=sys.stderr)


def summary(trace: Path, process: str | None, gaps: int) -> None:
    """GPU intervals per process and channel: count, time, busy share, CPU-to-GPU latency,
    and each process's largest GPU-idle gaps."""
    xml_path = trace.with_suffix(f".{GPU_INTERVALS_TABLE}.xml")
    export_table(trace, GPU_INTERVALS_TABLE, xml_path)
    intervals: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
    latencies: dict[tuple[str, str], list[int]] = defaultdict(list)
    for columns, cells in iter_rows(xml_path):
        row = dict(zip(columns, cells))
        process_label = row["process"].label
        if process and process not in process_label:
            continue
        start = int(row["start"].raw)
        key = (process_label, row["channel-name"].label)
        intervals[key].append((start, start + int(row["duration"].raw)))
        if row["start-latency"].raw.isdigit():
            latencies[key].append(int(row["start-latency"].raw))
    if not intervals:
        sys.exit("no GPU intervals matched (check --process; names look like 'python (1234)')")

    every_interval = [interval for spans in intervals.values() for interval in spans]
    window_ns = max(end for _, end in every_interval) - min(start for start, _ in every_interval)
    print(
        f"# {trace}: {len(every_interval)} GPU intervals, window {window_ns / 1e6:.1f} ms, "
        f"GPU busy (union, all channels) {100 * union_ns(every_interval) / window_ns:.1f}%"
    )
    print(
        f"{'process':<28} {'channel':<10} {'count':>7} {'sum_ms':>9} {'mean_us':>9} "
        f"{'busy%':>6} {'cpu->gpu_us':>12}"
    )
    by_time = sorted(intervals.items(), key=lambda row: -sum(end - start for start, end in row[1]))
    for (process_label, channel), spans in by_time:
        total_ns = sum(end - start for start, end in spans)
        latency = latencies[(process_label, channel)]
        latency_us = sum(latency) / len(latency) / 1e3 if latency else 0
        print(
            f"{process_label[:28]:<28} {channel:<10} {len(spans):>7} {total_ns / 1e6:>9.2f} "
            f"{total_ns / len(spans) / 1e3:>9.1f} {100 * union_ns(spans) / window_ns:>5.1f}% "
            f"{latency_us:>12.1f}"
        )
    if gaps:
        for process_label in sorted({label for label, _ in intervals}):
            busy = merge_intervals(
                [
                    interval
                    for (label, _), spans in intervals.items()
                    if label == process_label
                    for interval in spans
                ]
            )
            idle = sorted(
                ((start - end, end) for (_, end), (start, _) in zip(busy, busy[1:])),
                reverse=True,
            )
            print(f"# largest GPU-idle gaps for {process_label} (duration, at trace time):")
            for gap_ns, at_ns in idle[:gaps]:
                print(f"    {gap_ns / 1e3:9.1f} us  at {at_ns / 1e9:.6f} s")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subcommands = parser.add_subparsers(dest="subcommand", required=True)
    record_parser = subcommands.add_parser("record")
    record_parser.add_argument("--out", required=True)
    record_parser.add_argument("--template", default="Metal System Trace")
    record_parser.add_argument("--time-limit", type=float, default=5.0, help="seconds (default 5)")
    record_parser.add_argument("--attach", type=int)
    record_parser.add_argument("command", nargs=argparse.REMAINDER)
    tables_parser = subcommands.add_parser("tables")
    tables_parser.add_argument("trace")
    export_parser = subcommands.add_parser("export")
    export_parser.add_argument("trace")
    export_parser.add_argument("--table", required=True)
    export_parser.add_argument("--tsv")
    summary_parser = subcommands.add_parser("summary")
    summary_parser.add_argument("trace")
    summary_parser.add_argument("--process", help="substring of 'name (pid)' to keep")
    summary_parser.add_argument("--gaps", type=int, default=5)
    args = parser.parse_args()

    require_xcode()
    if args.subcommand == "record":
        command = args.command[1:] if args.command[:1] == ["--"] else args.command
        record(Path(args.out), args.template, args.time_limit, args.attach, command)
    elif args.subcommand == "tables":
        list_tables(Path(args.trace))
    elif args.subcommand == "export":
        export(Path(args.trace), args.table, Path(args.tsv) if args.tsv else None)
    else:
        summary(Path(args.trace), args.process, args.gaps)


if __name__ == "__main__":
    main()
