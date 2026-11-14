"""Capture an Xcode GPU trace (.gputrace) of one MLX workload from the command line.

    uv run python tools/profiling/gputrace_capture.py --out x.gputrace \\
        --setup "a = mx.random.normal((2048, 2048))" --expr "mx.softmax(a @ a, axis=-1)" \\
        [--warmup 2]
    PYTHONPATH=. uv run python tools/profiling/gputrace_capture.py --out x.gputrace \\
        --call mypkg.bench:run_once [--warmup 1]

Re-execs itself with MTL_CAPTURE_ENABLED=1 (must be in the environment before the Metal
device is created), evaluates the expression --warmup times (so JIT compiles and allocs
are outside the capture), then wraps exactly one evaluation in
mx.metal.start_capture / stop_capture. Capturing needs no Xcode; a capture holds MLX's
whole metallib, so it is at least about 190 MB. The .gputrace opens in Xcode (GUI only);
replay it on a Mac with the same GPU family + OS. mx is in scope for --setup / --expr;
--setup may be given several times. --call imports module:function and evaluates what the
function returns; a script's import path starts at its own directory, so put the module's
root on PYTHONPATH.
"""

from __future__ import annotations

import argparse
import importlib
import os
import shutil
import sys
from pathlib import Path

import mlx.core as mx
from kernel_bench import evaluate


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--out",
        required=True,
        help="path ending in .gputrace; a file or directory already there is removed first",
    )
    parser.add_argument(
        "--setup", action="append", default=[], help="python statements run once before timing"
    )
    parser.add_argument(
        "--expr", help="python expression returning an mx.array (or tuple/list of them)"
    )
    parser.add_argument(
        "--call", help="module:function called with no args; its return value is evaluated"
    )
    parser.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args()
    if not (args.expr or args.call):
        parser.error("give --expr or --call")
    output = Path(args.out).absolute()
    if output.suffix != ".gputrace":
        sys.exit("--out must end in .gputrace")

    if os.environ.get("MTL_CAPTURE_ENABLED") != "1":
        os.execve(
            sys.executable,
            [sys.executable, *sys.argv],
            dict(os.environ, MTL_CAPTURE_ENABLED="1"),
        )
    namespace: dict[str, object] = {"mx": mx}
    for statement in args.setup:
        exec(statement, namespace)
    if args.call:
        module_name, function_name = args.call.split(":")
        namespace["workload"] = getattr(importlib.import_module(module_name), function_name)
        expression = compile("workload()", "<call>", "eval")
    else:
        expression = compile(args.expr, "<expr>", "eval")

    for _ in range(args.warmup):
        evaluate(expression, namespace)
    mx.synchronize()
    # MLX's capture fails on an existing path.
    if output.is_dir() and not output.is_symlink():
        shutil.rmtree(output)
    elif output.exists() or output.is_symlink():
        output.unlink()
    mx.metal.start_capture(str(output))
    try:
        evaluate(expression, namespace)
        mx.synchronize()
    finally:
        mx.metal.stop_capture()
    size = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
    print(f"captured {output} ({size / 1e6:.1f} MB); open it in Xcode")
    sys.stdout.flush()
    os._exit(0)  # MLX teardown can hang after a capture


if __name__ == "__main__":
    main()
