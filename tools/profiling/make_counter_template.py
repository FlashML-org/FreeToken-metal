"""Clone the active Xcode's Metal System Trace template with a different Metal GPU
Counters profile, so xctrace record --template <path> records hardware counters without
the Instruments GUI.

    uv run python tools/profiling/make_counter_template.py --profile 13 --out mst13.tracetemplate
    uv run python tools/profiling/make_counter_template.py --profile 13 --shader-profiler 1 \\
        --out mst13s.tracetemplate

The template is an NSKeyedArchiver binary plist; the GPU instrument's settings dict holds
counterprofile / counterprofileinternal (int), shaderprofiler / shaderprofilerinternal
(bool) and counterdevice / counterdeviceinternal (accelerator id, as reported in the
gpu-counter-info table; --device-id, default the template's). --src defaults to the stock
template of the Xcode that xcrun resolves (set DEVELOPER_DIR to an Xcode install when
xcode-select points at the Command Line Tools). Result on M4 Pro / macOS 26.5.2 / Xcode
26.6: profile 0 (the stock one) records a single counter (RT Unit Active), profile 13
about 85 (occupancy, limiters, utilization, L1 residency, bandwidth), profile 14 the ray
tracing ones; every other profile tried (1-12, 15-31, 48, 64, 100, 128, 255, 256, 1024)
is rejected by the GPU service ("Selected counter profile is not supported on target
device") and leaves every counter table empty. --shader-profiler 1 is rejected with
profile 0; with profile 13 it keeps the counters and adds a sampled Shader Timeline
(metal-shader-profiler-intervals, one row per kernel), while gpu-shader-profiler-sample,
the per-line cost, stays empty.
"""

from __future__ import annotations

import argparse
import plistlib
import sys
from pathlib import Path

from xctrace_intervals import require_xcode

# The stock template, relative to the Xcode bundle's Contents directory.
STOCK_TEMPLATE = (
    "Applications/Instruments.app/Contents/Packages/GPU.instrdst/Contents/Templates/"
    "Metal System Trace.tracetemplate"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--src", help="template to clone (default: the active Xcode's stock one)")
    parser.add_argument(
        "--profile", type=int, required=True, help="Metal GPU Counters profile (13: occupancy...)"
    )
    parser.add_argument(
        "--shader-profiler", type=int, default=0, help="1 turns the shader profiler on"
    )
    parser.add_argument(
        "--device-id", type=int, help="accelerator id to force (default: keep template value)"
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    source = Path(args.src) if args.src else require_xcode().parent / STOCK_TEMPLATE

    replacements: dict[str, int | bool] = {
        "counterprofile": args.profile,
        "counterprofileinternal": args.profile,
        "shaderprofiler": bool(args.shader_profiler),
        "shaderprofilerinternal": bool(args.shader_profiler),
    }
    if args.device_id is not None:
        replacements["counterdevice"] = args.device_id
        replacements["counterdeviceinternal"] = args.device_id
    with source.open("rb") as template:
        archive = plistlib.load(template)
    objects = archive["$objects"]
    patched: list[str] = []
    for archived in objects:
        if not (isinstance(archived, dict) and "NS.keys" in archived):
            continue
        keys = [objects[key.data] for key in archived["NS.keys"]]
        if "counterprofile" not in keys:
            continue
        for index, key in enumerate(keys):
            if key in replacements:
                objects.append(replacements[key])
                archived["NS.objects"][index] = plistlib.UID(len(objects) - 1)
                patched.append(key)
    if "counterprofile" not in patched:
        sys.exit(f"{source} has no counterprofile setting to patch: not an Instruments template?")
    with Path(args.out).open("wb") as output:
        plistlib.dump(archive, output, fmt=plistlib.FMT_BINARY)
    print(f"wrote {args.out}; patched {patched}")


if __name__ == "__main__":
    main()
