#!/bin/sh
#
# Build the native profiling helpers with the Command Line Tools' clang; Xcode is not needed.
#   libmtlstat.dylib  DYLD_INSERT_LIBRARIES hook: dispatch and commit counts, GPU busy time and a
#                     per-command-buffer kernel log (mtlstat.m; kernel_table.py, kernel_bench.py)
#   pso_info          pipeline properties of the kernels of a .metal or .metallib (pso_info.m)
#
# Usage: sh tools/profiling/build.sh [output_dir]    (default ~/.cache/ftmetal/profiling)
set -eu

case "${1:-}" in
  -h | --help)
    echo "usage: sh tools/profiling/build.sh [output_dir]    (default ~/.cache/ftmetal/profiling)"
    exit 0
    ;;
esac
SOURCE_DIR=$(cd "$(dirname "$0")" && pwd)
OUT_DIR=${1:-$HOME/.cache/ftmetal/profiling}

mkdir -p "$OUT_DIR"
cc -O2 -fobjc-arc -dynamiclib -framework Foundation -framework Metal \
  "$SOURCE_DIR/mtlstat.m" -o "$OUT_DIR/libmtlstat.dylib"
echo "built $OUT_DIR/libmtlstat.dylib"
cc -O2 -fobjc-arc -framework Foundation -framework Metal \
  "$SOURCE_DIR/pso_info.m" -o "$OUT_DIR/pso_info"
echo "built $OUT_DIR/pso_info"
