#!/usr/bin/env bash
#
# Build the wheels: one per supported Python, for macOS 15 or later on Apple Silicon. Each wheel
# is checked (version, platform tag, its own extensions only) and, unless --no-check, installed
# into a fresh environment and imported.
#
# A development build is stamped with its commit (<version>+g<sha>, .dirty when the packaged
# files have uncommitted changes) and is not for PyPI. --release builds the bare version from a
# clean checkout whose HEAD is tagged v<version>: every supported Python, checked, into an empty
# output directory.
#
#   scripts/build-wheels.sh                      # development wheels into dist/
#   scripts/build-wheels.sh --python 3.13        # only these Pythons (repeatable)
#   scripts/build-wheels.sh --release            # the wheels that go to PyPI
#
# Options: --python X.Y, --out-dir DIR (default dist/ in the repository), --sdist, --no-check,
# --release. Needs uv; runs on an Apple Silicon Mac.
set -euo pipefail

SUPPORTED_PYTHONS=(3.11 3.12 3.13)
PYTHONS=()
OUT_DIR=""
SDIST=0
CHECK=1
RELEASE=0
VERSION_FILE=python/ftmetal/version.py
# The oldest macOS the wheels run on (the Metal kernels need Metal 3.2, macOS 15).
export MACOSX_DEPLOYMENT_TARGET=15.0
PLATFORM_TAG=macosx_15_0_arm64
# uv's own Pythons: setuptools ignores a deployment target below the interpreter's, and a
# Homebrew Python built for a newer macOS would tag the wheels for that macOS.
export UV_MANAGED_PYTHON=1
TWINE=twine==7.0.0
# What a wheel or the sdist is built from; anything else in the checkout does not mark a build
# dirty.
PACKAGED=(python setup.py pyproject.toml README.md LICENSE)

say() { printf '==> %s\n' "$*"; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

usage() {
  sed -n '3,17s/^# \{0,1\}//p' "${BASH_SOURCE[0]}"
}

while [ $# -gt 0 ]; do
  case "$1" in
    --python) [ $# -ge 2 ] || die "--python needs a version"; PYTHONS+=("$2"); shift 2 ;;
    --out-dir) [ $# -ge 2 ] || die "--out-dir needs a directory"; OUT_DIR="$2"; shift 2 ;;
    --sdist) SDIST=1; shift ;;
    --no-check) CHECK=0; shift ;;
    --release) RELEASE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; die "unknown option: $1" ;;
  esac
done

if [ "$RELEASE" = 1 ]; then
  [ ${#PYTHONS[@]} -eq 0 ] || die "--release builds every supported Python; drop --python"
  [ "$CHECK" = 1 ] || die "--release checks every wheel; drop --no-check"
  [ "$SDIST" = 0 ] || die "releases ship wheels only; drop --sdist"
fi
[ ${#PYTHONS[@]} -gt 0 ] || PYTHONS=("${SUPPORTED_PYTHONS[@]}")
for python in "${PYTHONS[@]}"; do
  [[ "$python" =~ ^3\.[0-9]+$ ]] || die "--python takes a version like 3.13, got $python"
done
# Each Python once, in the order given.
unique=()
for python in "${PYTHONS[@]}"; do
  case " ${unique[*]-} " in *" $python "*) ;; *) unique+=("$python") ;; esac
done
PYTHONS=("${unique[@]}")

[ "$(uname -s)-$(uname -m)" = Darwin-arm64 ] || die "build on an Apple Silicon Mac"
command -v uv >/dev/null || die "uv is not on PATH"

ROOT="$(CDPATH='' cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
[ -f "$ROOT/$VERSION_FILE" ] && [ -f "$ROOT/setup.py" ] || die "$ROOT is not an ftmetal checkout"
# A relative --out-dir is relative to where the script was run.
if [ -z "$OUT_DIR" ]; then
  OUT_DIR="$ROOT/dist"
fi
mkdir -p "$OUT_DIR"
OUT_DIR="$(CDPATH='' cd -- "$OUT_DIR" && pwd -P)"
case "$OUT_DIR/" in
  "$ROOT/build/"*) die "--out-dir must not be under build/, which every build deletes" ;;
esac
if [ "$RELEASE" = 1 ] && [ -n "$(ls -A "$OUT_DIR")" ]; then
  die "a release goes into an empty directory; $OUT_DIR is not empty"
fi
cd "$ROOT"

SCRATCH="$(mktemp -d)"
STAMPED=0
cleanup() {
  # Put version.py back byte for byte, so uncommitted edits to it survive.
  if [ "$STAMPED" = 1 ]; then
    cp "$SCRATCH/version.py" "$ROOT/$VERSION_FILE"
  fi
  rm -rf "$SCRATCH" "$ROOT/build" "$ROOT/python/freetoken_metal.egg-info"
}
trap cleanup EXIT

version="$(sed -nE 's/^__version__ = "([^"]+)"$/\1/p' "$VERSION_FILE")"
[ -n "$version" ] || die "cannot read __version__ from $VERSION_FILE"
case "$version" in
  *+*) die "$VERSION_FILE says $version: an interrupted build left its stamp; restore the file" ;;
esac
if [ "$RELEASE" = 1 ]; then
  [ -z "$(git status --porcelain)" ] || die "a release builds from a clean checkout"
  tagged="$(git rev-parse -q --verify "refs/tags/v$version^{commit}" || true)"
  [ "$tagged" = "$(git rev-parse HEAD)" ] \
    || die "a release of $version builds from the commit tagged v$version, not HEAD"
  say "release build of $version"
else
  dirty=""
  if [ -n "$(git status --porcelain --untracked-files=all -- "${PACKAGED[@]}")" ]; then
    dirty=.dirty
  fi
  version="$version+g$(git rev-parse --short HEAD)$dirty"
  cp "$VERSION_FILE" "$SCRATCH/version.py"
  STAMPED=1
  sed -E "s/^__version__ = \"[^\"]+\"$/__version__ = \"$version\"/" "$SCRATCH/version.py" \
    > "$VERSION_FILE"
  say "development build of $version (not for PyPI)"
fi

built=()
for python in "${PYTHONS[@]}"; do
  abi="cp${python/./}"
  say "building the $abi wheel"
  # build/ outlives a build: another Python's extensions in it would end up in this wheel.
  rm -rf build
  # Debug or architecture settings in the environment would change the C++ extensions.
  env -u DEBUG -u CMAKE_ARGS -u ARCHFLAGS \
    uv build --wheel --python "$python" --out-dir "$SCRATCH/$abi"

  wheels=("$SCRATCH/$abi"/*.whl)
  [ ${#wheels[@]} -eq 1 ] && [ -f "${wheels[0]}" ] || die "expected one wheel for $abi"
  wheel="${wheels[0]}"
  case "$(basename "$wheel")" in
    *-"$version"-"$abi"-"$abi"-"$PLATFORM_TAG".whl) ;;
    *) die "$(basename "$wheel") is not version $version tagged $abi-$abi-$PLATFORM_TAG" ;;
  esac
  extensions="$(unzip -Z1 "$wheel" | grep '\.so$' || true)"
  [ "$(printf '%s\n' "$extensions" | grep -c .)" -eq 2 ] \
    || die "expected the two C++ extensions in $(basename "$wheel"), got: $extensions"
  if printf '%s\n' "$extensions" | grep -qv "cpython-${python/./}-darwin\.so$"; then
    die "$(basename "$wheel") carries another Python's extension: $extensions"
  fi

  if [ "$CHECK" = 1 ]; then
    say "checking the $abi wheel in a fresh environment"
    venv="$SCRATCH/venv-$abi"
    uv venv --quiet --python "$python" "$venv"
    uv pip install --quiet --python "$venv" "$wheel"
    # -I and no PYTHONPATH: the installed wheel is what gets imported, never the checkout.
    env -u PYTHONPATH "$venv/bin/python" -I -c '
import sys
import mlx.core, ftmetal, ftmetal.kernel._C, ftmetal.command_graph._C
assert ftmetal.__file__.startswith(sys.prefix), ftmetal.__file__
' || die "the $abi wheel does not import from its own environment"
    reported="$(env -u PYTHONPATH "$venv/bin/ftm" --version)"
    [ "$reported" = "ftmetal version $version" ] \
      || die "ftm --version says '$reported', expected 'ftmetal version $version'"
  fi
  mv "$wheel" "$OUT_DIR/"
  built+=("$OUT_DIR/$(basename "$wheel")")
done

if [ "$SDIST" = 1 ]; then
  say "building the sdist"
  rm -rf build
  uv build --sdist --out-dir "$SCRATCH/sdist"
  mv "$SCRATCH"/sdist/*.tar.gz "$OUT_DIR/"
fi

if [ "$RELEASE" = 1 ]; then
  [ ${#built[@]} -eq ${#SUPPORTED_PYTHONS[@]} ] || die "a release has one wheel per Python"
  say "checking the wheels for PyPI"
  uvx "$TWINE" check --strict "${built[@]}"
fi

say "done, in $OUT_DIR:"
for file in "${built[@]}"; do
  basename "$file"
done
if [ "$SDIST" = 1 ]; then
  ls "$OUT_DIR" | grep '\.tar\.gz$'
fi
