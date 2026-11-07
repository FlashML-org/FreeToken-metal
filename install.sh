#!/usr/bin/env bash
#
# ftmetal installer (macOS on Apple Silicon), from this source checkout: the pinned uv, a venv with
# ftmetal built from the current sources and without the dev group, and `ftm` linked onto PATH.
# Rerun after any source change. Development uses `uv sync` instead, see README.md.
#
# Configurable via environment, absolute paths only:
#   FTMETAL_HOME      install root (default: ~/.ftmetal); uv at bin/uv, wheel at wheel/, venv at venv/
#   FTMETAL_BIN_DIR   where to link `ftm`, and uv if there is none (default: ~/.local/bin)
set -euo pipefail

UV_VERSION="0.13.7"
PY_VERSION="3.13"
FT_HOME="${FTMETAL_HOME:-$HOME/.ftmetal}"
UV="$FT_HOME/bin/uv"
WHEEL_DIR="$FT_HOME/wheel"
VENV="$FT_HOME/venv"
BIN_DIR="${FTMETAL_BIN_DIR:-$HOME/.local/bin}"

say() { printf '==> %s\n' "$*"; }
die() { printf '[error] %s\n' "$*" >&2; exit 1; }

# The build below deletes build/ in the current directory, so it must be this checkout.
CDPATH='' cd "$(dirname "${BASH_SOURCE[0]}")"
[ -f python/ftmetal/__init__.py ] || die "run install.sh from its checkout, not through a link or a pipe."

case "$FT_HOME:$BIN_DIR" in
  /*:/*) ;;
  *) die "FTMETAL_HOME and FTMETAL_BIN_DIR must be absolute paths." ;;
esac
[ "$(uname -sm)" = "Darwin arm64" ] || die "needs macOS on Apple Silicon, this is $(uname -sm)."
xcode-select -p >/dev/null 2>&1 || die "no C++ compiler: run xcode-select --install, then re-run."

# Private, so that a uv the user already has stays as it is. Unmanaged: no shell rc edits. The two
# unset variables would outrank the install dir.
if [ "$("$UV" --version 2>/dev/null | awk '{print $2}')" != "$UV_VERSION" ]; then
  say "installing uv $UV_VERSION into $FT_HOME/bin ..."
  mkdir -p "$FT_HOME/bin"
  curl -LsSf "https://astral.sh/uv/$UV_VERSION/install.sh" \
    | env -u UV_INSTALL_DIR -u CARGO_DIST_FORCE_INSTALL_DIR UV_UNMANAGED_INSTALL="$FT_HOME/bin" sh
fi

say "building ftmetal from $PWD ..."
# build/lib outlives a build, so a file deleted from the sources would stay in the wheel.
rm -rf build "$WHEEL_DIR"
# DEBUG in the environment would make the C++ extensions a debug build.
env -u DEBUG "$UV" build --wheel --python "$PY_VERSION" --out-dir "$WHEEL_DIR"

# Only now: a build that fails above leaves the previous install working.
say "installing into $VENV ..."
"$UV" venv "$VENV" --python "$PY_VERSION" --clear
"$UV" pip install --python "$VENV" "$WHEEL_DIR"/ftmetal-*.whl

mkdir -p "$BIN_DIR"
ln -sf "$VENV/bin/ftm" "$BIN_DIR/ftm"
command -v uv >/dev/null || [ -e "$BIN_DIR/uv" ] || ln -sf "$UV" "$BIN_DIR/uv"

# The CLI imports both C++ extensions, so this fails if they do not load.
"$BIN_DIR/ftm" --help >/dev/null

FOUND="$(command -v ftm || true)"
if [ "$FOUND" != "$BIN_DIR/ftm" ]; then
  printf '[warn] put %s first on PATH: ftm resolves to "%s" here.\n' "$BIN_DIR" "$FOUND" >&2
fi

cat <<EOF

ftmetal installed: $BIN_DIR/ftm -> $VENV/bin/ftm

  ftm serve --model <checkpoint dir>     # OpenAI API on 127.0.0.1:1919
  ftm shell                              # chat with the running server
EOF
