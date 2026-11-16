# Install

Requires an Apple Silicon Mac with macOS 15 or later. Building from source also needs the Xcode
Command Line Tools (`xcode-select --install`).

## From PyPI

With [uv](https://docs.astral.sh/uv/):

```bash
uv tool install --python 3.13 freetoken-metal
ftm --version
```

It installs into an environment of its own and links `ftm` into `~/.local/bin`; run
`uv tool update-shell` if that directory is not on your `PATH`, and
`uv tool upgrade freetoken-metal` to update. With pip, install into a virtual environment of
Python 3.11 to 3.13: `pip install freetoken-metal`.

## Install script

```bash
git clone https://github.com/FlashML-org/FreeToken-metal.git && cd FreeToken-metal
./install.sh
```

It installs into `~/.ftmetal` and links `ftm` into `~/.local/bin`; add that directory to your
`PATH` if it is not there. Rerun it after pulling.

## From source

With [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/FlashML-org/FreeToken-metal.git && cd FreeToken-metal
uv sync
uv run ftm --version
```

After editing a `.cpp`, `.h` or `.mm` file, rebuild with `uv sync --reinstall-package freetoken-metal`.
