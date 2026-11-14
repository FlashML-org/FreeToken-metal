# Install

Requires an Apple Silicon Mac with macOS 15 or later and the Xcode Command Line Tools
(`xcode-select --install`).

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

After editing a `.cpp`, `.h` or `.mm` file, rebuild with `uv sync --reinstall-package ftmetal`.
