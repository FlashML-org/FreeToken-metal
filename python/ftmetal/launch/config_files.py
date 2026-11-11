"""Reading and writing agents' config files: a missing file reads as empty, one that does not
parse to a mapping stops the launch untouched, and a file whose content changes is backed up
first, keeping the newest few backups, and then replaced whole, never left half-written."""

from __future__ import annotations

import glob
import json
import os
import re
import shutil
import tempfile
import time
from pathlib import Path

import yaml
from pydantic import JsonValue

OWNER_ONLY_MODE = 0o600
BACKUPS_KEPT = 5


def write_text_with_backup(path: Path, text: str, *, private: bool = False) -> None:
    """private: the file holds a secret, so it is written owner-only whatever its old mode."""
    # Written through a symlink, so a dotfile manager's link stays a link.
    target = path.resolve()
    content = text.encode("utf-8")
    try:
        original = target.read_bytes() if target.exists() else None
        if original == content:
            return
        elif original is not None:
            backup = path.with_name(f"{path.name}.{time.time_ns()}.bak")
            # Created writable whatever the original's mode, and owner-only until copystat
            # gives it that mode, so a private file's copy is never readable by others.
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            with os.fdopen(os.open(backup, flags, OWNER_ONLY_MODE), "wb") as stream:
                stream.write(original)
            shutil.copystat(target, backup)
            # The nanosecond stamps have one width, so the names sort oldest first.
            backups = sorted(path.parent.glob(f"{glob.escape(path.name)}.*.bak"))
            for stale in backups[:-BACKUPS_KEPT]:
                stale.unlink()
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=target.parent, prefix=f".{target.name}.", suffix=".tmp"
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            # mkstemp creates it owner-only.
            if original is not None and not private:
                shutil.copymode(target, temporary)
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
    except OSError as exc:
        raise RuntimeError(f"cannot write {path}: {exc}") from exc


def write_json_with_backup(
    path: Path, value: dict[str, JsonValue], *, private: bool = False
) -> None:
    text = json.dumps(value, indent=2, ensure_ascii=False) + "\n"
    write_text_with_backup(path, text, private=private)


def write_yaml_with_backup(
    path: Path, value: dict[str, JsonValue], keys: tuple[str, ...], *, private: bool = False
) -> None:
    """Rewrites only the top-level keys blocks in the file's text, so every other line, comments
    included, stays as it was; a whole new dump when that text would not load back as value."""
    text = read_config_text(path)
    # A block appended after a last line without its newline would join that line.
    lines = (text + "\n" if text and not text.endswith("\n") else text).splitlines(keepends=True)
    for key in keys:
        block = yaml.safe_dump({key: value[key]}, sort_keys=False, allow_unicode=True)
        key_line = re.compile(rf"""({re.escape(key)}|"{re.escape(key)}"|'{re.escape(key)}')\s*:""")
        start = next((i for i, line in enumerate(lines) if key_line.match(line)), None)
        if start is None:
            lines.append(block)
        else:
            end = start + 1
            while end < len(lines) and (not lines[end].strip() or lines[end][0] in " \t#"):
                end += 1
            # Blank and unindented comment lines at its end introduce whatever follows.
            while end > start + 1 and (not lines[end - 1].strip() or lines[end - 1][0] == "#"):
                end -= 1
            lines[start:end] = [block]
    patched = "".join(lines)
    try:
        is_faithful = yaml.safe_load(patched) == value
    except yaml.YAMLError:
        is_faithful = False
    written = patched if is_faithful else yaml.safe_dump(value, sort_keys=False, allow_unicode=True)
    write_text_with_backup(path, written, private=private)


def read_config_text(path: Path) -> str:
    """The text of path without a byte order mark; empty when the file is missing."""
    try:
        return path.read_text(encoding="utf-8-sig") if path.exists() else ""
    except (OSError, UnicodeDecodeError) as exc:
        raise RuntimeError(
            f"cannot read {path}: {exc}\nfix or move the file, then re-run ftm launch."
        ) from exc


def load_json_object(path: Path) -> dict[str, JsonValue]:
    """The JSON object in path; empty when the file is missing or blank."""
    text = read_config_text(path)
    try:
        value = json.loads(text) if text.strip() else {}
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"cannot parse {path}: {exc}\nThe file uses syntax ftm cannot rewrite safely (JSON5 "
            "or JSONC comments and trailing commas are not JSON), so it is left as it was; fix "
            "or move it, then re-run ftm launch."
        ) from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{path} does not hold a JSON object; fix or move it, then re-run.")
    else:
        return value


def load_yaml_object(path: Path) -> dict[str, JsonValue]:
    """The YAML mapping in path; empty when the file is missing or holds no document."""
    text = read_config_text(path)
    try:
        value = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise RuntimeError(
            f"cannot parse {path}: {exc}\nfix or move the file, then re-run ftm launch."
        ) from exc
    if value is None:
        return {}
    elif not isinstance(value, dict):
        raise RuntimeError(f"{path} does not hold a YAML mapping; fix or move it, then re-run.")
    else:
        return value


def dict_child(parent: dict[str, JsonValue], key: str) -> dict[str, JsonValue]:
    child = parent.get(key)
    if isinstance(child, dict):
        return child
    else:
        new_child: dict[str, JsonValue] = {}
        parent[key] = new_child
        return new_child


def nested_value(document: JsonValue, *keys: str) -> JsonValue:
    """document[keys[0]][keys[1]]...; None where a level is missing or not a mapping."""
    for key in keys:
        document = document.get(key) if isinstance(document, dict) else None
    return document


def env_path(name: str, default: Path) -> Path:
    """The path the environment variable name holds, with ~ expanded; default when unset."""
    value = os.environ.get(name, "").strip()
    return Path(value).expanduser() if value else default


def toml_string(value: str) -> str:
    return json.dumps(value)


__all__ = [
    "dict_child",
    "env_path",
    "load_json_object",
    "load_yaml_object",
    "nested_value",
    "toml_string",
    "write_json_with_backup",
    "write_text_with_backup",
    "write_yaml_with_backup",
]
