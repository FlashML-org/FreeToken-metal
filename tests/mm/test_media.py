"""Client image refs: collection from rendered messages, the accept gate, and fetching from
base64, data: URLs, the file:// allowlist and http(s) under the domain allowlist (urlopen is
faked, so no network)."""

from __future__ import annotations

import asyncio
import base64
import io
import urllib.request
from pathlib import Path

import pytest
from PIL import Image
from pydantic import JsonValue

from ftmetal.mm import media
from ftmetal.mm.config import ENCODER_KINDS, EncoderKind, MultimodalConfig
from ftmetal.mm.media import collect_image_refs, fetch_image_bytes, image_reject_reason
from ftmetal.server.args import ServerArgs
from tiny_vl_model import register_vision_encoder, write_gemma4_model


def png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), (1, 2, 3)).save(buffer, format="PNG")
    return buffer.getvalue()


PNG = png_bytes()


def fake_urlopen(url: str, timeout: float) -> io.BytesIO:
    return io.BytesIO(PNG)


def test_refs_are_popped_in_prompt_order() -> None:
    first: dict[str, JsonValue] = {"kind": "url", "data": "https://a.example/1.png"}
    second: dict[str, JsonValue] = {"kind": "b64", "data": "AA=="}
    messages: list[dict[str, JsonValue]] = [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": [{"type": "image", "freetoken_ref": first}]},
        {"role": "user", "content": [{"type": "text", "text": "and"}, {"type": "image"}]},
        {"role": "user", "content": [{"type": "image", "freetoken_ref": second}]},
    ]
    assert collect_image_refs(messages) == [first, second]
    # The bare parts stay for the chat template.
    assert messages[1]["content"] == [{"type": "image"}]


# (the family registers its vision encoder, the encoder kinds left unbuilt, the refusal or
# None)
REJECT_CASES: list[tuple[bool, frozenset[EncoderKind], str | None]] = [
    (False, frozenset(), "the served model has no vision support"),
    (False, frozenset(ENCODER_KINDS), "image input is disabled on this server (--text-model-only)"),
    (True, frozenset(ENCODER_KINDS), "image input is disabled on this server (--text-model-only)"),
    (False, frozenset({"vision"}), "image input is disabled on this server (--mm-disable)"),
    (True, frozenset({"vision"}), "image input is disabled on this server (--mm-disable)"),
    (True, frozenset({"audio"}), None),
    (True, frozenset(), None),
]


@pytest.mark.parametrize(("has_encoder", "disabled_encoders", "reason"), REJECT_CASES)
def test_reject_reason_follows_the_served_modalities(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    has_encoder: bool,
    disabled_encoders: frozenset[EncoderKind],
    reason: str | None,
) -> None:
    write_gemma4_model(tmp_path)
    if has_encoder:
        register_vision_encoder(monkeypatch, "Gemma4ForConditionalGeneration")
    config = ServerArgs(
        model_path=str(tmp_path), mm=MultimodalConfig(disabled_encoders=disabled_encoders)
    )
    assert image_reject_reason(config) == reason


# (ref kind, ref data with {root} for the media root and {b64} for a PNG's base64,
# --allowed-media-domains, whether --allowed-local-media-path is set, the error or None)
FETCH_CASES: list[tuple[str, str, str, bool, str | None]] = [
    ("b64", "{b64}", "", False, None),
    ("url", "data:image/png;base64,{b64}", "", False, None),
    ("url", "file://{root}/image.png", "", True, None),
    ("url", "file://{root}/image.png", "", False, "without --allowed-local-media-path"),
    ("url", "file://{root}/../outside/secret.png", "", True, "must be a subpath"),
    # A link inside the root that points out of it.
    ("url", "file://{root}/link.png", "", True, "must be a subpath"),
    ("url", "https://images.example.com/a.png", "", False, None),
    ("url", "https://images.example.com/a.png", "Images.Example.com.", False, None),
    ("url", "https://other.example.com/a.png", "images.example.com", False, "allowed domains"),
    ("b64", base64.b64encode(b"plain text").decode(), "", False, "cannot identify image"),
    ("url", "ftp://images.example.com/a.png", "", False, "unsupported image source"),
]


@pytest.mark.parametrize(("kind", "data", "domains", "has_root", "error"), FETCH_CASES)
def test_fetch_loads_or_refuses_each_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    data: str,
    domains: str,
    has_root: bool,
    error: str | None,
) -> None:
    root = tmp_path / "media"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / "image.png").write_bytes(PNG)
    (outside / "secret.png").write_bytes(PNG)
    (root / "link.png").symlink_to(outside / "secret.png")
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    config = ServerArgs(
        model_path=str(tmp_path),
        allowed_media_domains=domains,
        allowed_local_media_path=str(root.resolve()) if has_root else "",
    )
    ref: dict[str, JsonValue] = {
        "kind": kind,
        "data": data.format(root=root, b64=base64.b64encode(PNG).decode()),
    }
    if error is None:
        assert asyncio.run(fetch_image_bytes([ref], config)) == [PNG]
    else:
        with pytest.raises(ValueError, match=error):
            asyncio.run(fetch_image_bytes([ref], config))


# (ref kind, ref data): the size cap bounds every source.
CAPPED_SOURCE_CASES: list[tuple[str, str]] = [
    ("b64", "{b64}"),
    ("url", "https://images.example.com/a.png"),
    ("url", "file://{root}/image.png"),
]


@pytest.mark.parametrize(("kind", "data"), CAPPED_SOURCE_CASES)
def test_fetch_caps_the_image_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str, data: str
) -> None:
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(media, "MAX_IMAGE_BYTES", len(PNG) - 1)
    (tmp_path / "image.png").write_bytes(PNG)
    config = ServerArgs(model_path=str(tmp_path), allowed_local_media_path=str(tmp_path.resolve()))
    ref: dict[str, JsonValue] = {
        "kind": kind,
        "data": data.format(root=tmp_path, b64=base64.b64encode(PNG).decode()),
    }
    with pytest.raises(ValueError, match=f"exceeds {len(PNG) - 1} bytes"):
        asyncio.run(fetch_image_bytes([ref], config))
