"""Client-supplied image handling: ref collection, the accept gate and byte fetching.

Raises plain ValueError on bad input; the server layer maps it to its wire error type.

Port of FreeToken mm/media.py. Divergences: http(s) is fetched with urllib in a worker
thread instead of httpx; the size cap bounds every source, not only http(s); the fetched
bytes get a header probe, so a non-image fails the request before any stream starts
(FreeToken fails it in the tokenizer).
"""

from __future__ import annotations

import asyncio
import base64
import http.client
import io
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

from PIL import Image
from pydantic import JsonValue

from ftmetal.server.args import ServerArgs

MAX_IMAGE_BYTES = 32 << 20
URL_TIMEOUT_SECONDS = 30.0
# What a bad source raises: undecodable base64, an unreachable URL or missing file, an HTTP
# error status, bytes Pillow cannot identify or a decompression bomb.
FETCH_ERRORS = (ValueError, OSError, http.client.HTTPException, Image.DecompressionBombError)


def collect_image_refs(messages: list[dict[str, JsonValue]]) -> list[dict[str, JsonValue]]:
    """Pop the image refs out of rendered messages in prompt order; the bare {"type": "image"}
    parts stay for the chat template."""
    refs: list[dict[str, JsonValue]] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image":
                ref = part.pop("freetoken_ref", None)
                if isinstance(ref, dict):
                    refs.append(ref)
    return refs


def image_reject_reason(config: ServerArgs) -> str | None:
    """None when this server can accept image inputs, else the client-facing reason."""
    if "image" in config.served_modalities:
        return None
    elif config.mm.text_model_only:
        return "image input is disabled on this server (--text-model-only)"
    elif "vision" in config.mm.disabled_encoders:
        return "image input is disabled on this server (--mm-disable)"
    else:
        return "the served model has no vision support"


def check_media_domain(url: str, config: ServerArgs) -> None:
    """Reject a URL whose hostname is not in --allowed-media-domains; an empty allowlist
    admits any domain."""
    allowed = {
        domain.strip().lower().rstrip(".")
        for domain in config.allowed_media_domains.split(",")
        if domain.strip()
    }
    host = (urlparse(url).hostname or "").rstrip(".")
    if allowed and host not in allowed:
        raise ValueError(
            f"the URL must be from one of the allowed domains: {sorted(allowed)}; "
            f"input URL domain: {host or '<none>'}"
        )


def load_local_media(url: str, config: ServerArgs) -> bytes:
    """Read a file:// ref; the resolved path must be a strict subpath of
    --allowed-local-media-path."""
    root = config.allowed_local_media_path
    if not root:
        raise ValueError("cannot load local files without --allowed-local-media-path")
    spec = urlparse(url)
    file_path = Path(urllib.request.url2pathname(spec.netloc + spec.path))
    # resolve() follows symlinks, so a link inside the root escaping it is rejected too.
    resolved = file_path.resolve()
    if Path(root).resolve() not in resolved.parents:
        raise ValueError(
            f"the file path {file_path} must be a subpath of --allowed-local-media-path {root}"
        )
    with resolved.open("rb") as file:
        return file.read(MAX_IMAGE_BYTES + 1)


def read_url(url: str) -> bytes:
    """An http(s) URL's body, at most one byte past the cap; redirect targets are not
    re-checked against the allowlist."""
    with urllib.request.urlopen(url, timeout=URL_TIMEOUT_SECONDS) as response:
        return response.read(MAX_IMAGE_BYTES + 1)


async def fetch_image_bytes(refs: list[dict[str, JsonValue]], config: ServerArgs) -> list[bytes]:
    """Resolve collected image refs (URLs / base64) to raw bytes, in order, each one checked
    to start like an image."""
    images: list[bytes] = []
    for ref in refs:
        ref_data = ref.get("data")
        data = ref_data if isinstance(ref_data, str) else ""
        try:
            if ref.get("kind") == "b64":
                raw = base64.b64decode(data)
            elif data.startswith("data:"):
                raw = base64.b64decode(data.partition(",")[2])
            elif data.startswith(("http://", "https://")):
                check_media_domain(data, config)
                raw = await asyncio.to_thread(read_url, data)
            elif data.startswith("file://"):
                raw = load_local_media(data, config)
            else:
                raise ValueError("unsupported image source (expect http(s)/file url or base64)")
            if len(raw) > MAX_IMAGE_BYTES:
                raise ValueError(f"image exceeds {MAX_IMAGE_BYTES} bytes")
            # Image.open reads the header only; decoding waits for the tokenizer.
            with Image.open(io.BytesIO(raw)):
                pass
        except FETCH_ERRORS as exc:
            raise ValueError(f"could not load image: {exc}") from exc
        images.append(raw)
    return images


__all__ = [
    "MAX_IMAGE_BYTES",
    "check_media_domain",
    "collect_image_refs",
    "fetch_image_bytes",
    "image_reject_reason",
    "load_local_media",
    "read_url",
]
