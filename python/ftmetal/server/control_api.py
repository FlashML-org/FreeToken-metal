"""Read-only control-plane routes: GET /health (lifecycle), GET /v1/stats (runtime metrics),
GET /v1/requests (the request ring) and GET /v1/cache/status (cache geometry), plus the HTTP
middleware that times the requests the generation layer does not record into the ring.

All handlers read a shared FrontendManager snapshot via get_state; nothing here touches
the scheduler or blocks.

Port of FreeToken server/control_api.py, plus the request-ring middleware, /v1/cache/status
and its cache_geometry from FreeToken server/api_server.py. Divergences: /health has no
loading phase (the engine is built before the server accepts requests); the request ring is
the FrontendManager's; /v1/cache/status only reports (no rebuild, so last_rebuild stays
null), and of what FreeToken learns from the engine's readiness meta only the MoE offload
cache's geometry arrives (MoeCacheGeometry, when the scheduler thread starts): load-time
pool sizes, per-unit costs, floors, the cache budget and the window pool read as their
unknown 0, so every limit's max is 0; num_experts and num_moe_layers are 0 while the
experts are resident (FreeToken reports the model's); the pool sizes come from the reply
snapshots alone and the thinking gears from the profile probed before serving.
"""

# Comments and docstrings keep FreeToken's line breaks, and its loose typing stays unchecked.
# ruff: noqa: E501

from __future__ import annotations

import time
from collections.abc import Awaitable
from dataclasses import dataclass
from typing import Any, Callable

from fastapi import FastAPI, Request, Response

from .model_meta import derive_think_gears
from .request_ring import RequestRecord
from .stats import build_stats

# Paths the HTTP middleware logs into the request ring. The three chat protocols funnel through
# the shared generation layer and are recorded there instead (with real token totals) — kept out
# here to avoid a duplicate token-less row. These two run their own ack loop, so they stay logged
# here (without per-request tokens).
TRACKED_REQUEST_PREFIXES = (
    "/v1/completions",
    "/generate",
)


@dataclass(frozen=True, kw_only=True)
class MoeCacheGeometry:
    """The MoE offload cache the engine built: experts per layer, MoE layers, slots, policy."""

    num_experts: int
    num_moe_layers: int
    moe_cache_size: int
    moe_cache_policy: str


def build_health(state: Any, version: str) -> dict:
    """Lifecycle health doc: ok while serving (or stopping), error once a backend thread died."""
    instance_id = getattr(state, "instance_id", None)
    fatal = getattr(state, "fatal_error", None)
    if fatal:
        return {"status": "error", "message": fatal, "instance_id": instance_id}

    mstate = getattr(state, "maintenance_state", "serving")
    config = getattr(state, "config", None)
    model = getattr(config, "served_model_name", None)

    ready_at = getattr(state, "ready_at", None)
    uptime_s = max(0, int(time.monotonic() - ready_at)) if ready_at is not None else 0
    return {
        "status": "ok",
        "model": model,
        "instance_id": instance_id,
        "uptime_s": uptime_s,
        "maintenance": mstate,
        "version": version,
    }


def _reasoning_geometry(state: Any) -> dict | None:
    """The ``geometry.reasoning`` block, from the frontend tokenizer's probed
    thinking profile."""
    derived = derive_think_gears(
        state.thinking_profile,
        parser_configured=bool(getattr(state.config, "reasoning_parser", None)),
    )
    if derived is None:
        return None
    gears, default, kwargs = derived
    return {"gears": list(gears), "default": default, "kwargs": kwargs}


def cache_geometry(state: Any) -> dict:
    """Current cache geometry for a cache panel. Each pool size is the running UserReply
    snapshot's (updated per generation), 0 before the first one. num_experts /
    num_moe_layers / moe_cache_size are the MoE offload cache's, 0 without one."""
    tr = state.stats
    config = state.config
    page_size = int(getattr(config, "page_size", 1) or 1)
    moe: MoeCacheGeometry | None = getattr(state, "moe_cache", None)
    # Thinking control: the gears a client can offer and the chat_template_kwargs
    # each selects, derived from the checkpoint's own probed template (no
    # per-family registry).
    reasoning = _reasoning_geometry(state)
    geo = {
        "num_pages": tr.kv_total_pages,
        "page_size": page_size,
        "moe_cache_size": 0 if moe is None else moe.moe_cache_size,
        "num_mamba_slots": tr.mamba_total_slots,
        "num_experts": 0 if moe is None else moe.num_experts,
        "num_moe_layers": 0 if moe is None else moe.num_moe_layers,
        # Eviction policy of the MoE slot cache; None without one.
        "moe_cache_policy": None if moe is None else moe.moe_cache_policy,
        # Per-unit VRAM costs, unknown without the readiness meta.
        "unit_bytes": {
            "kv_per_token": 0,
            "moe_per_expert": 0,
            "mamba_per_slot": 0,
            "swa_per_token": 0,
        },
        "swa_full_tokens_ratio": 0.0,
        # The window pool's page unit and size, unknown without the readiness meta.
        "swa_page_size": 0,
        "num_swa_pages": 0,
        # The engine's total cache budget; 0 when unknown.
        "cache_budget_bytes": 0,
        "reasoning": reasoning,
    }
    # Per-pool slider bounds: with no budget known every max is 0 ("unknown"), and each min
    # is the floor fallback (page_size for KV, num_experts for MoE, else 0).
    geo["limits"] = {
        "kv_tokens": {"min": page_size, "max": 0},
        "moe_experts": {"min": 0 if moe is None else moe.num_experts, "max": 0},
        "mamba_slots": {"min": 0, "max": 0},
        "swa_tokens": {"min": 0, "max": 0},
    }
    return geo


def register_control_routes(
    app: FastAPI,
    get_state: Callable[[], Any],
    get_model_sampling: Callable[[], dict] | None = None,
) -> None:
    @app.middleware("http")
    async def record_request(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Time every generation request into the ring for /v1/requests + /v1/stats p95. Single-
        model server, so model = served_model_name; stream is inferred from the response media
        type. Token counts are unknown here (SSE usage arrives after the handler returns)."""
        path = request.url.path
        if not path.startswith(TRACKED_REQUEST_PREFIXES):
            return await call_next(request)
        start = time.monotonic()
        response = await call_next(request)
        duration_ms = int((time.monotonic() - start) * 1000)
        ctype = response.headers.get("content-type", "")
        state = get_state()
        state.request_ring.add(
            RequestRecord(
                ts=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                method=request.method,
                path=path,
                status=response.status_code,
                model=getattr(state.config, "served_model_name", None),
                duration_ms=duration_ms,
                ttft_ms=None,
                prompt_tokens=None,
                completion_tokens=None,
                stream=ctype.startswith("text/event-stream"),
                error=None,
            )
        )
        return response

    @app.get("/health")
    async def health():
        return build_health(get_state(), app.version)

    @app.get("/v1/requests")
    async def list_requests(since: int = 0, limit: int = 100):
        limit = max(1, min(limit, 512))
        entries, next_cursor = get_state().request_ring.since(since, limit)
        return {"entries": entries, "next_cursor": next_cursor}

    @app.get("/v1/stats")
    async def stats():
        state = get_state()
        doc = build_stats(state, state.request_ring.p95_ms(), state.request_ring.ttft_mean_ms())
        # Surface the model's recommended sampling (from its generation_config.json) so
        # clients can seed their sampling controls per-model instead of guessing.
        if get_model_sampling is not None:
            doc["model"]["sampling"] = get_model_sampling() or {}
        return doc

    @app.get("/v1/cache/status")
    async def cache_status():
        state = get_state()
        return {
            "state": state.maintenance_state,
            "last_rebuild": None,
            "geometry": cache_geometry(state),
        }


__all__ = ["MoeCacheGeometry", "build_health", "cache_geometry", "register_control_routes"]
