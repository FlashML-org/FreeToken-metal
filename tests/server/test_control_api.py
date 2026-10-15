"""The control plane end to end over the real tokenizer worker and a scripted scheduler:
/health, /v1/stats, /v1/requests, /v1/cache/status, prepare-stop, the request log, CORS and
the access-log filter that hides the polled routes."""

from __future__ import annotations

import dataclasses
import http.client
import json
import logging
import queue
import struct
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import JsonValue

from ftmetal import __version__
from ftmetal.env import ENV_PREFIXES
from ftmetal.mm.config import ENCODER_KINDS, EncoderKind, MultimodalConfig
from ftmetal.server.access_log_filter import PollingAccessLogFilter
from ftmetal.server.api_server import FrontendManager, create_app
from ftmetal.server.control_api import MoeCacheGeometry, cache_geometry
from scripted_server import (
    WAIT_SECONDS,
    ScriptedScheduler,
    running_server,
    sse_events,
    tiny_server_args,
    wait_for_abort,
)
from tiny_model import TINY_TEXT_CONFIG
from tiny_vl_model import QWEN_PREPROCESSOR_CONFIG, TINY_IMAGE_TOKEN_ID, TINY_VISION_CONFIG

CHAT: dict[str, JsonValue] = {"model": "tiny", "messages": [{"role": "user", "content": "hi"}]}
# prepare-stop answers loopback clients only.
LOOPBACK_CLIENT = ("127.0.0.1", 50000)


def write_model_config(model_dir: Path, tensor_names: tuple[str, ...] = ()) -> None:
    """A tiny Qwen3.6-shaped config.json and a checkpoint of one-float tensor_names, so the
    model card parses."""
    config = {
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "model_type": "qwen3_5_moe",
        "text_config": TINY_TEXT_CONFIG,
    }
    (model_dir / "config.json").write_text(json.dumps(config))
    entries = {
        name: {"dtype": "F32", "shape": [1], "data_offsets": [4 * i, 4 * i + 4]}
        for i, name in enumerate(tensor_names)
    }
    header = json.dumps(entries).encode()
    data = bytes(4 * len(tensor_names))
    (model_dir / "model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + data)


def test_health_follows_the_engine_lifecycle(
    tiny_model_dir: Path, scripted: type[ScriptedScheduler], monkeypatch: pytest.MonkeyPatch
) -> None:
    args = tiny_server_args(tiny_model_dir)
    with TestClient(create_app(args), client=LOOPBACK_CLIENT) as client:
        serving = client.get("/health").json()
        assert client.post("/v1/admin/prepare-stop").status_code == 200
        stopping = client.get("/health").json()
    instance_id = serving.pop("instance_id")
    assert isinstance(instance_id, str) and isinstance(serving.pop("uptime_s"), int)
    assert serving == {
        "status": "ok",
        "model": "tiny-model",
        "maintenance": "serving",
        "version": __version__,
    }
    assert (stopping["status"], stopping["maintenance"]) == ("ok", "stopping")
    assert stopping["instance_id"] == instance_id

    monkeypatch.setattr(scripted, "fails", True)
    with TestClient(create_app(args)) as client:
        assert client.post("/v1/chat/completions", json=CHAT).status_code == 400
        deadline = time.monotonic() + WAIT_SECONDS
        while any(thread.name == "ftmetal-scheduler" for thread in threading.enumerate()):
            assert time.monotonic() < deadline, "the scheduler thread outlived its failure"
            time.sleep(0.01)
        failed = client.get("/health").json()
    assert failed == {
        "status": "error",
        "message": "backend thread ftmetal-scheduler exited (restart required)",
        "instance_id": failed["instance_id"],
    }


def test_stats_count_requests_and_keep_the_last_pool_usage(
    tiny_model_dir: Path, scripted: type[ScriptedScheduler], monkeypatch: pytest.MonkeyPatch
) -> None:
    write_model_config(tiny_model_dir)
    monkeypatch.setattr(scripted, "mamba_total_slots", 8)
    with TestClient(create_app(tiny_server_args(tiny_model_dir))) as client:
        idle = client.get("/v1/stats").json()
        chat = client.post("/v1/chat/completions", json=CHAT).json()["usage"]
        completion = client.post("/v1/completions", json={"model": "tiny", "prompt": "hi"})
        # A prompt the scheduler refuses ends with an error reply, which carries no pool usage.
        monkeypatch.setattr(scripted, "max_prompt_tokens", 1)
        assert client.post("/v1/chat/completions", json=CHAT).status_code == 400
        busy = client.get("/v1/stats").json()
    usage = [chat, completion.json()["usage"]]
    assert (idle["kv"], idle["mamba"], idle["swa"]) == (None, None, None)
    assert idle["requests"]["completed"] == 0 and idle["throughput"]["decode_tps"] == 0.0
    assert busy["model"] == {
        "id": "tiny-model",
        "ctx": 512,
        "attn": "hybrid_linear",
        "moe": True,
        "input_modalities": ["text"],
        "sampling": {"temperature": 0.6, "top_k": 20, "top_p": 0.95},
    }
    # The two admitted requests' last step reported them; the refused prompt changed nothing.
    assert busy["kv"] == {"used_pages": 2, "total_pages": 64, "page_size": 1}
    assert busy["mamba"] == {"used_slots": 0, "total_slots": 8}
    assert (busy["swa"], busy["vram_bytes"], busy["gpus"]) == (None, 0, [])
    assert busy["throughput"]["decode_tps"] > 0 and busy["throughput"]["prefill_tps"] > 0
    requests = busy["requests"]
    assert (requests["active"], requests["completed"]) == (0, 3)
    assert requests["prompt_tokens_total"] == sum(item["prompt_tokens"] for item in usage)
    assert requests["completion_tokens_total"] == sum(item["completion_tokens"] for item in usage)
    assert requests["p95_ms"] >= 0 and requests["ttft_mean_ms"] >= 0
    assert busy["instance_id"] == idle["instance_id"]


# (whether config.json has a vision_config, the encoder kinds left unbuilt, the advertised
# modalities)
MODALITY_CASES: list[tuple[bool, frozenset[EncoderKind], list[str]]] = [
    (False, frozenset(), ["text"]),
    (True, frozenset(), ["text", "image"]),
    (True, frozenset(ENCODER_KINDS), ["text"]),
    (True, frozenset({"vision"}), ["text"]),
]


@pytest.mark.parametrize(("has_vision", "disabled_encoders", "modalities"), MODALITY_CASES)
def test_stats_and_models_advertise_the_served_modalities(
    tiny_model_dir: Path,
    scripted: type[ScriptedScheduler],
    has_vision: bool,
    disabled_encoders: frozenset[EncoderKind],
    modalities: list[str],
) -> None:
    write_model_config(tiny_model_dir, ("model.visual.proj.weight",) if has_vision else ())
    if has_vision:
        config = json.loads((tiny_model_dir / "config.json").read_text())
        config.update(vision_config=TINY_VISION_CONFIG, image_token_id=TINY_IMAGE_TOKEN_ID)
        (tiny_model_dir / "config.json").write_text(json.dumps(config))
        preprocessor_config = json.dumps(QWEN_PREPROCESSOR_CONFIG)
        (tiny_model_dir / "preprocessor_config.json").write_text(preprocessor_config)
    args = dataclasses.replace(
        tiny_server_args(tiny_model_dir), mm=MultimodalConfig(disabled_encoders=disabled_encoders)
    )
    with TestClient(create_app(args)) as client:
        stats = client.get("/v1/stats").json()
        models = client.get("/v1/models").json()
    assert stats["model"]["input_modalities"] == modalities
    assert models["data"][0]["input_modalities"] == modalities


# (route, request body, whether the record is a stream, whether it carries token counts): the
# chat protocols are recorded with their totals, /v1/completions and /generate by the
# middleware without them, and count_tokens not at all.
RING_CASES: list[tuple[str, dict[str, JsonValue], bool | None, bool]] = [
    ("/v1/chat/completions", CHAT, False, True),
    ("/v1/chat/completions", {**CHAT, "stream": True}, True, True),
    ("/v1/completions", {"model": "tiny", "prompt": "hi"}, False, False),
    ("/v1/messages/count_tokens", CHAT, None, False),
    ("/v1/messages", {**CHAT, "max_tokens": 64}, False, True),
    ("/v1/responses", {"model": "tiny", "input": "hi", "stream": True}, True, True),
    ("/generate", {"prompt": "hi", "max_tokens": 8}, True, False),
]


def test_requests_ring_pages_with_its_cursor(
    tiny_model_dir: Path, scripted: type[ScriptedScheduler]
) -> None:
    with TestClient(create_app(tiny_server_args(tiny_model_dir))) as client:
        for route, body, _, _ in RING_CASES:
            assert client.post(route, json=body).status_code == 200, route
        everything = client.get("/v1/requests").json()
        first = client.get("/v1/requests", params={"limit": 2}).json()
        rest = client.get("/v1/requests", params={"since": first["next_cursor"]}).json()
        none = client.get("/v1/requests", params={"since": rest["next_cursor"]}).json()
    recorded = [case for case in RING_CASES if case[2] is not None]
    entries = everything["entries"]
    assert [entry["path"] for entry in entries] == [route for route, _, _, _ in recorded]
    assert everything["next_cursor"] == len(recorded)
    for entry, (_, _, stream, has_tokens) in zip(entries, recorded, strict=True):
        assert (entry["method"], entry["status"], entry["model"]) == ("POST", 200, "tiny-model")
        assert entry["stream"] is stream and entry["error"] is None
        assert isinstance(entry["prompt_tokens"], int) is has_tokens
        assert isinstance(entry["completion_tokens"], int) is has_tokens
        assert isinstance(entry["duration_ms"], int)
    assert (first["entries"], first["next_cursor"]) == (entries[:2], 2)
    assert (rest["entries"], rest["next_cursor"]) == (entries[2:], len(recorded))
    assert none == {"entries": [], "next_cursor": len(recorded)}


def test_cache_status_reports_the_pools_and_thinking_gears(
    tiny_model_dir: Path, scripted: type[ScriptedScheduler]
) -> None:
    with TestClient(create_app(tiny_server_args(tiny_model_dir))) as client:
        before = client.get("/v1/cache/status").json()
        client.post("/v1/chat/completions", json=CHAT)
        after = client.get("/v1/cache/status").json()
    assert before["geometry"]["num_pages"] == 0
    assert (after["state"], after["last_rebuild"]) == ("serving", None)
    assert after["geometry"] == {
        "num_pages": 64,
        "page_size": 1,
        "moe_cache_size": 0,
        "num_mamba_slots": 0,
        "num_experts": 0,
        "num_moe_layers": 0,
        "moe_cache_policy": None,
        "unit_bytes": {
            "kv_per_token": 0,
            "moe_per_expert": 0,
            "mamba_per_slot": 0,
            "swa_per_token": 0,
        },
        "swa_full_tokens_ratio": 0.0,
        "swa_page_size": 0,
        "num_swa_pages": 0,
        "cache_budget_bytes": 0,
        "reasoning": {
            "gears": ["off", "on"],
            "default": "on",
            "kwargs": {
                "off": {"enable_thinking": False, "thinking_mode": "disabled"},
                "on": {"enable_thinking": True, "thinking_mode": "enabled"},
            },
        },
        "limits": {
            "kv_tokens": {"min": 1, "max": 0},
            "moe_experts": {"min": 0, "max": 0},
            "mamba_slots": {"min": 0, "max": 0},
            "swa_tokens": {"min": 0, "max": 0},
        },
    }


def test_cache_status_reports_the_moe_offload_cache(tiny_model_dir: Path) -> None:
    """The geometry the scheduler thread reports for an offload cache: Qwen3.6's 40 layers of
    256 experts at rate 0.7."""
    frontend = FrontendManager(
        config=tiny_server_args(tiny_model_dir),
        send_tokenizer=queue.Queue(),
        moe_cache=MoeCacheGeometry(
            num_experts=256, num_moe_layers=40, moe_cache_size=7168, moe_cache_policy="lru"
        ),
    )
    frontend.thinking_profile = frontend.warm_frontend_tokenizer()
    geometry = cache_geometry(frontend)
    assert (
        geometry["num_experts"],
        geometry["num_moe_layers"],
        geometry["moe_cache_size"],
        geometry["moe_cache_policy"],
    ) == (256, 40, 7168, "lru")
    assert geometry["limits"]["moe_experts"] == {"min": 256, "max": 0}


def post(port: int, route: str, body: dict[str, JsonValue]) -> http.client.HTTPResponse:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=WAIT_SECONDS)
    headers = {"Content-Type": "application/json"}
    connection.request("POST", route, body=json.dumps(body), headers=headers)
    return connection.getresponse()


def get(port: int, route: str) -> JsonValue:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=WAIT_SECONDS)
    connection.request("GET", route)
    return json.loads(connection.getresponse().read())


def test_prepare_stop_drains_then_aborts_and_seals_the_totals(
    tiny_model_dir: Path, scripted: type[ScriptedScheduler], monkeypatch: pytest.MonkeyPatch
) -> None:
    write_model_config(tiny_model_dir)
    monkeypatch.setattr(scripted, "step_seconds", 0.01)
    with running_server(create_app(tiny_server_args(tiny_model_dir))) as port:
        stuck = post(port, "/generate", {"prompt": "hi", "max_tokens": 100000, "ignore_eos": True})
        draining = post(port, "/generate", {"prompt": "hi", "max_tokens": 30})
        assert stuck.readline().startswith(b"data: ") and draining.readline().startswith(b"data: ")
        body: dict[str, JsonValue] = {"drain_timeout_s": 1.0, "abort_timeout_s": 2.0}
        stop = post(port, "/v1/admin/prepare-stop", body)
        assert stop.status == 200
        sealed = json.loads(stop.read())
        wait_for_abort(scripted.instances[0], 0)
        drained = sse_events(draining.read().decode())
        refused = post(port, "/generate", {"prompt": "hi", "max_tokens": 4})
        refused_chat = post(port, "/v1/chat/completions", CHAT)
        health = get(port, "/health")
        stats = get(port, "/v1/stats")
        again = json.loads(post(port, "/v1/admin/prepare-stop", body).read())
        aborted = sse_events(stuck.read().decode())
    # The aborted request's client is told, rather than left waiting.
    assert aborted[-2:] == [json.dumps({"error": "request aborted"}), "[DONE]"]
    assert sealed.pop("uptime_s") >= 0
    assert sealed["completion_tokens_total"] > 30
    assert sealed == {
        "instance_id": sealed["instance_id"],
        "model_id": "tiny-model",
        "prompt_tokens_total": 4,
        "completion_tokens_total": sealed["completion_tokens_total"],
        "drain_complete": True,
    }
    assert drained[-1] == "[DONE]" and not any("error" in event for event in drained)
    assert (refused.status, json.loads(refused.read())) == (
        503,
        {"error": "server unavailable: engine is stopping"},
    )
    assert refused_chat.status == 503
    assert isinstance(health, dict) and health["maintenance"] == "stopping"
    assert isinstance(stats, dict) and isinstance(stats["requests"], dict)
    assert (stats["requests"]["active"], stats["requests"]["completed"]) == (0, 1)
    assert again.pop("uptime_s") >= 0 and again == sealed


def test_prepare_stop_answers_loopback_clients_only(
    tiny_model_dir: Path, scripted: type[ScriptedScheduler]
) -> None:
    with TestClient(create_app(tiny_server_args(tiny_model_dir))) as client:
        response = client.post("/v1/admin/prepare-stop")
        assert client.post("/v1/chat/completions", json=CHAT).status_code == 200
    assert (response.status_code, response.json()) == (403, {"error": "loopback access required"})


# (environment variable naming the log directory, None for neither)
REQUEST_LOG_CASES: list[str | None] = ["FTMETAL_API_LOG_DIR", "FREETOKEN_API_LOG_DIR", None]


@pytest.mark.parametrize("variable", REQUEST_LOG_CASES)
def test_request_log_writes_one_line_per_dispatched_request(
    tiny_model_dir: Path,
    scripted: type[ScriptedScheduler],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    variable: str | None,
) -> None:
    for prefix in ENV_PREFIXES:
        monkeypatch.delenv(f"{prefix}API_LOG_DIR", raising=False)
    log_dir = tmp_path / "requests"
    if variable is not None:
        monkeypatch.setenv(variable, str(log_dir))
    count_body: dict[str, JsonValue] = {**CHAT, "system": "Be brief."}
    generate_body: dict[str, JsonValue] = {"prompt": "hi", "max_tokens": 4}
    with TestClient(create_app(tiny_server_args(tiny_model_dir))) as client:
        client.post("/v1/chat/completions", json=CHAT)
        client.post("/v1/messages/count_tokens", json=count_body)
        client.post("/generate", json=generate_body)
        # Refused by body validation: never dispatched.
        assert client.post("/v1/chat/completions", json={"model": "tiny"}).status_code == 422
    if variable is None:
        assert not log_dir.exists()
    else:
        [log_file] = log_dir.iterdir()
        assert log_file.stat().st_mode & 0o777 == 0o600
        records = [json.loads(line) for line in log_file.read_text().splitlines()]
        assert [(record["endpoint"], record["request"]) for record in records] == [
            ("/v1/chat/completions", CHAT),
            ("/v1/messages/count_tokens", count_body),
            ("/generate", generate_body),
        ]
        assert all(record["client"] == "testclient" for record in records)


# (--cors-origins, the request's Origin, the access-control-allow-origin it gets back)
CORS_CASES: list[tuple[str, str, str | None]] = [
    ("tauri://localhost,http://localhost:1420", "http://localhost:1420", "http://localhost:1420"),
    ("tauri://localhost,http://localhost:1420", "https://example.org", None),
    ("*", "https://example.org", "*"),
    ("", "http://localhost:1420", None),
]


@pytest.mark.parametrize(("cors_origins", "origin", "allowed"), CORS_CASES)
def test_cors_headers_follow_the_allow_list(
    tiny_model_dir: Path,
    scripted: type[ScriptedScheduler],
    cors_origins: str,
    origin: str,
    allowed: str | None,
) -> None:
    args = dataclasses.replace(tiny_server_args(tiny_model_dir), cors_origins=cors_origins)
    with TestClient(create_app(args)) as client:
        response = client.get("/v1/models", headers={"Origin": origin})
    assert response.headers.get("access-control-allow-origin") == allowed


# (request path as uvicorn logs it, whether ftmetal logs at DEBUG, whether the line is shown)
ACCESS_LOG_CASES: list[tuple[str, bool, bool]] = [
    ("/health", False, False),
    ("/v1/stats", False, False),
    ("/v1/requests?since=3&limit=50", False, False),
    ("/v1/cache/status", False, False),
    ("/v1", False, False),
    ("/v1/models", False, True),
    ("/v1/chat/completions", False, True),
    ("/v1/stats", True, True),
]


@pytest.mark.parametrize(("path", "debug", "shown"), ACCESS_LOG_CASES)
def test_access_log_hides_the_polled_routes(
    caplog: pytest.LogCaptureFixture, path: str, debug: bool, shown: bool
) -> None:
    record = logging.LogRecord(
        "uvicorn.access",
        logging.INFO,
        __file__,
        0,
        '%s - "%s %s HTTP/%s" %d',
        ("127.0.0.1:50000", "GET", path, "1.1", 200),
        None,
    )
    with caplog.at_level(logging.DEBUG if debug else logging.INFO, logger="ftmetal"):
        assert PollingAccessLogFilter().filter(record) is shown
