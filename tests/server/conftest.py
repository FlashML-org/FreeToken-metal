"""The scripted scheduler in place of the real one, and a client of the app it serves."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ftmetal.server import api_server
from ftmetal.server.api_server import create_app
from scripted_server import ScriptedScheduler, tiny_server_args


@pytest.fixture
def scripted(monkeypatch: pytest.MonkeyPatch) -> type[ScriptedScheduler]:
    ScriptedScheduler.instances.clear()
    monkeypatch.setattr(api_server, "Scheduler", ScriptedScheduler)
    return ScriptedScheduler


@pytest.fixture
def client(tiny_model_dir: Path, scripted: type[ScriptedScheduler]) -> Iterator[TestClient]:
    with TestClient(create_app(tiny_server_args(tiny_model_dir))) as test_client:
        yield test_client
