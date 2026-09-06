"""A lease will not load a model onto a card somebody else is measuring."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import pytest

from ml_stack.serve import Measuring, ServerInfo, ServerManager, ServerSpec, free_port


class _Backend:
    """Records what it was asked to start; never starts anything."""

    name = "fake"

    def __init__(self) -> None:
        self.started: list[int] = []

    def start(self, spec, *, lease, timeout=300.0, **starting):
        self.started.append(spec.port)
        return ServerInfo(base_url=f"http://127.0.0.1:{spec.port}", port=spec.port,
                          pid=4242, backend="fake")

    def stop(self, info, *, grace_s=5.0):
        return None


@pytest.fixture
def stranger():
    """A live process this one did not start and is not descended from."""
    held = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    for _ in range(50):
        if held.poll() is None:
            break
        time.sleep(0.05)
    yield held.pid
    held.kill()
    held.wait(timeout=10)


def _measuring(pid: int, argv: list[str] | None = None) -> None:
    """Write the measuring record and lock for ``pid``."""
    from ml_stack.bench.run import measuring_file, measuring_lock_file

    measuring_file().parent.mkdir(parents=True, exist_ok=True)
    measuring_lock_file().write_text(f"pid {pid}", encoding="utf-8")
    if argv is not None:
        measuring_file().write_text(
            json.dumps({"pid": pid, "argv": argv, "log": "", "how": {},
                        "started": "2026-01-02T03:04:05"}), encoding="utf-8")


def _spec(model: str = "m.gguf") -> ServerSpec:
    return ServerSpec(model=model, port=free_port(), context=4096)


def test_a_lease_is_refused_while_somebody_else_measures(tmp_path, stranger):
    _measuring(stranger, ["ask", "--model", "m.gguf"])
    backend = _Backend()
    manager = ServerManager(backend=backend, state_file=tmp_path / "servers.json")

    with pytest.raises(Measuring) as why:
        manager.lease(_spec(), roam=False, timeout=1.0)

    said = str(why.value)
    assert "the card is being measured by ml-stack-bench ask --model m.gguf" in said
    assert f"(pid {stranger})" in said
    assert "started 2026-01-02T03:04:05" in said
    assert "Wait for it to finish, stop it with 'ml-stack-bench stop', or pass --anyway" \
        in said
    assert backend.started == [], "nothing was loaded onto the card"


def test_a_lock_with_no_record_beside_it_still_refuses(tmp_path, stranger):
    _measuring(stranger)
    manager = ServerManager(backend=_Backend(), state_file=tmp_path / "servers.json")

    with pytest.raises(Measuring, match="wrote no record of itself"):
        manager.lease(_spec(), roam=False, timeout=1.0)


def test_anyway_starts_it_regardless(tmp_path, stranger):
    _measuring(stranger, ["ask", "--model", "m.gguf"])
    backend = _Backend()
    manager = ServerManager(backend=backend, state_file=tmp_path / "servers.json")

    info = manager.lease(_spec(), roam=False, timeout=1.0, anyway=True)
    assert backend.started == [info.port]


def test_the_measurement_adopts_the_server_it_is_serving_with(tmp_path, stranger,
                                                              monkeypatch):
    _measuring(stranger, ["ask", "--model", "m.gguf"])
    backend = _Backend()
    manager = ServerManager(backend=backend, state_file=tmp_path / "servers.json")
    monkeypatch.setattr("ml_stack.serve.manager.is_healthy", lambda *a, **k: True)
    monkeypatch.setattr("ml_stack.serve.manager.reported_models", lambda *a, **k: ["m.gguf"])
    monkeypatch.setattr("ml_stack.serve.manager.serving_params", lambda *a, **k: None)

    info = manager.lease(_spec(), roam=False, timeout=1.0)
    assert info.adopted, "a server already up costs the measurement no memory"
    assert backend.started == []


def test_the_process_holding_the_lock_serves_its_own_model(tmp_path):
    _measuring(os.getpid(), ["ask", "--model", "m.gguf"])
    backend = _Backend()
    manager = ServerManager(backend=backend, state_file=tmp_path / "servers.json")

    info = manager.lease(_spec(), roam=False, timeout=1.0)
    assert backend.started == [info.port], "the holder is not refused by its own lock"
