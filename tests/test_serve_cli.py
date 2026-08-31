"""``ml-serve``: what it prints, what it exits with, and what it refuses to do.

The adoption paths run against a real HTTP server standing in for a live llama-server,
so the refusals here are the library's own, reached over a real socket. No model is
loaded anywhere in this file.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import pytest
from conftest import json_reply
from ml_stack.client import is_healthy
from ml_stack.serve import cli
from ml_stack.serve.backend import ServerInfo, ServerSpec
from ml_stack.serve.ports import free_port

MODEL = "tinyfixture-4B-Q4_K_M.gguf"


@pytest.fixture
def state(tmp_path, monkeypatch):
    """Point the CLI at a lease file of its own."""
    path = tmp_path / "servers.json"
    monkeypatch.setattr(cli, "STATE_FILE", path)
    return path


def serving(model: str = MODEL, n_ctx: int = 4096, slots: int = 1):
    """A handler answering /health, /v1/models and /props like llama-server does."""
    def handle(method, path, body):
        if path.startswith("/props"):
            return json_reply({
                "model_path": f"/models/{model}",
                "total_slots": slots,
                "default_generation_settings": {"n_ctx": n_ctx},
            })
        return json_reply({"data": [{"id": model}]})

    return handle


def record(state, port: int, *, pid: int, owner_pid: int) -> None:
    state.write_text(json.dumps({str(port): {
        "port": port, "pid": pid, "owner_pid": owner_pid, "backend": "llama.cpp",
        "model": MODEL, "base_url": f"http://127.0.0.1:{port}",
    }}))


class TestStatus:
    def test_nothing_serving_exits_nonzero_and_says_so(self, state, capsys):
        """A shell gating on this needs a failing exit, not an empty list."""
        port = free_port()
        assert cli.main(["status", "--port", str(port)]) == 1
        assert f"nothing is serving on port {port}" in capsys.readouterr().out

    def test_nothing_serving_in_json_is_still_a_failing_exit(self, state, capsys):
        assert cli.main(["status", "--port", str(free_port()), "--json"]) == 1
        payload = json.loads(capsys.readouterr().out)
        assert payload["serving"] is False
        assert payload["servers"] == []

    def test_a_running_server_is_described(self, server, state, capsys):
        instance = server(serving(n_ctx=8192, slots=2))
        assert cli.main(["status", "--port", str(instance.port)]) == 0

        out = capsys.readouterr().out
        assert instance.base_url in out
        assert MODEL in out and "Q4_K_M" in out
        assert "8192 per slot" in out
        assert "slots    2" in out

    def test_a_running_server_in_json(self, server, state, capsys):
        instance = server(serving(n_ctx=8192, slots=2))
        assert cli.main(["status", "--port", str(instance.port), "--json"]) == 0

        payload = json.loads(capsys.readouterr().out)
        assert payload["serving"] is True
        row = payload["servers"][0]
        assert row["port"] == instance.port
        assert row["base_url"] == instance.base_url
        assert row["model"] == MODEL
        assert row["context"] == 8192
        assert row["slots"] == 2
        assert row["verdict"] == "adopt"

    def test_it_says_whether_a_lease_would_adopt_or_start(self, server, state, capsys):
        instance = server(serving(n_ctx=4096, slots=1))
        assert cli.main(["status", "--port", str(instance.port)]) == 0
        assert "would adopt this server" in capsys.readouterr().out

    def test_a_lease_that_would_be_refused_says_which_field(self, server, state, capsys):
        instance = server(serving(n_ctx=4096, slots=1))
        assert cli.main(["status", "--port", str(instance.port),
                         "--parallel", "4", "--context", "32768"]) == 0

        out = capsys.readouterr().out
        assert "would be refused" in out
        assert "slots: asked for 4, serving 1" in out
        assert "context: asked for 8192 per slot, serving 4096" in out

    def test_a_recorded_port_is_surveyed_without_being_named(self, server, state, capsys):
        instance = server(serving())
        record(state, instance.port, pid=4242, owner_pid=4242)

        assert cli.main(["status", "--port", str(free_port())]) == 0
        assert instance.base_url in capsys.readouterr().out

    def test_a_server_ml_serve_started_is_named_as_such(self, server, state, capsys):
        instance = server(serving())
        record(state, instance.port, pid=4242, owner_pid=4242)

        assert cli.main(["status", "--port", str(instance.port)]) == 0
        assert "started by 'ml-serve up'" in capsys.readouterr().out

    def test_the_process_holding_the_lease_is_named(self, server, state, capsys):
        instance = server(serving())
        record(state, instance.port, pid=999_999_998, owner_pid=os.getpid())

        assert cli.main(["status", "--port", str(instance.port)]) == 0
        assert f"held by process {os.getpid()}" in capsys.readouterr().out

    def test_an_unrecorded_server_says_down_will_not_touch_it(self, server, state, capsys):
        instance = server(serving())
        assert cli.main(["status", "--port", str(instance.port)]) == 0
        assert "none on record" in capsys.readouterr().out


class TestUp:
    def test_it_adopts_a_compatible_server(self, server, state, capsys):
        instance = server(serving(n_ctx=4096, slots=1))
        assert cli.main(["up", MODEL, "--port", str(instance.port)]) == 0

        out = capsys.readouterr().out
        assert out.startswith("adopted ")
        assert instance.base_url in out

    def test_adoption_is_reported_as_adoption_in_json(self, server, state, capsys):
        instance = server(serving(n_ctx=4096, slots=1))
        assert cli.main(["up", MODEL, "--port", str(instance.port), "--json"]) == 0

        payload = json.loads(capsys.readouterr().out)
        assert payload["adopted"] is True
        assert payload["base_url"] == instance.base_url

    def test_it_refuses_a_different_shape_on_a_busy_port(self, server, state, capsys):
        """Adopting the wrong shape is what makes a caller reload the weights it
        already had."""
        instance = server(serving(n_ctx=4096, slots=1))
        code = cli.main(["up", MODEL, "--port", str(instance.port),
                         "--parallel", "4", "--context", "32768"])

        assert code == 2
        err = capsys.readouterr().err
        assert "slots: asked for 4, serving 1" in err
        assert "context: asked for 8192 per slot, serving 4096" in err
        assert is_healthy(instance.base_url), "the refusal must not have stopped it"

    def test_it_refuses_a_different_model_on_a_busy_port(self, server, state, capsys):
        instance = server(serving())
        code = cli.main(["up", "other-8B-Q4_K_M.gguf", "--port", str(instance.port)])

        assert code == 2
        err = capsys.readouterr().err
        assert f"model: asked for 'other-8B-Q4_K_M.gguf', serving '{MODEL}'" in err

    def test_the_flags_default_to_the_modules_own_shape(self, state, monkeypatch, capsys):
        seen: dict[str, object] = {}

        def lease(self, spec, *, timeout=300.0):
            seen["spec"] = spec
            seen["timeout"] = timeout
            return ServerInfo(base_url=f"http://127.0.0.1:{spec.port}", port=spec.port,
                              pid=None, backend="llama.cpp", adopted=True)

        monkeypatch.setattr(cli.ServerManager, "lease", lease)
        assert cli.main(["up", MODEL]) == 0

        spec = seen["spec"]
        default = ServerSpec(model="")
        assert spec.port == default.port
        assert spec.context == default.context
        assert spec.parallel == default.parallel
        assert str(spec.model) == MODEL


class TestDown:
    def test_it_refuses_a_server_this_machine_has_no_record_of(self, server, state, capsys):
        instance = server(serving())
        assert cli.main(["down", "--port", str(instance.port)]) == 2

        err = capsys.readouterr().err
        assert "no record of starting it" in err
        assert is_healthy(instance.base_url), "it must still be running"

    def test_it_refuses_a_server_another_live_process_holds(self, server, state, capsys):
        instance = server(serving())
        record(state, instance.port, pid=999_999_998, owner_pid=os.getpid())

        assert cli.main(["down", "--port", str(instance.port)]) == 2
        assert f"held by process {os.getpid()}" in capsys.readouterr().err
        assert is_healthy(instance.base_url)

    def test_it_stops_a_server_this_machine_started(self, state, capsys):
        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        port = free_port()
        record(state, port, pid=proc.pid, owner_pid=proc.pid)
        try:
            assert cli.main(["down", "--port", str(port)]) == 0
            assert f"stopped http://127.0.0.1:{port}" in capsys.readouterr().out

            deadline = time.monotonic() + 10
            while proc.poll() is None and time.monotonic() < deadline:
                time.sleep(0.05)
            assert proc.poll() is not None, "the recorded process is still running"
            assert json.loads(state.read_text()) == {}
        finally:
            if proc.poll() is None:
                proc.kill()
            proc.wait()

    def test_a_stale_record_is_cleared_without_pretending_it_stopped_something(
            self, state, capsys):
        port = free_port()
        record(state, port, pid=999_999_998, owner_pid=999_999_998)

        assert cli.main(["down", "--port", str(port)]) == 0
        assert "removed the record" in capsys.readouterr().out
        assert json.loads(state.read_text()) == {}

    def test_a_quiet_port_exits_nonzero(self, state, capsys):
        port = free_port()
        assert cli.main(["down", "--port", str(port)]) == 1
        assert "nothing is serving" in capsys.readouterr().out
