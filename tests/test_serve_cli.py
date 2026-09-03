"""``ml-stack-serve``: what it prints, what it exits with, and what it refuses to do.

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
from conftest import (
    LLAMA_SERVER_HELP,
    fake_binary,
    fake_memory,
    fake_process,
    json_reply,
    write_gguf,
)
from ml_stack.client import is_healthy
from ml_stack.serve import cli
from ml_stack.serve.backend import ServerInfo, ServerSpec
from ml_stack.serve.ports import free_port
from ml_stack.testing import FakePreflight

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
        assert "started by 'ml-stack-serve up'" in capsys.readouterr().out

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


class TestTellingTheFleet:
    """A server nobody announced is a server no other machine can use.

    `ml-stack-serve up` leased and recorded the port in its own state file and stopped there, so a
    peer asking who is serving a model saw nothing and loaded its own copy — while a working
    server sat idle on this machine.
    """

    def beacon(self, root):
        from ml_stack.fleet.serving import Serving

        return Serving(root / "serving.json")

    def test_putting_a_model_up_announces_it(self, server, state, tmp_path):
        root = tmp_path / "traind"
        root.mkdir()
        instance = server(serving(slots=2))
        assert cli.main(["up", MODEL, "--port", str(instance.port), "--parallel", "2",
                         "--root", str(root)]) == 0
        served = self.beacon(root).all()
        assert [s.port for s in served] == [instance.port]
        assert served[0].slots == 2
        assert served[0].models == [MODEL]

    def test_taking_it_down_withdraws_it(self, state, tmp_path):
        """A registration outlives its server, and the beacon then points at a dead port."""
        root = tmp_path / "traind"
        root.mkdir()
        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        port = free_port()
        record(state, port, pid=proc.pid, owner_pid=proc.pid)
        self.beacon(root).register(port, models=[MODEL], slots=1)
        try:
            assert [s.port for s in self.beacon(root).all()] == [port]
            assert cli.main(["down", "--port", str(port), "--root", str(root)]) == 0
        finally:
            proc.kill()
            proc.wait(timeout=5)
        assert self.beacon(root).all() == [], "the beacon still points at a dead port"

    def test_a_machine_with_no_fleet_is_not_given_one(self, server, state, tmp_path):
        """Announcing is for machines in a fleet; the rest get no stray files."""
        root = tmp_path / "never-set-up"
        instance = server(serving())
        assert cli.main(["up", MODEL, "--port", str(instance.port), "--root", str(root)]) == 0
        assert not root.exists()
        assert cli.beacon(str(root)) is None


def test_a_draft_is_passed_through_and_auto_asks_the_one_chooser(monkeypatch, tmp_path, capsys):
    """`--draft auto` is `hub.choose_head`'s decision for the binary that will serve, and
    the decision is said out loud; anything else is taken as written."""
    import huggingface_hub

    import ml_stack.hub as hub
    from ml_stack.serve.cli import drafted

    hub._DRAFT_NOTES.clear()
    shelves = {"maker/thing-GGUF": [("weights.gguf", 4_000_000_000),
                                    ("mtp-model.gguf", 40_000_000)]}
    monkeypatch.setattr(hub, "files", lambda repo, **kw: shelves.get(repo, []))

    def no_readme(*a, **k):
        raise OSError("no readme")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", no_readme)
    binary = tmp_path / "llama-server"
    binary.write_text("#!/bin/sh\nexit 0\n")
    binary.chmod(0o755)

    assert drafted("hf:maker/thing-GGUF/weights.gguf", "auto", binary=binary) == \
        "hf:maker/thing-GGUF/mtp-model.gguf"
    assert "shipped beside the weights" in capsys.readouterr().err

    # anything else is taken as written, and asked for nothing
    assert drafted("hf:maker/thing-GGUF/weights.gguf", "/models/small.gguf") == "/models/small.gguf"
    assert drafted("hf:maker/thing-GGUF/weights.gguf", "") == ""

    # a local file from nowhere the Hub knows, with nothing beside it, has no head
    assert drafted(str(tmp_path / "big.gguf"), "auto", binary=binary) == ""
    assert "no head shipped" in capsys.readouterr().err


def test_up_withholds_a_fork_only_head_from_mainline_and_says_why(monkeypatch, tmp_path, capsys):
    """`up --draft auto` on a mainline binary serves no head whose README names a fork, and
    prints the chooser's reason with the README's own sentence under it. Mutation: pass
    `borrows=True` in `cmd_up` -- the head is served and the load fails at the far end."""
    import huggingface_hub

    import ml_stack.hub as hub
    from ml_stack.serve import cli

    hub._DRAFT_NOTES.clear()
    shelves = {"maker/flash-GGUF": [("flash-Q4.gguf", 4_000_000_000),
                                    ("MTP/mtp-flash-shared-Q8_0.gguf", 2_600_000_000)]}
    monkeypatch.setattr(hub, "files", lambda repo, **kw: shelves.get(repo, []))
    readme = tmp_path / "MTP-README.md"
    readme.write_text("These do not work on mainline ggml-org/llama.cpp yet.")

    def fake_download(repo, filename, **kw):
        if filename == "MTP/README.md":
            return str(readme)
        raise OSError("no readme")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    preflight = FakePreflight()
    monkeypatch.setattr("ml_stack.serve.preflight.Preflight", preflight)
    monkeypatch.setattr("ml_stack.hub.room", lambda: 0)
    monkeypatch.setattr(cli, "STATE_FILE", tmp_path / "servers.json")
    binary = tmp_path / "current" / "llama-server"
    binary.parent.mkdir()
    binary.write_text("#!/bin/sh\necho usage: llama-server\n")
    binary.chmod(0o755)

    code = cli.main(["up", "hf:maker/flash-GGUF/flash-Q4.gguf", "--preflight-only",
                     "--binary", str(binary), "--draft", "auto", "--port", str(free_port())])
    err = capsys.readouterr().err
    assert code == 0, err
    assert preflight.seen[0].draft is None
    assert not preflight.seen[0].spec_type
    assert "withheld: the repository's README says it needs a fork" in err
    assert "These do not work on mainline" in err
    assert "--build NAME" in err


def test_auto_finds_the_draft_head_lying_beside_a_local_model(tmp_path):
    """A cached repository puts the mtp- head in the model's own directory, so a local path
    resolves by looking rather than by asking the Hub about a file already on the disk."""
    from ml_stack.serve.cli import drafted

    where = tmp_path / "snapshots" / "abc123"
    where.mkdir(parents=True)
    (where / "thing-it-qat-UD-Q4_K_XL.gguf").write_bytes(b"weights")
    (where / "mmproj-BF16.gguf").write_bytes(b"vision, not a draft")

    model = where / "thing-it-qat-UD-Q4_K_XL.gguf"
    assert drafted(str(model), "auto") == ""          # nothing shipped one

    (where / "mtp-thing-it.gguf").write_bytes(b"draft head")
    assert drafted(str(model), "auto") == str(where / "mtp-thing-it.gguf")

    # and the projector is never mistaken for one
    assert "mmproj" not in drafted(str(model), "auto")


def test_a_head_is_found_across_revisions_of_the_same_repository(tmp_path):
    """A Hub cache keeps one folder per revision. Weights fetched in August and a draft head
    fetched today land in different ones, so "beside the weights" finds nothing -- which is
    what happened: `--draft auto` reported no head for a model that ships three, and would
    have run a whole experiment unaccelerated without saying so."""
    from ml_stack.serve.cli import drafted

    snaps = tmp_path / "models--maker--thing-GGUF" / "snapshots"
    old = snaps / "aaaaaaaa" / "UD-IQ4_XS"
    new = snaps / "bbbbbbbb" / "MTP"
    old.mkdir(parents=True)
    new.mkdir(parents=True)
    model = old / "thing-UD-IQ4_XS-00001-of-00003.gguf"
    model.write_bytes(b"weights")
    (new / "mtp-thing-Q8_0.gguf").write_bytes(b"head")

    assert not sorted(model.parent.glob("mtp-*.gguf")), "nothing beside the weights"
    assert drafted(str(model), "auto").endswith("mtp-thing-Q8_0.gguf")

    # the near cases still win over the far one
    (model.parent / "mtp-thing-right-here.gguf").write_bytes(b"head")
    assert drafted(str(model), "auto").endswith("mtp-thing-right-here.gguf")


def test_a_head_named_for_its_method_is_found_and_says_which_kind(tmp_path):
    """`mtp-` and `eagle3-` are both draft heads; a rule knowing only one reported no draft
    for gpt-oss, which ships two EAGLE3 heads and no mtp- file."""
    from ml_stack.hub import spec_for
    from ml_stack.serve.cli import drafted

    where = tmp_path / "m"
    where.mkdir()
    model = where / "oss-MXFP4.gguf"
    model.write_bytes(b"weights")
    (where / "eagle3-oss-Q8_0.gguf").write_bytes(b"head")

    found = drafted(str(model), "auto")
    assert found.endswith("eagle3-oss-Q8_0.gguf")
    assert spec_for(found) == "draft-eagle3"

    assert drafted(str(model), "") == "", "not asked for, not looked for"
    assert drafted(str(model), "/explicit/path.gguf") == "/explicit/path.gguf"


def test_build_subcommand_is_wired_to_cmd_build(monkeypatch):
    """The parser's job here is just getting every flag to `build.cmd_build` unmangled --
    what it does with them is `test_serve_build.py`'s job."""
    from ml_stack.serve import build

    seen: dict[str, object] = {}

    def fake(args):
        seen.update(vars(args))
        return 0

    monkeypatch.setattr(build, "cmd_build", fake)
    assert cli.main(["build", "--commit", "deadbeef", "--jobs", "4", "--from", "release",
                     "--force"]) == 0
    assert seen["commit"] == "deadbeef"
    assert seen["jobs"] == 4
    assert seen["source_kind"] == "release"
    assert seen["force"] is True
    assert seen["check"] is False
    assert seen["rollback"] is False
    assert seen["persist"] is False


def test_up_refuses_a_flag_the_build_lacks_before_loading(tmp_path, monkeypatch, capsys):
    """The whole path -- lease, then the backend's launch -- against a stand-in that answers
    `--help` without `--draft-max`. Nothing is started; the refusal names the nearest."""
    import subprocess as sp
    from dataclasses import replace

    from ml_stack.serve import backend
    from ml_stack.serve.backend import flags_of

    monkeypatch.setattr(cli, "STATE_FILE", tmp_path / "servers.json")
    monkeypatch.setattr(backend, "_FLAGS", {})
    binary = tmp_path / "llama-server"
    binary.write_text("#!/bin/sh\nif [ \"$1\" = --help ]; then cat <<'HELP'\n"
                      "-m,    --model FNAME                    model path\n"
                      "-c,    --ctx-size N                     size of the prompt context\n"
                      "-ngl,  --gpu-layers, --n-gpu-layers N   layers in VRAM\n"
                      "       --host HOST                      ip address to listen on\n"
                      "       --port PORT                      port to listen on\n"
                      "-fa,   --flash-attn [on|off|auto]       set Flash Attention use\n"
                      "       --jinja                          use jinja template\n"
                      "--spec-draft-n-max N                    tokens to draft (default: 3)\n"
                      "--draft, --draft-max N                  the argument has been removed\n"
                      "HELP\nfi\nexit 0\n")
    binary.chmod(0o755)
    gguf = tmp_path / MODEL
    gguf.write_bytes(b"GGUF" + b"\x00" * 64)
    flags_of(binary)    # the help is read here, once; after this nothing may run
    monkeypatch.setattr(sp, "Popen",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("started")))

    real_lease = cli.ServerManager.lease

    def lease(self, spec, **kw):
        # The real lease, with the retired flag put on the argv and a port nobody holds.
        from ml_stack.serve.ports import free_port
        return real_lease(self, replace(spec, extra_args=("--draft-max", "3"),
                                        port=free_port()), **kw)

    monkeypatch.setattr(cli.ServerManager, "lease", lease)
    code = cli.main(["up", str(gguf), "--binary", str(binary), "--port", str(free_port())])
    err = capsys.readouterr().err
    assert code == 2
    assert err.strip() == "this llama-server has no --draft-max; it has --spec-draft-n-max"


class TestPreflightOnly:
    """``up --preflight-only`` runs every check a load would run and prints the report,
    without ever leasing a server."""

    def test_a_passing_preflight_exits_zero_and_never_leases(self, tmp_path, monkeypatch, capsys):
        import ml_stack.setup as setup_module

        monkeypatch.setattr(cli, "STATE_FILE", tmp_path / "servers.json")
        monkeypatch.setattr(setup_module, "_arches", lambda binary: {"llama"})

        def lease(self, spec, *, timeout=None, roam=True):
            raise AssertionError("--preflight-only must never lease a server")

        monkeypatch.setattr(cli.ServerManager, "lease", lease)

        gguf = write_gguf(tmp_path / MODEL, {
            "general.architecture": "llama", "llama.block_count": 32,
            "llama.attention.head_count_kv": 8, "llama.attention.key_length": 128,
        })
        binary = fake_binary(tmp_path, help_text=LLAMA_SERVER_HELP)
        code = cli.main(["up", str(gguf), "--binary", str(binary), "--preflight-only",
                         "--port", str(free_port())])
        out = capsys.readouterr().out
        assert code == 0
        assert "ok    shards" in out
        assert "ok    architecture" in out

    def test_a_failing_preflight_exits_one(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr("ml_stack.serve.preflight.source_dir", lambda: tmp_path / "no-src")
        import ml_stack.setup as setup_module

        monkeypatch.setattr(cli, "STATE_FILE", tmp_path / "servers.json")
        monkeypatch.setattr(setup_module, "_arches", lambda binary: {"gemma4"})  # not llama

        gguf = write_gguf(tmp_path / MODEL, {
            "general.architecture": "llama", "llama.block_count": 32,
            "llama.attention.head_count_kv": 8, "llama.attention.key_length": 128,
        })
        binary = fake_binary(tmp_path, help_text=LLAMA_SERVER_HELP)
        code = cli.main(["up", str(gguf), "--binary", str(binary), "--preflight-only",
                         "--port", str(free_port())])
        out = capsys.readouterr().out
        assert code == 1
        assert "FAIL  architecture" in out


class TestResolveModel:
    """A bare model name -- no `/`, no `hf:` -- used to be read as a relative path and fail
    preflight saying 'shards missing' for a model already in the Hub cache."""

    def test_a_path_or_hf_reference_is_used_exactly_as_given(self):
        assert cli.resolve_model("hf:maker/thing-GGUF/thing.gguf") == \
            "hf:maker/thing-GGUF/thing.gguf"
        assert cli.resolve_model("some/dir/thing.gguf") == "some/dir/thing.gguf"

    def test_a_bare_name_found_in_the_hub_cache_resolves_to_its_real_path(
            self, tmp_path, monkeypatch):
        import ml_stack.hub as hub_module

        cache = tmp_path / "hub"
        snapshot = cache / "models--maker--thing-GGUF" / "snapshots" / "abc123"
        snapshot.mkdir(parents=True)
        gguf = snapshot / MODEL
        gguf.write_bytes(b"x")
        monkeypatch.setattr(hub_module, "HUB_CACHE", cache)

        assert cli.resolve_model(MODEL) == str(gguf.resolve())

    def test_a_bare_name_found_nowhere_is_returned_unchanged(self, tmp_path, monkeypatch):
        import ml_stack.hub as hub_module

        monkeypatch.setattr(hub_module, "HUB_CACHE", tmp_path / "empty")
        assert cli.resolve_model(MODEL) == MODEL

    def test_up_preflight_only_resolves_a_bare_name_and_reports_it(
            self, tmp_path, monkeypatch, capsys):
        import ml_stack.hub as hub_module
        import ml_stack.setup as setup_module

        monkeypatch.setattr(cli, "STATE_FILE", tmp_path / "servers.json")
        cache = tmp_path / "hub"
        snapshot = cache / "models--maker--thing-GGUF" / "snapshots" / "abc123"
        snapshot.mkdir(parents=True)
        gguf = write_gguf(snapshot / MODEL, {
            "general.architecture": "llama", "llama.block_count": 32,
            "llama.attention.head_count_kv": 8, "llama.attention.key_length": 128,
        })
        monkeypatch.setattr(hub_module, "HUB_CACHE", cache)
        monkeypatch.setattr(setup_module, "_arches", lambda binary: {"llama"})

        binary = tmp_path / "llama-server"
        binary.write_text("#!/bin/sh\nif [ \"$1\" = --help ]; then cat <<'HELP'\n"
                          + LLAMA_SERVER_HELP + "HELP\nfi\nexit 0\n")
        binary.chmod(0o755)

        def lease(self, spec, *, timeout=None, roam=True):
            raise AssertionError("--preflight-only must never lease a server")

        monkeypatch.setattr(cli.ServerManager, "lease", lease)

        code = cli.main(["up", MODEL, "--binary", str(binary), "--preflight-only",
                         "--port", str(free_port())])
        out = capsys.readouterr()
        assert code == 0
        assert "ok    shards" in out.out
        assert f"resolved {MODEL} -> {gguf.resolve()}" in out.err


class TestBuildFlag:
    """``up --build NAME`` picks the binary through the named build rather than 'current'."""

    @pytest.fixture
    def built(self, tmp_path, monkeypatch):
        """Every `LlamaServerBackend` built while `up` runs, in order, without leasing one.

        `Preflight` builds a second one (with the already-resolved path) to check flags, so
        every construction is recorded rather than one captured value overwritten -- only
        the *first* is `cmd_up`'s own.
        """
        from ml_stack.serve import backend as backend_module

        monkeypatch.setattr(cli, "STATE_FILE", tmp_path / "servers.json")
        calls: list[dict] = []

        class Spy(backend_module.LlamaServerBackend):
            def __init__(self, **kw):
                calls.append(dict(kw))
                super().__init__(**kw)

        monkeypatch.setattr(backend_module, "LlamaServerBackend", Spy)

        def lease(self, spec, *, timeout=None, roam=True):
            raise AssertionError("this fixture only records how the backend was built")

        monkeypatch.setattr(cli.ServerManager, "lease", lease)

        gguf = tmp_path / MODEL
        gguf.write_bytes(b"GGUF" + b"\x00" * 64)

        def run(*extra: str) -> list[dict]:
            calls.clear()
            cli.main(["up", str(gguf), *extra, "--preflight-only", "--port", str(free_port())])
            assert calls, "cmd_up must construct its own backend"
            return calls

        return run

    def test_the_named_build_is_threaded_into_the_backend(self, built):
        calls = built("--build", "unsloth")
        assert calls[0].get("build") == "unsloth"
        assert not calls[0].get("binary"), "no --binary was given, so none may be invented"

    def test_an_explicit_binary_is_passed_alongside_and_wins_at_resolution(self, built):
        calls = built("--binary", "/some/llama-server", "--build", "unsloth")
        assert calls[0].get("binary") == "/some/llama-server"
        assert calls[0].get("build") == "unsloth"


class TestModelsFetch:
    def test_fetch_downloads_and_prints_each_reference(self, monkeypatch, capsys, tmp_path):
        import huggingface_hub
        import ml_stack.hub as hub

        monkeypatch.setattr(hub, "files", lambda repo, **kw: [("thing-Q4_K_M.gguf", 4_000)])

        def fake_download(repo_id, filename, **kw):
            target = tmp_path / filename
            target.write_bytes(b"x" * 10)
            return str(target)

        monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
        assert hub.main(["fetch", "hf:maker/thing-GGUF/thing-Q4_K_M.gguf"]) == 0
        assert "thing-Q4_K_M.gguf" in capsys.readouterr().out


def test_preflight_only_resolves_a_draft_named_by_file(monkeypatch, tmp_path, capsys):
    """`up --preflight-only` with `--draft hf:owner/repo/MTP/head.gguf` refused the reference
    where a real start would have fetched it and served by path. Mutation: drop the
    resolved_draft call before Preflight."""
    from ml_stack.serve import cli

    head = tmp_path / "mtp-head-Q8_0.gguf"
    head.write_bytes(b"GGUF")
    model = tmp_path / "m.gguf"
    model.write_bytes(b"GGUF")
    monkeypatch.setattr("ml_stack.hub.fetch", lambda ref: head)
    preflight = FakePreflight()
    monkeypatch.setattr("ml_stack.serve.preflight.Preflight", preflight)
    monkeypatch.setattr("ml_stack.hub.room", lambda: 0)
    binary = tmp_path / "llama-server"
    binary.write_text("#!/bin/sh\necho usage: llama-server\n")
    binary.chmod(0o755)
    code = cli.main(["up", str(model), "--preflight-only", "--binary", str(binary),
                     "--draft", "hf:owner/repo-GGUF/MTP/mtp-head-Q8_0.gguf"])
    assert code == 0, capsys.readouterr()
    assert preflight.seen[0].draft == str(head)


def test_a_sharded_model_in_the_hub_cache_resolves_to_its_name_not_its_blob(tmp_path, monkeypatch):
    """A Hub cache entry is a symlink into blobs/; llama.cpp finds the other shards by the
    name, and handed the blob path said 'invalid split file name' (gpt-oss-120b, 2026-09-02).
    The link is returned, and its size still reads through to the blob."""
    from pathlib import Path

    import ml_stack.hub as hub_module

    cache = tmp_path / "hub"
    blobs = cache / "models--maker--big-GGUF" / "blobs"
    snapshot = cache / "models--maker--big-GGUF" / "snapshots" / "abc123"
    blobs.mkdir(parents=True)
    snapshot.mkdir(parents=True)
    for n, blob in ((1, "aa" * 32), (2, "bb" * 32)):
        (blobs / blob).write_bytes(b"x" * (100 * n))
        (snapshot / f"big-Q4_K_M-0000{n}-of-00002.gguf").symlink_to(blobs / blob)
    monkeypatch.setattr(hub_module, "HUB_CACHE", cache)

    found = cli.resolve_model("big-Q4_K_M-00001-of-00002.gguf")
    assert found == str(snapshot / "big-Q4_K_M-00001-of-00002.gguf")
    assert Path(found).is_symlink() and Path(found).stat().st_size == 100


def test_status_every_lists_each_llama_server_and_says_which_nobody_leased(monkeypatch, capsys):
    """A stray server -- a Homebrew one from before the managed build -- holds memory a lease
    cannot see; `status --every` names it, and `pgrep` by hand is what the guard refuses."""
    import psutil

    fakes = [fake_process(["/opt/homebrew/bin/llama-server", "--port", "8081", "-m",
                           "/models/embeddinggemma-300M-Q8_0.gguf"], 1 * 2**30, pid=11),
             fake_process(["/x/current/llama-server", "-m", "/models/thing-UD-Q4_K_XL.gguf",
                           "--port", "8082"], 60 * 2**30, pid=12),
             fake_process(["/usr/bin/python3", "-m", "something"], 5, pid=13),
             fake_process(["llama-server"], 0, pid=14, state=psutil.STATUS_ZOMBIE)]
    monkeypatch.setattr(psutil, "process_iter", lambda attrs=None: fakes)
    monkeypatch.setattr(cli, "recorded_servers", lambda path: {8082: {"pid": 12}})
    assert cli.main(["status", "--every"]) == 0
    out = capsys.readouterr().out
    assert ":8081  pid 11  embeddinggemma-300M (Q8_0)  1.0G resident  NOT leased" in out
    assert ":8082  pid 12  thing (Q4_K_XL)  60.0G resident  leased" in out
    assert "1 not leased" in out and "python3" not in out
    assert "pid 14  defunct" in out, "a zombie is named as such and not counted as a stray"
    monkeypatch.setattr(psutil, "process_iter", lambda attrs=None: [])
    assert cli.main(["status", "--every"]) == 1
    assert "no llama-server is running" in capsys.readouterr().out


def test_machine_memory_splits_the_servers_from_everything_else(monkeypatch):
    import psutil

    G = 2**30
    monkeypatch.setattr(psutil, "virtual_memory",
                        lambda: fake_memory(total=128 * G, available=17 * G, wired=67 * G))
    monkeypatch.setattr(psutil, "process_iter", lambda attrs=None: [
        fake_process(["/x/llama-server", "--port", "8099"], 73 * G),
        fake_process(["/Applications/Browser.app/Contents/MacOS/Browser"], 2 * G),
        fake_process(["/usr/bin/editor"], 1 * G)])
    got = cli.machine_memory()
    assert got["total"] == 128 * G and got["used"] == 111 * G and got["wired"] == 67 * G
    assert got["servers"] == 73 * G and got["others"] == 3 * G
    assert got["largest"] == ["Browser 2.0G", "editor 1.0G"]


def test_up_kv_stores_the_cache_as_asked(tmp_path, monkeypatch):
    """`--kv q8_0` reaches both cache types of the spec; nothing asked leaves the server's
    own default."""
    import ml_stack.hub as hub_module

    seen = {}

    class Manager:
        def __init__(self, *a, **k):
            self.backend = types.SimpleNamespace(binary=None)

        def lease(self, spec, **kw):
            seen["spec"] = spec
            raise SystemExit(0)

    import types

    monkeypatch.setattr(cli, "ServerManager", Manager)
    monkeypatch.setattr(hub_module, "HUB_CACHE", tmp_path)
    model = tmp_path / "tiny.gguf"
    model.write_bytes(b"x")
    try:
        cli.main(["up", str(model), "--kv", "q8_0", "--port", "1"])
    except SystemExit:
        pass
    assert seen["spec"].cache_type_k == "q8_0" and seen["spec"].cache_type_v == "q8_0"
    try:
        cli.main(["up", str(model), "--port", "1"])
    except SystemExit:
        pass
    assert seen["spec"].cache_type_k == "" and seen["spec"].cache_type_v == ""
    assert seen["spec"].embedding is False
    try:
        cli.main(["up", str(model), "--port", "1", "--embedding"])
    except SystemExit:
        pass
    assert seen["spec"].embedding is True
