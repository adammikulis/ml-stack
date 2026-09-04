"""ml-stack-claude: Claude Code on a served model, in its measured shape, off the network."""

import contextlib
import json

from ml_stack import claude


def test_the_environment_points_every_model_call_at_the_server_and_nothing_elsewhere():
    env = claude.environment("http://127.0.0.1:8080/", "kestrel-8B",
                             base={"ANTHROPIC_API_KEY": "sk-real", "HOME": "/h"})
    assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8080"
    assert env["ANTHROPIC_AUTH_TOKEN"] == "local" and "ANTHROPIC_API_KEY" not in env
    assert {env[name] for name in claude.MODEL_VARS} == {"kestrel-8B"}
    assert env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] == "1"
    assert env["CLAUDE_CODE_DISABLE_1M_CONTEXT"] == "1" and env["HOME"] == "/h"
    online = claude.environment("http://x", "m", offline=False, base={})
    assert "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC" not in online
    assert json.loads(claude.settings()) == {"skipWebFetchPreflight": True,
                                             "alwaysThinkingEnabled": False}


def test_launch_leases_the_measured_shape_and_runs_claude_inside_it(monkeypatch, tmp_path):
    from ml_stack.serve.profile import record

    seen = {}

    class Server:
        base_url = "http://127.0.0.1:8899"

    @contextlib.contextmanager
    def fake_serve(model, manager=None, **lease):
        seen["lease"] = lease
        seen["model"] = model
        yield Server()
        seen["released"] = True

    profile = record("kestrel-8B-UD-Q4_K_XL.gguf", cache_type="q8_0", tight=True, batch=True)
    monkeypatch.setattr("ml_stack.serve.manager.serve", fake_serve)
    monkeypatch.setattr("ml_stack.serve.profile.profile_for", lambda m: profile)
    monkeypatch.setattr("ml_stack.graph.bench.serve.find_model", lambda m: "/models/kestrel-8B-UD-Q4_K_XL.gguf")
    monkeypatch.setattr(claude, "alias_of", lambda url, model: "kestrel-8B")
    binary = tmp_path / "claude"
    binary.write_text("#!/bin/sh\nexit 0\n")
    binary.chmod(0o755)

    def run_claude(command, env):
        seen["command"], seen["env"] = command, env
        assert not seen.get("released"), "claude runs inside the lease"
        return 7

    code = claude.launch(["kestrel", "--port", "8899", "--claude", str(binary), "--",
                          "--print", "hello"], say=lambda _: None, run_claude=run_claude)
    assert code == 7
    assert seen["lease"]["port"] == 8899 and seen["lease"]["parallel"] == 1, "one conversation, one seat"
    assert seen["lease"]["cache_type_k"] == "q8_0", "the measured shape"
    assert seen["command"][0] == str(binary) and seen["command"][1] == "--settings"
    assert seen["command"][-2:] == ["--print", "hello"]
    assert seen["env"]["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8899"
    assert seen["env"]["ANTHROPIC_MODEL"] == "kestrel-8B"
    assert seen["released"], "and the server goes when claude exits"


def test_seats_asked_for_reach_the_lease(monkeypatch, tmp_path):
    """One seat holds the whole measured cache; `--seats N` divides it between N."""
    from ml_stack.serve.profile import record

    seen = {}

    class Server:
        base_url = "http://127.0.0.1:8899"

    @contextlib.contextmanager
    def fake_serve(model, manager=None, **lease):
        seen["lease"] = lease
        yield Server()

    profile = record("kestrel-8B-UD-Q4_K_XL.gguf", seat_context=32768, parallel=2)
    monkeypatch.setattr("ml_stack.serve.manager.serve", fake_serve)
    monkeypatch.setattr("ml_stack.serve.profile.profile_for", lambda m: profile)
    monkeypatch.setattr("ml_stack.graph.bench.serve.find_model",
                        lambda m: "/models/kestrel-8B-UD-Q4_K_XL.gguf")
    monkeypatch.setattr(claude, "alias_of", lambda url, model: "kestrel-8B")
    binary = tmp_path / "claude"
    binary.write_text("#!/bin/sh\nexit 0\n")
    binary.chmod(0o755)
    started = ["kestrel", "--port", "8899", "--claude", str(binary)]

    claude.launch(started, say=lambda _: None, run_claude=lambda command, env: 0)
    assert (seen["lease"]["parallel"], seen["lease"]["context"]) == (1, 65536), \
        "one seat holding the whole cache the record measured across two"

    claude.launch([*started, "--seats", "4"], say=lambda _: None,
                  run_claude=lambda command, env: 0)
    assert (seen["lease"]["parallel"], seen["lease"]["context"]) == (4, 131072), \
        "each seat asked for gets what one measured seat got"


def test_launch_refuses_without_a_claude_binary(monkeypatch, capsys):
    monkeypatch.setattr(claude.shutil, "which", lambda name: None)
    assert claude.launch(["kestrel"], say=print) == 2
    assert "no `claude` on PATH" in capsys.readouterr().out


def test_alias_falls_back_to_the_file_stem(monkeypatch):
    monkeypatch.setattr("ml_stack.client.reported_models", lambda url, **kw: [])
    assert claude.alias_of("http://127.0.0.1:1", "/m/kestrel-8B-UD-Q4_K_XL.gguf") == "kestrel-8B-UD-Q4_K_XL"
    monkeypatch.setattr("ml_stack.client.reported_models", lambda url, **kw: ["served-name"])
    assert claude.alias_of("http://127.0.0.1:1", "x.gguf") == "served-name"


class TestPicking:
    """`ml-stack-claude` with nothing named."""

    def _options(self):
        return [{"kind": "server", "port": 8080, "url": "http://127.0.0.1:8080",
                 "name": "a served model", "note": "already running"},
                {"kind": "model", "name": "quince-2b.gguf", "record": None, "note": ""},
                {"kind": "model", "name": "brack-9b.gguf", "record": None, "note": ""}]

    def test_a_bare_number_takes_that_one(self):
        from ml_stack.claude import pick

        assert pick(self._options(), say=lambda _: None, ask=lambda _: "2")["name"] == \
            "quince-2b.gguf"

    def test_nothing_typed_takes_the_first(self):
        from ml_stack.claude import pick

        got = pick(self._options(), say=lambda _: None, ask=lambda _: "")
        assert got["kind"] == "server"

    def test_a_name_takes_the_one_it_names(self):
        from ml_stack.claude import pick

        got = pick(self._options(), say=lambda _: None, ask=lambda _: "brack")
        assert got["name"] == "brack-9b.gguf"

    def test_a_name_matching_none_chooses_none(self):
        from ml_stack.claude import pick

        said: list[str] = []
        assert pick(self._options(), say=said.append, ask=lambda _: "velthorne") is None
        assert any("not a choice" in line for line in said)

    def test_nothing_to_run_says_so(self):
        from ml_stack.claude import pick

        said: list[str] = []
        assert pick([], say=said.append, ask=lambda _: "1") is None
        assert any("nothing to run" in line for line in said)

    def test_the_servers_are_offered_before_the_models(self):
        from ml_stack.claude import pick

        said: list[str] = []
        pick(self._options(), say=said.append, ask=lambda _: "1")
        assert said[0] == "already running:"
