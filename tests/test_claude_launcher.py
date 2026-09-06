"""ml-stack-claude: Claude Code on a served model, in its measured shape, off the network."""

import contextlib
import functools
import json
from pathlib import Path

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
    monkeypatch.setattr("ml_stack.serve.profile.profile_for", lambda m, **_: profile)
    monkeypatch.setattr("ml_stack.hub.located",
                        lambda *a, **k: Path("/models/kestrel-8B-UD-Q4_K_XL.gguf"))
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
    monkeypatch.setattr("ml_stack.serve.profile.profile_for", lambda m, **_: profile)
    monkeypatch.setattr("ml_stack.hub.located",
                        lambda *a, **k: Path("/models/kestrel-8B-UD-Q4_K_XL.gguf"))
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


@contextlib.contextmanager
def _lease(seen, model, manager=None, **lease):
    seen["lease"] = lease
    yield _Served()


class _Served:
    base_url = "http://127.0.0.1:8899"


def _leases(seen):
    """A stand-in for `serve` that records the lease and hands back a served address."""
    return functools.partial(_lease, seen)


class TestOfferingADraftHead:
    """A model with no measured record is offered the draft heads on this machine."""

    def _head(self, name="mtp-quince-2b-Q4_K_M.gguf", size=100, build=""):
        from ml_stack.hub import Head

        return Head(path=f"/models/{name}", bytes=size, spec_type="draft-mtp", build=build)

    def test_the_listing_says_what_each_model_would_draft_with(self):
        from ml_stack.claude import head_line

        said = head_line({"kind": "model", "name": "quince-2b.gguf", "record": None,
                          "heads": [self._head(), self._head("mtp-quince-2b-Q8_0.gguf", 200)]})
        assert "mtp-quince-2b-Q4_K_M.gguf" in said and "100B of memory" in said
        assert "or 1 more" in said

    def test_the_listing_says_when_there_is_no_head(self):
        from ml_stack.claude import head_line

        said = head_line({"kind": "model", "name": "brack-9b.gguf", "record": None,
                          "heads": []})
        assert said == "no draft head on this machine: it runs without speculative decoding"

    def test_a_measured_record_keeps_its_own_head(self):
        from ml_stack.serve.profile import record

        kept = record("quince-2b.gguf", draft="mtp-quince-2b-Q8_0.gguf", spec_type="draft-mtp")
        said = claude.head_line({"kind": "model", "name": "quince-2b.gguf", "record": kept,
                          "heads": [self._head()]})
        assert said == "drafts ahead with mtp-quince-2b-Q8_0.gguf, from its measured record"

    def test_a_record_with_no_head_names_the_one_lying_on_disk(self):
        from ml_stack.serve.profile import record

        said = claude.head_line({"kind": "model", "name": "quince-2b.gguf",
                          "record": record("quince-2b.gguf"), "heads": [self._head()]})
        assert "without a draft head" in said and "mtp-quince-2b-Q4_K_M.gguf" in said

    def test_nothing_typed_takes_the_cheapest_head(self):
        from ml_stack.claude import pick_head

        heads = [self._head(), self._head("mtp-quince-2b-Q8_0.gguf", 200)]
        assert pick_head(heads, say=lambda _: None, ask=lambda _: "").bytes == 100

    def test_zero_serves_without_a_head(self):
        from ml_stack.claude import pick_head

        assert pick_head([self._head()], say=lambda _: None, ask=lambda _: "0") is None

    def test_no_head_at_all_says_so_once(self):
        from ml_stack.claude import pick_head

        said: list[str] = []
        assert pick_head([], say=said.append, ask=lambda _: "1") is None
        assert said == ["serving without a draft head: nothing guesses tokens ahead for "
                        "the model to check in one pass, which is slower"]

    def test_the_build_a_borrowing_head_needs_is_said_when_it_is_offered(self):
        from ml_stack.claude import pick_head

        said: list[str] = []
        pick_head([self._head(build="mended")], say=said.append, ask=lambda _: "1")
        assert any("needs --build mended" in line for line in said)


class TestServingWithTheHead:
    """What a bare model is served with, and what the launcher says it is serving with."""

    def _serving(self, monkeypatch, tmp_path, argv, heads):
        seen: dict = {}
        monkeypatch.setattr("ml_stack.serve.manager.serve", _leases(seen))
        monkeypatch.setattr("ml_stack.serve.profile.profile_for", lambda m, **_: None)
        monkeypatch.setattr("ml_stack.hub.located",
                            lambda name, **kw: Path("/models/quince-2b-Q4_K_M.gguf"))
        monkeypatch.setattr("ml_stack.hub.heads_for", lambda *a, **k: heads)
        monkeypatch.setattr(claude, "alias_of", lambda url, model: "quince-2b")
        binary = tmp_path / "claude"
        binary.write_text("#!/bin/sh\nexit 0\n")
        binary.chmod(0o755)
        said: list[str] = []
        claude.launch([*argv, "--claude", str(binary)], say=said.append,
                      run_claude=lambda cmd, env: 0)
        return seen, said

    def test_the_cheapest_head_is_served_and_named_on_the_launch_line(self, monkeypatch,
                                                                     tmp_path):
        from ml_stack.hub import Head

        heads = [Head(path="/models/mtp-quince-2b-Q4_K_M.gguf", bytes=100,
                      spec_type="draft-mtp"),
                 Head(path="/models/mtp-quince-2b-Q8_0.gguf", bytes=200,
                      spec_type="draft-mtp")]
        seen, said = self._serving(monkeypatch, tmp_path, ["quince-2b"], heads)
        assert seen["lease"]["draft"] == "/models/mtp-quince-2b-Q4_K_M.gguf"
        assert seen["lease"]["spec_type"] == "draft-mtp"
        assert seen["lease"]["spec_draft_max"] == 4
        line = next(one for one in said if one.startswith("drafting ahead"))
        assert "mtp-quince-2b-Q4_K_M.gguf" in line and "draft-mtp" in line
        assert "4 tokens ahead" in line

    def test_no_head_on_this_machine_is_said_at_the_launch(self, monkeypatch, tmp_path):
        seen, said = self._serving(monkeypatch, tmp_path, ["quince-2b"], [])
        assert "draft" not in seen["lease"]
        assert any("without a draft head" in one for one in said)

    def test_asking_for_none_serves_without_one(self, monkeypatch, tmp_path):
        from ml_stack.hub import Head

        heads = [Head(path="/models/mtp-quince-2b-Q4_K_M.gguf", bytes=100,
                      spec_type="draft-mtp")]
        seen, _said = self._serving(monkeypatch, tmp_path,
                                    ["quince-2b", "--draft", "none"], heads)
        assert "draft" not in seen["lease"]

    def test_naming_a_head_that_is_not_there_refuses_before_the_load(self, monkeypatch,
                                                                    tmp_path):
        seen, said = self._serving(monkeypatch, tmp_path,
                                   ["quince-2b", "--draft", "mtp-brack.gguf"], [])
        assert "lease" not in seen, "nothing is served"
        assert any("no draft head called" in one for one in said)
