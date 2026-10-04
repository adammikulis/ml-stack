"""The rails around Claude Code and Codex on a local model: the role policy, the hook, the session files."""

import json
import os
import subprocess
import sys
import threading
import time
import tomllib
from pathlib import Path

import pytest

from ml_stack import claude, codex, harnesshook, harnessing, requests
from ml_stack.harnesspolicy import decide

SRC = str(Path(__file__).resolve().parent.parent / "src")


def _hook(event, payload, *args, env=None):
    """Run the hook as a harness does: a child process, the event on stdin."""
    base = {**os.environ, "PYTHONPATH": SRC, **(env or {})}
    return subprocess.run([sys.executable, "-m", "ml_stack.harnesshook", event, *args],
                          input=payload if isinstance(payload, str) else json.dumps(payload),
                          capture_output=True, text=True, env=base, timeout=60, check=False)


def _verdict(done):
    return json.loads(done.stdout)["hookSpecificOutput"]


class TestPolicy:
    def test_a_read_runs_in_every_role(self):
        for role in ("read-only", "approve-first", "plan-and-go"):
            assert decide(role, "Read", {"file_path": "/w/a.py"}, roots=("/w",)).action == "allow"
            assert decide(role, "Bash", {"command": "git status"}, roots=("/w",)).action == "allow"

    def test_a_destructive_call_asks_where_the_role_acts_and_is_denied_where_it_does_not(self):
        call = ("Bash", {"command": "rm -rf /tmp/somewhere"})
        assert decide("approve-first", *call, roots=("/w",)).action == "ask"
        assert decide("plan-and-go", *call, roots=("/w",)).kind == "tool_call_destructive"
        assert decide("read-only", *call, roots=("/w",)).action == "deny"

    def test_a_reversible_call_asks_in_approve_first_and_runs_in_plan_and_go(self):
        edit = ("Edit", {"file_path": "/w/a.py", "old_string": "a", "new_string": "b"})
        assert decide("approve-first", *edit, roots=("/w",)).action == "ask"
        assert decide("plan-and-go", *edit, roots=("/w",)).action == "allow"
        assert decide("read-only", *edit, roots=("/w",)).action == "deny"

    def test_a_tool_nobody_classified_asks_as_destructive(self):
        got = decide("plan-and-go", "SomeNewTool", {"x": "y"}, roots=("/w",))
        assert (got.action, got.kind) == ("ask", "tool_call_destructive")
        assert decide("plan-and-go", "Bash", None, roots=("/w",)).action == "ask"

    def test_text_naming_the_launchers_files_or_a_person_only_action_is_denied_in_every_role(self):
        for role in ("approve-first", "plan-and-go"):
            hit = decide(role, "Edit", {"file_path": "/state/harness/ab/settings.json", "old_string": "a",
                                        "new_string": "b"}, roots=("/state",), protected=("/state/harness/ab",))
            assert hit.action == "deny"
            answer = decide(role, "Bash", {"command": "ml-stack-requests answer rq_1 allow-once"})
            assert answer.action == "deny" and "ml-stack-requests answer" in answer.reason

    def test_nothing_in_the_call_or_the_environment_changes_the_role(self, monkeypatch):
        for name in ("ML_STACK_ROLE", "ML_STACK_GUARD", "MLSTACK_GUARD", "CLAUDE_ROLE"):
            monkeypatch.setenv(name, "off")
        call = {"command": "rm -rf /tmp/x", "role": "plan-and-go", "ml_stack_role": "plan-and-go",
                "permission_mode": "bypassPermissions", "dangerouslyDisableSandbox": True}
        assert decide("read-only", "Bash", call, roots=("/w",)).action == "deny"
        assert decide("approve-first", "Bash", call, roots=("/w",)).action == "ask"


class TestHook:
    def _rail(self, **over):
        return harnesshook.Rail("approve-first", "local-test", ("/w",), (), 5.0) if not over \
            else harnesshook.Rail(**{"role": "approve-first", "label": "local-test", "roots": ("/w",), **over})

    def test_a_safe_call_is_allowed_and_nothing_is_raised(self):
        inbox = requests.Inbox(memory=True)
        out = harnesshook.pre({"tool_name": "Read", "tool_input": {"file_path": "/w/a"}}, self._rail(), inbox)
        assert out["hookSpecificOutput"]["permissionDecision"] == "allow"
        assert inbox.list() == []

    def test_a_call_that_asks_waits_for_the_person_and_runs_only_on_their_yes(self):
        for choice, want in (("allow-once", "allow"), ("deny", "deny")):
            inbox = requests.Inbox(memory=True)
            payload = {"tool_name": "Bash", "tool_input": {"command": "rm -rf /tmp/x"}, "cwd": "/w",
                       "session_id": "s1"}

            def person(inbox=inbox, choice=choice):
                for _ in range(200):
                    waiting = requests.list_requests(inbox=inbox, state="pending")
                    if waiting:
                        requests.answer(waiting[0].id, choice, waiting[0].fingerprint, "ui",
                                        requests.Context(env={}, inbox=inbox))
                        return
                    time.sleep(0.02)

            thread = threading.Thread(target=person)
            thread.start()
            out = harnesshook.pre(payload, self._rail(wait_s=10.0), inbox)
            thread.join()
            assert out["hookSpecificOutput"]["permissionDecision"] == want
            raised = requests.list_requests(inbox=inbox)[0]
            assert raised.kind == "tool_call_destructive" and raised.raised_by.agent == "local-test"

    def test_an_unanswered_request_is_a_denial(self):
        inbox = requests.Inbox(memory=True)
        payload = {"tool_name": "Bash", "tool_input": {"command": "rm -rf /tmp/x"}}
        out = harnesshook.pre(payload, self._rail(wait_s=0.2), inbox)
        assert out["hookSpecificOutput"]["permissionDecision"] == "deny"

    def test_as_a_child_process_a_denied_call_is_denied_and_a_safe_one_allowed(self):
        deny = _verdict(_hook("pre", {"tool_name": "Bash", "tool_input": {"command": "rm -rf /tmp/x"}},
                              "--role", "read-only", "--root", "/w"))
        assert deny["permissionDecision"] == "deny" and deny["hookEventName"] == "PreToolUse"
        allow = _verdict(_hook("pre", {"tool_name": "Read", "tool_input": {"file_path": "/w/a"}},
                               "--role", "approve-first", "--root", "/w"))
        assert allow["permissionDecision"] == "allow"

    def test_a_call_that_asks_with_no_keystore_is_denied_at_once(self):
        began = time.time()
        done = _hook("pre", {"tool_name": "Bash", "tool_input": {"command": "rm -rf /tmp/x"}},
                     "--role", "approve-first", "--root", "/w", "--wait", "120",
                     env={"ML_STACK_HOME": str(Path(os.environ["ML_STACK_HOME"]) / "hook")})
        assert _verdict(done)["permissionDecision"] == "deny"
        assert time.time() - began < 60, "the hook does not wait on a keystore it cannot open"

    def test_no_environment_variable_or_argument_lifts_the_policy(self):
        done = _hook("pre", {"tool_name": "Bash", "tool_input": {"command": "rm -rf /tmp/x",
                                                                 "role": "plan-and-go"}},
                     "--role", "read-only", "--root", "/w",
                     env={"ML_STACK_ROLE": "plan-and-go", "MLSTACK_GUARD": "off", "ML_STACK_GUARD": "off"})
        assert _verdict(done)["permissionDecision"] == "deny"

    def test_a_hook_that_fails_blocks_the_call(self):
        done = _hook("pre", "not json", "--role", "plan-and-go")
        assert done.returncode == 2 and done.stdout == ""

    def test_the_post_hook_degrades_to_nothing_without_a_nudge(self, monkeypatch):
        monkeypatch.setenv("PATH", "/nonexistent")
        assert harnesshook.post("local-test") == {}


class TestSessionFiles:
    def test_the_settings_are_outside_the_tree_read_only_and_hold_no_secret(self, tmp_path):
        tree = tmp_path / "tree"
        tree.mkdir()
        files = harnessing.session_files(tree)
        try:
            pre = harnessing.hook_command("pre", role="approve-first", label="l", root=tree,
                                          protect=harnessing.protected_paths(files))
            path = files.write("settings.json", claude.settings(pre, "post-cmd", harnessing.WAIT_S))
            files.lock()
            assert tree not in path.parents and files.path not in (tree, *tree.parents)
            assert not os.access(path, os.W_OK) and not os.access(files.path, os.W_OK)
            text = path.read_text()
            for secret in ("ANTHROPIC", "sk-", "TOKEN", "KEY", "password"):
                assert secret not in text
            hooks = json.loads(text)["hooks"]
            assert "ml_stack.harnesshook" in hooks["PreToolUse"][0]["hooks"][0]["command"]
            assert "--role approve-first" in hooks["PreToolUse"][0]["hooks"][0]["command"]
            assert "disableAllHooks" not in text
        finally:
            files.release()
        assert not files.path.exists()

    def test_a_working_directory_that_holds_the_session_is_refused(self):
        from ml_stack import home

        with pytest.raises(ValueError, match="overlap"):
            harnessing.session_files(home.state("harness"))

    def test_the_hook_command_names_this_interpreter_and_quotes_its_paths(self, tmp_path):
        odd = tmp_path / "a dir"
        cmd = harnessing.hook_command("pre", role="plan-and-go", label="x y", root=odd, protect=["/p q"])
        assert cmd.startswith(sys.executable) and "'x y'" in cmd and "'/p q'" in cmd


class TestAdmission:
    def test_a_model_the_wiring_limit_cannot_hold_prints_the_command_and_is_not_raised(self):
        said = []

        class Plan:
            enough_now, needed_mb, current_mb, default_mb = False, 120000, 98000, 98000

        assert harnessing.admitted("/m/big.gguf", 262144, said.append, plan=lambda *a: Plan()) is False
        assert "ml-stack-serve memory --for big.gguf --ctx 262144 --kv q8_0 --apply" in said[0]
        assert harnessing.admitted("/m/big.gguf", 262144, said.append,
                                   plan=lambda *a: type("P", (), {"enough_now": True})()) is True


class TestCodex:
    def test_the_config_names_the_provider_the_window_and_the_hook(self):
        text = codex.config_toml("http://127.0.0.1:8080/", "qwen", 262144, ("PRE", "POST", 300.0))
        got = tomllib.loads(text)
        assert got["model_provider"] == "mlstack" and got["model"] == "qwen"
        provider = got["model_providers"]["mlstack"]
        assert provider["base_url"] == "http://127.0.0.1:8080/v1" and provider["wire_api"] == "responses"
        assert got["model_context_window"] == 262144
        assert got["model_auto_compact_token_limit"] == int(262144 * 0.9)
        assert got["features"]["hooks"] is True
        assert got["hooks"]["PreToolUse"][0]["hooks"][0]["command"] == "PRE"
        assert got["hooks"]["PostToolUse"][0]["hooks"][0]["command"] == "POST"
        assert text == codex.config_toml("http://127.0.0.1:8080/", "qwen", 262144, ("PRE", "POST", 300.0))

    def test_the_role_sets_codexs_own_sandbox_and_approval(self):
        assert codex.policy_flags("read-only") == ["--sandbox", "read-only", "--ask-for-approval", "never"]
        assert "workspace-write" in codex.policy_flags("approve-first")
        assert "danger-full-access" not in codex.policy_flags("plan-and-go")

    def test_the_environment_has_its_own_home_and_no_api_key(self, tmp_path):
        env = codex.environment(tmp_path, {"OPENAI_API_KEY": "sk-x", "CODEX_API_KEY": "k", "PATH": "/bin"})
        assert env["CODEX_HOME"] == str(tmp_path) and "OPENAI_API_KEY" not in env and "CODEX_API_KEY" not in env


class _Config:
    class serving:
        slot_context = 262144


def _fake_serving(seen):
    import contextlib

    @contextlib.contextmanager
    def serving(model, want, say, by):
        seen["model"], seen["want"] = model, want
        yield "http://127.0.0.1:8899", _Config(), "/models/Qwen3.8-27B-Q4_K_M.gguf"
        seen["released"] = True
    return serving


class TestLaunch:
    @pytest.fixture(autouse=True)
    def _quiet(self, monkeypatch, tmp_path):
        monkeypatch.setattr(harnessing, "join_workspace", lambda *a, **k: True)
        monkeypatch.setattr(claude, "alias_of", lambda url, model: "qwen-27b")
        monkeypatch.setattr(codex, "alias_of", lambda url, model: "qwen-27b")
        (tmp_path / "tree").mkdir()
        monkeypatch.chdir(tmp_path / "tree")

    def test_claude_gets_the_hooks_window_and_a_settings_file_that_goes_when_it_exits(self, monkeypatch, tmp_path):
        seen = {}
        monkeypatch.setattr(harnessing, "serving", _fake_serving(seen))
        binary = tmp_path / "claude"
        binary.write_text("#!/bin/sh\n")

        def run(command, env):
            seen["command"], seen["env"] = command, env
            seen["settings"] = json.loads(Path(command[2]).read_text())
            return 0

        assert claude.launch(["--claude", str(binary), "--role", "plan-and-go"], say=lambda _: None,
                             run_claude=run) == 0
        assert seen["model"] == "Qwen3.8-27B" and seen["want"].ctx == 262144 and seen["want"].slots == 1
        assert seen["env"]["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] == "262144"
        assert seen["env"]["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] == "262144"
        assert "--role plan-and-go" in seen["settings"]["hooks"]["PreToolUse"][0]["hooks"][0]["command"]
        assert not Path(seen["command"][2]).exists() and seen["released"]

    def test_codex_gets_its_own_home_with_the_provider_and_the_hook_flag(self, monkeypatch, tmp_path):
        seen = {}
        monkeypatch.setattr(harnessing, "serving", _fake_serving(seen))
        binary = tmp_path / "codex"
        binary.write_text("#!/bin/sh\n")

        def run(command, env):
            seen["command"] = command
            seen["config"] = tomllib.loads((Path(env["CODEX_HOME"]) / "config.toml").read_text())
            seen["home"] = Path(env["CODEX_HOME"])
            return 3

        assert codex.launch(["qwen", "--codex", str(binary), "--", "exec", "fix it"], say=lambda _: None,
                            run_codex=run) == 3
        assert seen["config"]["model_context_window"] == 262144
        assert "--dangerously-bypass-hook-trust" in seen["command"] and seen["command"][-2:] == ["exec", "fix it"]
        assert tmp_path / "tree" not in seen["home"].parents and not seen["home"].exists()

    def test_an_unknown_role_is_refused_before_anything_is_served(self, monkeypatch, tmp_path):
        monkeypatch.setattr(harnessing, "serving", lambda *a, **k: pytest.fail("served"))
        binary = tmp_path / "claude"
        binary.write_text("#!/bin/sh\n")
        assert claude.launch(["--claude", str(binary), "--role", "root"], say=lambda _: None) == 2
