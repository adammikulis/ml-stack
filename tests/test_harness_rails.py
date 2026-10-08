"""The rails around Claude Code and Codex on a local model: the role policy, the hook, the session files."""

import base64
import json
import os
import shlex
import subprocess
import sys
import threading
import time
import tomllib
from pathlib import Path

import pytest
from workspace_kit import Kit, clean_env

from ml_stack import claude, codex, coding, harnesshook, harnessid, harnessing, requests
from ml_stack.harnesspolicy import decide
from ml_stack.workspace import tokens
from ml_stack.workspace.project import describe

SRC = str(Path(__file__).resolve().parent.parent / "src")


def _hook(event, payload, *args, env=None):
    """Run the hook as a harness does: a child process, the event on stdin."""
    base = {**os.environ, "PYTHONPATH": SRC, **(env or {})}
    return subprocess.run([sys.executable, "-m", "ml_stack.harnesshook", event, *args],
                          input=payload if isinstance(payload, str) else json.dumps(payload),
                          capture_output=True, text=True, env=base, timeout=60, check=False)


def _verdict(done):
    return json.loads(done.stdout)["hookSpecificOutput"]


def _command_text(command):
    return base64.b64decode(command.rsplit(" ", 1)[1]).decode("utf-16-le") if os.name == "nt" else command


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

    def test_a_call_that_asks_waits_for_the_person_and_runs_only_on_their_yes(self, tmp_path, monkeypatch):
        kit = Kit(clean_env(monkeypatch, tmp_path))
        tokens.store(kit.base, 'local-test', kit.agent('local-test'))
        kit.ws.registry.set_project(kit.ws.auth(kit.owner), 'local-test', describe(str(tmp_path)))
        for choice, want in (("allow-once", "allow"), ("deny", "deny")):
            inbox = requests.Inbox(memory=True)
            payload = {"tool_name": "Bash", "tool_input": {"command": f"rm -rf {tmp_path}/x"}, "cwd": str(tmp_path),
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
            out = harnesshook.pre(payload, self._rail(wait_s=10.0, roots=(str(tmp_path),)), inbox)
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
        assert done.returncode == 2
        assert _verdict(done)["permissionDecision"] == "deny"
        assert "diagnostic=" in done.stdout

    def test_the_post_hook_warns_without_blocking_when_nudge_is_missing(self, monkeypatch):
        monkeypatch.setenv("PATH", "/nonexistent")
        answer = harnesshook.post("local-test")
        assert "workspace nudge unavailable" in answer["hookSpecificOutput"]["additionalContext"]


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
            assert not os.access(path, os.W_OK)
            if os.name == "nt":
                assert tokens.problem(files.path) == ""
                with pytest.raises(PermissionError):
                    path.write_text("replace protected settings")
            else:
                assert not os.access(files.path, os.W_OK)
            text = path.read_text()
            for secret in ("ANTHROPIC", "sk-", "TOKEN", "KEY", "password"):
                assert secret not in text
            hooks = json.loads(text)["hooks"]
            command = hooks["PreToolUse"][0]["hooks"][0]["command"]
            command = _command_text(command)
            if os.name == "nt":
                assert "'--role' 'approve-first'" in command
            else:
                assert "--role approve-first" in command
            assert "ml_stack.harnesshook" in command
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
        if os.name == "nt":
            script = _command_text(cmd)
            assert script.startswith("& '" + sys.executable + "'")
            assert "'x y'" in script and "'/p q'" in script
        else:
            args = shlex.split(cmd)
            assert args[0] == sys.executable and "x y" in args and "/p q" in args

    def test_generated_hook_executes_with_quoted_project_paths(self, tmp_path):
        odd = tmp_path / "a dir & user's project"
        odd.mkdir()
        cmd = harnessing.hook_command("pre", role="read-only", label="x y", root=odd, protect=[])
        payload = {"tool_name": "Read", "tool_input": {"file_path": str(odd / "source.py")}, "cwd": str(odd)}
        done = subprocess.run(cmd if os.name == "nt" else shlex.split(cmd), input=json.dumps(payload),
                              capture_output=True, text=True, timeout=60, check=False,
                              env={**os.environ, "PYTHONPATH": SRC})
        assert done.returncode == 0, done.stderr
        assert _verdict(done)["permissionDecision"] == "allow"


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
        assert got["features"]["code_mode"]["direct_only_tool_namespaces"] == ["mcp__workspace"]
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
        monkeypatch.setattr(codex.tokens, "load", lambda base, agent: "assigned-test-seat")
        monkeypatch.setattr(harnessid, "invite", lambda name, project, parent, say, **kwargs: harnessid.Seat(name, parent, base=tmp_path / "workspace"))
        monkeypatch.setattr(harnessid, "announce", lambda *a, **k: True)
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
        assert seen["model"] == harnessing.DEFAULT_MODEL and seen["want"].ctx == 0 and seen["want"].slots == 1
        assert seen["env"]["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] == "262144"
        assert seen["env"]["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] == "262144"
        command = _command_text(seen["settings"]["hooks"]["PreToolUse"][0]["hooks"][0]["command"])
        assert ("'--role' 'plan-and-go'" if os.name == "nt" else "--role plan-and-go") in command
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
        scope = seen["config"]["mcp_servers"]["workspace"]
        assert scope["required"] is True
        assert scope["command"] == sys.executable
        assert scope["args"][-1] == "--workspace-only"
        assert scope["env"]["ML_STACK_WORKSPACE_TOKEN"] == "assigned-test-seat"
        assert "--dangerously-bypass-hook-trust" in seen["command"] and seen["command"][-2:] == ["exec", "fix it"]
        assert tmp_path / "tree" not in seen["home"].parents and not seen["home"].exists()

    def test_an_unknown_role_is_refused_before_anything_is_served(self, monkeypatch, tmp_path):
        monkeypatch.setattr(harnessing, "serving", lambda *a, **k: pytest.fail("served"))
        binary = tmp_path / "claude"
        binary.write_text("#!/bin/sh\n")
        assert claude.launch(["--claude", str(binary), "--role", "root"], say=lambda _: None) == 2


class TestWorkspaceCommands:
    def _fake_workspace(self, tmp_path, monkeypatch):
        exe = tmp_path / "bin" / "ml-stack-workspace"
        exe.parent.mkdir()
        script = ('import sys\nfrom pathlib import Path\n'
                  'Path(sys.argv[0] + ".argv").write_text("\\n".join(sys.argv[1:]))\n'
                  'print("nudge text")\n')
        if sys.platform == "win32":
            from distlib.scripts import ScriptMaker

            maker = ScriptMaker(None, str(exe.parent))
            maker.variants = {""}
            maker.executable = sys.executable
            maker.script_template = script
            exe = Path(maker.make("ml-stack-workspace = unused:main")[0])
        else:
            exe.write_text(f"#!{sys.executable}\n{script}")
            exe.chmod(0o755)
        monkeypatch.setenv("PATH", f"{exe.parent}{os.pathsep}{os.environ['PATH']}")
        return exe

    def test_a_hostile_label_is_one_argument_and_never_a_shell_line(self, tmp_path, monkeypatch):
        exe = self._fake_workspace(tmp_path, monkeypatch)
        hostile = "x; touch /tmp/pwned $(id) `id`"
        assert harnessid.announce(harnessid.Seat(hostile, "claude"), "t", lambda _: None)
        argv = Path(f"{exe}.argv").read_text().splitlines()
        assert argv == ["announce", "joined", "t", "--agent", "claude", "--label", hostile]
        seen = []
        def run(command, **kwargs):
            seen.append((command, kwargs))
            return subprocess.CompletedProcess(command, 0, 'nudge text', '')
        monkeypatch.setattr(harnesshook.subprocess, 'run', run)
        assert harnesshook.nudge(hostile) == "nudge text"
        command, options = seen[0]
        assert command == [sys.executable, '-m', 'ml_stack.workspace.notification_reader',
                           hostile, str(Path.cwd()), '']
        assert options.get('shell', False) is False
        assert options['timeout'] == harnesshook.NUDGE_S

    def test_a_failed_announcement_identifies_the_workspace_trust_check(self, monkeypatch):
        said = []
        monkeypatch.setenv("PATH", "/nonexistent")
        assert harnessid.announce(harnessid.Seat("l", "claude"), "t", said.append) is False
        assert "selected authority and device trust" in said[0]


class TestSeat:
    @pytest.fixture(autouse=True)
    def _own_workspace(self, monkeypatch, tmp_path):
        clean_env(monkeypatch, tmp_path)
        monkeypatch.setattr(harnessid.project_connection, "auto_attach", lambda *a: None, raising=False)

    @pytest.fixture
    def person(self, monkeypatch):
        from ml_stack.sentinel import human

        real = human.require_person
        monkeypatch.setattr(human, "require_person",
                            lambda action, terminal=None, env=None: real(action, (True, True), env))

    def test_the_name_is_local_model_harness_cleaned_to_what_an_id_allows(self):
        assert harnessid.agent_name("Qwen3.8-27B-UD-Q4_K_XL", "codex") == "local-qwen3.8-27b-ud-q4_k_xl-codex"
        assert harnessid.agent_name("x" * 80, "codex") == harnessid.agent_name("x" * 80, "codex")
        assert len(harnessid.agent_name("x" * 80, "codex")) <= 48
        assert harnessid.agent_name("m", "codex", "mine") == "mine"

    def test_a_launcher_connects_a_persistent_agent_without_printing_a_token(self, person, tmp_path):
        from ml_stack.workspace import Workspace, tokens

        said = []
        project = tmp_path / "proj"
        project.mkdir()
        seat = harnessid.invite("local-test-codex", project, "claude", said.append)
        ws = Workspace()
        assert seat.persistent and not seat.minted and ws.registry.role_of("local-test-codex") == "agent"
        token_file = tokens.directory(ws.base) / "local-test-codex"
        assert tokens.problem(token_file) == ""
        secret = token_file.read_text().strip()
        assert secret not in "".join(said) and seat.flags() == ["--agent", "local-test-codex"]
        assert seat.revoke() is False and token_file.exists()
        assert ws.auth(secret).id == "local-test-codex"
        again = harnessid.invite("local-test-codex", project, "claude", said.append)
        assert again.persistent and tokens.load(ws.base, again.name) == secret

    def test_the_seat_is_on_the_project_board_with_the_quiet_defaults(self, person, tmp_path):
        from ml_stack.workspace import Workspace

        project = tmp_path / "proj"
        project.mkdir()
        seat = harnessid.invite("local-test-codex", project, "claude", lambda _: None)
        boards, subs = Workspace(seat.base).board.store.state()
        assert any("local-test-codex" in b["members"] for k, b in boards.items() if k != "#general")
        assert not subs.get("local-test-codex")

    def test_a_launcher_records_its_model_as_claimed(self, person, monkeypatch, tmp_path):
        from ml_stack.workspace import Workspace

        seat = harnessid.invite("local-test-codex", tmp_path, "claude", lambda _: None)
        assert seat.record_model("qwen-27b", "codex") is True
        assert "qwen-27b" in json.dumps(Workspace(seat.base).registry.info("local-test-codex"))
        assert Workspace(seat.base).registry.info(seat.name)["model_state"] == "claimed"
        monkeypatch.setenv("CLAUDECODE", "1")
        assert seat.record_model("other", "codex") is True
        assert harnessid.Seat("x", "p").record_model("m", "codex") is False

    def test_ending_the_session_keeps_identity_and_removes_harness_files(self, person, monkeypatch, tmp_path):
        from ml_stack.workspace import Workspace, tokens

        seen = {}
        monkeypatch.setattr(harnessing, "serving", _fake_serving(seen))
        monkeypatch.setattr(codex, "alias_of", lambda url, model: "qwen")
        monkeypatch.setattr(harnessid, "announce", lambda *a, **k: True)
        binary = tmp_path / "codex"
        binary.write_text("#!/bin/sh\n")
        (tmp_path / "proj").mkdir()

        def run(command, env):
            ws = Workspace()
            seen["token"] = tokens.load(ws.base, "local-qwen-codex")
            assert ws.auth(seen["token"]).id == "local-qwen-codex"
            seen["home"] = Path(env["CODEX_HOME"])
            return 0

        assert codex.launch(["--codex", str(binary), "--project", str(tmp_path / "proj")], say=lambda _: None,
                            run_codex=run) == 0
        assert Workspace().auth(seen["token"]).id == "local-qwen-codex"
        assert (tokens.directory(Workspace().base) / "local-qwen-codex").exists()
        assert not seen["home"].exists()

    def test_an_agent_launcher_initializes_itself_without_a_parent_credential(self, monkeypatch, tmp_path):
        monkeypatch.setenv("CLAUDECODE", "1")
        seat = harnessid.invite("local-test-codex", tmp_path, "claude", lambda _: None)
        assert seat.name == "local-test-codex" and seat.persistent

    @pytest.mark.parametrize("authority", ["selected", "discovered", "coordinator"])
    def test_remote_launcher_uses_selected_authority_without_local_bootstrap(self, monkeypatch, tmp_path, authority):
        calls = []
        class Remote:
            base = tmp_path
            def token(self, **kwargs):
                return "saved-session"
            def call(self, operation, token, *args):
                calls.append((operation, args))
                return {"id": "device-worker"}
            def ensure(self, base, name, **kwargs):
                calls.append(("ensure", kwargs))
                return "device-worker"
        remote = Remote()
        canonical = authority != "coordinator"
        connection = {"host": "https://192.0.2.1", "project_id": "a" * 32}
        selected = connection if authority == "selected" else None
        monkeypatch.setattr(harnessid.project_connection, "selected", lambda *a: selected)
        monkeypatch.setattr(harnessid.project_connection, "auto_attach",
                            lambda *a: connection if authority == "discovered" else None, raising=False)
        monkeypatch.setattr(harnessid.project_connection, "RemoteWorkspace", lambda *a, **k: remote)
        monkeypatch.setattr(harnessid.coordinator_client, "client", lambda *a: remote)
        monkeypatch.setattr(harnessid, "authoritative", lambda *a: {"key": "a" * 32, "name": "project"})
        monkeypatch.setattr(harnessid.limits, "root", lambda: tmp_path)
        monkeypatch.setattr(harnessid, "Workspace", lambda *a: pytest.fail("local fallback"))
        seat = harnessid.invite("worker", tmp_path, "", lambda _: None)
        assert seat.name == "device-worker" and seat.persistent
        assert seat.record_model("test-model", "test-harness")
        assert calls[-1][0] == ("claim_model" if canonical else "ensure")
        assert not seat.revoke()

    def test_a_fake_endpoint_cannot_verify_a_delegated_agents_model(self, monkeypatch, tmp_path):
        from ml_stack.testing import FakeLlamaServer, Served
        from ml_stack.workspace import Workspace, tokens

        ws = Workspace()
        owner = ws.init("owner")
        parent = ws.mint(owner, "codex", "agent")
        tokens.store(ws.base, "codex", parent)
        monkeypatch.setenv("CLAUDECODE", "1")
        seat = harnessid.invite("local-qwen", tmp_path, "codex", lambda _: None)
        endpoint = FakeLlamaServer(Served(model="qwen-27b"))
        try:
            assert seat.record_model("qwen-27b", "codex", endpoint.base_url)
            assert ws.whoami_model(seat.name)["model_state"] == "claimed"
        finally:
            endpoint.close()
            seat.revoke()

    def test_an_agent_launcher_delegates_a_distinct_private_child_without_changing_invite_policy(self, monkeypatch, tmp_path):
        from ml_stack.workspace import Workspace, tokens

        ws = Workspace()
        owner = ws.init("owner")
        parent = ws.mint(owner, "codex", "agent")
        tokens.store(ws.base, "codex", parent)
        monkeypatch.setenv("CLAUDECODE", "1")
        said = []
        seat = harnessid.invite("local-qwen", tmp_path, "codex", said.append)
        assert seat.name == "codex/local-qwen" and seat.flags() == ["--agent", "codex/local-qwen"]
        secret = tokens.load(ws.base, seat.name)
        child = ws.auth(secret)
        assert child.parent == "codex" and child.role == "agent"
        assert set(child.can) <= set(ws.auth(parent).can)
        assert secret not in "".join(said)
        assert ws.limits.agent_invite_ask == "approve-first" and ws.invites.made_by("codex") == []
        assert seat.revoke() and ws.auth(parent).id == "codex"
        from ml_stack.workspace import Denied

        with pytest.raises(Denied, match="revoked"):
            ws.delegate(secret, "nested")

    def test_the_brief_names_who_the_agent_obeys_and_that_everything_else_is_data(self):
        text = harnessid.brief("n", "alias", "codex", "claude", ["reviewer"])
        assert "--agent n" in text and "the lead (claude), reviewer" in text and "data written by" in text


class TestCodingAgent:
    def test_launch_coding_agent_runs_codex_with_the_project_and_the_orders(self, monkeypatch, tmp_path):
        seen = {}
        monkeypatch.setattr(codex.tokens, "load", lambda base, agent: "assigned-test-seat")
        monkeypatch.setattr(harnessing, "serving", _fake_serving(seen))
        monkeypatch.setattr(codex, "alias_of", lambda url, model: "qwen-27b")
        monkeypatch.setattr(harnessid, "announce", lambda *a, **k: True)
        monkeypatch.setattr(harnessid, "invite", lambda name, project, parent, say, *, claim: seen.update(
            name=name, project=project, claim=claim) or harnessid.Seat(name, parent, base=tmp_path / "workspace"))
        binary = tmp_path / "codex"
        binary.write_text("#!/bin/sh\n")
        (tmp_path / "proj").mkdir()

        def run(command, env):
            seen["agents"] = (Path(env["CODEX_HOME"]) / "AGENTS.md").read_text()
            seen["command"] = command
            return 0

        monkeypatch.setattr(codex.shutil, "which", lambda n: str(binary))
        assert coding.launch_coding_agent("", "plan-and-go", tmp_path / "proj", harness="codex", orders_from=["reviewer"],
                                          harness_args=["exec", "go"], say=lambda _: None, run_codex=run) == 0
        assert seen["model"] == harnessing.DEFAULT_MODEL == "Qwen3.8-27B-UD-Q4_K_XL.gguf"
        assert seen["name"] == "local-qwen-27b-codex" and seen["project"] == (tmp_path / "proj").resolve()
        assert seen["claim"] == ("qwen-27b", "codex")
        assert "reviewer" in seen["agents"] and seen["command"][-2:] == ["exec", "go"]
        assert "workspace-write" in seen["command"]

    def test_pi_is_the_default_coding_harness(self, monkeypatch, tmp_path):
        seen = {}
        monkeypatch.setitem(coding.HARNESSES, "pi", lambda argv, **options: seen.update(argv=argv, options=options) or 0)
        assert coding.launch_coding_agent("local-model.gguf", "plan-and-go", tmp_path,
                                          say=lambda _: None) == 0
        assert seen["argv"] == ["local-model.gguf", "--role", "plan-and-go", "--project", str(tmp_path)]

    def test_an_unknown_harness_is_refused(self):
        assert coding.launch_coding_agent("", "read-only", ".", "bash", say=lambda _: None) == 2


def test_explicit_head_and_none_override_measured_profile(monkeypatch):
    from types import SimpleNamespace

    from ml_stack import hub
    from ml_stack.serve.serving import Config, Serving

    measured = Config(serving=Serving(model="qwen.gguf", draft="old-head.gguf", spec_type="draft-mtp"))
    monkeypatch.setattr(harnessing.profile, "profile_for", lambda _: SimpleNamespace(config=lambda **_: measured))
    monkeypatch.setattr(harnessing.profile, "said", lambda _: "measured profile")
    monkeypatch.setattr(harnessing.chat_template, "trained_context", lambda _: 262144)
    monkeypatch.setattr(hub, "head_choice", lambda model, asked: None if asked == "none" else SimpleNamespace(
        serving=lambda: "requested MTP head", over=lambda: {"draft": asked, "spec_type": "draft-mtp"}))
    selected = harnessing.config_for("qwen.gguf", harnessing.Want(ctx=262144, draft="matching-head.gguf"), lambda _: None)
    assert selected.serving.draft == "matching-head.gguf"
    assert selected.serving.slot_context == 262144
    disabled = harnessing.config_for("qwen.gguf", harnessing.Want(ctx=262144, draft="none"), lambda _: None)
    assert disabled.serving.draft == "" and disabled.serving.mtp is False
    automatic = harnessing.config_for("qwen.gguf", harnessing.Want(ctx=262144, draft="auto"), lambda _: None)
    assert automatic.serving.draft == "old-head.gguf"


def test_automatic_context_comes_from_the_device_model_fit(monkeypatch):
    from types import SimpleNamespace

    from ml_stack import hub
    monkeypatch.setattr(hub, "located", lambda model, loose=True: Path("qwen.gguf"))
    monkeypatch.setattr(harnessing.chat_template, "trained_context", lambda _: 200000)
    monkeypatch.setattr(harnessing.suggest, "suggest", lambda *a, **k: SimpleNamespace(context=98304, verdict="yellow", n_gpu_layers="auto",
                            kv_cache_type="q8_0", flash_attn=True, batch=512))
    monkeypatch.setattr(harnessing.profile, "profile_for", lambda _: None)
    monkeypatch.setattr(hub, "head_choice", lambda *_: None)
    said = []
    config = harnessing.config_for("qwen.gguf", harnessing.Want(ctx=0), said.append)
    assert config.serving.slot_context == 98304
    assert any("automatically selected 98,304" in line for line in said)


@pytest.mark.parametrize("line, expected", [
    ("ml-stack-workspace inbox --agent own", "allow"),
    ("ml-stack-workspace send codex note 'sensor count 8' --agent own", "allow"),
    ("ml-stack-workspace inbox --agent other", "deny"),
    ("env ML_STACK_WORKSPACE_AGENT=other ml-stack-workspace inbox --agent other", "deny"),
    ("command ml-stack-workspace inbox --agent other", "deny"),
    ("ml-stack-workspace inbox --agent own --agent other", "deny"),
    ("ml-stack-workspace inbox --agent own --token-file /tmp/other", "deny"),
    ("ml-stack-workspace inbox", "deny"),
    ("ml-stack-workspace setup --agents own --agent own", "deny"),
    ("ml-stack-workspace inbox --agent own && rm README.md", "deny"),
])
def test_workspace_hook_binds_own_identity_and_keeps_human_commands_blocked(line, expected, tmp_path):
    payload = {"tool_name": "Bash", "tool_input": {"command": line}, "cwd": str(tmp_path)}
    rail = harnesshook.Rail("plan-and-go", "own", roots=(str(tmp_path),), wait_s=0)
    answer = harnesshook.pre(payload, rail)
    assert answer["hookSpecificOutput"]["permissionDecision"] == expected


@pytest.mark.parametrize("name,role,action", [
    ("mcp__workspace__workspace_inbox", "read-only", "allow"),
    ("mcp__workspace__workspace_reputation", "read-only", "allow"),
    ("mcp__workspace__workspace_send", "plan-and-go", "allow"),
    ("mcp__workspace__workspace_send", "approve-first", "ask"),
    ("mcp__workspace__workspace_send", "read-only", "deny"),
    ("mcp__foreign__workspace_send", "plan-and-go", "ask"),
])
def test_bound_workspace_mcp_obeys_role(name, role, action):
    assert decide(role, name, {"to": "codex", "kind": "status", "text": "ready"}).action == action
