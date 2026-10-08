"""Distinct and resumable Codex identities on shared project Boards."""

import json
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from ml_stack import home
from ml_stack.workspace import (
    automatic_connection,
    cli,
    project_connection,
    project_session,
    tokens,
)
from ml_stack.workspace.identity import Denied

PROJECT = "a" * 32


@pytest.fixture
def sessions(tmp_path, monkeypatch):
    monkeypatch.setenv("ML_STACK_HOME", str(tmp_path / "state"))
    monkeypatch.delenv("CODEX_SESSION_ID", raising=False)
    monkeypatch.delenv("CODEX_THREAD_ID", raising=False)
    root = tmp_path / "project"
    root.mkdir()
    enrolled = []
    revoked = set()
    class Remote:
        host = "https://board.invalid:8770"
        project_id = PROJECT
        cluster_key = ""
        def __init__(self, *args, **kwargs):
            pass
        def token(self, *, agent, **kwargs):
            self.call("whoami", agent)
            return agent
        def call(self, operation, token):
            if token in revoked:
                raise Denied("the agent was revoked")
            return {"id": token, "role": "agent", "project": {"key": PROJECT}, "can": []}
        def enroll(self, name, **kwargs):
            enrolled.append(name)
            return {"id": name}
    remote = Remote()
    project_connection.bind(remote, root, "legacy-agent", "development")
    choice = {"host": remote.host, "project_id": PROJECT,
              "cluster": "development", "root": str(root), "automatic": True, "agent": ""}
    local = SimpleNamespace(registry=SimpleNamespace(info=lambda actor: {"model": "fixture-model", "harness": "codex"}))
    monkeypatch.setattr(cli, "Workspace", lambda: local)
    monkeypatch.setattr(cli, "_local_token", lambda args: "local-agent-token")
    monkeypatch.setattr(automatic_connection.device_agent, "owned_project_session",
                        lambda *args: nullcontext(SimpleNamespace(id=args[2])))
    monkeypatch.setattr(automatic_connection, "discover", lambda root: choice)
    monkeypatch.setattr(automatic_connection, "RemoteWorkspace", Remote)
    monkeypatch.setattr(project_connection, "RemoteWorkspace", Remote)
    monkeypatch.setattr(project_connection, "CanonicalWorkspace", lambda remote, token: (remote, token))
    return root, choice, enrolled, revoked


def join(root, label="", agent="codex"):
    args = SimpleNamespace(agent=agent, token_file="", label=label)
    cli._context(args, project_connection.selected(root))
    return args.agent


def test_independent_threads_join_distinctly_and_resume_without_reenrollment(sessions, monkeypatch):
    root, _, enrolled, _ = sessions
    monkeypatch.setenv("CODEX_THREAD_ID", "fixture-thread-one")
    first = join(root)
    monkeypatch.setenv("CODEX_THREAD_ID", "fixture-thread-two")
    second = join(root)
    assert first != second and first.startswith("codex-") and second.startswith("codex-")
    monkeypatch.setenv("CODEX_THREAD_ID", "fixture-thread-one")
    assert join(root) == first
    assert join(root, "helper") == first
    assert enrolled == [first, second]
    saved = project_connection._saved()[str(root)]
    assert saved["agent"] == "legacy-agent" and len(saved["sessions"]) == 2
    assert "fixture-thread-one" not in json.dumps(saved)
    assert tokens.problem(home.state("workspace-connections.json")) == ""


def test_revoked_session_is_not_replaced(sessions, monkeypatch):
    root, _, enrolled, revoked = sessions
    monkeypatch.setenv("CODEX_THREAD_ID", "fixture-thread")
    actor = join(root)
    revoked.add(actor)
    with pytest.raises(Denied, match="revoked"):
        join(root)
    assert enrolled == [actor]


def test_codex_workspace_command_infers_session_agent(sessions, monkeypatch):
    root, _, enrolled, _ = sessions
    monkeypatch.setenv("CODEX_THREAD_ID", "fixture-thread")
    monkeypatch.delenv(tokens.AGENT_ENV, raising=False)
    args = SimpleNamespace(agent="", token_file="")
    cli._context(args, project_connection.selected(root))
    assert args.agent == project_session.name("codex") and enrolled == [args.agent]


def test_same_thread_concurrent_join_enrolls_once(sessions, monkeypatch):
    root, _, enrolled, _ = sessions
    monkeypatch.setenv("CODEX_THREAD_ID", "fixture-thread")
    with ThreadPoolExecutor(max_workers=2) as pool:
        first, second = list(pool.map(lambda _: join(root), range(2)))
    assert first == second and enrolled == [first]


def test_invalid_saved_session_does_not_fall_back(sessions, monkeypatch):
    root, _, _, _ = sessions
    monkeypatch.setenv("CODEX_THREAD_ID", "fixture-thread")
    join(root)
    path = home.state("workspace-connections.json")
    saved = json.loads(path.read_text())
    saved[str(root)]["sessions"][project_session.current()]["project_id"] = "b" * 32
    path.write_text(json.dumps(saved))
    with pytest.raises(Denied, match="session is invalid"):
        project_connection.selected(root)


def test_sibling_worktree_inherits_authority_and_current_session(sessions, monkeypatch, tmp_path):
    root, _, enrolled, _ = sessions
    sibling = tmp_path / "sibling"
    sibling.mkdir()
    monkeypatch.setattr(project_connection.worktreerules, "checkouts", lambda cwd: (sibling, root))
    monkeypatch.setattr(project_connection.projects, "identity", lambda cwd: PROJECT)
    inherited = project_connection.selected(sibling)
    assert inherited["host"] == "https://board.invalid:8770"
    assert inherited["agent"] == "" and inherited["automatic"] is True
    monkeypatch.setenv("CODEX_THREAD_ID", "fixture-thread")
    actor = join(root)
    assert project_connection.selected(sibling)["agent"] == actor
    assert join(sibling, "helper") == actor and enrolled == [actor]


def test_sibling_project_mismatch_is_refused(sessions, monkeypatch, tmp_path):
    root, _, _, _ = sessions
    sibling = tmp_path / "sibling"
    sibling.mkdir()
    monkeypatch.setattr(project_connection.worktreerules, "checkouts", lambda cwd: (sibling, root))
    monkeypatch.setattr(project_connection.projects, "identity", lambda cwd: "b" * 32)
    with pytest.raises(Denied, match="differs"):
        project_connection.selected(sibling)


def test_new_session_preserves_existing_authority(sessions, monkeypatch):
    root, choice, enrolled, _ = sessions
    monkeypatch.setenv("CODEX_THREAD_ID", "fixture-thread")
    monkeypatch.setattr(automatic_connection, "discover", lambda root: {**choice, "host": "https://other.invalid"})
    with pytest.raises(Denied, match="another Board"):
        join(root)
    assert not enrolled


def test_local_authentication_precedes_dev_enrollment(sessions, monkeypatch):
    root, _, enrolled, _ = sessions
    monkeypatch.setenv("CODEX_THREAD_ID", "fixture-thread")
    def denied(*args):
        raise Denied("local agent authorization failed")
    monkeypatch.setattr(automatic_connection.device_agent, "owned_project_session", denied)
    with pytest.raises(Denied, match="authorization failed"):
        join(root)
    assert not enrolled


def test_unexpected_enrollment_identity_preserves_existing_connection(sessions, monkeypatch):
    root, _, _, _ = sessions
    monkeypatch.setenv("CODEX_THREAD_ID", "fixture-thread")
    monkeypatch.setattr(automatic_connection.RemoteWorkspace, "enroll", lambda *a, **kw: {"id": "other-worker"})
    with pytest.raises(Denied, match="another native session"):
        join(root)
    assert project_connection._saved()[str(root)]["agent"] == "legacy-agent"


def test_session_identifier_precedence_and_manual_names(monkeypatch):
    monkeypatch.setenv("CODEX_SESSION_ID", "fallback-session")
    monkeypatch.setenv("CODEX_THREAD_ID", "preferred-thread")
    first = project_session.name("codex")
    assert project_session.name("worker") == "worker"
    monkeypatch.setenv("CODEX_SESSION_ID", "other-fallback")
    assert project_session.name("codex") == first
    monkeypatch.delenv("CODEX_THREAD_ID")
    assert project_session.name("codex") != first
    monkeypatch.setenv("CODEX_THREAD_ID", "invalid\nthread")
    with pytest.raises(Denied, match="identifier"):
        project_session.name("codex")


def test_claim_owner_is_readable_and_json_is_unchanged(capsys):
    owner = "canonical:" + "a" * 32 + ":codex-fixture"
    row = {"kind": "file", "key": "source.py", "owner": owner}
    cli._show(SimpleNamespace(json=False), row)
    assert "codex-fixture on shared project Board" in capsys.readouterr().out
    cli._show(SimpleNamespace(json=True), row)
    assert json.loads(capsys.readouterr().out)["owner"] == owner


@pytest.fixture(autouse=True)
def native_environment(monkeypatch):
    monkeypatch.delenv("ML_STACK_SESSION_ID", raising=False)
    monkeypatch.delenv("ML_STACK_SESSION_HARNESS", raising=False)


@pytest.mark.parametrize("native", ["claude-code", "codex"])
def test_native_sessions_are_distinct_stable_and_preserve_legacy_binding(sessions, monkeypatch, native):
    root, _, enrolled, _ = sessions
    monkeypatch.setenv("CODEX_THREAD_ID", "inherited-parent-codex")
    monkeypatch.setenv("ML_STACK_SESSION_HARNESS", native)
    monkeypatch.setenv("ML_STACK_SESSION_ID", "native-one")
    first = join(root, agent=native)
    monkeypatch.setenv("ML_STACK_SESSION_ID", "native-two")
    second = join(root, agent=native)
    assert first != second
    monkeypatch.setenv("ML_STACK_SESSION_ID", "native-one")
    assert join(root, agent=native) == first
    assert join(root, "child-label", agent=native) == first
    assert enrolled == [first, second]
    assert project_connection._saved()[str(root)]["agent"] == "legacy-agent"
    assert "native-one" not in json.dumps(project_connection._saved())


def test_native_context_does_not_select_inherited_other_harness(monkeypatch):
    monkeypatch.setenv("CODEX_THREAD_ID", "same-native-session")
    codex = project_session.current()
    monkeypatch.setenv("ML_STACK_SESSION_ID", "same-native-session")
    monkeypatch.setenv("ML_STACK_SESSION_HARNESS", "claude-code")
    assert project_session.current() != codex
    assert project_session.current("codex") == ""
    assert project_session.name("codex") == "codex"
    assert project_session.name("claude-code").startswith("claude-code-")


@pytest.mark.parametrize("native,session", [("claude-code", ""), ("", "event"),
    ("claude-code", "bad\nvalue"), ("claude-code", "bad value"),
    ("claude-code", "x" * 257), ("not/a/harness", "event")])
def test_invalid_native_context_is_refused(monkeypatch, native, session):
    monkeypatch.setenv("ML_STACK_SESSION_HARNESS", native)
    monkeypatch.setenv("ML_STACK_SESSION_ID", session)
    with pytest.raises(Denied, match="native session"):
        project_session.current()


def test_native_saved_foreign_harness_and_revocation_are_not_replaced(sessions, monkeypatch):
    root, _, enrolled, revoked = sessions
    monkeypatch.setenv("ML_STACK_SESSION_HARNESS", "claude-code")
    monkeypatch.setenv("ML_STACK_SESSION_ID", "native-session")
    actor = join(root, agent="claude-code")
    revoked.add(actor)
    with pytest.raises(Denied, match="revoked"):
        join(root, agent="claude-code")
    assert enrolled == [actor]
    saved = project_connection._saved()
    saved[str(root)]["sessions"][project_session.current()]["local_agent"] = "codex"
    home.state("workspace-connections.json").write_text(json.dumps(saved))
    with pytest.raises(Denied, match="session is invalid"):
        project_connection.selected(root)
