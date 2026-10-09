"""Records that a lookup or an unchanged rebind used to rewrite are left alone (docs/disk-writes.md)."""

from types import SimpleNamespace

import pytest

from ml_stack import home
from ml_stack.graph.store import GraphStore
from ml_stack.workspace import (
    integration_git as repo,
    project_connection as connection,
    worktree_lifecycle as lifecycle,
)
from ml_stack.workspace.device_accounts import account_for

PROJECT = "a" * 32


class Remote:
    host = "http://127.0.0.1:8770"
    project_id = PROJECT
    cluster_key = ""

    def token(self, **kwargs):
        return "project-agent-capability"

    def call(self, operation, token, *args, **kwargs):
        return {"id": "mac", "role": "agent", "can": ["send", "read", "claim"], "project": {"key": PROJECT}}


def test_binding_an_unchanged_connection_does_not_rewrite_the_record(tmp_path, monkeypatch):
    monkeypatch.setenv("ML_STACK_HOME", str(tmp_path / "state"))
    project, other = tmp_path / "project", tmp_path / "second"
    project.mkdir()
    other.mkdir()
    connection.bind(Remote(), project, "mac")
    path = home.state("workspace-connections.json")
    first = path.stat()
    connection.bind(Remote(), project, "mac")
    again = path.stat()
    assert (again.st_mtime_ns, again.st_ino) == (first.st_mtime_ns, first.st_ino)
    connection.bind(Remote(), other, "mac")
    assert str(other.resolve()) in path.read_text()


def test_an_account_lookup_writes_nothing(tmp_path):
    ws = SimpleNamespace(base=tmp_path)
    assert account_for(ws, "worker") is None
    assert not list(tmp_path.iterdir()), "a lookup must not create the store"
    with GraphStore(tmp_path / "device-accounts.db") as graph:
        graph.upsert_node({"id": "agent:worker", "kind": "agent", "label": "worker"})
    database = tmp_path / "device-accounts.db"
    seen = (database.stat().st_mtime_ns, database.read_bytes())
    assert account_for(ws, "worker") is None
    assert (database.stat().st_mtime_ns, database.read_bytes()) == seen


@pytest.fixture
def checkout(tmp_path):
    primary = tmp_path / "repository"
    primary.mkdir()
    repo.git(primary, "init", "-b", "development")
    repo.git(primary, "config", "user.name", "Fixture")
    repo.git(primary, "config", "user.email", "fixture@example.test")
    (primary / "source.py").write_text("value = 1\n")
    repo.git(primary, "add", "source.py")
    repo.git(primary, "commit", "-m", "chore: fixture")
    made = tmp_path / "checkout"
    repo.git(primary, "worktree", "add", "-b", "worker/change", str(made))
    return made


def test_remembering_an_unchanged_checkout_does_not_rewrite_the_store(tmp_path, checkout):
    base = tmp_path / "lifecycle"
    base.mkdir()
    lifecycle.remember(base, "worker", "helper", str(checkout))
    database = base / "worktree-lifecycle.db"
    first = (database.stat().st_mtime_ns, database.read_bytes())
    lifecycle.remember(base, "worker", "helper", str(checkout))
    assert (database.stat().st_mtime_ns, database.read_bytes()) == first
    (checkout / "more.py").write_text("x = 1\n")
    repo.git(checkout, "add", "more.py")
    repo.git(checkout, "commit", "-m", "chore: more")
    lifecycle.remember(base, "worker", "helper", str(checkout))
    assert len(lifecycle.scopes(base, "worker")[0]["commits"]) == 2
