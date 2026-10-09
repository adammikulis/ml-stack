"""An agent acting on the owner's order deploys the runtime only while the `runtime.deploy` gate is delegated.

Each test runs the real `poolhouse runtime` command in-process as an agent-started process against a temporary
git checkout and a temporary runtimes root; only the wheel build is replaced by the deploy tests' fake builder, and the activity log
(silent under test) by a list that keeps what the command asked it to record.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from test_runtime_deploy import builder, commit, world  # noqa: F401

from poolhouse import authority, runtime_cli, runtime_deploy, runtime_store, runtime_trust
from poolhouse.activity import writer

pytestmark = pytest.mark.slow
RECORDS: list[SimpleNamespace] = []


@pytest.fixture
def lead(world, monkeypatch):  # noqa: F811
    """The deploy world as a lead agent's process: marked, naming the recorded checkout, building with the fake builder."""
    repo, launchers, _ = world
    monkeypatch.setenv("CLAUDECODE", "1")
    # the scrub of POOLHOUSE_HOME has its own tests (test_runtime_trust); here the state root must stay the temporary one
    monkeypatch.setattr(runtime_trust, "ignore_redirects", lambda: [])
    monkeypatch.delenv("POOLHOUSE_AUTHORITY_FLOOR", raising=False)
    runtime_deploy.prepare_root()
    runtime_store.write_state({"checkout": str(repo), "launchers": str(launchers)})
    RECORDS.clear()
    monkeypatch.setattr(writer, "record", lambda kind, **said: RECORDS.append(SimpleNamespace(kind=kind, **said)) or True)
    fake = {"build": builder()}
    real = runtime_deploy.ensure
    monkeypatch.setattr(runtime_deploy, "ensure", lambda plan, **kw: real(plan, builder=fake["build"], **kw))
    return repo, launchers, fake


def deploys():
    return [row for row in RECORDS if row.kind == "runtime.deploy"]


def selected():
    return runtime_store.selection().get("commit", "")


def test_a_delegated_agent_s_ensure_deploys_and_is_audited_with_the_agent(lead, capsys):
    repo, _, _ = lead
    head = commit(repo, "a")
    assert runtime_cli.main(["ensure"]) == 0, capsys.readouterr().err
    assert selected() == head
    [entry] = deploys()
    assert (entry.outcome, entry.meta["command"], entry.meta["agent"]) == ("switched", "ensure", "runtime-agent")
    assert entry.meta["person"] is False and entry.meta["authority"] == authority.DELEGATED
    used = [row for row in authority.audit_rows() if row["event"] == "authority.use"]
    assert [(row["gate"], row["agent"]) for row in used] == [("runtime.deploy", "runtime-agent")]


def test_an_agent_is_refused_while_the_gate_is_the_person_s_and_nothing_is_built(lead, capsys):
    repo, _, _ = lead
    commit(repo, "a")
    authority.set_state(["runtime.deploy"], authority.PERSON, by="lead")
    assert runtime_cli.main(["ensure"]) == 1
    assert "for a person" in capsys.readouterr().err
    assert selected() == "" and not runtime_store.candidates()
    [entry] = deploys()
    assert (entry.outcome, entry.meta["command"]) == ("refused", "ensure") and entry.meta["agent"] == "runtime-agent"


def test_revoking_the_delegation_refuses_the_next_deploy_and_keeps_the_selection(lead):
    repo, _, _ = lead
    first = commit(repo, "a")
    assert runtime_cli.main(["ensure"]) == 0
    authority.set_state(["runtime.deploy"], authority.PERSON, by="lead")
    commit(repo, "b")
    assert runtime_cli.main(["ensure"]) == 1
    assert selected() == first
    authority.set_state(["runtime.deploy"], authority.DELEGATED, by="lead")
    assert runtime_cli.main(["ensure"]) == 0 and selected() != first


def test_rollback_follows_the_same_gate(lead):
    repo, _, _ = lead
    first = commit(repo, "a")
    runtime_cli.main(["ensure"])
    commit(repo, "b")
    runtime_cli.main(["ensure"])
    authority.set_state(["runtime.deploy"], authority.PERSON, by="lead")
    assert runtime_cli.main(["rollback"]) == 1 and selected() != first
    authority.set_state(["runtime.deploy"], authority.DELEGATED, by="lead")
    assert runtime_cli.main(["rollback"]) == 0 and selected() == first
    assert [e.meta["command"] for e in deploys()].count("rollback") == 2


def test_a_failed_smoke_under_a_delegated_deploy_leaves_the_selection_alone(lead):
    repo, launchers, fake = lead
    first = commit(repo, "a")
    runtime_cli.main(["ensure"])
    before = (launchers / "poolhouse-workspace").read_bytes()
    commit(repo, "b")
    fake["build"] = builder(hook_code=3)
    assert runtime_cli.main(["ensure"]) == 1
    assert selected() == first and (launchers / "poolhouse-workspace").read_bytes() == before
    assert deploys()[-1].outcome == "failed"


def test_a_process_no_agent_started_is_not_gated(lead, monkeypatch):
    repo, _, _ = lead
    head = commit(repo, "a")
    monkeypatch.delenv("CLAUDECODE")
    monkeypatch.delenv("POOLHOUSE_WORKSPACE_AGENT")
    authority.set_state(["ALL"], authority.PERSON, by="lead")
    assert runtime_cli.main(["ensure"]) == 0 and selected() == head


def test_dev_delegates_the_deploy_gate_and_prod_keeps_it_with_the_person(tmp_path, monkeypatch):
    monkeypatch.setenv("POOLHOUSE_HOME", str(tmp_path / "state"))
    monkeypatch.delenv("POOLHOUSE_AUTHORITY_FLOOR", raising=False)
    assert authority.GATES["runtime.deploy"] == "runtime"
    assert "runtime.deploy" not in authority.PROD_DELEGATED
    assert authority.state_of("runtime.deploy") == authority.DELEGATED


def test_restart_host_follows_the_same_gate_and_is_audited(lead, capsys):
    repo, _, _ = lead
    commit(repo, "a")
    runtime_cli.main(["ensure"])
    authority.set_state(["runtime.deploy"], authority.PERSON, by="lead")
    assert runtime_cli.main(["restart-host", "--port", "1"]) == 1
    assert deploys()[-1].outcome == "refused" and deploys()[-1].meta["command"] == "restart-host"
    authority.set_state(["runtime.deploy"], authority.DELEGATED, by="lead")
    assert runtime_cli.main(["restart-host", "--port", "1"]) == 0
    assert "not-running" in capsys.readouterr().out
    last = deploys()[-1]
    assert (last.outcome, last.meta["command"], last.meta["authority"]) == ("not-running", "restart-host", authority.DELEGATED)

