"""Authorized issue selection, durable leases, retries and native worker dispatch."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest
from workspace_kit import Kit, clean_env

from ml_stack.workspace import backlog, issuepump, localagent as la, tokens
from ml_stack.workspace.identity import Denied


@pytest.fixture
def setup(monkeypatch, tmp_path):
    kit = Kit(clean_env(monkeypatch, tmp_path))
    parent = kit.agent("lead")
    project = tmp_path / "isolated"
    project.mkdir()
    (project / ".git").write_text("gitdir: /scratch\n")
    child = kit.ws.delegate(parent, "worker")
    agent = la.Agent("worker", "local.gguf", identity=child["id"], profile="coding", project=str(project))
    la.save(kit.ws, agent)
    backlog.configure(kit.ws, parent, agent.name, "sample/project", str(project))
    return kit, parent, agent, project


def issue(number=1, **extra):
    return {"number": number, "title": "Add a missing feature", "body": "Acceptance: a real patch.",
            "url": f"https://github.com/sample/project/issues/{number}", "updatedAt": "revision1",
            "assignees": [], "labels": [], **extra}


def test_worker_cannot_authorize_itself_or_another_parents_worker(setup):
    kit, _, agent, project = setup
    for token in (tokens.load(kit.base, agent.identity), kit.agent("other")):
        with pytest.raises(Denied):
            backlog.configure(kit.ws, token, agent.name, "sample/project", str(project))
    with pytest.raises(ValueError, match="owner/name"):
        backlog.configure(kit.ws, kit.owner, agent.name, "../outside", str(project))
    with pytest.raises(ValueError, match="isolated git worktree"):
        backlog.configure(kit.ws, kit.owner, agent.name, "sample/project", str(project.parent))


def test_two_workers_cannot_claim_the_same_issue_and_reopen_preserves_result(setup):
    kit, parent, agent, project = setup
    child = kit.ws.delegate(parent, "second")
    other = replace(agent, name="second", identity=child["id"])
    la.save(kit.ws, other)
    backlog.configure(kit.ws, parent, other.name, "sample/project", str(project))
    with ThreadPoolExecutor(2) as pool:
        jobs = list(pool.map(lambda a: backlog.pick(kit.ws, a, fetcher=lambda _: [issue()]), (agent, other)))
    chosen = next(row for row, _ in jobs if row)
    assert sum(row is not None for row, _ in jobs) == 1
    with backlog._store(kit.ws) as graph:
        record = backlog._record(graph, chosen["key"])
        assert {edge["rel"] for edge in graph.edges()} == {"issue", "work"}
    owner = agent if record["owner"] == agent.identity else other
    backlog.finish(kit.ws, chosen, owner, ("answer", "Reviewable patch proposed"))
    assert backlog.pick(kit.reopen(), agent, fetcher=lambda _: [issue()])[0] is None


def test_failure_backoff_is_bounded_and_an_updated_issue_reopens_work(setup):
    kit, _, agent, _ = setup
    now = [1000.0]
    source = [issue()]
    def select():
        return backlog.pick(kit.ws, agent, fetcher=lambda _: source, clock=lambda: now[0])
    for _ in range(3):
        job, _ = select()
        assert job
        backlog.finish(kit.ws, job, agent, ("status", "A bounded failure"), clock=lambda: now[0])
        assert select()[0] is None
        now[0] += 1000
    assert select()[0] is None
    source[0] = issue(updatedAt="revision2")
    now[0] += backlog.POLL_S
    assert select()[0]


def test_assigned_and_blocked_issues_are_skipped_and_fetch_failures_are_visible(setup):
    kit, _, agent, _ = setup
    rows = [issue(1, assignees=[{"login": "someone"}]), issue(2, labels=[{"name": "blocked"}]), issue(3)]
    job, _ = backlog.pick(kit.ws, agent, fetcher=lambda _: rows)
    assert job["number"] == 3
    def broken(_):
        raise RuntimeError("GitHub unavailable")
    _, detail = backlog.pick(kit.ws, agent, fetcher=broken, clock=lambda: 10**12)
    assert "blocked" in detail and "GitHub unavailable" in detail


def test_idle_dispatch_posts_one_authenticated_job_and_reconciles_its_native_reply(setup, monkeypatch):
    kit, parent, agent, project = setup
    monkeypatch.setattr(backlog, "fetch", lambda _: [issue()])
    monkeypatch.setattr(la, "alive", lambda _: True)
    la.Status(kit.ws, agent.name).update(state="idle")
    original = backlog.pick
    monkeypatch.setattr(backlog, "pick", lambda ws, a: original(ws, a, fetcher=backlog.fetch))
    issuepump.step(kit.ws, parent, agent.name)
    child = tokens.load(kit.base, agent.identity)
    tasks = kit.ws.inbox(child, raw=True, limit=1)
    assert len(tasks) == 1 and tasks[0]["from"] == "lead"
    assert str(project) in tasks[0]["text"] and "bare pytest" in tasks[0]["text"]
    issuepump.step(kit.ws, parent, agent.name)
    assert len(kit.ws.inbox(child, raw=True)) == 1
    kit.ws.send(child, "lead", "answer", "Patch ready for parent review", reply_to=tasks[0]["seq"])
    issuepump.step(kit.ws, parent, agent.name)
    assert original(kit.ws, agent, fetcher=backlog.fetch)[0] is None


def test_interrupted_dispatch_recovers_outbox_receipt_without_duplicate_task(setup, monkeypatch):
    kit, parent, agent, _ = setup
    monkeypatch.setattr(la, "alive", lambda _: True)
    la.Status(kit.ws, agent.name).update(state="idle")
    job, _ = backlog.pick(kit.ws, agent, fetcher=lambda _: [issue()])
    key = f"dispatch:{agent.identity}"
    with backlog._store(kit.ws) as graph:
        graph.put_doc(key, {"issue": job, "seq": 0, "subject": "Unique dispatch"})
    kit.ws.send(parent, agent.identity, "task", "existing job", subject="Unique dispatch")
    issuepump.step(kit.ws, parent, agent.name)
    assert len(kit.ws.inbox(tokens.load(kit.base, agent.identity), raw=True)) == 1
