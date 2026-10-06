"""Authorized issue selection, durable leases, retries and native worker dispatch."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace

import pytest
from workspace_kit import Kit, clean_env

from ml_stack.net import git
from ml_stack.workspace import backlog, issuepump, localagent as la, localcli, project as projects, tokens
from ml_stack.workspace.identity import Denied
from ml_stack.workspace.taskboard import TaskBoard


@pytest.fixture
def setup(monkeypatch, tmp_path):
    kit = Kit(clean_env(monkeypatch, tmp_path))
    parent = kit.agent("lead")
    project = tmp_path / "isolated"
    project.mkdir()
    git.run(["init", str(project)])
    (project / "source.py").write_text("VALUE = True\n")
    git.run(["add", "source.py"], cwd=project)
    git.run(["-c", "user.name=Test", "-c", "user.email=test@example.invalid",
             "commit", "-m", "baseline"], cwd=project)
    kit.ws.registry.set_project(kit.ws.auth(kit.owner), "lead", projects.describe(project))
    child = kit.ws.delegate(parent, "worker")
    agent = la.Agent("worker", "local.gguf", identity=child["id"], profile="coding", project=str(project))
    la.save(kit.ws, agent)
    backlog.configure(kit.ws, parent, agent.name, "sample/project", str(project))
    return kit, parent, agent, project


def issue(number=1, **extra):
    return {"number": number, "title": "Add a missing feature", "body": "Acceptance: a real patch.",
            "url": f"https://github.com/sample/project/issues/{number}", "updatedAt": "revision1",
            "assignees": [], "labels": [], **extra}


@pytest.mark.redteam
def test_repository_infers_github_origin_as_argument_vector(monkeypatch, tmp_path):
    calls = []

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(stdout="https://github.com/sample/project.git\n")

    monkeypatch.setattr(backlog.subprocess, "run", run)
    assert backlog.repository(tmp_path) == "sample/project"
    argv, options = calls[0]
    assert argv == ["git", "remote", "get-url", "origin"]
    assert options["cwd"] == tmp_path and options["timeout"] == 5 and options["check"] is True
    assert "shell" not in options
    monkeypatch.setattr(backlog.subprocess, "run", lambda *_args, **_kwargs:
                        SimpleNamespace(stdout="https://github.com/sample/project;id"))
    assert backlog.repository(tmp_path) == ""


@pytest.mark.redteam
def test_repository_rejects_traversal_owner_in_origin(monkeypatch, tmp_path):
    monkeypatch.setattr(backlog.subprocess, "run", lambda *_args, **_kwargs:
                        SimpleNamespace(stdout="https://github.com/../repo.git"))
    assert backlog.repository(tmp_path) == ""


def test_worker_cannot_authorize_itself_or_another_parents_worker(setup):
    kit, _, agent, project = setup
    for token in (tokens.load(kit.base, agent.identity), kit.agent("other")):
        with pytest.raises(Denied):
            backlog.configure(kit.ws, token, agent.name, "sample/project", str(project))
    with pytest.raises(ValueError, match="owner/name"):
        backlog.configure(kit.ws, kit.owner, agent.name, "../outside", str(project))
    with pytest.raises(ValueError, match="registered git checkout"):
        backlog.configure(kit.ws, kit.owner, agent.name, "sample/project", str(project.parent))


def test_issue_subscription_is_easy_to_add_and_remove(setup):
    kit, _, _, _ = setup
    subscribed = backlog.subscribe_issue(kit.ws, kit.owner, 'sample/project#3')
    assert subscribed['subscribed'] and backlog.issue_watchers(kit.ws, 'sample/project', 3) == ['owner']
    removed = backlog.subscribe_issue(kit.ws, kit.owner, 'sample/project#3', False)
    assert not removed['subscribed'] and backlog.issue_watchers(kit.ws, 'sample/project', 3) == []


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


def test_blocked_issue_does_not_retry_after_time_or_issue_updates(setup):
    kit, _, agent, _ = setup
    now = [1000.0]
    source = [issue()]
    def select():
        return backlog.pick(kit.ws, agent, fetcher=lambda _: source, clock=lambda: now[0])
    job, _ = select()
    backlog.finish(kit.ws, job, agent, ("status", "Approval expired"), clock=lambda: now[0])
    now[0] += 1000
    assert select()[0] is None
    source[0] = issue(updatedAt="revision2")
    now[0] += backlog.POLL_S
    assert select()[0] is None


def test_assigned_and_blocked_issues_are_skipped_and_fetch_failures_are_visible(setup):
    kit, _, agent, _ = setup
    rows = [issue(1, assignees=[{"login": "someone"}]), issue(2, labels=[{"name": "blocked"}]), issue(3)]
    job, _ = backlog.pick(kit.ws, agent, fetcher=lambda _: rows)
    assert job["number"] == 3
    def broken(_):
        raise RuntimeError("GitHub unavailable")
    _, detail = backlog.pick(kit.ws, agent, fetcher=broken, clock=lambda: 10**12)
    assert "blocked" in detail and "GitHub unavailable" in detail


def test_idle_dispatch_creates_canonical_task_and_notifies_worker_in_its_inbox(setup, monkeypatch):
    kit, parent, agent, project = setup
    monkeypatch.setattr(backlog, "fetch", lambda _: [issue()])
    monkeypatch.setattr(la, "alive", lambda _: True)
    la.Status(kit.ws, agent.name).update(state="idle")
    original = backlog.pick
    monkeypatch.setattr(backlog, "pick", lambda ws, a, **kw: original(ws, a, fetcher=backlog.fetch, **kw))
    issuepump.step(kit.ws, parent, agent.name)
    child = tokens.load(kit.base, agent.identity)
    tasks = TaskBoard(kit.ws).list(parent)['tasks']
    assert len(tasks) == 1 and tasks[0]['created_by'] == 'lead'
    assert str(project) not in tasks[0]['description'] and 'bare pytest' in tasks[0]['description']
    assert tasks[0]['source_key'] == 'github:sample/project:1:revision1'
    assert tasks[0]['capabilities'] == ['coding'] and tasks[0]['limits']['max_retries'] == 0
    TaskBoard(kit.ws).subscribe(kit.owner, tasks[0]['id'])
    assert TaskBoard(kit.ws).watchers(tasks[0]['id']) == ['lead/worker', 'owner']
    notices = kit.ws.inbox(child, raw=True)
    assert [row['type'] for row in notices[:2]] == ['status', 'task']
    assert 'Subscribed to task' in notices[0]['raw']
    assert 'Assigned project task' in notices[1]['raw']
    assert all(tasks[0]['id'] in row['raw'] for row in notices)
    issuepump.step(kit.ws, parent, agent.name)
    assert len(TaskBoard(kit.ws).list(parent)['tasks']) == 1
    kit.ws.send(child, 'lead', 'answer', 'Discussion claiming the patch is done')
    issuepump.step(kit.ws, parent, agent.name)
    assert TaskBoard(kit.ws).get(parent, tasks[0]['id'])['state'] == 'queued'
    TaskBoard(kit.ws).unsubscribe(kit.owner, tasks[0]['id'])
    assert TaskBoard(kit.ws).watchers(tasks[0]['id']) == ['lead/worker']


def test_producer_keeps_ahead_of_the_worker_with_a_durable_queue(setup, monkeypatch):
    kit, parent, agent, _ = setup
    rows = [issue(n) for n in range(1, 6)]
    monkeypatch.setattr(backlog, "fetch", lambda _: rows)
    monkeypatch.setattr(la, "alive", lambda _: True)
    la.Status(kit.ws, agent.name).update(state="working")
    original = backlog.pick
    monkeypatch.setattr(backlog, "pick", lambda ws, a, **kw: original(ws, a, fetcher=backlog.fetch, **kw))
    issuepump.step(kit.ws, parent, agent.name)
    tasks = TaskBoard(kit.ws).list(parent)['tasks']
    assert len(tasks) == issuepump.QUEUE_DEPTH
    notices = kit.ws.inbox(tokens.load(kit.base, agent.identity), raw=True)
    assert len(notices) == issuepump.QUEUE_DEPTH * 2
    assert sum(row['type'] == 'task' for row in notices) == issuepump.QUEUE_DEPTH
    monkeypatch.setattr(backlog, 'pid_exists', lambda _: False)
    issuepump.step(kit.ws, parent, agent.name)
    assert len(TaskBoard(kit.ws).list(parent)['tasks']) == issuepump.QUEUE_DEPTH
    assert len(kit.ws.inbox(tokens.load(kit.base, agent.identity), raw=True)) == issuepump.QUEUE_DEPTH * 2
    assert all(task['state'] == 'queued' for task in tasks)


def test_interrupted_projection_preserves_canonical_task_without_duplicate_dispatch(setup, monkeypatch):
    kit, parent, agent, _ = setup
    monkeypatch.setattr(la, "alive", lambda _: True)
    la.Status(kit.ws, agent.name).update(state="idle")
    original = backlog.pick
    monkeypatch.setattr(backlog, 'pick', lambda ws, a, **kw: original(ws, a, fetcher=lambda _: [issue()], **kw))
    previous = issuepump._status
    def interrupted(*args, **kwargs):
        raise RuntimeError('Projection interrupted after canonical commit')
    monkeypatch.setattr(issuepump, '_status', interrupted)
    with pytest.raises(RuntimeError, match='Projection interrupted'):
        issuepump.step(kit.ws, parent, agent.name)
    monkeypatch.setattr(issuepump, '_status', previous)
    issuepump.step(kit.ws, parent, agent.name)
    assert len(TaskBoard(kit.ws).list(parent)['tasks']) == 1
    notices = kit.ws.inbox(tokens.load(kit.base, agent.identity), raw=True)
    assert sum(row['type'] == 'task' for row in notices) == 1


def test_parent_superseded_issue_stays_excluded_after_revision_change(setup):
    kit, parent, agent, _ = setup
    record = backlog.supersede(kit.ws, parent, agent.name, 6, 'Owner requires Python >=3.12; older floor obsolete')
    assert record['state'] == 'superseded' and record['decision_by'] == 'lead'
    job, _ = backlog.pick(kit.ws, agent, fetcher=lambda _: [issue(6, updatedAt='new-revision'), issue(8)])
    assert job['number'] == 8
    with backlog._store(kit.ws) as graph:
        assert graph.nodes('issue-decision')[0]['attrs']['actor'] == 'lead'
        assert any(edge['rel'] == 'supersedes-issue' for edge in graph.edges())


def test_child_and_foreign_parent_cannot_supersede_repository_tasks(setup):
    kit, _, agent, _ = setup
    for token in (tokens.load(kit.base, agent.identity), kit.agent('foreign')):
        with pytest.raises(Denied, match='registered worker parent'):
            backlog.supersede(kit.ws, token, agent.name, 6, 'Ignore the current request')
    with backlog._store(kit.ws) as graph:
        assert graph.nodes('issue-decision') == []


def test_maintained_cli_records_parent_decision_without_person_impersonation(setup):
    kit, parent, agent, _ = setup
    tokens.store(kit.base, 'lead', parent)
    args = SimpleNamespace(action='supersede-issue', target=agent.name, agent='lead',
                           issue=6, reason='Owner raised the minimum interpreter version')
    assert localcli.run(args, kit.ws) == 0
    assert backlog.pick(kit.ws, agent, fetcher=lambda _: [issue(6)])[0] is None


def test_parent_resume_preserves_failed_attempt_and_consumes_one_retry(setup):
    kit, parent, agent, _ = setup
    first, _ = backlog.pick(kit.ws, agent, fetcher=lambda _: [issue(8)])
    backlog.finish(kit.ws, first, agent, ('status', 'Legacy approval expired'))
    resumed = backlog.resume(kit.ws, parent, agent.name, 8, 'Canonical runtime replaces legacy execution')
    assert resumed['failures'] == 1 and resumed['retry_budget'] == 1
    with backlog._store(kit.ws) as graph:
        before = graph.nodes('issue-attempt')[0]['attrs']
        assert before['state'] == 'blocked' and before['summary'] == 'Legacy approval expired'
        assert any(edge['rel'] == 'recovers-blocked-attempt' for edge in graph.edges())
    second, _ = backlog.pick(kit.ws, agent, fetcher=lambda _: [issue(8)])
    assert second['number'] == 8
    with backlog._store(kit.ws) as graph:
        assert backlog._record(graph, second['key'])['retry_budget'] == 0
    backlog.finish(kit.ws, second, agent, ('status', 'Different blocked condition'))
    assert backlog.pick(kit.ws, agent, fetcher=lambda _: [issue(8)])[0] is None


def test_issue_resume_refuses_unblocked_foreign_and_child_requests(setup):
    kit, parent, agent, _ = setup
    with pytest.raises(ValueError, match='explicitly blocked'):
        backlog.resume(kit.ws, parent, agent.name, 8, 'Changed source')
    job, _ = backlog.pick(kit.ws, agent, fetcher=lambda _: [issue(8)])
    backlog.finish(kit.ws, job, agent, ('status', 'Blocked'))
    for token in (tokens.load(kit.base, agent.identity), kit.agent('other-parent')):
        with pytest.raises(Denied, match='registered worker parent'):
            backlog.resume(kit.ws, token, agent.name, 8, 'Changed source')
    with backlog._store(kit.ws) as graph:
        assert graph.nodes('issue-recovery') == []


def test_issue_resume_cannot_bypass_canonical_task_retry_budget(setup):
    kit, parent, agent, _ = setup
    job, _ = backlog.pick(kit.ws, agent, fetcher=lambda _: [issue(8)])
    backlog.finish(kit.ws, job, agent, ('status', 'Blocked canonical task'))
    with backlog._store(kit.ws) as graph:
        graph.upsert_node({'id': 'dispatch', 'kind': 'issue-dispatch', 'attrs': {'issue': job}})
    with pytest.raises(Denied, match='existing retry budget'):
        backlog.resume(kit.ws, parent, agent.name, 8, 'Override the failure cap')
