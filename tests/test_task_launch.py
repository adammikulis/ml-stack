"""Task-scoped existing worker launch preserves live grants and native execution limits."""

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
from taskboard_kit import board

from ml_stack.workspace import (
    localagent as la,
    localloop,
    localmodel,
    localstart,
    project,
    task_authority,
    task_caps,
    task_launch,
    tokens,
)
from ml_stack.workspace.identity import Denied

__all__ = ['board']


@pytest.fixture
def audit(board):
    token = board.agent('audit-worker')
    tokens.store(board.ws.base, 'audit-worker', token)
    grant = project.describe(str(board.source))
    person = board.ws.auth(board.owner)
    for ident in ('lead', 'audit-worker'):
        board.ws.registry.set_project(person, ident, grant)
    runner = la.Agent('audit-worker', 'qwen', role='read-only', project=str(board.source),
                      ctx=131072, max_effort='low', orders_from=('lead',), harness='claude',
                      max_output_tokens=12345)
    la.save(board.ws, runner)
    spec = {**board.spec, 'source_key': 'audit-request', 'project': grant, 'assignees': ['audit-worker']}
    task = board.board.create(board.parent, spec)
    return board, runner, task


@pytest.mark.parametrize('harness', ['pi', 'claude', 'codex'])
def test_existing_nonchild_launch_preserves_identity_configuration_and_caps(audit, monkeypatch, harness):
    kit, runner, task = audit
    runner = replace(runner, harness=harness)
    la.save(kit.ws, runner)
    assert not kit.ws.auth(tokens.load(kit.ws.base, runner.name)).parent
    monkeypatch.setattr(localstart.localmodel, 'choose', lambda *_args, **_kw: localmodel.Pick(ref='qwen', name='qwen'))
    monkeypatch.setattr(localstart.lp, 'admit', lambda *_: ('', ''))
    monkeypatch.setattr(localstart.jobs, 'detach', lambda *_args, **_kw: SimpleNamespace(pid=123, log='runtime.log'))
    monkeypatch.setattr(localstart, 'started_at', lambda _: 10)
    monkeypatch.setattr(localstart, '_mint', lambda *_: pytest.fail('minted identity'))
    monkeypatch.setattr(localstart, '_record_model', lambda *_: pytest.fail('changed verified model'))
    task_launch.start(kit.ws, kit.parent, runner.name, task['id'])
    saved = la.load(kit.ws, runner.name)
    assert saved.identity == runner.name and saved.profile == 'coding' and saved.harness == harness
    assert saved.max_output_tokens == runner.max_output_tokens == 12345
    assert (saved.model, saved.role, saved.ctx, saved.max_effort, saved.orders_from) == (
        runner.model, runner.role, runner.ctx, runner.max_effort, runner.orders_from)
    assert localloop.caps_of(saved) == localloop.caps_of(runner)


@pytest.mark.redteam
@pytest.mark.parametrize('cause', ['caller', 'assignee', 'project', 'orders', 'source', 'active'])
def test_exact_task_authority_denies_foreign_or_changed_assignment(audit, cause):
    kit, runner, task = audit
    token, ident = kit.parent, task['id']
    if cause == 'caller':
        token = kit.agent('unrelated')
    elif cause == 'assignee':
        other = kit.board.create(kit.parent, {**kit.spec, 'source_key': 'other-task', 'project': task['project']})
        ident = other['id']
    elif cause == 'project':
        kit.ws.registry.set_project(kit.ws.auth(kit.owner), runner.name, {'key': 'different'})
    elif cause == 'orders':
        la.save(kit.ws, replace(runner, orders_from=('unrelated',)))
    elif cause == 'source':
        la.save(kit.ws, replace(runner, project=str(kit.base.parent)))
    else:
        la.Status(kit.ws, runner.name).update(state='working', task='task:' + 'b' * 32)
    with pytest.raises(Denied):
        task_authority.authorize(kit.ws, token, runner.name, ident)


@pytest.mark.redteam
def test_native_tool_admission_refuses_exhausted_and_invalid_counter(tmp_path):
    path = tmp_path / 'counter.json'
    path.write_text(json.dumps({'version': 1, 'calls': 0, 'limit': 1}))
    assert task_caps.admit(path)
    assert not task_caps.admit(path)
    path.write_text(json.dumps({'version': 1, 'calls': False, 'limit': 1}))
    with pytest.raises(ValueError):
        task_caps.admit(path)


@pytest.mark.redteam
def test_fractional_saved_counter_cap_is_rejected():
    runner = la.Agent('audit-worker', 'qwen', extra={'task_caps': {'calls': 0.5}})
    with pytest.raises(ValueError):
        localloop.caps_of(runner)
