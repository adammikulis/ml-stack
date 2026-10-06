"""Local launch callers use the authenticated project agent's authority."""

import json
from types import SimpleNamespace

import pytest
from workspace_kit import Kit, clean_env

from ml_stack.workspace import localcli, localroute, localstart, tokens


@pytest.mark.parametrize('caller', ['cli', 'browser'])
def test_launch_callers_pass_own_agent_parent_without_person_credentials(monkeypatch, caller):
    seen = {}
    workspace = object()
    monkeypatch.setattr(tokens, 'read_file', lambda *_: pytest.fail('read person credential'))
    monkeypatch.setattr(localstart, 'launch_parent', lambda ws, project: 'own-parent-token')
    def start(ws, ask, **authority):
        seen.update(workspace=ws, ask=ask, authority=authority)
        return localstart.Started('worker', 123, 'qwen', 'read-only', already=True)
    monkeypatch.setattr(localstart, 'start', start)
    if caller == 'cli':
        monkeypatch.setattr(localcli.localmodel, 'choose', lambda *_args, **_kw:
                            SimpleNamespace(ok=True, name='qwen', note='fits'))
        args = SimpleNamespace(agent='', profile='coding', ctx='', project='.', role='read-only',
            model='qwen', name='worker', effort='off', max_effort='medium', orders_from='lead',
            harness='claude', repo='sample/project', no_wait=True)
        assert localcli._start(args, workspace) == 0
        assert seen['ask'].repo == 'sample/project' and seen['ask'].authority is None
    else:
        code, _ = localroute._write(workspace, 'start', json.dumps({
            'model': 'qwen', 'name': 'worker', 'profile': 'coding', 'project': '.',
            'role': 'read-only', 'repo': 'sample/project'}).encode())
        assert code == 200 and seen['ask'].repo == 'sample/project'
    assert seen['workspace'] is workspace
    assert seen['authority']['parent_token'] == 'own-parent-token'
    assert 'person_token' not in seen['authority']


def test_parent_launch_binds_only_its_owned_worker(monkeypatch):
    workspace = SimpleNamespace(auth=lambda token: SimpleNamespace(role='agent'))
    calls = []
    monkeypatch.setattr(localstart, '_start', lambda *_args, **_kw:
                        localstart.Started('worker', 123, 'qwen', 'read-only', already=True))
    monkeypatch.setattr(localstart.device_agent, 'enroll', lambda *_: pytest.fail('person enrollment'))
    monkeypatch.setattr(localstart.device_agent, 'bind_owned_worker',
                        lambda *args: calls.append(args), raising=False)
    localstart.start(workspace, localstart.Ask(), parent_token='own-parent-token')
    assert calls == [(workspace, 'own-parent-token', 'worker')]


def test_device_membership_is_bound_before_worker_process_starts(tmp_path, monkeypatch):
    kit = Kit(clean_env(monkeypatch, tmp_path))
    parent = kit.agent('launcher')
    events = []
    monkeypatch.setattr(localstart.lp, 'admit', lambda *_: ('', ''))
    monkeypatch.setattr(localstart, '_mint', lambda *_: pytest.fail('minted top-level worker'))
    monkeypatch.setattr(localstart, '_record_model', lambda *_: None)
    monkeypatch.setattr(localstart, 'started_at', lambda *_: 42)
    def bind(ws, token, name):
        saved = localstart.la.load(ws, name)
        assert ws.auth(tokens.load(ws.base, saved.identity)).parent == ws.auth(token).id
        events.append('bound')
    def spawn(*_args, **_kwargs):
        assert events == ['bound']
        events.append('started')
        return SimpleNamespace(pid=123, log=tmp_path / 'worker.log')
    monkeypatch.setattr(localstart.device_agent, 'bind_owned_worker', bind, raising=False)
    pick = localstart.localmodel.Pick(ref='qwen', name='qwen')
    localstart.start(kit.ws, localstart.Ask(name='worker'), pick=pick, spawn=spawn, parent_token=parent)
    assert events == ['bound', 'started']
