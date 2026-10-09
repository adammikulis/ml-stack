"""Completed worker executions survive Board transport and process restarts."""

import threading
from types import SimpleNamespace

import pytest
from test_workspace_remote import PROJECT, joined

from ml_stack.http import ServerError
from ml_stack.workspace import localagent, localloop, worker_completion, worker_reconnect
from ml_stack.workspace.identity import Denied
from ml_stack.workspace.project_connection import BoardWorkspace

pytest_plugins = ['test_workspace_remote']


class Transport:
    def __init__(self, host, identity):
        self.host, self.identity = host, identity
        self.current = SimpleNamespace(host='https://board.example.invalid', project_id=PROJECT)
        self.lost_reply = False
        self.outage = False
        self.calls = []

    def call(self, operation, token, *args, **kwargs):
        self.calls.append(operation)
        if self.outage:
            self.outage = False
            raise ConnectionResetError('fixture disconnected')
        code, reply = self.host.answer(PROJECT, 'board', {'agent_token': token,
                                  'operation': operation, 'args': list(args), 'kwargs': kwargs})
        if code != 200:
            error = Denied(reply.get('error', 'refused'))
            error.__cause__ = ServerError('fixture refusal', status=code)
            raise error
        if operation == 'send' and self.lost_reply:
            self.lost_reply = False
            raise ConnectionResetError('fixture lost the committed reply')
        return reply['result']


class Worker(BoardWorkspace):
    def __init__(self, base, transport, actor):
        super().__init__(transport, actor['token'])
        base.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.base, self.worker_identity = base, actor['id']

    def worker_token(self, agent):
        return self.token


@pytest.fixture
def worker(host, tmp_path, monkeypatch):
    actor, lead = joined(host, name='worker'), joined(host, name='lead')
    transport = Transport(host, actor['id'])
    made = Worker(tmp_path / 'local', transport, actor)
    code, _sent = host.answer(PROJECT, 'board', {'agent_token': lead['token'], 'operation': 'send',
                   'args': ['worker', 'task', 'perform the fixture'], 'kwargs': {}})
    assert code == 200
    row = made.inbox(made.token, raw=True, limit=1)[0]
    monkeypatch.setattr(localloop.la, 'obeys', lambda *args: 'authorized fixture lead')
    monkeypatch.setattr(worker_reconnect, 'POLL_S', 0)
    return made, transport, row


def loop_for(ws, executed):
    agent = localagent.Agent('local-worker', 'fixture-model', identity=ws.worker_identity)
    settings = localloop.Settings(execute=lambda *args: executed.append('performed') or ('answer', 'done', 1))
    status = SimpleNamespace(update=lambda **fields: None)
    return localloop.Loop(ws, agent, localloop.Held(None, {'id': 'held-model'}), settings,
                         (lambda: False, status))


def test_lost_committed_reply_is_recovered_without_duplicate_execution_or_message(worker):
    ws, transport, row = worker
    executed = []
    loop = loop_for(ws, executed)
    transport.lost_reply = True
    loop.handle(row)
    assert executed == ['performed']
    assert len(ws.outbox(ws.token, limit=100)) == 1
    assert transport.calls.count('send') == 1
    restarted = loop_for(ws, executed)
    restarted.handle(row)
    assert executed == ['performed'] and transport.calls.count('send') == 1
    ws.ack(ws.token, row['seq'])
    restarted.completion.acknowledged(row)
    assert restarted.completion.record() == {}


def test_interrupted_execution_is_not_repeated(worker):
    ws, _, row = worker
    executed = []
    loop = loop_for(ws, executed)
    loop.completion.begin(row)
    restarted = loop_for(ws, executed)
    with pytest.raises(worker_completion.Interrupted, match='side effects require review'):
        restarted.handle(row)
    assert executed == []


def test_completed_reply_cannot_reinstate_a_revoked_identity(worker):
    ws, transport, row = worker
    loop = loop_for(ws, [])
    loop.completion.begin(row)
    loop.completion.finish(row, 'answer', 'done', 1)
    transport.host.workspace(PROJECT).registry.revoke_self(ws.token, lambda who: None)
    with pytest.raises(Denied, match='revoked'):
        loop.completion.deliver(ws, ws.token, row)
    assert transport.calls.count('send') == 1


def test_unverifiable_lost_reply_fails_closed_instead_of_resending(worker, monkeypatch):
    ws, transport, row = worker
    loop = loop_for(ws, [])
    loop.completion.begin(row)
    saved = loop.completion.finish(row, 'answer', 'done', 1)
    loop.completion.record({**saved, 'state': 'sending'})
    monkeypatch.setattr(ws, 'outbox', lambda *args, **kwargs: [
        {'from': 'worker', 'reply_to': 999, 'subject': 'unrelated'} for _ in range(100)])
    with pytest.raises(worker_completion.Interrupted, match='bounded outbox'):
        loop.completion.deliver(ws, ws.token, row)
    assert transport.calls.count('send') == 0


def test_read_outage_retains_model_lease_until_worker_is_cancelled(worker, monkeypatch):
    ws, transport, _ = worker
    agent = localagent.Agent('local-worker', 'fixture-model', identity=ws.worker_identity)
    monkeypatch.setattr(localloop.la, 'load', lambda *args: agent)
    cancelled, released = threading.Event(), []
    transport.outage = True
    def execute(*args):
        assert released == []
        cancelled.set()
        return 'answer', 'done', 1
    settings = localloop.Settings(cancel=cancelled, execute=execute,
                                 serve=lambda actor: localloop.Held(None, {'id': 'held-model'},
                                                                   lambda: released.append('released')))
    assert localloop.run(ws, agent.name, settings) == 0
    assert released == ['released']
    cancelled.clear()
    assert len(ws.outbox(ws.token, limit=100)) == 1



def test_uncommitted_ambiguous_send_is_not_retried_or_reexecuted(worker, monkeypatch):
    ws, transport, row = worker
    executed = []
    loop = loop_for(ws, executed)
    original = transport.call
    def lost(operation, token, *args, **kwargs):
        if operation == 'send':
            transport.calls.append(operation)
            raise ConnectionResetError('fixture disconnected before its outcome was known')
        return original(operation, token, *args, **kwargs)
    monkeypatch.setattr(transport, 'call', lost)
    with pytest.raises(worker_completion.Interrupted, match='outcome is unresolved'):
        loop.handle(row)
    assert executed == ['performed'] and transport.calls.count('send') == 1
    with pytest.raises(worker_completion.Interrupted, match='outcome is unresolved'):
        loop.handle(row)
    assert executed == ['performed'] and transport.calls.count('send') == 1


def test_definite_draining_response_resumes_delivery_without_reexecution(worker, monkeypatch):
    ws, transport, row = worker
    loop, executed = None, []
    loop = loop_for(ws, executed)
    original, draining = transport.call, [True]
    def call(operation, token, *args, **kwargs):
        if operation == 'send' and draining[0]:
            draining[0] = False
            transport.calls.append(operation)
            error = Denied('fixture daemon draining before admission')
            error.__cause__ = ServerError('fixture draining', status=503)
            raise error
        return original(operation, token, *args, **kwargs)
    monkeypatch.setattr(transport, 'call', call)
    loop.handle(row)
    assert executed == ['performed'] and transport.calls.count('send') == 2
    assert len(ws.outbox(ws.token, limit=100)) == 1


def test_checkpoint_storage_failure_never_starts_execution(worker, monkeypatch):
    ws, _, row = worker
    executed = []
    loop = loop_for(ws, executed)
    def unavailable(*args, **kwargs):
        raise RuntimeError('fixture checkpoint disk unavailable')
    monkeypatch.setattr(worker_completion, 'GraphStore', unavailable)
    with pytest.raises(worker_completion.Interrupted, match='storage is unavailable'):
        loop.handle(row)
    assert executed == []


def test_checkpoint_identity_uses_clear_body_instead_of_rendered_attribution(worker):
    ws, _, row = worker
    loop = loop_for(ws, [])
    loop.handle(row)
    changed = {**row, 'text': 'updated attribution header; same clear message'}
    assert loop.completion.begin(changed)['state'] == 'sent'



@pytest.mark.parametrize('suffix', ['.wal', '.shadow', '.tmp'])
def test_completion_refuses_redirected_graph_companions_before_open(worker, tmp_path, monkeypatch, suffix):
    ws, _, _ = worker
    loop = loop_for(ws, [])
    outside = tmp_path / 'outside'
    outside.write_text('protected fixture content')
    companion = loop.completion.path.with_name(loop.completion.path.name + suffix)
    companion.symlink_to(outside)
    monkeypatch.setattr(worker_completion, 'GraphStore', lambda *args, **kwargs: pytest.fail('unsafe graph open'))
    with pytest.raises(Denied, match='symlink'):
        loop.completion.record()
    assert outside.read_text() == 'protected fixture content'


def test_completion_refuses_public_storage_parent_before_open(worker, monkeypatch):
    ws, _, _ = worker
    loop = loop_for(ws, [])
    loop.completion.parents[1].chmod(0o755)
    monkeypatch.setattr(worker_completion, 'GraphStore', lambda *args, **kwargs: pytest.fail('unsafe graph open'))
    with pytest.raises(Denied, match='mode'):
        loop.completion.record()
