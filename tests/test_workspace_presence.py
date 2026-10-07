"""Authenticated remote presence without durable read traffic."""

from concurrent.futures import ThreadPoolExecutor

import pytest
import test_workspace_remote as remote_tests


@pytest.fixture
def host(tmp_path):
    return remote_tests.host.__wrapped__(tmp_path)


def test_reads_refresh_presence_without_durable_writes(host):
    actor = remote_tests.joined(host)
    ws = host.workspace(remote_tests.PROJECT)
    before = {p: p.read_bytes() for p in ws.base.rglob('*') if p.is_file()}
    for operation in ('whoami', 'nudge'):
        assert remote_tests.call(host, actor, operation)[0] == 200
    after = {p: p.read_bytes() for p in ws.base.rglob('*') if p.is_file()}
    assert after == before
    status = host.status(remote_tests.PROJECT)
    agent = next(row for row in status['agents'] if row['id'] == actor['id'])
    assert agent['online'] and agent['last_seen'] > 0


def test_denied_and_invalid_calls_do_not_refresh_presence(host):
    actor = remote_tests.joined(host)
    host._presence.clear()
    assert remote_tests.call(host, actor, 'whoami', project=remote_tests.OTHER)[0] == 403
    assert remote_tests.call(host, actor, 'unavailable')[0] == 403
    assert remote_tests.call(host, actor, 'who', 'area')[0] == 400
    assert remote_tests.call(host, {'token': 'invalid'}, 'whoami')[0] == 403
    assert not host._presence
    assert not host.status(remote_tests.PROJECT)['agents'][0]['online']


def test_revoked_and_expired_actors_are_offline(host):
    actor = remote_tests.joined(host)
    ws = host.workspace(remote_tests.PROJECT)
    agents = ws.registry._load()
    agents[actor['id']]['expires'] = ws.clock() - 1
    ws.registry._save(agents)
    assert remote_tests.call(host, actor, 'whoami')[0] == 403
    assert not host.status(remote_tests.PROJECT)['agents'][0]['online']
    agents[actor['id']]['expires'] = ws.clock() + 100
    agents[actor['id']]['revoked'] = True
    ws.registry._save(agents)
    assert remote_tests.call(host, actor, 'nudge')[0] == 403
    assert not any(row['online'] for row in host.status(remote_tests.PROJECT)['agents'])


def test_dev_admission_is_required_before_presence(host):
    actor = remote_tests.joined(host)
    ws = host.workspace(remote_tests.PROJECT)
    agents = ws.registry._load()
    agents[actor['id']]['project'].update(cluster='development', cluster_id='selected')
    ws.registry._save(agents)
    host._presence.clear()
    body = {'agent_token': actor['token'], 'operation': 'whoami', 'args': [], 'kwargs': {}}
    assert host.answer(remote_tests.PROJECT, 'board', body)[0] == 403
    assert host.answer(remote_tests.PROJECT, 'board', body,
                       admission=('development', 'wrong', True))[0] == 403
    assert not host._presence
    assert host.answer(remote_tests.PROJECT, 'board', body,
                       admission=('development', 'selected', True))[0] == 200
    assert (remote_tests.PROJECT, actor['id']) in host._presence


def test_presence_is_project_scoped_expires_and_bounded(host):
    actor = remote_tests.joined(host)
    host._presence.clear()
    host._seen(remote_tests.OTHER, actor['id'], host.workspace(remote_tests.PROJECT).clock())
    assert not host.status(remote_tests.PROJECT)['agents'][0]['online']
    host._seen(remote_tests.PROJECT, actor['id'], 1)
    assert not host.status(remote_tests.PROJECT)['agents'][0]['online']
    host._presence.clear()
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(lambda i: host._seen(remote_tests.PROJECT, str(i), 100), range(4100)))
    assert len(host._presence) == 4096
    host._seen(remote_tests.PROJECT, 'fresh', 191)
    assert host._presence == {(remote_tests.PROJECT, 'fresh'): 191}


def test_mutations_keep_durable_audit(host):
    first = remote_tests.joined(host, name='first')
    remote_tests.joined(host, name='second')
    ws = host.workspace(remote_tests.PROJECT)
    before = list(ws.audit_log.rows())
    assert remote_tests.call(host, first, 'send', 'second', 'status', 'hello')[0] == 200
    after = list(ws.audit_log.rows())
    assert len(after) > len(before)
    assert any(row['event'] == 'remote.seen' for row in before)
    assert after[-1]['event'] != 'remote.seen'
