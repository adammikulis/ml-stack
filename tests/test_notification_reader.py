"""Session-scoped notification readers without enrollment."""
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from ml_stack import harnesshook
from ml_stack.workspace import notification_reader as reader
from ml_stack.workspace.identity import Denied


def record(session):
    slot = hashlib.sha256(session.encode()).hexdigest()[:32]
    return slot, {'host': 'https://board.invalid', 'project_id': 'a' * 32,
                  'local_agent': 'codex', 'session': slot, 'agent': f'codex-{slot}'}


@pytest.fixture
def saved(monkeypatch, tmp_path):
    first, one = record('root-one')
    second, two = record('root-two')
    configured = {'host': one['host'], 'project_id': one['project_id'], 'agent': 'codex',
                  'sessions': {first: one, second: two}}
    monkeypatch.setattr(reader.project_connection, '_saved', lambda: {str(tmp_path): configured})
    return tmp_path, configured, one, two


def test_hook_session_selects_each_root_without_environment_fallback(saved, monkeypatch):
    root, _, one, two = saved
    monkeypatch.setenv('CODEX_THREAD_ID', 'unrelated-shell')
    assert reader.binding('codex', root, 'root-one') == one
    assert reader.binding('codex', root, 'root-two') == two


def test_unbound_child_never_reads_default_or_parent_inbox(saved):
    root, _, _, _ = saved
    with pytest.raises(Denied, match='no authenticated saved binding'):
        reader.binding('codex', root, 'child-thread')
    with pytest.raises(Denied, match='valid hook session'):
        reader.binding('codex', root, '')


def test_explicit_launcher_parent_requires_saved_binding(saved):
    root, configured, one, _ = saved
    configured['agent'] = one['agent']
    assert reader.binding(one['agent'], root, 'child-thread') is configured
    with pytest.raises(Denied, match='differs'):
        reader.binding('other-parent', root, 'child-thread')


def test_legacy_root_requires_its_exact_saved_principal_without_thread_namespace(saved):
    root, configured, _, _ = saved
    configured.pop('sessions')
    assert reader.binding('codex', root, 'root-one') is configured
    configured['agent'] = 'other-root'
    configured['local_agent'] = 'codex'
    with pytest.raises(Denied, match='differs'):
        reader.binding('codex', root, 'root-one')


def test_local_launcher_alias_resolves_only_its_exact_saved_actor(saved):
    root, configured, _, _ = saved
    configured['local_agent'] = 'fixture-code'
    configured['agent'] = 'fixture-project-actor'
    assert reader.binding('fixture-code', root, '') is configured
    with pytest.raises(Denied, match='differs'):
        reader.binding('foreign-code', root, '')


def test_authenticated_context_reuses_reader_without_discovery_or_reauthentication(saved, monkeypatch):
    root, _, one, _ = saved
    calls = []
    remote = SimpleNamespace(base=root, project_id=one['project_id'], host=one['host'],
                             call=lambda operation, token: calls.append(operation) or 'one unread')
    who = SimpleNamespace(id=one['agent'], role='agent')
    monkeypatch.setattr(reader.project_connection, 'RemoteWorkspace',
                        lambda *a, **kw: pytest.fail('repeated discovery'))
    monkeypatch.setattr(reader.tokens, 'load', lambda base, actor: 'existing-fixture')
    assert reader.read('codex', root, 'root-one', canonical=(remote, who)) == 'one unread'
    assert calls == ['nudge']
    who.id = 'foreign-actor'
    with pytest.raises(Denied, match='does not match'):
        reader.read('codex', root, 'root-one', canonical=(remote, who))


def test_saved_slot_authority_tampering_is_refused(saved):
    root, _, one, _ = saved
    one['project_id'] = 'b' * 32
    with pytest.raises(Denied, match='no authenticated saved binding'):
        reader.binding('codex', root, 'root-one')


def test_reader_uses_existing_token_and_authenticates_before_nudge(saved, monkeypatch):
    root, _, one, _ = saved
    calls = []
    class Remote:
        base = Path('/fixture-capabilities')
        def __init__(self, *args, **kwargs):
            pass
        def token(self, **kwargs):
            pytest.fail('notification attempted credential enrollment')
        def call(self, operation, token):
            calls.append((operation, token))
            return ({'id': one['agent'], 'role': 'agent', 'project': {'key': one['project_id']}}
                    if operation == 'whoami' else 'one unread')
    monkeypatch.setattr(reader.project_connection, 'RemoteWorkspace', Remote)
    monkeypatch.setattr(reader.tokens, 'load', lambda base, actor: actor + '-existing')
    assert reader.read('codex', root, 'root-one') == 'one unread'
    assert [operation for operation, _ in calls] == ['whoami', 'nudge']


def test_revoked_or_wrong_capability_never_reads(saved, monkeypatch):
    root, _, one, _ = saved
    def call(operation, token):
        assert operation == 'whoami'
        return {'id': 'other', 'role': 'agent', 'project': {'key': one['project_id']}}
    monkeypatch.setattr(reader.project_connection, 'RemoteWorkspace',
                        lambda *a, **kw: SimpleNamespace(base=root, call=call))
    monkeypatch.setattr(reader.tokens, 'load', lambda *a: 'existing-fixture-capability')
    with pytest.raises(Denied, match='capability does not match'):
        reader.read('codex', root, 'root-one')


def test_nudge_subprocess_is_bounded_and_session_explicit(monkeypatch):
    captured = []
    def run(command, **kwargs):
        captured.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout='one unread', stderr='')
    monkeypatch.setattr(harnesshook, '_reader_run', run)
    rail = harnesshook.Rail('read-only', 'codex', ['/fixture'], session_id='root-one')
    assert harnesshook.nudge('codex', rail) == 'one unread'
    command, options = captured[0]
    assert command[-3:] == ['codex', '/fixture', 'root-one']
    assert options['timeout'] == harnesshook.NUDGE_S


def test_unbound_notification_does_not_block_completed_tools(saved, monkeypatch, capsys):
    import io
    monkeypatch.setattr(harnesshook.harness_remote, 'context', lambda *a, **kw: None)
    monkeypatch.setattr(harnesshook, 'Workspace', lambda: SimpleNamespace(base=saved[0], auth=lambda token: SimpleNamespace(id='codex')))
    monkeypatch.setattr(harnesshook.tokens, 'load', lambda *a: 'fixture-existing')
    monkeypatch.setattr(harnesshook.worktree_lifecycle, 'checkpoint', lambda *a: None)
    monkeypatch.setattr(harnesshook, '_reader_run', lambda *a, **kw:
                        SimpleNamespace(returncode=1, stdout='', stderr='notification session has no authenticated saved binding'))
    assert harnesshook.run(['post', '--label', 'codex'], io.StringIO('{"session_id":"child-thread"}'), io.StringIO()) == 0
    assert 'no authenticated saved binding' in capsys.readouterr().err


def test_explicit_parent_hook_preserves_launcher_root(monkeypatch):
    import io
    captured = []
    monkeypatch.setattr(harnesshook.harness_remote, 'context', lambda *a, **kw: (_ for _ in ()).throw(Denied('offline')))
    monkeypatch.setattr(harnesshook, 'nudge', lambda label, rail: captured.append((label, rail.roots)) or '')
    assert harnesshook.run(['post', '--label', 'project-parent', '--root', '/launcher-root'],
                           io.StringIO('{"cwd":"/unrelated"}'), io.StringIO()) == 0
    assert captured == [('project-parent', ['/launcher-root'])]


def test_checkout_authority_mismatch_is_refused(saved):
    import json
    root, _, _, _ = saved
    (root / '.ml-stack-project.json').write_text(json.dumps(
        {'kind': 'project-checkout', 'project_id': 'b' * 32, 'authority': {}}))
    with pytest.raises(Denied, match='authority disagree'):
        reader.binding('codex', root, 'root-one')


def test_sibling_project_identity_mismatch_is_refused(saved, monkeypatch):
    root, _, _, _ = saved
    sibling = root.parent / 'other-sibling'
    sibling.mkdir()
    monkeypatch.setattr(reader.worktreerules, 'checkouts', lambda cwd: (sibling, root))
    monkeypatch.setattr(reader.project_connection.projects, 'identity', lambda cwd: 'b' * 32)
    with pytest.raises(Denied, match='differs'):
        reader.binding('codex', sibling, 'root-one')
