"""Hook outage diagnostics and completion admission."""
import io
import json
from types import SimpleNamespace

import pytest

from ml_stack import harnesshook
from ml_stack.workspace.identity import BoardUnavailable, Denied


def unavailable(*args, **kwargs):
    raise BoardUnavailable('project board unavailable: connection refused')


def test_post_outage_does_not_reverse_a_completed_tool(monkeypatch, capsys):
    monkeypatch.setattr(harnesshook.harness_remote, 'context', unavailable)
    monkeypatch.setattr(harnesshook, '_reader_run', lambda *args, **kwargs:
                        SimpleNamespace(returncode=1, stdout='', stderr='connection refused'))
    out = io.StringIO()
    assert harnesshook.run(['post', '--agent', 'worker', '--root', '/project'], io.StringIO('{}'), out) == 0
    context = json.loads(out.getvalue())['hookSpecificOutput']['additionalContext']
    assert 'connection refused' in context
    assert 'connection refused' in capsys.readouterr().err


def test_stop_is_not_blocked_when_the_board_is_unavailable(monkeypatch):
    monkeypatch.setattr(harnesshook.harness_remote, 'context', unavailable)
    assert harnesshook.stop(harnesshook.Rail('plan-and-go', 'worker', ['/project'])) == {}


def test_stop_stays_blocked_by_a_real_refusal(monkeypatch):
    def refused(*args, **kwargs):
        raise Denied('completion refused')
    monkeypatch.setattr(harnesshook.harness_remote, 'context', refused)
    result = harnesshook.stop(harnesshook.Rail('plan-and-go', 'worker', ['/project']))
    assert result['decision'] == 'block' and 'completion refused' in result['reason']


def test_pre_outage_degrades_to_a_warning_and_a_real_denial_still_denies(monkeypatch):
    payload = {'tool_name': 'Write', 'tool_input': {'file_path': '/project/test.py'}, 'cwd': '/project'}
    rail = harnesshook.Rail('plan-and-go', 'worker', ['/project'])
    monkeypatch.setattr(harnesshook.harness_claims, 'conflict', unavailable)
    monkeypatch.setattr(harnesshook.harness_claims, 'reserve', unavailable)
    answer = harnesshook.pre(payload, rail)['hookSpecificOutput']
    assert answer['permissionDecision'] == 'allow'
    assert 'connection refused' in answer['permissionDecisionReason']

    def refused(*args, **kwargs):
        raise Denied('another agent owns this')
    monkeypatch.setattr(harnesshook.harness_claims, 'conflict', refused)
    answer = harnesshook.pre(payload, rail)['hookSpecificOutput']
    assert answer['permissionDecision'] == 'deny' and 'another agent owns this' in answer['permissionDecisionReason']


@pytest.mark.parametrize('secret', ['mlws1.worker.' + 'a' * 43, 'mlws1.worker.one/child.two.' + 'b' * 43,
                                   'mlsk1.' + 'c' * 43, 'sk-' + 'a' * 30,
                                   'https://owner:private-password@example.invalid',
                                   '-----BEGIN PRIVATE KEY-----hidden-----END PRIVATE KEY-----'])
def test_diagnostic_redacts_credentials(secret, monkeypatch):
    monkeypatch.setenv('ML_STACK_TEST_TOKEN', 'opaque-local-credential')
    result = harnesshook._diagnostic(Denied('cannot connect ' + secret + ' opaque-local-credential'))
    assert secret not in result and 'opaque-local-credential' not in result
    assert 'cannot connect' in result
    assert len(harnesshook._diagnostic(Denied('x' * 2000))) < 1100


@pytest.mark.parametrize('role', ['read-only', 'approve-first', 'plan-and-go'])
def test_launcher_keeps_normal_role_policy_without_admission(monkeypatch, tmp_path, role):
    def no_board_context(*args, **kwargs):
        pytest.fail('resource-free launcher requested mutation authority')
    monkeypatch.setattr(harnesshook.harness_remote, 'context', no_board_context)
    monkeypatch.setattr(harnesshook, '_ask', lambda *args, **kwargs: (False, 'test declined'))
    args = {'command': 'ml-stack --no-browser'}
    rail = harnesshook.Rail(role, 'worker', [str(tmp_path)])
    expected = harnesshook.decide(role, 'Bash', args, roots=rail.roots)
    result = harnesshook.pre({'tool_name': 'Bash', 'tool_input': args, 'cwd': str(tmp_path)}, rail)
    action = result['hookSpecificOutput']['permissionDecision']
    assert action == ('allow' if expected.action == 'allow' else 'deny')


def test_a_board_outage_never_waives_the_destructive_call_guard(monkeypatch, tmp_path):
    monkeypatch.setattr(harnesshook.harness_remote, 'context', unavailable)
    monkeypatch.setattr(harnesshook, '_ask', lambda *args, **kwargs: (False, 'test declined'))
    args = {'command': f'ml-stack --no-browser; rm -rf {tmp_path / "owned"}'}
    result = harnesshook.pre({'tool_name': 'Bash', 'tool_input': args, 'cwd': str(tmp_path)},
                            harnesshook.Rail('plan-and-go', 'worker', [str(tmp_path)]))
    assert result['hookSpecificOutput']['permissionDecision'] == 'deny'


@pytest.mark.parametrize('error', [Denied('board offline'), OSError('connection refused'),
                                  RuntimeError('daemon unavailable')])
def test_outer_post_failure_keeps_completed_tool_results(error, monkeypatch, capsys):
    def broken(_label, _rail):
        raise error
    monkeypatch.setattr(harnesshook, 'post', broken)
    assert harnesshook.run(['post'], io.StringIO('{}'), io.StringIO()) == 0
    diagnostic = capsys.readouterr().err
    assert 'notification unavailable' in diagnostic and str(error) in diagnostic


def test_unhandled_post_failure_is_nonblocking(monkeypatch, capsys):
    exits = []
    monkeypatch.setattr(harnesshook.sys, 'argv', ['harnesshook', 'post'])
    monkeypatch.setattr(harnesshook.os, '_exit', exits.append)
    harnesshook._block(Denied, Denied('daemon unavailable'), None)
    assert exits == [0]
    assert 'Denied: daemon unavailable' in capsys.readouterr().err


def test_post_checkpoint_outage_preserves_pending_message_alert(monkeypatch):
    monkeypatch.setattr(harnesshook.harness_remote, 'context', unavailable)
    seen = []

    def pending(label, rail=None):
        seen.append(label)
        return '1 waiting for you; run inbox\nworkspace notification unavailable: connection refused'

    monkeypatch.setattr(harnesshook, 'nudge', pending)
    out = io.StringIO()
    assert harnesshook.run(['post', '--agent', 'worker', '--root', '/project'], io.StringIO('{}'), out) == 0
    context = json.loads(out.getvalue())['hookSpecificOutput']['additionalContext']
    assert 'connection refused' in context
    assert '1 waiting for you; run inbox' in context
    assert seen == ['worker']


def test_post_lifecycle_failure_preserves_stop_message_alert(monkeypatch):
    monkeypatch.setattr(harnesshook, '_reader_run', lambda *args, **kwargs:
                        SimpleNamespace(returncode=1, stdout='urgent message waiting; run inbox',
                                        stderr='checkpoint missing shadow'))
    result = harnesshook.post('worker', harnesshook.Rail('plan-and-go', 'worker', ['/project']))
    context = result['hookSpecificOutput']['additionalContext']
    assert 'checkpoint missing shadow' in context
    assert 'urgent message waiting; run inbox' in context
