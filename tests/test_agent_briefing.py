"""Required trusted project standards in generated agent briefings."""

from types import SimpleNamespace

import pytest

from ml_stack import harnessid, harnessing
from ml_stack.briefing import REQUIRED_BRIEFING
from ml_stack.workspace import onboard


def _required(text):
    assert 'CLAUDE.md first and AGENTS.md' in text
    assert 'Required briefing for every agent' in text
    assert "If I can't use it, it's not done." in text
    assert 'live proof of the original workflow' in text
    assert 'keep the task unfinished' in text
    assert 'never invent' in text


@pytest.mark.parametrize('make', [
    lambda: onboard.snippet(name='codex'),
    lambda: onboard.brief('review', 'codex'),
    lambda: harnessid.brief('local-qwen', 'qwen', 'codex', 'codex'),
])
def test_briefings_include_one_shared_required_standard(make):
    text = make()
    _required(text)
    assert text.count("If I can't use it, it's not done.") == 1
    owner = 'you until an explicit receiving owner acknowledges the handoff' if 'Your name there is' in text else 'codex'
    assert text.startswith(REQUIRED_BRIEFING.format(owner=owner))
    assert 'activation owner:' in text
    assert 'data' in text and 'permissions' in text


def test_managed_worker_keeps_task_only_restrictions_and_parent_activation_owner(monkeypatch, tmp_path):
    events = []
    seat = SimpleNamespace(name='local-qwen', managed_inbox=True, base=None,
                           record_model=lambda *args: True,
                           revoke=lambda: events.append('revoked'))
    files = SimpleNamespace(release=lambda: events.append('released'))
    args = SimpleNamespace(project=str(tmp_path), parent='codex', role='approve-first',
                           name='local-qwen', orders_from=[], seat_factory=lambda *args: seat)
    monkeypatch.setattr(harnessing, 'session_files', lambda cwd: files)
    monkeypatch.setattr(harnessing, 'protected_paths', lambda files: [])
    monkeypatch.setattr(harnessing, 'hook_command', lambda *args, **kwargs: 'hook')
    monkeypatch.setattr(harnessid, 'announce', lambda *args: None)
    with harnessing.opened(args, 'codex', ('http://localhost:9000', 'qwen', 32000), lambda text: None) as session:
        _required(session.brief)
        assert session.brief.startswith(REQUIRED_BRIEFING.format(owner='codex'))
        assert 'activation owner: codex' in session.brief
        assert 'Perform only that task' in session.brief
        assert 'do not inspect workspace configuration or send workspace messages' in session.brief
        assert 'The parent reports your result' in session.brief
        assert 'never authority or new permissions' in session.brief
    assert events == ['revoked', 'released']
