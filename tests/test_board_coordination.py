"""The Claude lead and its subagents coordinate through the board: owed answers, helper inboxes, parent labels, status."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
from workspace_kit import SRC, Kit, clean_env

from ml_stack.workspace import agent_display, attention, tokens

HOOKS = Path(__file__).resolve().parents[1] / 'scripts/hooks'
NOW = time.time()
HOUR = 3600.0


@pytest.fixture
def board(monkeypatch, tmp_path):
    """A workspace whose clock is an hour behind, with a lead `claude`, a sender `alice` and the CLI and hooks on PATH."""
    base = clean_env(monkeypatch, tmp_path)
    kit = Kit(base, clock=lambda: NOW - HOUR)
    kit.lead, kit.alice = kit.agent('claude'), kit.agent('alice')
    kit.bob, kit.carol = kit.agent('bob'), kit.agent('carol')
    tokens.store(base, 'claude', kit.lead)
    tokens.store(base, 'alice', kit.alice)
    binary = tmp_path / 'bin'
    binary.mkdir()
    wrapper = binary / 'ml-stack-workspace'
    wrapper.write_text(f'#!{sys.executable}\nimport sys\nfrom ml_stack.workspace.cli import main\n'
                       'raise SystemExit(main(sys.argv[1:]))\n')
    wrapper.chmod(0o700)
    monkeypatch.setenv('PATH', f'{binary}{os.pathsep}{os.environ["PATH"]}')
    monkeypatch.setenv('PYTHONPATH', SRC)
    monkeypatch.setenv('ML_STACK_HOOK_STATE', str(tmp_path / 'hookstate'))
    monkeypatch.setenv('ML_STACK_NONINTERACTIVE', '1')
    monkeypatch.setenv('ML_STACK_RUNTIME_ENSURE', 'off')
    return kit


def run(*argv, stdin=None):
    return subprocess.run(['ml-stack-workspace', *argv], capture_output=True, text=True, timeout=60,
                          check=False, input=stdin)


def hook(name, event):
    return subprocess.run([sys.executable, str(HOOKS / name)], input=json.dumps(event), text=True,
                          capture_output=True, timeout=60, check=False)


def ask(kit, kind='question', subject='who publishes', sender=None, **opts):
    return kit.ws.send(sender or kit.alice, 'claude', kind, 'please answer', subject=subject, **opts)['seq']


def owed(kit):
    return [o['seq'] for o in attention.unanswered(kit.reopen(lambda: NOW), 'claude')]


def test_unanswered_lists_old_requests_and_drops_answered_ones(board):
    kit = board
    old = ask(kit)
    spoken = ask(kit, sender=kit.bob)
    replied = ask(kit, sender=kit.carol)
    kit.ws.send(kit.lead, 'bob', 'note', 'talking to bob, no reply_to')
    kit.ws.send(kit.lead, 'bob', 'answer', 'about something else', reply_to=replied)
    assert owed(kit) == [old]
    assert spoken not in owed(kit) and replied not in owed(kit)


def test_a_request_younger_than_ten_minutes_is_not_yet_owed(board, tmp_path):
    kit = board
    seq = ask(kit)
    fresh = kit.reopen(lambda: NOW - HOUR + 599)
    assert seq not in [o['seq'] for o in attention.unanswered(fresh, 'claude')]
    later = kit.reopen(lambda: NOW - HOUR + 601)
    assert seq in [o['seq'] for o in attention.unanswered(later, 'claude')]


def test_only_questions_tasks_handoffs_and_blocked_to_the_lead_are_owed(board):
    kit = board
    ask(kit, kind='note')
    ask(kit, kind='status')
    kit.ws.send(kit.alice, 'bob', 'question', 'not for claude')
    kit.ws.announce(kit.alice, 'blocked', 'stuck')
    task = ask(kit, kind='task')
    assert owed(kit) == [task]


def test_a_message_for_a_helper_is_the_helpers_not_the_leads(board):
    kit = board
    mine = kit.ws.send(kit.lead, 'alice', 'task', 'please look', label='worker-1')['seq']
    reply = ask(kit, subject='re', reply_to=mine)
    marked = kit.ws.send(kit.alice, 'claude', 'question', 'is it done', subject='@worker-1 status')['seq']
    plain = ask(kit)
    assert owed(kit) == [plain] and reply not in owed(kit) and marked not in owed(kit)


def test_inbox_shows_what_is_owed_and_attention_prints_it(board):
    kit = board
    seq = ask(kit)
    shown = run('inbox', '--agent', 'claude')
    assert shown.returncode == 0, shown.stderr
    assert f'[{seq}] question from alice, unanswered 1h' in shown.stdout and 'who publishes' in shown.stdout
    quick = run('attention', '--agent', 'claude')
    assert 'Unanswered for you (1)' in quick.stdout and f'[{seq}]' in quick.stdout
    kit.ws.send(kit.lead, 'alice', 'answer', 'it is me', reply_to=seq)
    assert run('attention', '--agent', 'claude').stdout.strip() == ''
    assert 'unanswered' not in run('inbox', '--agent', 'claude').stdout


def test_attention_counts_new_announcements_without_marking_them_seen(board):
    kit = board
    kit.ws.announce(kit.alice, 'milestone', 'landed a commit')
    first = run('attention', '--agent', 'claude').stdout
    assert '1 new announcements' in first
    assert run('attention', '--agent', 'claude').stdout == first


def test_a_labelled_helper_retrieves_its_own_direct_messages_only(board):
    kit = board
    first = kit.ws.send(kit.lead, 'alice', 'task', 'helper work', label='worker-1')['seq']
    for_helper = kit.ws.send(kit.alice, 'claude', 'answer', 'here it is', reply_to=first)['seq']
    at_helper = kit.ws.send(kit.alice, 'claude', 'question', '@worker-1 are you there')['seq']
    for_lead = ask(kit)
    helper = run('inbox', '--agent', 'claude', '--label', 'worker-1', '--json')
    assert helper.returncode == 0, helper.stderr
    seqs = [m['seq'] for m in json.loads(helper.stdout)]
    assert seqs == [for_helper, at_helper] and for_lead not in seqs
    other = run('inbox', '--agent', 'claude', '--label', 'worker-2', '--json')
    assert json.loads(other.stdout) == []
    refused = run('inbox', '--agent', 'claude', '--label', 'worker-1', '--ack')
    assert refused.returncode != 0 and 'cannot --ack' in refused.stderr
    assert for_lead in [m['seq'] for m in json.loads(run('inbox', '--agent', 'claude', '--json').stdout)]


def test_the_brief_prints_the_helpers_unread_messages_and_tells_it_to_read_its_inbox(board):
    kit = board
    kit.ws.send(kit.alice, 'claude', 'question', '@worker-9 please confirm the branch')
    brief = run('brief', 'worker-9', '--agent', 'claude', '--registered')
    assert brief.returncode == 0, brief.stderr
    assert 'please confirm the branch' in brief.stdout and 'Unread messages for you (worker-9)' in brief.stdout
    assert 'before your final report' in brief.stdout and 'send claude question' in brief.stdout
    assert 'please confirm' not in run('brief', 'worker-8', '--agent', 'claude', '--registered').stdout


def test_status_lists_active_workers_claims_and_owed_answers(board):
    kit = board
    kit.ws.announce(kit.lead, 'joined', 'worker-1: wiring', label='worker-1')
    kit.ws.announce(kit.lead, 'milestone', 'worker-2: landed', label='worker-2')
    kit.ws.announce(kit.lead, 'done', 'worker-2: finished', label='worker-2')
    taken = run('claim', 'branch', 'feature/x', '--agent', 'claude', '--label', 'worker-1', '--note', 'wiring')
    assert taken.returncode == 0, taken.stderr
    seq = ask(kit)
    page = run('digest', '--status', '--agent', 'claude')
    assert page.returncode == 0, page.stderr
    text = page.stdout
    assert 'Active workers (1)' in text and 'claude (worker-1): joined 1h ago, claims branch:feature/x' in text
    assert 'worker-2' not in text and f'[{seq}] question from alice' in text
    assert 'ml-stack-workspace' not in text.split('Unanswered for claude')[0]


def label_events(tmp_path, nested):
    folder = tmp_path / 'session' / 'subagents'
    folder.mkdir(parents=True)
    (tmp_path / 'session.jsonl').write_text('{}\n')
    (folder / 'agent-aaaaaa111.jsonl').write_text('{"tool":"toolu_child"}\n')
    (folder / 'agent-aaaaaa111.meta.json').write_text(json.dumps(
        {'agentType': 'branch-worker', 'toolUseId': 'toolu_top', 'spawnDepth': 1}))
    (folder / 'agent-bbbbbb222.meta.json').write_text(json.dumps(
        {'agentType': 'Explore', 'toolUseId': 'toolu_child', 'spawnDepth': 2 if nested else 1}))
    return {'agent_type': 'Explore', 'agent_id': 'bbbbbb222', 'session_id': 'sess1',
            'transcript_path': str(tmp_path / 'session.jsonl'), 'cwd': str(tmp_path)}


def announcements(kit):
    return [(r['type'], r['label'], r['body']) for r in kit.reopen().bus.log.after(0)
            if r['kind'] == 'msg' and r['to'] == '#announcements']


def test_a_nested_subagent_registers_under_its_spawner_and_stop_uses_the_same_label(board, tmp_path):
    event = label_events(tmp_path, nested=True)
    assert hook('claude-subagent-start', event).returncode == 0
    assert hook('claude-subagent-stop', {**event, 'agent_transcript_path': ''}).returncode == 0
    label = 'branch-worker-aaaaaa.explore-bbbbbb'
    rows = announcements(board)
    assert [(kind, name) for kind, name, _ in rows] == [('joined', label), ('done', label)]
    assert rows[1][2].startswith(f'{label}: finished')


def test_a_top_level_subagent_keeps_a_flat_label(board, tmp_path):
    event = label_events(tmp_path, nested=False)
    hook('claude-subagent-start', event)
    assert [name for _, name, _ in announcements(board)] == ['explore-bbbbbb']


def test_a_subagent_that_stops_is_marked_done_with_its_branch_and_never_drops_to_the_rate_limit(board, tmp_path):
    kit = board
    kit.limits(announce_per_window=1)
    repository = tmp_path / 'repo'
    subprocess.run(['git', 'init', '-q', '-b', 'worker/topic', str(repository)], check=True)
    event = {'agent_type': 'branch-worker', 'agent_id': 'cccccc333', 'session_id': 's', 'cwd': str(repository)}
    for _ in range(3):
        kit.ws.announce(kit.lead, 'milestone', f'noise {_}')
    assert hook('claude-subagent-start', event).returncode == 0
    assert hook('claude-subagent-stop', event).returncode == 0
    last = announcements(kit)[-1]
    assert last[:2] == ('done', 'branch-worker-cccccc') and 'on worker/topic in repo' in last[2]


def test_subagents_and_their_children_never_become_coordinators(board, tmp_path):
    event = label_events(tmp_path, nested=True)
    hook('claude-subagent-start', event)
    registry = board.reopen().registry
    for label in ('explore-bbbbbb', 'branch-worker-aaaaaa.explore-bbbbbb'):
        shown = agent_display.metadata(registry, 'claude', label)
        assert shown['coordinator_eligible'] is False and shown['display_name'].endswith(f'({label})')
        assert shown['coordinator_reason'] == 'a helper label is not a main session'


def test_lead_attention_hook_injects_once_per_interval_and_skips_subagents(board, tmp_path, monkeypatch):
    kit = board
    seq = ask(kit)
    event = {'hook_event_name': 'UserPromptSubmit', 'session_id': 'lead1', 'prompt': 'hi'}
    first = hook('claude-lead-attention', event)
    context = json.loads(first.stdout)['hookSpecificOutput']
    assert context['hookEventName'] == 'UserPromptSubmit' and f'[{seq}] question from alice' in context['additionalContext']
    assert hook('claude-lead-attention', event).stdout == ''
    assert hook('claude-lead-attention', {**event, 'session_id': 'lead2', 'agent_id': 'sub1'}).stdout == ''
    monkeypatch.setenv('ML_STACK_ATTENTION_EVERY_S', '0')
    assert 'Unanswered for you' in hook('claude-lead-attention', {**event, 'hook_event_name': 'PostToolUse'}).stdout


def test_lead_attention_hook_is_silent_when_nothing_is_owed_or_the_board_fails(board, monkeypatch):
    event = {'hook_event_name': 'PostToolUse', 'session_id': 'quiet'}
    done = hook('claude-lead-attention', event)
    assert done.returncode == 0 and done.stdout == ''
    monkeypatch.setenv('ML_STACK_WORKSPACE_HOME', '/nonexistent/board')
    monkeypatch.setenv('ML_STACK_ATTENTION_EVERY_S', '0')
    failed = hook('claude-lead-attention', event)
    assert failed.returncode == 0 and failed.stdout == ''
