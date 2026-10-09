"""The Claude lead and its subagents coordinate through the board: owed answers, each subagent's own inbox and name, status."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
from workspace_kit import SRC, Kit, clean_env

from ml_stack.workspace import agent_display, attention, session_name, tokens

HOOKS = Path(__file__).resolve().parents[1] / 'scripts/hooks'
NOW = time.time()
LEAD = ''
HOUR = 3600.0


@pytest.fixture
def board(monkeypatch, tmp_path):
    """A workspace whose clock is an hour behind, with a lead session, a sender `alice` and the CLI and hooks on PATH."""
    base = clean_env(monkeypatch, tmp_path)
    kit = Kit(base, clock=lambda: NOW - HOUR)
    kit.lead_name = session_name.assign(base, 'claude-sonnet-5-5', 'claude-code', 'sess1')
    kit.lead, kit.alice = kit.agent(kit.lead_name), kit.agent('alice')
    kit.bob, kit.carol = kit.agent('bob'), kit.agent('carol')
    tokens.store(base, kit.lead_name, kit.lead)
    kit.ws.register_session(kit.lead, None, 'claude-code')
    kit.ws.set_model(kit.lead_name, 'claude-sonnet-5-5', terminal=(True, True), env={})
    tokens.store(base, 'alice', kit.alice)
    monkeypatch.setitem(globals(), 'LEAD', kit.lead_name)
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
    return kit.ws.send(sender or kit.alice, LEAD, kind, 'please answer', subject=subject, **opts)['seq']


def owed(kit):
    return [o['seq'] for o in attention.unanswered(kit.reopen(lambda: NOW), LEAD)]


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
    assert seq not in [o['seq'] for o in attention.unanswered(fresh, LEAD)]
    later = kit.reopen(lambda: NOW - HOUR + 601)
    assert seq in [o['seq'] for o in attention.unanswered(later, LEAD)]


def test_only_questions_tasks_handoffs_and_blocked_to_the_lead_are_owed(board):
    kit = board
    ask(kit, kind='note')
    ask(kit, kind='status')
    kit.ws.send(kit.alice, 'bob', 'question', 'not for claude')
    kit.ws.announce(kit.alice, 'blocked', 'stuck')
    task = ask(kit, kind='task')
    assert owed(kit) == [task]


def test_a_request_to_a_subagent_is_addressed_to_its_name_and_is_not_the_leads(board, tmp_path):
    kit = board
    child = start(kit, spawn_events(tmp_path, nested=False)['child'])
    kit.ws.send(kit.alice, child, 'question', 'is it done')
    plain = ask(kit)
    assert owed(kit) == [plain]


def test_inbox_shows_what_is_owed_and_attention_prints_it(board):
    kit = board
    seq = ask(kit)
    shown = run('inbox', '--agent', LEAD)
    assert shown.returncode == 0, shown.stderr
    assert f'[{seq}] question from alice, unanswered 1h' in shown.stdout and 'who publishes' in shown.stdout
    quick = run('attention', '--agent', LEAD)
    assert 'Unanswered for you (1)' in quick.stdout and f'[{seq}]' in quick.stdout
    kit.ws.send(kit.lead, 'alice', 'answer', 'it is me', reply_to=seq)
    assert run('attention', '--agent', LEAD).stdout.strip() == ''
    assert 'unanswered' not in run('inbox', '--agent', LEAD).stdout


def test_attention_counts_new_announcements_without_marking_them_seen(board):
    kit = board
    kit.ws.announce(kit.alice, 'milestone', 'landed a commit')
    first = run('attention', '--agent', LEAD).stdout
    assert '1 new announcements' in first
    assert run('attention', '--agent', LEAD).stdout == first


def test_a_subagent_reads_its_own_inbox_and_acks_only_that(board, tmp_path):
    kit = board
    child = start(kit, spawn_events(tmp_path, nested=False)['child'])
    for_child = kit.ws.send(kit.alice, child, 'question', '@ is anyone there')['seq']
    for_lead = ask(kit)
    helper = run('inbox', '--agent', child, '--json')
    assert helper.returncode == 0, helper.stderr
    assert [m['seq'] for m in json.loads(helper.stdout)] == [for_child]
    acked = run('inbox', '--agent', child, '--ack')
    assert acked.returncode == 0, acked.stderr
    assert json.loads(run('inbox', '--agent', child, '--json').stdout) == []
    assert for_lead in [m['seq'] for m in json.loads(run('inbox', '--agent', LEAD, '--json').stdout)]


def test_the_brief_prints_the_subagents_unread_messages_and_names_its_parent(board, tmp_path):
    kit = board
    child = start(kit, spawn_events(tmp_path, nested=False)['child'])
    kit.ws.send(kit.alice, child, 'question', 'please confirm the branch')
    brief = run('brief', '--agent', child, '--registered')
    assert brief.returncode == 0, brief.stderr
    assert 'please confirm the branch' in brief.stdout and 'Unread messages for you' in brief.stdout
    assert f'You are the subagent {child}, spawned by {kit.lead_name}' in brief.stdout
    assert 'before your final report' in brief.stdout and f'send {kit.lead_name} question' in brief.stdout
    refused = run('brief', '--agent', kit.lead_name, '--registered')
    assert refused.returncode != 0 and 'was not spawned' in refused.stderr


def test_status_lists_a_leads_active_subagents_claims_and_owed_answers(board, tmp_path):
    kit = board
    events = spawn_events(tmp_path, nested=False)
    first, second = start(kit, events['child']), start(kit, {**events['child'], 'agent_id': 'dddddd444'})
    kit.ws.announce(kit.lead, 'milestone', 'noise from the lead itself')
    child_token = tokens.load(kit.base, first)
    kit.ws.announce(child_token, 'milestone', 'landed')
    kit.ws.announce(tokens.load(kit.base, second), 'done', 'finished')
    taken = run('claim', 'branch', f'{first}/x', '--agent', first, '--note', 'wiring')
    assert taken.returncode == 0, taken.stderr
    seq = ask(kit)
    page = run('digest', '--status', '--agent', LEAD)
    assert page.returncode == 0, page.stderr
    text = page.stdout
    assert 'Active subagents (1)' in text and f'{first}: milestone 1h ago, claims branch:{first}/x' in text
    assert second not in text and f'[{seq}] question from alice' in text


def spawn_events(tmp_path, nested):
    """The SubagentStart events of a top-level subagent and, when ``nested``, one it spawned."""
    folder = tmp_path / 'session' / 'subagents'
    folder.mkdir(parents=True)
    (tmp_path / 'session.jsonl').write_text('{}\n')
    (folder / 'agent-aaaaaa111.jsonl').write_text('{"tool":"toolu_child"}\n')
    (folder / 'agent-aaaaaa111.meta.json').write_text(json.dumps(
        {'agentType': 'branch-worker', 'toolUseId': 'toolu_top', 'spawnDepth': 1}))
    (folder / 'agent-bbbbbb222.meta.json').write_text(json.dumps(
        {'agentType': 'Explore', 'toolUseId': 'toolu_child', 'spawnDepth': 2 if nested else 1}))
    common = {'session_id': 'sess1', 'transcript_path': str(tmp_path / 'session.jsonl'), 'cwd': str(tmp_path)}
    return {'child': {**common, 'agent_type': 'branch-worker', 'agent_id': 'aaaaaa111'},
            'grandchild': {**common, 'agent_type': 'Explore', 'agent_id': 'bbbbbb222'}}


def start(kit, event):
    """Run the real SubagentStart hook for ``event`` and return the unique name the board gave the subagent."""
    done = hook('claude-subagent-start', event)
    assert done.returncode == 0 and not done.stderr, done.stderr
    return session_name.lookup(kit.base, 'claude-code', event['agent_id'])


def announcements(kit):
    return [(r['type'], r['from'], r['body']) for r in kit.reopen().bus.log.after(0)
            if r['kind'] == 'msg' and r['to'] == '#announcements']


def test_a_nested_subagent_is_a_child_of_its_spawner_and_stop_speaks_as_itself(board, tmp_path):
    kit = board
    events = spawn_events(tmp_path, nested=True)
    first, second = start(kit, events['child']), start(kit, events['grandchild'])
    registry = kit.reopen().registry
    assert len({kit.lead_name, first, second}) == 3
    assert registry.info(first)['parent'] == kit.lead_name and registry.info(second)['parent'] == first
    assert hook('claude-subagent-stop', {**events['grandchild'], 'agent_transcript_path': ''}).returncode == 0
    rows = announcements(kit)
    assert [(kind, name) for kind, name, _ in rows] == [('joined', first), ('joined', second), ('done', second)]
    assert rows[2][2].startswith('finished')


def test_subagents_and_their_children_never_become_coordinators(board, tmp_path):
    kit = board
    events = spawn_events(tmp_path, nested=True)
    names = [start(kit, events['child']), start(kit, events['grandchild'])]
    registry = kit.reopen().registry
    assert agent_display.metadata(registry, kit.lead_name)['coordinator_eligible'] is True
    for name in names:
        shown = agent_display.metadata(registry, name)
        assert shown['coordinator_eligible'] is False and shown['session_kind'] == 'subagent'
        assert shown['coordinator_reason'] == 'not a main session'
        assert shown['spawned_by'] == registry.info(name)['parent']
    assert agent_display.spoken(agent_display.metadata(registry, names[1])) == f'{names[1]} (spawned by {names[0]})'


def test_a_subagent_that_stops_is_marked_done_with_its_branch_and_never_drops_to_the_rate_limit(board, tmp_path):
    kit = board
    kit.limits(announce_per_window=1)
    repository = tmp_path / 'repo'
    subprocess.run(['git', 'init', '-q', '-b', 'worker/topic', str(repository)], check=True)
    event = {'agent_type': 'branch-worker', 'agent_id': 'cccccc333', 'session_id': 'sess1', 'cwd': str(repository)}
    name = start(kit, event)
    token = tokens.load(kit.base, name)
    for number in range(4):
        kit.ws.announce(token, 'milestone', f'noise {number}')
    assert hook('claude-subagent-stop', event).returncode == 0
    last = announcements(kit)[-1]
    assert last[:2] == ('done', name) and 'on worker/topic in repo' in last[2]
    assert any(row['event'] == 'announce.over_quota' for row in kit.reopen().audit_log.rows())
    assert kit.reopen().registry.info(name)['revoked'] is True


def test_lead_attention_hook_injects_once_per_interval_and_skips_subagents(board, tmp_path, monkeypatch):
    kit = board
    seq = ask(kit)
    event = {'hook_event_name': 'UserPromptSubmit', 'session_id': 'sess1', 'prompt': 'hi'}
    first = hook('claude-lead-attention', event)
    context = json.loads(first.stdout)['hookSpecificOutput']
    assert context['hookEventName'] == 'UserPromptSubmit' and f'[{seq}] question from alice' in context['additionalContext']
    assert hook('claude-lead-attention', event).stdout == ''
    assert hook('claude-lead-attention', {**event, 'agent_id': 'sub1'}).stdout == ''
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
