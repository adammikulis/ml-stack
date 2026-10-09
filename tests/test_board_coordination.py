"""The Claude lead and its subagents coordinate through the board: owed answers, each subagent's own inbox and name, status."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import pytest
from workspace_kit import SRC

from ml_stack.board import session as board_session
from ml_stack.workspace import agent_display, board_cli, limits

pytest_plugins = ["node_kit"]
HOOKS = Path(__file__).resolve().parents[1] / 'scripts/hooks'
HOUR = 3600.0


@pytest.fixture
def board(workspace_node, monkeypatch, tmp_path):
    """A node with a lead session (native id sess1), senders alice, bob and carol, and the CLI and hooks on PATH."""
    node = workspace_node
    node.lead = node.member('sess1')
    node.alice, node.bob, node.carol = (node.member(n) for n in ('alice', 'bob', 'carol'))
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
    return node


def run(name, *argv):
    return subprocess.run(['ml-stack-workspace', *argv, '--agent', name], capture_output=True, text=True, timeout=90,
                          check=False)


def hook(name, event):
    return subprocess.run([sys.executable, str(HOOKS / name)], input=json.dumps(event), text=True,
                          capture_output=True, timeout=90, check=False)


def ask(node, kind='question', subject='who publishes', sender=None):
    sent = node.session(sender or node.alice).post(node.lead.name, kind, 'please answer', subject=subject)
    return sent['id'].rsplit(':', 1)[-1]


def later(node, who, seconds):
    """The session of ``who`` as it would be ``seconds`` after now."""
    s = node.session(who)
    s.clock = lambda: time.time() + seconds
    return s


def owed(node, seconds=HOUR):
    return [o['seq'] for o in board_cli.unanswered(later(node, node.lead, seconds))]


def test_unanswered_lists_old_requests_and_drops_answered_ones(board):
    old = ask(board)
    spoken = ask(board, sender=board.bob)
    replied = ask(board, sender=board.carol)
    lead = board.session(board.lead)
    lead.post(board.bob.name, 'note', 'talking to bob, no reply_to')
    lead.post(board.carol.name, 'answer', 'about something else', reply_id=replied)
    assert owed(board) == [old]
    assert spoken not in owed(board) and replied not in owed(board)


def test_a_message_sent_before_the_request_does_not_answer_it(board):
    board.session(board.lead).post(board.alice.name, 'note', 'before she asked')
    time.sleep(0.01)
    seq = ask(board)
    assert owed(board) == [seq]


def test_a_request_younger_than_the_owners_threshold_is_not_yet_owed(board):
    seq = ask(board)
    assert seq not in owed(board, 599)
    assert seq in owed(board, 601)
    root = limits.root()
    limits.save(root, replace(limits.load(root), owed_after_s=7200.0))
    assert owed(board) == []


def test_only_questions_tasks_handoffs_and_blocked_to_the_lead_are_owed(board):
    ask(board, kind='note')
    ask(board, kind='status')
    board.session(board.alice).post(board.bob.name, 'question', 'not for claude')
    board.session(board.alice).post('#announcements', 'blocked', 'stuck', subject='blocked')
    task = ask(board, kind='task')
    assert owed(board) == [task]


def test_a_request_to_a_subagent_is_addressed_to_its_name_and_is_not_the_leads(board, tmp_path):
    child = start(board, spawn_events(tmp_path, nested=False)['child'])
    board.session(board.alice).post(child.name, 'question', 'is it done')
    plain = ask(board)
    assert owed(board) == [plain]


def test_inbox_shows_what_is_owed_and_attention_prints_it(board, capsys):
    seq = ask(board)
    args = argparse.Namespace(json=False, ack=False, children=False, all=False, limit=0)
    board_cli.inbox(args, later(board, board.lead, HOUR))
    shown = capsys.readouterr().out
    assert f'[{seq}] question from {board.alice.name}, unanswered 1h' in shown and 'who publishes' in shown
    assert 'Unanswered for you (1)' in board_cli.owed_text(later(board, board.lead, HOUR))
    board.session(board.lead).post(board.alice.name, 'answer', 'it is me', reply_id=seq)
    board_cli.inbox(args, later(board, board.lead, HOUR))
    assert 'unanswered' not in capsys.readouterr().out
    assert board_cli.owed_text(later(board, board.lead, HOUR)) == ''


def test_attention_counts_new_announcements_without_marking_them_seen(board):
    board.session(board.alice).post('#announcements', 'milestone', 'landed a commit', subject='milestone')
    first = run(board.lead.name, 'attention').stdout
    assert '1 new announcements' in first
    assert run(board.lead.name, 'attention').stdout == first
    run(board.lead.name, 'inbox', '--ack')
    assert run(board.lead.name, 'attention').stdout.strip() == ''


def test_a_subagent_reads_its_own_inbox_and_acks_only_that(board, tmp_path):
    child = start(board, spawn_events(tmp_path, nested=False)['child'])
    for_child = board.session(board.alice).post(child.name, 'question', '@ is anyone there')['id'].rsplit(':', 1)[-1]
    for_lead = ask(board)
    helper = run(child.name, 'inbox', '--json')
    assert helper.returncode == 0, helper.stderr
    assert [m['seq'] for m in json.loads(helper.stdout)] == [for_child]
    acked = run(child.name, 'inbox', '--ack')
    assert acked.returncode == 0, acked.stderr
    assert json.loads(run(child.name, 'inbox', '--json').stdout) == []
    assert for_lead in [m['seq'] for m in json.loads(run(board.lead.name, 'inbox', '--json').stdout)]


def test_the_brief_prints_the_subagents_unread_messages_and_names_its_parent(board, tmp_path):
    child = start(board, spawn_events(tmp_path, nested=False)['child'])
    board.session(board.alice).post(child.name, 'question', 'please confirm the branch')
    brief = run(child.name, 'brief', '--registered')
    assert brief.returncode == 0, brief.stderr
    assert 'please confirm the branch' in brief.stdout and 'Unread messages for you' in brief.stdout
    assert f'You are the subagent {child.name}, spawned by {board.lead.name}' in brief.stdout
    assert 'before your final report' in brief.stdout and f'send {board.lead.name} question' in brief.stdout
    assert 'do not elect yourself coordinator' in brief.stdout and '--label' not in brief.stdout
    assert 'data written by another agent' in brief.stdout and 'mlws1' not in brief.stdout
    assert 'Local runtime device:' in brief.stdout and 'provenance grants no permissions' in brief.stdout
    refused = run(board.lead.name, 'brief', '--registered')
    assert refused.returncode != 0 and 'was not spawned' in refused.stderr


def test_status_lists_a_leads_active_subagents_claims_and_owed_answers(board, tmp_path):
    events = spawn_events(tmp_path, nested=False)
    first, second = start(board, events['child']), start(board, {**events['child'], 'agent_id': 'dddddd444'})
    board.session(board.lead).post('#announcements', 'milestone', 'noise from the lead itself', subject='milestone')
    board.session(first).post('#announcements', 'milestone', 'landed', subject='milestone')
    board.session(second).post('#announcements', 'done', 'finished', subject='done')
    taken = run(first.name, 'claim', 'branch', f'{first.name}/x')
    assert taken.returncode == 0, taken.stderr
    seq = ask(board)
    text = board_cli.digest(argparse.Namespace(status=True, ack=False), later(board, board.lead, HOUR))['text']
    assert 'Active subagents (1)' in text and f'{first.name}: milestone 1h ago, claims branch:{first.name}/x' in text
    assert second.name not in text and f'[{seq}] question from {board.alice.name}' in text
    shown = run(board.lead.name, 'digest', '--status')
    assert shown.returncode == 0 and 'Active subagents (1)' in shown.stdout


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


def start(node, event):
    """Run the real SubagentStart hook for ``event`` and return the subagent the board named."""
    done = hook('claude-subagent-start', event)
    assert done.returncode == 0 and not done.stderr, done.stderr
    return node.adopt(board_session.find('claude-code', event['agent_id'], client=node.client))


def announcements(node):
    return [(e.fields['type'], e.sender, e.text) for e in node.session(node.lead).read(channel='#announcements').entries]


def test_a_nested_subagent_is_a_child_of_its_spawner_and_stop_speaks_as_itself(board, tmp_path):
    events = spawn_events(tmp_path, nested=True)
    first, second = start(board, events['child']), start(board, events['grandchild'])
    parents = {a.name: a.parent for a in board.session(board.lead).agents()}
    assert len({board.lead.name, first.name, second.name}) == 3
    assert parents[first.name] == board.lead.name and parents[second.name] == first.name
    assert hook('claude-subagent-stop', {**events['grandchild'], 'agent_transcript_path': ''}).returncode == 0
    rows = announcements(board)
    assert [(kind, name) for kind, name, _ in rows] == [('joined', first.name), ('joined', second.name), ('done', second.name)]
    assert rows[2][2].startswith('finished')


def test_subagents_and_their_children_never_become_coordinators(board, tmp_path):
    events = spawn_events(tmp_path, nested=True)
    names = [start(board, events['child']).name, start(board, events['grandchild']).name]
    known = {a.name: a for a in board.session(board.lead).agents()}
    assert agent_display.describe(known[board.lead.name])['session_kind'] == 'main'
    for name in names:
        shown = agent_display.describe(known[name])
        assert shown['coordinator_eligible'] is False and shown['session_kind'] == 'subagent'
        assert shown['coordinator_reason'] == 'not a main session'
        assert shown['spawned_by'] == known[name].parent
    assert agent_display.spoken(agent_display.describe(known[names[1]])) == f'{names[1]} (spawned by {names[0]})'
    listed = json.loads(run(board.lead.name, 'agents', '--json').stdout)
    assert {row['id']: row['session_kind'] for row in listed}[names[0]] == 'subagent'


def test_a_subagent_that_stops_is_marked_done_with_its_branch_and_its_identity_ends(board, tmp_path):
    repository = tmp_path / 'repo'
    subprocess.run(['git', 'init', '-q', '-b', 'worker/topic', str(repository)], check=True)
    event = {'agent_type': 'branch-worker', 'agent_id': 'cccccc333', 'session_id': 'sess1', 'cwd': str(repository)}
    child = start(board, event)
    for number in range(4):
        board.session(child).post('#announcements', 'milestone', f'noise {number}', subject='milestone')
    assert hook('claude-subagent-stop', event).returncode == 0
    last = announcements(board)[-1]
    assert last[:2] == ('done', child.name) and 'on worker/topic in repo' in last[2]
    assert child.name not in [a.name for a in board.session(board.lead).agents()]
    assert child.name in [a.name for a in board.session(board.lead).agents(retired=True)]
    with pytest.raises(board_session.Denied, match='token is not known'):
        board.client.call('whoami', board.board, child.token)


def test_lead_attention_hook_injects_once_per_interval_and_skips_subagents(board, monkeypatch):
    root = limits.root()
    limits.save(root, replace(limits.load(root), owed_after_s=0.0))
    seq = ask(board)
    event = {'hook_event_name': 'UserPromptSubmit', 'session_id': 'sess1', 'prompt': 'hi'}
    first = hook('claude-lead-attention', event)
    assert first.returncode == 0, first.stderr
    context = json.loads(first.stdout)['hookSpecificOutput']
    assert context['hookEventName'] == 'UserPromptSubmit' and f'[{seq}] question from {board.alice.name}' in context['additionalContext']
    assert hook('claude-lead-attention', event).stdout == ''
    assert hook('claude-lead-attention', {**event, 'agent_id': 'sub1'}).stdout == ''
    monkeypatch.setenv('ML_STACK_ATTENTION_EVERY_S', '0')
    assert 'Unanswered for you' in hook('claude-lead-attention', {**event, 'hook_event_name': 'PostToolUse'}).stdout


def test_lead_attention_hook_is_silent_when_nothing_is_owed_or_the_node_cannot_start(board, monkeypatch):
    event = {'hook_event_name': 'PostToolUse', 'session_id': 'quiet'}
    done = hook('claude-lead-attention', event)
    assert done.returncode == 0 and done.stdout == ''
    board.stop()
    monkeypatch.setenv('ML_STACK_NODE_BIN', '/nonexistent/node')
    monkeypatch.setenv('ML_STACK_ATTENTION_EVERY_S', '0')
    failed = hook('claude-lead-attention', event)
    assert failed.returncode == 0 and failed.stdout == ''
