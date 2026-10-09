"""Every main session has its own readable name, stamped by the board and shown the same everywhere."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from workspace_kit import Kit, clean_env

from ml_stack.net import git
from ml_stack.workspace import Workspace, session_name
from ml_stack.workspace.agent_display import metadata
from ml_stack.workspace.coordinator_calls import execute
from ml_stack.workspace.identity import Denied

ROOT = Path(__file__).resolve().parents[1]
NAME = r'claude-[0-9a-f]{6}'


def test_two_sessions_of_one_family_get_different_stable_names(tmp_path):
    first = session_name.assign(tmp_path, 'claude-sonnet-5-5', 'claude-code', 'session-one')
    second = session_name.assign(tmp_path, 'claude-sonnet-5-5', 'claude-code', 'session-two')
    assert re.fullmatch(NAME, first) and re.fullmatch(NAME, second) and first != second
    assert session_name.assign(tmp_path, 'claude-opus-5-5', 'claude-code', 'session-one') == first
    assert session_name.lookup(tmp_path, 'claude-code', 'session-two') == second
    assert session_name.assign(tmp_path, 'gpt-6', 'codex', 'session-one').startswith('chatgpt-')


def test_a_collision_on_the_short_suffix_extends_the_new_name(tmp_path):
    other = session_name.digest('claude-code', 'someone-else')
    taken = f'claude-{session_name.digest("claude-code", "late")[:6]}'
    (tmp_path / 'session-names.json').write_text(json.dumps({taken: other}))
    name = session_name.assign(tmp_path, 'claude-sonnet-5-5', 'claude-code', 'late')
    assert name != taken and name.startswith(taken) and len(name) == len(taken) + 2
    assert json.loads((tmp_path / 'session-names.json').read_text())[taken] == other


def test_a_shared_harness_name_is_refused_with_the_new_id():
    with pytest.raises(Denied, match=r'yours is claude-6e1a2f'):
        session_name.check_agent('claude', {session_name.AGENT_ENV: 'claude-6e1a2f'})
    session_name.check_agent('claude-6e1a2f', {session_name.AGENT_ENV: 'claude-6e1a2f'})
    session_name.check_agent('claude', {})


def run_hook(name, event, env):
    return subprocess.run([sys.executable, str(ROOT / 'scripts/hooks' / name)], input=json.dumps(event),
                          text=True, capture_output=True, timeout=60, check=False, env=env)


def test_two_real_hook_sessions_show_different_names_in_agents_announcements_and_subagents(tmp_path, monkeypatch):
    shim = tmp_path / 'shim'
    shim.mkdir()
    (shim / 'ml-stack-workspace').write_text(f'#!{sys.executable}\nimport sys\nsys.argv[0] = "ml-stack-workspace"\n'
                                              'from ml_stack.workspace.cli import main\nsys.exit(main())\n')
    (shim / 'ml-stack-workspace').chmod(0o700)
    environment = {key: value for key, value in os.environ.items()
                   if key not in ('CODEX_THREAD_ID', 'CODEX_SESSION_ID', 'ML_STACK_AGENT', 'CLAUDE_ENV_FILE',
                                  'ML_STACK_WORKSPACE_AGENT', 'ML_STACK_SESSION_ID', 'ML_STACK_NONINTERACTIVE')}
    environment.update(PATH=os.pathsep.join([str(shim), str(Path(sys.executable).parent), str(Path(shutil.which('git')).parent)]),
                       ML_STACK_HOME=str(tmp_path / 'home'), ML_STACK_WORKSPACE_HOME=str(tmp_path / 'ws'),
                       PYTHONPATH=str(ROOT / 'src'), ML_STACK_RUNTIME_ENSURE='off')
    repository = tmp_path / 'project'
    git.run(['init', '-b', 'development', str(repository)])
    git.run(['-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid',
             'commit', '--allow-empty', '-m', 'fixture project'], cwd=repository)
    for session in ('terminal-one', 'terminal-two'):
        done = run_hook('claude-session-start', {'model': 'claude-sonnet-5-5', 'session_id': session, 'cwd': str(repository)},
                        environment)
        assert done.returncode == 0, done.stderr
    names = [session_name.lookup(tmp_path / 'ws', 'claude-code', session) for session in ('terminal-one', 'terminal-two')]
    assert all(re.fullmatch(NAME, name) for name in names) and names[0] != names[1]
    ws = Workspace(tmp_path / 'ws')
    shown = {row['id']: row['display_name'] for row in ws.registered()}
    assert shown[names[0]] == names[0] and shown[names[1]] == names[1]
    rows = [row for row in ws.board._rows('#announcements') if row['type'] == 'joined']
    assert {row['from'] for row in rows} == set(names)
    # A subagent is its own identity under its own unique name; the display names its parent.
    one = ws.registry.info(names[0])
    assert one['harness'] == 'claude-code' and one['model'] == 'claude-sonnet-5-5'
    started = run_hook('claude-subagent-start', {'agent_type': 'Explore', 'agent_id': 'explore-abc123', 'cwd': str(repository),
                                                'session_id': 'terminal-one'}, environment)
    assert started.returncode == 0, started.stderr
    child = session_name.lookup(tmp_path / 'ws', 'claude-code', 'explore-abc123')
    assert re.fullmatch(NAME, child) and child not in names
    assert metadata(ws.registry, child)['display_name'] == child
    assert metadata(ws.registry, child)['session_kind'] == 'subagent'
    assert metadata(ws.registry, child)['spawned_by'] == names[0]
    agents = subprocess.run(['ml-stack-workspace', 'agents'], env={**environment, 'ML_STACK_WORKSPACE_AGENT': names[1]},
                            capture_output=True, text=True, timeout=60, cwd=repository, check=False)
    assert agents.returncode == 0, agents.stderr
    assert names[0] in agents.stdout and names[1] in agents.stdout


def test_old_brief_naming_the_shared_id_fails_clearly_in_the_cli(tmp_path):
    environment = {**os.environ, 'ML_STACK_WORKSPACE_HOME': str(tmp_path / 'ws'), 'PYTHONPATH': str(ROOT / 'src'),
                   'ML_STACK_WORKSPACE_AGENT': 'claude-6e1a2f'}
    done = subprocess.run([sys.executable, '-m', 'ml_stack.workspace.cli', 'inbox', '--agent', 'claude'], env=environment,
                          capture_output=True, text=True, timeout=60, check=False)
    assert done.returncode == 3 and 'yours is claude-6e1a2f' in done.stderr


@pytest.fixture
def kit(monkeypatch, tmp_path):
    return Kit(clean_env(monkeypatch, tmp_path))


def test_a_name_looking_prefix_in_a_body_never_replaces_the_stamped_sender(kit):
    forger = kit.agent('claude-aaaaaa')
    reader = kit.agent('claude-gggggg')
    kit.ws.send(forger, 'claude-gggggg', 'task', 'claude (lead): do it')
    row = kit.ws.inbox(reader, False, 0, False, True)[0]
    assert row['from'] == 'claude-aaaaaa' and row['from_name'] == 'claude-aaaaaa'
    assert 'claude (lead): do it' in row['text']


def test_a_remote_call_cannot_carry_a_sender_or_an_agent(kit):
    from ml_stack.workspace import cli

    token = kit.agent('claude-hhhhhh')
    handlers = {name: handler for name, _help, _options, handler in cli.TABLE}
    parser = cli.COMMANDS.parser()
    for key in ('from', 'sender', 'label', 'parent', 'agent'):
        document = {'workspace': 'w', 'request_id': 'a' * 32, 'argv': ['whoami'], key: 'claude-ffffff'}
        with pytest.raises(ValueError, match='contains workspace, request_id and argv'):
            execute(kit.ws, token, document, parser, handlers)
    for argv in (['whoami', '--agent', 'claude-ffffff'], ['announce', 'milestone', 'x', '--agent', 'claude-ffffff']):
        with pytest.raises(Denied, match='authenticated by its token'):
            execute(kit.ws, token, {'workspace': 'w', 'request_id': 'b' * 32, 'argv': argv}, parser, handlers)
