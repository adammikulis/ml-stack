"""Native hook ownership rejects competing mutations before their filesystem effects."""

import concurrent.futures
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from workspace_kit import Kit, clean_env

from ml_stack import harness_claims, harnesshook
from ml_stack.workspace import Conflict, Denied, tokens
from ml_stack.workspace.project import describe

pytestmark = pytest.mark.redteam


@pytest.fixture
def kit(tmp_path, monkeypatch):
    kit = Kit(clean_env(monkeypatch, tmp_path))
    for name in ('alpha', 'beta'):
        tokens.store(kit.base, name, kit.agent(name))
    kit.project = tmp_path / 'project'
    kit.project.mkdir()
    for name in ('alpha', 'beta'):
        kit.ws.registry.set_project(kit.ws.auth(kit.owner), name, describe(str(kit.project)))
    return kit


def test_actual_hook_blocks_other_owner_and_symlink_alias_before_write(kit):
    target = kit.project / 'result.txt'
    alias = kit.project / 'alias'
    alias.symlink_to(kit.project, target_is_directory=True)
    kit.ws.claim(tokens.load(kit.base, 'beta'), 'file', str(target))
    payload = {'tool_name': 'Write', 'tool_input': {'file_path': str(alias / target.name), 'content': 'hostile'},
               'cwd': str(kit.project)}
    done = subprocess.run([sys.executable, '-m', 'ml_stack.harnesshook', 'pre', '--role', 'plan-and-go',
                           '--label', 'alpha', '--root', str(kit.project)],
                          input=json.dumps(payload), text=True, capture_output=True,
                          env={**os.environ, 'PYTHONPATH': str(Path(__file__).parents[1] / 'src')},
                          timeout=10, check=True)
    decision = json.loads(done.stdout)['hookSpecificOutput']
    if decision['permissionDecision'] == 'allow':
        target.write_text('hostile')
    assert decision['permissionDecision'] == 'deny' and 'beta' in decision['permissionDecisionReason']
    assert not target.exists()


def test_atomic_multi_resource_conflict_does_not_leave_partial_claims(kit):
    alpha, beta = kit.ws.auth(tokens.load(kit.base, 'alpha')), kit.ws.auth(tokens.load(kit.base, 'beta'))
    kit.ws.claims.claim(beta, 'port', '8080')
    with pytest.raises(Conflict):
        kit.ws.claims.reserve(alpha, [('file', str(kit.project / 'untouched')), ('port', '8080')])
    assert kit.ws.claims.listing(owner='alpha') == []
    def take(who):
        try:
            kit.ws.claims.reserve(who, [('file', str(kit.project / 'shared'))])
            return who.id
        except Conflict:
            return None
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        assert len([value for value in pool.map(take, [alpha, beta]) if value]) == 1


def test_launcher_identity_and_root_cannot_be_overridden_by_call(kit):
    target = kit.project / 'ok.txt'
    args = {'file_path': str(target), 'actor': 'beta', 'roots': ['/']}
    harness_claims.reserve('Write', args, str(kit.project), 'alpha', [str(kit.project)])
    assert kit.ws.claims.who('file', str(target))['owner'] == 'alpha'
    with pytest.raises(Denied, match='outside the launcher-approved'):
        harness_claims.reserve('Write', {'file_path': str(kit.project.parent / 'outside')},
                               str(kit.project), 'alpha', [str(kit.project)])


def test_explicit_port_worktree_and_install_mutations_reserve_native_resources(kit, monkeypatch):
    python = kit.project / 'venv' / 'bin' / 'python'
    which = harness_claims.shutil.which
    monkeypatch.setattr(harness_claims.shutil, 'which', lambda name: str(python) if name == 'python' else which(name))
    command = f'git -C {kit.project} add result.txt; python -m pip install wheel; ml-stack-serve down --port 51548'
    required = harness_claims.resources('Bash', {'command': command}, str(kit.project))
    assert ('worktree', str(kit.project)) in required
    assert ('install', str(python.parent.parent)) in required
    assert ('port', '51548') in required
    harness_claims.reserve('Bash', {'command': command}, str(kit.project), 'alpha', [str(kit.project)])
    decision = harnesshook.pre({'tool_name': 'Bash', 'tool_input': {'command': command}, 'cwd': str(kit.project)},
                               harnesshook.Rail('plan-and-go', 'beta', roots=[str(kit.project)], wait_s=0))
    assert decision['hookSpecificOutput']['permissionDecision'] == 'deny'
    assert kit.ws.claims.who('install', str(python.parent.parent))['owner'] == 'alpha'


def test_separate_worktrees_share_source_area_but_allow_independent_files(kit, tmp_path):
    git = harness_claims.shutil.which('git')
    def run(*args):
        return subprocess.run([git, '-C', str(kit.project), *args], capture_output=True,
                              text=True, check=True, timeout=10)
    run('init')
    (kit.project / 'shared.py').write_text('original\n')
    run('add', 'shared.py')
    run('-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.test', 'commit', '-m', 'fixture')
    other = tmp_path / 'other-worktree'
    run('worktree', 'add', '-b', 'independent', str(other))
    kit.ws.registry.set_project(kit.ws.auth(kit.owner), 'beta', describe(str(other)))
    harness_claims.reserve('Write', {'file_path': str(kit.project / 'shared.py')},
                           str(kit.project), 'alpha', [str(kit.project)])
    with pytest.raises(Conflict):
        harness_claims.reserve('Write', {'file_path': str(other / 'shared.py')},
                               str(other), 'beta', [str(other)])
    harness_claims.reserve('Write', {'file_path': str(other / 'independent.py')},
                           str(other), 'beta', [str(other)])
    owner = kit.ws.claims.who('area', str(other / 'shared.py'))
    assert owner['owner'] == 'alpha'
    assert owner['commit'] == run('rev-parse', 'HEAD').stdout.strip()
    assert owner['owner_pid'] > 0 and owner['owner_started'] > 0
    assert owner['environment'] and owner['interpreter']
