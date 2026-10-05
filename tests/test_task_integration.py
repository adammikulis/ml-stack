"""Real Git worktrees and a local bare remote enforce reviewed development integration."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
from taskboard_kit import accepted, board as _board_fixture

from ml_stack.graph.store import GraphStore
from ml_stack.workspace import (
    claim_handoff,
    integration_git as repo,
    localagent,
    resource_allocations,
    task_integration,
    task_worktrees,
)
from ml_stack.workspace.identity import Denied

board = _board_fixture

GATES = '''import json, os, sys
with open(os.environ['INTEGRATION_GATE_LOG'], 'a') as stream:
    stream.write(json.dumps(sys.argv[1:]) + '\\n')
if os.environ.get('FAIL_INTEGRATION_GATE') == sys.argv[1]:
    sys.exit(1)
'''


@pytest.fixture
def project(board, tmp_path, monkeypatch, request):
    project_root = tmp_path / 'integration-project'
    project_root.mkdir()
    primary, origin = project_root / 'repo', project_root / 'remote.git'
    primary.mkdir()
    repo.git(primary, 'init', '--initial-branch=0.2dev')
    repo.git(primary, 'config', 'user.name', 'Isolated reviewer')
    repo.git(primary, 'config', 'user.email', 'reviewer@example.test')
    (primary / 'scripts').mkdir()
    (primary / 'scripts' / 'test').write_text(GATES)
    (primary / 'sim.py').write_text('speed = 1\n')
    repo.git(primary, 'add', '--', 'scripts/test', 'sim.py')
    repo.git(primary, 'commit', '-m', 'chore: isolated baseline')
    repo.git(tmp_path, 'init', '--bare', str(origin))
    repo.git(primary, 'remote', 'add', 'origin', str(origin))
    repo.git(primary, 'push', '-u', 'origin', '0.2dev')
    template = project_root / 'source-template'
    repo.git(primary, 'worktree', 'add', '-b', 'source-template', str(template))
    runner = localagent.load(board.ws, 'native-worker')
    localagent.save(board.ws, replace(runner, project=str(template)))
    with GraphStore(board.base / 'coordination.db') as graph:
        graph.drop([node['id'] for node in graph.nodes('task-worktree')])
    worktree = task_worktrees.prepare(board.ws, board.parent, board.worker_id, board.task['id'])
    source = project_root / ('task-' + board.task['id'].split(':')[1])
    allocation = resource_allocations.assign(board.ws, board.parent, board.worker_id,
                                             board.task['id'], 'native-grant')
    board.board.claim(board.child, board.task['id'], allocation['allocation_id'])
    claim_handoff.acquire(board.ws, board.ws.auth(board.child), [str(source)])
    (source / 'sim.py').write_text('speed = 2\n')
    repo.git(source, 'add', '--', 'sim.py')
    repo.git(source, 'commit', '-m', 'fix: reviewed simulation speed')
    patch = repo.git(source, 'diff', '--binary', '--full-index', worktree['baseline_commit'], 'HEAD', binary=True)
    if getattr(request, 'param', '') == 'bad_patch':
        patch += b'not the reviewed full diff\n'
    (source / '.task.patch').write_bytes(patch)
    (source / '.task-report.md').write_text('Independent replay ready.\n')
    repo.git(source, 'add', '--', '.task.patch', '.task-report.md')
    repo.git(source, 'commit', '-m', 'chore: record canonical task artifacts')
    artifacts = {name: hashlib.sha256((source / name).read_bytes()).hexdigest()
                 for name in ('sim.py', '.task.patch', '.task-report.md')}
    proposal = board.board.submit(board.child, board.task['id'],
                                 {'summary': 'Native source is committed.', 'artifacts': artifacts,
                                  'checks': [{'name': 'Worker claim', 'passed': True}],
                                  'provenance': {'commit': repo.git(source, 'rev-parse', 'HEAD')}})
    hook = Path(__file__).resolve().parents[1] / 'scripts' / 'hooks' / 'pre-push'
    (primary / '.git' / 'hooks' / 'pre-push').symlink_to(hook)
    log = tmp_path / 'gate-commands.jsonl'
    monkeypatch.setenv('INTEGRATION_GATE_LOG', str(log))
    return {'primary': primary, 'source': source, 'origin': origin, 'log': log,
            'proposal': proposal, 'baseline': worktree['baseline_commit'], 'worktree': worktree}


def test_reviewed_native_patch_runs_maintained_gates_then_fast_forwards_development(board, project):
    review = board.board.review(board.parent, board.task['id'], accepted())
    result = task_integration.integrate(board.ws, board.parent, board.task['id'])
    assert result['state'] == 'published' and result['review_hash'] == review['review_hash']
    commands = [json.loads(line) for line in project['log'].read_text().splitlines()]
    assert commands == [['quick', '--base', project['baseline']], ['gate'],
                        ['all', 'tests/test_serve_no_bypass.py', '--redteam']]
    assert repo.git(project['primary'], 'branch', '--show-current') == '0.2dev'
    assert repo.git(project['primary'], 'rev-parse', 'HEAD') == result['commit']
    assert repo.git(project['origin'], 'rev-parse', 'refs/heads/0.2dev') == result['commit']
    assert repo.ancestor(project['primary'], project['baseline'], result['commit'])
    assert repo.git(project['origin'], 'for-each-ref', '--format=%(refname)', 'refs/heads') == 'refs/heads/0.2dev'
    assert task_integration.integrate(board.ws, board.parent, board.task['id']) == result
    with GraphStore(board.base / 'coordination.db') as graph:
        assert len(graph.nodes('integration')) == 1
        assert {node['attrs']['state'] for node in graph.nodes('integration-event')} == {'scope_returned', 'candidate', 'integrated', 'published'}
        assert len(graph.nodes('integration-gate')) == 3
        assert graph.nodes('git-commit')[0]['attrs']['sha'] == result['commit']
    assert board.ws.who_owns('worktree', str(project['source']))['owner'] == 'lead'


@pytest.mark.redteam
def test_unaccepted_and_unrelated_requests_make_no_git_changes(board, project):
    for token in (board.parent, board.agent('outsider')):
        with pytest.raises(Denied):
            task_integration.integrate(board.ws, token, board.task['id'])
    assert repo.git(project['primary'], 'rev-parse', 'HEAD') == project['baseline']
    assert not project['log'].exists()
    assert board.ws.who_owns('worktree', str(project['source']))['owner'] == board.worker_id


@pytest.mark.redteam
@pytest.mark.parametrize('change', ['new_commit', 'dirty', 'main', 'artifact'])
def test_stale_sources_and_release_target_are_refused_before_gates(board, project, change):
    board.board.review(board.parent, board.task['id'], accepted())
    if change == 'main':
        repo.git(project['primary'], 'branch', '-m', 'main')
    elif change in ('new_commit', 'dirty'):
        (project['source'] / 'sim.py').write_text('speed = 99\n')
        if change == 'new_commit':
            repo.git(project['source'], 'add', '--', 'sim.py')
            repo.git(project['source'], 'commit', '-m', 'fix: unreviewed source')
    else:
        with GraphStore(board.base / 'coordination.db') as graph:
            proposal = graph.nodes('proposal')[0]
            graph.upsert_node({**proposal, 'attrs': {**proposal['attrs'], 'artifacts': {'sim.py': 'b' * 64}}})
    with pytest.raises((Denied, ValueError)):
        task_integration.integrate(board.ws, board.parent, board.task['id'])
    assert not project['log'].exists()


def test_parallel_reviewed_branch_merges_in_owned_candidate_without_rebasing(board, project):
    board.board.review(board.parent, board.task['id'], accepted())
    (project['primary'] / 'parallel.py').write_text('independent = True\n')
    repo.git(project['primary'], 'add', '--', 'parallel.py')
    repo.git(project['primary'], 'commit', '-m', 'feat: independent parallel development')
    before = repo.git(project['primary'], 'rev-parse', 'HEAD')
    result = task_integration.integrate(board.ws, board.parent, board.task['id'])
    assert result['state'] == 'published'
    assert repo.ancestor(project['primary'], before, result['commit'])
    assert repo.ancestor(project['primary'], project['proposal']['provenance']['commit'], result['commit'])
    assert repo.git(project['source'], 'rev-parse', 'HEAD') == project['proposal']['provenance']['commit']
    assert (project['primary'] / 'parallel.py').exists()


@pytest.mark.redteam
@pytest.mark.parametrize('failure', ['conflict', 'gate'])
def test_failed_merge_or_gate_preserves_candidate_and_never_publishes(board, project, monkeypatch, failure):
    board.board.review(board.parent, board.task['id'], accepted())
    if failure == 'conflict':
        (project['primary'] / 'sim.py').write_text('speed = 3\n')
        repo.git(project['primary'], 'add', '--', 'sim.py')
        repo.git(project['primary'], 'commit', '-m', 'fix: conflicting independent change')
    else:
        monkeypatch.setenv('FAIL_INTEGRATION_GATE', 'gate')
    before = repo.git(project['primary'], 'rev-parse', 'HEAD')
    result = task_integration.integrate(board.ws, board.parent, board.task['id'])
    assert result['state'] == 'blocked'
    assert repo.git(project['primary'], 'rev-parse', 'HEAD') == before
    assert repo.git(project['origin'], 'rev-parse', 'refs/heads/0.2dev') == project['baseline']
    assert repo.git(project['source'], 'rev-parse', 'HEAD') == project['proposal']['provenance']['commit']


@pytest.mark.redteam
def test_maintained_push_hook_preserves_foreign_merged_worktrees_and_reports_the_block(board, project, tmp_path):
    board.board.review(board.parent, board.task['id'], accepted())
    foreign = tmp_path / 'foreign-worktree'
    repo.git(project['primary'], 'worktree', 'add', '-b', 'other-agent/active', str(foreign))
    (foreign / 'foreign.py').write_text('independent = True\n')
    repo.git(foreign, 'add', '--', 'foreign.py')
    repo.git(foreign, 'commit', '-m', 'feat: other agent change')
    repo.git(project['primary'], 'merge', '--ff-only', 'other-agent/active')
    other = board.agent('other-agent')
    board.ws.claim(other, 'worktree', str(foreign))
    result = task_integration.integrate(board.ws, board.parent, board.task['id'])
    assert result['state'] == 'blocked' and 'other-agent/active' in result['reason']
    assert 'commit' in result and result['checks']
    assert foreign.exists() and not Path(repo.git(foreign, 'rev-parse', '--git-path', 'locked')).exists()
    assert board.ws.who_owns('worktree', str(foreign))['owner'] == 'other-agent'
    assert repo.git(project['origin'], 'rev-parse', 'refs/heads/0.2dev') == project['baseline']


@pytest.mark.redteam
@pytest.mark.parametrize('project', ['bad_patch'], indirect=True)
def test_accepted_artifact_hashes_cannot_substitute_an_unrelated_native_patch(board, project):
    board.board.review(board.parent, board.task['id'], accepted())
    with pytest.raises(Denied, match='exact reviewed source changes'):
        task_integration.integrate(board.ws, board.parent, board.task['id'])
    assert not project['log'].exists()


@pytest.mark.redteam
def test_worker_or_unrelated_agent_cannot_return_the_reviewed_child_scope(board, project):
    for token in (board.child, board.agent('unrelated')):
        with pytest.raises(Denied):
            board.ws.claims.return_worktree(board.ws.auth(token), project['worktree'])
    assert board.ws.who_owns('worktree', str(project['source']))['owner'] == board.worker_id


@pytest.mark.redteam
def test_same_worker_claim_for_another_assignment_cannot_be_returned(board, project):
    board.board.review(board.parent, board.task['id'], accepted())
    claim = board.ws.who_owns('worktree', str(project['source']))
    assert claim['assignment'] == project['worktree']['id']
    wrong = {**project['worktree'], 'id': 'task-worktree:' + '0' * 32}
    with pytest.raises(Denied, match='exact task delegation'):
        board.ws.claims.return_worktree(board.ws.auth(board.parent), wrong)
    assert board.ws.who_owns('worktree', str(project['source'])) == claim
