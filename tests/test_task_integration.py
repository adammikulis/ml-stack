"""Real Git worktrees and a local bare remote enforce reviewed development integration."""

import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from taskboard_kit import accepted, board as _board_fixture

from ml_stack import harness_claims
from ml_stack.graph.store import GraphStore
from ml_stack.workspace import (
    claim_handoff,
    integration_git as repo,
    integration_staging,
    localagent,
    resource_allocations,
    task_integration,
    task_scheduler,
    task_worktrees,
)
from ml_stack.workspace.identity import Denied
from ml_stack.workspace.project import describe

board = _board_fixture

GATES = '''import json, os, sys
with open(os.environ['INTEGRATION_GATE_LOG'], 'a') as stream:
    stream.write(json.dumps(sys.argv[1:]) + '\\n')
if os.environ.get('FAIL_INTEGRATION_GATE') == sys.argv[1]:
    sys.exit(1)
'''


def install_push_hook(primary):
    hook = Path(__file__).resolve().parents[1] / 'scripts' / 'hooks' / 'pre-push'
    interpreter = Path(sys.executable).as_posix()
    checker = (hook.parent / 'pushed').as_posix()
    installed = primary / '.git' / 'hooks' / 'pre-push'
    installed.write_text(hook.read_text().replace('"${PYTHON:-python3}" "$here/pushed"', f'\"{interpreter}\" \"{checker}\"'))
    installed.chmod(0o755)


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
    (primary / '.gitignore').write_text('.claude/worktrees/\n')
    (primary / 'sim.py').write_text('speed = 1\n')
    repo.git(primary, 'add', '--', '.gitignore', 'scripts/test', 'sim.py')
    repo.git(primary, 'commit', '-m', 'chore: isolated baseline')
    repo.git(tmp_path, 'init', '--bare', str(origin))
    repo.git(primary, 'remote', 'add', 'origin', str(origin))
    repo.git(primary, 'push', '-u', 'origin', '0.2dev')
    template = project_root / 'source-template'
    repo.git(primary, 'worktree', 'add', '-b', 'source-template', str(template))
    runner = localagent.load(board.ws, 'native-worker')
    localagent.save(board.ws, replace(runner, project=str(template)))
    if getattr(request, 'param', '') == 'human_creator':
        board.task = board.board.create(board.owner, {**board.spec, 'source_key': 'human:native-integration'})
    with GraphStore(board.base / 'coordination.db') as graph:
        graph.drop([node['id'] for node in graph.nodes('task-worktree')])
    worktree = task_worktrees.prepare(board.ws, board.parent, board.worker_id, board.task['id'])
    source = project_root / ('task-' + board.task['id'].split(':')[1])
    allocation = resource_allocations.assign(board.ws, board.parent, board.worker_id,
                                             board.task['id'], 'native-grant')
    board.board.claim(board.child, board.task['id'], allocation['allocation_id'])
    claim_handoff.acquire(board.ws, board.ws.auth(board.child), [str(source)])
    if getattr(request, 'param', '') == 'native_claims':
        monkeypatch.setattr(harness_claims, 'Workspace', lambda: board.ws)
        board.ws.registry.set_project(board.ws.auth(board.owner), 'lead', describe(str(source)))
        harness_claims.reserve('Write', {'file_path': str(source / 'sim.py')},
                               str(source), board.worker_id, [str(source)])
        harness_claims.reserve('Bash', {'command': 'git add sim.py'},
                               str(source), board.worker_id, [str(source)])
    (source / 'sim.py').write_text('speed = 2\n')
    repo.git(source, 'add', '--', 'sim.py')
    repo.git(source, 'commit', '-m', 'fix: reviewed simulation speed')
    patch = repo.git(source, 'diff', '--binary', '--full-index', worktree['baseline_commit'], 'HEAD', binary=True)
    if getattr(request, 'param', '') == 'bad_patch':
        patch += b'not the reviewed full diff\n'
    (source / '.task.patch').write_bytes(patch)
    (source / '.task-report.md').write_text('Task artifacts recorded.\n')
    repo.git(source, 'add', '--', '.task.patch', '.task-report.md')
    repo.git(source, 'commit', '-m', 'chore: record canonical task artifacts')
    artifacts = {name: hashlib.sha256(repo.git(source, 'show', f'HEAD:{name}', binary=True)).hexdigest()
                 for name in ('sim.py', '.task.patch', '.task-report.md')}
    proposal = board.board.submit(board.child, board.task['id'],
                                 {'summary': 'Native source is committed.', 'artifacts': artifacts,
                                  'checks': [{'name': 'Worker claim', 'passed': True}],
                                  'provenance': {'commit': repo.git(source, 'rev-parse', 'HEAD')}})
    install_push_hook(primary)
    log = tmp_path / 'gate-commands.jsonl'
    monkeypatch.setenv('INTEGRATION_GATE_LOG', str(log))
    return {'primary': primary, 'source': source, 'origin': origin, 'log': log,
            'proposal': proposal, 'baseline': worktree['baseline_commit'], 'worktree': worktree}


def test_reviewed_native_patch_runs_maintained_gates_then_fast_forwards_development(board, project):
    review = board.board.review(board.parent, board.task['id'], accepted())
    result = task_integration.integrate(board.ws, board.parent, board.task['id'], publish=True)
    assert result['state'] == 'published' and result['review_hash'] == review['review_hash']
    commands = [json.loads(line) for line in project['log'].read_text().splitlines()]
    assert commands == [['quick', '--base', project['baseline']], ['gate'],
                        ['all', 'tests/test_serve_no_bypass.py', '--redteam']]
    assert repo.git(project['primary'], 'branch', '--show-current') == '0.2dev'
    assert repo.git(project['primary'], 'rev-parse', 'HEAD') == result['commit']
    assert repo.git(project['origin'], 'rev-parse', 'refs/heads/0.2dev') == result['commit']
    assert repo.ancestor(project['primary'], project['baseline'], result['commit'])
    assert repo.git(project['origin'], 'for-each-ref', '--format=%(refname)', 'refs/heads') == 'refs/heads/0.2dev'
    assert task_integration.integrate(board.ws, board.parent, board.task['id'], publish=True) == result
    with GraphStore(board.base / 'coordination.db') as graph:
        assert len(graph.nodes('integration')) == 1
        assert {node['attrs']['state'] for node in graph.nodes('integration-event')} == {'scope_returned', 'candidate', 'integrated', 'publication_confirmed', 'cleanup_required', 'published'}
        assert len(graph.nodes('integration-gate')) == 3
        assert graph.nodes('git-commit')[0]['attrs']['sha'] == result['commit']
    assert not project['source'].exists()
    assert board.ws.who_owns('worktree', str(project['source'])) is None
    assert board.board.get(board.parent, board.task['id'])['state'] == 'completed'


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
        task_integration.integrate(board.ws, board.parent, board.task['id'], publish=True)
    assert not project['log'].exists()


def test_parallel_reviewed_branch_merges_in_owned_candidate_without_rebasing(board, project):
    board.board.review(board.parent, board.task['id'], accepted())
    (project['primary'] / 'parallel.py').write_text('independent = True\n')
    repo.git(project['primary'], 'add', '--', 'parallel.py')
    repo.git(project['primary'], 'commit', '-m', 'feat: independent parallel development')
    before = repo.git(project['primary'], 'rev-parse', 'HEAD')
    result = task_integration.integrate(board.ws, board.parent, board.task['id'], publish=True)
    assert result['state'] == 'published'
    assert repo.ancestor(project['primary'], before, result['commit'])
    assert repo.ancestor(project['primary'], project['proposal']['provenance']['commit'], result['commit'])
    assert not project['source'].exists()
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
    result = task_integration.integrate(board.ws, board.parent, board.task['id'], publish=True)
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
    result = task_integration.integrate(board.ws, board.parent, board.task['id'], publish=True)
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
        task_integration.integrate(board.ws, board.parent, board.task['id'], publish=True)
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
    current = board.ws.who_owns('worktree', str(project['source']))
    assert {key: value for key, value in current.items() if key != 'expires_in_s'} == \
        {key: value for key, value in claim.items() if key != 'expires_in_s'}


@pytest.mark.redteam
def test_reviewed_artifact_names_stay_literal_git_arguments_and_cannot_escape(tmp_path):
    repo.git(tmp_path, 'init', '--initial-branch=0.2dev')
    repo.git(tmp_path, 'config', 'user.name', 'Isolated reviewer')
    repo.git(tmp_path, 'config', 'user.email', 'reviewer@example.test')
    name = '$(touch ESCAPED);--upload-pack.py'
    content = b'authoritative tracked content\n'
    (tmp_path / name).write_bytes(content)
    repo.git(tmp_path, 'add', '--', name)
    repo.git(tmp_path, 'commit', '-m', 'test: literal hostile artifact path')
    commit = repo.git(tmp_path, 'rev-parse', 'HEAD')
    digest = hashlib.sha256(content).hexdigest()
    repo.reviewed_files(tmp_path, commit, {name: digest})
    for unsafe in ('../outside.py', '/outside.py', 'nested\\outside.py'):
        with pytest.raises(Denied, match='relative repository paths'):
            repo.reviewed_files(tmp_path, commit, {unsafe: digest})
    assert not (tmp_path / 'ESCAPED').exists()


@pytest.mark.redteam
def test_foreign_development_branch_owner_blocks_integration_before_gates(board, project):
    board.board.review(board.parent, board.task['id'], accepted())
    other = board.agent('development-owner')
    claim = board.ws.claim(other, 'branch', '0.2dev')
    result = task_integration.integrate(board.ws, board.parent, board.task['id'], publish=True)
    assert result['state'] == 'blocked' and result['blocking_owner'] == 'development-owner'
    assert result['blocking_claim'] == claim
    current = board.ws.who_owns('branch', '0.2dev')
    assert {key: current[key] for key in claim} == claim
    assert not project['log'].exists()
    assert repo.git(project['primary'], 'rev-parse', 'HEAD') == project['baseline']


def test_standard_nested_harness_worktrees_do_not_dirty_or_delete_primary(board, project):
    foreign = project['primary'] / '.claude' / 'worktrees' / 'foreign-agent'
    foreign.mkdir(parents=True)
    marker = foreign / 'work.txt'
    marker.write_text('active foreign work\n')
    repo.clean(project['primary'])
    assert marker.read_text() == 'active foreign work\n'
    unrelated = project['primary'] / '.claude' / 'unreviewed.py'
    unrelated.write_text('unreviewed = True\n')
    with pytest.raises(Denied, match='uncommitted changes'):
        repo.clean(project['primary'])
    assert marker.exists() and unrelated.exists()


def test_task_integrate_interfaces_use_native_token_and_exact_task_id(board, project, monkeypatch):
    from ml_stack.workspace import cli, tools

    board.board.review(board.parent, board.task['id'], accepted())
    monkeypatch.setattr(tools, 'Workspace', lambda: board.ws)
    monkeypatch.setattr(tools, '_token', lambda: board.parent)
    result = tools.workspace_task_integrate(board.task['id'])
    assert result['state'] == 'completed'
    command = next(item for item in cli.TABLE if item[0] == 'task-integrate')
    assert command[3](SimpleNamespace(id=board.task['id']), board.ws, board.parent) == result


@pytest.mark.parametrize('project', ['human_creator'], indirect=True)
def test_registered_parent_integrates_human_created_independently_reviewed_task(board, project):
    task = board.board.get(board.owner, board.task['id'])
    assert task['created_by'] != board.ws.auth(board.parent).id
    review = board.board.review(board.owner, board.task['id'], accepted())
    assert review['verifier'] != board.ws.auth(board.parent).id
    result = task_integration.integrate(board.ws, board.parent, board.task['id'], publish=True)
    assert result['state'] == 'published'
    assert repo.git(project['origin'], 'rev-parse', 'refs/heads/0.2dev') == result['commit']


@pytest.mark.redteam
@pytest.mark.parametrize('project', ['native_claims'], indirect=True)
def test_native_reservations_release_only_the_exact_accepted_task_assignment(board, project, tmp_path):
    source, scope = project['source'], project['worktree']
    target = str(source / 'sim.py')
    before = board.ws.claims.listing(owner=board.worker_id)
    assigned = [claim for claim in before if claim.get('assignment') == scope['id']]
    assert {claim['kind'] for claim in assigned} == {'file', 'area', 'worktree'}
    assert all(claim.get('task') == scope['task'] and claim.get('project') == scope['project']
               for claim in assigned)
    assert next(claim for claim in assigned if claim['kind'] == 'worktree')['delegated_by'] == 'lead'
    with pytest.raises(Denied, match='different task assignment'):
        board.ws.claims.reserve(board.ws.auth(board.child), [('file', target)],
                                {'assignment': 'task-worktree:other'})
    assert board.ws.claims.who('file', target)['assignment'] == scope['id']
    unrelated = str(tmp_path / 'unrelated-worker-file')
    board.ws.claims.reserve(board.ws.auth(board.child), [('file', unrelated)])
    wrong = str(tmp_path / 'different-task-file')
    board.ws.claims.reserve(board.ws.auth(board.child), [('file', wrong)],
                            {'assignment': 'task-worktree:other', 'task': 'task:other', 'project': str(source)})
    board.board.review(board.parent, board.task['id'], accepted())
    result = task_integration.integrate(board.ws, board.parent, board.task['id'], publish=True)
    assert result['state'] == 'published'
    assert not any(claim.get('assignment') == scope['id']
                   for claim in board.ws.claims.listing(owner=board.worker_id))
    assert board.ws.claims.who('file', unrelated)['owner'] == board.worker_id
    assert board.ws.claims.who('file', wrong)['assignment'] == 'task-worktree:other'
    with pytest.raises(Denied, match='different ownership claim'):
        board.ws.claims.return_worktree(board.ws.auth(board.parent),
                                        {**scope, 'id': 'task-worktree:other'})
    assert board.ws.claims.who('file', wrong)['assignment'] == 'task-worktree:other'
    with GraphStore(board.base / 'coordination.db') as graph:
        returned = next(row['attrs'] for row in graph.nodes('integration-event')
                        if row['attrs']['state'] == 'scope_returned')
        assert {claim['kind'] for claim in returned['released_claims']} == {'file', 'area'}


def test_authenticated_task_inspection_exposes_exact_published_review_and_once_only_attempt(board, project):
    review = board.board.review(board.parent, board.task['id'], accepted())
    outcome = task_scheduler.integrate_completed(board.ws, board.parent, board.worker_id)[0]
    assert outcome['state'] == 'completed'
    assert task_scheduler.integrate_completed(board.ws, board.parent, board.worker_id) == []
    detail = board.board.get(board.owner, board.task['id'])
    integration = detail['integration']
    assert integration['review_id'] == review['id'] and integration['review_hash'] == review['review_hash']
    assert integration['proposal_id'] == detail['proposal']['id']
    assert integration['proposal_hash'] == detail['proposal']['proposal_hash']
    assert integration['source_commit'] == project['proposal']['provenance']['commit']
    assert integration['commit'] == outcome['commit'] and len(integration['checks']) == 3
    assert all(check['passed'] and len(check['output_hash']) == 64 for check in integration['checks'])
    assert all(check['command'][0] == 'scripts/test' for check in integration['checks'])
    assert len(detail['integration_attempts']) == 1
    assert detail['integration_attempts'][0]['outcome'] == integration
    listed = next(task for task in board.board.list(board.owner)['tasks'] if task['id'] == board.task['id'])
    assert listed['integration'] == integration and listed['integration_attempts'] == detail['integration_attempts']
    assert {event['state'] for event in detail['integration_events']} == {
        'scope_returned', 'candidate', 'integrated', 'cleanup_required', 'completed'}
    assert 'candidate' not in integration and 'released_claims' not in detail['integration_events'][0]


@pytest.mark.redteam
def test_blocked_integration_inspection_redacts_secrets_and_retains_authentication(board, project):
    board.board.review(board.parent, board.task['id'], accepted())
    other = board.agent('publication-owner')
    board.ws.claim(other, 'branch', '0.2dev')
    outcome = task_scheduler.integrate_completed(board.ws, board.parent, board.worker_id)[0]
    assert outcome['state'] == 'blocked'
    with GraphStore(board.base / 'coordination.db') as graph:
        node = graph.nodes('integration')[0]
        graph.upsert_node({**node, 'attrs': {**node['attrs'], 'reason':
            f"token=very-private-value {board.base}/tokens/private.token unavailable"}})
    detail = board.board.get(board.owner, board.task['id'])
    assert detail['integration']['blocking_owner'] == 'publication-owner'
    assert detail['integration']['blocking_claim'] == {'kind': 'branch', 'owner': 'publication-owner', 'key': '0.2dev'}
    exposed = json.dumps({key: detail[key] for key in
                          ('integration', 'integrations', 'integration_attempts', 'integration_events')})
    assert 'very-private-value' not in exposed and 'private.token' not in exposed
    assert 'candidate' not in detail['integration'] and 'interpreter' not in exposed
    assert task_scheduler.integrate_completed(board.ws, board.parent, board.worker_id) == []
    board.ws.registry.revoke(board.ws.auth(board.owner), 'lead')
    with pytest.raises(Denied):
        board.board.get(board.parent, board.task['id'])
    with pytest.raises(Denied):
        board.board.list(board.parent)


def test_accepted_task_is_not_done_until_local_landing_and_cleanup(board, project):
    board.board.review(board.parent, board.task['id'], accepted())
    assert board.board.get(board.parent, board.task['id'])['state'] == 'accepted'
    result = task_integration.integrate(board.ws, board.parent, board.task['id'])
    assert result['state'] == 'completed' and result['cleanup_verified']
    assert not project['source'].exists()
    assert not Path(result['candidate']).exists()
    assert repo.git(project['origin'], 'rev-parse', 'refs/heads/0.2dev') == project['baseline']
    assert task_integration.integrate(board.ws, board.parent, board.task['id']) == result


def test_unique_ignored_files_keep_task_pending_until_preserved(board, project, tmp_path):
    board.board.review(board.parent, board.task['id'], accepted())
    excluded = Path(repo.git(project['source'], 'rev-parse', '--git-path', 'info/exclude'))
    excluded.write_text('notes.private\n')
    notes = project['source'] / 'notes.private'
    notes.write_text('unique work')
    result = task_integration.integrate(board.ws, board.parent, board.task['id'])
    assert result['state'] == 'blocked' and 'requiring preservation' in result['reason']
    assert notes.read_text() == 'unique work'
    assert board.board.get(board.parent, board.task['id'])['state'] == 'accepted'
    notes.rename(tmp_path / 'preserved.private')
    result = task_integration.integrate(board.ws, board.parent, board.task['id'])
    assert result['state'] == 'completed'
    assert not project['source'].exists()
    assert (tmp_path / 'preserved.private').read_text() == 'unique work'


def test_completed_task_refuses_recreated_branch_without_deleting_it(board, project):
    board.board.review(board.parent, board.task['id'], accepted())
    result = task_integration.integrate(board.ws, board.parent, board.task['id'])
    assert result['state'] == 'completed'
    repo.git(project['primary'], 'branch', project['worktree']['branch'])
    with pytest.raises(Denied, match='branch reappeared'):
        task_integration.integrate(board.ws, board.parent, board.task['id'])
    assert repo.git(project['primary'], 'rev-parse', project['worktree']['branch']) == result['commit']


@pytest.mark.redteam
@pytest.mark.parametrize('staged', [False, True])
def test_lead_preserves_exact_reviewed_primary_bytes_before_fast_forward(board, project, staged, monkeypatch):
    board.board.review(board.parent, board.task['id'], accepted())
    primary = project['primary']
    content = repo.git(project['source'], 'show', 'HEAD:sim.py', binary=True)
    (primary / 'sim.py').write_bytes(content)
    if staged:
        repo.git(primary, 'add', '--', 'sim.py')
    git = repo.git

    def read_only_primary(root, *arguments, **kwargs):
        assert root != primary or arguments[0] not in ('add', 'commit', 'restore', 'reset', 'checkout')
        return git(root, *arguments, **kwargs)

    monkeypatch.setattr(repo, 'git', read_only_primary)
    if not staged:
        index = primary / git(primary, 'rev-parse', '--git-path', 'index')
        before = index.read_bytes()
        with pytest.raises(Denied, match='already staged'):
            task_integration.integrate(board.ws, board.parent, board.task['id'])
        assert index.read_bytes() == before and (primary / 'sim.py').read_bytes() == content
        assert not project['log'].exists()
        return
    result = task_integration.integrate(board.ws, board.parent, board.task['id'])
    assert result['state'] == 'completed'
    assert (primary / 'sim.py').read_bytes() == content
    assert repo.git(primary, 'status', '--porcelain') == ''
    assert board.ws.who_owns('file', str(primary / 'sim.py')) is None


@pytest.mark.redteam
@pytest.mark.parametrize('change', ['working', 'staged', 'untracked', 'child'])
def test_unreviewed_primary_bytes_and_child_staging_are_refused(board, project, change):
    board.board.review(board.parent, board.task['id'], accepted())
    primary = project['primary']
    (primary / 'sim.py').write_text('speed = 2\n')
    token = board.parent
    if change == 'working':
        (primary / 'sim.py').write_text('speed = 99\n')
    elif change == 'staged':
        (primary / 'sim.py').write_text('speed = 99\n')
        repo.git(primary, 'add', '--', 'sim.py')
        (primary / 'sim.py').write_text('speed = 2\n')
    elif change == 'untracked':
        (primary / 'unknown.py').write_text('unknown = True\n')
    else:
        token = board.child
    before = repo.git(primary, 'diff', '--cached', binary=True)
    with pytest.raises(Denied):
        task_integration.integrate(board.ws, token, board.task['id'])
    assert repo.git(primary, 'diff', '--cached', binary=True) == before
    assert repo.git(primary, 'rev-parse', 'HEAD') == project['baseline']
    assert not project['log'].exists()


@pytest.mark.redteam
def test_primary_preservation_verification_keeps_index_bytes(board, project):
    primary = project['primary']
    content = repo.git(project['source'], 'show', 'HEAD:sim.py', binary=True)
    (primary / 'sim.py').write_bytes(content)
    repo.git(primary, 'add', '--', 'sim.py')
    index = primary / repo.git(primary, 'rev-parse', '--git-path', 'index')
    before = index.read_bytes()
    names = integration_staging.files(primary, project['proposal']['provenance']['commit'])
    integration_staging.verify(primary, project['proposal']['provenance']['commit'], names)
    assert index.read_bytes() == before


@pytest.mark.redteam
def test_primary_preservation_refuses_symlink_bytes(board, project, monkeypatch):
    primary = project['primary']
    target = primary / 'sim.py'
    target.write_bytes(repo.git(project['source'], 'show', 'HEAD:sim.py', binary=True))
    repo.git(primary, 'add', '--', 'sim.py')
    is_symlink = Path.is_symlink
    monkeypatch.setattr(Path, 'is_symlink', lambda path: path == target or is_symlink(path))
    with pytest.raises(Denied, match='symlinks'):
        integration_staging.files(primary, project['proposal']['provenance']['commit'])


@pytest.mark.redteam
@pytest.mark.parametrize('contained', [False, True])
def test_unfinished_merge_metadata_requires_reviewed_commit_containment(board, project, contained):
    primary = project['primary']
    commit = project['proposal']['provenance']['commit']
    marker = primary / repo.git(primary, 'rev-parse', '--git-path', 'MERGE_HEAD')
    marker.write_text((commit if contained else '0' * 40) + '\n')
    (primary / 'sim.py').write_bytes(repo.git(project['source'], 'show', 'HEAD:sim.py', binary=True))
    repo.git(primary, 'add', '--', 'sim.py')
    if not contained:
        with pytest.raises((Denied, RuntimeError)):
            integration_staging.files(primary, commit)
        assert marker.exists()
        return
    names = integration_staging.files(primary, commit)
    index = primary / repo.git(primary, 'rev-parse', '--git-path', 'index')
    before = index.read_bytes()
    integration_staging.verify(primary, commit, names)
    assert marker.exists() and index.read_bytes() == before
    board.board.review(board.parent, board.task['id'], accepted())
    result = task_integration.integrate(board.ws, board.parent, board.task['id'])
    assert result['state'] == 'completed' and not marker.exists()


@pytest.mark.redteam
def test_cleanup_refuses_a_foreign_worktree_lock(board, project):
    source = project['source']
    locked = source / repo.git(source, 'rev-parse', '--git-path', 'locked')
    repo.git(project['primary'], 'worktree', 'lock', '--reason', 'foreign live operation', str(source))
    board.board.review(board.parent, board.task['id'], accepted())
    result = task_integration.integrate(board.ws, board.parent, board.task['id'])
    assert result['state'] == 'blocked' and 'lock owned by another operation' in result['reason']
    assert source.exists() and locked.read_text().strip() == 'foreign live operation'
