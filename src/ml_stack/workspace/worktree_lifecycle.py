"""Durable ownership and completion checks for isolated coding checkouts."""

import hashlib
import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path

from ml_stack import worktreerules
from ml_stack.files import promote
from ml_stack.graph.store import GraphStore
from ml_stack.workspace import integration_git as repo
from ml_stack.workspace.chain import held
from ml_stack.workspace.identity import Denied


@contextmanager
def _storage(base: Path, *, write: bool = False):
    database = base / 'worktree-lifecycle.db'
    with held(base / 'worktree-lifecycle.lock'):
        if not write:
            with GraphStore(database, read_only=True) as graph:
                yield graph
            return
        if any(Path(str(database) + suffix).exists() for suffix in ('.wal', '.wal.checkpoint', '.shadow')):
            raise Denied('lifecycle storage requires preserved checkpoint recovery')
        staging = Path(tempfile.mkdtemp(prefix='worktree-lifecycle-stage-', dir=base))
        staged = staging / database.name
        if database.exists():
            shutil.copyfile(database, staged)
        with GraphStore(staged) as graph:
            yield graph
            expected = (graph.nodes(), graph.edges())
        with GraphStore(staged, read_only=True) as graph:
            actual = (graph.nodes(), graph.edges())
        def normalize(rows):
            return sorted(rows, key=lambda row: repr(sorted(row.items())))
        if tuple(map(normalize, actual)) != tuple(map(normalize, expected)):
            raise Denied('lifecycle staged checkpoint differs from the recorded graph')
        if any(Path(str(staged) + suffix).exists() for suffix in ('.wal', '.wal.checkpoint', '.shadow')):
            raise Denied('lifecycle staged checkpoint is incomplete; recovery evidence preserved')
        staged.chmod(0o600)
        with staged.open('rb') as source:
            os.fsync(source.fileno())
        promote(staged, database)
        if os.name != 'nt':
            directory = os.open(base, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        shutil.rmtree(staging)


def scopes(base: Path, owner: str, label: str = '') -> list[dict]:
    database = base / 'worktree-lifecycle.db'
    if not database.exists():
        return []
    with _storage(base) as graph:
        return [row['attrs'] for row in graph.nodes('worktree-lifecycle')
                if row['attrs']['owner'] == owner
                and (not label or row['attrs']['label'] in ('', label))]


def remember(base: Path, owner: str, label: str, path: str) -> None:
    """Record an authenticated worker's reserved or materialized coding checkout."""
    found = worktreerules.checkouts(path)
    if (found and found[0] == found[1]) or (not found and Path(path).exists()):
        return
    checkout = found[0] if found else Path(path).resolve()
    primary = found[1] if found else None
    branch = repo.git(checkout, 'branch', '--show-current') if found else ''
    if found and not branch:
        raise Denied('coding worktree ownership requires a named branch')
    key = 'worktree-lifecycle:' + hashlib.sha256(f'{owner}:{label}:{checkout}'.encode()).hexdigest()
    commit = repo.git(checkout, 'rev-parse', 'HEAD') if found else ''
    development = repo.git(primary, 'branch', '--show-current') if primary else ''
    value = {'owner': owner, 'label': label, 'path': str(checkout),
             'primary': str(primary) if primary else '', 'branch': branch, 'development': development}
    def merged(graph) -> tuple[dict, dict]:
        previous = next((row['attrs'] for row in graph.nodes('worktree-lifecycle') if row['id'] == key), {})
        wanted = dict(value)
        if not found and previous.get('primary'):
            wanted.update({name: previous[name] for name in ('primary', 'branch', 'development')})
        branches = sorted(set(previous.get('branches', [])) | ({branch} if branch else set()))
        commits = sorted(set(previous.get('commits', [])) | ({commit} if commit else set()))
        return previous, {**wanted, 'branches': branches, 'commits': commits}

    if (base / 'worktree-lifecycle.db').exists():
        with _storage(base) as graph:
            previous, attrs = merged(graph)
        if previous == attrs:
            return                    # already recorded: the checkpoint rewrite is the whole cost
    with _storage(base, write=True) as graph:
        attrs = merged(graph)[1]
        graph.upsert_node({'id': key, 'kind': 'worktree-lifecycle', 'label': label, 'attrs': attrs})


def checkpoint(base: Path, owner: str) -> None:
    """Capture current commits from the authenticated worker's surviving scopes."""
    for scope in scopes(base, owner):
        path = Path(scope['path'])
        if path.exists():
            remember(base, owner, scope['label'], str(path))


def record_cleanup(base: Path, path: Path, proof: tuple[str, str],
                   primary: Path, development: str) -> None:
    """Record a completed maintained removal with its captured source and landed commits."""
    commit, landed = proof
    with _storage(base, write=True) as graph:
        for row in graph.nodes('worktree-lifecycle'):
            scope = row['attrs']
            if scope['path'] == str(path):
                if path.exists():
                    raise Denied('cleanup proof requires an absent checkout')
                if scope['primary'] and Path(scope['primary']) != primary:
                    raise Denied('cleanup proof repository differs from the attributed checkout')
                if not repo.ancestor(primary, commit, landed):
                    raise Denied('cleanup proof requires the exact source commit to be landed')
                attrs = {**scope, 'primary': str(primary), 'development': development,
                         'commits': sorted(set(scope['commits']) | {commit}),
                         'cleanup': {'commit': commit, 'landed': landed}}
                graph.upsert_node({**row, 'attrs': attrs})


def cleanup(base: Path, owner: str, path: str, claims, *, claim_owner: str = '') -> dict:
    """Remove a claimed landed checkout and record its final commit after verified cleanup."""
    target = Path(path).resolve()
    owned = [scope for scope in scopes(base, owner) if scope['path'] == str(target)]
    if not owned or not target.exists():
        raise Denied('cleanup requires this worker\'s existing attributed checkout')
    claim = claims.who('worktree', str(target))
    if not claim or claim['owner'] != (claim_owner or owner):
        raise Denied('cleanup requires the live checkout claim of this worker')
    checkpoint(base, owner)
    owned = [scope for scope in scopes(base, owner) if scope['path'] == str(target)]
    source = owned[0]
    primary = Path(source['primary'])
    if worktreerules.checkouts(target) != (target, primary):
        raise Denied('cleanup checkout identity changed')
    branch = repo.git(target, 'branch', '--show-current')
    if branch not in source['branches'] or branch in ('main', source['development']):
        raise Denied('cleanup requires an attributed isolated branch')
    locked = target / repo.git(target, 'rev-parse', '--git-path', 'locked')
    if locked.exists():
        raise Denied('cleanup preserves locked worktrees')
    commit = repo.git(target, 'rev-parse', 'HEAD')
    landed = repo.git(primary, 'rev-parse', f"refs/heads/{source['development']}")
    repo.remove_merged(primary, target, branch, landed)
    record_cleanup(base, target, (commit, landed), primary, source['development'])
    return {'path': str(target), 'commit': commit, 'landed': landed, 'cleanup_verified': True}


def pending(base: Path, owner: str, label: str = '') -> list[dict]:
    """Inspect the worker's durable scopes without removing files or Git references."""
    result = []
    for scope in scopes(base, owner, label):
        path = Path(scope['path'])
        if scope['primary'] and path.exists():
            remember(base, scope['owner'], scope['label'], str(path))
            scope = next(row for row in scopes(base, owner, label) if row['path'] == str(path)
                         and row['label'] == scope['label'])
        if not scope['primary']:
            found = worktreerules.checkouts(path)
            if not found or found[0] == found[1]:
                if path.exists():
                    result.append({**scope, 'reasons': ['reserved checkout path remains']})
                continue
            remember(base, scope['owner'], scope['label'], scope['path'])
            with _storage(base) as graph:
                scope = next(row['attrs'] for row in graph.nodes('worktree-lifecycle')
                             if row['attrs']['path'] == str(found[0])
                             and all(row['attrs'][key] == scope[key] for key in ('owner', 'label')))
            path = Path(scope['path'])
        primary = Path(scope['primary'])
        registered = repo.git(primary, 'worktree', 'list', '--porcelain').splitlines()
        refs = repo.git(primary, 'for-each-ref', '--format=%(refname)', 'refs/heads/').splitlines()
        reasons = []
        tip = repo.git(primary, 'rev-parse', f"refs/heads/{scope['development']}")
        unlanded = [commit for commit in scope['commits'] if not repo.ancestor(primary, commit, tip)]
        if unlanded:
            reasons.append('recorded commits are not landed: ' + ', '.join(unlanded))
        if path.exists():
            reasons.append('checkout remains')
        elif not scope.get('cleanup'):
            reasons.append('verified cleanup proof is missing')
        if any(line.startswith('worktree ') and Path(line[9:]).resolve() == path
               for line in registered):
            reasons.append('worktree registration remains')
        remaining = [branch for branch in scope['branches'] if f'refs/heads/{branch}' in refs]
        if remaining:
            reasons.append('branches remain: ' + ', '.join(remaining))
        if reasons:
            result.append({**scope, 'reasons': reasons})
    return result


def require_clean(base: Path, owner: str, label: str = '') -> None:
    """Refuse completion while an attributed checkout, registration or branch remains."""
    scopes = pending(base, owner, label)
    if scopes:
        detail = '; '.join(f"{row['path']} ({row['branch']}): {', '.join(row['reasons'])}" for row in scopes)
        raise Denied('completion requires landed work and verified cleanup: ' + detail)
