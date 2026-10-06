"""Durable ownership and completion checks for isolated coding checkouts."""

import hashlib
from pathlib import Path

from ml_stack import worktreerules
from ml_stack.graph.store import GraphStore
from ml_stack.workspace import integration_git as repo
from ml_stack.workspace.chain import held
from ml_stack.workspace.identity import Denied


def remember(base: Path, owner: str, label: str, path: str) -> None:
    """Record an authenticated worker's existing linked checkout and branch."""
    found = worktreerules.checkouts(path)
    if not found or found[0] == found[1]:
        return
    checkout, primary = found
    branch = repo.git(checkout, 'branch', '--show-current')
    if not branch:
        raise Denied('coding worktree ownership requires a named branch')
    key = 'worktree-lifecycle:' + hashlib.sha256(f'{owner}:{label}:{checkout}'.encode()).hexdigest()
    value = {'owner': owner, 'label': label, 'path': str(checkout),
             'primary': str(primary), 'branch': branch}
    with held(base / 'coordination.lock'), GraphStore(base / 'coordination.db') as graph:
        previous = next((row['attrs'] for row in graph.nodes('worktree-lifecycle') if row['id'] == key), {})
        branches = sorted(set(previous.get('branches', [])) | {branch})
        graph.upsert_node({'id': key, 'kind': 'worktree-lifecycle', 'label': label,
                           'attrs': {**value, 'branches': branches}})


def pending(base: Path, owner: str, label: str = '') -> list[dict]:
    """Inspect the worker's durable scopes without removing files or Git references."""
    database = base / 'coordination.db'
    if not database.exists():
        return []
    with held(base / 'coordination.lock'), GraphStore(database) as graph:
        scopes = [row['attrs'] for row in graph.nodes('worktree-lifecycle')
                  if row['attrs']['owner'] == owner
                  and (not label or row['attrs']['label'] == label)]
    result = []
    for scope in scopes:
        path, primary = Path(scope['path']), Path(scope['primary'])
        registered = repo.git(primary, 'worktree', 'list', '--porcelain').splitlines()
        refs = repo.git(primary, 'for-each-ref', '--format=%(refname)', 'refs/heads/').splitlines()
        reasons = []
        if path.exists():
            reasons.append('checkout remains')
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
