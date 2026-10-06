"""Durable ownership and completion checks for isolated coding checkouts."""

import hashlib
from pathlib import Path

from ml_stack import worktreerules
from ml_stack.graph.store import GraphStore
from ml_stack.workspace import integration_git as repo
from ml_stack.workspace.chain import held
from ml_stack.workspace.identity import Denied


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
    with held(base / 'worktree-lifecycle.lock'), GraphStore(base / 'worktree-lifecycle.db') as graph:
        previous = next((row['attrs'] for row in graph.nodes('worktree-lifecycle') if row['id'] == key), {})
        branches = sorted(set(previous.get('branches', [])) | ({branch} if branch else set()))
        commits = sorted(set(previous.get('commits', [])) | ({commit} if commit else set()))
        graph.upsert_node({'id': key, 'kind': 'worktree-lifecycle', 'label': label,
                           'attrs': {**value, 'branches': branches, 'commits': commits}})


def pending(base: Path, owner: str, label: str = '') -> list[dict]:
    """Inspect the worker's durable scopes without removing files or Git references."""
    database = base / 'worktree-lifecycle.db'
    if not database.exists():
        return []
    with held(base / 'worktree-lifecycle.lock'), GraphStore(database) as graph:
        scopes = [row['attrs'] for row in graph.nodes('worktree-lifecycle')
                  if row['attrs']['owner'] == owner
                  and (not label or row['attrs']['label'] in ('', label))]
    result = []
    for scope in scopes:
        path = Path(scope['path'])
        if not scope['primary']:
            found = worktreerules.checkouts(path)
            if not found:
                if path.exists():
                    result.append({**scope, 'reasons': ['reserved checkout path remains']})
                continue
            branch = repo.git(found[0], 'branch', '--show-current')
            scope = {**scope, 'primary': str(found[1]), 'branch': branch,
                     'branches': [branch] if branch else [],
                     'development': repo.git(found[1], 'branch', '--show-current')}
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
