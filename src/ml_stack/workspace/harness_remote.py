"""Native mutation identities and physical checkout scopes."""

import hashlib
import os
import uuid
from pathlib import Path

from ml_stack import worktreerules
from ml_stack.guard.shellscan import has_expansion, has_glob, segments
from ml_stack.harnesspolicy import SHELL_TOOLS, _shell_line
from ml_stack.workspace import (
    integration_git as repo,
    limits,
    project_connection,
    worktree_lifecycle,
)
from ml_stack.workspace.chain import held
from ml_stack.workspace.claims import Claims, normal
from ml_stack.workspace.identity import AGENT, Denied, Identity


def context(actor, cwd, roots, *, require_claim=True):
    """Authenticate a worker and the launcher's fixed project roots."""
    selected = project_connection.selected(Path(cwd))
    if selected is None:
        return None
    for root in roots:
        found = project_connection.selected(Path(root))
        if not found or any(found.get(key) != selected.get(key) for key in ('host', 'project_id')):
            raise Denied('native roots must belong to the selected project')
    remote = project_connection.RemoteWorkspace(
        selected['host'], selected['project_id'], cluster=selected.get('cluster', ''),
        cluster_key=Path(selected['cluster_key']) if selected.get('cluster_key') else None)
    info = remote.call('whoami', remote.token(agent=actor))
    if info.get('id') != actor or info.get('role') != AGENT or info.get('project', {}).get('key') != remote.project_id:
        raise Denied('native ownership requires the launcher-bound agent identity')
    who = Identity(actor, AGENT, info.get('parent', ''), tuple(info.get('can', ())))
    if require_claim and 'claim' not in who.can:
        raise Denied('agent capability has no mutation claim permission')
    return remote, who


def physical_owner(remote, who):
    """Namespace the authenticated identity in the shared physical claim store."""
    authority = hashlib.sha256(f'{remote.host}/{remote.project_id}'.encode()).hexdigest()[:32]
    return Identity(f'board:{authority}:{who.id}', AGENT, can=who.can)


def claims():
    """Open the shared local physical ownership store."""
    base = limits.root()
    return Claims(base, limits.load(base).claim_ttl_s)


def source_resources(remote, required, *, branch_only=False):
    """Map authenticated local targets to repository-relative source claims."""
    source = []
    for kind, key in required:
        if kind == 'branch':
            source.append((kind, key))
        elif kind in ('file', 'worktree'):
            target = Path(key)
            found = worktreerules.checkouts(target.parent if target.is_file() else target)
            if not found:
                raise Denied('source mutation requires a Git checkout')
            bound = project_connection.selected(found[0])
            if not bound or (bound.get('host'), bound.get('project_id')) != (remote.host, remote.project_id):
                raise Denied('mutation target belongs to another project')
            relative = target.relative_to(found[0]).as_posix()
            if kind == 'worktree' and branch_only:
                branch = repo.git(found[0], 'branch', '--show-current')
                if not branch:
                    raise Denied('Git mutations require a named branch')
                source.append(('branch', branch))
            elif kind == 'worktree' or relative == '.':
                raise Denied('mutations must name bounded source paths')
            else:
                source.append(('area', relative))
    return list(dict.fromkeys(source))


def require_clean(remote, who):
    """Inspect this worker's local durable checkout namespace."""
    worktree_lifecycle.require_clean(remote.base, who.id)


def _rollback(store, made, previous):
    with held(store.lock):
        current = store._load()
        for row in made:
            key = f"{row['kind']}:{row['key']}"
            if key not in previous and current.get(key) == row:
                current.pop(key)
        store._save(current)


def physical_resources(required):
    resources = [(kind, key) for kind, key in required if kind != 'branch']
    resources.extend(('area', normal('area', key)) for kind, key in required if kind == 'file')
    return list(dict.fromkeys(resources))


def inspect_shell(name, args):
    if name not in SHELL_TOOLS:
        return
    commands, findings = segments(_shell_line(name, args))
    if findings or any(has_expansion(word) or has_glob(word) for command in commands
                       for word in (*command.argv, *(target for _, target in command.redirects))):
        raise Denied('shell mutations require fixed inspectable targets')


def reserve(remote, who, required, fields, *, branch_only=False):
    """Reserve physical targets and source areas before a mutation."""
    physical = physical_resources(required)
    source = source_resources(remote, required, branch_only=branch_only)
    store = claims()
    provenance = {**fields, 'note': f"[{uuid.uuid4().hex}] {fields.get('note', '')}"}
    made, previous = store.reserve(physical_owner(remote, who), physical, provenance, with_previous=True)
    try:
        if source:
            remote.native_reserve(who.id, source)
    except (OSError, RuntimeError, ValueError):
        _rollback(store, made, previous)
        raise
    for kind, key in physical:
        if kind in ('file', 'worktree'):
            target = Path(key)
            worktree_lifecycle.remember(remote.base, who.id,
                                        str(target.parent if target.is_file() else target))


def conflict(remote, who, required):
    """Return a foreign physical claim before admission runs."""
    store = claims()
    principal = physical_owner(remote, who)
    for kind, key in physical_resources(required):
        owner = store.who(kind, key)
        if owner and owner['owner'] != principal.id:
            return f"{kind} {key} belongs to {owner['owner']}"
    return ''


def staging_only(name, args):
    """Identify shell commands that only stage paths or commit the current branch."""
    selectors = {'GIT_DIR', 'GIT_WORK_TREE', 'GIT_COMMON_DIR', 'GIT_INDEX_FILE',
                 'GIT_OBJECT_DIRECTORY', 'GIT_ALTERNATE_OBJECT_DIRECTORIES', 'GIT_NAMESPACE', 'GIT_PREFIX'}
    if name not in SHELL_TOOLS or any(key in selectors or key.startswith('GIT_CONFIG') for key in os.environ):
        return False
    commands, findings = segments(_shell_line(name, args))
    if findings or not commands:
        return False
    for command in commands:
        words = list(command.argv)
        if (not words or words[0] != 'git' or command.redirects or command.piped
                or words.count('-C') > 1 or any(has_expansion(word) or has_glob(word) for word in words)):
            return False
        at = 1
        while at < len(words) and words[at] in ('-C', '--no-pager'):
            at += 1 if words[at] == '--no-pager' else 2
        if at >= len(words) or words[at] not in ('add', 'commit'):
            return False
    return True


def revocation_snapshot(remote, identity):
    """Capture exact physical claim records before requesting self revocation."""
    store = claims()
    principal = physical_owner(remote, Identity(identity, AGENT))
    with held(store.lock):
        return {key: row for key, row in store._load().items() if row['owner'] == principal.id}


def release_revoked(remote, identity, snapshot):
    """Remove unchanged physical claims after self revocation succeeds."""
    store = claims()
    principal = physical_owner(remote, Identity(identity, AGENT))
    with held(store.lock):
        current = store._load()
        for key, row in snapshot.items():
            if row['owner'] == principal.id and current.get(key) == row:
                current.pop(key)
        store._save(current)


LOCAL_KINDS = ('port', 'server', 'install')


def local_command(remote, who, args):
    """Claim, release or look up a device-local port, server or install in the physical store."""
    kind, key = args.kind, args.key
    if kind == 'install':
        key = str((Path.cwd() / key).resolve())
    key = normal(kind, key)
    store, principal = claims(), physical_owner(remote, who)
    if args.cmd == 'who':
        return store.who(kind, key) or {'owner': None}
    if args.cmd == 'release':
        return {'released': [store.release(principal, kind, key)]}
    if args.ttl or args.pid:
        raise Denied('reservation uses bounded defaults; renew TTL with heartbeat')
    store.reserve(principal, [(kind, key)], {'note': args.note})
    return {'reserved': [{'kind': kind, 'key': key}]}


def cli_command(remote, token, args):
    """Dispatch project-confined claim operations under the exact CLI identity."""
    info = remote.call('whoami', token)
    selected = project_connection.selected()
    expected = args.agent or (selected or {}).get('agent', '')
    if (not expected or info.get('id') != expected or info.get('role') != AGENT
            or info.get('project', {}).get('key') != remote.project_id):
        raise Denied('claims require the selected exact agent identity')
    who = Identity(expected, AGENT, info.get('parent', ''), tuple(info.get('can', ())))
    if args.cmd == 'worktrees':
        if getattr(args, 'cleanup', ''):
            if 'claim' not in who.can:
                raise Denied('capability has no claim permission')
            principal = physical_owner(remote, who)
            store = claims()
            target = str(Path(args.cleanup).resolve())
            claim = store.who('worktree', target)
            if not claim or claim['owner'] != principal.id:
                raise Denied('cleanup requires the live checkout claim of this worker')
            return worktree_lifecycle.cleanup(remote.base, who.id, target, store, claim_owner=principal.id)
        return worktree_lifecycle.pending(remote.base, who.id)
    if args.cmd in ('announce', 'send'):
        require_clean(remote, who)
        return None
    if 'claim' not in who.can:
        raise Denied('capability has no claim permission')
    if args.cmd == 'heartbeat':
        rows = remote.call('native.heartbeat', token, ttl_s=args.ttl)
        local = claims().renew(physical_owner(remote, who), args.ttl)
        rows = [*rows, *local]
        return {'renewed': len(rows), 'capped': [f"{r['kind']}:{r['key']}" for r in rows if r.get('capped')]}
    kind, key = args.kind, args.key
    if kind in LOCAL_KINDS:
        return local_command(remote, who, args)
    if kind not in ('branch', 'area', 'file', 'worktree'):
        raise Denied('CLI claims require a branch, a bounded checkout path, a port, a server or an install')
    required = [(kind, key)]
    if kind == 'area':
        required = [('file', str((Path.cwd() / key).resolve()))]
    if kind in ('file', 'worktree'):
        required = [(kind, str((Path.cwd() / key).resolve()))]
    source = source_resources(remote, required, branch_only=kind == 'worktree')
    if args.cmd == 'who':
        rows = [remote.call('who', token, k, v) for k, v in source]
        return next((row for row in rows if row), {'owner': None})
    if args.cmd == 'release':
        keys = {f'{k}:{normal(k, v)}' for k, v in physical_resources(required)}
        snapshot = {k: row for k, row in revocation_snapshot(remote, who.id).items() if k in keys}
        rows = [remote.native_release(who.id, k, v) for k, v in source]
        release_revoked(remote, who.id, snapshot)
        return {'released': rows}
    if args.ttl or args.pid:
        raise Denied('reservation uses bounded defaults; renew TTL with heartbeat')
    reserve(remote, who, required, {'note': args.note}, branch_only=kind == 'worktree')
    return {'reserved': [{'kind': k, 'key': v} for k, v in source]}
