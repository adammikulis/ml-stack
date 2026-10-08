"""Who owns which branch, worktree, port, file, area, install environment or server."""

from __future__ import annotations

import os
import re
import time
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from ml_stack import worktreerules
from ml_stack.files import read_json, write_json
from ml_stack.serve.process import pid_exists, started_at
from ml_stack.workspace.chain import held
from ml_stack.workspace.identity import AGENT, HUMAN, Denied, Identity

__all__ = ["EXPIRING_SOON_S", "KINDS", "MAX_LIFETIME_S", "MAX_RENEW_S", "Claims", "Conflict", "alive", "normal"]

KINDS = ("branch", "worktree", "port", "file", "server", "install", "area")
PATH_KINDS = ("worktree", "file", "install")
WORD = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/@:+-]{0,199}$")
VERSION = 1
EXPIRING_SOON_S = 300.0
MAX_RENEW_S = 3600.0
MAX_LIFETIME_S = 8 * 3600.0


class Conflict(RuntimeError):
    """Something else owns what was asked for; ``owner`` is its claim."""

    def __init__(self, message: str, owner: dict[str, Any]) -> None:
        super().__init__(message)
        self.owner = owner


def alive(pid: int) -> bool:
    """Whether a process with this pid exists."""
    if os.name == "nt":
        return pid_exists(pid)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _holder_alive(claim: dict[str, Any]) -> bool:
    """Whether the claim's pid runs and, when its start time was recorded, is the process that took it."""
    pid = int(claim["pid"])
    if not alive(pid):
        return False
    recorded = claim.get("pid_started")
    seen = started_at(pid) if recorded else None
    return not recorded or seen is None or abs(seen - float(recorded)) < 2.0


def normal(kind: str, key: str) -> str:
    """The canonical spelling of ``key`` for ``kind``; raises ValueError for one that is not valid."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {', '.join(KINDS)}")
    if kind == "port":
        if not key.isdigit() or not 0 < int(key) < 65536:
            raise ValueError("a port is a number from 1 to 65535")
        return str(int(key))
    if kind == 'area':
        expanded = Path(key).expanduser()
        if not expanded.is_absolute():
            raise ValueError('an area claim needs an absolute project path')
        target = expanded.resolve()
        directory = next((parent for parent in (target, *target.parents) if parent.is_dir()), None)
        checkout = worktreerules.checkouts(directory) if directory else None
        if not checkout:
            raise ValueError('an area claim needs a Git checkout')
        top, common = checkout
        return str(common / target.relative_to(top))
    if kind in PATH_KINDS:
        expanded = Path(key).expanduser()
        if not expanded.is_absolute():
            raise ValueError("a path claim needs an absolute path")
        return os.path.realpath(expanded)
    if not WORD.match(key) or ".." in key:
        raise ValueError(f"{key!r} is not a usable {kind} name")
    return key


def _nested(a: str, b: str) -> bool:
    return a == b or a.startswith(b + os.sep) or b.startswith(a + os.sep)


def _covers(claim: dict[str, Any], kind: str, key: str) -> bool:
    if kind == 'area':
        return claim['kind'] == 'area' and _nested(claim['key'], key)
    if kind in PATH_KINDS:
        return claim["kind"] in PATH_KINDS and _nested(claim["key"], key)
    return claim["kind"] == kind and claim["key"] == key


class Claims:
    """Claims with a TTL renewed by heartbeat; one whose pid has died is released on next look.

    ``on_swept(claim)`` hears each claim released as expired or dead; ``on_stolen(old, new)``
    hears a claim taken by another owner over one that had expired or died.
    """

    def __init__(self, base: Path, ttl_s: float, clock: Callable[[], float] = time.time,
                 on_swept: Callable[[dict[str, Any]], None] | None = None,
                 on_stolen: Callable[[dict[str, Any], dict[str, Any]], None] | None = None) -> None:
        self.path = base / "claims.json"
        self.lock = base / "claims.lock"
        self.ttl_s, self.clock, self.on_swept, self.on_stolen = ttl_s, clock, on_swept, on_stolen

    def _load(self) -> dict[str, dict[str, Any]]:
        data = read_json(self.path, {})
        found = data.get("claims") if isinstance(data, dict) else None
        return dict(found) if isinstance(found, dict) else {}

    def _save(self, claims: dict[str, dict[str, Any]]) -> None:
        write_json(self.path, {"version": VERSION, "claims": claims})
        self.path.chmod(0o600)

    def _sweep(self, claims: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
        now = self.clock()
        dead = [{**c, "reason": "expired" if c["expires"] <= now else "dead-pid"}
                for c in claims.values()
                if c["expires"] <= now or (c["pid"] and not _holder_alive(c))]
        for claim in dead:
            claims.pop(f"{claim['kind']}:{claim['key']}", None)
            if self.on_swept:
                self.on_swept(claim)
        return dead

    def claim(self, who: Identity, kind: str, key: str,
              fields: Mapping[str, Any] | None = None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Take ``key``; ``fields`` may hold ``ttl_s``, ``pid`` and ``note``. Returns the claim
        and any dead claims that were released on the way."""
        ttl_s, pid = float((fields or {}).get("ttl_s", 0.0)), int((fields or {}).get("pid", 0))
        note = str((fields or {}).get("note", ""))
        key = normal(kind, key)
        with held(self.lock):
            claims = self._load()
            swept = self._sweep(claims)
            if swept:
                self._save(claims)
            for other in claims.values():
                if _covers(other, kind, key) and other["owner"] != who.id:
                    raise Conflict(f"{kind} {key} is held by {other['owner']} until "
                                   f"{time.strftime('%H:%M:%S', time.localtime(other['expires']))}",
                                   other)
            now = self.clock()
            made = {"kind": kind, "key": key, "owner": who.id, "pid": int(pid), "since": now,
                    "expires": now + (ttl_s or self.ttl_s), "note": note[:200]}
            if pid:
                made["pid_started"] = float((fields or {}).get("pid_started") or started_at(int(pid)) or 0.0)
            claims[f"{kind}:{key}"] = made
            self._save(claims)
            if self.on_stolen:
                for old in swept:
                    if old["owner"] != who.id and _covers(old, kind, key):
                        self.on_stolen(old, made)
            return made, swept

    def reserve(self, who: Identity, resources: list[tuple[str, str]],
                fields: Mapping[str, Any] | None = None, *, with_previous: bool = False):
        """Atomically reserve a mutation's resources or refuse its entire conflicting set."""
        if len(resources) > 128:
            raise ValueError('a mutation reserves at most 128 resources')
        resources = list(dict.fromkeys((kind, normal(kind, key)) for kind, key in resources))
        with held(self.lock):
            claims = self._load()
            self._sweep(claims)
            for kind, key in resources:
                for other in claims.values():
                    if _covers(other, kind, key) and other['owner'] != who.id:
                        raise Conflict(f"{kind} {key} belongs to {other['owner']}", other)
                    if _covers(other, kind, key) and other.get('assignment') \
                            and other['assignment'] != (fields or {}).get('assignment'):
                        raise Denied('the resource is reserved for a different task assignment')
            previous = {f'{kind}:{key}' for kind, key in resources if f'{kind}:{key}' in claims}
            now, made = self.clock(), []
            for kind, key in resources:
                old = claims.get(f'{kind}:{key}')
                since = old['since'] if old else now
                expiry = min(now + self.ttl_s, since + MAX_LIFETIME_S)
                if expiry <= now:
                    raise Denied('ownership lifetime exhausted; release the resource before continuing')
                entry = {'kind': kind, 'key': key, 'owner': who.id, 'pid': 0, 'since': since,
                         'expires': expiry, 'note': str((fields or {}).get('note', ''))[:200]}
                entry.update({key: value for key, value in (fields or {}).items()
                              if key in ('owner_pid', 'owner_started', 'interpreter', 'environment', 'commit')})
                if kind in ('file', 'area', 'worktree'):
                    entry.update({key: value for key, value in (fields or {}).items()
                                  if key in ('assignment', 'task', 'project')})
                if kind == 'worktree' and old and old.get('delegated_by'):
                    entry['delegated_by'] = old['delegated_by']
                claims[f'{kind}:{key}'] = entry
                made.append(entry)
            self._save(claims)
            return (made, previous) if with_previous else made

    def handoff(self, who: Identity, kind: str, key: str, owner: str, assignment: str) -> dict[str, Any]:
        """Transfer one parent's exact claim under an already verified task assignment."""
        key = normal(kind, key)
        if kind != 'worktree' or who.parent != owner:
            raise Denied('only an explicit parent worktree assignment can be handed off')
        with held(self.lock):
            claims = self._load()
            self._sweep(claims)
            claim = claims.get(f'{kind}:{key}')
            if claim is None or claim['owner'] not in (owner, who.id):
                raise Denied('the assigned worktree has no matching parent ownership claim')
            if claim['owner'] == owner:
                claim.update(owner=who.id, delegated_by=owner, assignment=assignment)
                self._save(claims)
            return dict(claim)

    def return_worktree(self, who: Identity, scope: dict[str, Any]) -> dict[str, Any]:
        """Return one child's exact claim after its canonical acceptance was independently checked."""
        key = normal('worktree', scope['project'])
        if who.role != HUMAN and who.id != scope['owner']:
            raise Denied('only the registered task parent or person may return its reviewed worktree')
        with held(self.lock):
            claims = self._load()
            self._sweep(claims)
            claim = claims.get(f'worktree:{key}')
            if not claim or claim['owner'] not in (scope['owner'], scope['worker']):
                raise Denied('the reviewed worktree has a different ownership claim')
            if claim['owner'] == scope['owner'] and claim.get('reviewed_assignment') != scope['id']:
                raise Denied('the returned worktree does not match this exact task delegation')
            if claim['owner'] == scope['worker']:
                if claim.get('delegated_by') != scope['owner'] or claim.get('assignment') != scope['id']:
                    raise Denied('the worktree claim does not match this exact task delegation')
                claim.update(owner=scope['owner'], returned_by=who.id, reviewed_assignment=scope['id'])
                claim.pop('delegated_by', None)
                claim.pop('assignment', None)
            released = [value for value in claims.values()
                        if value['kind'] in ('file', 'area') and value['owner'] == scope['worker']
                        and value.get('assignment') == scope['id'] and value.get('task') == scope['task']
                        and value.get('project') == scope['project']]
            for value in released:
                claims.pop(f"{value['kind']}:{value['key']}")
            self._save(claims)
            return {**claim, 'released_claims': released}

    @contextmanager
    def inactive_worktree(self, who: Identity, scope: dict[str, Any]):
        """Hold exact inactive task ownership through checkout recovery and preserve failed claims."""
        if who.role != HUMAN and who.id != scope['owner']:
            raise Denied('inactive worktree recovery requires its registered parent or person')
        resources = [('worktree', normal('worktree', scope['project'])), ('branch', scope['branch'])]
        repository_area = normal('area', scope['source_project'])
        with held(self.lock):
            claims = self._load()
            self._sweep(claims)
            before = dict(claims)
            for claim in claims.values():
                assigned = claim.get('assignment') == scope['id'] or claim.get('task') == scope['task'] \
                    or claim.get('project') == scope['project']
                if claim['kind'] == 'area' and _nested(claim['key'], repository_area) \
                        and not claim.get('project'):
                    raise Denied('an unbound repository area reservation requires an explicit handoff')
                if claim['kind'] in ('file', 'area') and assigned:
                    raise Denied('inactive recovery requires release of exact task file and area reservations')
                if any(_covers(claim, kind, key) for kind, key in resources):
                    if claim['kind'] in ('file', 'install'):
                        raise Denied('the inactive checkout has a live physical mutation reservation')
                    if claim['owner'] != who.id or claim.get('assignment') not in (None, scope['id']):
                        raise Denied('the inactive checkout has another live ownership assignment')
            now = self.clock()
            for kind, key in resources:
                name = f'{kind}:{key}'
                if name not in claims:
                    claims[name] = {'kind': kind, 'key': key, 'owner': who.id, 'pid': 0,
                                    'since': now, 'expires': now + self.ttl_s,
                                    'assignment': scope['id'], 'task': scope['task'],
                                    'project': scope['project'], 'note': 'Inactive task recovery'}
            self._save(claims)
            completed = False
            try:
                yield
                completed = True
            finally:
                if completed:
                    for kind, key in resources:
                        claims.pop(f'{kind}:{key}', None)
                self._save(claims if completed else before)

    def release(self, who: Identity, kind: str, key: str) -> dict[str, Any]:
        """Give up a claim. Its owner, a lead or a human may."""
        key = normal(kind, key)
        with held(self.lock):
            claims = self._load()
            self._sweep(claims)
            found = claims.get(f"{kind}:{key}")
            if found is None:
                raise ValueError(f"{kind} {key} is not claimed")
            if found["owner"] != who.id and who.role == AGENT:
                raise Denied(f"{kind} {key} belongs to {found['owner']}")
            claims.pop(f"{kind}:{key}")
            self._save(claims)
            return found

    def renew(self, who: Identity, ttl_s: float = 0.0, *,
              predicate: Callable[[dict[str, Any]], bool] | None = None) -> list[dict[str, Any]]:
        """Extend every claim ``who`` holds by ``ttl_s`` (at most ``MAX_RENEW_S``) from now, never
        past ``MAX_LIFETIME_S`` after it was first taken; returns the renewed claims, each marked
        ``capped`` when the lifetime cap held it back."""
        step = min(ttl_s or self.ttl_s, MAX_RENEW_S)
        with held(self.lock):
            claims = self._load()
            self._sweep(claims)
            now = self.clock()
            mine = [c for c in claims.values() if c["owner"] == who.id
                    and (predicate is None or predicate(c))]
            for claim in mine:
                limit = claim["since"] + MAX_LIFETIME_S
                claim["expires"] = max(claim["expires"], min(now + step, limit))
                claim["capped"] = now + step > limit
            self._save(claims)
            return [dict(c) for c in mine]

    def heartbeat(self, who: Identity, ttl_s: float = 0.0) -> int:
        """Renew every claim ``who`` holds; returns how many."""
        return len(self.renew(who, ttl_s))

    def listing(self, owner: str = "", kind: str = "") -> list[dict[str, Any]]:
        """Live claims, optionally of one owner or kind, each with ``expires_in_s`` and
        ``expiring_soon``."""
        with held(self.lock):
            claims = self._load()
            if self._sweep(claims):
                self._save(claims)
        now, soon = self.clock(), min(EXPIRING_SOON_S, self.ttl_s / 3)
        return sorted(({**c, "expires_in_s": round(c["expires"] - now, 1),
                        "expiring_soon": c["expires"] - now <= soon}
                       for c in claims.values()
                       if (not owner or c["owner"] == owner) and (not kind or c["kind"] == kind)),
                      key=lambda c: (c["kind"], c["key"]))

    def who(self, kind: str, key: str) -> dict[str, Any] | None:
        """The claim that covers ``key``, or None when nobody owns it."""
        key = normal(kind, key)
        return next((c for c in self.listing() if _covers(c, kind, key)), None)
