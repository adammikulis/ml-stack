"""The `authority` command of `poolhouse-workspace`: show, set and preset the delegable gates."""

from __future__ import annotations

import argparse
from typing import Any

from poolhouse import authority
from poolhouse.command import flag
from poolhouse.workspace import enforcement, enforcement_check, project
from poolhouse.workspace.identity import Denied
from poolhouse.workspace.service import Workspace

ACTIONS = ('show', 'set', 'preset')
OPTIONS = [
    flag('action', choices=ACTIONS, help='show the gates, set some to person or delegated, or apply the dev or prod preset'),
    flag('words', nargs='*', default=[],
         help='set: person|delegated then ALL, a group or gates; preset: dev|prod'),
    flag('--project', default='', help='a project key: set its own state; preset: the project whose enforcement mode follows'),
]


def _flip(ws: Workspace, who: str, gates: list[str], new: str, where: tuple[str, str]) -> list[dict[str, str]]:
    scope, preset = where
    changes = authority.set_state(gates, new, by=who, project=scope, preset=preset)
    ws.audit('authority.set', who, project=scope, gates=gates, preset=preset,
             **{'from': sorted({c['from'] for c in changes}), 'to': new})
    return changes


def _preset(ws: Workspace, token: str, who: str, name: str, scope: str) -> dict[str, Any]:
    if name not in authority.PRESETS:
        raise ValueError(f'preset must be one of {", ".join(authority.PRESETS)}')
    keep = authority.PROD_DELEGATED if name == 'prod' else set(authority.GATES)
    changes = [*_flip(ws, who, sorted(keep), authority.DELEGATED, (scope, name)),
               *_flip(ws, who, sorted(set(authority.GATES) - keep), authority.PERSON, (scope, name))]
    key = scope or project.describe().get('key', '')
    mode = 'strict' if name == 'prod' else 'open'
    report = enforcement_check.check(ws, key) if mode == 'strict' else None
    return {'preset': name, 'changes': [c for c in changes if c['from'] != c['to']],
            'enforcement': enforcement.set_mode(ws, token, key, mode), **({'preflight': report} if report else {})}


def run(args: argparse.Namespace, ws: Workspace, token: str) -> Any:
    """Show the registry, or change it as a lead or a person; a helper identity is refused."""
    who = ws.auth(token)
    if args.action == 'show':
        return authority.show(args.project)
    if who.parent:
        ws.audit('auth.denied', who.id, reason='authority is flipped by a lead, not a helper')
        raise Denied('only a lead or a person changes authority, not a helper')
    if args.action == 'preset':
        if len(args.words) != 1:
            raise ValueError('preset takes dev or prod')
        return _preset(ws, token, who.id, args.words[0], args.project)
    if len(args.words) < 2 or args.words[0] not in (authority.PERSON, authority.DELEGATED):
        raise ValueError('set takes person or delegated, then ALL, a group or gates')
    try:
        gates = authority.resolve(args.words[1:])
    except KeyError as err:
        raise ValueError(str(err.args[0])) from err
    return {'changes': _flip(ws, who.id, gates, args.words[0], (args.project, ''))}
