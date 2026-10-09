"""The `enforcement` command of `poolhouse-workspace`."""

from __future__ import annotations

import argparse
from typing import Any

from poolhouse.command import flag
from poolhouse.workspace import enforcement, enforcement_check, project
from poolhouse.workspace.service import Workspace

ACTIONS = ('show', 'check', 'set', 'promote', 'demote')
OPTIONS = [
    flag('action', nargs='?', choices=ACTIONS, default='show',
         help='show the mode, check what strict would strand, set open|strict, promote or demote'),
    flag('value', nargs='?', default='', help='open or strict, for set'),
    flag('--project', default='', help='the project key (default: this checkout\'s project)'),
]


def run(args: argparse.Namespace, ws: Workspace, token: str) -> Any:
    """Show, check or change the enforcement mode of one project."""
    ws.auth(token)
    key = args.project or project.describe().get('key', '')
    new = {'promote': 'strict', 'demote': 'open'}.get(args.action) or args.value
    if args.action in ('promote', 'demote', 'set'):
        report = enforcement_check.check(ws, key) if new == 'strict' else None
        return {**enforcement.set_mode(ws, token, key, new), **({'preflight': report} if report else {})}
    if args.action == 'check':
        return enforcement_check.check(ws, key)
    return enforcement.current(ws, key)
