"""Launcher-bound resource ownership for inspectable native harness mutations."""

import os
import shlex
import shutil
import sys
from pathlib import Path

from ml_stack import worktreerules
from ml_stack.guard.destructive import classify
from ml_stack.guard.shellscan import segments
from ml_stack.harnesspolicy import CATALOG, SHELL_TOOLS, _shell_line
from ml_stack.interventions import Call
from ml_stack.net import git
from ml_stack.serve.process import started_at
from ml_stack.workspace import tokens
from ml_stack.workspace.claims import normal
from ml_stack.workspace.identity import Denied
from ml_stack.workspace.project import describe
from ml_stack.workspace.service import Workspace

FILE_TOOLS = frozenset({'Edit', 'Write', 'MultiEdit', 'NotebookEdit', 'apply_patch'})
FILE_COMMANDS = frozenset({'touch', 'mkdir', 'rm', 'rmdir', 'cp', 'mv', 'chmod', 'chown', 'tee'})


def _path(value, cwd):
    return str((Path(cwd) / value).resolve())


def resources(name, args, cwd):
    """Extract fixed mutation targets without evaluating a shell command."""
    found = []
    if name in FILE_TOOLS:
        found.extend(('file', _path(value, cwd)) for key, value in args.items()
                     if key in ('file_path', 'notebook_path', 'path') and isinstance(value, str) and value)
        for value in args.values():
            if isinstance(value, str) and '*** Begin Patch' in value:
                found.extend(('file', _path(path, cwd)) for path in worktreerules.patch_paths(value))
    elif name in SHELL_TOOLS:
        commands, findings = segments(_shell_line(name, args))
        if findings:
            found.append(('worktree', _path('.', cwd)))
        here = cwd
        for command in commands:
            words = command.argv
            if not words:
                continue
            verb = Path(words[0]).name
            if verb == 'cd' and len(words) == 2:
                here = _path(words[1], here)
                continue
            found.extend(('file', _path(target, here)) for operator, target in command.redirects
                         if operator.startswith('>') or operator.startswith('2>'))
            if verb in FILE_COMMANDS:
                found.extend(('file', _path(word, here)) for word in words[1:] if not word.startswith('-'))
            known = verb in FILE_COMMANDS or verb in ('git', 'ml-stack-serve', 'ml-stack-workspace') or ('install' in words and 'pip' in words[:3])
            if not known and classify(Call('Bash', {'command': shlex.join(words)}), roots=(here,)).label != 'safe':
                found.append(('worktree', _path('.', here)))
            if verb == 'git' and classify(Call('Bash', {'command': shlex.join(words)}), roots=(here,)).label != 'safe':
                where = words[words.index('-C') + 1] if '-C' in words else here
                found.append(('worktree', _path(where, here)))
            if 'install' in words and (verb in ('pip', 'pip3', 'uv') or 'pip' in words[:3]):
                explicit = next((word.partition('=')[2] for word in words
                                 if word.startswith(('--prefix=', '--target='))), '') or next(
                                     (words[at + 1] for at, word in enumerate(words[:-1])
                                      if word in ('--prefix', '--target')), '')
                executable = shutil.which(words[0]) or (words[0] if Path(words[0]).is_absolute() else '')
                if not explicit and (not executable or 'shims' in Path(executable).parts or verb == 'uv'):
                    raise Denied('install ownership requires an explicit interpreter or target environment')
                target = _path(explicit, here) if explicit else str(Path(executable).resolve().parent.parent)
                found.append(('install', target))
            if verb == 'ml-stack-serve' and any(word in ('up', 'down', 'restart') for word in words[1:]):
                for at, word in enumerate(words):
                    if word == '--port' and at + 1 < len(words):
                        found.append(('port', words[at + 1]))
                    elif word.startswith('--port='):
                        found.append(('port', word.partition('=')[2]))
    physical = list(found)
    for kind, key in physical:
        if kind == 'file' and any((parent / '.git').exists() for parent in Path(key).parents):
            found.append(('area', normal('area', key)))
    return found


def conflict(name, args, cwd, actor):
    """Return an existing foreign ownership conflict before approval is requested."""
    if not isinstance(args, dict):
        return ''
    required = resources(name, args, cwd)
    if not required:
        return ''
    ws = Workspace()
    if not ws.registry.role_of(actor):
        return ''
    for kind, key in required:
        owner = ws.claims.who(kind, key)
        if owner and owner['owner'] != actor:
            return f"{kind} {key} belongs to {owner['owner']}"
    return ''


def reserve(name, args, cwd, actor, roots):
    """Reserve an inspectable mutation using the launcher's authenticated workspace seat."""
    if not isinstance(args, dict) or (name not in SHELL_TOOLS and CATALOG.get(name) != 'reversible'):
        return
    required = resources(name, args, cwd)
    if not required:
        return
    approved = [Path(root).resolve() for root in roots]
    for kind, key in required:
        if kind in ('file', 'worktree') and not any(Path(key).is_relative_to(root) for root in approved):
            raise Denied('mutation target is outside the launcher-approved project')
    ws = Workspace()
    who = ws.auth(tokens.load(ws.base, actor))
    if who.id != actor:
        raise Denied('mutation ownership requires the launcher-bound identity')
    ws._may(who, 'claim')
    grant = ws.registry.info(who.id).get('project') or ws.registry.info(who.parent).get('project')
    if not grant or any(describe(str(root)).get('key') != grant.get('key') for root in approved):
        raise Denied('mutation ownership requires an existing person-set project grant')
    try:
        commit = git.head(Path(cwd))
    except git.GitFailed:
        commit = ''
    ws.claims.reserve(who, required, {'note': 'native harness mutation', 'commit': commit,
                                     'owner_pid': os.getppid(), 'owner_started': started_at(os.getppid()),
                                     'interpreter': str(Path(sys.executable).resolve()),
                                     'environment': str(Path(sys.prefix).resolve())})
