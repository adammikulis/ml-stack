"""Bounded argv-only Git operations for reviewed development integration."""

import hashlib
import os
import re
import subprocess
import sys
from pathlib import Path

from ml_stack import worktreerules
from ml_stack.sentinel.redaction import redact
from ml_stack.workspace.identity import Denied

COMMIT = re.compile(r'[0-9a-f]{40}')


class GateFailed(Denied):
    def __init__(self, phase: str, checks: list[dict]):
        super().__init__(f'integration {phase} gate failed; candidate preserved for rework')
        self.checks = checks


def git(root: Path, *arguments: str, binary: bool = False):
    result = subprocess.run(['git', '-C', str(root), *arguments], capture_output=True,
                            text=not binary, check=False, timeout=120,
                            env={**os.environ, **({'CLAUDECODE': '1'} if arguments[0] == 'push' else {})})
    if result.returncode:
        detail = result.stderr.decode(errors='replace') if binary else result.stderr
        raise RuntimeError(f'Git {arguments[0]} failed (exit {result.returncode}): {redact(detail)[-2000:]}')
    return result.stdout if binary else result.stdout.strip()


def clean(root: Path) -> None:
    if git(root, 'status', '--porcelain'):
        raise Denied(f'{root} has uncommitted changes; commit or preserve them before integration')


def ancestor(root: Path, before: str, after: str) -> bool:
    result = subprocess.run(['git', '-C', str(root), 'merge-base', '--is-ancestor', before, after],
                            capture_output=True, check=False, timeout=30)
    if result.returncode not in (0, 1):
        raise RuntimeError('Git could not establish the reviewed commit ancestry')
    return result.returncode == 0


def reviewed_files(root: Path, commit: str, artifacts: dict[str, str]) -> None:
    for name, digest in artifacts.items():
        path = Path(name)
        if path.is_absolute() or '..' in path.parts or '\\' in name or '\x00' in name:
            raise Denied('integration artifacts must be tracked relative repository paths')
        entry = git(root, 'ls-tree', commit, '--', name)
        if not entry.startswith(('100644 blob ', '100755 blob ')):
            raise Denied(f'the reviewed artifact is not a tracked regular file: {name}')
        content = git(root, 'show', f'{commit}:{name}', binary=True)
        if hashlib.sha256(content).hexdigest() != digest:
            raise Denied(f'the reviewed artifact hash changed: {name}')


def native_patch(root: Path, commit: str, baseline: str, artifacts: dict[str, str]) -> None:
    if not {'.task.patch', '.task-report.md'} <= set(artifacts):
        raise Denied('native integration requires the reviewed full patch and committed task report')
    parents = git(root, 'rev-list', '--parents', '-n', '1', commit).split()
    if len(parents) != 2 or not ancestor(root, baseline, parents[1]):
        raise Denied('the native artifact commit must follow its assigned source baseline')
    metadata = set(git(root, 'diff', '--name-only', parents[1], commit).splitlines())
    if metadata != {'.task.patch', '.task-report.md'}:
        raise Denied('the final artifact commit changed source after recording the reviewed patch')
    patch = git(root, 'show', f'{commit}:.task.patch', binary=True)
    expected = git(root, 'diff', '--binary', '--full-index', baseline, parents[1], binary=True)
    if patch != expected:
        raise Denied('the committed native patch does not describe the exact reviewed source changes')


def repository(source: Path) -> tuple[Path, str, str]:
    checkout = worktreerules.checkouts(source)
    if not checkout or checkout[0] != source.resolve() or checkout[0] == checkout[1]:
        raise Denied('reviewed work must belong to a prepared isolated Git worktree')
    primary = checkout[1]
    branch = git(primary, 'branch', '--show-current')
    if not branch or branch == 'main':
        raise Denied('integration only targets the checked-out development branch, never main')
    git(primary, 'check-ref-format', '--branch', branch)
    clean(primary)
    return primary, branch, git(primary, 'rev-parse', 'HEAD')


def gates(candidate: Path, baseline: str) -> list[dict]:
    outcomes = []
    before = git(candidate, 'rev-parse', 'HEAD')
    for arguments in (['quick', '--base', baseline], ['gate'],
                      ['all', 'tests/test_serve_no_bypass.py', '--redteam']):
        command = [sys.executable, 'scripts/test', *arguments]
        result = subprocess.run(command, cwd=candidate, capture_output=True,
                                check=False, timeout=900)
        outcomes.append({'command': command, 'passed': result.returncode == 0,
                         'exit': result.returncode,
                         'output_hash': hashlib.sha256(result.stdout + result.stderr).hexdigest()})
        if result.returncode:
            raise GateFailed(arguments[0], outcomes)
    clean(candidate)
    if git(candidate, 'rev-parse', 'HEAD') != before:
        raise Denied('the candidate commit changed during its gates')
    return outcomes
