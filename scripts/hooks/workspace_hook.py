"""Nonblocking workspace calls and redacted warnings for Claude notification hooks."""

from __future__ import annotations

import json
import os
import shlex
import stat
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from ml_stack.guard.secrets import env_secrets, redact


def warning(stage: str, reason: object) -> None:
    text, _ = redact(str(reason), env_secrets(os.environ))
    print(f'workspace {stage}: {" ".join(text.split())[:1000]}', file=sys.stderr)


def event(stage: str) -> dict | None:
    try:
        text = sys.stdin.read(65537)
        if len(text) > 65536:
            raise ValueError('hook event exceeds 64 KiB')
        value = json.loads(text)
        if not isinstance(value, dict):
            raise ValueError('hook event must be a JSON object')
        return value
    except (ValueError, OSError) as error:
        warning(stage, error)
        return None


def run(command: list[str], stage: str, *, environment: dict | None = None):
    try:
        done = subprocess.run(command, capture_output=True, text=True, check=False, timeout=30, env=environment)
    except (OSError, subprocess.TimeoutExpired) as error:
        warning(stage, error)
        return None
    if done.returncode:
        warning(stage, f'command exited {done.returncode}: {done.stderr or done.stdout or "no diagnostic returned"}')
        return None
    return done


def session_environment(value: dict, stage: str) -> dict | None:
    environment = dict(os.environ)
    for name in ('ML_STACK_SESSION_ID', 'ML_STACK_SESSION_HARNESS'):
        environment.pop(name, None)
    session = value.get('session_id')
    if session is None:
        return environment
    if not isinstance(session, str) or not session or len(session) > 256 or any(ord(char) < 33 or ord(char) > 126 for char in session):
        warning(stage, 'invalid native session_id; no session context recorded')
        return None
    environment.update(ML_STACK_SESSION_ID=session, ML_STACK_SESSION_HARNESS='claude-code')
    return environment


def persist_session(environment: dict) -> None:
    target = os.environ.get('CLAUDE_ENV_FILE')
    session = environment.get('ML_STACK_SESSION_ID')
    if not target or not session:
        return
    try:
        root = Path(os.environ.get('CLAUDE_CONFIG_DIR', str(Path.home() / '.claude')))
        path = Path(target)
        parent = root / 'session-env' / session
        if Path(session).name != session or session in ('.', '..') or path.parent != parent:
            raise ValueError('CLAUDE_ENV_FILE is not in this native session directory')
        for directory in (root, root / 'session-env', parent):
            info = directory.lstat()
            if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o022:
                raise ValueError('native session directory must be owned and not writable by other users')
        descriptor = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            file = os.open(path.name, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW,
                           0o600, dir_fd=descriptor)
            with os.fdopen(file, 'a') as output:
                info = os.fstat(output.fileno())
                if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
                    raise ValueError('native session exports must be a private owned regular file')
                for name in ('ML_STACK_SESSION_ID', 'ML_STACK_SESSION_HARNESS'):
                    output.write(f'export {name}={shlex.quote(environment[name])}\n')
        finally:
            os.close(descriptor)
    except (OSError, ValueError, AttributeError) as error:
        warning('SessionStart', error)
