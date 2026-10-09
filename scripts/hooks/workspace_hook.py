"""Nonblocking workspace calls and redacted warnings for Claude notification hooks."""

from __future__ import annotations

import json
import os
import shlex
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from poolhouse.guard.secrets import env_secrets, redact


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


def primary_checkout(source: Path) -> Path:
    """The primary checkout of the repository holding `source`, read from its .git file without running git."""
    link = source / '.git'
    try:
        text = link.read_text().strip() if link.is_file() else ''
    except (OSError, ValueError):
        return source
    if text.startswith('gitdir:'):
        gitdir = Path(text.split(':', 1)[1].strip())
        if gitdir.parent.name == 'worktrees':
            return gitdir.parent.parent.parent
    return source


def refresh_runtime(stage: str, checkout: Path | None = None, agent: str = '') -> None:
    """Start a detached runtime build when the selected runtime is behind the checkout's HEAD; takes at most 6 seconds."""
    if os.environ.get('POOLHOUSE_RUNTIME_ENSURE') == 'off':
        return
    source = Path(__file__).resolve().parents[2]
    environment = dict(os.environ, PYTHONPATH=str(source / 'src'))
    try:
        done = subprocess.run([sys.executable, '-m', 'poolhouse.runtime_cli', 'ensure', '--background',
                               '--checkout', str(checkout or primary_checkout(source)),
                               *(['--agent', agent] if agent else [])], capture_output=True,
                              text=True, timeout=6, check=False, env=environment)
    except (OSError, subprocess.TimeoutExpired) as error:
        warning(stage, f'runtime refresh: {error}')
        return
    if done.returncode:
        warning(stage, f'runtime refresh exited {done.returncode}: {done.stderr or done.stdout}')


def session_environment(value: dict, stage: str) -> dict | None:
    environment = dict(os.environ)
    for name in ('POOLHOUSE_SESSION_ID', 'POOLHOUSE_SESSION_HARNESS', 'POOLHOUSE_WORKSPACE_AGENT'):
        environment.pop(name, None)
    session = value.get('session_id')
    if session is None:
        return environment
    if not isinstance(session, str) or not session or len(session) > 256 or any(ord(char) < 33 or ord(char) > 126 for char in session):
        warning(stage, 'invalid native session_id; no session context recorded')
        return None
    environment.update(POOLHOUSE_SESSION_ID=session, POOLHOUSE_SESSION_HARNESS='claude-code')
    known = agent_name(session, str(value.get('cwd') or ''))
    if known:
        environment['POOLHOUSE_WORKSPACE_AGENT'] = known
    return environment


def agent_name(native_id: str, cwd: str = '') -> str:
    """The unique name the board gave the native session with this id in the project of ``cwd`` (default the
    working directory), or an empty string."""
    if not native_id:
        return ''
    try:
        from poolhouse.board import session
        return session.find('claude-code', native_id, cwd=Path(cwd or Path.cwd()))
    except (ImportError, OSError, ValueError, RuntimeError, LookupError, KeyError) as error:
        warning('lookup', error)
        return ''


def name_session(environment: dict, model: str, event: dict | None = None) -> str:
    """Give the session its unique name and put it in `environment`; empty (with a warning) when it has no native session."""
    session = environment.get('POOLHOUSE_SESSION_ID', '')
    if not session:
        warning('SessionStart', 'no native session id; the session cannot be given its own name')
        return ''
    try:
        from poolhouse.board import client, place, session as board_session
        node = client.Client()
        board = place.resolve(node, Path(str((event or {}).get('cwd') or Path.cwd())))
        name = board_session.register(node, board, board_session.Native(model, 'claude-code', session)).name
    except (ImportError, OSError, ValueError, RuntimeError, LookupError, client.NodeError) as error:
        warning('SessionStart', error)
        return ''
    environment['POOLHOUSE_WORKSPACE_AGENT'] = name
    return name


def persist_session(environment: dict) -> None:
    target = os.environ.get('CLAUDE_ENV_FILE')
    session = environment.get('POOLHOUSE_SESSION_ID')
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
                for name in ('POOLHOUSE_SESSION_ID', 'POOLHOUSE_SESSION_HARNESS', 'POOLHOUSE_WORKSPACE_AGENT'):
                    if name in environment:
                        output.write(f'export {name}={shlex.quote(environment[name])}\n')
        finally:
            os.close(descriptor)
    except (OSError, ValueError, AttributeError) as error:
        warning('SessionStart', error)


def person(value: dict, stage: str) -> int:
    """Hands a hook event to the person-record handlers; never blocks, so every failure is a warning."""
    try:
        from poolhouse.workspace import person_hook
        output = person_hook.on_event(value)
    except (ImportError, OSError, ValueError, KeyError, TypeError, AttributeError, RuntimeError) as error:
        warning(stage, error)
        return 0
    if output:
        print(json.dumps(output))
    return 0


def hook_notice(environment: dict, source: Path | None = None) -> str:
    """Check that git will run this checkout's hooks; post a changed problem to the board once; return the text for the agent."""
    from poolhouse import hookcheck
    repo = source or Path(__file__).resolve().parents[2]
    if not (repo / '.git').exists():
        return ''
    try:
        problems = hookcheck.inspect(repo)
        fingerprint = hookcheck.digest(problems)
        if fingerprint != hookcheck.remembered(repo):
            if problems:
                run(['poolhouse-workspace', 'announce', 'blocked', hookcheck.line(repo, problems)],
                    'SessionStart', environment=environment)
            hookcheck.remember(repo, fingerprint)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        warning('SessionStart', f'hook check: {error}')
        return ''
    return hookcheck.context(repo, problems) if problems else ''


def state_dir() -> Path:
    """Where hooks keep their small per-session memory (attention shown); private to the user."""
    named = os.environ.get('POOLHOUSE_HOOK_STATE')
    # Windows has no uid; its temporary directory is already the user's own.
    owner = os.getuid() if hasattr(os, 'getuid') else 'user'
    path = Path(named) if named else Path(tempfile.gettempdir()) / f'poolhouse-hooks-{owner}'
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    return path


def spawner_id(event: dict) -> str:
    """The agent id of the subagent that started this one, or an empty string when the main session did.

    The event itself may name it (`parent_agent_id`); otherwise the subagent's meta file in its session
    directory names the tool use that started it (`toolUseId`, with `spawnDepth`), and the transcript that
    holds that tool use is the spawner's.
    """
    if event.get('parent_agent_id'):
        return str(event['parent_agent_id'])
    session, agent = Path(str(event.get('transcript_path') or '')), str(event.get('agent_id') or '')
    if not session.name or not agent:
        return ''
    folder = session.with_suffix('') / 'subagents'
    try:
        meta = json.loads((folder / f'agent-{agent}.meta.json').read_text())
        if int(meta.get('spawnDepth') or 1) <= 1 or not meta.get('toolUseId'):
            return ''
        for other in sorted(folder.glob('agent-*.jsonl')):
            if other.stem != f'agent-{agent}' and meta['toolUseId'] in other.read_text(errors='replace'):
                return other.stem.removeprefix('agent-')
    except (OSError, ValueError, TypeError):
        return ''
    return ''


def parent_environment(event: dict, stage: str) -> dict | None:
    """The environment of the identity that started this subagent: the subagent that spawned it, else the main session."""
    environment = session_environment(event, stage)
    if environment is None:
        return None
    above = agent_name(spawner_id(event), str(event.get('cwd') or ''))
    if above:
        environment['POOLHOUSE_WORKSPACE_AGENT'] = above
    return environment


def own_environment(event: dict, stage: str) -> dict | None:
    """The environment of a subagent acting as itself, or None when the board has not named it."""
    environment = session_environment(event, stage)
    name = agent_name(str(event.get('agent_id') or ''), str(event.get('cwd') or ''))
    if environment is None or not name:
        return None
    environment['POOLHOUSE_WORKSPACE_AGENT'] = name
    return environment
