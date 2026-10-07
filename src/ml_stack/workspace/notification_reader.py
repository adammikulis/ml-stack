"""Read-only notification identities from saved project sessions."""

import hashlib
import sys
from pathlib import Path

from ml_stack import worktreerules
from ml_stack.workspace import project_connection, tokens
from ml_stack.workspace.identity import AGENT, Denied, valid_id


def binding(label: str, cwd: Path, session: str) -> dict:
    """Return the saved project binding for the hook reader."""
    current = cwd.resolve()
    saved = project_connection._saved()
    configured = None
    for root in (current, *current.parents):
        metadata_path = root / '.ml-stack-project.json'
        metadata = project_connection.read_json(metadata_path, {})
        if not isinstance(metadata, dict) or (metadata_path.exists()
                and metadata.get('kind') != 'project-checkout'):
            raise Denied('notification checkout metadata is invalid')
        configured = saved.get(str(root))
        if metadata.get('kind') == 'project-checkout':
            authority = metadata.get('authority', {})
            if (not configured or configured.get('project_id') != metadata.get('project_id')
                    or (authority.get('host') and authority['host'] != configured.get('host'))):
                raise Denied('notification checkout and saved authority disagree')
        if configured is not None:
            break
    if configured is None:
        checkout = worktreerules.checkouts(current)
        configured = saved.get(str(checkout[1])) if checkout else None
        if configured and configured.get('project_id') != project_connection.projects.identity(checkout[0]):
            raise Denied('notification worktree differs from its saved project Board')
    if configured is None:
        raise Denied('notification requires an existing project binding')
    if label == 'codex' and 'sessions' in configured:
        if not isinstance(session, str) or not session or len(session) > 256 or any(
                ord(char) < 33 or ord(char) > 126 for char in session):
            raise Denied('notification requires a valid hook session identifier')
        slot = hashlib.sha256(session.encode()).hexdigest()[:32]
        sessions = configured.get('sessions', {})
        chosen = sessions.get(slot) if isinstance(sessions, dict) else None
        if (not isinstance(chosen, dict) or chosen.get('session') != slot
                or chosen.get('local_agent') != 'codex'
                or not valid_id(str(chosen.get('agent', '')))
                or not (chosen.get('agent') == f'codex-{slot}'
                        or chosen.get('agent', '').startswith(f'codex-{slot}-'))
                or any(chosen.get(key) != configured.get(key) for key in ('host', 'project_id'))):
            raise Denied('notification session has no authenticated saved binding')
        return chosen
    if (configured.get('agent') != label
            and (label == 'codex' or configured.get('local_agent') != label)):
        raise Denied('notification reader differs from the saved launcher identity')
    return configured


def read(label: str, cwd: Path, session: str, *, canonical=None) -> str:
    """Return unread-message metadata under an existing authenticated capability."""
    chosen = binding(label, cwd, session)
    actor = chosen['agent']
    if canonical is not None:
        remote, who = canonical
        if (who.id != actor or who.role != AGENT or remote.project_id != chosen['project_id']
                or remote.host != chosen['host'].rstrip('/')):
            raise Denied('notification context does not match the saved project reader')
        return remote.call('nudge', tokens.load(remote.base, actor))
    remote = project_connection.RemoteWorkspace(
        chosen['host'], chosen['project_id'], cluster=chosen.get('cluster', ''),
        cluster_key=Path(chosen['cluster_key']) if chosen.get('cluster_key') else None)
    token = tokens.load(remote.base, actor)
    who = remote.call('whoami', token)
    if (who.get('id') != actor or who.get('role') != AGENT
            or who.get('project', {}).get('key') != chosen['project_id']):
        raise Denied('notification capability does not match the saved project reader')
    return remote.call('nudge', token)


if __name__ == '__main__':
    try:
        print(read(sys.argv[1], Path(sys.argv[2]), sys.argv[3]))
    except (Denied, OSError, RuntimeError, ValueError, KeyError) as error:
        from ml_stack.harnesshook import _diagnostic
        print(_diagnostic(error), file=sys.stderr)
        raise SystemExit(1) from None
