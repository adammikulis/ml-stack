"""Best-effort execution metadata from maintained coding harness events."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import uuid
from http.client import HTTPException
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path

from ml_stack import hook_diagnostics
from ml_stack.workspace import harness_remote, tokens
from ml_stack.workspace.execution_profile import OPAQUE, _document
from ml_stack.workspace.service import Workspace

EVENTS = frozenset({'SessionStart', 'PostModelSwitch', 'PreToolUse', 'PostToolUse', 'Stop', 'SubagentStop'})
EFFORT = frozenset({'low', 'medium', 'high', 'xhigh', 'max'})
MAX_INPUT = 65536


def metadata(payload):
    """Extract whitelisted reported metadata without inspecting session content."""
    if type(payload) is not dict or payload.get('hook_event_name') not in EVENTS:
        return None
    session = payload.get('session_id')
    agent = payload.get('agent_id')
    if type(session) is not str or not OPAQUE.fullmatch(session):
        raise ValueError('metadata requires a bounded opaque session ID')
    if agent is not None and (type(agent) is not str or not OPAQUE.fullmatch(agent)):
        raise ValueError('metadata requires a bounded opaque helper ID')
    fields = {'harness': 'claude-code'}
    event = payload['hook_event_name']
    model = payload.get('to_model') if event == 'PostModelSwitch' else payload.get('model') if event == 'SessionStart' else None
    if model is not None:
        fields.update(model=model, model_version=None, effective_effort=None)
    effort = payload.get('effort')
    if type(effort) is dict and effort.get('level') in EFFORT:
        fields['effective_effort'] = effort['level']
    if agent:
        fields['harness_agent_id'] = agent
    if payload.get('agent_type') is not None:
        fields['harness_agent_type'] = payload['agent_type']
    key = hashlib.sha256(f'{session}\0{agent or ""}'.encode()).hexdigest()
    return _document({'session': f'claude:{key}', 'event_id': uuid.uuid4().hex, 'event': event, 'fields': fields})


def runtime_facts():
    """Return observer runtime metadata from its matching installed distribution."""
    facts = {'python_version': sys.version.split()[0]}
    try:
        installed = distribution('ml-stack')
        module = Path(installed.locate_file('ml_stack/workspace/profilehook.py')).resolve()
        if module != Path(__file__).resolve() or not module.is_relative_to(Path(sys.prefix).resolve()):
            return facts
        facts['runtime_version'] = installed.version
        marker = Path(installed.locate_file('ml_stack/fleet/built-from'))
        if not marker.is_symlink() and marker.is_file() and marker.stat().st_size <= 128:
            commit = marker.read_text().strip()
            if re.fullmatch(r'[a-f0-9]{40}', commit):
                facts['runtime_commit'] = commit
    except (PackageNotFoundError, OSError, ValueError):
        return facts
    return facts


def launched(args, harness, served):
    """Return the maintained launcher's observed model and independent request settings."""
    fields = {'harness': harness, 'model': served[1], **runtime_facts()}
    for option, field in (('model', 'requested_model'), ('ctx', 'requested_context'),
                          ('effort', 'requested_effort'), ('max_output_tokens', 'requested_output_tokens'),
                          ('max_turns', 'requested_turns')):
        value = getattr(args, option, None)
        if value is not None and value != '':
            fields[field] = value
    return _document({'session': f'launcher:{uuid.uuid4().hex}', 'event_id': uuid.uuid4().hex,
                      'event': 'LauncherStart', 'fields': fields})


def send(document, label, root, base=None):
    """Send metadata through the launcher's saved authenticated workspace identity."""
    canonical = harness_remote.context(label, root, [root], require_claim=False)
    if canonical:
        remote, who = canonical
        return remote.call('record_execution_profile', remote.token(agent=who.id), document)
    ws = Workspace(base)
    token = tokens.load(ws.base, label)
    who = ws.auth(token)
    if who.id != label:
        raise ValueError('metadata requires the saved launcher identity')
    return ws.record_execution_profile(token, document)


def notify(payload, label, root):
    """Record reported hook metadata and emit a redacted warning on failure."""
    try:
        document = metadata(payload)
        if document is not None:
            document['fields'].update(runtime_facts())
            send(_document(document), label, root)
    except (OSError, ValueError, TypeError, RuntimeError, HTTPException) as error:
        sys.stderr.write(f'execution metadata unavailable: {hook_diagnostics.reason(error)}\n')


class _Parser(argparse.ArgumentParser):
    def error(self, _message):
        raise ValueError('execution metadata command arguments are invalid')


def run(argv=None, stdin=None):
    """Read one bounded metadata notification and always return success."""
    parser = _Parser()
    parser.add_argument('event', choices=('observe',))
    parser.add_argument('--label', required=True)
    parser.add_argument('--root', required=True)
    parser.add_argument('--role')
    parser.add_argument('--wait')
    try:
        args = parser.parse_args(argv)
        body = (stdin or sys.stdin).read(MAX_INPUT + 1)
        if len(body) > MAX_INPUT:
            raise ValueError('execution metadata input exceeds its size limit')
        notify(json.loads(body or '{}'), args.label, Path(args.root))
    except (OSError, ValueError, TypeError, RuntimeError, HTTPException) as error:
        sys.stderr.write(f'execution metadata unavailable: {hook_diagnostics.reason(error)}\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(run())
