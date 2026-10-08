"""Private worker execution checkpoints and mailbox completion recovery."""

import hashlib
import json
import time

from ml_stack.graph.store import GraphStore
from ml_stack.workspace import tokens, worker_reconnect
from ml_stack.workspace.chain import held
from ml_stack.workspace.identity import Denied

MAX_REPLY = 3500


class Interrupted(RuntimeError):
    """A prior worker execution ended before its result was durably recorded."""


def fingerprint(row):
    selected = {key: row.get(key) for key in ('id', 'seq', 'from', 'to', 'type')}
    selected['body'] = row.get('raw', row.get('text', ''))
    return hashlib.sha256(json.dumps(selected, sort_keys=True).encode()).hexdigest()


class Journal:
    """Keep one authenticated worker's pending execution in an owner-only graph."""

    def __init__(self, ws):
        remote = ws.remote
        scope = f'{remote.current.host}/{remote.current.project_id}/{ws.worker_identity}'
        self.identity = ws.worker_identity
        self.parents = (ws.base, ws.base / 'worker-completions',
                        ws.base / 'worker-completions' / hashlib.sha256(scope.encode()).hexdigest())
        for parent in self.parents:
            if not parent.exists() and not parent.is_symlink():
                parent.mkdir(parents=True, mode=0o700)
            if problem := tokens.problem(parent):
                raise Denied(f'worker completion storage: {problem}')
        self.base = self.parents[-1]
        self.path, self.lock = self.base / 'completion.db', self.base / 'completion.lock'

    def record(self, update=None):
        for path in (*self.parents, self.path, self.lock,
                     *(self.path.with_name(self.path.name + suffix) for suffix in ('.wal', '.shadow', '.tmp'))):
            if (path.exists() or path.is_symlink()) and (problem := tokens.problem(path)):
                raise Denied(f'worker completion storage: {problem}')
        try:
            with held(self.lock), GraphStore(self.path) as graph:
                self.path.chmod(0o600)
                saved = graph.get_doc('pending') or {}
                if update is not None:
                    graph.put_doc('pending', update)
                    saved = update
            for suffix in ('.wal', '.shadow', '.tmp'):
                companion = self.path.with_name(self.path.name + suffix)
                if companion.exists() and not companion.is_symlink():
                    companion.chmod(0o600)
            return saved
        except (OSError, RuntimeError) as error:
            raise Interrupted('worker execution checkpoint storage is unavailable; replay is disabled') from error

    def begin(self, row):
        if len(json.dumps(row).encode()) > 128 * 1024:
            raise ValueError('worker execution checkpoint exceeds its size limit')
        execution = self.base / 'execution.lock'
        if (execution.exists() or execution.is_symlink()) and (problem := tokens.problem(execution)):
            raise Denied(f'worker completion storage: {problem}')
        with held(execution):
            key = fingerprint(row)
            saved = self.record()
            if saved:
                if saved.get('fingerprint') != key:
                    raise Interrupted('a different task has an unresolved worker execution checkpoint')
                if saved['state'] == 'executing':
                    raise Interrupted('task execution was interrupted; side effects require review before retry')
                return saved
            self.record({'fingerprint': key, 'seq': row['seq'], 'state': 'executing', 'row': row})
            return None

    def finish(self, row, kind, text, rounds):
        body = text[:MAX_REPLY]
        if len(text) > MAX_REPLY:
            notice = ' [Reply shortened to the workspace message size limit.]'
            body = text[:MAX_REPLY - len(notice)] + notice
        saved = {'fingerprint': fingerprint(row), 'seq': row['seq'], 'state': 'completed',
                 'kind': kind, 'body': body, 'rounds': rounds,
                 'subject': 'completion:' + fingerprint(row)[:32], 'row': row}
        return self.record(saved)

    def deliver(self, ws, token, row):
        saved = self.record()
        if saved.get('fingerprint') != fingerprint(row) or saved['state'] == 'executing':
            raise Interrupted('worker reply requires its durably completed task')
        if saved['state'] == 'sent':
            return
        deadline = time.monotonic() + ws.remote.window_s
        while True:
            if saved['state'] == 'sending':
                previous = ws.outbox(token, limit=100, _reconnect_deadline=deadline)
                if any(item['from'] == self.identity and item.get('reply_to') == row['seq']
                       and item.get('type') == saved['kind'] and item.get('raw') == saved['body']
                       for item in previous):
                    self.record({**saved, 'state': 'sent'})
                    return
                if len(previous) >= 100:
                    raise Interrupted('completion receipt is outside the bounded outbox; retry requires review')
                raise Interrupted('completion send outcome is unresolved; retry requires review')
            if time.monotonic() >= deadline:
                raise worker_reconnect.Unavailable('completion delivery exceeded its bounded reconnect window')
            saved = self.record({**saved, 'state': 'sending'})
            recipient = row['to'] if str(row['to']).startswith('#') else row['from']
            try:
                ws.send(token, recipient, saved['kind'], saved['body'],
                        reply_to=row['seq'], subject=saved['subject'])
            except (Denied, OSError, RuntimeError) as error:
                if not worker_reconnect.transient(error):
                    raise
                if worker_reconnect.unaccepted(error):
                    saved = self.record({**saved, 'state': 'completed'})
                ws.remote.pause(deadline)
                continue
            self.record({**saved, 'state': 'sent'})
            return

    def acknowledged(self, row):
        saved = self.record()
        if saved.get('fingerprint') == fingerprint(row) and saved.get('state') == 'sent':
            self.record({})
