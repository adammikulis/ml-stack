"""Exact clear agent reports reuse their verified journal entry for sixty seconds."""

from contextlib import nullcontext

from ml_stack.workspace.chain import held
from ml_stack.workspace.identity import HUMAN
from ml_stack.workspace.screen import injection_markers

WINDOW_S = 60.0
KINDS = frozenset({'status', 'done', 'milestone', 'blocked'})
FIELDS = ('from', 'role', 'label', 'to', 'type', 'subject', 'reply_to', 'thread',
          'body', 'model', 'model_state', 'report_ttl_s')


def eligible(row: dict) -> bool:
    return row['role'] != HUMAN and row['type'] in KINDS and not row.get('file')


def duplicate(ws, row: dict) -> dict | None:
    now = ws.clock()
    # The maintained sender index is derived from the verified journal, not another store.
    for previous in ws.bus.recent_outbox(row['from'], now - WINDOW_S):
        if previous['ts'] <= now and not previous.get('held') and ws.bus.live(previous) \
                and all(previous.get(key) == row.get(key) for key in FIELDS):
            return previous
    return None


def emit(ws, who, row: dict, *, announce: bool, ttl_s: float):
    report = eligible(row)
    with held(ws.base / 'reports.lock') if report else nullcontext():
        authenticated = ws._message_sender(who)
        ws._message_rights(who, row, announce)
        if authenticated and who.role != HUMAN and row["type"] == "done":
            from ml_stack.workspace.worktree_lifecycle import require_clean
            require_clean(ws.base, who.id, row["label"])
        row['model'], row['model_state'] = ('', '') if who.role == HUMAN else ws.registry.model_of(who.id, row['label'])
        if report and authenticated and not injection_markers(row['subject'] + '\n' + row['body']):
            row['report_ttl_s'] = ttl_s
            if previous := duplicate(ws, row):
                return ws.deliver(previous, raw=True)
        if announce:
            ws._announcement_quota(who)
        return ws._post_new(who, row, ttl_s)
