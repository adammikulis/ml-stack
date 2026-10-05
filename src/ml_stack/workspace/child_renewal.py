"""Registered parents renew existing delegated seats without replacing their tokens."""

from ml_stack.workspace.chain import held
from ml_stack.workspace.identity import AGENT, CAPS, LEAD, Denied


def renew(ws, token: str, child: str) -> dict:
    """Extend a live child's expiry within its current parent's authority and TTL."""
    parent = ws.auth(token)
    if parent.parent or parent.role not in (AGENT, LEAD):
        raise Denied('only the registered top-level parent renews a delegated seat')
    registry = ws.registry
    with held(registry.path.with_name('agents.lock')):
        rows = registry._load()
        entry = rows.get(child)
        if not entry or entry.get('parent') != parent.id or not registry._live(rows, entry):
            raise Denied('the delegated seat is not a live child of this parent')
        if not set(entry.get('can', ())) <= set(parent.can) & set(CAPS):
            raise Denied('the child capabilities exceed current parent authority')
        expires = ws.clock() + ws.limits.child_ttl_s
        if stop := rows[parent.id].get('expires'):
            expires = min(expires, stop)
        entry['expires'] = expires
        registry._save(rows)
    ws.audit('delegate.renew', parent.id, child=child, expires=expires)
    return {'id': child, 'expires': expires}
