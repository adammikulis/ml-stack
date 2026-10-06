"""Cluster-scoped transport admission for automatic project agents."""

import hashlib
import hmac
import ssl

from .discovery import derive_token, memberships


def _matched(opening, rows):
    if opening is None or not opening[2]:
        return None
    found = [member for member in rows
             if hmac.compare_digest(opening[1].secret, derive_token(member.key))]
    return found[0] if len(found) == 1 else None


def authenticated_cluster(opening, path) -> tuple[str, str]:
    """Return the current membership proven by an authenticated sealed request."""
    member = _matched(opening, memberships(path))
    return (member.group, hashlib.sha256(member.key).hexdigest()) if member else ("", "")


def _dev(connection, opening, rows):
    member = _matched(opening, rows)
    return member if (isinstance(connection, ssl.SSLSocket) and rows
                      and member is rows[0] and getattr(member, "mode", "prod") == "dev") else None


def visible(connection, opening, path) -> bool:
    """Whether a sealed TLS request proves the active Dev membership."""
    return _dev(connection, opening, memberships(path)) is not None


def admit(connection, opening, path, body) -> bool:
    """Whether this request proves the selected Dev cluster over encrypted transport."""
    member = _dev(connection, opening, memberships(path))
    return bool(member and body.get("cluster") == member.group
                and body.get("cluster_id") == hashlib.sha256(member.key).hexdigest())
