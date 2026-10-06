"""Cluster-scoped transport admission for automatic project agents."""

import hashlib
import hmac
import ssl

from .discovery import derive_token, memberships


def authenticated_cluster(opening, path) -> tuple[str, str]:
    """Return the current membership proven by an authenticated sealed request."""
    if opening is None or not opening[2]:
        return "", ""
    secret = opening[1].secret
    found = [member for member in memberships(path)
             if hmac.compare_digest(secret, derive_token(member.key))]
    return (found[0].group, hashlib.sha256(found[0].key).hexdigest()) if len(found) == 1 else ("", "")


def visible(connection, opening, path) -> bool:
    """Whether a sealed TLS request proves an active Dev membership."""
    if not isinstance(connection, ssl.SSLSocket):
        return False
    cluster, _ = authenticated_cluster(opening, path)
    rows = memberships(path)
    return bool(rows and cluster and rows[0].group == cluster
                and getattr(rows[0], "mode", "prod") == "dev")


def admit(connection, opening, path, body) -> bool:
    """Whether this request proves the selected Dev cluster over encrypted transport."""
    if not visible(connection, opening, path):
        return False
    cluster, cluster_id = authenticated_cluster(opening, path)
    if not cluster or body.get("cluster") != cluster or body.get("cluster_id") != cluster_id:
        return False
    return True
