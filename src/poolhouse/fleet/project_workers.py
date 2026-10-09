"""Sealed project workspace and Dev worker routes."""

import re
from urllib.parse import urlparse

from . import project_enrollment

IDENTITY_HEADERS = ("X-Poolhouse-Agent", "X-Poolhouse-Sender", "X-Poolhouse-From", "X-Poolhouse-Parent",
                    "X-Poolhouse-Name", "X-Poolhouse-Label")
"""Headers that would name a sender: a project call is attributed to its token alone."""


def answer(handler, host, body, cluster_key_path):
    match = re.fullmatch(r"/workspace/v1/projects/([a-f0-9]{32})/(join|board|ensure|enroll|renew|worker)",
                         urlparse(handler.path).path)
    if not match:
        return False
    opening = handler._sealing()
    if any(handler.headers.get(name) for name in IDENTITY_HEADERS):
        handler._send(400, {"error": "a project call is attributed to its token; it carries no sender, "
                                     "parent, name or label header"})
    elif host is None:
        handler._send(501, {"error": "project workspace hosting is unavailable"})
    elif opening is None or not opening[2]:
        handler._send(403, {"error": "project agent capabilities require sealed fleet requests"})
    else:
        request = handler._object(body)
        cluster, cluster_id = project_enrollment.authenticated_cluster(opening, cluster_key_path)
        if match[2] in {"enroll", "renew", "worker"}:
            if not project_enrollment.admit(handler.connection, opening, cluster_key_path, request):
                handler._send(403, {"error": "automatic project operations require the active Dev cluster over TLS"})
                return True
            if match[2] == "worker":
                code, reply = host.start_worker(match[1], request, admission=(cluster, cluster_id),
                                                   cluster_key=cluster_key_path)
            else:
                operation = host.enroll if match[2] == "enroll" else host.renew
                code, reply = operation(match[1], request, cluster=cluster, cluster_id=cluster_id)
        else:
            code, reply = host.answer(match[1], match[2], request, device=handler._workspace_device,
                                      admission=(cluster, cluster_id, project_enrollment.visible(
                                          handler.connection, opening, cluster_key_path)))
        handler._send(code, reply)
    return True
