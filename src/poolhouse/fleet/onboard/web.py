"""The one HTTP handler the onboarding servers share.

Pairing, file sharing and the bootstrap offer each say what a request means as a function
from `Call` to `Reply`; this module is the only place that speaks HTTP for them. It reads a
bounded body, writes the reply (a stream for files) and turns a framing error into a status.
"""

from __future__ import annotations

import json
import ssl
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

from poolhouse.fleet.framing import Limited, LimitedServer, Malformed, read_body
from poolhouse.graph.serve import ReplyHandler

__all__ = ["Call", "Dispatch", "Listener", "Reply", "json_reply"]


@dataclass(slots=True)
class Call:
    """A request as the route functions see it."""

    method: str
    path: str
    headers: Any
    client: str
    secure: bool
    body: Callable[[int], bytes]
    """Reads the request body, refusing more than the given number of bytes."""

    def json(self, most: int = 16 * 1024) -> dict[str, Any]:
        data = json.loads(self.body(most) or b"{}")
        if not isinstance(data, dict):
            raise Malformed(400, "the body is a JSON object")
        return data


@dataclass(slots=True)
class Reply:
    status: int
    body: bytes = b""
    headers: dict[str, str] = field(default_factory=dict)
    content_type: str = "application/octet-stream"
    stream: Iterator[bytes] | None = None
    length: int = 0
    """The size of ``stream``, which is sent in place of ``body``."""


Dispatch = Callable[[Call], Reply]


def json_reply(status: int, document: dict[str, Any]) -> Reply:
    return Reply(status, json.dumps(document).encode(), content_type="application/json")


def _handler(dispatch: Dispatch) -> type[ReplyHandler]:
    class Handler(Limited, ReplyHandler):
        def answer(self) -> Reply:
            call = Call(self.command, self.path, self.headers, self.client_address[0],
                        isinstance(self.connection, ssl.SSLSocket),
                        lambda most: read_body(self, most))
            try:
                return dispatch(call)
            except Malformed as bad:
                self.close_connection = True
                return json_reply(bad.status, {"error": bad.message})
            except ValueError:
                return json_reply(400, {"error": "the body is not JSON"})

    return Handler


class Listener:
    """A threaded server running ``dispatch``; TLS when given a context."""

    def __init__(self, dispatch: Dispatch, address: tuple[str, int],
                 context: ssl.SSLContext | None = None) -> None:
        self.httpd = LimitedServer(address, _handler(dispatch), tls=context)
        self._thread: threading.Thread | None = None

    @property
    def port(self) -> int:
        return int(self.httpd.server_address[1])

    def start(self) -> None:
        self._thread = threading.Thread(target=self.httpd.serve_forever, daemon=True,
                                        name="onboard-listener")
        self._thread.start()

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
