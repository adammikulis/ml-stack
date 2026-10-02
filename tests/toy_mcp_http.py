"""An MCP server built with the ``mcp`` SDK, served over streamable HTTP by uvicorn on a real
socket behind a bearer-token check. ``running(token)`` is a context manager yielding the
URL; ``seen`` lists the headers of each request."""

from __future__ import annotations

import socket
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Annotated

import uvicorn
from mcp.server.mcpserver import (
    AcceptedElicitation,
    Elicit,
    ElicitationResult,
    MCPServer,
    Resolve,
)
from mcp_types import ToolAnnotations
from pydantic import BaseModel


class Sure(BaseModel):
    sure: bool


def ask_sure(path: str) -> Elicit[Sure]:
    return Elicit(f"Really delete {path}?", Sure)


def build(token: str, *, leak: bool = False) -> MCPServer:
    app = MCPServer("toy")

    @app.tool(annotations=ToolAnnotations(read_only_hint=True))
    def echo(text: str) -> str:
        """Echo text, or the token when the server was built to leak it."""
        return token if leak else text

    @app.tool()
    def delete(path: str, sure: Annotated[ElicitationResult[Sure], Resolve(ask_sure)]) -> str:
        """Delete a path after asking the user."""
        return f"deleted {path}" if isinstance(sure, AcceptedElicitation) and sure.data.sure \
            else "kept"

    return app


class Guarded:
    """ASGI middleware refusing any request without ``Authorization: Bearer <token>``."""

    def __init__(self, app, token: str, seen: list[dict[str, str]]) -> None:
        self.app, self.token, self.seen = app, token, seen

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] == "http":
            headers = {k.decode().lower(): v.decode() for k, v in scope["headers"]}
            self.seen.append(headers)
            if headers.get("authorization") != f"Bearer {self.token}":
                body = b'{"error": "unauthorized"}'
                await send({"type": "http.response.start", "status": 401, "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode())]})
                await send({"type": "http.response.body", "body": body})
                return
        await self.app(scope, receive, send)


@contextmanager
def running(token: str, *, leak: bool = False) -> Iterator[tuple[str, list[dict[str, str]]]]:
    seen: list[dict[str, str]] = []
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    app = Guarded(build(token, leak=leak).streamable_http_app(), token, seen)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        time.sleep(0.02)
    try:
        yield f"http://127.0.0.1:{port}/mcp", seen
    finally:
        server.should_exit = True
        thread.join(timeout=5)
