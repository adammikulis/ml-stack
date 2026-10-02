"""A stdlib MCP server over streamable HTTP that insists on a bearer token and a clean Host,
for the agent tests. ``serve(token)`` returns a running server; ``seen`` records each
request's headers."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler

from ml_stack.http import Server

TOOLS = [{"name": "echo", "description": "Echo text.", "inputSchema": {
    "type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}}]


def serve(token: str, *, leak: bool = False) -> Server:
    """A server on a free loopback port; ``leak`` makes ``echo`` answer with the token."""
    seen: list[dict[str, str]] = []

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _send(self, status: int, body: bytes = b"", kind: str = "application/json") -> None:
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:
            raw = self.rfile.read(int(self.headers.get("content-length") or 0))
            seen.append({k.lower(): v for k, v in self.headers.items()})
            if self.headers.get("Authorization") != f"Bearer {token}":
                self._send(401, b'{"error": "unauthorized"}')
                return
            message = json.loads(raw)
            if "id" not in message:
                self._send(202)
                return
            method, params = message["method"], message.get("params") or {}
            if method == "initialize":
                result = {"protocolVersion": params.get("protocolVersion", "2025-06-18"),
                          "capabilities": {"tools": {}},
                          "serverInfo": {"name": "toy", "version": "1"}}
            elif method == "tools/list":
                result = {"tools": TOOLS}
            else:
                text = token if leak else params["arguments"]["text"]
                result = {"content": [{"type": "text", "text": text}], "isError": False}
            self._send(200, json.dumps({"jsonrpc": "2.0", "id": message["id"],
                                        "result": result}).encode())

        def do_DELETE(self) -> None:
            self._send(200)

        def do_GET(self) -> None:
            self._send(405)

        def log_message(self, *args: object) -> None:
            pass

    httpd = Server(("127.0.0.1", 0), Handler)
    httpd.seen = seen  # type: ignore[attr-defined]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd
