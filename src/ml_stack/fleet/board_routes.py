"""The person's workspace Board behind the Fleet UI authorization guard."""

from ml_stack.workspace import boardroute
from ml_stack.workspace.service import Workspace


class BoardRoutes:
    def route(self) -> bool:
        if not self.path.startswith("/ui/board/"):
            return super().route()
        headers = {key.lower(): value for key, value in self.handler.headers.items()}
        raw = b""
        if self.method == "POST":
            size = headers.get("content-length", "0")
            if not size.isascii() or not size.isdigit() or len(size) > 8:
                self.send(400, {"error": "a post needs a valid Content-Length"})
                return True
            if int(size) > boardroute.POST_MAX:
                self.send(413, {"error": "the message is too large"})
                return True
            raw = self.handler.rfile.read(int(size))
        request = boardroute.Request(
            self.method, self.handler.path.replace("/ui/board/", "/board/", 1),
            headers, self.handler.server.server_address[1], True, raw)
        status, extra, body = boardroute.respond(Workspace(), request)
        self.handler.send_response(status)
        for key, value in extra.items():
            self.handler.send_header(key, value)
        self.handler.send_header("Content-Length", str(len(body)))
        self.handler.end_headers()
        if self.method != "HEAD":
            self.handler.wfile.write(body)
        return True
