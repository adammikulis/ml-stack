"""The person's workspace Board behind the Fleet UI authorization guard."""

from ml_stack.workspace import boardroute
from ml_stack.workspace.service import Workspace


def route(request, *, workspace=None, prefix="/ui/board/") -> bool:
    if not request.path.startswith(prefix):
        return False
    headers = {key.lower(): value for key, value in request.handler.headers.items()}
    raw = b""
    if request.method == "POST":
        size = headers.get("content-length", "0")
        if not size.isascii() or not size.isdigit() or len(size) > 8:
            request.send(400, {"error": "a post needs a valid Content-Length"})
            return True
        if int(size) > boardroute.POST_MAX:
            request.send(413, {"error": "the message is too large"})
            return True
        raw = request.handler.rfile.read(int(size))
    call = boardroute.Request(
        request.method, request.handler.path.replace(prefix, "/board/", 1),
        headers, request.handler.server.server_address[1], True, raw)
    status, extra, body = boardroute.respond(Workspace() if workspace is None else workspace, call)
    request.handler.send_response(status)
    for key, value in extra.items():
        request.handler.send_header(key, value)
    request.handler.send_header("Content-Length", str(len(body)))
    request.handler.end_headers()
    if request.method != "HEAD":
        request.handler.wfile.write(body)
    return True
