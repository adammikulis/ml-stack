"""Maintained local-agent controls behind Fleet authentication."""

from poolhouse.workspace import localroute
from poolhouse.workspace.boardroute import Request
from poolhouse.workspace.service import Workspace


def route(request) -> bool:
    if not request.path.startswith('/ui/agents/'):
        return False
    if request.method not in ('GET', 'POST'):
        request.send(405, {'error': 'agent controls support GET and POST'})
        return True
    headers = {key.lower(): value for key, value in request.handler.headers.items()}
    raw = b''
    if request.method == 'POST':
        length = headers.get('content-length', '0')
        if not length.isascii() or not length.isdigit() or len(length) > 8:
            request.send(400, {'error': 'invalid request length'})
            return True
        if int(length) > localroute.BODY_MAX:
            request.send(413, {'error': 'agent configuration is too large'})
            return True
        raw = request.handler.rfile.read(int(length))
    call = Request(request.method, request.handler.path.replace('/ui/agents/', '/agents/', 1),
                   headers, request.handler.server.server_address[1], True)
    code, extra, body = localroute.respond(Workspace(), call, raw)
    handler = request.handler
    handler.send_response(code)
    for key, value in extra.items():
        handler.send_header(key, value)
    handler.send_header('Content-Length', str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)
    return True
