"""One-use local browser handoff from an already authenticated UI session."""

from __future__ import annotations

from .session import parse_cookie


class LaunchRoutes:
    def route(self) -> bool:
        if self.path != '/ui/launch-ticket':
            return super().route()
        if self.method != 'POST':
            self.send(405, {'error': 'POST required'})
            return True
        if self.ui.may_setup(self.client_ip, self.host_header, ''):
            self.send(403, {'error': 'browser launch requires the local machine'})
            return True
        if self.ui.sessions.get(parse_cookie(self.cookie)) is None:
            self.send(401, {'error': 'sign in first'})
            return True
        ticket, expires = self.ui.sessions.mint_ticket()
        self.send(200, {'ticket': ticket, 'expires_at': expires}, {'Cache-Control': 'no-store'})
        return True
