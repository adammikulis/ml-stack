"""One-use local browser handoff: tickets minted for the daemon's own window and the owner's
terminal, or from an already credentialed UI session."""

from __future__ import annotations

from . import invite_routes
from .launch_secret import HEADER
from .session import parse_cookie


class LaunchRoutes:
    def public_route(self) -> bool:
        if self.path != '/ui/launch/ticket':
            return super().public_route()
        if self.method != 'POST':
            self.send(405, {'error': 'POST required'})
            return True
        ui, source = self.ui, self.client_ip
        if (ui.launch is None or not invite_routes.local(source) or not ui.host_ok(self.host_header)
                or self.header('Origin') or self.header('Sec-Fetch-Site')):
            self.send(403, {'error': 'a launch ticket is asked for by this machine itself'})
            return True
        held = ui.launch_throttle.blocked_for(source)
        if held:
            self.send(429, {'error': f'too many attempts -- wait {held:.0f}s'})
            return True
        if not ui.launch.matches(self.header(HEADER)):
            ui.launch_throttle.failed(source)
            ui.record('launch.refused', reason='secret', source=source)
            self.send(403, {'error': 'the launch secret does not match this daemon'})
            return True
        return self._issue('launch-secret')

    def route(self) -> bool:
        if self.path != '/ui/launch-ticket':
            return super().route()
        if self.method != 'POST':
            self.send(405, {'error': 'POST required'})
            return True
        if self.ui.may_setup(self.client_ip, self.host_header, ''):
            self.send(403, {'error': 'browser launch requires the local machine'})
            return True
        session = self.ui.sessions.get(parse_cookie(self.cookie))
        if session is None:
            self.send(401, {'error': 'sign in first'})
            return True
        if not session.credentialed:
            self.ui.record('launch.refused', reason='uncredentialed-session', source=self.client_ip)
            self.send(403, {'error': 'open the app from its own window to hand a session to a browser'})
            return True
        return self._issue('session')

    def _issue(self, by: str) -> bool:
        try:
            ticket, expires = self.ui.sessions.mint_ticket(by)
        except ValueError as error:
            self.send(429, {'error': str(error)})
            return True
        self.ui.launch_throttle.succeeded(self.client_ip)
        self.ui.record('launch.ticket', by=by, source=self.client_ip)
        self.send(200, {'ticket': ticket, 'expires_at': expires}, {'Cache-Control': 'no-store'})
        return True
