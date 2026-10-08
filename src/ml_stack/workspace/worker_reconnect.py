"""Bounded reconnection for authenticated worker reads and acknowledgements."""

import errno
import time

from ml_stack.http import ServerError, ServerUnreachable
from ml_stack.workspace.identity import Denied

WINDOW_S = 90.0
POLL_S = 0.25
READS = frozenset({'whoami', 'agents', 'claims', 'who', 'inbox', 'outbox', 'thread', 'nudge',
                  'wait', 'history', 'board.list', 'board.read', 'board.threads', 'board.subs',
                  'board.digest', 'board.summary', 'board.mentions', 'work.reputation'})


class Unavailable(RuntimeError):
    """The authenticated worker transport did not recover within its reconnect window."""


class Cancelled(RuntimeError):
    """The worker was stopped while waiting for its authenticated transport."""


def transient(error):
    """Whether the failure chain contains a transport outage without an authorization refusal."""
    seen, failures = set(), []
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        failures.append(error)
        error = error.__cause__
    if any(isinstance(item, ServerError) and item.status in {400, 401, 403, 404, 409, 429, 501}
           for item in failures):
        return False
    return any(isinstance(item, ServerUnreachable)
               or (isinstance(item, ServerError) and item.status in {502, 503, 504})
               or isinstance(item, ConnectionError | TimeoutError)
               or (isinstance(item, OSError) and item.errno in {
                   errno.ECONNRESET, errno.ECONNREFUSED, errno.ETIMEDOUT,
                   errno.EHOSTUNREACH, errno.ENETUNREACH, errno.EPIPE}) for item in failures)


def unaccepted(error):
    """Whether the failure proves the daemon did not admit the request."""
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        if ((isinstance(error, ServerError) and error.status == 503)
                or (isinstance(error, OSError) and error.errno == errno.ECONNREFUSED)):
            return True
        error = error.__cause__
    return False


class Recovering:
    """Retry reads and cursor acknowledgements without replaying arbitrary mutations."""

    def __init__(self, transport, stopped, status, *, window_s=WINDOW_S):
        self.transport, self.stopped, self.status = transport, stopped, status
        self.window_s = window_s

    def __getattr__(self, name):
        return getattr(self.transport, name)

    def pause(self, deadline):
        if self.stopped():
            raise Cancelled('worker stopped during Board reconnection')
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise Unavailable('canonical Board did not recover within the bounded reconnect window')
        self.status.update(state='reconnecting', detail='Waiting for the authenticated project Board')
        time.sleep(min(POLL_S, remaining))

    def call(self, operation, token, *args, **kwargs):
        deadline = min(time.monotonic() + self.window_s,
                       kwargs.pop('_reconnect_deadline', float('inf')))
        if operation not in READS and operation != 'ack':
            return self.transport.call(operation, token, *args, **kwargs)
        while True:
            if self.stopped() and operation != 'ack':
                raise Cancelled('worker stopped during Board reconnection')
            if time.monotonic() >= deadline:
                raise Unavailable('canonical Board did not recover within the bounded reconnect window')
            try:
                return self.transport.call(operation, token, *args, **kwargs)
            except (Denied, OSError, RuntimeError) as error:
                if not transient(error):
                    raise
                self.pause(deadline)
