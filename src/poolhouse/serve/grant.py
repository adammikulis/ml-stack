"""The capability a model server needs to be started: a grant only the broker gives.

A backend launches nothing without a `Lease`, and a `Lease` cannot be made unless the code
making it is running inside `broker_grant()`. The broker opens that block around every start
and growth it performs for a lease it granted; nothing else opens it (a structural test and a
budget gate read the source for any other use). So a CLI, a helper or a test that calls the
low-level start by accident gets `NoGrant` instead of a server nobody queued or admitted.

This is a guard against accidents, not a sandbox: code that wants to defeat it can, and the
two checks above are what notice it doing so.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from contextvars import ContextVar

__all__ = ["NoGrant", "broker_grant", "granted", "require"]

_OPEN: ContextVar[bool] = ContextVar("poolhouse_broker_grant", default=False)


class NoGrant(RuntimeError):
    """A server start with no lease from the broker behind it."""


@contextlib.contextmanager
def broker_grant() -> Iterator[None]:
    """Mark the code inside as acting on a lease the broker granted. The broker's alone."""
    token = _OPEN.set(True)
    try:
        yield
    finally:
        _OPEN.reset(token)


def granted() -> bool:
    """Whether the caller is inside a broker grant."""
    return _OPEN.get()


def require(lease: object = None, kind: type | None = None) -> None:
    """Raise `NoGrant` unless inside a broker grant; with ``kind``, also unless ``lease``
    is the proof the grant made."""
    if kind is not None and not isinstance(lease, kind):
        raise NoGrant("launch needs the Lease the broker's grant made")
    if not granted():
        raise NoGrant(
            "a model server starts only on a lease the broker granted: `poolhouse-serve up "
            "MODEL` (or ServerManager.lease) asks for one; nothing else may start llama-server")
