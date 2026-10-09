"""What keeps the keystore from hanging, prompting nobody, or quietly storing a key in the clear.

`poolhouse.keystore` asks these questions before and around every backend call. This module never
imports `keyring` (only `keystore` may); it is handed the backend or the work to run.

* `unsafe_backend` names why a backend must not hold the master key: a plaintext or file-backed
  `keyrings.alt` store, or a null store.
* `bounded` runs one backend call and gives up after a wait, because an OS prompt nobody can see
  (a locked Secret Service collection on a machine with no display, a Keychain dialog on a screen
  nobody is at) never returns.
* `forbid_prompts` tells macOS to fail an access that needs a dialog instead of showing one.
* `service_session` is true for a Windows process in session 0, where no person can answer.
"""

from __future__ import annotations

import sys
import threading
from collections.abc import Callable
from typing import Any

__all__ = ["BACKGROUND_WAIT_S", "PERSON_WAIT_S", "bounded", "forbid_prompts", "service_session", "unsafe_backend"]

PERSON_WAIT_S = 300.0
"""How long a person at a screen has to answer the one OS prompt."""
BACKGROUND_WAIT_S = 20.0
"""How long a process nobody watches waits for a backend that should answer at once."""

UNSAFE_MODULES = ("keyrings.alt", "keyring.backends.null", "keyring.backends.fail")
"""Backends that keep an item in a file in the clear, keep nothing, or always fail."""


def unsafe_backend(ring: Any) -> str:
    """Why ``ring`` (or a backend a chainer holds) must not hold the master key, else an empty string."""
    for part in getattr(ring, "backends", None) or [ring]:
        module = type(part).__module__
        if module.startswith(UNSAFE_MODULES):
            return f"{module}.{type(part).__qualname__} keeps keys in a plain file or nowhere"
    return ""


class _Carry:
    """A ``with`` block that keeps the exception raised inside it for the caller instead of raising it."""

    error: BaseException | None = None

    def __enter__(self) -> _Carry:
        return self

    def __exit__(self, kind: object, error: BaseException | None, trace: object) -> bool:
        self.error = error
        return True


def bounded(work: Callable[[], Any], wait_s: float) -> Any:
    """What ``work()`` returns, or its exception; `TimeoutError` when it has not finished in ``wait_s``.

    ``work`` runs in a daemon thread, so a call stuck on a dialog dies with the process.
    """
    done: list[Any] = []
    carry = _Carry()

    def run() -> None:
        with carry:
            done.append(work())
        if carry.error is not None:
            done.append(None)

    worker = threading.Thread(target=run, name="keystore-call", daemon=True)
    worker.start()
    worker.join(wait_s)
    if not done:
        raise TimeoutError(f"the OS keystore did not answer within {wait_s:.0f} seconds")
    if carry.error is not None:
        raise carry.error
    return done[0]


def _security_framework() -> Any:
    import ctypes
    import ctypes.util
    return ctypes.CDLL(ctypes.util.find_library("Security") or "/System/Library/Frameworks/Security.framework/Security")


def forbid_prompts() -> bool:
    """Make macOS fail a Keychain access that needs a dialog rather than show one; whether it did."""
    if sys.platform != "darwin":
        return False
    try:
        import ctypes
        framework = _security_framework()
        framework.SecKeychainSetUserInteractionAllowed.argtypes = [ctypes.c_ubyte]
        return framework.SecKeychainSetUserInteractionAllowed(0) == 0
    except (OSError, AttributeError):
        return False


def _session_id() -> int:
    import ctypes
    session = ctypes.c_ulong()
    kernel = ctypes.windll.kernel32  # type: ignore[attr-defined]
    if not kernel.ProcessIdToSessionId(kernel.GetCurrentProcessId(), ctypes.byref(session)):
        raise OSError("ProcessIdToSessionId failed")
    return session.value


def service_session() -> bool:
    """Whether this is a Windows process in session 0: a service or boot task, no desktop to answer."""
    if sys.platform != "win32":
        return False
    try:
        return _session_id() == 0
    except (OSError, AttributeError):
        return False
