"""Serialized OS keychain access and cancellation state."""

import threading

SERVICE = "ml-stack"
_LOCK = threading.RLock()
_BLOCKED = False
HELP = ("OS keychain access is unavailable or was cancelled. Unlock or create the default "
        "keychain in your operating system's credential manager, then explicitly retry "
        "the credential or signing operation.")


class KeychainError(RuntimeError):
    """An OS keychain operation could not complete."""


def retry():
    """Allow one explicit operation to access the OS keychain again."""
    global _BLOCKED
    with _LOCK:
        _BLOCKED = False


def blocked():
    """Return whether keychain access is paused after an error or cancellation."""
    with _LOCK:
        return _BLOCKED


def perform(backend, operation, account, value=None):
    """Perform one keychain call without repeating a cancelled access."""
    global _BLOCKED
    with _LOCK:
        if _BLOCKED:
            raise KeychainError(HELP)
        args = (SERVICE, account) if value is None else (SERVICE, account, value)
        try:
            return getattr(backend, f"{operation}_password")(*args)
        except backend.errors.PasswordDeleteError:
            if operation == "delete":
                return False
            _BLOCKED = True
            raise KeychainError(HELP) from None
        except (backend.errors.KeyringError, OSError):
            _BLOCKED = True
            raise KeychainError(HELP) from None
