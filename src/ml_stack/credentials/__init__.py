"""Credentials: one resolver for every token and key this library needs.

`get(name)` looks in order at the explicit argument, the environment variable ``NAME``, the
file named by ``NAME_FILE``, the credentials file, Hugging Face's own token file for
``HF_TOKEN``, and the OS keychain when ``keyring`` is installed. Values are never logged or
put in an exception; `describe` and `status` say where a credential came from, not what it is.
"""

from __future__ import annotations

import contextlib
import logging
import os
import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ml_stack import home
from ml_stack.credentials.environment import child_environment
from ml_stack.credentials.reading import (
    INSECURE_ENV,
    CredentialError,
    clean,
    read_text,
    valid_name,
)
from ml_stack.credentials.writing import write

__all__ = ["FILE_ENV", "INSECURE_ENV", "CredentialError", "Secret", "child_environment",
           "describe", "file_path", "get", "set", "status", "unset"]

logger = logging.getLogger(__name__)

FILE_ENV = "ML_STACK_CREDENTIALS_FILE"
KEYRING_SERVICE = "ml-stack"
HF_NAMES = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")
KNOWN = ("HF_TOKEN", "ANTHROPIC_API_KEY")


class Secret(str):
    """A credential. It is a ``str`` where it is used and says nothing where it is printed."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "Secret(<redacted>)"

    def __reduce__(self) -> Any:
        raise TypeError("a Secret is not pickled")


def file_path() -> Path:
    """Where the credentials file is: ``ML_STACK_CREDENTIALS_FILE``, else under the state root."""
    named = os.environ.get(FILE_ENV)
    return home.expand(named) if named else home.state("credentials.toml")


def hf_token_path() -> Path:
    """Where Hugging Face keeps the token ``huggingface-cli login`` saved."""
    named = os.environ.get("HF_TOKEN_PATH")
    if named:
        return home.expand(named)
    root = os.environ.get("HF_HOME")
    base = home.expand(root) if root else home.user_home() / ".cache" / "huggingface"
    return base / "token"


def _from_env(name: str) -> str | None:
    value = os.environ.get(name)
    return clean(value, f"environment variable {name}") if value and value.strip() else None


def _from_named_file(name: str) -> str | None:
    named = os.environ.get(f"{name}_FILE")
    if not named:
        return None
    return clean(read_text(home.expand(named), private=False, anchor=Path("/")),
                 f"the file {name}_FILE names")


def _entries(path: Path) -> dict[str, str]:
    """Every credential in the file at ``path``; empty when the file is not there."""
    if not path.exists() and not path.is_symlink():
        return {}
    text = read_text(path)
    try:
        table = tomllib.loads(text)
    except tomllib.TOMLDecodeError:
        raise CredentialError(f"{path} is not valid TOML") from None
    odd = sorted(key for key, value in table.items() if not isinstance(value, str))
    if odd:
        raise CredentialError(f"{path} holds entries that are not strings: {', '.join(odd)}")
    return {key: clean(value, f"{key} in {path}") for key, value in table.items()}


def _from_file(name: str) -> str | None:
    return _entries(file_path()).get(name)


def _from_hf_file(name: str) -> str | None:
    if name not in HF_NAMES:
        return None
    path = hf_token_path()
    if not path.exists():
        return None
    text = read_text(path, private=False, anchor=path.parent)
    return clean(text, "Hugging Face's token file")


def _keyring() -> Any | None:
    try:
        import keyring
        import keyring.errors
    except ImportError:
        return None
    return keyring


def _from_keyring(name: str) -> str | None:
    backend = _keyring()
    if backend is None:
        return None
    try:
        value = backend.get_password(KEYRING_SERVICE, name)
    except (backend.errors.KeyringError, OSError) as exc:
        logger.debug("keyring unavailable: %s", type(exc).__name__)
        return None
    return clean(value, "the keychain entry") if value else None


SOURCES: tuple[tuple[str, Callable[[str], str | None]], ...] = (
    ("environment", _from_env),
    ("environment file", _from_named_file),
    ("credentials file", _from_file),
    ("huggingface token file", _from_hf_file),
    ("keychain", _from_keyring),
)


def _lookup(name: str) -> tuple[str, str] | None:
    names = HF_NAMES if name in HF_NAMES else (name,)
    for source, reader in SOURCES:
        for alias in names:
            found = reader(alias)
            if found:
                return source, found
    return None


def get(name: str, explicit: str | None = None, *, required: bool = False) -> Secret | None:
    """The credential ``name``: ``explicit`` if given, else the first source that holds it.

    None when nothing does, or `CredentialError` when ``required``. A source that exists but
    cannot be trusted (a file other users can read) raises rather than being skipped.
    """
    valid_name(name)
    if explicit is not None and explicit.strip():
        return Secret(clean(explicit, "the explicit value"))
    found = _lookup(name)
    if found is not None:
        return Secret(found[1])
    if required:
        raise CredentialError(
            f"no credential {name}: set {name} or {name}_FILE, or run: ml-stack credentials "
            f"set {name}")
    return None


def status(name: str) -> dict[str, Any]:
    """``{"present": bool, "source": str | None}`` for ``name``, and ``"error"`` when a source
    exists but cannot be used. Never the value."""
    valid_name(name)
    try:
        found = _lookup(name)
    except CredentialError as exc:
        return {"present": False, "source": None, "error": str(exc)}
    return {"present": found is not None, "source": found[0] if found else None}


def describe() -> list[dict[str, Any]]:
    """One row per credential this machine knows of: its name, where it comes from and
    whether that source can be used. Never a value."""
    names = list(KNOWN)
    with contextlib.suppress(CredentialError):
        names += sorted({*_entries(file_path())} - {*names})
    return [{"name": name, **status(name)} for name in names]


def set(name: str, value: str, *, keychain: bool = False) -> str:
    """Store ``value`` as ``name`` in the credentials file (or the OS keychain); returns where."""
    valid_name(name)
    value = clean(value, "the value")
    if keychain:
        backend = _keyring()
        if backend is None:
            raise CredentialError("the keychain needs: pip install keyring")
        backend.set_password(KEYRING_SERVICE, name, value)
        return "keychain"
    path = file_path()
    entries = _entries(path)
    entries[name] = value
    write(path, entries)
    return str(path)


def unset(name: str, *, keychain: bool = False) -> bool:
    """Remove ``name`` from the credentials file (or the keychain). Whether it was there."""
    valid_name(name)
    if keychain:
        backend = _keyring()
        if backend is None:
            return False
        try:
            backend.delete_password(KEYRING_SERVICE, name)
        except backend.errors.PasswordDeleteError:
            return False
        return True
    path = file_path()
    entries = _entries(path)
    if name not in entries:
        return False
    del entries[name]
    write(path, entries)
    return True
