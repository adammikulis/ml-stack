"""Credentials from explicit inputs, environment, files, and configured keychain entries."""

from __future__ import annotations

import contextlib
import os
import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ml_stack import home
from ml_stack.credentials import keychain as keystore
from ml_stack.credentials.environment import child_environment
from ml_stack.credentials.reading import (
    INSECURE_ENV,
    CredentialError,
    clean,
    read_text,
    valid_name,
)
from ml_stack.credentials.writing import write
from ml_stack.files import read_json, write_json
from ml_stack.platform import private_file

__all__ = ["FILE_ENV", "INSECURE_ENV", "CredentialError", "Secret", "child_environment",
           "describe", "file_path", "get", "set", "status", "unset"]


FILE_ENV = "ML_STACK_CREDENTIALS_FILE"
KEYRING_SERVICE = keystore.SERVICE
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
    if name not in _keychain_names():
        return None
    return _read_keyring(name)


def _keychain_names() -> dict[str, bool]:
    names = read_json(home.state("keychain-credentials.json"), {})
    return names if isinstance(names, dict) else {}


def _mark_keychain(name: str, present: bool) -> None:
    names = _keychain_names()
    if present:
        names[name] = True
    else:
        names.pop(name, None)
    path = home.state("keychain-credentials.json")
    write_json(path, names)
    private_file(path)


def _read_keyring(name: str) -> str | None:
    backend = _keyring()
    if backend is None:
        return None
    try:
        value = keystore.perform(backend, "get", name)
    except keystore.KeychainError as exc:
        raise CredentialError(str(exc)) from None
    return clean(value, "the keychain entry") if value else None


SOURCES: tuple[tuple[str, Callable[[str], str | None]], ...] = (
    ("environment", _from_env),
    ("environment file", _from_named_file),
    ("credentials file", _from_file),
    ("huggingface token file", _from_hf_file),
    ("keychain", _from_keyring),
)


def _lookup(name: str, *, keychain: bool | None = None) -> tuple[str, str] | None:
    names = HF_NAMES if name in HF_NAMES else (name,)
    for source, reader in SOURCES:
        if source == "keychain" and keychain is False:
            continue
        for alias in names:
            found = _read_keyring(alias) if source == "keychain" and keychain is True else reader(alias)
            if found:
                return source, found
    return None


def get(name: str, explicit: str | None = None, *, required: bool = False,
        keychain: bool | None = None) -> Secret | None:
    """The credential ``name``: ``explicit`` if given, else the first source that holds it.

    None when nothing does, or `CredentialError` when ``required``. A source that exists but
    cannot be trusted (a file other users can read) raises rather than being skipped.
    ``keychain=True`` explicitly retries an OS lookup; automatic lookups use enrolled names.
    """
    valid_name(name)
    if explicit is not None and explicit.strip():
        return Secret(clean(explicit, "the explicit value"))
    if keychain is True:
        keystore.retry()
    found = _lookup(name, keychain=keychain)
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
        found = _lookup(name, keychain=False)
    except CredentialError as exc:
        return {"present": False, "source": None, "error": str(exc)}
    if found is None and any(alias in _keychain_names() for alias in (HF_NAMES if name in HF_NAMES else (name,))):
        row = {"present": True, "source": "keychain", "checked": False}
        if keystore.blocked():
            row["error"] = keystore.HELP
        return row
    return {"present": found is not None, "source": found[0] if found else None}


def describe() -> list[dict[str, Any]]:
    """One row per credential this machine knows of: its name, where it comes from and
    whether that source can be used. Never a value."""
    names = list(KNOWN)
    with contextlib.suppress(CredentialError):
        names += sorted({*_entries(file_path()), *_keychain_names()} - {*names})
    return [{"name": name, **status(name)} for name in names]


def set(name: str, value: str, *, keychain: bool = False) -> str:
    """Store ``value`` as ``name`` in the credentials file (or the OS keychain); returns where."""
    valid_name(name)
    value = clean(value, "the value")
    if keychain:
        backend = _keyring()
        if backend is None:
            raise CredentialError("the keychain needs: pip install keyring")
        keystore.retry()
        try:
            keystore.perform(backend, "set", name, value)
        except keystore.KeychainError as exc:
            raise CredentialError(str(exc)) from None
        _mark_keychain(name, True)
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
            keystore.retry()
            deleted = keystore.perform(backend, "delete", name)
        except keystore.KeychainError as exc:
            raise CredentialError(str(exc)) from None
        if deleted is False:
            return False
        _mark_keychain(name, False)
        return True
    path = file_path()
    entries = _entries(path)
    if name not in entries:
        return False
    del entries[name]
    write(path, entries)
    return True
