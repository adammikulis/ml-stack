"""Reading a secret out of a file only its owner can read."""

from __future__ import annotations

import os
import re
import stat
import sys
from pathlib import Path

MOST_BYTES = 64 * 1024
MOST_CHARS = 8192
INSECURE_ENV = "POOLHOUSE_ALLOW_INSECURE_CREDENTIALS"
NAME_SHAPE = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,63}")


class CredentialError(RuntimeError):
    """A credential cannot be read, written or found. Never carries a value."""


def valid_name(name: str) -> str:
    """``name`` if it is a legal credential name; ``CredentialError`` otherwise."""
    if not isinstance(name, str) or not NAME_SHAPE.fullmatch(name):
        raise CredentialError("a credential name is letters, digits and underscores, "
                              "starting with a letter, at most 64 long")
    return name


def clean(value: str, where: str) -> str:
    """``value`` without surrounding whitespace; ``CredentialError`` for anything that is
    empty, too long or not one printable line."""
    value = value.strip()
    if not value:
        raise CredentialError(f"{where} is empty")
    if len(value) > MOST_CHARS:
        raise CredentialError(f"{where} is longer than {MOST_CHARS} characters")
    if not value.isprintable():
        raise CredentialError(f"{where} holds a control character or a line break")
    return value


def insecure_allowed() -> bool:
    """Whether the loud environment flag permits reading a file others can read."""
    return os.environ.get(INSECURE_ENV) == "yes-read-it-anyway"


def read_text(path: Path, *, anchor: Path | None = None, private: bool = True) -> str:
    """The text of ``path``, refusing what a secret file must not be.

    A symlink is followed only when its target stays inside ``anchor`` (default: the
    directory holding the link); a file other users can read is refused on POSIX unless
    ``private`` is False or the loud flag is set; a file owned by another user, one that is
    not a regular file and one over `MOST_BYTES` are refused. ``private=False`` is for a
    file an operator named, such as a mounted secret, and checks only the kind and size.
    """
    path = Path(path)
    anchor = (anchor or path.parent).resolve()
    try:
        real = path.resolve(strict=True)
    except OSError as exc:
        raise CredentialError(f"{path} cannot be read: {exc.strerror or 'not found'}") from None
    if path.is_symlink() and not real.is_relative_to(anchor):
        raise CredentialError(f"{path} is a symlink to {real}, outside {anchor}")
    try:
        fd = os.open(real, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise CredentialError(f"{path} cannot be opened: {exc.strerror}") from None
    try:
        info = os.fstat(fd)
        _check(path, info, private=private)
        raw = os.read(fd, MOST_BYTES + 1)
    finally:
        os.close(fd)
    if len(raw) > MOST_BYTES:
        raise CredentialError(f"{path} is larger than {MOST_BYTES} bytes")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        raise CredentialError(f"{path} is not UTF-8 text") from None


def _check(path: Path, info: os.stat_result, *, private: bool) -> None:
    if not stat.S_ISREG(info.st_mode):
        raise CredentialError(f"{path} is not a regular file")
    if info.st_size > MOST_BYTES:
        raise CredentialError(f"{path} is larger than {MOST_BYTES} bytes")
    if sys.platform == "win32" or not private:
        return
    if info.st_uid != os.geteuid():
        raise CredentialError(f"{path} belongs to another user")
    if info.st_mode & 0o077 and not insecure_allowed():
        raise CredentialError(
            f"{path} can be read by other users (mode {stat.S_IMODE(info.st_mode):04o}); "
            f"run: chmod 600 {path}")
