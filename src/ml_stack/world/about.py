"""The ``world.json`` record beside an invented world: what it says about itself."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ml_stack.files import UNVERSIONED, read_json, version_of, versioned, write_json

__all__ = ["NAME", "VERSION", "UnknownVersion", "read", "write"]

NAME = "world.json"

#: 1 -- kind, size, seed, people, and the organisation the graph was built from.
VERSION = 1


class UnknownVersion(ValueError):
    """The record was written by a newer ml-stack than this one."""


def write(where: Path | str, record: Mapping[str, Any]) -> Path:
    """Write ``record`` as ``where/world.json`` carrying `VERSION`. Returns the path."""
    path = Path(where).expanduser() / NAME
    write_json(path, versioned(record, VERSION))
    return path


def read(where: Path | str) -> dict[str, Any]:
    """What ``where/world.json`` says -- kind, size, seed, people -- or ``{}`` when absent.

    A record with no version key was written before the key existed; its fields are the
    ones `VERSION` names, so it is read as it stands. A higher version raises.
    """
    path = Path(where).expanduser() / NAME
    record = read_json(path, {})
    if not isinstance(record, Mapping):
        return {}
    found = version_of(record)
    if found not in (UNVERSIONED, VERSION):
        raise UnknownVersion(
            f"{path} is a version {found} world record and this ml-stack reads {VERSION}")
    return dict(record)
