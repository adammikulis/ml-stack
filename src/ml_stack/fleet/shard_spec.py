"""A test-shard request: its framing on the wire and every check made before anything runs."""

from __future__ import annotations

import json
import re
import struct

from .shard_tree import MOST_PACKED, TreeError, clean_name

MOST_HEADER = 64 << 10
MOST_FILES = 200
MOST_SECONDS = 3600
FIELDS = frozenset({"id", "tree_sha256", "files", "timeout_s"})
TEST_FILE = re.compile(r"^tests/[A-Za-z0-9_][A-Za-z0-9_.-]*\.py$")
SHARD_ID = re.compile(r"^[0-9a-f]{32}$")
DIGEST = re.compile(r"^[0-9a-f]{64}$")


class Refused(ValueError):
    """A shard request that is refused; the message says which rule it broke."""


def frame(header: dict, tree: bytes) -> bytes:
    """The request body: a length, the JSON header, then the packed tree."""
    raw = json.dumps(header, sort_keys=True).encode()
    return struct.pack(">I", len(raw)) + raw + tree


def split(body: bytes) -> tuple[dict, bytes]:
    """The header and the packed tree a request body holds."""
    if len(body) < 4:
        raise Refused("the shard request is too short")
    (size,) = struct.unpack(">I", body[:4])
    if size > MOST_HEADER or 4 + size > len(body):
        raise Refused("the shard header is malformed or too large")
    try:
        header = json.loads(body[4:4 + size])
    except ValueError:
        raise Refused("the shard header is not JSON") from None
    if not isinstance(header, dict):
        raise Refused("the shard header is not an object")
    tree = body[4 + size:]
    if not tree or len(tree) > MOST_PACKED:
        raise Refused("the shard carries no tree or too large a one")
    return header, tree


def check_header(header: dict) -> dict:
    """The header once every field is known, typed and in range; Refused otherwise."""
    extra = sorted(set(header) - FIELDS)
    if extra:
        raise Refused(f"a shard takes no field called {extra[0][:40]!r}: only {', '.join(sorted(FIELDS))}")
    if not isinstance(header.get("id"), str) or not SHARD_ID.match(header["id"]):
        raise Refused("the shard id is 32 lowercase hex digits")
    if not isinstance(header.get("tree_sha256"), str) or not DIGEST.match(header["tree_sha256"]):
        raise Refused("the tree digest is 64 lowercase hex digits")
    seconds = header.get("timeout_s", MOST_SECONDS)
    if type(seconds) is not int or not 1 <= seconds <= MOST_SECONDS:
        raise Refused(f"the time limit is a whole number of seconds from 1 to {MOST_SECONDS}")
    return {**header, "timeout_s": seconds, "files": check_files(header.get("files"))}


def check_files(files: object) -> list[str]:
    """The test files, each one a plain ``tests/NAME.py`` path with no repeats."""
    if not isinstance(files, list) or not 1 <= len(files) <= MOST_FILES:
        raise Refused(f"a shard names from 1 to {MOST_FILES} test files")
    for name in files:
        if not isinstance(name, str) or not TEST_FILE.match(name):
            raise Refused(f"only tests/NAME.py files run in a shard, not {str(name)[:80]!r}")
    if len(set(files)) != len(files):
        raise Refused("a shard names each test file once")
    return list(files)


def check_present(files: list[str], shipped: list[str]) -> None:
    """Every test file is in the tree that came with the request."""
    have = set(shipped)
    for name in files:
        try:
            clean_name(name)
        except TreeError as exc:
            raise Refused(str(exc)) from None
        if name not in have:
            raise Refused(f"{name} is not in the shipped tree")
