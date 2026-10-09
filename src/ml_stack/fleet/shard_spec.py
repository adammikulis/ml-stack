"""A test-shard job: every check made on its header before anything runs.

The node checks the same rules when the request arrives (`app/poolside-node/src/shard/spec.rs`);
the executor checks them again, because it reads a file and trusts nothing it did not verify.
"""

from __future__ import annotations

import re

from .shard_tree import TreeError, clean_name

MOST_FILES = 200
MOST_SECONDS = 3600
TIERS = ("all", "fast", "full", "gate", "slow")
FIELDS = frozenset({"id", "tree_sha256", "tier", "files", "timeout_s"})
TEST_FILE = re.compile(r"^tests/[A-Za-z0-9_][A-Za-z0-9_.-]*\.py$")
SHARD_ID = re.compile(r"^[0-9a-f]{32}$")
DIGEST = re.compile(r"^[0-9a-f]{64}$")


class Refused(ValueError):
    """A shard job that is refused; the message says which rule it broke."""


def check_header(header: dict) -> dict:
    """The header once every field is known, typed and in range; Refused otherwise."""
    extra = sorted(set(header) - FIELDS)
    if extra:
        raise Refused(f"a shard takes no field called {extra[0][:40]!r}: only {', '.join(sorted(FIELDS))}")
    if not isinstance(header.get("id"), str) or not SHARD_ID.match(header["id"]):
        raise Refused("the shard id is 32 lowercase hex digits")
    if not isinstance(header.get("tree_sha256"), str) or not DIGEST.match(header["tree_sha256"]):
        raise Refused("the tree digest is 64 lowercase hex digits")
    tier = header.get("tier", "all")
    if tier not in TIERS:
        raise Refused(f"the tier is one of {', '.join(TIERS)}")
    seconds = header.get("timeout_s", MOST_SECONDS)
    if type(seconds) is not int or not 1 <= seconds <= MOST_SECONDS:
        raise Refused(f"the time limit is a whole number of seconds from 1 to {MOST_SECONDS}")
    files = check_files(header.get("files", []))
    if tier == "gate" and files:
        raise Refused("the gate takes no files")
    return {**header, "tier": tier, "timeout_s": seconds, "files": files}


def check_files(files: object) -> list[str]:
    """The test files, each one a plain ``tests/NAME.py`` path with no repeats; none means the whole tier."""
    if not isinstance(files, list) or len(files) > MOST_FILES:
        raise Refused(f"a shard names at most {MOST_FILES} test files")
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
