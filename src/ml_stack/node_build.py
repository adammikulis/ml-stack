"""Put the node into a runtime being built: compile it from the same commit, record its checksum, prove it answers."""

from __future__ import annotations

from pathlib import Path

from ml_stack import node_binary, node_launch, runtime

CACHE = "cargo-target"


def add_to(built: runtime.Runtime, source: Path, commit: str, *, timeout: float = node_binary.BUILD_TIMEOUT) -> dict:
    """Build the node from ``source``, install it in the runtime tree and start it once on a scratch state to see it answer.

    The cargo target directory is shared by every build on this machine, so a later commit compiles only what changed.
    Returns the binary's record; raises when the build fails or the node does not answer.
    """
    binary = node_binary.build(source, cache=runtime.directory() / CACHE, timeout=timeout)
    record = node_binary.install(built.prefix, binary, commit=commit)
    node_launch.smoke(built.prefix)
    return record


def mark(prefix: Path) -> dict:
    """The fields `mark_verified` adds for a tree's node binary: its checksum (nothing on a tree with none)."""
    try:
        return {"node_sha256": node_binary.record_of(prefix)["sha256"]}
    except OSError:
        return {}

