"""The Hugging Face hub cache as poolhouse's one model store.

``hub.hub_cache()`` is the folder (``$HF_HUB_CACHE``, ``$HF_HOME/hub``, ``$XDG_CACHE_HOME/huggingface/hub``
or ``~/.cache/huggingface/hub``). A repository there is laid out the way ``huggingface_hub`` writes it, so
``hf``, transformers and every other tool read what poolhouse downloads and the other way round:

    models--owner--repo/blobs/<sha256 | git sha1>      the bytes, once, named by their digest
    models--owner--repo/snapshots/<commit>/<path>      a relative symlink into blobs/
    models--owner--repo/refs/<revision>                the commit a name such as ``main`` was resolved to

Nothing else keeps models: ``<state>/models`` no longer exists as a store.
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

from poolhouse import files, hub
from poolhouse.hub import remote
from poolhouse.safenames import safe_filename, safe_join

COMMIT = re.compile(r"[a-f0-9]{40}")
DIGEST = re.compile(r"[a-f0-9]{40}|[a-f0-9]{64}")

__all__ = ["COMMIT", "blob_name", "git_blob_sha1", "link", "main_commit", "point_ref", "repo_root"]


def repo_root(repo: str) -> Path:
    """The cache folder of ``owner/name``."""
    owner, _, name = repo.partition("/")
    for part in (owner, name):
        safe_filename(part)
    return hub.hub_cache() / f"models--{owner}--{name}"


def git_blob_sha1(path: Path, size: int) -> str:
    """The git object id of a file of ``size`` bytes: what the Hub calls a small file's oid."""
    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(f"blob {size}\0".encode())
    with path.open("rb") as stream:
        while block := stream.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def blob_name(one: remote.RemoteFile) -> str:
    """What the cache calls ``one``'s bytes: its LFS sha256, else its git blob id. Refuses a file
    the Hub listed with neither, since a name that is not a digest could not be checked."""
    name = one.sha256 or one.oid
    if not DIGEST.fullmatch(name):
        raise OSError(f"the Hub listed no digest for {one.path}; it cannot be placed in the Hub cache")
    return name


def link(target: Path, blob: Path) -> None:
    """Make the snapshot file ``target`` a relative symlink to ``blob``. A link already right stays,
    a real file there (one ``poolhouse-models snapshot`` wrote) is left, a wrong link is replaced."""
    target.parent.mkdir(parents=True, exist_ok=True)
    want = os.path.relpath(blob, target.parent)
    if target.is_symlink():
        if target.readlink().as_posix() == want:
            return
    elif target.exists():
        return
    staged = target.with_name(target.name + ".linking")
    staged.unlink(missing_ok=True)
    try:
        staged.symlink_to(want)
    except OSError:
        staged.hardlink_to(blob)
    files.promote(staged, target)


def main_commit(repo: str) -> str:
    """The commit ``refs/main`` of ``repo`` names, or '' when there is none or the file holds anything
    but a commit id: the cache is a folder another tool writes, and what is read from it is not trusted."""
    try:
        text = (repo_root(repo) / "refs" / "main").read_text(encoding="utf-8").strip()
    except OSError:
        return ""
    return text if COMMIT.fullmatch(text) else ""


def point_ref(repo: str, revision: str, commit: str) -> None:
    """Record that ``revision`` meant ``commit``, as the cache does for a branch or tag name."""
    if COMMIT.fullmatch(revision):
        return
    for part in revision.split("/"):
        safe_filename(part)
    files.write_text(safe_join(repo_root(repo), f"refs/{revision}"), commit)
