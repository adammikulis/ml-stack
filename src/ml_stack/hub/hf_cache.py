"""Public, revisioned Hub snapshots downloaded through the net pipeline."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

from ml_stack import files, hub, lock, net, worktreerules
from ml_stack.hub import remote
from ml_stack.hub.transfer import PICKLES
from ml_stack.net.sniff import expected_kind
from ml_stack.safenames import safe_filename, safe_join

COMMIT = re.compile(r"[a-f0-9]{40}")
REPO = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*")
MAX_FILES = 20_000
MAX_FILE = 1 << 40


@dataclass(frozen=True, slots=True)
class Member:
    name: str
    size: int
    digest: str
    lfs: bool

    def verify(self, path: Path, _headers: dict[str, str]) -> str:
        if path.stat().st_size != self.size:
            return "snapshot file size differs from its revision metadata"
        if self.lfs:
            actual = files.sha256_file(path)
        else:
            digest = hashlib.sha1(usedforsecurity=False)
            digest.update(f"blob {self.size}\0".encode())
            with path.open("rb") as stream:
                while block := stream.read(1 << 20):
                    digest.update(block)
            actual = digest.hexdigest()
        return "" if actual == self.digest else "snapshot file differs from its revision digest"


def _members(data: dict) -> list[Member]:
    rows = data.get("siblings")
    if not isinstance(rows, list) or len(rows) > MAX_FILES:
        raise ValueError("invalid snapshot file listing")
    members = []
    seen = set()
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("rfilename"), str):
            raise ValueError("invalid snapshot file metadata")
        name = row["rfilename"]
        for part in name.split("/"):
            safe_filename(part)
        if len(name) > 1024 or name.casefold() in seen:
            raise ValueError("invalid or duplicate snapshot file path")
        seen.add(name.casefold())
        if any(part.startswith(".git") for part in name.split("/")) or name.lower().endswith(PICKLES):
            continue
        size = row.get("size")
        lfs = row.get("lfs")
        if lfs is not None and not isinstance(lfs, dict):
            raise ValueError("invalid snapshot LFS metadata")
        digest = lfs.get("sha256") if lfs is not None else row.get("blobId")
        width = 64 if lfs is not None else 40
        if (type(size) is not int or not 0 <= size <= MAX_FILE
                or not isinstance(digest, str) or not re.fullmatch(r"[a-f0-9]{" + str(width) + "}", digest)
                or (lfs is not None and lfs.get("size") != size)):
            raise ValueError("snapshot files need bounded sizes and revision digests")
        members.append(Member(name, size, digest, lfs is not None))
    if not members:
        raise ValueError("snapshot contains no supported files")
    return members


def fetch(repo: str, *, revision: str = "main", repo_type: str = "model") -> Path:
    """Download a public model or dataset revision into the standard Hub cache."""
    if not REPO.fullmatch(repo) or repo_type not in {"model", "dataset"}:
        raise ValueError("snapshot needs an owner/repository and model or dataset type")
    if not isinstance(revision, str) or not revision or len(revision) > 200:
        raise ValueError("snapshot needs a bounded revision")
    for part in revision.split("/"):
        safe_filename(part)
    cache = hub.hub_cache().resolve()
    if worktreerules.checkouts(cache) is not None:
        raise ValueError("the Hub cache must be outside a Git checkout")
    base = remote.endpoint()
    url = f"{base}/api/{repo_type}s/{repo}/revision/{quote(revision, safe='')}?blobs=true"
    data = net.default().json(url, net.Ask(purpose="public Hub snapshot", max_bytes=8 << 20, tries=3))
    if not isinstance(data, dict) or not isinstance(data.get("sha"), str) or not COMMIT.fullmatch(data["sha"]):
        raise ValueError("snapshot metadata needs a resolved commit")
    commit = data["sha"]
    if COMMIT.fullmatch(revision) and commit != revision:
        raise ValueError("snapshot metadata names another revision")
    members = _members(data)
    root = safe_join(cache, f"{repo_type}s--{repo.replace('/', '--')}")
    folder = safe_join(root, f"snapshots/{commit}")
    destinations = [safe_join(root, f"snapshots/{commit}/{member.name}") for member in members]
    if not COMMIT.fullmatch(revision):
        destinations.append(safe_join(root, f"refs/{revision}"))
    if any(worktreerules.checkouts(path) is not None for path in (root, folder, *destinations)):
        raise ValueError("the Hub snapshot must be outside a Git checkout")
    prefix = "datasets/" if repo_type == "dataset" else ""
    with lock.only_one(root / ".ml-stack-snapshot.lock", announce=lambda _text: None):
        for member in members:
            target = safe_join(root, f"snapshots/{commit}/{member.name}")
            if worktreerules.checkouts(target) is not None:
                raise ValueError("snapshot members must be outside a Git checkout")
            if target.is_file() and not member.verify(target, {}):
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            url = f"{base}/{prefix}{repo}/resolve/{commit}/{quote(member.name, safe='/')}"
            net.download(url, target, net.Want(
                kind=expected_kind(member.name), sha256=member.digest if member.lfs else "", size=member.size,
                require_digest=member.lfs, max_bytes=max(1, member.size), verify=member.verify,
                purpose="public Hub snapshot"))
        if not COMMIT.fullmatch(revision):
            files.write_text(safe_join(root, f"refs/{revision}"), commit)
    return folder
