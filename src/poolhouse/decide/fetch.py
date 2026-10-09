"""Pinned model files: each is a repository, a commit, a size and a SHA-256, and nothing is
used that does not match all four."""

from __future__ import annotations

import json
import os
import urllib.parse
from dataclasses import dataclass
from pathlib import Path

from poolhouse import home, net
from poolhouse.decide.types import BackendUnavailable, DecideError
from poolhouse.files import sha256_file, write_json
from poolhouse.httpguard import Refused
from poolhouse.safenames import safe_filename


@dataclass(frozen=True, slots=True)
class Pin:
    """One file of one commit of a Hub repository, with the size and hash it must have."""

    repo: str
    revision: str
    filename: str
    sha256: str
    size: int

    def __post_init__(self) -> None:
        if len(self.revision) != 40 or len(self.sha256) != 64:
            raise ValueError(f"{self.repo}/{self.filename}: revision and sha256 must be full hashes")


def _marks() -> Path:
    return home.cache("decide", "verified.json")


def _verified(pin: Pin, path: Path) -> bool:
    stat = path.stat()
    if stat.st_size != pin.size:
        return False
    held = json.loads(_marks().read_text()) if _marks().is_file() else {}
    if held.get(pin.sha256) == [stat.st_size, stat.st_mtime_ns]:
        return True
    if sha256_file(path) != pin.sha256:
        return False
    held[pin.sha256] = [stat.st_size, stat.st_mtime_ns]
    write_json(_marks(), held)
    return True


ENDPOINT = "https://huggingface.co"


def _url(pin: Pin) -> str:
    base = (os.environ.get("HF_ENDPOINT") or ENDPOINT).rstrip("/")
    return (f"{base}/{pin.repo}/resolve/{urllib.parse.quote(pin.revision, safe='')}/"
            f"{urllib.parse.quote(pin.filename)}")


def _where(pin: Pin) -> Path:
    """Where the verified copy of ``pin`` lives: the repository's own layout under its pinned
    revision, so files that belong together (a config beside its weights, an adapter folder) sit
    together as a loader expects."""
    return home.cache("decide", "snapshots", safe_filename(pin.repo.replace("/", "--")),
                      safe_filename(pin.revision), *(safe_filename(part)
                                                      for part in pin.filename.split("/")))


def locate(pin: Pin, *, download: bool = False) -> Path:
    """The verified file for ``pin``, downloading it only when ``download`` is set.

    The download goes through the net pipeline (allow-listed host, pinned size and SHA-256,
    format check, scan); a file that fails any check is held, never kept, and `DecideError` is
    raised.
    """
    path = _where(pin)
    if not path.is_file():
        if not download:
            raise BackendUnavailable(
                f"{pin.repo}/{pin.filename} is not downloaded; "
                "run `poolhouse decide fetch` to download it")
        want = net.Want(sha256=pin.sha256, size=pin.size, require_digest=True,
                        max_bytes=pin.size + (1 << 20), purpose="decide model")
        try:
            net.download(_url(pin), path, want)
        except net.Blocked as exc:
            raise DecideError(f"{pin.repo}/{pin.filename} does not match its pinned size and "
                              f"hash; the file was held: {exc}") from None
        except (Refused, OSError) as exc:
            raise DecideError(f"{pin.repo}/{pin.filename} could not be fetched: {exc}") from None
    if not _verified(pin, path):
        path.unlink(missing_ok=True)
        raise DecideError(f"{pin.repo}/{pin.filename} does not match its pinned size and hash; "
                          "the file was removed")
    return path
