"""Pinned model files: each is a repository, a commit, a size and a SHA-256, and nothing is
used that does not match all four."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ml_stack import home
from ml_stack.decide.types import BackendUnavailable, DecideError
from ml_stack.files import sha256_file, write_json

MAX_FILE_BYTES = 8 << 30


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


def _hub():
    try:
        import huggingface_hub
    except ImportError as exc:
        raise BackendUnavailable("fetching model files needs huggingface_hub: "
                                 "pip install 'ml-stack[hub]'") from exc
    return huggingface_hub


def locate(pin: Pin, *, download: bool = False) -> Path:
    """The verified file for ``pin``, downloading it only when ``download`` is set.

    The size is checked against the Hub's listing before any bytes are fetched, the hash
    after; a file that fails either is deleted and `DecideError` raised.
    """
    if pin.size > MAX_FILE_BYTES:
        raise DecideError(f"{pin.filename}: {pin.size} bytes is over the {MAX_FILE_BYTES} limit")
    hub = _hub()
    try:
        path = Path(hub.hf_hub_download(pin.repo, pin.filename, revision=pin.revision,
                                        local_files_only=True))
    except hub.errors.LocalEntryNotFoundError:
        if not download:
            raise BackendUnavailable(
                f"{pin.repo}/{pin.filename} is not downloaded; "
                "run `ml-stack decide fetch` to download it") from None
        meta = hub.get_hf_file_metadata(hub.hf_hub_url(pin.repo, pin.filename,
                                                       revision=pin.revision))
        if meta.size != pin.size:
            raise DecideError(f"{pin.repo}/{pin.filename}: the Hub lists {meta.size} bytes, "
                              f"expected {pin.size}") from None
        path = Path(hub.hf_hub_download(pin.repo, pin.filename, revision=pin.revision))
    if not _verified(pin, path):
        path.unlink(missing_ok=True)
        raise DecideError(f"{pin.repo}/{pin.filename} does not match its pinned size and hash; "
                          "the file was removed")
    return path
