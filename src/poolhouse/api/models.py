"""``ph.models``: model files in the Hugging Face hub cache, where poolhouse keeps and finds every model.

`pull` downloads one file of a Hub repository (every shard of a sharded build) into the standard cache layout
(``$HF_HUB_CACHE``, else ``$HF_HOME/hub``, else ``~/.cache/huggingface/hub``); `path` says where it is without
downloading anything. Anything that serves or loads a model (`ph.serve`, `poolhouse-serve`) finds it there.
"""

from __future__ import annotations

import re
from pathlib import Path

from poolhouse import errors, hub

__all__ = ["cache_dir", "path", "pull"]

REPO = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*")


def cache_dir() -> Path:
    """The Hub cache folder models are kept in."""
    return hub.hub_cache()


def pull(repo: str, file: str) -> Path:
    """Download ``file`` of the Hub repository ``repo`` (``owner/name``) into the cache; the path of the file there.

    Raises `Error` for a repository or file that does not exist or a download that fails; a file already in the
    cache is not downloaded again.
    """
    if not REPO.fullmatch(repo) or not file:
        raise errors.Error("name a repository as owner/name and a file in it")
    try:
        return hub.fetch(f"hf:{repo}/{file}")
    except (ValueError, OSError) as exc:
        raise errors.Error(str(exc)) from None


def path(repo: str, file: str) -> Path | None:
    """The cached path of ``file`` of ``repo``, or None when it has not been pulled. Downloads nothing."""
    if not REPO.fullmatch(repo):
        raise errors.Error("name a repository as owner/name")
    snapshots = cache_dir() / f"models--{repo.replace('/', '--')}" / "snapshots"
    found = sorted(snapshots.glob(f"*/{file}")) if snapshots.is_dir() else []
    return found[-1] if found else None
