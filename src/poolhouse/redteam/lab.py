"""The fixtures a run stands up: a page server, a honeypot, a scratch directory, the alias
host names, and the guard that keeps every connection on this machine."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path

from poolhouse import home
from poolhouse.redteam.daemon import Running
from poolhouse.redteam.egress import local_only
from poolhouse.redteam.evidence import Canary, Honeypot, Pages, secret_for
from poolhouse.redteam.tools import Toolbox, dns_aliases

__all__ = ["Lab", "lab"]


@dataclass(slots=True)
class Lab:
    """One run's fixtures and the model it attacks. ``model_url`` is the base URL of an
    OpenAI-compatible server ("" for none); ``seed`` fixes every secret."""

    root: Path
    pages: Pages
    honeypot: Honeypot
    model_url: str = ""
    model_name: str = ""
    seed: str = "redteam"
    daemon_url: str = ""
    token: str = ""
    files: Path = Path()
    counter: int = 0

    def canary(self, label: str = "") -> Canary:
        """A fresh canary in its own directory, with the honeypot emptied."""
        self.counter += 1
        where = self.root / f"attempt-{self.counter}"
        where.mkdir()
        self.honeypot.reset()
        return Canary(where, self.honeypot, secret_for(f"{self.seed}:{label}", self.counter))

    def toolbox(self, canary: Canary, *, guarded: bool = True) -> Toolbox:
        return Toolbox(canary, self.pages, guarded=guarded)


@contextmanager
def own_home(path: Path) -> Iterator[Path]:
    """Point ``POOLHOUSE_HOME`` at ``path`` for the block and put the old value back after. The lab forges requests, plants
    decoys and rotates keys, and every one of those lands in whatever home is current: without this the red-team
    command quarantined 127.0.0.1 in the real sentinel store and left key backups beside the real cluster key."""
    path.mkdir(parents=True, exist_ok=True)
    before = os.environ.get(home.ROOT_ENV)
    os.environ[home.ROOT_ENV] = str(path)
    try:
        yield path
    finally:
        if before is None:
            os.environ.pop(home.ROOT_ENV, None)
        else:
            os.environ[home.ROOT_ENV] = before


@contextmanager
def lab(*, model_url: str = "", model_name: str = "", seed: str = "redteam",
        served: Running | None = None) -> Iterator[Lab]:
    """A `Lab` for the block: servers up, aliases resolving, connections confined to this
    machine, everything torn down at the end. ``served`` is the daemon to attack, if any."""
    with ExitStack() as stack:
        stack.enter_context(local_only())
        stack.enter_context(dns_aliases())
        scratch = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="redteam-")))
        stack.enter_context(own_home(scratch / "machine-state"))   # the attacks quarantine peers and write key backups: never in the real home
        pages, honeypot = Pages(), Honeypot()
        stack.callback(pages.close)
        stack.callback(honeypot.close)
        yield Lab(scratch, pages, honeypot, model_url, model_name, seed, *(
            (served.url, served.token, served.files) if served else ("", "", Path())))
