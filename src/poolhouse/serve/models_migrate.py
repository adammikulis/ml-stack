"""``poolhouse-models migrate plan|run``: move the old model store into the Hugging Face hub cache.

Models used to be kept in ``<state>/models/<owner>/<repo>/<file>``. They now live only in the standard
hub cache (``hub.hub_cache()``), one copy per machine, so ``hf``, transformers and every other tool share
them. This is the one explicit step that moves what is left in the old store; no other command does.

``plan`` counts and sizes what would move and says what stops it, in seconds, hashing nothing. ``run``
asks the Hub which commit each repository is at, checks every file's size and digest against the Hub's own
listing, and only then puts it in ``models--owner--repo/blobs/<digest>`` with a link in
``snapshots/<commit>/`` and ``refs/main``. The old file is removed after the link is read back and checked.
A file the Hub cannot vouch for (offline, repository gone, another revision's bytes) stays where it is and
is listed. Progress is appended to ``<state>/models-migrate.log``; a cut-off run is finished by the next.
"""

from __future__ import annotations

import errno
import json
import os
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from poolhouse import files, home, hub
from poolhouse.hub import hfstore, remote
from poolhouse.log import say
from poolhouse.serve import lease_cli
from poolhouse.serve.broker import BrokerError
from poolhouse.units import human_bytes

__all__ = ["Report", "migrate", "old_store"]

LOG = "models-migrate.log"
PARTIAL = (".part", ".incomplete", ".linking")
"""Endings of a file that is half written; it is never moved."""


def old_store() -> Path:
    """The folder models used to be kept in."""
    return home.state("models")


def log_file() -> Path:
    return home.state(LOG)


@dataclass(frozen=True, slots=True)
class Item:
    """One file of the old store: ``repo`` is ``owner/name`` and ``rel`` its path inside it."""

    repo: str
    rel: str
    path: Path
    size: int


@dataclass
class Report:
    """What a plan or run found, did and left, in the words it says them."""

    mode: str
    root: str = ""
    cache: str = ""
    files: int = 0
    bytes: int = 0
    repos: int = 0
    moved: int = 0
    deduped: int = 0
    moved_bytes: int = 0
    left: list[str] = field(default_factory=list)
    blocked: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.left and not self.blocked

    def as_dict(self) -> dict[str, object]:
        return {"mode": self.mode, "store": self.root, "cache": self.cache, "files": self.files,
                "bytes": self.bytes, "repos": self.repos, "moved": self.moved,
                "deduplicated": self.deduped, "moved_bytes": self.moved_bytes,
                "left_in_place": self.left, "blocked": self.blocked, "notes": self.notes}

    def lines(self) -> list[str]:
        out = [f"old store   {self.root}", f"hub cache   {self.cache}"]
        if self.mode == "plan":
            out.append(f"to move     {self.files} file(s), {human_bytes(self.bytes)}, {self.repos} repo(s)")
        else:
            out.append(f"moved       {self.moved} file(s), {human_bytes(self.moved_bytes)}"
                       f" ({self.deduped} already in the cache)")
        out += self.notes
        out += [f"left in place: {row}" for row in self.left]
        out += [f"blocked: {row}" for row in self.blocked]
        return out


def scan(root: Path) -> tuple[list[Item], list[str]]:
    """Every file of the old store as an `Item`, and the paths that are not in an ``owner/repo`` folder
    or are half written, which are never moved. Reads names and sizes only."""
    items: list[Item] = []
    odd: list[str] = []
    for here, dirs, names in os.walk(root):
        dirs.sort()
        for name in sorted(names):
            path = Path(here) / name
            parts = path.relative_to(root).parts
            if len(parts) < 3 or path.is_symlink() or name.endswith(PARTIAL):
                odd.append(f"{path}: not a finished file inside an owner/repo folder")
                continue
            items.append(Item(f"{parts[0]}/{parts[1]}", "/".join(parts[2:]), path, path.stat().st_size))
    return items, odd


def _volume(path: Path) -> Path:
    while not path.exists() and path != path.parent:
        path = path.parent
    return path


def _cache_problem(cache: Path) -> str:
    """Why the hub cache cannot be written, or '': judged off the nearest folder that exists, so a plan
    creates nothing."""
    near = _volume(cache)
    if not near.is_dir() or not os.access(near, os.W_OK | os.X_OK):
        return f"the hub cache {cache} cannot be written ({near} is not a writable folder)"
    return ""


def broker_models() -> list[str]:
    """The model of every server the broker holds; none when no broker answers."""
    try:
        return [str(one["model"]) for one in lease_cli.snapshot()["servers"]]
    except (BrokerError, OSError):
        return []


def leased(root: Path, models: list[str]) -> list[str]:
    """The repo folders of ``root`` that a server's model sits in: their files stay put."""
    base = root.resolve()
    held: set[str] = set()
    for model in models:
        try:
            parts = Path(model).resolve().relative_to(base).parts
        except (OSError, ValueError):
            continue
        if len(parts) >= 3:
            held.add(f"{parts[0]}/{parts[1]}")
    return sorted(held)


def _log(lines: list[str]) -> None:
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    where = log_file()
    where.parent.mkdir(parents=True, exist_ok=True)
    with where.open("a", encoding="utf-8") as out:
        out.writelines(f"{stamp} {line}\n" for line in lines)


def plan(root: Path, cache: Path, models: list[str]) -> Report:
    """What `migrate` would do, off the names and sizes alone."""
    items, odd = scan(root)
    report = Report("plan", str(root), str(cache), len(items), sum(i.size for i in items),
                    len({i.repo for i in items}), left=odd)
    if not items:
        report.notes.append("nothing to move: the old store is empty or gone")
        return report
    held = leased(root, models)
    if held:
        report.blocked.append("a model server holds " + ", ".join(held)
                              + "; stop it (poolhouse-serve leases lists them) and run again")
    problem = _cache_problem(cache)
    if problem:
        report.blocked.append(problem)
    same = root.stat().st_dev == _volume(cache).stat().st_dev
    free = shutil.disk_usage(_volume(cache)).free
    report.notes.append("same volume: each file is linked into the cache and the old name removed; "
                        "no extra disk" if same else
                        f"another volume: each file is copied then removed; the cache has "
                        f"{human_bytes(free)} free and needs {human_bytes(report.bytes)}")
    return report


def _digest_problem(item: Item, one: remote.RemoteFile) -> str:
    """Why this file is not the bytes the Hub lists, or ''; the digest read off the file itself."""
    if item.size != one.size:
        return f"{item.rel} is {item.size} bytes, the Hub lists {one.size}"
    if one.sha256:
        actual, wanted = files.sha256_file(item.path), one.sha256
    else:
        actual, wanted = hfstore.git_blob_sha1(item.path, item.size), one.oid
    return "" if actual == wanted else f"{item.rel} differs from the digest the Hub lists"


def _digest_of(path: Path, name: str) -> str:
    """``path``'s digest in the form ``name`` is written in: sha256 for an LFS file, else the object id."""
    return files.sha256_file(path) if len(name) == 64 else hfstore.git_blob_sha1(path, path.stat().st_size)


def _existing(blob: Path, item: Item, name: str) -> str:
    """``'have'`` when the blob already in the cache is these bytes, else why it is not."""
    if blob.stat().st_size == item.size and _digest_of(blob, name) == name:
        return "have"
    return f"{blob.name[:12]} is in the cache with other bytes; it is not replaced"


def _into_blobs(item: Item, blob: Path, name: str) -> str:
    """Put the file's bytes at ``blob`` without ever replacing one: ``'new'``, ``'have'`` when a blob
    of that name was already there and is the same bytes, or the reason it could not be."""
    blob.parent.mkdir(parents=True, exist_ok=True)
    if blob.exists():
        return _existing(blob, item, name)
    try:
        os.link(item.path, blob)
        return "new"
    except FileExistsError:
        return _existing(blob, item, name)
    except OSError as exc:
        if exc.errno not in (errno.EXDEV, errno.EPERM, errno.EMLINK, errno.ENOTSUP):
            raise
    staged = blob.with_name(f"{blob.name}.{os.getpid()}.incomplete")
    try:
        shutil.copyfile(item.path, staged)
        if _digest_of(staged, name) != name:
            return f"the copy of {item.rel} did not hash to {name[:12]}"
        try:
            os.link(staged, blob)
        except FileExistsError:
            return _existing(blob, item, name)
        return "new"
    finally:
        staged.unlink(missing_ok=True)


def place(item: Item, one: remote.RemoteFile | None, commit: str, cache: Path) -> tuple[str, str]:
    """Move one file into the cache. Returns ``(outcome, why)``: ``new`` or ``have`` when done, else
    ``left`` and the reason; the old file is removed only after the link reads back right."""
    if one is None:
        return "left", f"{item.rel} is not in {item.repo} at {commit[:8]}"
    try:
        name = hfstore.blob_name(one)
        problem = _digest_problem(item, one)
        if problem:
            return "left", problem
        root = cache / f"models--{item.repo.replace('/', '--')}"
        got = _into_blobs(item, root / "blobs" / name, name)
        if got not in ("new", "have"):
            return "left", got
        target = root / "snapshots" / commit / item.rel
        hfstore.link(target, root / "blobs" / name)
        if not target.is_file() or target.stat().st_size != item.size:
            return "left", f"{item.rel}: the link in the cache does not read back"
    except OSError as exc:
        return "left", f"{item.rel}: {exc}"
    item.path.unlink()
    return got, ""


def _repo_state(repo: str) -> tuple[str, dict[str, remote.RemoteFile]]:
    commit = remote.commit(repo, "main")
    return commit, {f.path: f for f in remote.listing(repo, commit)}


def _tidy(root: Path) -> None:
    """Remove the folders the move emptied, and the old store itself when nothing is left in it."""
    for here, _dirs, _names in os.walk(root, topdown=False):
        try:
            Path(here).rmdir()
        except OSError:
            continue


def _one_repo(repo: str, items: list[Item], cache: Path, report: Report) -> None:
    try:
        commit, listed = _repo_state(repo)
    except (remote.RemoteError, OSError, ValueError) as exc:
        report.left += [f"{i.path}: {repo} could not be read from {remote.endpoint()} ({exc})" for i in items]
        return
    for item in items:
        outcome, why = place(item, listed.get(item.rel), commit, cache)
        if outcome == "left":
            report.left.append(f"{item.path}: {why}")
            _log([f"left {item.path}: {why}"])
            continue
        report.moved += 1
        report.deduped += outcome == "have"
        report.moved_bytes += item.size
        _log([f"{'deduplicated' if outcome == 'have' else 'moved'} {item.path} -> {repo}@{commit[:8]}/{item.rel}"])
    if report.moved and not (hfstore.repo_root(repo) / "refs" / "main").exists():
        hfstore.point_ref(repo, "main", commit)


def migrate(*, apply: bool, models: Callable[[], list[str]] = broker_models) -> Report:
    """Plan or run the move of the old store into ``hub.hub_cache()``. ``models`` says which models the
    broker's servers hold (a test names its own)."""
    root, cache = old_store(), hub.hub_cache()
    report = plan(root, cache, models())
    if not apply or report.blocked or not report.files:
        return report
    report.mode, report.left = "run", []
    items, odd = scan(root)
    report.left += odd
    _log([f"run: {len(items)} file(s) from {root} into {cache}"])
    grouped: dict[str, list[Item]] = {}
    for item in items:
        grouped.setdefault(item.repo, []).append(item)
    for repo, rows in grouped.items():
        _one_repo(repo, rows, cache, report)
    _tidy(root)
    _log([f"done: moved {report.moved}, left {len(report.left)}"])
    return report


def command(action: str, *, as_json: bool = False) -> int:
    """The ``migrate`` subcommand: print the report; exit 1 when anything is blocked or left."""
    report = migrate(apply=action == "run")
    say(json.dumps(report.as_dict(), indent=2) if as_json else "\n".join(report.lines()))
    return 0 if report.ok else 1
