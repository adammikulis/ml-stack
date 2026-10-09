"""A cache of attack verdicts, a seeded sample of attacks, and the limits a run asks a server for.

An attack that passed is not run again while everything its verdict depends on is
byte-identical: the model file, the attack, the guard or prompt version, and the source of the
modules the attack touches. An attack that failed is run every time.
"""

from __future__ import annotations

import hashlib
import importlib.util
import math
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from random import Random

from poolhouse import home
from poolhouse.files import read_json, sha256_file, version_of, versioned, write_json

__all__ = ["VERSION", "Attack", "Cache", "Limits", "Outcome", "Subject", "Verdict", "key",
           "model_hash", "run", "sample", "surface_files"]

VERSION = 1
SCRIPTED = "scripted"


@dataclass(frozen=True)
class Attack:
    """One attack: its id, its class, and the modules or files whose code it exercises."""

    id: str
    klass: str = ""
    surfaces: tuple[str, ...] = ()


@dataclass(frozen=True)
class Verdict:
    """Whether the defence held, and what was seen."""

    passed: bool
    detail: str = ""


@dataclass(frozen=True)
class Outcome:
    """A verdict, whether it came from the cache, and the seconds spent getting it."""

    attack: str
    verdict: Verdict
    cached: bool
    seconds: float


@dataclass(frozen=True)
class Subject:
    """What is under attack: the model file's hash and the guard or prompt version."""

    model: str
    guard: str


@dataclass(frozen=True)
class Limits:
    """What a run asks the server for: few tokens, no thinking, a warm prompt prefix."""

    max_tokens: int = 256
    max_steps: int = 4
    stop: tuple[str, ...] = ()
    thinking: bool = False
    cache_prompt: bool = True

    def body(self, base: Mapping[str, object] | None = None) -> dict[str, object]:
        """A chat-completions request body carrying these limits over ``base``."""
        out: dict[str, object] = {**(base or {}), "max_tokens": self.max_tokens,
                                  "cache_prompt": self.cache_prompt}
        if self.stop:
            out["stop"] = list(self.stop)
        if not self.thinking:
            out["chat_template_kwargs"] = {"enable_thinking": False}
        return out


def model_hash(path: Path | str, *, directory: Path | None = None) -> str:
    """The sha256 of a model file, kept by path, size and mtime so it is read once."""
    target = Path(path).resolve()
    stat = target.stat()
    where = (directory or home.cache("verdicts")) / "models.json"
    seen = read_json(where, {})
    stamp = f"{target}\0{stat.st_size}\0{stat.st_mtime_ns}"
    if stamp not in seen:
        seen[stamp] = sha256_file(target)
        write_json(where, seen)
    return str(seen[stamp])


def surface_files(surface: str) -> list[Path]:
    """The files behind a surface: a path, a directory's ``.py`` files, or a dotted module."""
    path = Path(surface)
    if path.is_dir():
        return sorted(path.rglob("*.py"))
    if path.is_file():
        return [path]
    try:
        spec = importlib.util.find_spec(surface)
    except (ImportError, ValueError):
        return []
    if spec is None or spec.origin is None:
        return []
    origin = Path(spec.origin)
    return sorted(origin.parent.rglob("*.py")) if spec.submodule_search_locations else [origin]


def surface_hash(surfaces: Iterable[str]) -> str:
    """One hash over the bytes of every file the surfaces name."""
    digest = hashlib.sha256()
    for surface in sorted(surfaces):
        digest.update(surface.encode())
        files = surface_files(surface)
        digest.update(b"\0missing" if not files else b"")
        for file in files:
            digest.update(file.read_bytes())
    return digest.hexdigest()


def key(attack: Attack, subject: Subject) -> str:
    """The cache key: everything the attack's verdict depends on."""
    parts = (str(VERSION), subject.model, attack.id, subject.guard, surface_hash(attack.surfaces))
    return hashlib.sha256("\0".join(parts).encode()).hexdigest()


class Cache:
    """Verdicts on disk, one JSON file per key, under the poolhouse cache directory."""

    def __init__(self, directory: Path | None = None) -> None:
        self.directory = directory or home.cache("verdicts", "attacks")

    def lookup(self, wanted: str) -> Verdict | None:
        """The verdict recorded under ``wanted``, or None when there is none of this shape."""
        record = read_json(self.directory / f"{wanted}.json", None)
        if version_of(record) != VERSION or not isinstance(record.get("verdict"), dict):
            return None
        said = record["verdict"]
        return Verdict(bool(said.get("passed")), str(said.get("detail", "")))

    def store(self, wanted: str, attack: Attack, verdict: Verdict, seconds: float) -> None:
        """Record a verdict with the attack it answers and what it cost."""
        write_json(self.directory / f"{wanted}.json", versioned(
            {"attack": attack.id, "verdict": asdict(verdict), "seconds": round(seconds, 3)},
            VERSION))


def run(attacks: Sequence[Attack], execute: Callable[[Attack], Verdict], subject: Subject, *,
        cache: Cache | None = None, use_cache: bool = True) -> list[Outcome]:
    """Execute attacks one after another, serving a passing verdict from the cache when its key
    is unchanged. ``use_cache=False`` runs every attack and records nothing."""
    kept = cache if cache is not None else Cache()
    out: list[Outcome] = []
    for attack in attacks:
        wanted = key(attack, subject)
        said = kept.lookup(wanted) if use_cache else None
        if said is not None and said.passed:
            out.append(Outcome(attack.id, said, True, 0.0))
            continue
        began = time.monotonic()
        verdict = execute(attack)
        spent = time.monotonic() - began
        if use_cache:
            kept.store(wanted, attack, verdict, spent)
        out.append(Outcome(attack.id, verdict, False, spent))
    return out


def sample(attacks: Sequence[Attack], *, fraction: float = 0.1, seed: str,
           changed: Iterable[str] = ()) -> list[Attack]:
    """A seeded share of the attacks, plus every attack in a class that touches a changed file."""
    moved = {Path(c).resolve() for c in changed}
    touching = [a for a in attacks
                if moved & {f.resolve() for s in a.surfaces for f in surface_files(s)}]
    classes = {a.klass for a in touching if a.klass}
    chosen = {a.id for a in touching} | {a.id for a in attacks if a.klass in classes}
    ordered = sorted(a.id for a in attacks)
    count = max(1, math.ceil(fraction * len(ordered))) if ordered else 0
    chosen |= set(Random(seed).sample(ordered, count))  # noqa: S311
    return [a for a in attacks if a.id in chosen]

