"""What ``ml-stack-surface`` does, as functions that take values and return them.

`walk` asks a checkout for every command and subcommand it answers to; `capture` writes
their help and output under a directory; `compare` reads two of those and returns what
differs. `ml_stack.surface.cli` parses and prints.
"""

from __future__ import annotations

import difflib
import json
import os
import platform
import re
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ml_stack import home

__all__ = ["SAFE", "Change", "Node", "Run", "SurfaceFailed", "Taken", "capture",
           "compare", "declared", "installed", "invocations", "settle", "walk"]

PROBE = Path(__file__).with_name("probe.py")
"""The worker one command is asked through, run by path against the checkout being asked."""

SRC = Path(__file__).resolve().parents[2]
"""The ``src`` directory this module was imported from, asked when no other is named."""

WIDTH = 100
"""The terminal width every capture is taken at, so wrapping is not a difference."""

SAFE: dict[str, tuple[str, ...]] = {
    "": ("--list",),
    "help": (),
    "setup": (),
    "jobs status": (),
    "serve status": (),
    "serve limits": (),
    "serve fit": ("--room", "24G"),
    "bench status": (),
    "speech providers": (),
    "suite list": (),
}
"""The invocations that read this machine and change nothing, by the words that reach them."""


@dataclass(frozen=True, slots=True)
class Node:
    """One command or subcommand: its words, its help, and whether a bare argv is refused."""

    words: tuple[str, ...]
    help: str
    refuses: bool

    @property
    def slug(self) -> str:
        """The file name this node's capture is written under."""
        return _slug(self.words)

    @property
    def spoken(self) -> str:
        """The words as a person types them after ``ml-stack``."""
        return " ".join(self.words)


@dataclass(frozen=True, slots=True)
class Run:
    """One invocation a capture makes: where it is filed, the words typed, and why."""

    slug: str
    words: tuple[str, ...]
    verdict: str


@dataclass(frozen=True, slots=True)
class Taken:
    """What one capture came to: the nodes whose help was written, and the runs made."""

    nodes: tuple[Node, ...]
    runs: tuple[Run, ...]

    @property
    def help_only(self) -> tuple[Node, ...]:
        """The nodes nothing was run for."""
        ran = {run.slug for run in self.runs}
        return tuple(node for node in self.nodes if node.slug not in ran)


@dataclass(frozen=True, slots=True)
class Change:
    """One capture that is not the same in both directories, and the diff between them."""

    name: str
    diff: str


def _env(src: Path) -> dict[str, str]:
    """The environment a child is asked in: ``src`` first on the path, a fixed width."""
    already = os.environ.get("PYTHONPATH", "")
    path = os.pathsep.join([str(src), already]) if already else str(src)
    return {**os.environ, "PYTHONPATH": path, "COLUMNS": str(WIDTH),
            "TERM": "dumb", "NO_COLOR": "1"}


def _ask(src: Path, argv: Sequence[str], timeout: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, *argv], env=_env(src), text=True,
                          capture_output=True, stdin=subprocess.DEVNULL,
                          timeout=timeout, check=False)


class SurfaceFailed(Exception):
    """A checkout could not be asked what it answers to."""


def probe(src: Path, word: str, timeout: float = 120.0) -> dict[str, Any]:
    """What ``src`` says the command ``word`` answers to, as the probe returns it."""
    done = _ask(src, [str(PROBE), word], timeout)
    if done.returncode or not done.stdout:
        return {"word": word, "error": (done.stderr.strip().splitlines() or ["no output"])[-1]}
    return json.loads(done.stdout)


def installed(src: Path, timeout: float = 120.0) -> list[str]:
    """Every command ``src`` installs: ``""`` for ``ml-stack`` and one word per script."""
    done = _ask(src, [str(PROBE)], timeout)
    if done.returncode or not done.stdout:
        raise SurfaceFailed(f"{src} named no commands: "
                            f"{(done.stderr.strip().splitlines() or ['no output'])[-1]}")
    return list(json.loads(done.stdout))


def walk(src: Path, only: Sequence[str] = (), timeout: float = 120.0) -> list[Node]:
    """Every command and subcommand ``src`` answers to, in the order each declares them."""
    wanted = list(only) if only else installed(src, timeout)
    out: list[Node] = []
    for word in wanted:
        found = probe(src, word, timeout)
        if "error" in found:
            out.append(Node((word,) if word else (), f"[{found['error']}]\n", False))
            continue
        out.extend(Node(tuple(one["words"]), one["help"], bool(one["refuses"]))
                   for one in found["nodes"])
    return out


_RUNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(:\d{2})?"), "<when>"),
    (re.compile(r"\d{4}-\d{2}-\d{2}"), "<date>"),
    (re.compile(r"\b\d{1,2}:\d{2}:\d{2}\b"), "<clock>"),
    (re.compile(r"\b(pid|process) \d+"), r"\1 <pid>"),
    (re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), "<ip>"),
    (re.compile(r"\b0x[0-9a-f]{4,}\b"), "<address>"),
    (re.compile(r"\b\d+(?:\.\d+)?\s?(?:tok/s|tokens/s|t/s)"), "<rate>"),
    (re.compile(r"\b\d+(?:\.\d+)?\s?(?:TiB|GiB|MiB|KiB|TB|GB|MB|kB|[TGMK])\b"), "<size>"),
    (re.compile(r"\b\d+(?:\.\d+)?\s?(?:ms|us|s|min|h)\b"), "<took>"),
    (re.compile(r"\b\d+(?:\.\d+)?%"), "<share>"),
    (re.compile(r"\bport \d{2,5}\b"), "port <port>"),
    (re.compile(r":\d{4,5}\b"), ":<port>"),
)
"""What varies between two runs of the same invocation, and what stands in for it."""


def _paths(text: str, src: Path) -> str:
    node = platform.node()
    for was, now in ((str(src.parent), "<repo>"), (str(home.user_home()), "~"),
                     (node, "<host>"), (node.removesuffix(".local"), "<host>")):
        text = text.replace(was, now) if was else text
    return re.sub(r"/(?:private/)?(?:tmp|var/folders)/\S+", "<tmp>", text)


def settle(text: str, src: Path, *, output: bool) -> str:
    """``text`` with this machine's paths replaced, and for ``output`` what a rerun changes."""
    text = _paths(text, src)
    if output:
        for pattern, stands in _RUNS:
            text = pattern.sub(stands, text)
    return text


def _slug(words: Sequence[str]) -> str:
    return ".".join(words) or "ml-stack"


def declared(nodes: Sequence[Node]) -> list[Run]:
    """The `SAFE` invocations that reach a command among ``nodes``."""
    reached = {node.words[0] if node.words else "" for node in nodes}
    out = []
    for spoken, rest in SAFE.items():
        said = spoken.split()
        if (said[0] if said else "") in reached:
            out.append(Run(_slug(said), (*said, *rest), "ran"))
    return out


def invocations(nodes: Sequence[Node], help_only: bool = False) -> list[Run]:
    """Every invocation a capture makes: the declared ones, and the ones a parser refuses."""
    if help_only:
        return []
    out = declared(nodes)
    taken = {one.slug for one in out}
    out.extend(Run(node.slug, node.words, "refused")
               for node in nodes if node.refuses and node.slug not in taken)
    return out


def _printed(src: Path, run: Run, timeout: float) -> str:
    """The record of one invocation: what was typed, what it exited with, what it printed."""
    typed = " ".join(["ml-stack", *run.words])
    try:
        done = _ask(src, ["-m", "ml_stack.cli", *run.words], timeout)
    except subprocess.TimeoutExpired:
        return f"$ {typed}\nexit: timed out\n"
    parts = [f"$ {typed}", f"exit: {done.returncode}", "--- out", done.stdout.rstrip("\n"),
             "--- err", done.stderr.rstrip("\n")]
    return "\n".join(parts) + "\n"


def _write(where: Path, name: str, text: str) -> None:
    where.mkdir(parents=True, exist_ok=True)
    (where / name).write_text(text, encoding="utf-8")


def capture(out: Path, *, src: Path = SRC, only: Sequence[str] = (),
            help_only: bool = False, timeout: float = 120.0) -> Taken:
    """Write every command's help, and every safe invocation's output, under ``out``."""
    nodes = walk(src, only, timeout)
    runs = invocations(nodes, help_only)
    for node in nodes:
        _write(out / "help", f"{node.slug}.txt", settle(node.help, src, output=False))
    for run in runs:
        _write(out / "run", f"{run.slug}.txt",
               settle(_printed(src, run, timeout), src, output=True))
    said = {run.slug: run for run in runs}
    listed = {node.slug: {"words": list(node.words), "help": True,
                          "run": list(said[node.slug].words) if node.slug in said else None}
              for node in nodes}
    for slug, run in said.items():
        listed.setdefault(slug, {"words": list(run.words), "help": False,
                                 "run": list(run.words)})
    _write(out, "manifest.json", json.dumps(dict(sorted(listed.items())), indent=1) + "\n")
    _write(out, "about.json", json.dumps({"src": str(src), "width": WIDTH}, indent=1) + "\n")
    return Taken(tuple(nodes), tuple(runs))


def _files(where: Path) -> dict[str, str]:
    """Every captured file under ``where``, keyed by the name a reader would say."""
    found: dict[str, str] = {}
    for kind in ("help", "run"):
        for path in sorted((where / kind).glob("*.txt")):
            said = path.stem.replace(".", " ")
            found[f"{said} --help" if kind == "help" else said] = path.read_text(
                encoding="utf-8")
    manifest = where / "manifest.json"
    if manifest.exists():
        found["what was captured"] = manifest.read_text(encoding="utf-8")
    return found


def compare(before: Path, after: Path, lines: int = 2) -> list[Change]:
    """Every capture that is not the same in both directories, with the diff between them."""
    was, now = _files(before), _files(after)
    out: list[Change] = []
    for name in sorted(set(was) | set(now)):
        left, right = was.get(name, ""), now.get(name, "")
        if left == right:
            continue
        diff = difflib.unified_diff(left.splitlines(), right.splitlines(),
                                    fromfile=f"{before.name}: {name}",
                                    tofile=f"{after.name}: {name}", lineterm="", n=lines)
        out.append(Change(name, "\n".join(diff)))
    return out
