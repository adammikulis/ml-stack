"""Every tracked file of a repository read for a person's details, reported and never blocked.

    ml-stack-audit                     the repository around the working directory
    ml-stack-audit --root ../other     another checkout
    ml-stack-audit --staged            the pre-commit hook's check, without a commit
    ml-stack-audit --all               the noisy kinds too: places, dates, URLs, addresses
    ml-stack-audit --floor 0.4         report a hit the recogniser is less sure of
    ml-stack-audit --json              one object per finding on stdout

The reader is the hook's: names from ``NAMES_GRAPH`` and ``NAMES_SCRAPE``, the allow-lists
``NAMES_FIXTURES`` (``--fixtures`` overrides it), ``tests/known-fixtures.txt`` and
``~/.config/pii-allow.txt``, the shape rules and the recogniser -- asked here for every kind
in ``SIGNAL`` rather than people alone. Exit 1 when there is anything to look at.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any, TextIO

from ml_stack.redact import hook
from ml_stack.redact.hook import DEFAULT_FIXTURES, FLOOR, from_database, permitted, recogniser

__all__ = ["Finding", "NOISY", "SIGNAL", "audit", "main", "tracked"]

# email addresses and phone numbers are the hook's own patterns, not the recogniser's
SIGNAL = frozenset({"PERSON", "CREDIT_CARD", "IBAN_CODE", "US_SSN", "US_PASSPORT", "CRYPTO",
                    "MEDICAL_LICENSE"})
# a driver licence is a letter and some digits, which is also h2, M1 and x00
NOISY = frozenset({"URL", "LOCATION", "DATE_TIME", "NRP", "UK_NHS", "IP_ADDRESS",
                   "US_BANK_NUMBER", "US_DRIVER_LICENSE"})
SKIP_SUFFIXES = (".json", ".jsonl", ".ipynb", ".svg")


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    what: str


def tracked(root: str, rules: hook.Shapes) -> list[str]:
    """Every file git tracks under ``root`` that the hook would read."""
    listed = hook._git(root, "ls-files", "-z").split("\0")
    skip = tuple(rules.skip_suffixes) + SKIP_SUFFIXES
    return [f for f in listed if f and not f.endswith(skip)]


def audit(root: str, *, env: Mapping[str, str] | None = None, kinds: frozenset[str] = SIGNAL,
          floor: float = FLOOR, fixtures: str | None = None, engine: Any = None,
          ) -> list[Finding]:
    """Every finding in the tracked files under ``root``, in path order. ``engine`` is the
    recogniser to ask; None builds the shared one."""
    env = os.environ if env is None else env
    rules = hook._rules(env)
    fixtures = fixtures or env.get("NAMES_FIXTURES", DEFAULT_FIXTURES)
    home = env.get("HOME") or os.path.expanduser("~")
    allowed = permitted(root, fixtures, f"{home}/.config/pii-allow.txt")
    known = {n for n in from_database(env.get("NAMES_GRAPH", ""), env.get("NAMES_SCRAPE", ""))
             if n.casefold() not in allowed}
    found: list[Finding] = []
    for path in tracked(root, rules):
        try:
            with open(os.path.join(root, path), encoding="utf-8") as fh:
                blob = fh.read()
        except (OSError, UnicodeDecodeError):
            continue
        if not blob or "\0" in blob[:2048]:
            continue
        for _, line, what in hook._findings(path, blob, known, allowed, engine, rules,
                                            kinds=kinds, floor=floor):
            found.append(Finding(path, line, what))
    return list(dict.fromkeys(found))


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ml-stack-audit",
        description="Every tracked file of a repository read for a person's details, "
                    "reported and never blocked. Exit status 1 when there is anything to "
                    "look at.")
    p.add_argument("--root", metavar="DIR",
                   help="the checkout to read (default: the repository around the working "
                        "directory)")
    which = p.add_mutually_exclusive_group()
    which.add_argument("--tracked", action="store_true",
                       help="every file git tracks, as it is in the working tree (the default)")
    which.add_argument("--staged", action="store_true",
                       help="the index: the pre-commit hook's own check, without a commit")
    p.add_argument("--all", action="store_true",
                   help="the noisy kinds too: places, dates, URLs, addresses, driver licences")
    p.add_argument("--floor", type=float, default=FLOOR, metavar="F",
                   help=f"the recogniser's confidence under which a hit is ignored "
                        f"(default: {FLOOR})")
    p.add_argument("--fixtures", metavar="PATH",
                   help="the allow-list of invented names (default: NAMES_FIXTURES, else "
                        f"{DEFAULT_FIXTURES})")
    p.add_argument("--json", action="store_true", help="one object per finding on stdout")
    return p


def main(argv: list[str] | None = None, *, env: Mapping[str, str] | None = None,
         stdout: TextIO | None = None, engine: Any = None) -> int:
    """Read the tracked files (or, with ``--staged``, the index) and print what to look at.
    Returns 1 when there is anything, else 0."""
    env = os.environ if env is None else env
    out = sys.stdout if stdout is None else stdout
    opts = _parser().parse_args(list(argv or ()))
    root = os.path.abspath(opts.root or hook._git(None, "rev-parse", "--show-toplevel").strip())
    if opts.staged:
        staged_env = dict(env)
        if opts.fixtures:
            staged_env["NAMES_FIXTURES"] = opts.fixtures
        return hook.main([], env=staged_env, root=root, stdout=out)
    if engine is None:
        engine = recogniser()
    kinds = SIGNAL | (NOISY if opts.all else frozenset())
    found = audit(root, env=env, kinds=kinds, floor=opts.floor, fixtures=opts.fixtures,
                  engine=engine)
    fixtures = opts.fixtures or env.get("NAMES_FIXTURES", DEFAULT_FIXTURES)
    if opts.json:
        json.dump([asdict(f) for f in found], out, indent=1)
        print(file=out)
        return 1 if found else 0
    if engine is None:
        print("audit: presidio is not installed, so only known names and shapes are checked",
              file=out)
    if not found:
        print("nothing to look at", file=out)
        return 0
    print(f"{len(found)} to look at (a guess, not a verdict):\n", file=out)
    for f in found:
        print(f"  {f.path}:{f.line}  {f.what}", file=out)
    print(f"\nIf a name here is invented, add it to {fixtures}:"
          f"  python -m ml_stack.redact.hook allow \"the name\"", file=out)
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
