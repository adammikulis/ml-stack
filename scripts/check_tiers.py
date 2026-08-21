#!/usr/bin/env python3
"""Enforce the tier rule. This is the check that must never go red.

Three tiers, by where the code runs:

    device  embedded / mobile / host app -- standard library ONLY
    host    a desktop that serves models  -- + psutil, hf_hub, httpx, ...
    lab     a desktop that trains         -- + mlx, torch, transformers, datasets

The rule is mechanical: a package may import the standard library, its own tier, and any
*lower* tier. Never a higher one. A device package that grows a torch import cannot be
cross-compiled onto an embedded target, and without this check the failure surfaces at
deploy time rather than at review time.

Two checks:

1. **Static** -- walk every import in every package and confirm the tier ordering, and
   that device packages import nothing outside the standard library.
2. **Live** (``--live``) -- actually import each device package in a subprocess whose
   ``sys.path`` carries no site-packages. This is the one that cannot be fooled by a
   lazy import inside a function, which is exactly how the rule usually gets broken.

Run: ``python scripts/check_tiers.py [--live]``
"""

from __future__ import annotations

import argparse
import ast
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PACKAGES = REPO / "packages"

TIER_ORDER = {"device": 0, "host": 1, "lab": 2}

# The tier each mainspring subpackage belongs to. Adding a package means adding it here --
# an unlisted package fails the check rather than defaulting to permissive.
TIERS: dict[str, str] = {
    "contracts": "device",
    "media": "device",
    "client": "device",
    "serve": "host",
    "gguf": "host",
    "speech": "host",
    "vision": "host",
    "backend": "lab",
    "graph": "lab",
    "train": "lab",
    "testing": "lab",
}

STDLIB = set(sys.stdlib_module_names)


def tier_of(package: str) -> str:
    try:
        return TIERS[package]
    except KeyError:
        raise SystemExit(
            f"package 'mainspring.{package}' has no tier in scripts/check_tiers.py. "
            "Add it to TIERS -- an unlisted package is a design decision that was "
            "never made, not a package that is exempt."
        ) from None


def imports_of(path: Path) -> set[str]:
    """Every top-level module name imported by a file, including inside functions."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError as exc:
        raise SystemExit(f"{path}: {exc}") from exc

    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # relative import, same package
                continue
            if node.module:
                found.add(node.module.split(".")[0])
    return found


def mainspring_subpackages_of(path: Path) -> set[str]:
    """Which ``mainspring.<x>`` subpackages a file imports."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError:
        return set()

    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("mainspring."):
            parts = node.module.split(".")
            if len(parts) >= 2:
                found.add(parts[1])
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("mainspring."):
                    parts = alias.name.split(".")
                    if len(parts) >= 2:
                        found.add(parts[1])
    return found


def check_static() -> list[str]:
    problems: list[str] = []

    for source in sorted(PACKAGES.glob("*/src/mainspring/*/**/*.py")):
        package = source.relative_to(PACKAGES).parts[3]
        tier = tier_of(package)
        rank = TIER_ORDER[tier]
        where = source.relative_to(REPO)

        for other in sorted(mainspring_subpackages_of(source)):
            if other == package:
                continue
            other_rank = TIER_ORDER[tier_of(other)]
            if other_rank > rank:
                problems.append(
                    f"{where}: {tier} package 'mainspring.{package}' imports "
                    f"{tier_of(other)} package 'mainspring.{other}'. "
                    f"A {tier} package may not depend on a {tier_of(other)} one."
                )

        if tier == "device":
            external = {
                name
                for name in imports_of(source)
                if name not in STDLIB and name != "mainspring" and not name.startswith("_")
            }
            for name in sorted(external):
                problems.append(
                    f"{where}: device package 'mainspring.{package}' imports "
                    f"non-stdlib module '{name}'. Device packages must import on a "
                    f"machine with nothing installed."
                )

    return problems


def check_live() -> list[str]:
    """Import each device package with site-packages stripped from sys.path.

    A lazy ``import torch`` inside a function body passes the static check trivially and
    still breaks the target. This catches it.
    """
    problems: list[str] = []
    device = [name for name, tier in TIERS.items() if tier == "device"]

    for name in sorted(device):
        src = PACKAGES / f"mainspring-{name}" / "src"
        if not src.is_dir():
            continue

        program = (
            "import sys\n"
            "sys.path = [p for p in sys.path if 'site-packages' not in p "
            "and 'dist-packages' not in p]\n"
            f"sys.path.insert(0, {str(src)!r})\n"
            f"import mainspring.{name} as m\n"
            "print(sorted(getattr(m, '__all__', []))[:3])\n"
        )
        result = subprocess.run(
            [sys.executable, "-S", "-c", program],
            capture_output=True,
            text=True,
            cwd=REPO,
        )
        if result.returncode != 0:
            tail = (result.stderr or result.stdout).strip().splitlines()
            problems.append(
                f"mainspring.{name} does not import without site-packages:\n    "
                + "\n    ".join(tail[-6:])
            )

    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--live",
        action="store_true",
        help="also import device packages in a stdlib-only subprocess",
    )
    args = parser.parse_args()

    problems = check_static()
    if args.live:
        problems += check_live()

    if problems:
        print(f"tier check FAILED ({len(problems)} problem(s)):\n", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1

    counts = ", ".join(
        f"{sum(1 for t in TIERS.values() if t == tier)} {tier}"
        for tier in ("device", "host", "lab")
    )
    print(f"tier check OK ({counts})" + (" [live]" if args.live else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
