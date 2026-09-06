"""The README's command table is the registry's, and the registry names every command."""

from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path

from ml_stack.cli import PREFIX, commands
from ml_stack.cli.reference import HELP, TABLE, table

REPO = Path(__file__).resolve().parent.parent
README = REPO / "README.md"


def committed() -> list[str]:
    """The table as README.md carries it, under "## The commands"."""
    lines = README.read_text(encoding="utf-8").splitlines()
    start = lines.index("## The commands")
    first = next(i for i in range(start, len(lines)) if lines[i].startswith("| "))
    last = next(i for i in range(first, len(lines)) if not lines[i].startswith("| "))
    return lines[first:last]


def test_the_committed_table_is_the_one_the_registry_makes():
    assert committed() == table().splitlines(), (
        "README.md's command table and ml_stack.cli.reference disagree; "
        "scripts/reference --write writes the registry's")


def test_every_command_has_a_line_and_every_line_a_command():
    installed = set(commands())
    assert set(HELP) == installed, (
        f"in the registry and not installed: {sorted(set(HELP) - installed)}; "
        f"installed and not in the registry: {sorted(installed - set(HELP))}")


def test_every_row_names_a_command_that_exists():
    scripts = set(tomllib.loads((REPO / "pyproject.toml").read_text())["project"]["scripts"])
    for row in TABLE:
        assert row.command in scripts, f"{row.invocation!r} names no console script"


def test_every_command_has_a_row_of_its_own():
    named = {row.command for row in TABLE}
    missing = sorted(w for w in HELP if f"{PREFIX}{w}" not in named)
    assert not missing, f"no row in the README's table for: {missing}"


def test_the_script_prints_what_the_readme_holds():
    ran = subprocess.run([sys.executable, str(REPO / "scripts" / "reference")],
                         capture_output=True, text=True, check=True)
    assert ran.stdout.splitlines() == committed()
