"""These packages must import with no third-party libraries installed."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

STDLIB_ONLY = ["contracts", "media", "client", "fleet"]


@pytest.mark.parametrize("name", STDLIB_ONLY)
def test_it_imports_with_nothing_installed(name):
    src = REPO / "packages" / f"ml-stack-{name}" / "src"
    program = (
        "import sys\n"
        "sys.path = [p for p in sys.path "
        "if 'site-packages' not in p and 'dist-packages' not in p]\n"
        f"sys.path.insert(0, {str(src)!r})\n"
        f"import ml_stack.{name} as m\n"
        "print(len(getattr(m, '__all__', [])))\n"
    )
    done = subprocess.run([sys.executable, "-S", "-c", program],
                          capture_output=True, text=True, cwd=REPO)
    assert done.returncode == 0, (
        f"ml_stack.{name} needs something that is not in the standard library:\n"
        f"{done.stderr}")


def test_the_stdlib_only_packages_declare_no_dependencies():
    for name in STDLIB_ONLY:
        text = (REPO / "packages" / f"ml-stack-{name}" / "pyproject.toml").read_text()
        for line in text.splitlines():
            if line.startswith("dependencies ="):
                deps = line.split("=", 1)[1].strip()
                assert deps in ("[]", '[""]'), f"ml-stack-{name}: {line}"


def test_the_web_assets_are_not_python():
    from ml_stack.fleet.ui import ASSETS

    assert not list(ASSETS.glob("*.py"))
