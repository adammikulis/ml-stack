"""These parts of ml_stack must import with no third-party libraries installed."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

STDLIB_ONLY = ["contracts", "media", "client", "fleet"]


@pytest.mark.parametrize("name", STDLIB_ONLY)
def test_it_imports_with_nothing_installed(name):
    src = REPO / "src"
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


def test_installing_ml_stack_brings_in_nothing():
    """`pip install ml-stack` has to be enough on a machine that only joins the
    cluster and passes work about. Everything heavier is an extra."""
    import tomllib

    meta = tomllib.load((REPO / "pyproject.toml").open("rb"))["project"]
    assert meta["dependencies"] == []
    assert set(meta["optional-dependencies"]) >= {"app", "train", "serve", "all"}


def test_the_web_assets_are_not_python():
    """web/ is data, not code -- which is what keeps this package device tier. The tier
    check only globs *.py, so it would not notice a module smuggled in here."""
    from ml_stack.fleet.ui import ASSETS

    assert list(ASSETS.glob("*.html")), "an empty asset directory would pass anything"
    assert not list(ASSETS.glob("*.py"))
