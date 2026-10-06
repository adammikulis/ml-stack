"""Package imports and the base installation dependency contract."""

from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement

REPO = Path(__file__).resolve().parent.parent

STDLIB_ONLY = ["contracts", "fleet"]


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


@pytest.mark.parametrize("name", ["media", "client"])
def test_device_clients_import_with_only_base_dependencies(name):
    program = """
import sys, tomllib
from importlib.metadata import packages_distributions
from pathlib import Path
from packaging.requirements import Requirement
core = {Requirement(value).name.lower().replace('_', '-') for value in
        tomllib.loads(Path('pyproject.toml').read_text())['project']['dependencies']}
allowed = {module for module, distributions in packages_distributions().items()
           if any(distribution.lower().replace('_', '-') in core for distribution in distributions)}
""" + f"import ml_stack.{name}\n" + """
outside = {module.split('.')[0] for module in sys.modules
           if not module.startswith('_') and module.split('.')[0] not in sys.stdlib_module_names}
print(sorted(outside - allowed - {'ml_stack'}))
"""
    done = subprocess.run([sys.executable, "-c", program], cwd=REPO,
                          capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "[]"


def test_base_install_declares_device_security_and_board_dependencies():
    meta = tomllib.load((REPO / "pyproject.toml").open("rb"))["project"]
    dependencies = {Requirement(d).name for d in meta["dependencies"]}
    assert dependencies == {"cryptography", "ladybug", "packaging", "psutil", "pywin32", "qrcode"}
    assert set(meta["optional-dependencies"]) >= {"train", "all"}
    assert "serve" not in meta["optional-dependencies"]


def test_the_web_assets_are_not_python():
    """web/ is data, not code -- which is what keeps this package device tier. The tier
    check only globs *.py, so it would not notice a module smuggled in here."""
    from ml_stack.fleet.ui import ASSETS

    assert list(ASSETS.glob("*.html")), "an empty asset directory would pass anything"
    assert not list(ASSETS.glob("*.py"))
