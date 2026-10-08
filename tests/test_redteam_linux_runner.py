"""Linux tests use admitted selectors and immutable installed assets."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.redteam
ROOT = Path(__file__).resolve().parents[1]


def test_container_spawn_keeps_arguments_literal_and_mounts_read_only():
    from test_container_launch import ContainerRun

    for hostile in ("$(touch /tmp/injected)", "tests/test_layers.py; --privileged", "--mount=type=bind,src=/,dst=/host"):
        with pytest.raises(RuntimeError, match="fixture admission"):
            ContainerRun([sys.executable, "-m", "pytest", hostile], {})


def test_setup_selects_one_built_wheel_and_refuses_missing_artifacts(tmp_path):
    setup = (ROOT / "scripts/test-on-linux-setup").read_text()
    selection = setup[setup.index('WHEEL=(/wheel/'):setup.index('# Keyed by')]
    selection = selection.replace('/wheel/', f'{tmp_path}/')
    command = 'set -euo pipefail\n' + selection + '\nprintf "%s" "$WHEEL"'
    missing = subprocess.run(["bash", "-c", command], capture_output=True, text=True)
    assert missing.returncode != 0
    wheel = tmp_path / "ml_stack-0.0.0-py3-none-any.whl"
    wheel.touch()
    selected = subprocess.run(["bash", "-c", command], capture_output=True, text=True)
    assert selected.returncode == 0, selected.stderr
    assert selected.stdout == str(wheel)
    (tmp_path / "ml_stack-0.0.1-py3-none-any.whl").touch()
    ambiguous = subprocess.run(["bash", "-c", command], capture_output=True, text=True)
    assert ambiguous.returncode != 0


def test_setup_installs_wheel_for_dependencies_and_private_entrypoints():
    setup = (ROOT / "scripts/test-on-linux-setup").read_text()
    assert 'python -m pip wheel --quiet --no-deps --wheel-dir /wheel /work' in setup
    assert '"$ENV_DIR/bin/pip" install -q "$WHEEL$EXTRAS"' in setup
    assert '"$PRIV/bin/pip" install -q --no-deps "$WHEEL"' in setup
    assert 'wheel-v1 $DEPS $EXTRAS $IMAGE' in setup
    assert ' -e ' not in setup and '--editable' not in setup


def test_container_installed_distribution_and_child_import_provenance():
    if sys.prefix != "/priv-venv":
        pytest.skip("requires the maintained Linux container environment")
    from importlib.metadata import distribution

    installed = distribution("ml-stack")
    direct = json.loads(installed.read_text("direct_url.json"))
    assert direct["url"].startswith("file:///wheel/ml_stack-")
    assert direct["url"].endswith(".whl")
    assert not direct.get("dir_info", {}).get("editable", False)
    assert Path(installed.locate_file("ml_stack/__init__.py")).is_file()
    child = subprocess.run(
        [sys.executable, "-c", "import ml_stack; print(ml_stack.__file__)"],
        capture_output=True, text=True, check=True,
    )
    assert Path(child.stdout.strip()) == ROOT / "src/ml_stack/__init__.py"
