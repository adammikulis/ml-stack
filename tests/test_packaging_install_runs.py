"""Running `install.sh --headless`, end to end, on this machine.

Everything else about the installers is read rather than run. This one executes the
script against a scratch prefix with a locally built wheel, so the path a first-time
user takes is one this suite has actually walked: a virtualenv appears, the console
scripts are linked, and every step after the install exits without a traceback.

`ML_STACK_OFFLINE_ZIP` keeps it off the network. `ML_STACK_MODELS=none` keeps it off
the disk. No terminal means `join_fleet` asks for nothing.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SH = REPO / "packaging" / "install.sh"
SCRIPTS = ("ml-stack", "ml-stack-serve", "ml-stack-models", "ml-stack-setup",
           "ml-stack-doctor", "ml-stack-fleet", "ml-stack-traind")

pytestmark = pytest.mark.slow


def wheel() -> Path:
    """The built wheel, or a skip saying how to make one."""
    built = sorted((REPO / "dist").glob("ml_stack-*.whl"))
    if not built:
        pytest.skip("no wheel in dist/; run `python packaging/build.py` first")
    return built[-1]


def run_installer(root: Path, *args: str, **extra: str) -> subprocess.CompletedProcess[str]:
    home = root / "home"
    home.mkdir(parents=True, exist_ok=True)
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(home),
        "ML_STACK_PREFIX": str(root / "prefix"),
        "ML_STACK_DEST": str(root / "bin"),
        "ML_STACK_MODELS": "none",
        "ML_STACK_OFFLINE_ZIP": str(wheel()),
        **extra,
    }
    for keep in ("PYENV_ROOT", "SSL_CERT_FILE", "TMPDIR"):
        if keep in os.environ:
            env[keep] = os.environ[keep]
    return subprocess.run(["sh", str(SH), *args], stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True,
                          timeout=900, stdin=subprocess.DEVNULL, env=env)


@pytest.mark.skipif(sys.platform == "win32", reason="install.ps1 is the Windows installer")
def test_the_headless_installer_leaves_a_machine_that_works(tmp_path):
    done = run_installer(tmp_path, "--headless")
    assert done.returncode == 0, f"install.sh --headless exited {done.returncode}\n{done.stdout}"

    venv = tmp_path / "prefix" / "venv"
    assert (venv / "bin" / "python").is_file(), "no virtualenv where ML_STACK_PREFIX said"
    linked = {name for name in SCRIPTS if (tmp_path / "bin" / name).exists()}
    assert linked == set(SCRIPTS), f"not linked into ML_STACK_DEST: {sorted(set(SCRIPTS) - linked)}"

    # Every step after the install runs under `|| true`, so a crash is printed to
    # stderr rather than returned; the run merges it into stdout for that reason.
    assert "Traceback (most recent call last)" not in done.stdout, done.stdout
    assert "ModuleNotFoundError" not in done.stdout, done.stdout
    for step in ("== headless", "== what this machine can do", "== models",
                 "== checking it over", "== done"):
        assert step in done.stdout, f"{step} never ran:\n{done.stdout}"


@pytest.mark.skipif(sys.platform == "win32", reason="install.ps1 is the Windows installer")
def test_the_installed_machine_says_which_version_it_runs(tmp_path):
    done = run_installer(tmp_path, "--headless")
    assert done.returncode == 0, done.stdout
    python = tmp_path / "prefix" / "venv" / "bin" / "python"
    said = subprocess.run(
        [str(python), "-c",
         "from ml_stack.fleet import updates; print(updates.state()['version'])"],
        capture_output=True, text=True, timeout=120)
    assert said.stdout.strip(), "an installed machine reports no version at all"
    assert said.stdout.strip() in wheel().name.replace("_", "-")


@pytest.mark.skipif(sys.platform == "win32", reason="install.ps1 is the Windows installer")
def test_uninstall_takes_the_venv_away(tmp_path):
    assert run_installer(tmp_path, "--headless").returncode == 0
    venv = tmp_path / "prefix" / "venv"
    assert venv.is_dir()
    gone = run_installer(tmp_path, "--headless", "--uninstall")
    assert gone.returncode == 0, gone.stdout
    assert not venv.exists(), f"the venv survived --uninstall\n{gone.stdout}"


@pytest.mark.skipif(not shutil.which("pwsh"), reason="pwsh is not on this machine")
def test_the_windows_installer_parses_and_offers_every_mode():
    done = subprocess.run(
        ["pwsh", "-NoProfile", "-Command",
         f"$null = [System.Management.Automation.Language.Parser]::ParseFile("
         f"'{REPO / 'packaging' / 'install.ps1'}', [ref]$null, [ref]$e); "
         f"if ($e) {{ $e | ForEach-Object {{ $_.Message }}; exit 1 }}"],
        capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stdout + done.stderr
