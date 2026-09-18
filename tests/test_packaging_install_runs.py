"""Running `install.sh --headless`, end to end, on this machine.

Everything else about the installers is read rather than run. This one executes the
script against a scratch prefix with a locally built wheel, so the path a first-time
user takes is one this suite has actually walked: a virtualenv appears, the console
scripts are linked, and every step after the install exits without a traceback.

`ML_STACK_OFFLINE_ZIP` keeps it off the network. `ML_STACK_MODELS=none` keeps it off
the disk. No terminal means `join_fleet` asks for nothing. `ML_STACK_OFFLINE_WHEELS`
points the extras at wheels on the disk; the ones built here stand in for the real
distributions, which are tens of megabytes and on the network.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from ml_stack.installed import STANDARD, extras

REPO = Path(__file__).resolve().parent.parent
SH = REPO / "packaging" / "install.sh"
SCRIPTS = ("ml-stack", "ml-stack-serve", "ml-stack-models", "ml-stack-setup",
           "ml-stack-doctor", "ml-stack-fleet", "ml-stack-traind")

pytestmark = pytest.mark.slow

# Parses install.ps1 and reads its syntax tree: every switch the header documents is in the
# param block, and something assigns each of the four modes.
READS_THE_PS1 = r"""
$errors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile(
    $env:ML_STACK_PS1, [ref]$null, [ref]$errors)
if ($errors) { $errors | ForEach-Object { $_.Message }; exit 1 }
$switches = @($ast.ParamBlock.Parameters | ForEach-Object { $_.Name.VariablePath.UserPath })
foreach ($want in 'Headless', 'Dev', 'System', 'Uninstall') {
    if ($switches -notcontains $want) { "no -$want switch"; exit 1 }
}
$modes = @($ast.FindAll({
    param($node)
    $node -is [System.Management.Automation.Language.AssignmentStatementAst] -and
    $node.Left.Extent.Text -eq '$mode'
}, $true) | ForEach-Object { $_.Right.Extent.Text.Trim('"') })
foreach ($want in 'app', 'headless', 'dev', 'system') {
    if ($modes -notcontains $want) { "nothing sets the $want mode"; exit 1 }
}
"""


def wheel() -> Path:
    """The wheel in dist/, built first when it is missing or older than src/."""
    built = sorted((REPO / "dist").glob("ml_stack-*.whl"))
    source = max((path.stat().st_mtime for path in (REPO / "src").rglob("*.py")), default=0.0)
    if not built or built[-1].stat().st_mtime < source:
        subprocess.run([sys.executable, str(REPO / "packaging" / "build.py")],
                       check=True, stdout=subprocess.DEVNULL, timeout=900)
        built = sorted((REPO / "dist").glob("ml_stack-*.whl"))
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


# Pulls the offline install out of install.ps1 by name and runs it against a real pip: the
# wheelhouse it picks, the file:// URL it builds, and the extras that land in the venv.
DRIVES_THE_PS1 = r"""
$ErrorActionPreference = "Stop"
$ast = [System.Management.Automation.Language.Parser]::ParseFile(
    $env:ML_STACK_PS1, [ref]$null, [ref]$null)
$all = $ast.FindAll({ param($node)
    $node -is [System.Management.Automation.Language.FunctionDefinitionAst] }, $true)
foreach ($want in 'Find-Wheelhouse', 'Local-Uri', 'Install-Offline') {
    $fn = @($all | Where-Object { $_.Name -eq $want })
    if (-not $fn) { "install.ps1 has no $want"; exit 1 }
    Invoke-Expression $fn[0].Extent.Text
}
$extras = $env:ML_STACK_EXTRAS
$offZip = $env:ML_STACK_OFFLINE_ZIP
$offWhl = $env:ML_STACK_OFFLINE_WHEELS
"windows uri: " + (Local-Uri "C:\dir\pkg.whl")
"wheelhouse: " + (Find-Wheelhouse $offZip)
Install-Offline $env:ML_STACK_PIP
"""


@pytest.mark.skipif(not shutil.which("pwsh"), reason="pwsh is not on this machine")
def test_the_windows_installer_takes_its_extras_from_the_wheels_on_the_disk(tmp_path):
    """The Windows offline path installed ml-stack alone, where the online one installs
    every extra. pwsh runs the functions that changed that; the venv and pip are this
    machine's, and the paths are checked in the Windows shape the machine cannot run."""
    venv = tmp_path / "venv"
    subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True, timeout=300)
    pip = venv / ("Scripts/pip.exe" if sys.platform == "win32" else "bin/pip")
    done = subprocess.run(
        ["pwsh", "-NoProfile", "-Command", DRIVES_THE_PS1],
        capture_output=True, text=True, timeout=900,
        env={**os.environ,
             "ML_STACK_PS1": str(REPO / "packaging" / "install.ps1"),
             "ML_STACK_PIP": str(pip),
             "ML_STACK_EXTRAS": extras(),
             "ML_STACK_OFFLINE_ZIP": str(wheel()),
             "ML_STACK_OFFLINE_WHEELS": str(wheelhouse(tmp_path / "wheels"))})
    assert done.returncode == 0, done.stdout + done.stderr
    assert "windows uri: file:///C:/dir/pkg.whl" in done.stdout, done.stdout
    assert "extras from the wheels in" in done.stdout, done.stdout
    python = venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    for one in STANDARD:
        got = subprocess.run([str(python), "-c", f"import {one.module}"],
                             capture_output=True, text=True, timeout=120)
        assert got.returncode == 0, f"{one.extra} did not install: {got.stderr}"
    said = subprocess.run([str(python), "-m", "ml_stack.installed"],
                          capture_output=True, text=True, timeout=120)
    assert said.returncode == 0, said.stdout + said.stderr


@pytest.mark.skipif(not shutil.which("pwsh"), reason="pwsh is not on this machine")
def test_the_windows_installer_falls_back_to_ml_stack_alone_without_the_wheels(tmp_path):
    """A wheelhouse missing one distribution used to be the whole install failing. It
    installs ml-stack and says which parts it did not get."""
    venv = tmp_path / "venv"
    subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True, timeout=300)
    pip = venv / ("Scripts/pip.exe" if sys.platform == "win32" else "bin/pip")
    half = tmp_path / "wheels"
    stand_in_wheel(half, "numpy", "1.26.0", "numpy")
    done = subprocess.run(
        ["pwsh", "-NoProfile", "-Command", DRIVES_THE_PS1],
        capture_output=True, text=True, timeout=900,
        env={**os.environ,
             "ML_STACK_PS1": str(REPO / "packaging" / "install.ps1"),
             "ML_STACK_PIP": str(pip),
             "ML_STACK_EXTRAS": extras(),
             "ML_STACK_OFFLINE_ZIP": str(wheel()),
             "ML_STACK_OFFLINE_WHEELS": str(half)})
    assert done.returncode == 0, done.stdout + done.stderr
    assert "does not hold every wheel" in done.stdout, done.stdout
    python = venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    said = subprocess.run([str(python), "-m", "ml_stack.installed"],
                          capture_output=True, text=True, timeout=120)
    assert said.returncode == 1, "a fallback install reports nothing missing"
    for one in STANDARD:
        assert f"{one.name}: not installed" in said.stdout, said.stdout


@pytest.mark.skipif(not shutil.which("pwsh"), reason="pwsh is not on this machine")
def test_the_windows_installer_parses_and_offers_every_mode():
    done = subprocess.run(["pwsh", "-NoProfile", "-Command", READS_THE_PS1],
                          capture_output=True, text=True, timeout=120,
                          env={**os.environ,
                               "ML_STACK_PS1": str(REPO / "packaging" / "install.ps1")})
    assert done.returncode == 0, done.stdout + done.stderr


def stand_in_wheel(into: Path, name: str, version: str, module: str) -> Path:
    """A wheel holding one empty module, named so pip reads it as ``name==version``."""
    dist = name.replace("-", "_")
    info = f"{dist}-{version}.dist-info"
    files = {
        f"{module}.py": f"__version__ = {version!r}\n",
        f"{info}/METADATA": f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n",
        f"{info}/WHEEL": ("Wheel-Version: 1.0\nGenerator: tests\n"
                          "Root-Is-Purelib: true\nTag: py3-none-any\n"),
    }
    into.mkdir(parents=True, exist_ok=True)
    path = into / f"{dist}-{version}-py3-none-any.whl"
    with zipfile.ZipFile(path, "w") as whl:
        for at, body in files.items():
            whl.writestr(at, body)
        whl.writestr(f"{info}/RECORD",
                     "".join(f"{at},,\n" for at in [*files, f"{info}/RECORD"]))
    return path


def wheelhouse(into: Path) -> Path:
    """A directory holding one stand-in wheel for every extra a full install has."""
    versions = {"ladybug": "0.20.4", "huggingface_hub": "1.32.0", "ddgs": "9.0.0",
                "matplotlib": "3.11.2", "numpy": "1.26.0", "trafilatura": "2.0.0"}
    for module, version in versions.items():
        stand_in_wheel(into, module.replace("_", "-"), version, module)
    return into


@pytest.mark.skipif(sys.platform == "win32", reason="install.ps1 is the Windows installer")
def test_an_offline_install_says_which_parts_it_did_not_get(tmp_path):
    """No wheels on the disk means no store, no downloads and no graph maths. A machine
    that is quietly missing half of what the online one has is worse than one that says
    so, because nothing it does later names the reason."""
    done = run_installer(tmp_path, "--headless")
    assert done.returncode == 0, done.stdout
    assert "== what came with it" in done.stdout, done.stdout
    for one in STANDARD:
        assert f"{one.name}: not installed" in done.stdout, done.stdout
        assert one.fix in done.stdout, done.stdout
    assert "ML_STACK_OFFLINE_WHEELS" in done.stdout, done.stdout


@pytest.mark.skipif(sys.platform == "win32", reason="install.ps1 is the Windows installer")
def test_an_offline_install_takes_its_extras_from_the_wheels_beside_it(tmp_path):
    """With them it is the machine the online path produces: every extra installed, from
    the disk, with no index consulted."""
    done = run_installer(tmp_path, "--headless",
                         ML_STACK_OFFLINE_WHEELS=str(wheelhouse(tmp_path / "wheels")))
    assert done.returncode == 0, done.stdout
    assert "every part of a full install is here" in done.stdout, done.stdout
    python = tmp_path / "prefix" / "venv" / "bin" / "python"
    for one in STANDARD:
        got = subprocess.run([str(python), "-c", f"import {one.module}"],
                             capture_output=True, text=True, timeout=120)
        assert got.returncode == 0, f"{one.extra} did not install: {got.stderr}"
