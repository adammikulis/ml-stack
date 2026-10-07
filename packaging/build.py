"""Build wheels, dependency wheelhouses and standalone platform bundles."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import zipfile
from email.parser import BytesParser
from pathlib import Path

from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "dist"
APP = ROOT / "app"
EXTERNAL = ("pyinstaller", "packaging", "psutil", "ladybug>=0.20.4,<0.21", "py-machineid")
SIDECAR = "ml-stack-headless"


def run(argv: list[str], **kw) -> None:
    done = subprocess.run(argv, cwd=kw.pop("cwd", ROOT), **kw)
    if done.returncode != 0:
        raise SystemExit(f"failed: {' '.join(argv)}")


def wheels() -> list[Path]:
    DIST.mkdir(exist_ok=True)
    for old in DIST.glob("ml_stack-*.whl"):
        old.unlink()
    run([sys.executable, "-m", "build", "--wheel", "--outdir", str(DIST), str(ROOT)],
        stdout=subprocess.DEVNULL)
    return sorted(DIST.glob("*.whl"))


def wheelhouse(out: Path) -> list[Path]:
    """Download every extra a full install has into ``out``, as wheels."""
    sys.path.insert(0, str(ROOT / "src"))
    from ml_stack.installed import extras

    out.mkdir(parents=True, exist_ok=True)
    run([sys.executable, "-m", "pip", "wheel", "--wheel-dir", str(out),
         "--find-links", str(DIST), "--find-links", str(out),
         f"ml-stack[{extras()}] @ file://{ROOT}"], stdout=subprocess.DEVNULL)
    for mine in out.glob("ml_stack-*.whl"):
        mine.unlink()
    return sorted(out.glob("*.whl"))


def owned_telemetry(source: Path) -> Path:
    """Build a validated metal-smi wheel from an isolated source snapshot."""
    source = source.expanduser().resolve()
    if not source.is_dir() or not (source / 'pyproject.toml').is_file():
        raise SystemExit('--metal-smi-source must name a project directory with pyproject.toml')
    with tempfile.TemporaryDirectory(prefix='ml-stack-metal-smi-') as temporary:
        stage = Path(temporary)
        copied, output = stage / 'source', stage / 'wheels'
        shutil.copytree(source, copied, ignore=shutil.ignore_patterns(
            '.git', '.venv', 'venv', '__pycache__', '*.pyc', '*.egg-info',
            'build', 'dist', '.pytest_cache', '.ruff_cache'))
        run([sys.executable, '-m', 'pip', 'wheel', '--no-deps', '--wheel-dir', str(output), str(copied)],
            cwd=stage, stdout=subprocess.DEVNULL)
        found = list(output.glob('*.whl'))
        if len(found) != 1:
            raise SystemExit('metal-smi source must build exactly one wheel')
        _telemetry_metadata(found[0])
        DIST.mkdir(parents=True, exist_ok=True)
        into = DIST / found[0].name
        shutil.copy2(found[0], into)
    return into


def _telemetry_metadata(wheel: Path) -> None:
    """Require the owned telemetry distribution and declared minimum version."""
    with zipfile.ZipFile(wheel) as archive:
        metadata = [name for name in archive.namelist() if name.endswith('.dist-info/METADATA')]
        if len(metadata) != 1:
            raise SystemExit('metal-smi wheel must contain one distribution metadata record')
        headers = BytesParser().parsebytes(archive.read(metadata[0]))
    try:
        version = Version(headers.get('Version', '0'))
    except InvalidVersion as exc:
        raise SystemExit('telemetry wheel has invalid version metadata') from exc
    if canonicalize_name(headers.get('Name', '')) != 'metal-smi' or version < Version('1.1.0'):
        raise SystemExit('telemetry wheel must provide metal-smi>=1.1.0')


def built_from() -> Path:
    """Write the current commit marker into the build directory."""
    sys.path.insert(0, str(ROOT / "src"))
    from ml_stack.fleet.measuring import BUILT_FROM, installed_commit

    where = ROOT / ".build-work" / BUILT_FROM
    where.parent.mkdir(parents=True, exist_ok=True)
    where.write_text(installed_commit() + "\n", encoding="utf-8")
    return where


def daemon() -> Path:
    """Freeze the daemon with PyInstaller. Returns the binary it wrote."""
    env = ROOT / ".build-venv"
    if not env.exists():
        run([sys.executable, "-m", "venv", str(env)])
    pip = env / ("Scripts" if sys.platform == "win32" else "bin") / "pip"
    run([str(pip), "install", "-q", "--upgrade", *EXTERNAL])
    run([str(pip), "install", "-q", "--no-index", "--find-links", str(DIST),
         "--force-reinstall", "--no-deps", "ml-stack"])
    run([str(pip), "install", "-q", "--find-links", str(DIST),
         "ml-stack[agents,hub,fleet-onboard,coordinator]"])

    built_from()
    tool = env / ("Scripts" if sys.platform == "win32" else "bin") / "pyinstaller"
    run([str(tool), "--clean", "--noconfirm", "--distpath", str(DIST / "bundle"),
         "--workpath", str(ROOT / ".build-work"), "ml-stack.spec"],
        cwd=ROOT / "packaging")
    made = DIST / "bundle" / (SIDECAR + (".exe" if sys.platform == "win32" else ""))
    if not made.is_file():
        raise SystemExit(f"PyInstaller wrote no {made.name}")
    return made


def target_triple() -> str:
    """The Rust target this machine builds for, which is how the sidecar is named."""
    out = subprocess.run(["rustc", "-vV"], capture_output=True, text=True)
    for line in out.stdout.splitlines():
        if line.startswith("host: "):
            return line.split(" ", 1)[1].strip()
    raise SystemExit("rustc could not say what it builds for; install Rust")


def window(frozen: Path) -> list[Path]:
    """Build the window around the frozen daemon. Returns what landed in dist/bundle."""
    binaries = APP / "src-tauri" / "binaries"
    binaries.mkdir(parents=True, exist_ok=True)
    suffix = ".exe" if sys.platform == "win32" else ""
    beside = binaries / f"{SIDECAR}-{target_triple()}{suffix}"
    shutil.copy2(frozen, beside)
    beside.chmod(beside.stat().st_mode | 0o111)

    npm = shutil.which("npm")
    if npm is None:
        raise SystemExit("the window is built with node and npm; neither is installed")
    run([npm, "ci", "--silent"], cwd=APP)
    run([npm, "run", "--silent", "build"], cwd=APP)

    artifacts = _artifacts(APP / "src-tauri" / "target" / "release" / "bundle")
    if not artifacts:
        raise SystemExit("the native build produced no application bundle")
    made: list[Path] = []
    for found in artifacts:
        into = DIST / "bundle" / found.name
        shutil.rmtree(into, ignore_errors=True)
        into.unlink(missing_ok=True)
        (shutil.copytree if found.is_dir() else shutil.copy2)(found, into)
        made.append(into)
    if sys.platform == "darwin":
        for app in made:
            _adhoc_sign(app)
    return made


def _artifacts(out: Path) -> list[Path]:
    """What the bundler wrote for this platform."""
    if sys.platform == "darwin":
        return sorted((out / "macos").glob("*.app"))
    if sys.platform == "win32":
        return sorted((out / "nsis").glob("*-setup.exe"))
    return sorted((out / "appimage").glob("*.AppImage"))


def _adhoc_sign(app: Path) -> None:
    """Ad-hoc sign and verify the native application bundle."""
    if app.exists():
        run(["codesign", "--force", "--deep", "--sign", "-", str(app)],
            stdout=subprocess.DEVNULL)
        run(["codesign", "--verify", "--deep", "--strict", str(app)],
            stdout=subprocess.DEVNULL)


def report(made: Path) -> None:
    print(f"\nbundle: {made}")
    for item in sorted(made.iterdir()):
        size = sum(f.stat().st_size for f in item.rglob("*") if f.is_file()) \
            if item.is_dir() else item.stat().st_size
        print(f"  {item.name}  {size / 2**20:.1f} MB")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="build")
    ap.add_argument("--bundle", action="store_true",
                    help="also build a standalone app for this platform")
    ap.add_argument("--no-window", action="store_true",
                    help="freeze the daemon, and stop before the window around it")
    ap.add_argument("--wheelhouse", action="store_true",
                    help="also download the extras, for a machine with no network")
    ap.add_argument("--clean", action="store_true")
    ap.add_argument('--metal-smi-source', type=Path,
                    help='build and bundle owned metal-smi>=1.1.0 from this local source directory')
    a = ap.parse_args(argv)

    if a.clean:
        for path in (DIST, ROOT / ".build-venv", ROOT / ".build-work",
                     APP / "src-tauri" / "target"):
            shutil.rmtree(path, ignore_errors=True)

    built = wheels()
    if a.metal_smi_source:
        built = sorted({*built, owned_telemetry(a.metal_smi_source)})
    print(f"{len(built)} wheels in {DIST}")
    for w in built:
        print(f"  {w.name}  {w.stat().st_size / 1024:.0f} KB")
    if a.wheelhouse:
        house = wheelhouse(DIST / "wheels")
        print(f"{len(house)} extra wheels in {DIST / 'wheels'}")
    if a.bundle:
        frozen = daemon()
        if not a.no_window:
            window(frozen)
        report(DIST / "bundle")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
