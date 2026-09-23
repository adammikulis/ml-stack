"""Build wheels and a standalone bundle for this platform.

    python packaging/build.py             wheels only
    python packaging/build.py --bundle    wheels, the daemon, and the window around it
    python packaging/build.py --bundle --no-window    the daemon on its own
    python packaging/build.py --wheelhouse            wheels, and the extras beside them
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "dist"
APP = ROOT / "app"
EXTERNAL = ("pyinstaller", "packaging", "psutil", "numpy")
SIDECAR = "ml-stack-headless"


def run(argv: list[str], **kw) -> None:
    done = subprocess.run(argv, cwd=kw.pop("cwd", ROOT), **kw)
    if done.returncode != 0:
        raise SystemExit(f"failed: {' '.join(argv)}")


def wheels() -> list[Path]:
    DIST.mkdir(exist_ok=True)
    # The bundle carries every wheel it finds here, including a previous version's.
    for old in DIST.glob("*.whl"):
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
         f"ml-stack[{extras()}] @ file://{ROOT}"], stdout=subprocess.DEVNULL)
    for mine in out.glob("ml_stack-*.whl"):
        mine.unlink()
    return sorted(out.glob("*.whl"))


def built_from() -> Path:
    """Write the commit this tree is at where ``ml-stack.spec`` puts it beside
    `ml_stack.fleet.measuring`, which is what the frozen daemon answers as its commit."""
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
    # force-reinstall: the version has not changed between builds, so pip would keep
    # the wheel already in the build venv and bundle code that is one edit behind.
    run([str(pip), "install", "-q", "--no-index", "--find-links", str(DIST),
         "--force-reinstall", "--no-deps", "ml-stack"])

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

    made: list[Path] = []
    for found in _artifacts(APP / "src-tauri" / "target" / "release" / "bundle"):
        into = DIST / "bundle" / found.name
        shutil.rmtree(into, ignore_errors=True)
        into.unlink(missing_ok=True)
        (shutil.copytree if found.is_dir() else shutil.copy2)(found, into)
        made.append(into)
    if sys.platform == "darwin":
        _adhoc_sign(DIST / "bundle" / "ml-stack.app")
    return made


def _artifacts(out: Path) -> list[Path]:
    """What the bundler wrote for this platform."""
    if sys.platform == "darwin":
        return sorted((out / "macos").glob("*.app"))
    if sys.platform == "win32":
        return sorted((out / "nsis").glob("*-setup.exe"))
    return sorted((out / "appimage").glob("*.AppImage"))


def _adhoc_sign(app: Path) -> None:
    """Sign with no identity, which is what macOS needs to open it at all."""
    if app.exists():
        run(["codesign", "--force", "--deep", "--sign", "-", str(app)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


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
    a = ap.parse_args(argv)

    if a.clean:
        for path in (DIST, ROOT / ".build-venv", ROOT / ".build-work",
                     APP / "src-tauri" / "target"):
            shutil.rmtree(path, ignore_errors=True)

    built = wheels()
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
