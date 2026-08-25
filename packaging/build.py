"""Build wheels and a standalone bundle for this platform.

    python packaging/build.py            wheels only
    python packaging/build.py --bundle   wheels plus a standalone app
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "dist"
EXTERNAL = ("pyinstaller", "pywebview", "psutil")


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


def bundle(built: list[Path]) -> Path:
    env = ROOT / ".build-venv"
    if not env.exists():
        run([sys.executable, "-m", "venv", str(env)])
    pip = env / ("Scripts" if sys.platform == "win32" else "bin") / "pip"
    run([str(pip), "install", "-q", "--upgrade", *EXTERNAL])
    # force-reinstall: the version has not changed between builds, so pip would keep
    # the wheel already in the build venv and bundle code that is one edit behind.
    run([str(pip), "install", "-q", "--no-index", "--find-links", str(DIST),
         "--force-reinstall", "--no-deps", "ml-stack"])

    tool = env / ("Scripts" if sys.platform == "win32" else "bin") / "pyinstaller"
    for spec in ("ml-stack-app.spec", "ml-stack.spec"):
        run([str(tool), "--clean", "--noconfirm", "--distpath", str(DIST / "bundle"),
             "--workpath", str(ROOT / ".build-work"), spec],
            cwd=ROOT / "packaging")
    made = DIST / "bundle"
    print(f"\nbundle: {made}")
    for item in sorted(made.iterdir()):
        size = sum(f.stat().st_size for f in item.rglob("*") if f.is_file()) \
            if item.is_dir() else item.stat().st_size
        print(f"  {item.name}  {size / 2**20:.1f} MB")
    return made


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="build")
    ap.add_argument("--bundle", action="store_true",
                    help="also build a standalone app for this platform")
    ap.add_argument("--clean", action="store_true")
    a = ap.parse_args(argv)

    if a.clean:
        for path in (DIST, ROOT / ".build-venv", ROOT / ".build-work"):
            shutil.rmtree(path, ignore_errors=True)

    built = wheels()
    print(f"{len(built)} wheels in {DIST}")
    for w in built:
        print(f"  {w.name}  {w.stat().st_size / 1024:.0f} KB")
    if a.bundle:
        bundle(built)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
