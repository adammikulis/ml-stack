"""Build immutable runtime wheels from a git revision."""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

from ml_stack.safenames import unpack

ORIGIN = "source-checkout"


def source_checkout() -> Path | None:
    """Return the tracked source checkout recorded in the installed runtime."""
    marker = Path(__file__).with_name(ORIGIN)
    if not marker.is_file():
        return None
    source = Path(marker.read_text(encoding="utf-8").strip())
    return source if source.is_absolute() and (source / ".git").exists() else None


def stamp(wheel: Path, commit: str, checkout: Path) -> None:
    """Record runtime provenance and update the wheel's integrity manifest."""
    with zipfile.ZipFile(wheel) as archive:
        contents = {name: archive.read(name) for name in archive.namelist()}
    records = [name for name in contents if name.endswith(".dist-info/RECORD")]
    if len(records) != 1:
        raise ValueError("wheel must contain one RECORD")
    record = records[0]
    contents["ml_stack/fleet/built-from"] = (commit + "\n").encode()
    contents[f"ml_stack/fleet/{ORIGIN}"] = (str(checkout.resolve()) + "\n").encode()
    manifest = io.StringIO(newline="")
    writer = csv.writer(manifest)
    for name, data in contents.items():
        if name != record:
            digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
            writer.writerow((name, f"sha256={digest}", str(len(data))))
    writer.writerow((record, "", ""))
    contents[record] = manifest.getvalue().encode()
    with zipfile.ZipFile(wheel, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in contents.items():
            archive.writestr(name, data)


def install_checkout(checkout: Path, *, timeout: float) -> tuple[int, str]:
    """Install a wheel built from checkout HEAD and resolve its dependencies."""
    try:
        with tempfile.TemporaryDirectory(prefix="ml-stack-runtime-") as temporary:
            stage = Path(temporary)
            commit = _run(["git", "-C", str(checkout), "rev-parse", "HEAD"], timeout)
            snapshot = stage / "source.zip"
            _run(["git", "-C", str(checkout), "archive", "--format=zip",
                  "--output", str(snapshot), commit], timeout)
            source, wheels = stage / "source", stage / "wheels"
            unpack(snapshot, source)
            _run([sys.executable, "-m", "pip", "wheel", "--no-deps",
                  "--wheel-dir", str(wheels), str(source)], timeout)
            found = list(wheels.glob("ml_stack-*.whl"))
            if len(found) != 1:
                return 1, "source revision must build exactly one ml-stack wheel"
            stamp(found[0], commit, checkout)
            _run([sys.executable, "-m", "pip", "install", "--upgrade", str(found[0])], timeout)
            said = _run([sys.executable, "-m", "pip", "install", "--force-reinstall",
                         "--no-deps", str(found[0])], timeout)
            return 0, said[-2000:]
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        return 1, str(exc)[-2000:]


def _run(argv: list[str], timeout: float) -> str:
    done = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    output = f"{done.stdout}{done.stderr}".strip()
    if done.returncode:
        raise ValueError(output or f"{argv[0]} exited {done.returncode}")
    return output
