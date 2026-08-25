"""A Python environment for training jobs, separate from the app itself."""

from __future__ import annotations

import json
import re
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["Environment", "Library", "CATALOG", "catalog_for"]

MIN_PYTHON = (3, 11)
WANT_PYTHON = "3.12"
STANDALONE = ("https://api.github.com/repos/astral-sh/"
              "python-build-standalone/releases/latest")


@dataclass(frozen=True, slots=True)
class Library:
    """One thing that can be installed, and what it is for."""

    name: str
    title: str
    blurb: str
    packages: tuple[str, ...]
    index: str = ""
    size_mb: int = 0
    default: bool = False
    platforms: tuple[str, ...] = ()
    vendors: tuple[str, ...] = ()

    def applies(self, vendor: str = "") -> bool:
        if self.platforms and sys.platform not in self.platforms:
            return False
        return not self.vendors or vendor in self.vendors


CATALOG: tuple[Library, ...] = (
    Library("core", "Training essentials",
            "Arrays, checkpoint files, and ml-stack's own training code. "
            "Needed by everything below.",
            ("ml-stack[train]",),
            size_mb=40, default=True),
    Library("torch-cuda", "PyTorch for NVIDIA",
            "Training on an NVIDIA card.",
            ("torch",), size_mb=2500, default=True, vendors=("nvidia",)),
    Library("torch-rocm", "PyTorch for AMD",
            "Training on an AMD card through ROCm.",
            ("torch",), index="https://download.pytorch.org/whl/rocm6.2",
            size_mb=2200, default=True, vendors=("amd",)),
    Library("torch-cpu", "PyTorch",
            "Training on the processor. Slower, but works anywhere.",
            ("torch",), index="https://download.pytorch.org/whl/cpu",
            size_mb=200, default=False, vendors=("cpu", "apple")),
    Library("mlx", "MLX",
            "Training on Apple silicon, using the GPU.",
            ("mlx>=0.18",), size_mb=120, default=True,
            platforms=("darwin",), vendors=("apple",)),
    Library("vision", "Images",
            "Reading and resizing pictures.",
            ("pillow>=10.0",), size_mb=15),
    Library("huggingface", "Hugging Face models",
            "Starting from a downloaded model rather than from scratch.",
            ("transformers>=4.40", "datasets>=2.19"), size_mb=300),
    Library("telemetry", "Temperature and clocks",
            "Reporting this machine's temperature and GPU clock.",
            ("darwin-perf>=0.1",), size_mb=5, default=True,
            platforms=("darwin",)),
)


def catalog_for(vendor: str = "") -> list[Library]:
    """The libraries worth offering on this machine."""
    return [lib for lib in CATALOG if lib.applies(vendor)]


@dataclass
class Environment:
    """A virtual environment the daemon owns and runs training jobs with."""

    root: Path
    _cache: dict[str, Any] = field(default_factory=dict)

    @property
    def path(self) -> Path:
        return Path(self.root).expanduser() / "env"

    @property
    def python(self) -> Path:
        bindir = "Scripts" if sys.platform == "win32" else "bin"
        name = "python.exe" if sys.platform == "win32" else "python"
        return self.path / bindir / name

    @property
    def exists(self) -> bool:
        return self.python.exists()

    # -- finding an interpreter -----------------------------------------
    def host_python(self) -> Path | None:
        """A Python on this machine new enough to build the environment with."""
        if not getattr(sys, "frozen", False):
            if sys.version_info[:2] >= MIN_PYTHON:
                return Path(sys.executable)
        for name in ("python3.13", "python3.12", "python3.11", "python3", "python"):
            found = shutil.which(name)
            if not found:
                continue
            try:
                out = subprocess.run([found, "-c",
                                      "import sys;print('%d.%d' % sys.version_info[:2])"],
                                     capture_output=True, text=True, timeout=10)
            except (OSError, subprocess.SubprocessError):
                continue
            if out.returncode != 0:
                continue
            try:
                major, minor = (int(x) for x in out.stdout.strip().split("."))
            except ValueError:
                continue
            if (major, minor) >= MIN_PYTHON:
                return Path(found)
        return None

    # -- fetching one --------------------------------------------------
    def standalone_python(self) -> Path | None:
        """A Python this app downloaded for itself, if it has one."""
        base = Path(self.root).expanduser() / "python"
        found = base / ("python.exe" if sys.platform == "win32" else "bin/python3")
        return found if found.exists() else None

    def _asset_name(self) -> str:
        machine = platform.machine().lower()
        arch = "aarch64" if machine in ("arm64", "aarch64") else "x86_64"
        target = {"darwin": "apple-darwin",
                  "win32": "pc-windows-msvc"}.get(sys.platform, "unknown-linux-gnu")
        return f"-{arch}-{target}-install_only_stripped"

    def fetch_python(self, *, on_progress: Any = None) -> Path:
        """Download a Python to build the environment with.

        The system Python on a stock machine is often older than 3.11 -- macOS ships
        3.9 -- and a bundled app has no interpreter of its own to lend.
        """
        found = self.standalone_python()
        if found:
            return found
        if on_progress:
            on_progress("Downloading Python")

        req = urllib.request.Request(STANDALONE, headers={"User-Agent": "ml-stack"})
        with urllib.request.urlopen(req, timeout=60) as r:
            release = json.loads(r.read())
        want = self._asset_name()
        assets = [a for a in release.get("assets", ())
                  if want in a["name"] and a["name"].endswith(".tar.gz")
                  and f"cpython-{WANT_PYTHON}." in a["name"]]
        if not assets:
            raise EnvironmentError(
                f"no Python {WANT_PYTHON} build for this machine ({want.strip('-')})")

        base = Path(self.root).expanduser() / "python"
        base.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "python.tar.gz"
            with urllib.request.urlopen(
                    urllib.request.Request(assets[0]["browser_download_url"],
                                           headers={"User-Agent": "ml-stack"}),
                    timeout=600) as r, archive.open("wb") as fh:
                shutil.copyfileobj(r, fh)
            if on_progress:
                on_progress("Unpacking Python")
            with tarfile.open(archive) as tf:
                for member in tf.getmembers():
                    if member.name.startswith("/") or ".." in Path(member.name).parts:
                        raise EnvironmentError("refusing an archive that escapes its "
                                               "directory")
                tf.extractall(tmp)
            unpacked = Path(tmp) / "python"
            if not unpacked.is_dir():
                raise EnvironmentError("the download did not contain a python directory")
            shutil.rmtree(base, ignore_errors=True)
            shutil.move(str(unpacked), str(base))

        got = self.standalone_python()
        if got is None:
            raise EnvironmentError("the downloaded Python is not where it was expected")
        return got

    # -- building it ----------------------------------------------------
    def create(self, *, on_progress: Any = None) -> Path:
        """Make the environment if it is not there. Returns the interpreter."""
        if self.exists:
            return self.python
        base = self.host_python() or self.standalone_python()
        if base is None:
            base = self.fetch_python(on_progress=on_progress)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if on_progress:
            on_progress("Creating the environment")
        made = subprocess.run([str(base), "-m", "venv", str(self.path)],
                              capture_output=True, text=True)
        if made.returncode != 0:
            raise EnvironmentError(
                f"could not build the environment: {_last_error(made.stderr)}")
        return self.python

    def wheels(self) -> Path | None:
        """Wheels shipped alongside the app, for the ml-stack packages themselves.

        They are not on an index, so without these the environment can hold torch and
        still not be able to run a training job.
        """
        if getattr(sys, "frozen", False):
            bundled = Path(getattr(sys, "_MEIPASS", "")) / "wheels"
            return bundled if bundled.is_dir() else None
        for parent in Path(__file__).resolve().parents:
            candidate = parent / "dist"
            if candidate.is_dir() and any(candidate.glob("ml_stack-*.whl")):
                return candidate
        return None

    def pip(self, args: list[str], *, timeout: float = 3600.0
            ) -> subprocess.CompletedProcess:
        found = self.wheels()
        if found and args and args[0] == "install":
            args = [args[0], "--find-links", str(found), *args[1:]]
        return subprocess.run([str(self.python), "-m", "pip", *args],
                              capture_output=True, text=True, timeout=timeout)

    # -- what is in it --------------------------------------------------
    def installed(self) -> dict[str, str]:
        """Package name to version, for what the environment holds."""
        if not self.exists:
            return {}
        try:
            out = self.pip(["list", "--format=json"], timeout=60)
        except (OSError, subprocess.SubprocessError):
            return {}
        if out.returncode != 0:
            return {}
        try:
            return {p["name"].lower(): p["version"] for p in json.loads(out.stdout)}
        except (ValueError, KeyError, TypeError):
            return {}

    def has(self, library: Library) -> bool:
        have = self.installed()
        return all(_base(spec) in have for spec in library.packages)

    def state(self, vendor: str = "") -> dict[str, Any]:
        have = self.installed()
        return {
            "ready": self.exists,
            "python": str(self.python) if self.exists else "",
            "host_python": str(self.host_python() or self.standalone_python() or ""),
            "can_build": True,
            "libraries": [
                {"name": lib.name, "title": lib.title, "blurb": lib.blurb,
                 "size_mb": lib.size_mb, "default": lib.default,
                 "installed": all(_base(s) in have for s in lib.packages),
                 "version": have.get(_base(lib.packages[0]), "")}
                for lib in catalog_for(vendor)
            ],
        }

    # -- changing it ----------------------------------------------------
    def install(self, names: list[str], *, on_progress: Any = None) -> dict[str, Any]:
        """Install the named libraries. Returns what happened, per library."""
        self.create(on_progress=on_progress)
        wanted = {lib.name: lib for lib in CATALOG}
        done: dict[str, Any] = {}
        for name in names:
            lib = wanted.get(name)
            if lib is None:
                done[name] = {"ok": False, "error": "no such library"}
                continue
            if on_progress:
                on_progress(f"Installing {lib.title}")
            args = ["install", "--upgrade", *lib.packages]
            if lib.index:
                args += ["--index-url", lib.index]
            try:
                out = self.pip(args)
            except subprocess.TimeoutExpired:
                done[name] = {"ok": False, "error": "timed out"}
                continue
            done[name] = ({"ok": True} if out.returncode == 0
                          else {"ok": False, "error": _last_error(out.stderr)})
        return done

    def uninstall(self, names: list[str]) -> dict[str, Any]:
        wanted = {lib.name: lib for lib in CATALOG}
        done: dict[str, Any] = {}
        for name in names:
            lib = wanted.get(name)
            if lib is None or not self.exists:
                done[name] = {"ok": False, "error": "not installed"}
                continue
            out = self.pip(["uninstall", "-y", *(_base(s) for s in lib.packages)])
            done[name] = ({"ok": True} if out.returncode == 0
                          else {"ok": False, "error": _last_error(out.stderr)})
        return done

    def remove(self) -> None:
        shutil.rmtree(self.path, ignore_errors=True)


def _base(spec: str) -> str:
    return re.split(r"[<>=!~\[]", spec, maxsplit=1)[0].strip().lower()


def _last_error(stderr: str) -> str:
    lines = [ln for ln in (stderr or "").splitlines() if ln.strip()]
    for line in reversed(lines):
        if "error" in line.lower():
            return line.strip()[:200]
    return (lines[-1][:200] if lines else "failed")
