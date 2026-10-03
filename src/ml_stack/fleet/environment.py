"""A Python environment for training jobs, separate from the app itself."""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

from ml_stack import net
from ml_stack.files import promote
from ml_stack.http import ServerError
from ml_stack.httpguard import Refused

__all__ = ["CATALOG", "Environment", "Library", "catalog_for"]

PYTHON = "3.13"
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
    Library("bench", "Measuring",
            "Running the model sweeps other machines send this one.",
            ("ml-stack[graph,store,hub]",), size_mb=150),
    Library("vision", "Images",
            "Reading and resizing pictures.",
            ("pillow>=10.0",), size_mb=15),
    Library("huggingface", "Hugging Face models",
            "Starting from a downloaded model rather than from scratch.",
            ("transformers>=4.40", "datasets>=2.19"), size_mb=300),
    Library("decide-pointer", "Decision models",
            "CPU pointer-head inference, trained checkpoints and the Strands 2B decision model. Download model weights in Tools.",
            ("ml-stack[decide-pointer]",), size_mb=500),
    Library("gym", "All live environments",
            "MetaDrive, RWARE, SUMO-RL and Stable-Baselines3 including traffic-driving co-simulation.",
            ("ml-stack[gym]",), size_mb=1400),
    Library("gym-driving", "Smart car · MetaDrive",
            "Native 3D driving, lidar, traffic and vehicle dynamics.",
            ("ml-stack[gym-driving]",), size_mb=800),
    Library("gym-warehouse", "Warehouse · RWARE",
            "Cooperative warehouse robot environments.",
            ("ml-stack[gym-warehouse]",), size_mb=30),
    Library("gym-traffic", "Traffic · SUMO-RL",
            "Traffic simulation and reinforcement-learning signal control.",
            ("ml-stack[gym-traffic]",), size_mb=250),
    Library("gym-rl", "Reinforcement learning · Stable-Baselines3",
            "PPO training, checkpoints and policy evaluation.",
            ("ml-stack[gym-rl]",), size_mb=300),
    Library("telemetry", "Temperature and clocks",
            "Reporting this machine's temperature and GPU clock.",
            ("metal-smi>=1.1.0",), size_mb=5, default=True,
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
        """A Python on this machine to build the environment with."""
        if not getattr(sys, "frozen", False):
            return Path(sys.executable)
        found = shutil.which(f"python{PYTHON}")
        return Path(found) if found else None

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
        """Download a Python to build the environment with."""
        found = self.standalone_python()
        if found:
            return found
        if on_progress:
            on_progress("Downloading Python")

        try:
            release = net.default().json(STANDALONE, net.Ask(purpose="python builds", tries=3)) or {}
        except (ServerError, ValueError) as exc:
            raise OSError(f"could not reach the Python builds: {exc}") from None
        want = self._asset_name()
        assets = [a for a in release.get("assets", ())
                  if want in a["name"] and a["name"].endswith(".tar.gz")
                  and f"cpython-{PYTHON}." in a["name"]]
        if not assets:
            raise OSError(
                f"no Python {PYTHON} build for this machine ({want.strip('-')})")

        base = Path(self.root).expanduser() / "python"
        base.parent.mkdir(parents=True, exist_ok=True)
        asset = assets[0]
        digest = str(asset.get("digest") or "").removeprefix("sha256:")
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "python.tar.gz"
            try:
                net.download(asset["browser_download_url"], archive, net.Want(
                    sha256=digest, require_digest=True, size=int(asset.get("size") or 0),
                    max_bytes=2 << 30, purpose="python download"))
            except (net.Blocked, net.Truncated, ServerError, Refused) as exc:
                raise OSError(f"could not download Python: {exc}") from None
            if on_progress:
                on_progress("Unpacking Python")
            with tarfile.open(archive) as tf:
                try:
                    tf.extractall(tmp, filter="data")
                except tarfile.FilterError as exc:
                    raise OSError(f"refusing an archive that escapes its directory: "
                                  f"{exc}") from None
            unpacked = Path(tmp) / "python"
            if not unpacked.is_dir():
                raise OSError("the download did not contain a python directory")
            shutil.rmtree(base, ignore_errors=True)
            promote(unpacked, base)

        got = self.standalone_python()
        if got is None:
            raise OSError("the downloaded Python is not where it was expected")
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
            raise OSError(
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
    def daemon_installed(self) -> dict[str, str]:
        """Package name to version, for what the daemon's own interpreter can import."""
        out: dict[str, str] = {}
        for dist in metadata.distributions():
            name = dist.metadata.get("Name") if dist.metadata else None
            if name:
                out[canonicalize_name(name)] = dist.version
        return out

    def installed(self) -> dict[str, str]:
        """Package versions available to the managed job interpreter."""
        if not self.exists:
            return {}
        try:
            out = self.pip(["inspect", "--local"], timeout=60)
        except (OSError, subprocess.SubprocessError):
            return {}
        if out.returncode != 0:
            return {}
        try:
            rows = json.loads(out.stdout)["installed"]
            self._cache["metadata"] = {canonicalize_name(row["metadata"]["name"]): row["metadata"] for row in rows}
            self._cache["direct_urls"] = {canonicalize_name(row["metadata"]["name"]): row.get("direct_url") for row in rows}
            return {canonicalize_name(row["metadata"]["name"]): row["metadata"]["version"] for row in rows}
        except (ValueError, KeyError, TypeError):
            return {}

    def has(self, library: Library) -> bool:
        have = self.installed()
        return _library_installed(library, have, self._cache)

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
                 "installed": _library_installed(lib, have, self._cache),
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
        have = self.installed()
        protected = {canonicalize_name(req.name)
                     for lib in CATALOG if lib.name not in names and _library_installed(lib, have, self._cache)
                     for spec in lib.packages for req in (_requirements(spec, self._cache.get("metadata")) or [])}
        for name in names:
            lib = wanted.get(name)
            if lib is None or not self.exists:
                done[name] = {"ok": False, "error": "not installed"}
                continue
            requirements = [req for spec in lib.packages for req in (_requirements(spec, self._cache.get("metadata")) or [])]
            packages = sorted({canonicalize_name(req.name) for req in requirements} - protected)
            if not packages:
                done[name] = {"ok": True, "kept_shared": True}
                continue
            out = self.pip(["uninstall", "-y", *packages])
            done[name] = ({"ok": True} if out.returncode == 0
                          else {"ok": False, "error": _last_error(out.stderr)})
        return done

    def remove(self) -> None:
        shutil.rmtree(self.path, ignore_errors=True)


def _base(spec: str) -> str:
    return canonicalize_name(Requirement(spec).name)


def _last_error(stderr: str) -> str:
    lines = [ln for ln in (stderr or "").splitlines() if ln.strip()]
    for line in reversed(lines):
        if "error" in line.lower():
            return line.strip()[:200]
    return (lines[-1][:200] if lines else "failed")


def _requirements(spec: str, installed_metadata=None) -> list[Requirement] | None:
    requirement = Requirement(spec)
    if not requirement.extras:
        return [requirement]
    if installed_metadata is not None:
        info = installed_metadata.get(canonicalize_name(requirement.name))
        if info is None:
            return None
        provided = set(info.get("provides_extra", []))
        requires = info.get("requires_dist", [])
    else:
        try:
            distribution = metadata.distribution(requirement.name)
        except metadata.PackageNotFoundError:
            return None
        provided = set(distribution.metadata.get_all("Provides-Extra", []))
        requires = distribution.requires or []
    if not requirement.extras <= provided:
        return None
    dependencies = [Requirement(text) for text in requires]
    return [requirement, *(dependency for dependency in dependencies
             if dependency.marker is None or any(dependency.marker.evaluate({"extra": extra})
                                                 for extra in requirement.extras))]


def _library_installed(library: Library, have: dict[str, str], cache=None) -> bool:
    cache = cache or {}
    for spec in library.packages:
        requirements = _requirements(spec, cache.get("metadata"))
        if requirements is None:
            return False
        for requirement in requirements:
            version = have.get(canonicalize_name(requirement.name))
            if version is None or version not in requirement.specifier:
                return False
            if requirement.url and not _direct_matches(requirement, cache.get("direct_urls", {})):
                return False
    return True


def _direct_matches(requirement: Requirement, urls) -> bool:
    direct = urls.get(canonicalize_name(requirement.name)) or {}
    expected = requirement.url or ""
    if expected.startswith("git+"):
        base, separator, revision = expected.removeprefix("git+").rpartition("@")
        if not separator:
            return False
        vcs = direct.get("vcs_info", {})
        return (direct.get("url") == base and vcs.get("vcs") == "git"
                and revision == vcs.get("commit_id"))
    return direct.get("url") == expected
