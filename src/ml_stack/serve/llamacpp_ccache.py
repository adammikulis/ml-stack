"""ccache for the llama.cpp source build: neighbouring upstream commits share most translation
units, so a compile cache turns an update across many commits from minutes into seconds.

The cache lives in one project-owned directory (``<llama.cpp state>/ccache``), is capped at
`MAX_SIZE` and is the one extra directory the build's sandbox may write. Without ccache on this
machine the build is exactly what it was, and says so in one line."""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path

from ml_stack import home, platform as host

__all__ = ["ABSENT", "LANGUAGES", "MAX_SIZE", "Ccache", "find"]

MAX_SIZE = "5G"
"""The cache's size limit (``CCACHE_MAXSIZE``); ccache evicts the least recently used past it."""
LANGUAGES = ("C", "CXX", "OBJC", "OBJCXX")
"""The languages that get a ``CMAKE_<lang>_COMPILER_LAUNCHER``; one a build never enables is ignored."""
ABSENT = "ccache not found: building without a compile cache (install ccache to speed up updates)"
SECONDS = 60.0
"""The longest any one ccache or loader-listing process may run before it is killed."""
OUTPUT_BYTES = 1_000_000
"""The most output read from one of them; the process is killed past it."""
FIELDS = ("direct_cache_hit", "preprocessed_cache_hit", "cache_miss", "uncacheable")


def cache_dir() -> Path:
    """The cache directory, ``<ml-stack home>/llama.cpp/ccache``."""
    return home.state("llama.cpp", "ccache")


def _spawn(argv: list[str], env: dict[str, str] | None = None) -> tuple[int, str]:
    """Run ``argv`` (never through a shell) for at most `SECONDS`, reading at most `OUTPUT_BYTES`
    of its output; (-1, "") when it cannot start, hangs, or floods. Its group is killed."""
    try:
        proc = host.start_process(argv, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL)
    except (OSError, ValueError):
        return -1, ""

    def stop() -> None:
        try:
            host.terminate_process_group(proc, force=True)
        except OSError:
            proc.kill()

    timer = threading.Timer(SECONDS, stop)
    timer.start()
    try:
        data = proc.stdout.read(OUTPUT_BYTES + 1) if proc.stdout else b""
        flooded = len(data) > OUTPUT_BYTES
        if flooded:
            stop()
        code = proc.wait()
    finally:
        timer.cancel()
        stop()
        if proc.stdout:
            proc.stdout.close()
    expired = code < 0
    return (-1, "") if flooded or expired else (code, data.decode("utf-8", errors="replace"))


def _libraries(exe: str) -> list[str]:
    """The shared libraries ``exe`` loads from outside the system directories."""
    query = ["otool", "-L", exe] if platform.system() == "Darwin" else ["ldd", exe]
    _, out = _spawn(query)
    paths = re.findall(r"(?:=>\s+|^\s+)(/\S+)", out, re.M)
    return [p for p in paths if not p.startswith(("/usr/lib", "/System", "/lib"))]


def _tree(path: str) -> str:
    """The install prefix of a program or library: the keg under a ``Cellar``, else its directory."""
    real = Path(os.path.realpath(path))
    parts = real.parts
    if "Cellar" in parts and len(parts) > parts.index("Cellar") + 2:
        return str(Path(*parts[:parts.index("Cellar") + 3]))
    return str(real.parent.parent if real.parent.name == "bin" else real.parent)


def _link_parent(path: str) -> str:
    """The directory holding the first symlink on ``path`` ("" when there is none): the loader
    looks up a library by its install name, and Homebrew's is ``/opt/homebrew/opt/<name>/...``,
    a link into the keg, which the policy cannot grant itself but may read from its directory."""
    walked = Path("/")
    for part in Path(path).parts[1:]:
        if (walked / part).is_symlink():
            return str(walked)
        walked = walked / part
    return ""


@dataclass(frozen=True, slots=True)
class Ccache:
    """A ccache to build with: its real path, its cache directory and the trees it reads."""

    exe: str
    directory: Path
    reads: tuple[str, ...]

    def arguments(self) -> list[str]:
        """The cmake flags that put ccache in front of every compiler."""
        return [f"-DCMAKE_{lang}_COMPILER_LAUNCHER={self.exe}" for lang in LANGUAGES]

    def environment(self, basedir: Path) -> dict[str, str]:
        """What ccache is told: where the cache is, how big, and ``basedir`` so paths under it
        are hashed relative to the working directory and one commit's objects hit in another's
        work directory; the working directory itself is not hashed."""
        return {"CCACHE_DIR": str(self.directory), "CCACHE_MAXSIZE": MAX_SIZE,
                "CCACHE_BASEDIR": str(basedir), "CCACHE_NOHASHDIR": "1"}

    def cache(self) -> str:
        """The one directory outside the work directory the sandbox lets the build write: the
        policy's ``cache`` grant."""
        return str(self.directory)

    def reset(self, basedir: Path) -> None:
        """Zero the counters, so the numbers at the end are this build's."""
        self._run(basedir, "-z")

    def summary(self, basedir: Path) -> str:
        """One line of this build's hits, misses and the cache size limit."""
        stats = self._run(basedir, "--print-stats")
        if not stats.strip():
            return "ccache: statistics unavailable"
        counts = dict.fromkeys(FIELDS, 0)
        for line in stats.splitlines():
            key, _, value = line.partition("\t")
            if key in counts and value.strip().isdigit():
                counts[key] = int(value)
        hits = counts["direct_cache_hit"] + counts["preprocessed_cache_hit"]
        total = hits + counts["cache_miss"]
        rate = f"{100 * hits // total}%" if total else "n/a"
        return (f"ccache: {hits} hits, {counts['cache_miss']} misses ({rate} hit rate), "
                f"{counts['uncacheable']} uncacheable; cache {self.directory} (limit {MAX_SIZE})")

    def _run(self, basedir: Path, *args: str) -> str:
        code, out = _spawn([self.exe, *args], {**os.environ, **self.environment(basedir)})
        return out if code == 0 else ""


def find() -> Ccache | None:
    """The ccache on this machine with its cache directory made, or None when there is none."""
    found = shutil.which("ccache")
    if not found:
        return None
    exe = os.path.realpath(found)
    if _spawn([exe, "--version"])[0] != 0:
        return None
    directory = cache_dir()
    directory.mkdir(parents=True, exist_ok=True)
    libs = _libraries(exe)
    trees = {_tree(exe), *(_tree(lib) for lib in libs), *(_link_parent(lib) for lib in libs)} - {""}
    return Ccache(exe, Path(os.path.realpath(directory)), tuple(sorted(trees)))
