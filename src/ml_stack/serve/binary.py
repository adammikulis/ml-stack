"""Find ``llama-server``, and give it an environment it can actually load in."""

from __future__ import annotations

import logging
import os
import platform
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

SERVER_NAMES = ("llama-server", "llama-server.exe")

CACHE_ROOT = Path(
    os.environ.get("ML_STACK_CACHE", Path.home() / ".cache" / "ml_stack")
).expanduser()

# Where `ml-stack-serve build` installs what it builds or downloads, and which build is
# trusted right now. `build.py` only repoints MANAGED_CURRENT once a new build answers
# --help and reads every architecture the old one did -- so finding it here is finding
# something already verified, never a build in progress.
MANAGED_ROOT = Path.home() / ".ml-stack" / "llama.cpp"
MANAGED_CURRENT = MANAGED_ROOT / "current"

# A build kept beside `current` rather than replacing it -- a fork whose fixes have not
# reached master, selected by name instead of becoming the default. `build.py --name NAME`
# points `MANAGED_NAMED / NAME` at it once it is verified.
MANAGED_NAMED = MANAGED_ROOT / "named"

# Directories that are on PATH only in a login shell, so a subprocess never sees them.
_LOGIN_SHELL_DIRS = (
    Path.home() / "bin",
    Path.home() / ".local" / "bin",
    Path("/opt/homebrew/bin"),
    Path("/usr/local/bin"),
)


class BinaryNotFound(RuntimeError):
    """No ``llama-server`` could be located, and none could be fetched."""


def is_windows() -> bool:
    return platform.system() == "Windows"


def find_binary(
    name: str = "llama-server",
    *,
    explicit: str | Path | None = None,
    vendor_dir: Path | None = None,
    build: str | None = None,
) -> Path | None:
    """Locate a llama.cpp binary. ``None`` if it is nowhere to be found.

    ``build`` (or, absent that, ``$MLSTACK_LLAMA_BUILD``) names a build ``ml-stack-serve
    build --name NAME`` made and kept beside ``current`` rather than replacing it -- a fork
    whose fixes have not reached master. It outranks ``current`` but never an explicit path
    or ``$LLAMA_CPP_SERVER``, so a caller that names a build gets it even while ``current``
    stays mainline.
    """
    candidates = _name_variants(name)

    if explicit:
        path = Path(explicit).expanduser()
        if path.is_file():
            return path.resolve()
        logger.debug("explicit binary %s does not exist; falling through", path)

    env_key = "LLAMA_CPP_SERVER" if name == "llama-server" else None
    if env_key and (value := os.environ.get(env_key)):
        path = Path(value).expanduser()
        if path.is_file():
            return path.resolve()

    if env_dir := os.environ.get("LLAMA_CPP_DIR"):
        for candidate in candidates:
            path = Path(env_dir).expanduser() / candidate
            if path.is_file():
                return path.resolve()

    if name == "llama-server":
        named = build or os.environ.get("MLSTACK_LLAMA_BUILD")
        if named:
            for candidate in candidates:
                path = MANAGED_NAMED / named / candidate
                if path.is_file():
                    return path.resolve()
            logger.debug("named build %r has no %s; falling through", named, name)

        # A verified `ml-stack-serve build` outranks a login shell's PATH and the stale
        # bottle a release lags behind -- but never an explicit path or $LLAMA_CPP_SERVER,
        # both handled above.
        for candidate in candidates:
            path = MANAGED_CURRENT / candidate
            if path.is_file():
                return path.resolve()

    for directory in (vendor_dir, CACHE_ROOT, *_LOGIN_SHELL_DIRS):
        if directory is None:
            continue
        for candidate in candidates:
            path = Path(directory).expanduser() / candidate
            if path.is_file():
                return path.resolve()

    if found := shutil.which(name):
        return Path(found).resolve()

    return None


def require_binary(name: str = "llama-server", **kwargs: object) -> Path:
    """``find_binary`` or raise with the search path spelled out."""
    found = find_binary(name, **kwargs)  # type: ignore[arg-type]
    if found is not None:
        return found
    raise BinaryNotFound(
        f"{name} not found. Looked at: $LLAMA_CPP_SERVER, $LLAMA_CPP_DIR, a named build "
        f"($MLSTACK_LLAMA_BUILD or build=) under {MANAGED_NAMED}, {MANAGED_CURRENT}, "
        f"a vendor dir, PATH, {CACHE_ROOT}, and "
        f"{', '.join(str(d) for d in _LOGIN_SHELL_DIRS)}.\n"
        f"ml-stack-serve build   builds llama.cpp's own master (or downloads the newest "
        f"release, on a machine with no compiler) -- usually what you want, since a "
        f"release lags master by an architecture or two.\n"
        f"On macOS: brew install llama.cpp"
    )


def child_env(binary: Path | str, extra: dict[str, str] | None = None) -> dict[str, str]:
    """The environment to launch ``binary`` with, with its own directory on PATH."""
    env = dict(os.environ)
    bindir = str(Path(binary).resolve().parent)
    env["PATH"] = bindir + os.pathsep + env.get("PATH", "")
    if extra:
        env.update(extra)
    return env


def _name_variants(name: str) -> tuple[str, ...]:
    if name.endswith(".exe"):
        return (name,)
    return (name, f"{name}.exe") if is_windows() else (name,)
