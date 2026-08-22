"""Find ``llama-server``, and give it an environment it can actually load in.

``shutil.which`` alone is not enough. The usual install locations reach PATH only through
a login shell's profile, so anything spawned from a subprocess, an editor or a TUI gets
"No such file or directory" for a binary that is plainly there in a terminal.
"""

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
) -> Path | None:
    """Locate a llama.cpp binary. ``None`` if it is nowhere to be found.

    Order, first hit wins:

    1. ``explicit`` -- a path from a config file or a CLI flag.
    2. ``$LLAMA_CPP_SERVER`` (for llama-server) or ``$LLAMA_CPP_DIR/<name>``.
    3. ``vendor_dir`` -- a self-contained build the caller owns.
    4. ``PATH``.
    5. ``ML_STACK_CACHE`` -- a previous auto-download.
    6. The login-shell directories PATH does not carry into a subprocess.

    An ``explicit`` path that does not exist falls through rather than hard-failing, so a
    stale config still boots off a vendored or cached copy.
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
        f"{name} not found. Looked at: $LLAMA_CPP_SERVER, $LLAMA_CPP_DIR, a vendor dir, "
        f"PATH, {CACHE_ROOT}, and {', '.join(str(d) for d in _LOGIN_SHELL_DIRS)}.\n"
        f"On macOS: brew install llama.cpp"
    )


def child_env(binary: Path | str, extra: dict[str, str] | None = None) -> dict[str, str]:
    """The environment to launch ``binary`` with, with its own directory on PATH.

    This is what makes a self-contained build work on Windows. An absolute-path launch of
    ``llama-server.exe`` does **not** find its sibling DLLs (``ggml*.dll``, ``llama*.dll``,
    cudart/cublas) unless their directory is on the loader search path; prepending the
    binary's own directory makes the bundled DLLs resolve.

    Harmless on Unix, and applied unconditionally so the two platforms do not diverge
    into two code paths that get tested separately and drift.
    """
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
