"""Install into a virtual environment with uv's shared cache: cloned or hard-linked files, no bytecode written."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from ml_stack.home import user_home
from ml_stack.net import packages

REDIRECTS = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")
KEPT = {"PIP_CONFIG_FILE", "PIP_DISABLE_PIP_VERSION_CHECK"}


def program(path: str | None = None) -> str | None:
    """The uv executable on the search path or in the pyenv shims, or None."""
    return shutil.which("uv", path=path) or shutil.which("uv", path=str(user_home() / ".pyenv" / "shims"))


def link_mode() -> str:
    """Clone on macOS (copy-on-write, nothing written), hard links elsewhere."""
    return "clone" if sys.platform == "darwin" else "hardlink"


def _plain(environment: dict[str, str]) -> bool:
    """Whether nothing in the environment redirects package sources or the network; only the pip path admits those."""
    return not any(name.startswith(("UV_", "PIP_")) and name not in KEPT for name in environment) \
        and not any(environment.get(name) for name in REDIRECTS)


def install(python: Path, spec: str, *, timeout: float, environment: dict[str, str]) -> subprocess.CompletedProcess | None:
    """`uv pip install` the spec into the interpreter's environment; None when uv is missing or the environment redirects sources."""
    tool = program(environment.get("PATH"))
    if tool is None or not _plain(environment):
        return None
    url = spec.partition(" @ ")[2]
    if url:
        packages._admit(url, packages.policy.default())
    clean = {name: value for name, value in environment.items() if not name.startswith(("UV_", "PIP_"))}
    return subprocess.run([tool, "pip", "install", "--no-config", "--link-mode", link_mode(),
                           "--python", str(python), spec], capture_output=True, text=True, timeout=timeout, env=clean)
