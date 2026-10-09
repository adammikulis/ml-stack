"""The built wheel, made once for every test process and run that shares a checkout state.

Building the wheel and installing it into a target directory took 19 to 41 s the first time a
worker reached a test that needs real distribution metadata (see ``artifact_cache``).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from artifact_cache import cached, expand

ROOT = Path(__file__).resolve().parents[1]
TREES = ("src/ml_stack", "contracts", "patches")
FILES = ("pyproject.toml", "README.md", "LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md")


def _build(into: Path) -> None:
    environment = {**os.environ, "PIP_NO_INDEX": "1", "PIP_DISABLE_PIP_VERSION_CHECK": "1"}
    subprocess.run([sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation",
                    "--no-cache-dir", "--wheel-dir", str(into), str(ROOT)],
                   env=environment, check=True, capture_output=True, text=True, timeout=180)
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "--no-index",
                    "--target", str(into / "installed"), str(next(into.glob("*.whl")))],
                   env=environment, check=True, capture_output=True, text=True, timeout=180)


def built() -> Path:
    """The directory holding the wheel (``*.whl``) and, under ``installed/``, its contents
    installed with pip. Built on first use for these sources, found afterwards."""
    return cached("wheel", expand(*(ROOT / name for name in (*FILES, *TREES))), _build)
