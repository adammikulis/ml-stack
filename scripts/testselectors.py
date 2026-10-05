"""Validate explicit test paths with pytest's maintained argument parser."""

from pathlib import Path

import pytest


def validate(arguments: list[str], root: Path) -> None:
    """Reject missing files before broker admission while preserving pytest selectors."""
    config = pytest.Config.fromdictargs({}, arguments)
    try:
        for selector in config.args:
            path = Path(selector.split('::', 1)[0])
            target = path if path.is_absolute() else root / path
            if not target.exists():
                raise ValueError(f"test path does not exist: {path}")
            if not (target.is_file() or target.is_dir()):
                raise ValueError(f"test path is not a file or directory: {path}")
    finally:
        config._ensure_unconfigure()
