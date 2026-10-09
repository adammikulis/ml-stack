"""Validate macOS paths statically and other platforms with pytest's parser."""

import os
import sys
from pathlib import Path


def validate(arguments: list[str], root: Path) -> None:
    """Reject missing files before broker admission while preserving pytest selectors."""
    if sys.platform == "darwin" and os.environ.get("DEV_TEST_CONFINE") == "1":
        from test_kernel_isolation import check_selectors
        try:
            check_selectors([sys.executable, "-m", "pytest", *arguments])
        except RuntimeError as error:
            raise ValueError(str(error)) from None
        for selector in arguments:
            if selector.startswith("tests/"):
                target = root / selector.split("::", 1)[0]
                if not target.is_file():
                    raise ValueError(f"test path does not exist: {selector}")
        return
    import pytest
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
