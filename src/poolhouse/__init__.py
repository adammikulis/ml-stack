"""Train and run models across every machine on your network."""

from __future__ import annotations

import logging
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("poolhouse")
except PackageNotFoundError:
    __version__ = "0+unknown"

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = ["__version__"]
