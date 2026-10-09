"""What the project kept under its old name, ml-stack, moved once to the new one.

The old names of the state directory, the cache directory, the keychain entries and the
project file. Only `poolhouse migrate` (an explicit step, run once at the cutover) moves
anything under them, and `poolhouse.sentinel.moves` maps a path recorded under the old state
root. The per-feature keychain items an even older version left are migrated into the master
key under the names they have.
"""

from __future__ import annotations

__all__ = ["CACHE_DIR", "KEYCHAIN_SERVICE", "MEMORY_SERVICE", "PROJECT_FILE", "STATE_DIR"]

STATE_DIR = ".ml-stack"
"""The old state directory name, under the account's home."""
CACHE_DIR = "ml_stack"
"""The old cache directory name, under ``~/.cache``."""
KEYCHAIN_SERVICE = "ml-stack"
"""The old keychain service: the master key, and the signing keys older versions kept beside it."""
MEMORY_SERVICE = "ml-stack-memory"
"""The old keychain service of the memory vault's random key."""
PROJECT_FILE = ".ml-stack-project.json"
"""The old name of the file a project checkout keeps about the board it belongs to."""
