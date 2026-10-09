"""Turn test shards on or off for this device's daemon: ``python -m ml_stack.fleet.shard_consent on|off|status [ROOT]``.

Run by the person at this device. The daemon reads the setting at each request, so no restart is needed.
"""

from __future__ import annotations

import sys

from ml_stack import home
from ml_stack.log import say

from .runtime_paths import default_root
from .settings import Settings

ACTIONS = ("on", "off", "status")


def switch(action: str, root: str = "") -> bool:
    """Apply ``action`` to the settings under ``root`` (default: the daemon's usual one); whether shards are on."""
    path = home.expand(root or str(default_root())) / "settings.json"
    settings = Settings.load(path)
    if action != "status":
        settings.test_shards = action == "on"
        settings.save(path)
    return settings.test_shards


def run(argv: list[str]) -> int:
    """``ACTION [ROOT]``: set or show the setting and say what it is; 2 for a usage error."""
    if not argv or argv[0] not in ACTIONS or len(argv) > 2:
        say(f"usage: python -m ml_stack.fleet.shard_consent {'|'.join(ACTIONS)} [ROOT]")
        return 2
    say(f"test shards: {'on' if switch(argv[0], *argv[1:]) else 'off'}")
    return 0


if __name__ == "__main__":
    sys.exit(run(sys.argv[1:]))
