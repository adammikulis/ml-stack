"""What a boot-time (`--system`) install says about the keystore.

A service started at boot is a background process: it never creates the master key and reads it only
after a person ran `ml-stack-security unlock`. Without one it runs, but every sealed store stays locked
and nothing is written in the clear, so the install warns and goes on rather than refusing: the machine
is safe, only less useful, until somebody unlocks.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from ml_stack import home
from ml_stack.files import read_json
from ml_stack.keystore import UNLOCK_COMMAND

__all__ = ["notice"]

LOCKED = ("sealed stores (memory, the request inbox, the activity log, the reputation ledger, wrapped "
          "credentials) stay locked and the fleet signing key cannot be unwrapped; the service fails "
          "closed and writes nothing in the clear")


PLAIN = ("Credentials kept in the plain credentials file (`ml-stack credentials set` without `--keychain`) "
         "are not sealed and stay readable either way.")


def notice(user: str, home_dir: Path | str, platform: str = "") -> str:
    """A warning for a boot service whose user has no unlocked keystore, else an empty string.

    Reads the state file under the service's state root (``ML_STACK_HOME`` when the installing
    environment names one, else ``<home>/.ml-stack``), not a keystore backend, so it works under sudo
    and touches no prompt.
    """
    root = home.home() if os.environ.get(home.ROOT_ENV) else home.expand(home_dir) / home.DEFAULT_NAME
    held = read_json(root / "keystore" / "provisioned.json", {})
    if not (held.get("at") if isinstance(held, dict) else None):
        return (f"{user} has no ml-stack keystore key yet. At boot {LOCKED}. Run `{UNLOCK_COMMAND}` once "
                f"as {user} in a terminal, then restart the service. {PLAIN}")
    if (platform or sys.platform) == "darwin":
        return (f"On a Mac the login keychain is locked until {user} logs in, so after a reboot with nobody "
                f"logged in {LOCKED}, until the first login and a service restart.")
    return ""
