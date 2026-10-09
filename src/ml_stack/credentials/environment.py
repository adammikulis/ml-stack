"""The environment a child process gets: this process's, less anything that looks like a secret."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping

SECRET_NAME = re.compile(r"TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|API_?KEY|ACCESS_?KEY|"
                         r"PRIVATE_?KEY|AUTH|COOKIE|SESSION_?ID", re.IGNORECASE)
KEPT = frozenset({"SSH_AUTH_SOCK", "XAUTHORITY", "GPG_AGENT_INFO"})


def child_environment(extra: Mapping[str, str] | None = None, *,
                      source: Mapping[str, str] | None = None) -> dict[str, str]:
    """``source`` (default: this process's environment) without any variable whose name
    looks like a token, key, password or credential, with ``extra`` added.

    A child that needs one credential is handed exactly that one, in ``extra``."""
    env = {name: value for name, value in (os.environ if source is None else source).items()
           if name in KEPT or not SECRET_NAME.search(name)}
    env.update(extra or {})
    return env
