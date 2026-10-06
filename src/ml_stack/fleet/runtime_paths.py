"""State roots and startup output for a Fleet daemon process."""

from __future__ import annotations

import contextlib
import os
from pathlib import Path

from ml_stack import home, person
from ml_stack.log import say


def configure(root: Path) -> None:
    """Route custom daemon installations to their own process state and cache."""
    if os.environ.get(home.ROOT_ENV) or root.resolve() == home.state("traind").resolve():
        return
    state = root.resolve() / "state"
    os.environ[home.ROOT_ENV] = str(state)
    if not os.environ.get(home.CACHE_ENV):
        os.environ[home.CACHE_ENV] = str(state / "cache")


def announce_token(token: str) -> None:
    """Display the daemon token to a person at an interactive terminal."""
    with contextlib.suppress(person.HumanRequired):
        person.require_person("display the Fleet token")
        say(f"  token {token}")
