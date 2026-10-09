"""Experimental features: named, staged, off until somebody turns one on.

A feature has a stage (``experimental``, ``beta`` or ``stable``), a one-line description and a
note on what it risks. A code gate asks ``enabled(NAME)``; ``ml-stack features enable NAME``
turns it on for this machine and ``disable`` off, each recorded in the authority audit log with
who did it. The state is the ``features`` map in the daemon's ``settings.json``, the file
`ml_stack.fleet.settings.Settings` already keeps; there is no second store.

The safety floors are not features and cannot become one: ``register`` refuses a name that says
it would loosen a push to main, an agent posing as a person, the keystore, or sharing the GPU.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

from ml_stack import authority, home
from ml_stack.files import read_json, write_json

__all__ = ["FEATURES", "FLOORS", "STAGES", "Feature", "UnknownFeature", "enabled", "listing", "register", "switch"]

STAGES = ("experimental", "beta", "stable")
NAME = re.compile(r"[a-z][a-z0-9]*(-[a-z0-9]+)*")
ACTOR_ENV = "ML_STACK_WORKSPACE_AGENT"

FLOORS: dict[str, tuple[tuple[str, ...], ...]] = {
    "no push to main": (("push", "main"), ("pushes", "main"), ("pushing", "main")),
    "an agent is never a human": (("agent", "human"), ("agent", "person"), ("agents", "human"),
                                  ("agents", "person"), ("human", "identity"), ("person", "identity")),
    "the keystore is a person's": (("keystore",), ("key", "store")),
    "one GPU at a time": tuple(("gpu", word) for word in (
        "share", "shared", "sharing", "concurrent", "parallel", "two", "multi", "lease", "bypass")),
}
"""Each floor with the word sets that identify a name aiming at it: all words of one set present."""


class UnknownFeature(KeyError):
    """A name nothing registered: a typo in a gate or on the command line, never a silent off."""


@dataclass(frozen=True, slots=True)
class Feature:
    """One registered feature."""

    name: str
    stage: str
    about: str
    risk: str


FEATURES: dict[str, Feature] = {}


def _aimed_at(name: str) -> str:
    """The floor ``name`` says it would loosen, or an empty string."""
    words = set(name.split("-"))
    for floor, shapes in FLOORS.items():
        if any(words.issuperset(shape) for shape in shapes):
            return floor
    return ""


def register(name: str, stage: str, about: str, risk: str) -> Feature:
    """Add a feature, off by default. Refuses a floor, a malformed or repeated name, an unknown stage."""
    if not NAME.fullmatch(name or ""):
        raise ValueError(f"a feature name is lowercase words joined by '-', not {name!r}")
    if floor := _aimed_at(name):
        raise ValueError(f"{name!r} would loosen a safety floor ({floor}); a floor is never a feature")
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {', '.join(STAGES)}, not {stage!r}")
    if not about.strip() or not risk.strip() or "\n" in about or "\n" in risk:
        raise ValueError("a feature needs a one-line description and a one-line risk note")
    if name in FEATURES:
        raise ValueError(f"feature {name!r} is already registered")
    FEATURES[name] = Feature(name, stage, about.strip(), risk.strip())
    return FEATURES[name]


def _known(name: str) -> Feature:
    try:
        return FEATURES[name]
    except KeyError:
        raise UnknownFeature(f"no feature named {name!r}; known: {', '.join(sorted(FEATURES))}") from None


def _settings_path(root: str = "") -> Path:
    """The daemon's ``settings.json``: ``root`` when one is named, else the usual daemon directory."""
    return home.expand(root) / "settings.json" if root else home.state("traind", "settings.json")


def _file(path: Path) -> dict:
    """What the settings file holds as an object; anything else, or a file nobody can parse, is empty."""
    try:
        held = read_json(path, {})
    except RecursionError:
        return {}
    return held if isinstance(held, dict) else {}


def _held(root: str = "") -> dict[str, bool]:
    one = _file(_settings_path(root)).get("features")
    return {k: v for k, v in one.items() if isinstance(v, bool)} if isinstance(one, dict) else {}


def enabled(name: str, root: str = "") -> bool:
    """Whether this machine has turned ``name`` on. Off for anything unreadable; an unregistered name raises."""
    _known(name)
    return _held(root).get(name) is True


def listing(root: str = "") -> list[dict[str, object]]:
    """Every feature with its stage, description, risk and whether it is on, in name order."""
    return [{"name": f.name, "stage": f.stage, "about": f.about, "risk": f.risk, "enabled": enabled(f.name, root)}
            for f in sorted(FEATURES.values(), key=lambda f: f.name)]


def switch(name: str, on: bool, *, root: str = "") -> bool:
    """Turn ``name`` on or off, write it down and audit it. Returns whether the state changed."""
    _known(name)
    path = _settings_path(root)
    held = _file(path)
    before = _held(root).get(name) is True
    write_json(path, {**held, "features": {**_held(root), name: on}})
    authority.record("features.set", by=os.environ.get(ACTOR_ENV) or authority.PERSON, feature=name,
                     stage=FEATURES[name].stage, changed=before != on, to=on)
    return before != on


register("test-runner-extras", "experimental",
         "The test runner's learned test order and its full-tier timing record, beyond plain affected tests.",
         "Reorders tests by past timings and writes a timing baseline into the checkout.")
register("guard-change", "experimental",
         "A person-approved change to the guards and the code that makes them (no gate is wired yet).",
         "Would let an approved diff alter the guard code itself; nothing reads it until it is wired.")
register("remote-tests", "experimental",
         "Run tests on other devices of the pool (scripts/test --on), and take their tests here when this device's node allows it.",
         "While a device allows it, any member of its pool runs a checkout's tests there as its user: a stolen pool certificate is code execution.")
