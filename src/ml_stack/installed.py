"""The extras a full install has, which of them this one holds, and at what versions."""

from __future__ import annotations

import importlib.util
import platform
import re
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, requires
from importlib.metadata import version as installed_version

from ml_stack.log import say

__all__ = ["STANDARD", "STORE_MACOS_FLOOR", "Capability", "behind", "declared", "extras",
           "missing", "report", "standard", "store_wants_newer_macos", "unmet"]

_NAMED = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*(?:\[[^\]]*\])?\s*(.*)$")
_FLOOR = re.compile(r">=\s*([0-9][0-9A-Za-z.]*)")
_FOR_EXTRA = re.compile(r"""extra\s*==\s*['"]([^'"]+)['"]""")

STORE_MACOS_FLOOR = 15
"""The oldest macOS the graph store publishes a wheel for."""


@dataclass(frozen=True, slots=True)
class Capability:
    """One extra: what it is called, what it lets this machine do, and the import that
    proves it is here."""

    extra: str
    name: str
    does: str
    module: str

    @property
    def fix(self) -> str:
        return f"pip install 'ml-stack[{self.extra}]'"

    def present(self) -> bool:
        """Whether the module behind this extra can be imported on this machine."""
        try:
            return importlib.util.find_spec(self.module) is not None
        except (ImportError, ValueError):
            return False


STANDARD: tuple[Capability, ...] = (
    Capability("store", "the graph store",
               "keeps a graph, and the bench's runs, in one file on disk", "ladybug"),
    Capability("hub", "model downloads",
               "fetches a model by name into the one cache on this machine",
               "huggingface_hub"),
    Capability("web", "reading the web",
               "searches the web and reads a page into text", "ddgs"),
    Capability("plot", "plots", "draws the panels ml-stack-serve fit --plot writes",
               "matplotlib"),
    Capability("graph", "graph maths", "builds a graph out of what it has read", "numpy"),
)


def extras() -> str:
    """The extras a full install asks pip for, as pip is given them."""
    return ",".join(c.extra for c in STANDARD)


def standard() -> list[tuple[Capability, bool]]:
    """Every extra a full install has, with whether this machine holds it."""
    return [(c, c.present()) for c in STANDARD]


def missing() -> list[Capability]:
    """The extras a full install has and this one does not."""
    return [c for c, here in standard() if not here]


def _numbers(version: str) -> tuple[int, ...]:
    """The leading numeric parts of a version, so 1.2.3rc1 reads as (1, 2, 3)."""
    out = []
    for part in version.split("."):
        digits = re.match(r"\d+", part)
        if not digits:
            break
        out.append(int(digits.group()))
    return tuple(out)


def _under(have: str, floor: str) -> bool:
    mine, wanted = _numbers(have), _numbers(floor)
    return mine + (0,) * (len(wanted) - len(mine)) < wanted


def declared(distribution: str = "ml-stack") -> dict[str, list[str]]:
    """The extras of an installed distribution, read from its own metadata.

    An installed wheel carries no pyproject.toml, so this is where a real machine reads
    its pins.
    """
    out: dict[str, list[str]] = {}
    for line in requires(distribution) or []:
        requirement, _, marker = line.partition(";")
        named = _FOR_EXTRA.search(marker)
        if named:
            out.setdefault(named.group(1), []).append(requirement.strip())
    return out


def unmet(extras_of: dict[str, list[str]]) -> list[tuple[str, str, str, str]]:
    """``(extra, package, installed version, requirement)`` for each floor not met.

    A package nothing installed is not a finding, and a requirement naming another extra
    is followed rather than looked up.
    """
    out = []
    for extra, requirements in extras_of.items():
        for requirement in requirements:
            named = _NAMED.match(requirement)
            if not named or named.group(1) == "ml-stack":
                continue
            floor = _FLOOR.search(named.group(2))
            if not floor:
                continue
            try:
                have = installed_version(named.group(1))
            except PackageNotFoundError:
                continue
            if _under(have, floor.group(1)):
                out.append((extra, named.group(1), have, requirement))
    return out


def behind() -> list[tuple[str, str, str, str]]:
    """Every package this machine holds below the version an extra asks for."""
    return unmet(declared())


def store_wants_newer_macos() -> str:
    """Why the store extra cannot install here, or empty when it can.

    ladybug tags its macOS wheels macosx_15_0, so on macOS 14 and older pip falls back to
    the sdist and compiles against a libc++ with no std::atomic_ref.
    """
    if platform.system() != "Darwin":
        return ""
    release = _numbers(platform.mac_ver()[0])
    if not release or release[0] >= STORE_MACOS_FLOOR:
        return ""
    return (f"the graph store needs macOS {STORE_MACOS_FLOOR} or newer; this is "
            f"{platform.mac_ver()[0]}, where pip has no wheel to install and builds from "
            f"source instead")


def report() -> int:
    """Print what this install cannot do, and the line that fixes each. 0 when it can do
    everything."""
    gone = missing()
    for one in gone:
        say(f"  ! {one.name}: not installed -- {one.does}")
        say(f"    fix: {one.fix}")
    old = behind()
    for _, name, have, wanted in old:
        say(f"  ! {name} {have} is installed, and {wanted} is asked for")
    if old:
        say("    fix: pip install -U " + " ".join(sorted({name for _, name, _, _ in old})))
    floor = store_wants_newer_macos() if any(c.extra == "store" for c in gone) else ""
    if floor:
        say(f"    {floor}")
    if not gone and not old:
        say("  every part of a full install is here")
    return 1 if gone or old else 0


if __name__ == "__main__":
    raise SystemExit(report())
