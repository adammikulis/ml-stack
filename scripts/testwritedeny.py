"""Deny the pytest process tree every write under the real state root.

The after-the-fact check in `tests/conftest.py` (`_real_home`) fails a run once the real
`~/.ml-stack` has changed. This stops the write when it is made: the pytest tree runs under a
kernel rule that refuses it, so the test that reached for the real root gets an `EPERM` at the
line that did it.

macOS: `sandbox-exec` with an allow-default profile that denies `file-write*` under the root.
Linux: `bwrap` with the root bound read-only, used only when a probe shows it works and that a
nested `bwrap` (the sandbox tests) still works inside it. Elsewhere, and whenever the probe
fails, nothing is wrapped and only the after-the-fact check applies.

A process that is already sandboxed cannot apply a second `sandbox-exec` profile (checked: the
call fails with EPERM unless the profiles are identical), so the tests that run `sandbox-exec`
themselves carry the `seatbelt` marker (`tests/seatbelt-modules.txt`). `passes` splits a run in
two: everything else under the denial, then those modules without it and with the
after-the-fact check alone.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ml_stack import home

ENV = "DEV_TEST_WRITE_DENY"
"""``0`` turns the denial off for a run; ``1`` marks a process already under it."""
SANDBOX_EXEC = "/usr/bin/sandbox-exec"
MARKER = "seatbelt"


def protected(repo: Path) -> list[Path]:
    """The real state roots a test must not write: the account's own and the one the launching
    environment names, minus any that would contain the checkout or the temporary directory."""
    roots = {home.account_roots()[0].resolve(), home.home().resolve()}
    fenced = (repo.resolve(), Path(tempfile.gettempdir()).resolve(), Path.home().resolve())
    return sorted(root for root in roots if not any(root == keep or root in keep.parents for keep in fenced))


def quote(path: str) -> str:
    """``path`` as a Seatbelt string literal."""
    if any(ord(char) < 32 or ord(char) == 127 for char in path):
        raise ValueError(f"{path!r} holds a control character")
    return '"' + path.replace("\\", "\\\\").replace('"', '\\"') + '"'


def profile(roots: list[Path]) -> str:
    """The Seatbelt profile: everything allowed except writing under ``roots``."""
    denied = " ".join(f"(subpath {quote(str(root))})" for root in roots)
    return f"(version 1)(allow default)(deny file-write* {denied})"


def bubblewrap(roots: list[Path], command: list[str]) -> list[str]:
    """``command`` under bwrap with ``roots`` bound read-only over a bind of the whole filesystem."""
    binds = [arg for root in roots for arg in ("--ro-bind", str(root), str(root))]
    return ["bwrap", "--bind", "/", "/", "--dev-bind", "/dev", "/dev", *binds, "--", *command]


def wrapper(roots: list[Path], command: list[str]) -> list[str] | None:
    """``command`` under the denial on this platform, or None when there is no way to apply it."""
    if not roots:
        return None
    if sys.platform == "darwin" and os.access(SANDBOX_EXEC, os.X_OK):
        return [SANDBOX_EXEC, "-p", profile(roots), *command]
    if sys.platform.startswith("linux") and shutil.which("bwrap"):
        return bubblewrap(roots, command)
    return None


def works() -> bool:
    """Whether the denial can be applied here, probed on a throwaway directory (never the real
    root): the wrapped probe cannot write there, and on Linux a nested bwrap still starts."""
    with tempfile.TemporaryDirectory(prefix="write-deny-probe-") as scratch:
        root = Path(scratch).resolve()
        refused = ("import sys\ntry:\n open(sys.argv[1], 'w')\nexcept OSError:\n sys.exit(0)\nsys.exit(7)")
        probe = wrapper([root], [sys.executable, "-c", refused, str(root / "probe")])
        if probe is None:
            return False
        done = subprocess.run(probe, capture_output=True, check=False, timeout=60)
        if done.returncode != 0 or (root / "probe").exists():
            return False
        if sys.platform.startswith("linux"):  # the sandbox tests start bwrap themselves, inside this one
            nested = wrapper([root], ["bwrap", "--ro-bind", "/", "/", "true"])
            return nested is not None and subprocess.run(nested, capture_output=True, check=False, timeout=60).returncode == 0
        return True


@dataclass(frozen=True)
class Pass:
    """One pytest invocation of a split run: the command, and whether it runs under the denial."""

    command: list[str]
    denied: bool


def select(command: list[str], marked: bool) -> list[str]:
    """``command`` narrowed to the tests with the marker (``marked``) or without it, by `-m`."""
    out = list(command)
    clause = MARKER if marked else f"not {MARKER}"
    first = out.index("pytest") + 1
    if "-m" in out[first:]:
        at = first + out[first:].index("-m") + 1
        out[at] = f"({out[at]}) and {clause}"
    else:
        out[first:first] = ["-m", clause]
    return out


def listed(repo: Path) -> frozenset[str]:
    """The test modules in ``tests/seatbelt-modules.txt`` that run `sandbox-exec` themselves."""
    try:
        text = (repo / "tests" / "seatbelt-modules.txt").read_text(encoding="utf-8")
    except OSError:
        return frozenset()
    return frozenset(line.split("#")[0].strip() for line in text.splitlines() if line.split("#")[0].strip())


def named(command: list[str]) -> set[str]:
    """The test file names a command selects by path or node id; empty when it selects a directory or all."""
    out = set()
    for arg in command[command.index("pytest") + 1:]:
        if arg.startswith("tests/") and ".py" in arg:
            out.add(Path(arg.split("::")[0]).name)
    return out


def passes(command: list[str], environment: dict[str, str], repo: Path) -> list[Pass] | None:
    """The passes of a run under the denial, or None to run it as given.

    None when the run opted out (``DEV_TEST_WRITE_DENY=0``), is already inside a denial, is not a
    plain ``python -m pytest`` command, or the platform cannot apply it. Two passes when the
    selection reaches both ordinary tests and ``seatbelt`` modules, else the one that is needed.
    """
    if environment.get(ENV) in ("0", "1") or "pytest" not in command:
        return None
    roots = protected(repo)
    if not roots or not works():
        return None
    nested, picked = listed(repo), named(command)
    if not nested:
        return [Pass(list(command), True)]
    result = []
    if not picked or picked - nested:
        result.append(Pass(select(command, False), True))
    if not picked or picked & nested:
        result.append(Pass(select(command, True), False))
    return result


def merge_junit(into: Path, other: Path) -> None:
    """Fold the testcases and counts of the JUnit file ``other`` into ``into``."""
    from xml.etree import ElementTree

    first, second = ElementTree.parse(into), ElementTree.parse(other)  # noqa: S314 - files pytest just wrote
    head = first.getroot().find("testsuite") if first.getroot().tag == "testsuites" else first.getroot()
    tail = second.getroot().find("testsuite") if second.getroot().tag == "testsuites" else second.getroot()
    if head is None or tail is None:
        return
    for name in ("tests", "failures", "errors", "skipped"):
        head.set(name, str(int(head.get(name, "0")) + int(tail.get(name, "0"))))
    head.set("time", f"{float(head.get('time', '0')) + float(tail.get('time', '0')):.3f}")
    head.extend(list(tail))
    first.write(into, encoding="utf-8", xml_declaration=True)


def combined(statuses: list[int]) -> int:
    """One exit status for several passes: the first real failure, else 0 if any ran a test, else 5."""
    for status in statuses:
        if status not in (0, 5):
            return status
    return 0 if 0 in statuses else 5
