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
"""Input: ``0`` turns the denial off for a run. Output: the supervisor sets ``1`` in a child only after
it wrapped that child, so a value a user sets is never proof (`denied_here` checks by effect)."""
WRAP_ENV = "DEV_TEST_WRITE_DENY_WRAP"
"""Set by `run_pytest` on a pass that is to be wrapped."""
SANDBOX_EXEC = "/usr/bin/sandbox-exec"
MARKER = "seatbelt"


def say(text: str) -> None:
    """A notice on stderr, where a run's other supervisor lines go."""
    print(f"test: {text}", file=sys.stderr, flush=True)


def candidates() -> list[Path]:
    """Every real path a test must not write: the account's state and cache roots, the roots the
    launching environment names (``ML_STACK_HOME``, ``ML_STACK_CACHE`` and each ``home.OVERRIDES``
    variable that is set)."""
    state, cache = home.account_roots()
    named = [home.home(), home.cache(), *(home.expand(os.environ[v]) for v in home.OVERRIDES.values() if os.environ.get(v))]
    return sorted({path.resolve() for path in (state, cache, *named)})


def protected(repo: Path) -> list[Path]:
    """`candidates` minus any that would contain the checkout, the temporary directory or HOME,
    which would make the run unable to work; each one dropped is said out loud."""
    fenced = {"the checkout": repo.resolve(), "the temporary directory": Path(tempfile.gettempdir()).resolve(),
              "HOME": Path.home().resolve()}
    kept = []
    for root in candidates():
        holds = [name for name, keep in fenced.items() if root == keep or root in keep.parents]
        if holds:
            say(f"write denial: {root} is not denied because it holds {' and '.join(holds)}")
        else:
            kept.append(root)
    return kept


def probe_file(root: Path) -> Path | None:
    """An existing regular file under ``root`` outside the keystore, to open for writing without
    changing it, or None."""
    for here, dirs, names in os.walk(root):
        dirs[:] = [d for d in dirs if d != "keystore"][:8]
        for name in names:
            path = Path(here, name)
            if path.is_file() and not path.is_symlink():
                return path
    return None


def denied_here(root: Path) -> bool | None:
    """Whether this process cannot write under ``root``, by effect: opening an existing file there
    for writing (no create, no truncate, nothing written) is refused. None when there is no file to try."""
    target = probe_file(root)
    if target is None:
        return None
    try:
        os.close(os.open(target, os.O_WRONLY))
    except PermissionError:
        return True
    except OSError as exc:  # a read-only bind mount is EROFS
        return exc.errno == 30
    return False


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
    binds = [arg for root in roots if root.exists() for arg in ("--ro-bind", str(root), str(root))]
    return ["bwrap", "--die-with-parent", "--new-session", "--bind", "/", "/", "--dev-bind", "/dev", "/dev",
            *binds, "--", *command]


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
    """``command`` narrowed to the tests with the marker (``marked``) or without it, by `-m`
    (written ``-m EXPR``, ``-mEXPR`` or ``-m=EXPR``)."""
    out = list(command)
    clause = MARKER if marked else f"not {MARKER}"
    first = out.index("pytest") + 1
    for at in range(first, len(out)):
        part = out[at]
        if part == "-m" and at + 1 < len(out):
            out[at + 1] = f"({out[at + 1]}) and {clause}"
            return out
        if part.startswith("-m") and len(part) > 2:
            out[at] = f"-m=({part[3:] if part[2] == '=' else part[2:]}) and {clause}"
            return out
    out[first:first] = ["-m", clause]
    return out


def listed(repo: Path) -> frozenset[str]:
    """The test modules in ``tests/seatbelt-modules.txt`` that run `sandbox-exec` themselves, as
    paths relative to ``tests/`` (so a module of the same name in a subdirectory is not one)."""
    try:
        text = (repo / "tests" / "seatbelt-modules.txt").read_text(encoding="utf-8")
    except OSError:
        return frozenset()
    return frozenset(line.split("#")[0].strip() for line in text.splitlines() if line.split("#")[0].strip())


def named(command: list[str]) -> set[str]:
    """The test files a command selects by path or node id, relative to ``tests/``; empty when it
    selects a directory or everything."""
    out = set()
    for arg in command[command.index("pytest") + 1:]:
        if arg.startswith("tests/") and ".py" in arg:
            out.add(arg.split("::")[0][len("tests/"):])
    return out


def passes(command: list[str], environment: dict[str, str], repo: Path) -> list[Pass] | None:
    """The passes of a run under the denial, or None to run it as given.

    None (with a notice when it is not routine) when the run opted out (``DEV_TEST_WRITE_DENY=0``),
    is already inside a denial (found by effect, not by a variable), is not a plain
    ``python -m pytest`` command, or the platform cannot apply it. Two passes when the selection
    reaches both ordinary tests and ``seatbelt`` modules, else the one that is needed.
    """
    if "pytest" not in command:
        return None
    if environment.get(ENV) == "0":
        say("real-state write denial is off (DEV_TEST_WRITE_DENY=0); only the after-the-fact check applies")
        return None
    roots = protected(repo)
    if not roots:
        return None
    if all(denied_here(root) for root in roots):
        return None
    if not works():
        say("real-state write denial cannot be applied on this host; only the after-the-fact check applies")
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


def junit_at(command: list[str]) -> int | None:
    """The index of the argument holding the JUnit path: ``--junitxml=PATH`` itself, or the value
    after a bare ``--junitxml``."""
    for index, part in enumerate(command):
        if part.startswith("--junitxml="):
            return index
        if part == "--junitxml" and index + 1 < len(command):
            return index + 1
    return None


def junit_path(command: list[str]) -> Path | None:
    """Where the command writes its JUnit file, or None."""
    at = junit_at(command)
    if at is None:
        return None
    return Path(command[at].split("=", 1)[1] if command[at].startswith("--junitxml=") else command[at])


def with_junit(command: list[str], path: Path) -> list[str]:
    """``command`` writing its JUnit file to ``path`` instead."""
    out = list(command)
    at = junit_at(out)
    if at is not None:
        out[at] = f"--junitxml={path}" if out[at].startswith("--junitxml=") else str(path)
    return out


def merge_junit(into: Path, other: Path) -> None:
    """Fold the testcases and counts of the JUnit file ``other`` into ``into``."""
    from xml.etree import ElementTree

    try:
        second = ElementTree.parse(other)  # noqa: S314 - a file pytest just wrote
    except (OSError, ElementTree.ParseError):
        return  # that pass died before writing one; its exit status still counts
    try:
        first = ElementTree.parse(into)  # noqa: S314 - a file pytest just wrote
    except (OSError, ElementTree.ParseError):
        second.write(into, encoding="utf-8", xml_declaration=True)  # the first died: the second is all there is
        return
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
