"""Keeping this machine's ml-stack current: a newer release, or the head of a branch.

Two modes, and a machine picks one. **Releases** is what a bundled install does: ask
GitHub for the newest release, download the zip for this platform, swap the whole bundle
into place -- the daemon, the CLI and the app window are one download -- and restart.
**A branch** is for a machine that is a git checkout with an editable install: poll
``git ls-remote`` for the head of, say, ``main``, fast-forward onto it, reinstall only if
the packaging changed, and restart. That one runs unreviewed code the moment it is pushed,
so it is off unless asked for.

Neither ever interrupts work. `quiet` is the gate both loops pass through: no job running,
no benchmark measuring, no model loaded. A machine part way through a run is left alone
until it is not, however new the code is.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from collections.abc import Callable
from pathlib import Path
from typing import Any

__all__ = ["Pulled", "Release", "UpdateError", "apply_if_newer", "asset_for", "check",
           "checkout_here", "current_version", "in_the_way", "quiet", "state", "track",
           "track_once", "restart_after_update", "watch",
           "download", "install", "REPO", "GIT_URL"]

REPO = "adammikulis/ml-stack"
GIT_URL = f"https://github.com/{REPO}"
API = "https://api.github.com/repos/{repo}/releases/latest"
TIMEOUT = 30.0
CHUNK = 1 << 20
GIT_TIMEOUT = 300.0
PIP_TIMEOUT = 1800.0
EVERY_S = 300.0
"""How often a tracked branch is looked at: five minutes, the same order as a push."""

COMPANIONS = ("ml-stack", "ml-stack-headless", "ml-stack.exe", "ml-stack-headless.exe")
"""What a release download holds beside the thing that is running. An update replaces the
whole install, not the one binary that happened to notice it: the daemon and the CLI on
different versions is the bug this list exists to prevent."""

INSTALL_TRIGGERS = ("pyproject.toml", "setup.py", "setup.cfg", "uv.lock", "poetry.lock",
                    "requirements.txt", "requirements-dev.txt")
"""A pull that changed one of these needs ``pip install -e .`` again; any other pull does
not, because an editable install already reads the files that moved."""


class UpdateError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class Release:
    version: str
    url: str
    notes: str
    assets: tuple[dict[str, Any], ...]
    checked_at: float

    def newer_than(self, version: str) -> bool:
        """False when the running version is unknown: nothing is newer than nothing."""
        if not version.strip():
            return False
        return _parse(self.version) > _parse(version)


def _parse(version: str) -> tuple[int, ...]:
    cleaned = version.strip().lstrip("vV").split("-")[0].split("+")[0]
    out = []
    for part in cleaned.split("."):
        digits = "".join(c for c in part if c.isdigit())
        out.append(int(digits) if digits else 0)
    return tuple(out) or (0,)


def current_version() -> str:
    """The running version, or empty when there is no way to tell."""
    try:
        from importlib.metadata import version
        return version("ml-stack-fleet")
    except Exception:                                 # noqa: BLE001
        pass
    told = os.environ.get("ML_STACK_VERSION", "").strip()
    if told:
        return told
    return _version_in_source() or ""


def _version_in_source() -> str:
    """The version in this checkout, when running from one rather than an install."""
    for parent in Path(__file__).resolve().parents:
        found = parent / "pyproject.toml"
        if not found.is_file():
            continue
        for line in found.read_text().splitlines():
            if line.startswith("version"):
                _, _, value = line.partition("=")
                value, _, _ = value.partition("#")
                return value.strip().strip('"').strip("'")
    return ""


def platform_key() -> str:
    """The asset name fragment for this machine."""
    machine = platform.machine().lower()
    arch = "arm64" if machine in ("arm64", "aarch64") else "x86_64"
    if sys.platform == "darwin":
        return f"macos-{arch}"
    if sys.platform == "win32":
        return f"windows-{arch}"
    return f"linux-{arch}"


def check(repo: str = REPO, *, timeout: float = TIMEOUT) -> Release:
    """Ask GitHub for the newest release."""
    req = urllib.request.Request(API.format(repo=repo),
                                 headers={"Accept": "application/vnd.github+json",
                                          "User-Agent": "ml-stack"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = json.loads(r.read())
    except urllib.error.HTTPError as exc:
        raise UpdateError(f"could not reach GitHub: {exc.code}") from None
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise UpdateError(f"could not reach GitHub: {exc}") from None
    return Release(
        version=str(body.get("tag_name") or "").lstrip("v"),
        url=str(body.get("html_url") or ""),
        notes=str(body.get("body") or ""),
        assets=tuple(body.get("assets") or ()),
        checked_at=time.time(),
    )


def asset_for(release: Release, key: str = "") -> dict[str, Any] | None:
    """The download for this machine, or None if the release has none."""
    key = key or platform_key()
    for asset in release.assets:
        if key in str(asset.get("name", "")):
            return asset
    return None


def download(asset: dict[str, Any], into: Path | str,
             *, on_progress: Any = None, timeout: float = 600.0) -> Path:
    """Fetch one asset and check it against the digest GitHub reports for it.

    The digest is not a signature: it proves the bytes match what that release holds, not
    who built them. Trust here is the same as downloading it by hand -- TLS to github.com
    and the repository name below.
    """
    into = Path(into).expanduser()
    into.mkdir(parents=True, exist_ok=True)
    target = into / str(asset["name"])
    url = str(asset["browser_download_url"])
    total = int(asset.get("size") or 0)

    req = urllib.request.Request(url, headers={"User-Agent": "ml-stack"})
    digest = hashlib.sha256()
    done = 0
    partial = target.with_suffix(target.suffix + ".part")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r, partial.open("wb") as fh:
            while True:
                block = r.read(CHUNK)
                if not block:
                    break
                fh.write(block)
                digest.update(block)
                done += len(block)
                if on_progress:
                    on_progress(done, total)
    except (urllib.error.URLError, OSError) as exc:
        partial.unlink(missing_ok=True)
        raise UpdateError(f"download failed: {exc}") from None

    want = str(asset.get("digest") or "").removeprefix("sha256:").strip().lower()
    if want and digest.hexdigest() != want:
        partial.unlink(missing_ok=True)
        raise UpdateError("the download does not match the digest GitHub reports for it")
    if total and done != total:
        partial.unlink(missing_ok=True)
        raise UpdateError(f"got {done} of {total} bytes")
    os.replace(partial, target)
    return target


def install(archive: Path | str, *, app_path: Path | str | None = None) -> Path:
    """Unpack a downloaded release over the running one. Returns what it replaced.

    The replacement is atomic per item: the new copy is unpacked beside the old, and only
    then swapped in, so an interrupted install leaves the working copy alone.
    """
    archive = Path(archive).expanduser()
    target = Path(app_path).expanduser() if app_path else running_path()
    if target is None:
        raise UpdateError("cannot tell what to replace; install it by hand")

    staging = Path(tempfile.mkdtemp(prefix="ml-stack-update-", dir=str(target.parent)))
    try:
        with zipfile.ZipFile(archive) as zf:
            for member in zf.namelist():
                if member.startswith("/") or ".." in Path(member).parts:
                    raise UpdateError(f"refusing an archive entry named {member!r}")
            zf.extractall(staging)
        found = _pick(staging, target.name)
        if found is None:
            raise UpdateError(f"the download has no {target.name} in it")
        _restore_modes(found)

        backup = target.with_name(target.name + ".old")
        shutil.rmtree(backup, ignore_errors=True)
        backup.unlink(missing_ok=True)
        if target.exists():
            os.replace(target, backup)
        os.replace(found, target)
        shutil.rmtree(backup, ignore_errors=True)
        _replace_companions(staging, target)
        return target
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _replace_companions(staging: Path, target: Path) -> list[Path]:
    """The rest of the download, put beside what was just replaced.

    A headless install is two files -- ``ml-stack`` and ``ml-stack-headless`` -- from one
    zip. Replacing only the one that noticed the release leaves the CLI a version behind
    the daemon it drives, which is the shape of bug that costs an afternoon. Only names
    that are already there are replaced: this puts nothing new on a machine.
    """
    done: list[Path] = []
    for name in COMPANIONS:
        if name == target.name:
            continue
        beside = target.parent / name
        if not beside.is_file():
            continue
        new = _pick(staging, name)
        if new is None or not new.is_file():
            continue
        _restore_modes(new)
        try:
            os.replace(new, beside)
        except OSError:                               # a busy file on Windows; not fatal
            continue
        done.append(beside)
    return done


def _pick(staging: Path, name: str) -> Path | None:
    direct = staging / name
    if direct.exists():
        return direct
    for candidate in staging.rglob(name):
        return candidate
    return None


def _restore_modes(path: Path) -> None:
    """Zip does not carry the executable bit on every platform."""
    if path.is_file():
        path.chmod(path.stat().st_mode | 0o111)
        return
    for item in path.rglob("*"):
        if item.is_file() and (item.parent.name == "MacOS" or not item.suffix):
            item.chmod(item.stat().st_mode | 0o111)


def running_path() -> Path | None:
    """The app or binary currently executing, when it is a bundle."""
    if not getattr(sys, "frozen", False):
        return None
    exe = Path(sys.executable).resolve()
    for parent in exe.parents:
        if parent.suffix == ".app":
            return parent
    return exe


def relaunch(*, delay_s: float = 1.5, stop: bool = True) -> bool:
    """Start the replaced copy and stop this one. False when this is not a bundle.

    The wait is so an answer already being written reaches the browser first.
    """
    target = running_path()
    if target is None:
        return False

    def go() -> None:
        time.sleep(delay_s)
        try:
            if target.suffix == ".app":
                subprocess.Popen(["open", "-n", str(target)])
            else:
                subprocess.Popen([str(target)], start_new_session=True)
        finally:
            if stop:
                os._exit(0)

    threading.Thread(target=go, daemon=True, name="relaunch").start()
    return True



# -- what an update must never walk over ------------------------------------------------
def in_the_way(*, jobs: "Callable[[], bool] | None" = None,
               measuring: "Callable[[], bool] | None" = None,
               leases: "Callable[[], bool] | None" = None) -> str:
    """Why an update has to wait, or "" when nothing is in its way.

    Three things, and any one of them is enough: a training job, a benchmark measuring
    (the same lock ``ml-stack-bench status`` reads, so a run somebody started at the
    keyboard counts), and a model server this machine holds a lease on. Replacing the code
    under any of them turns a measurement into a mixture of two builds, or drops a served
    model mid-answer. A check that raises counts as busy: not being able to tell is not a
    reason to go ahead.
    """
    for why, look in (("a job is running", jobs),
                      ("a benchmark is measuring", measuring),
                      ("a model is loaded", leases)):
        if look is None:
            continue
        try:
            if look():
                return why
        except Exception:                             # noqa: BLE001
            return f"could not tell whether {why}"
    return ""


def quiet(**checks: "Callable[[], bool] | None") -> "Callable[[], bool]":
    """`in_the_way` as the ``idle`` gate `watch` and `track` take."""
    return lambda: not in_the_way(**checks)


# -- what this machine says about how it updates ----------------------------------------
LAST: dict[str, Any] = {"tracking": "off", "checked_at": 0.0, "error": "", "commit": ""}
"""The last look either loop took, for ``/health`` and so ``ml-stack-fleet status`` can
show a peer's mode and when it last asked. Written by the loops, read by `state`."""

_AGE: dict[str, float] = {}
_COMMIT: list[str] = []


def note(**fields: Any) -> None:
    """Record what a loop just did. Every field lands in `state`."""
    LAST.update(fields)


def commit_age_s(commit: str = "", checkout: "Path | None" = None) -> float:
    """How old the commit this machine runs is, in seconds; 0 when there is no telling.

    Cached on the sha, because the beacon rebuilds its report every ten seconds and the
    answer cannot change without the process restarting onto a different commit.
    """
    if not commit:
        return 0.0
    if commit in _AGE:
        return _AGE[commit]
    where = checkout if checkout is not None else checkout_here()
    made = 0.0
    if where is not None:
        try:
            done = subprocess.run(["git", "-C", str(where), "log", "-1", "--format=%ct"],
                                  capture_output=True, text=True, timeout=15)
            if done.returncode == 0 and done.stdout.strip().isdigit():
                made = max(0.0, time.time() - float(done.stdout.strip()))
        except (OSError, subprocess.SubprocessError, ValueError):
            made = 0.0
    _AGE[commit] = made
    return made


def _installed_commit() -> str:
    """`bench.installed_commit`, asked once: it shells out to git, and the beacon rebuilds
    its report every ten seconds. It cannot change without the process restarting."""
    if not _COMMIT:
        from .bench import installed_commit

        _COMMIT.append(installed_commit())
    return _COMMIT[0]


def state() -> dict[str, Any]:
    """What this machine runs and how it keeps current, for the beacon and ``/health``.

    ``tracking`` is a branch name, ``releases``, or ``off``; ``update_checked_at`` is when
    that was last looked at, so `fleet.join.table` can print "main, 4m ago" rather than a
    claim nobody checked.
    """
    commit = str(LAST.get("commit") or "") or _installed_commit()
    return {"version": current_version(), "commit": commit,
            "commit_age_s": commit_age_s(commit),
            "tracking": str(LAST.get("tracking") or "off"),
            "update_checked_at": float(LAST.get("checked_at") or 0.0),
            "update_error": str(LAST.get("error") or "")}


# -- following a branch -----------------------------------------------------------------
Git = Callable[[Any], "tuple[int, str]"]
"""``(argv without 'git') -> (returncode, output)``. The seam a test replaces."""


@dataclass(frozen=True, slots=True)
class Pulled:
    """One look at a tracked branch, and what it did about what it found."""

    branch: str
    was: str = ""
    now: str = ""
    remote: str = ""
    pulled: bool = False
    installed: bool = False
    restarted: str = ""
    diverged: bool = False
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error

    def public(self) -> dict[str, Any]:
        return {"branch": self.branch, "was": self.was[:7], "now": self.now[:7],
                "remote": self.remote[:7], "pulled": self.pulled,
                "installed": self.installed, "restarted": self.restarted,
                "diverged": self.diverged, "error": self.error}


def checkout_here() -> "Path | None":
    """The git working tree this package is imported from, or None for a plain install."""
    from ml_stack.paths import repo_root

    return repo_root(Path(__file__).resolve().parent)


def git_in(checkout: "Path | str") -> Git:
    """The real git, rooted in ``checkout``. Output is stdout and stderr together, because
    what a failed pull says is on stderr and the report has to carry it."""
    where = str(Path(checkout).expanduser())

    def run(args: Any) -> "tuple[int, str]":
        try:
            done = subprocess.run(["git", "-C", where, *[str(a) for a in args]],
                                  capture_output=True, text=True, timeout=GIT_TIMEOUT)
        except (OSError, subprocess.SubprocessError) as exc:
            return 1, str(exc)
        return done.returncode, f"{done.stdout}{done.stderr}".strip()

    return run


def pip_install(checkout: "Path | str") -> "tuple[int, str]":
    """``pip install -e .`` in the checkout, with the interpreter that is running."""
    try:
        done = subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."],
                              cwd=str(Path(checkout).expanduser()), capture_output=True,
                              text=True, timeout=PIP_TIMEOUT)
    except (OSError, subprocess.SubprocessError) as exc:
        return 1, str(exc)
    return done.returncode, f"{done.stdout}{done.stderr}".strip()[-2000:]


def _same(a: str, b: str) -> bool:
    """Two shas, one of which may be short."""
    a, b = a.strip(), b.strip()
    if not a or not b:
        return False
    n = min(len(a), len(b))
    return n >= 7 and a[:n] == b[:n]


def track_once(repo_url: str, branch: str, install_dir: "Path | str", *,
               git: Git | None = None,
               pip: "Callable[[Path], tuple[int, str]]" = pip_install,
               restart: "Callable[[], Any] | None" = None) -> Pulled:
    """One look at ``branch``: fast-forward onto it if it moved, and restart on the new code.

    Never a merge. A checkout holding commits the branch does not have is *reported and
    left alone* -- resolving that is a person's decision, and a daemon that reset someone's
    work in progress at three in the morning would be unforgivable. ``pip install -e .``
    runs only when the pull touched packaging (`INSTALL_TRIGGERS`); an editable install
    already sees every other file that moved. A pull that fails changes nothing, so the
    daemon keeps running the code it started with.

    ``git``, ``pip`` and ``restart`` are the three seams; everything else is what the
    machine does.
    """
    checkout = Path(install_dir).expanduser()
    run = git if git is not None else git_in(checkout)
    bring_back = restart if restart is not None else restart_after_update

    rc, out = run(["ls-remote", repo_url, branch])
    head = out.split()[0] if rc == 0 and out.split() else ""
    if rc != 0 or not head:
        return Pulled(branch, error=f"could not read {branch} on {repo_url}: "
                                    f"{out or 'no such branch'}")

    rc, out = run(["rev-parse", "HEAD"])
    local = out.split()[0] if rc == 0 and out.split() else ""
    if rc != 0 or not local:
        return Pulled(branch, remote=head,
                      error=f"{checkout} is not a git checkout: {out or 'no HEAD'}")
    if _same(local, head):
        return Pulled(branch, was=local, now=local, remote=head)

    rc, out = run(["fetch", repo_url, branch])
    if rc != 0:
        return Pulled(branch, was=local, now=local, remote=head,
                      error=f"could not fetch {branch}: {out}")

    rc, _ = run(["merge-base", "--is-ancestor", "HEAD", "FETCH_HEAD"])
    if rc != 0:
        return Pulled(branch, was=local, now=local, remote=head, diverged=True,
                      error=f"{checkout} has commits {branch} does not, so it is left "
                            f"alone. Merge or reset it by hand, then it follows again.")

    rc, changed = run(["diff", "--name-only", "HEAD", "FETCH_HEAD"])
    moved = [line.strip() for line in changed.splitlines() if line.strip()]
    # Not being able to list what moved means installing anyway: a stale install is worse
    # than a wasted minute.
    needs_install = rc != 0 or any(Path(f).name in INSTALL_TRIGGERS for f in moved)

    rc, out = run(["pull", "--ff-only", repo_url, branch])
    if rc != 0:
        return Pulled(branch, was=local, now=local, remote=head,
                      error=f"git pull --ff-only failed, so this machine keeps the code "
                            f"it has: {out}")

    rc, out = run(["rev-parse", "HEAD"])
    now = out.split()[0] if rc == 0 and out.split() else head

    if needs_install:
        code, said = pip(checkout)
        if code != 0:
            return Pulled(branch, was=local, now=now, remote=head, pulled=True,
                          error=f"pulled {now[:7]}, but 'pip install -e .' failed and it "
                                f"was not restarted: {said}")
        return Pulled(branch, was=local, now=now, remote=head, pulled=True,
                      installed=True, restarted=str(bring_back() or ""))
    return Pulled(branch, was=local, now=now, remote=head, pulled=True,
                  restarted=str(bring_back() or ""))


def track(repo_url: str, branch: str, install_dir: "Path | str", *,
          interval: float = EVERY_S, first_after_s: float = 30.0,
          idle: "Callable[[], bool]" = lambda: True,
          git: Git | None = None,
          pip: "Callable[[Path], tuple[int, str]]" = pip_install,
          restart: "Callable[[], Any] | None" = None,
          rounds: int = 0) -> threading.Thread:
    """Follow ``branch`` on a timer, on a machine that is a checkout with an editable install.

    ``idle`` is `quiet`: nothing is pulled over a job, a measurement or a loaded model. The
    thread stops once it has restarted, because the restart is what puts the new code in
    charge -- either the process is gone or it re-execs. ``rounds`` bounds the loop for a
    test; 0 is forever.
    """
    note(tracking=branch)

    def loop() -> None:
        time.sleep(first_after_s)
        seen = 0
        while not rounds or seen < rounds:
            seen += 1
            try:
                if idle():
                    got = track_once(repo_url, branch, install_dir, git=git, pip=pip,
                                     restart=restart)
                    note(checked_at=time.time(), error=got.error,
                         commit=got.now or LAST.get("commit", ""))
                    if got.restarted:
                        return
            except Exception as exc:                  # noqa: BLE001 - a loop that dies stops following
                note(checked_at=time.time(), error=str(exc))
            time.sleep(interval)

    thread = threading.Thread(target=loop, daemon=True, name="track")
    thread.start()
    return thread


# -- putting the new code in charge -------------------------------------------------------
def restart_after_update() -> str:
    """Run the code that is now on disk. Says how, or "" when it could do nothing.

    A bundle relaunches itself -- that is the window coming back, and the headless binary
    too. Anything else asks `autostart`, which lets the login service bring the daemon back
    where there is one and re-execs where there is not.
    """
    if relaunch():
        return "relaunched"
    from . import autostart

    return autostart.restart()


def apply_if_newer() -> dict[str, Any]:
    """Put the newest release in place, if there is one. Says what happened."""
    if running_path() is None:
        return {"ok": False, "installed": False,
                "error": "this copy was installed with pip; update it with pip"}
    now = current_version()
    try:
        release = check()
        if not release.newer_than(now):
            return {"ok": True, "installed": False, "version": now}
        asset = asset_for(release)
        if asset is None:
            return {"ok": False, "installed": False,
                    "error": f"release {release.version} has no download for "
                             "this machine"}
        archive = download(asset, tempfile.mkdtemp(prefix="ml-stack-update-"))
        install(archive)
    except UpdateError as exc:
        return {"ok": False, "installed": False, "error": str(exc)}
    return {"ok": True, "installed": True, "version": release.version}


def watch(*, wanted: "Callable[[], bool]", idle: "Callable[[], bool]",
          every_s: float = 24 * 3600, first_after_s: float = 300.0,
          restart: "Callable[[], Any] | None" = None,
          rounds: int = 0) -> threading.Thread:
    """Check for a newer release on a timer, and put it on when nothing is running.

    A machine part way through a training run, a measurement or an answer is left alone
    until it is not (``idle`` is `quiet`). ``restart`` is the seam: the release is the
    whole install, so what comes back is the new daemon, the new CLI and, if this is the
    windowed copy, the new window. ``rounds`` bounds the loop for a test; 0 is forever.
    """
    bring_back = restart if restart is not None else restart_after_update
    # Recorded here rather than from inside the thread: which mode this machine is in is
    # known the moment the watcher is set up, and a loop writing it on every turn is a loop
    # scribbling over shared state for no reason.
    note(tracking="releases")

    def loop() -> None:
        time.sleep(first_after_s)
        seen = 0
        while not rounds or seen < rounds:
            seen += 1
            try:
                if wanted() and idle():
                    got = apply_if_newer()
                    note(checked_at=time.time(), error=str(got.get("error") or ""))
                    if got.get("installed") and bring_back():
                        return
            except Exception:                         # noqa: BLE001
                pass
            time.sleep(every_s)

    thread = threading.Thread(target=loop, daemon=True, name="updates")
    thread.start()
    return thread
