"""Install signed releases and immutable wheels from tracked branches at idle boundaries."""

from __future__ import annotations

import contextlib
import json
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import zipfile
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from ml_stack import net, runtime
from ml_stack.files import promote
from ml_stack.http import ServerError, ServerUnreachable
from ml_stack.httpguard import Refused
from ml_stack.lock import Busy
from ml_stack.net import git as netgit, provenance
from ml_stack.paths import repo_root
from ml_stack.safenames import Unsafe, safe_filename, unpack

from . import signing
from .measuring import installed_commit
from .runtime_wheel import source_checkout

__all__ = [
    "GIT_URL",
    "REPO",
    "Pulled",
    "Release",
    "TrackedBranch",
    "UpdateError",
    "UpdateRuntime",
    "UpdateSchedule",
    "apply_if_newer",
    "asset_for",
    "check",
    "checkout_here",
    "current_version",
    "download",
    "download_release",
    "follow_runtime",
    "in_the_way",
    "install",
    "quiet",
    "restart_after_update",
    "state",
    "track",
    "track_once",
    "watch",
]

_LOG = logging.getLogger(__name__)

REPO = "adammikulis/ml-stack"
GIT_URL = f"https://github.com/{REPO}"
API = "https://api.github.com/repos/{repo}/releases/latest"
TIMEOUT = 30.0
CHUNK = 1 << 20
GIT_TIMEOUT = 300.0
PIP_TIMEOUT = 1800.0
EVERY_S = 300.0
"""How often a tracked branch is looked at: five minutes, the same order as a push."""

SIGNATURE_SUFFIX = ".sig"
SIGNATURE_LIMIT = 16384

COMPANIONS = ("ml-stack", "ml-stack-headless", "ml-stack.exe", "ml-stack-headless.exe")
"""What a release download holds beside the thing that is running. An update replaces the
whole install, not the one binary that happened to notice it: the daemon and the CLI on
different versions is the bug this list exists to prevent."""




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
    from importlib.metadata import PackageNotFoundError, version

    with contextlib.suppress(PackageNotFoundError, LookupError):
        return version("ml-stack")
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
    try:
        body = net.default().json(API.format(repo=repo), net.Ask(
            purpose="update check", tries=3, headers={"Accept": "application/vnd.github+json"}))
    except ServerUnreachable as exc:
        raise UpdateError(f"could not reach GitHub: {exc}") from None
    except ServerError as exc:
        raise UpdateError(f"could not reach GitHub: {exc.status}") from None
    except (OSError, ValueError) as exc:
        raise UpdateError(f"could not reach GitHub: {exc}") from None
    if not isinstance(body, dict):
        raise UpdateError("GitHub answered with something that is not a release")
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
        name = str(asset.get("name", ""))
        if key in name and not name.endswith(SIGNATURE_SUFFIX):
            return asset
    return None


def download(asset: dict[str, Any], into: Path | str, *, on_progress: Any = None,
             allow_unscanned: bool = False, library_links: bool = False) -> Path:
    """Fetch one asset through the net pipeline and check it against the digest GitHub reports
    for it; an asset with no digest, or a name that is not one plain file name, is refused
    before anything is fetched.

    The file is format-checked and virus-scanned; an archive that no
    scanner could look at is kept only with ``allow_unscanned``.
    """
    want = str(asset.get("digest") or "").removeprefix("sha256:").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", want):
        raise UpdateError(f"{asset.get('name')!r} comes with no sha256 digest to check it against")
    try:
        name = safe_filename(str(asset["name"]))
    except Unsafe as exc:
        raise UpdateError(f"the asset's name is not a usable file name: {exc}") from None
    into = Path(into).expanduser()
    into.mkdir(parents=True, exist_ok=True)
    target = into / name
    progress = (lambda done, total: on_progress(done, total)) if on_progress else None
    hooks = on_progress if isinstance(on_progress, net.Hooks) else net.Hooks(progress=progress)
    try:
        net.download(str(asset["browser_download_url"]), target, net.Want(
            sha256=want, size=int(asset.get("size") or 0), require_digest=True,
            allow_unscanned=allow_unscanned, library_links=library_links,
            max_bytes=8 << 30, purpose="release download"),
            hooks=hooks)
    except net.ChecksumMismatch:
        raise UpdateError("the download does not match the digest GitHub reports for it") from None
    except (net.Blocked, net.Truncated, ServerError, OSError, Refused) as exc:
        raise UpdateError(f"download failed: {exc}") from None
    return target


def download_release(release: Release, asset: dict[str, Any], into: Path | str, *,
                     on_progress: Any = None) -> Path:
    """Fetch a release asset and keep it only if its ``.sig`` verifies against the pinned key."""
    name = str(asset["name"])
    sig = next((a for a in release.assets if a.get("name") == name + SIGNATURE_SUFFIX), None)
    if sig is None:
        raise UpdateError(f"{name} has no signature, so it is not installed")
    with tempfile.TemporaryDirectory(prefix="ml-stack-sig-") as tmp:
        signature = download(sig, tmp, allow_unscanned=True).read_bytes()[:SIGNATURE_LIMIT]
    target = download(asset, into, on_progress=on_progress)
    try:
        signing.verify_file(target, signature)
    except signing.SignatureError as exc:
        target.unlink(missing_ok=True)
        provenance.sidecar(target).unlink(missing_ok=True)
        raise UpdateError(f"{name} is not installed: {exc}") from None
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
        try:
            unpack(archive, staging)
        except (Unsafe, zipfile.BadZipFile) as exc:
            raise UpdateError(f"refusing the download: {exc}") from None
        found = _pick(staging, target.name)
        if found is None:
            raise UpdateError(f"the download has no {target.name} in it")
        _restore_modes(found)

        backup = target.with_name(target.name + ".old")
        shutil.rmtree(backup, ignore_errors=True)
        backup.unlink(missing_ok=True)
        if target.exists():
            promote(target, backup)
        promote(found, target)
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
            promote(new, beside)
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
def in_the_way(*, jobs: Callable[[], bool] | None = None,
               measuring: Callable[[], bool] | None = None,
               leases: Callable[[], bool] | None = None) -> str:
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


def quiet(**checks: Callable[[], bool] | None) -> Callable[[], bool]:
    """`in_the_way` as the ``idle`` gate `watch` and `track` take."""
    return lambda: not in_the_way(**checks)


# -- what this machine says about how it updates ----------------------------------------
LAST: dict[str, Any] = {"tracking": "off", "checked_at": 0.0, "error": "", "commit": ""}
"""The last look either loop took, for ``/health`` and so ``ml-stack-cluster status`` can
show a peer's mode and when it last asked. Written by the loops, read by `state`."""

_AGE: dict[str, float] = {}
_COMMIT: list[str] = []


def note(**fields: Any) -> None:
    """Record what a loop just did. Every field lands in `state`."""
    LAST.update(fields)


def commit_age_s(commit: str = "", checkout: Path | None = None) -> float:
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
    """`measuring.installed_commit`, asked once: it shells out to git, and the beacon
    rebuilds its report every ten seconds. It cannot change without the process
    restarting."""
    if not _COMMIT:
        _COMMIT.append(installed_commit())
    return _COMMIT[0]


def state() -> dict[str, Any]:
    """What this machine runs and how it keeps current, for the beacon and ``/health``.

    ``tracking`` is a branch name, ``releases``, or ``off``; ``update_checked_at`` is when
    that was last looked at, so `fleet.join.table` can print "main, 4m ago" rather than a
    claim nobody checked.
    """
    commit = _installed_commit()
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


def checkout_here() -> Path | None:
    """The git working tree this package is imported from, or None for a plain install."""
    return repo_root(Path(__file__).resolve().parent) or source_checkout()


def git_in(checkout: Path | str) -> Git:
    """The real git, rooted in ``checkout``. Output is stdout and stderr together, because
    what a failed pull says is on stderr and the report has to carry it."""
    where = str(Path(checkout).expanduser())

    def run(args: Any) -> tuple[int, str]:
        words = [str(a) for a in args]
        try:
            if words and words[0] in netgit.NETWORK:
                done = netgit.run(words, cwd=Path(where), url=_remote_in(words),
                                  protocols="https:ssh")
            else:
                done = subprocess.run(["git", "-C", where, *words], capture_output=True,
                                      text=True, timeout=GIT_TIMEOUT)
        except netgit.GitFailed as exc:
            return 1, str(exc)
        except (OSError, subprocess.SubprocessError, Refused) as exc:
            return 1, str(exc)
        return done.returncode, f"{done.stdout}{done.stderr}".strip()

    return run


def _remote_in(words: list[str]) -> str:
    """The address after ``--`` in a git network command, or ''."""
    return words[words.index("--") + 1] if "--" in words and words.index("--") + 1 < len(words) else ""


def pip_install(checkout: Path | str) -> tuple[int, str]:
    """Build, verify and select the checkout's HEAD as an immutable runtime in a separate process."""
    argv = [sys.executable, "-m", "ml_stack.runtime_cli", "ensure", "--checkout", str(Path(checkout).expanduser())]
    try:
        done = subprocess.run(argv, capture_output=True, text=True, timeout=PIP_TIMEOUT)
    except (OSError, subprocess.SubprocessError) as exc:
        return 1, str(exc)[-2000:]
    return done.returncode, f"{done.stdout}{done.stderr}".strip()[-2000:]


@dataclass(frozen=True, slots=True)
class TrackedBranch:
    repo_url: str
    branch: str
    install_dir: Path | str


@dataclass(frozen=True, slots=True)
class UpdateRuntime:
    git: Git | None = None
    pip: Callable[[Path], tuple[int, str]] = pip_install
    restart: Callable[[], Any] | None = None
    admission: Callable[[], contextlib.AbstractContextManager[Any]] = contextlib.nullcontext


@dataclass(frozen=True, slots=True)
class UpdateSchedule:
    interval: float = EVERY_S
    first_after_s: float = 30.0
    rounds: int = 0


_DEFAULT_RUNTIME = UpdateRuntime()
_TRACK_SCHEDULE = UpdateSchedule()
_RELEASE_SCHEDULE = UpdateSchedule(interval=24 * 3600, first_after_s=300.0)
_FOLLOW_SCHEDULE = UpdateSchedule(interval=60.0, first_after_s=60.0)


def _same(a: str, b: str) -> bool:
    """Two shas, one of which may be short."""
    a, b = a.strip(), b.strip()
    if not a or not b:
        return False
    n = min(len(a), len(b))
    return n >= 7 and a[:n] == b[:n]


REMOTE = re.compile(r"(https://[A-Za-z0-9][A-Za-z0-9.-]*(:[0-9]{1,5})?/[A-Za-z0-9._/~-]+"
                    r"|ssh://[A-Za-z0-9_.@-]+(:[0-9]{1,5})?/[A-Za-z0-9._/~-]+"
                    r"|[A-Za-z0-9_.-]+@[A-Za-z0-9][A-Za-z0-9.-]*:[A-Za-z0-9._/~-]+)")
BRANCH = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]*")


def check_remote(repo_url: str, branch: str) -> None:
    """`UpdateError` unless ``repo_url`` is an https or ssh git address and ``branch`` a
    branch name: neither may start with a dash or name a transport that runs a program."""
    if not REMOTE.fullmatch(repo_url) or ".." in repo_url:
        raise UpdateError(f"{repo_url!r} is not an https or ssh git address")
    if not BRANCH.fullmatch(branch) or ".." in branch or branch.endswith((".lock", "/")):
        raise UpdateError(f"{branch!r} is not a branch name")


def _unsigned(run: Git, target: str = "FETCH_HEAD") -> str:
    """Return the signature verification failure for a commit, or an empty string."""
    key = signing.RELEASE_KEY.split()
    if len(key) < 2:
        return "no release key is set"
    with tempfile.TemporaryDirectory(prefix="ml-stack-signers-") as tmp:
        signers = Path(tmp) / "allowed_signers"
        signers.write_text(f"* {key[0]} {key[1]}\n")
        rc, out = run(["-c", "gpg.format=ssh", "-c", f"gpg.ssh.allowedSignersFile={signers}",
                       "verify-commit", target])
    return "" if rc == 0 else (out or "verify-commit failed")


def track_once(source: TrackedBranch, *, runtime: UpdateRuntime = _DEFAULT_RUNTIME) -> Pulled:
    """Fast-forward a signed branch and install its immutable wheel before restarting."""
    repo_url, branch = source.repo_url, source.branch
    checkout = Path(source.install_dir).expanduser()
    run = runtime.git if runtime.git is not None else git_in(checkout)
    bring_back = runtime.restart if runtime.restart is not None else restart_after_update

    try:
        check_remote(repo_url, branch)
    except UpdateError as exc:
        return Pulled(branch, error=str(exc))
    rc, out = run(["ls-remote", "--", repo_url, branch])
    head = out.split()[0] if rc == 0 and out.split() else ""
    if rc != 0 or not head:
        return Pulled(branch, error=f"could not read {branch} on {repo_url}: "
                                    f"{out or 'no such branch'}")

    rc, out = run(["rev-parse", "HEAD"])
    local = out.split()[0] if rc == 0 and out.split() else ""
    if rc != 0 or not local:
        return Pulled(branch, remote=head,
                      error=f"{checkout} is not a git checkout: {out or 'no HEAD'}")
    if _same(local, head) and _same(_installed_commit(), local):
        return Pulled(branch, was=local, now=local, remote=head)

    rc, out = run(["fetch", "--", repo_url, branch])
    if rc != 0:
        return Pulled(branch, was=local, now=local, remote=head,
                      error=f"could not fetch {branch}: {out}")

    refused = _unsigned(run, local if _same(local, head) else "FETCH_HEAD")
    if refused:
        return Pulled(branch, was=local, now=local, remote=head,
                      error=f"{branch} at {head[:7]} is not signed by the release key, "
                            f"so it is not pulled: {refused}")

    if _same(local, head):
        return _install_branch(Pulled(branch, was=local, now=local, remote=head),
                               checkout, runtime.pip, bring_back)

    rc, _ = run(["merge-base", "--is-ancestor", "HEAD", "FETCH_HEAD"])
    if rc != 0:
        return Pulled(branch, was=local, now=local, remote=head, diverged=True,
                      error=f"{checkout} has commits {branch} does not, so it is left "
                            f"alone. Merge or reset it by hand, then it follows again.")

    rc, out = run(["merge", "--ff-only", "FETCH_HEAD"])
    if rc != 0:
        return Pulled(branch, was=local, now=local, remote=head,
                      error=f"git merge --ff-only failed, so this machine keeps the code "
                            f"it has: {out}")

    rc, out = run(["rev-parse", "HEAD"])
    now = out.split()[0] if rc == 0 and out.split() else head

    return _install_branch(Pulled(branch, was=local, now=now, remote=head, pulled=True),
                           checkout, runtime.pip, bring_back)


def _install_branch(result: Pulled, checkout: Path,
                    pip: Callable[[Path], tuple[int, str]], restart: Callable[[], Any]) -> Pulled:
    code, said = pip(checkout)
    if code != 0:
        return replace(result, error=f"source is at {result.now[:7]}, but immutable wheel "
                       f"installation failed and it was not restarted: {said}")
    return replace(result, installed=True, restarted=str(restart() or ""))


def track(source: TrackedBranch, *, idle: Callable[[], bool] = lambda: True,
          runtime: UpdateRuntime = _DEFAULT_RUNTIME,
          schedule: UpdateSchedule = _TRACK_SCHEDULE) -> threading.Thread:
    """Follow a signed branch and replace the installed wheel while idle."""
    note(tracking=source.branch)

    def loop() -> None:
        time.sleep(schedule.first_after_s)
        seen = 0
        while not schedule.rounds or seen < schedule.rounds:
            seen += 1
            try:
                with runtime.admission():
                    if idle():
                        got = track_once(source, runtime=runtime)
                        note(checked_at=time.time(), error=got.error,
                             commit=got.now or LAST.get("commit", ""))
                        if got.restarted:
                            return
            except Busy:
                pass
            except Exception as exc:                  # noqa: BLE001 - a loop that dies stops following
                note(checked_at=time.time(), error=str(exc))
            time.sleep(schedule.interval)

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


def selected_commit() -> str:
    """The commit the machine's selected runtime was built from, or "" when none is selected."""
    try:
        row = json.loads((runtime.directory() / "selected.json").read_text(encoding="utf-8"))
        return str(row.get("commit", "")) if isinstance(row, dict) else ""
    except (OSError, ValueError):
        return ""


def follow_runtime(*, idle: Callable[[], bool], schedule: UpdateSchedule = _FOLLOW_SCHEDULE,
                   admission: Callable[[], contextlib.AbstractContextManager[Any]] = contextlib.nullcontext
                   ) -> threading.Thread | None:
    """Restart this process onto the selected runtime when it is behind it and nothing is in the way.

    Returns None when no runtime is selected.
    """
    if not selected_commit():
        return None

    def loop() -> None:
        time.sleep(schedule.first_after_s)
        while True:
            try:
                with admission():
                    chosen = selected_commit()
                    if chosen and not _same(chosen, _installed_commit()) and idle():
                        note(commit=chosen)
                        if restart_after_update():
                            return
            except Busy:
                pass
            except Exception as exc:                  # noqa: BLE001 - a loop that dies stops following
                note(checked_at=time.time(), error=str(exc))
            time.sleep(schedule.interval)

    thread = threading.Thread(target=loop, daemon=True, name="follow-runtime")
    thread.start()
    return thread


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
        archive = download_release(release, asset,
                                   tempfile.mkdtemp(prefix="ml-stack-update-"))
        install(archive)
    except UpdateError as exc:
        return {"ok": False, "installed": False, "error": str(exc)}
    return {"ok": True, "installed": True, "version": release.version}


def watch(*, wanted: Callable[[], bool], idle: Callable[[], bool],
          restart: Callable[[], Any] | None = None,
          schedule: UpdateSchedule = _RELEASE_SCHEDULE,
          admission: Callable[[], contextlib.AbstractContextManager[Any]] = contextlib.nullcontext) -> threading.Thread:
    """Check releases on a schedule and install them when the machine is idle."""
    bring_back = restart if restart is not None else restart_after_update
    # Recorded here rather than from inside the thread: which mode this machine is in is
    # known the moment the watcher is set up, and a loop writing it on every turn is a loop
    # scribbling over shared state for no reason.
    note(tracking="releases")

    def loop() -> None:
        time.sleep(schedule.first_after_s)
        seen = 0
        while not schedule.rounds or seen < schedule.rounds:
            seen += 1
            try:
                with admission():
                    if wanted() and idle():
                        got = apply_if_newer()
                        note(checked_at=time.time(), error=str(got.get("error") or ""))
                        if got.get("installed") and bring_back():
                            return
            except Busy:
                pass
            except Exception as exc:                  # noqa: BLE001
                _LOG.exception("Release update check failed")
                note(checked_at=time.time(), error=str(exc))
            time.sleep(schedule.interval)

    thread = threading.Thread(target=loop, daemon=True, name="updates")
    thread.start()
    return thread
