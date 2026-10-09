"""Build a runtime from one commit, verify it, select it, recover from a broken selection and roll back."""

from __future__ import annotations

import argparse
import json
import os
import re
import stat
import subprocess
import tempfile
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path

from poolhouse import node_build, runtime, runtime_launchers, runtime_store
from poolhouse.files import writing
from poolhouse.fleet import runtime_wheel
from poolhouse.fleet.environment import Environment
from poolhouse.lock import Busy, only_one
from poolhouse.workspace import cli, harness_remote, limits, project_connection, tokens
from poolhouse.workspace.claims import Claims, Conflict
from poolhouse.workspace.identity import AGENT, Denied, Identity

BUILD_TIMEOUT = 1800.0
CLAIM_WAIT_S = 60.0
CLAIM_TTL_S = 3600.0
CLAIM_POLL_S = 1.0
SMOKE_TIMEOUT = 60.0
FLOOR = "packaging/runtime-floor"
HOOKS = ("claude-session-start", "claude-subagent-start", "claude-subagent-stop")
FAULTS = ("Traceback", "ModuleNotFoundError", "No module named", "ImportError")
COMMIT = re.compile(r"[0-9a-f]{40}")
FAILURES = (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError)


class DeployError(RuntimeError):
    """A build, smoke, claim or switch step failed."""


RECOVERABLE = (*FAILURES, DeployError)
BACKOFF_S = 600.0


@dataclass(frozen=True, slots=True)
class Plan:
    """What to deploy: the source checkout, the commit built from it, the launcher directory (or None) and the epoch floor."""

    checkout: Path
    commit: str
    launchers: Path | None
    floor: int = 0
    timeout: float = BUILD_TIMEOUT
    wait_s: float = CLAIM_WAIT_S
    agent: str = ""


@dataclass(frozen=True, slots=True)
class Outcome:
    """What an ensure, recover or rollback did."""

    action: str
    commit: str = ""
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.action != "failed"


Builder = Callable[[Plan, Path, Path], runtime.Runtime]


def git(checkout: Path, *words: str, check: bool = True) -> str:
    """Run git in a checkout and return stdout."""
    done = subprocess.run(["git", "-C", str(checkout), *words], capture_output=True, text=True, timeout=60)
    if check and done.returncode:
        raise DeployError(f"git {words[0]} failed: {(done.stderr or done.stdout).strip()[:300]}")
    return done.stdout.strip() if done.returncode == 0 else ""


def resolve_commit(checkout: Path, ref: str = "HEAD") -> str:
    """The full commit a ref names in a checkout."""
    if ref.startswith("-"):
        raise DeployError(f"{ref!r} is not a ref")
    commit = git(checkout, "rev-parse", "--verify", f"{ref}^{{commit}}")
    if not COMMIT.fullmatch(commit):
        raise DeployError(f"{ref} does not name a commit")
    return commit


def parse_epoch(text: str) -> int:
    """The integer in a runtime-floor file; 0 for an empty file, DeployError for anything else."""
    text = text.strip()
    if text and not text.isdigit():
        raise DeployError(f"{FLOOR} must hold one integer, not {text[:40]!r}")
    return int(text or 0)


def floor_of(checkout: Path, commit: str) -> int:
    """The lowest runtime epoch the source at `commit` accepts."""
    return parse_epoch(git(checkout, "show", f"{commit}:{FLOOR}", check=False))


def source_epoch(source: Path) -> int:
    """The runtime epoch a built source tree declares."""
    path = source / FLOOR
    return parse_epoch(path.read_text(encoding="utf-8")) if path.is_file() else 0


def acting(plan: Plan) -> Identity | None:
    """The authenticated workspace agent running this command, as a physical claim owner; None when there is none."""
    name = plan.agent or os.environ.get(tokens.AGENT_ENV, "")
    if not name:
        return None
    args = argparse.Namespace(agent=name, token_file="")
    try:
        ws, token = cli._context(args, cli._project_connection(plan.checkout))
        if isinstance(ws, project_connection.BoardWorkspace):
            info = ws.remote.call("whoami", token)
            if (info.get("id") != name or info.get("role") != AGENT or "claim" not in info.get("can", ())
                    or info.get("project", {}).get("key") != ws.remote.project_id):
                return None
            return harness_remote.physical_owner(ws.remote, Identity(name, AGENT, info.get("parent", ""),
                                                                      tuple(info.get("can", ()))))
        who = ws.auth(token)
    except (Denied, OSError, ValueError, KeyError, SystemExit):
        return None
    return who if who.role == AGENT and "claim" in who.can else None


@contextmanager
def owning(paths: list[Path], *, wait_s: float, who: Identity | None, note: str = "runtime deploy",
           store: Claims | None = None) -> Iterator[None]:
    """Hold an `install` claim as `who` on each path, waiting for a live other owner up to `wait_s` for each.

    With no agent there is no claim; the deploy lock alone serialises builds.
    """
    if who is None:
        yield
        return
    if store is None:
        base = limits.root()
        store = Claims(base, limits.load(base).claim_ttl_s)
    me = who
    taken: list[Path] = []
    try:
        for path in paths:
            deadline = time.monotonic() + wait_s
            while True:
                try:
                    store.claim(me, "install", str(path), {"ttl_s": CLAIM_TTL_S, "pid": os.getpid(),
                                                           "note": note})
                    break
                except Conflict as held:
                    if time.monotonic() >= deadline:
                        raise DeployError(f"{path} is claimed by {held.owner['owner']}") from held
                    time.sleep(CLAIM_POLL_S)
            taken.append(path)
        yield
    finally:
        for path in taken:
            with suppress(ValueError, Denied):
                store.release(me, "install", str(path))


def host_python() -> Path:
    """The Python that builds runtime environments."""
    found = Environment(runtime.directory() / "bootstrap").host_python()
    if found is None:
        raise OSError("install Python 3.13 to prepare an isolated runtime")
    return found


def build(plan: Plan, stage: Path, prefix: Path) -> runtime.Runtime:
    """Build the plan's commit into a new immutable runtime tree at `prefix`."""
    creator = runtime_store.creator_record(plan.agent or os.environ.get(tokens.AGENT_ENV, ""), "ensure")
    target = runtime_wheel.Target(prefix, creator, host_python())
    built = runtime_wheel.build(plan.checkout, plan.commit, stage, timeout=plan.timeout, target=target)
    try:
        node_build.add_to(built, stage / "source", plan.commit, timeout=plan.timeout)
    except RECOVERABLE:
        runtime_store.discard(built.prefix)
        raise
    return built


def _clean_environment(home: Path, bin_dir: Path) -> dict[str, str]:
    kept = {key: value for key, value in runtime.environment().items()
            if not key.startswith(("POOLHOUSE_", "POOLHOUSE_")) and key not in {"CLAUDE_ENV_FILE", "CLAUDE_CONFIG_DIR"}}
    return {**kept, "HOME": str(home), "POOLHOUSE_HOME": str(home / "state"), "POOLHOUSE_NO_REAL_KEYSTORE": "1",
            "POOLHOUSE_NONINTERACTIVE": "1", "POOLHOUSE_NOTIFY": "off", "POOLHOUSE_RUNTIME_ENSURE": "off",
            "PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"}


def smoke(chosen: runtime.Runtime, source: Path, stage: Path) -> None:
    """Import the runtime under -I, run its CLI help and the Claude hooks against a temporary state root."""
    runtime.verify(chosen)
    home, bin_dir = stage / "smoke-home", stage / "smoke-bin"
    home.mkdir(parents=True, exist_ok=True)
    bin_dir.mkdir(parents=True, exist_ok=True)
    runtime_launchers.install(bin_dir, chosen)
    env = _clean_environment(home, bin_dir)
    done = subprocess.run([str(bin_dir / "poolhouse-workspace"), "--help"], capture_output=True, text=True,
                          timeout=SMOKE_TIMEOUT, env=env, cwd=home)
    if done.returncode or not done.stdout.strip():
        raise DeployError(f"poolhouse-workspace --help exited {done.returncode}: {done.stderr.strip()[-300:]}")
    event = json.dumps({"session_id": "runtime-smoke", "cwd": str(home), "model": "runtime-smoke",
                        "agent_id": "smoke", "agent_type": "smoke", "description": "runtime smoke"})
    for name in HOOKS:
        script = source / "scripts" / "hooks" / name
        if not script.is_file():
            continue
        done = subprocess.run([str(chosen.python), str(script)], input=event, capture_output=True, text=True,
                              timeout=SMOKE_TIMEOUT, env={**env, "CLAUDE_PROJECT_DIR": str(source)}, cwd=home)
        if done.returncode or any(fault in done.stderr for fault in FAULTS):
            raise DeployError(f"hook {name} exited {done.returncode}: {done.stderr.strip()[-300:]}")


def _note(update: dict) -> None:
    with suppress(OSError):
        runtime_store.write_state(update)


def _record(plan: Plan) -> None:
    _note({"checkout": str(plan.checkout), "launchers": str(plan.launchers) if plan.launchers else None})


def switch(chosen: runtime.Runtime, plan: Plan) -> None:
    """Rewrite every launcher atomically, publish the selection, then record the layout."""
    if plan.launchers is not None:
        runtime_launchers.install(plan.launchers, chosen)
    runtime.publish(chosen)
    _record(plan)


def _switch_paths(plan: Plan) -> list[Path]:
    return [*([plan.launchers] if plan.launchers is not None else []), runtime.directory() / "selected.json"]


def _snapshot(directory: Path | None) -> dict[Path, tuple[bytes, int]]:
    """The bytes and mode of every console launcher in a directory."""
    if directory is None:
        return {}
    return {path: (path.read_bytes(), stat.S_IMODE(path.stat().st_mode))
            for path in directory.glob("poolhouse*") if path.is_file() and not path.is_symlink()}


def _restore(directory: Path | None, original: dict[Path, tuple[bytes, int]]) -> None:
    """Put the snapshotted launchers back and remove console launchers that were not there."""
    if directory is None:
        return
    with suppress(OSError):
        for path in directory.glob("poolhouse*"):
            if path not in original and path.is_file() and not path.is_symlink():
                path.unlink()
        for path, (data, mode) in original.items():
            with writing(path) as temporary:
                temporary.write_bytes(data)
                temporary.chmod(mode)


def build_and_switch(plan: Plan, builder: Builder = build, who: Identity | None = None) -> runtime.Runtime:
    """Build, smoke and select the plan's commit under install claims; a failure before the selection keeps the previous runtime."""
    who = who or acting(plan)
    prefix = runtime.directory() / plan.commit / uuid.uuid4().hex
    with owning([prefix, *_switch_paths(plan)], wait_s=plan.wait_s, who=who, note=f"runtime build {plan.commit[:7]}"), \
            tempfile.TemporaryDirectory(prefix="poolhouse-runtime-") as temporary:
        stage, original = Path(temporary).resolve(), _snapshot(plan.launchers)
        built, published = builder(plan, stage, prefix), False
        try:
            runtime_store.mark_created(built.prefix, who.id if who else plan.agent or os.environ.get(tokens.AGENT_ENV, ""), "ensure")
            if built.commit != plan.commit:
                raise DeployError(f"built {built.commit}, expected {plan.commit}")
            smoke(built, stage / "source", stage)
            epoch = source_epoch(stage / "source")
            if epoch < plan.floor:
                raise DeployError(f"built epoch {epoch} is below the floor {plan.floor}")
            runtime_store.mark_verified(built, epoch, node_build.mark(built.prefix))
            if plan.launchers is not None:
                runtime_launchers.install(plan.launchers, built)
            runtime.publish(built)
            published = True
        finally:
            if not published:
                _restore(plan.launchers, original)
                runtime_store.discard(built.prefix)
        _record(plan)
        return built


def acceptable(plan: Plan, chosen: runtime.Runtime) -> bool:
    """Whether a runtime is not rejected and carries at least the plan's epoch floor."""
    return not runtime_store.rejected(chosen.prefix) and runtime_store.epoch_of(chosen.prefix) >= plan.floor


def healthy(plan: Plan) -> runtime.Runtime | None:
    """The selected runtime when it verifies and meets the floor, else None."""
    chosen = runtime.available()
    return chosen if chosen is not None and acceptable(plan, chosen) else None


def fallback(plan: Plan) -> runtime.Runtime | None:
    """The newest verified runtime that still verifies and meets the floor."""
    held = runtime_store.read_state().get("held")
    held = held.get("commit") if isinstance(held, dict) else None
    ranked = sorted(runtime_store.candidates(), key=lambda candidate: candidate.commit == held)
    for candidate in ranked:
        if runtime_store.epoch_of(candidate.prefix) < plan.floor:
            continue
        try:
            return runtime.verify(candidate)
        except FAILURES:
            continue
    return None


def settled(plan: Plan) -> str:
    """"current" or "held" when the selection file names an intact tree that needs no build, else empty; runs nothing."""
    row = runtime_store.selection()
    prefix = Path(str(row.get("prefix", "")))
    if not runtime_store.intact(prefix) or runtime_store.epoch_of(prefix) < plan.floor:
        return ""
    if row.get("commit") == plan.commit:
        return "current"
    held = runtime_store.read_state().get("held")
    return "held" if isinstance(held, dict) and held.get("commit") == plan.commit else ""


def current(plan: Plan) -> bool:
    """Whether the selection file already names this commit's intact tree."""
    return settled(plan) == "current"


def _reject_below_floor(plan: Plan, keep: Path) -> None:
    for candidate in runtime_store.candidates():
        if candidate.prefix != keep and runtime_store.epoch_of(candidate.prefix) < plan.floor:
            runtime_store.reject(candidate.prefix, "below the source checkout's runtime floor")


def prepare_root() -> Path:
    root = runtime.directory()
    if not root.exists():
        root.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        root.mkdir(mode=0o700)
        runtime.protect(root)
    return root


def ensure(plan: Plan, *, force: bool = False, force_build: bool = False, builder: Builder = build) -> Outcome:
    """Make the plan's commit the selected runtime, recovering a broken selection first."""
    state = "" if force else settled(plan)
    if state:
        held = runtime_store.read_state().get("held") or {}
        return Outcome(state, plan.commit, str(held.get("reason", "")) if state == "held" else "")
    root = prepare_root()
    try:
        with only_one(root / "deploy.lock", wait=False, note="ensure"):
            return _ensured(plan, force, force_build, builder)
    except Busy as exc:
        return Outcome("busy", plan.commit, str(exc))


def _recovered(plan: Plan, who: Identity) -> tuple[runtime.Runtime | None, list[str]]:
    chosen = healthy(plan)
    if chosen is not None:
        return chosen, []
    chosen = fallback(plan)
    if chosen is None:
        return None, []
    try:
        with owning(_switch_paths(plan), wait_s=plan.wait_s, who=who):
            switch(chosen, plan)
    except RECOVERABLE as exc:
        return chosen, [f"could not select {chosen.commit[:7]}: {exc}"]
    return chosen, [f"selected {chosen.commit[:7]} in place of a broken selection"]


def _failed_recently(plan: Plan) -> bool:
    failure = runtime_store.read_state().get("last_failure")
    return (isinstance(failure, dict) and failure.get("commit") == plan.commit
            and time.time() - float(failure.get("at", 0.0)) < BACKOFF_S)


def _ensured(plan: Plan, force: bool, force_build: bool, builder: Builder) -> Outcome:
    who = acting(plan)
    if force_build and who is None:
        raise DeployError("--force-build needs an authenticated agent: pass --agent or set POOLHOUSE_WORKSPACE_AGENT")
    chosen, steps = _recovered(plan, who)
    held = runtime_store.read_state().get("held")
    if not force and chosen is not None and chosen.commit == plan.commit:
        runtime_store.collect({chosen.prefix})
        return Outcome("recovered" if steps else "current", plan.commit, "; ".join(steps))
    if not force and chosen is not None and isinstance(held, dict) and held.get("commit") == plan.commit:
        return Outcome("recovered" if steps else "held", plan.commit, "; ".join([*steps, str(held.get("reason", ""))]))
    if not force_build and _failed_recently(plan):
        return Outcome("recovered" if steps else "backoff", plan.commit,
                       "; ".join([*steps, f"the last build of {plan.commit[:7]} failed; retrying after {BACKOFF_S:.0f}s or with --force-build"]))
    try:
        built = build_and_switch(plan, builder, who)
    except RECOVERABLE as exc:
        _note({"last_failure": {"commit": plan.commit, "at": time.time(), "detail": str(exc)[-1000:]}})
        return Outcome("recovered" if steps else "failed", plan.commit, "; ".join([*steps, f"build failed: {exc}"])[-1500:])
    _note({"last_failure": None, "held": None, "last_success": {"commit": plan.commit, "at": time.time()}})
    _reject_below_floor(plan, built.prefix)
    runtime_store.collect({built.prefix, *([chosen.prefix] if chosen else [])})
    detail = [*steps, f"selected {built.prefix}", *([] if plan.launchers else ["launchers not updated"])]
    return Outcome("switched", plan.commit, "; ".join(detail))


def rollback(plan: Plan, to: str = "") -> Outcome:
    """Select an earlier verified runtime and hold the commit that was selected."""
    root = prepare_root()
    try:
        with only_one(root / "deploy.lock", wait=False, note="rollback"):
            who = acting(plan)
            if who is None:
                raise DeployError("rollback needs an authenticated agent: pass --agent or set POOLHOUSE_WORKSPACE_AGENT")
            now = runtime_store.selection(root)
            target = next((c for c in runtime_store.candidates(root)
                           if str(c.prefix) != now.get("prefix") and (not to or c.commit.startswith(to))), None)
            if target is None:
                return Outcome("failed", detail="no earlier verified runtime to select")
            try:
                runtime.verify(target)
                with owning(_switch_paths(plan), wait_s=plan.wait_s, who=who):
                    switch(target, plan)
            except RECOVERABLE as exc:
                return Outcome("failed", target.commit, str(exc)[-500:])
            if now.get("commit"):
                _note({"held": {"commit": now["commit"], "reason": f"rolled back to {target.commit[:7]}"}})
            return Outcome("switched", target.commit, f"selected {target.prefix}")
    except Busy as exc:
        return Outcome("busy", detail=str(exc))
