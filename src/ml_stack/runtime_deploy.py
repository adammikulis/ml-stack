"""Build a runtime from one commit, verify it, select it, recover from a broken selection and roll back."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path

from ml_stack import runtime, runtime_launchers, runtime_store
from ml_stack.fleet import runtime_wheel
from ml_stack.lock import Busy, only_one
from ml_stack.workspace import limits
from ml_stack.workspace.claims import Claims, Conflict
from ml_stack.workspace.identity import AGENT, Denied, Identity

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


@dataclass(frozen=True, slots=True)
class Plan:
    """What to deploy: the source checkout, the commit built from it and where launchers live."""

    checkout: Path
    commit: str
    launchers: Path
    floor: str = ""
    timeout: float = BUILD_TIMEOUT
    wait_s: float = CLAIM_WAIT_S


@dataclass(frozen=True, slots=True)
class Outcome:
    """What an ensure, recover or rollback did."""

    action: str
    commit: str = ""
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.action != "failed"


Builder = Callable[[Plan, Path], runtime.Runtime]


def git(checkout: Path, *words: str, check: bool = True) -> str:
    """Run git in a checkout and return stdout."""
    done = subprocess.run(["git", "-C", str(checkout), *words], capture_output=True, text=True, timeout=60)
    if check and done.returncode:
        raise DeployError(f"git {words[0]} failed: {(done.stderr or done.stdout).strip()[:300]}")
    return done.stdout.strip() if done.returncode == 0 else ""


def resolve_commit(checkout: Path, ref: str = "HEAD") -> str:
    """The full commit a ref names in a checkout."""
    commit = git(checkout, "rev-parse", "--verify", f"{ref}^{{commit}}")
    if not COMMIT.fullmatch(commit):
        raise DeployError(f"{ref} does not name a commit")
    return commit


def _succeeds(checkout: Path, *words: str) -> bool:
    return subprocess.run(["git", "-C", str(checkout), *words], capture_output=True, timeout=60).returncode == 0


def floor_of(checkout: Path, commit: str) -> str:
    """The oldest commit a runtime must descend from, as the source at `commit` names it, or empty."""
    named = git(checkout, "show", f"{commit}:{FLOOR}", check=False).split()
    found = named[0] if named else ""
    return found if COMMIT.fullmatch(found) and _succeeds(checkout, "cat-file", "-e", f"{found}^{{commit}}") else ""


def meets(checkout: Path, floor: str, commit: str) -> bool:
    """Whether a runtime built from `commit` descends from the floor commit."""
    return not floor or _succeeds(checkout, "merge-base", "--is-ancestor", floor, commit)


@contextmanager
def owning(paths: list[Path], *, wait_s: float, note: str = "runtime deploy",
           store: Claims | None = None) -> Iterator[None]:
    """Hold an `install` claim on each path, waiting for a live other owner up to `wait_s` for each."""
    if store is None:
        base = limits.root()
        store = Claims(base, limits.load(base).claim_ttl_s)
    me = Identity(f"runtime-deploy:{os.getpid()}", AGENT)
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


def build(plan: Plan, stage: Path) -> runtime.Runtime:
    """Build the plan's commit into a new immutable runtime tree."""
    return runtime_wheel.build(plan.checkout, plan.commit, stage, timeout=plan.timeout)


def _clean_environment(home: Path, bin_dir: Path) -> dict[str, str]:
    kept = {key: value for key, value in runtime.environment().items()
            if not key.startswith(("ML_STACK_", "MLSTACK_")) and key not in {"CLAUDE_ENV_FILE", "CLAUDE_CONFIG_DIR"}}
    return {**kept, "HOME": str(home), "ML_STACK_HOME": str(home / "state"), "ML_STACK_NO_REAL_KEYSTORE": "1",
            "ML_STACK_NONINTERACTIVE": "1", "ML_STACK_NOTIFY": "off",
            "PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"}


def smoke(chosen: runtime.Runtime, source: Path, stage: Path) -> None:
    """Import the runtime under -I, run its CLI help and the Claude hooks against a temporary state root."""
    runtime.verify(chosen)
    home, bin_dir = stage / "smoke-home", stage / "smoke-bin"
    home.mkdir(parents=True, exist_ok=True)
    bin_dir.mkdir(parents=True, exist_ok=True)
    runtime_launchers.install(bin_dir, chosen)
    env = _clean_environment(home, bin_dir)
    done = subprocess.run([str(bin_dir / "ml-stack-workspace"), "--help"], capture_output=True, text=True,
                          timeout=SMOKE_TIMEOUT, env=env, cwd=home)
    if done.returncode or not done.stdout.strip():
        raise DeployError(f"ml-stack-workspace --help exited {done.returncode}: {done.stderr.strip()[-300:]}")
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


def switch(chosen: runtime.Runtime, plan: Plan) -> None:
    """Rewrite every launcher atomically, then publish the selection."""
    runtime_launchers.install(plan.launchers, chosen)
    runtime.publish(chosen)
    runtime_store.write_state({"checkout": str(plan.checkout), "launchers": str(plan.launchers)})


def _guarded_paths(plan: Plan, commit: str) -> list[Path]:
    root = runtime.directory()
    return [root / commit, plan.launchers, root / "selected.json"]


def build_and_switch(plan: Plan, builder: Builder = build) -> runtime.Runtime:
    """Build, smoke and select the plan's commit under install claims; a failure leaves the selection alone."""
    with owning(_guarded_paths(plan, plan.commit), wait_s=plan.wait_s, note=f"runtime build {plan.commit[:7]}"), \
            tempfile.TemporaryDirectory(prefix="ml-stack-runtime-") as temporary:
        stage = Path(temporary).resolve()
        built = builder(plan, stage)
        try:
            if built.commit != plan.commit:
                raise DeployError(f"built {built.commit}, expected {plan.commit}")
            smoke(built, stage / "source", stage)
            runtime_store.mark_verified(built)
            switch(built, plan)
        except BaseException:
            shutil.rmtree(built.prefix, ignore_errors=True)
            raise
        return built


def healthy(plan: Plan) -> runtime.Runtime | None:
    """The selected runtime when it verifies and descends from the floor, else None."""
    try:
        chosen = runtime.selected()
    except FAILURES:
        return None
    ok = chosen is not None and not runtime_store.rejected(chosen.prefix) \
        and meets(plan.checkout, plan.floor, chosen.commit)
    return chosen if ok else None


def fallback(plan: Plan, avoid: Path | None = None) -> runtime.Runtime | None:
    """The newest verified runtime that still verifies and meets the floor."""
    for candidate in runtime_store.candidates():
        if candidate.prefix == avoid or not meets(plan.checkout, plan.floor, candidate.commit):
            continue
        try:
            return runtime.verify(candidate)
        except FAILURES:
            continue
    return None


def current(plan: Plan) -> bool:
    """Whether the selection file already names this commit's verified tree, without running it."""
    row = runtime_store.selection()
    prefix = Path(str(row.get("prefix", "")))
    return (row.get("commit") == plan.commit and prefix.is_dir() and bool(runtime_store.verified_at(prefix))
            and not runtime_store.rejected(prefix) and meets(plan.checkout, plan.floor, plan.commit))


def _reject_below_floor(plan: Plan, keep: Path) -> None:
    for candidate in runtime_store.candidates():
        if candidate.prefix != keep and not meets(plan.checkout, plan.floor, candidate.commit):
            runtime_store.reject(candidate.prefix, "older than the source checkout's runtime floor")


def _prepare_root() -> Path:
    root = runtime.directory()
    if not root.exists():
        root.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        root.mkdir(mode=0o700)
        runtime.protect(root)
    return root


def ensure(plan: Plan, *, force: bool = False, builder: Builder = build) -> Outcome:
    """Make the plan's commit the selected runtime, recovering a broken selection first."""
    if not force and current(plan):
        return Outcome("current", plan.commit)
    root = _prepare_root()
    if not force and runtime_store.read_state(root).get("held", {}).get("commit") == plan.commit:
        return Outcome("held", plan.commit, runtime_store.read_state(root)["held"].get("reason", ""))
    try:
        with only_one(root / "deploy.lock", wait=False, note="ensure"):
            return _ensured(plan, force, builder)
    except Busy as exc:
        return Outcome("busy", plan.commit, str(exc))


def _ensured(plan: Plan, force: bool, builder: Builder) -> Outcome:
    steps, chosen = [], healthy(plan)
    if chosen is None:
        chosen = fallback(plan)
        if chosen is not None:
            try:
                with owning(_guarded_paths(plan, chosen.commit)[1:], wait_s=plan.wait_s):
                    switch(chosen, plan)
                steps.append(f"selected {chosen.commit[:7]} in place of a broken selection")
            except FAILURES + (DeployError,) as exc:
                steps.append(f"could not select {chosen.commit[:7]}: {exc}")
    if chosen is not None and chosen.commit == plan.commit and not force:
        runtime_store.collect({chosen.prefix})
        return Outcome("recovered" if steps else "current", plan.commit, "; ".join(steps))
    try:
        built = build_and_switch(plan, builder)
    except FAILURES + (DeployError,) as exc:
        runtime_store.write_state({"last_failure": {"commit": plan.commit, "at": time.time(), "detail": str(exc)[-1000:]}})
        return Outcome("recovered" if steps else "failed", plan.commit, "; ".join([*steps, f"build failed: {exc}"])[-1500:])
    runtime_store.write_state({"last_failure": None, "held": None, "last_success": {"commit": plan.commit, "at": time.time()}})
    _reject_below_floor(plan, built.prefix)
    runtime_store.collect({built.prefix, *([chosen.prefix] if chosen else [])})
    return Outcome("switched", plan.commit, "; ".join([*steps, f"selected {built.prefix}"]))


def rollback(plan: Plan, to: str = "") -> Outcome:
    """Select an earlier verified runtime and hold the commit that was selected."""
    root = _prepare_root()
    try:
        with only_one(root / "deploy.lock", wait=False, note="rollback"):
            now = runtime_store.selection(root)
            target = next((c for c in runtime_store.candidates(root)
                           if str(c.prefix) != now.get("prefix") and (not to or c.commit.startswith(to))), None)
            if target is None:
                return Outcome("failed", detail="no earlier verified runtime to select")
            try:
                runtime.verify(target)
                with owning(_guarded_paths(plan, target.commit)[1:], wait_s=plan.wait_s):
                    switch(target, plan)
            except FAILURES + (DeployError,) as exc:
                return Outcome("failed", target.commit, str(exc)[-500:])
            if now.get("commit"):
                runtime_store.write_state({"held": {"commit": now["commit"], "reason": f"rolled back to {target.commit[:7]}"}}, root)
            return Outcome("switched", target.commit, f"selected {target.prefix}")
    except Busy as exc:
        return Outcome("busy", detail=str(exc))
