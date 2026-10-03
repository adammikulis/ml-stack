"""``llama-cpp update``: build an upstream commit, smoke-test it, and only then make it active."""

from __future__ import annotations

import contextlib
import os
import platform
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path

from ml_stack import files, net, sandbox
from ml_stack.httpguard import Refused
from ml_stack.sandbox import policies
from ml_stack.serve import llamacpp_compile, llamacpp_smoke, llamacpp_state, llamacpp_trust
from ml_stack.serve.build_paths import BuildFailed, builds_dir, root
from ml_stack.serve.build_platform import now_iso, server_name
from ml_stack.serve.llamacpp_upstream import Unresolved, Upstream, latest, resolve, tag_number

__all__ = ["Env", "Outcome", "build_name", "update"]

SHORT = 9


@dataclass(frozen=True, slots=True)
class Outcome:
    """What `update` did: ``built`` (a new active build), ``current`` (already built) or ``failed``
    (the smoke test did not pass; ``kept`` is where the build was kept)."""

    status: str
    build: str
    detail: str = ""
    kept: Path | None = None


def build_name(number: int | None, sha: str) -> str:
    return f"b{number}-{sha[:SHORT]}" if number is not None else f"dev-{sha[:SHORT]}"


def _number(tag: str, sha: str, upstream: Upstream, pipeline: net.Pipeline) -> int | None:
    """The llama.cpp build number of ``sha``: its own tag's, else the newest build tag's
    shifted by the commits between them, as GitHub counts them."""
    if (known := tag_number(tag)) is not None:
        return known
    newest = latest(upstream, pipeline)
    base = tag_number(newest.tag)
    if base is None:
        return None
    if newest.tagged == sha:
        return base
    try:
        got = pipeline.json(f"{upstream.api_base}/repos/{upstream.repo}/compare/{newest.tag}...{sha}",
                            net.Ask(purpose="llama.cpp upstream", max_bytes=4 << 20))
    except net.NeedsApproval:
        raise
    except (OSError, ValueError, Refused):
        return None
    if not isinstance(got, dict):
        return None
    if got.get("status") == "ahead":
        return base + int(got.get("ahead_by") or 0)
    if got.get("status") == "behind":
        return base - int(got.get("behind_by") or 0)
    return base if got.get("status") == "identical" else None


def _version(binary: Path) -> str:
    """The first line ``binary --version`` prints, run under a policy with no network or GPU."""
    policy = policies.model_server(binary, binary.parent, gpu=False)
    policy = replace(policy, exec=(*policy.exec, *policies.SYSTEM_EXEC),
                     limits=replace(policy.limits, wall_seconds=60.0))
    done = sandbox.run([str(binary), "--version"], policy, cwd=str(binary.parent))
    lines = [ln for ln in ((done.stdout or "") + (done.stderr or "")).splitlines() if "version" in ln]
    return lines[0].strip() if lines else ""


@dataclass(slots=True)
class Env:
    """What an update runs with: where upstream is, the net pipeline, the toolchain (found
    when unset), compile jobs, the smoke test and where progress goes."""

    upstream: Upstream = field(default_factory=Upstream)
    pipeline: net.Pipeline = field(default_factory=net.default)
    tools: llamacpp_compile.Toolchain | None = None
    jobs: int = 0
    smoke: Callable[..., llamacpp_smoke.Result] = llamacpp_smoke.run
    say: Callable[[str], None] = lambda _text: None


def update(ref: str = "master", env: Env | None = None, *, force: bool = False,
           model: Path | None = None) -> Outcome:
    """Build ``ref`` (a branch, tag or commit; default master) from upstream and make it
    active when it passes the smoke test. `BuildFailed` for a missing tool, a pinned track, a
    ref upstream does not know or a build that does not compile; `net.NeedsApproval` when the
    source host is not approved. A build that fails the smoke test leaves the active build
    where it is and is kept under ``failed/``."""
    env = env or Env()
    if pinned := llamacpp_state.load()["pinned"]:
        raise BuildFailed(f"the track is pinned to {pinned}; `llama-cpp pin --off` resumes tracking")
    tools = env.tools or llamacpp_compile.toolchain()
    try:
        sha, tag = resolve(ref, env.upstream, env.pipeline)
    except Unresolved as exc:
        raise BuildFailed(str(exc)) from exc
    number = _number(tag, sha, env.upstream, env.pipeline)
    name = build_name(number, sha)
    existing = next((b for b in llamacpp_state.builds() if b.name == name and b.good), None)
    if existing is not None and not force:
        if (here := llamacpp_state.active()) is None or here.name != name:
            llamacpp_state.activate(existing)
        return Outcome("current", name, f"{name} is already built and active")
    work = root() / "work" / name
    shutil.rmtree(work, ignore_errors=True)
    (work / "out").mkdir(parents=True)
    try:
        info = _stage(env, tools, Stamp(ref, tag, sha, number, name), work)
        result = env.smoke(work / "out" / server_name(), model, say=env.say)
        info["smoke"] = {**result.as_dict(), "at": now_iso()}
        files.write_json(work / "out" / "BUILD.json", info)
        if not result.passed:
            failed = llamacpp_state.failed_dir()
            failed.mkdir(parents=True, exist_ok=True)
            kept = files.promote(work / "out", failed / f"{name}-{int(time.time())}")
            return Outcome("failed", name, _why(result), kept)
        _install(work / "out", name, str(info["sha256"]))
        return Outcome("built", name, str(info["version"]))
    finally:
        shutil.rmtree(work, ignore_errors=True)
        with contextlib.suppress(OSError):
            work.parent.rmdir()


@dataclass(frozen=True, slots=True)
class Stamp:
    """Which commit is being built and what it is called."""

    ref: str
    tag: str
    sha: str
    number: int | None
    name: str


def _stage(env: Env, tools: llamacpp_compile.Toolchain, stamp: Stamp, work: Path) -> dict:
    """Fetch and compile the commit into ``work/out`` and write its ``BUILD.json``."""
    env.say(f"fetching {stamp.sha} from {env.upstream.git_url}")
    llamacpp_compile.checkout(env.upstream, stamp.sha, work / "src", env.pipeline)
    jobs = env.jobs or _jobs()
    env.say(f"building {stamp.name} ({jobs} jobs); this takes several minutes")
    (work / "scratch").mkdir()
    job = llamacpp_compile.Job(work / "src", work / "scratch", stamp.number, stamp.sha, jobs, tools)
    built = llamacpp_compile.compile_source(job, env.say)
    binary = work / "out" / server_name()
    shutil.copy2(built, binary)
    info = {"schema": 1, "track": True, "source": "track", "commit": stamp.sha, "short": stamp.sha[:SHORT],
            "build": stamp.number, "ref": stamp.ref, "tag": stamp.tag, "name": stamp.name,
            "built_at": now_iso(), "sha256": files.sha256_file(binary), "version": _version(binary) or "unknown",
            "patches": [], "platform": f"{platform.system().lower()}-{platform.machine()}",
            "toolchain": {"cmake": tools.cmake, "cc": tools.cc, "cxx": tools.cxx}}
    if not env.upstream.mainline:
        info["repo"] = env.upstream.repo
    files.write_json(work / "out" / "BUILD.json", info)
    env.say("smoke test")
    return info


def _install(staged: Path, name: str, digest: str) -> None:
    """Move a build that passed into ``builds/``, check it is the bytes that were tested, pin
    it and make it active."""
    dest = builds_dir() / name
    builds_dir().mkdir(parents=True, exist_ok=True)
    shutil.rmtree(dest, ignore_errors=True)
    files.promote(staged, dest)
    if files.sha256_file(dest / server_name()) != digest:
        raise BuildFailed(f"{dest / server_name()} changed while it was being installed")
    llamacpp_trust.pin(dest / server_name(), digest)
    llamacpp_state.activate(next(b for b in llamacpp_state.builds() if b.name == name))


def _jobs() -> int:
    return max(2, (os.cpu_count() or 4) - 2)


def _why(result: llamacpp_smoke.Result) -> str:
    bad = [f"{name}: {detail}" for name, ok, detail in result.checks if not ok]
    return "; ".join(bad) or "the smoke test ran no checks"
