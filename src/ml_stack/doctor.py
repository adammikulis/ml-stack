"""What the checkouts and the working state are in.

`look_checkouts()` reads the repositories (hooks, working tree, branch, worktrees, the
editable install), the bench store and the managed llama.cpp builds. Each is a `Finding`.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from ml_stack.bench.underway import measuring, measuring_file
from ml_stack.checks import CHECKOUT, Finding, ask
from ml_stack.log import say
from ml_stack.serve.binary import child_env, managed_current, managed_named
from ml_stack.serve.build_platform import server_name
from ml_stack.serve.build_report import manifest_of

__all__ = ["HOOKS", "STALE_BUILD_DAYS", "ahead_of", "bench_of", "builds_of",
           "hooks_of", "install_of", "look_checkouts", "main", "repositories", "status_of",
           "worktrees_of"]

STALE_BUILD_DAYS = 14
"""A managed llama.cpp older than this is noted."""

HOOKS = ("pre-commit", "commit-msg", "pre-push")
"""The git hooks every repository here installs, from the directory it ships them in."""

_SHIPPED = ("scripts/hooks", "services/hooks")
_STAMP = re.compile(r"(\d{8}T\d{6})\.log$")


def _git(repo: Path, *args: str, strip: bool = True) -> str:
    """``git args`` in ``repo``; '' when git refuses, since every caller treats that as
    'no answer' rather than an error. ``strip=False`` keeps the leading space a porcelain
    status line carries -- stripped, ``" M note.txt"`` reads as ``"ote.txt"``."""
    try:
        got = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True,
                             timeout=30)
    except Exception:  # noqa: BLE001
        return ""
    if got.returncode != 0:
        return ""
    return got.stdout.strip() if strip else got.stdout


def repositories(given: list[str] | None = None) -> list[Path]:
    """The checkouts to look at: those named, or those of the current directory and the
    two known ones that are git repositories. Each once, whatever path it was reached by."""
    wanted = [Path(p).expanduser().absolute() for p in given] if given else [
        Path.cwd(), Path("~/ai_ceo").expanduser(), CHECKOUT]
    out: list[Path] = []
    for one in wanted:
        if not given and not (one.is_dir() and _git(one, "rev-parse", "--show-toplevel")):
            continue
        if one.resolve() not in [p.resolve() for p in out]:
            out.append(one)
    return out


def _shipped_hooks(repo: Path) -> str:
    """Which directory this repository ships its hooks in, or '' when it ships none."""
    return next((d for d in _SHIPPED if (repo / d).is_dir()), "")


def _installed(hook: Path, shipped: str) -> bool:
    """A hook is installed when it *is* the shipped script -- a symlink into ``shipped``
    -- or execs it: the untracked wrapper a machine writes to hand the script its own
    database is as installed as the link is."""
    if not hook.exists() and not hook.is_symlink():
        return False
    if hook.is_symlink():
        return f"{shipped}/" in str(hook.readlink())
    try:
        return f"{shipped}/" in hook.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False


def hooks_of(repo: Path) -> Finding | None:
    """Whether the hooks the repository ships are the ones git will run."""
    shipped = _shipped_hooks(repo)
    if not shipped:
        return None
    common = _git(repo, "rev-parse", "--git-common-dir")
    hooks = Path(common) if Path(common).is_absolute() else repo / common
    missing = [h for h in HOOKS if not _installed(hooks / "hooks" / h, shipped)]
    installer = repo / "scripts" / "install-hooks.sh"
    if installer.is_file():
        fix = f"cd {repo} && sh scripts/install-hooks.sh"
    else:
        fix = f"cd {repo} && " + " && ".join(
            f"ln -sf ../../{shipped}/{h} .git/hooks/{h}" for h in missing)
    return Finding(
        name=f"{repo.name}: hooks", good=not missing,
        said=("installed, from " + shipped) if not missing else
             "not installed: " + ", ".join(missing),
        fix="" if not missing else fix,
        note="" if not missing else
             f"git runs nothing from {shipped}/ until they are; a real name or a scrape "
             "goes into a commit unrefused")


def status_of(repo: Path) -> Finding:
    """Clean, or what is not."""
    lines = [ln for ln in _git(repo, "status", "--porcelain", strip=False).splitlines()
             if ln.strip()]
    paths = [ln[3:].strip() for ln in lines]
    shown = ", ".join(paths[:6]) + (f" ... ({len(paths)} in all)" if len(paths) > 6 else "")
    return Finding(name=f"{repo.name}: working tree", good=not lines,
                   said="clean" if not lines else f"{len(lines)} not committed: {shown}")


def ahead_of(repo: Path) -> Finding:
    """How far the branch is past its upstream. A note: nothing here pushes."""
    branch = _git(repo, "rev-parse", "--abbrev-ref", "HEAD") or "?"
    upstream = _git(repo, "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}")
    if not upstream:
        return Finding(name=f"{repo.name}: branch", good=True,
                       said=f"{branch}, no upstream to compare with")
    count = _git(repo, "rev-list", "--count", "@{u}..HEAD")
    ahead = int(count) if count.isdigit() else 0
    return Finding(
        name=f"{repo.name}: branch", good=True,
        said=f"{branch} is {ahead} commit(s) ahead of {upstream}" if ahead else
             f"{branch} is in step with {upstream}",
        note="" if not ahead else "not pushed, and not pushed by this" + (
             " -- a push to ml-stack cuts a release" if repo.name == "ml-stack" else ""))


def worktrees_of(repo: Path) -> list[Finding]:
    """Every other worktree of this repository, and how far behind HEAD its pin is."""
    text = _git(repo, "worktree", "list", "--porcelain")
    if not text:
        return []
    blocks = [b for b in text.split("\n\n") if b.strip()]
    out: list[Finding] = []
    for block in blocks[1:]:              # the first is this checkout itself
        fields = dict(ln.split(" ", 1) for ln in block.splitlines() if " " in ln)
        where, commit = fields.get("worktree", "?"), fields.get("HEAD", "")
        behind = _git(repo, "rev-list", "--count", f"{commit}..HEAD") if commit else ""
        n = int(behind) if behind.isdigit() else 0
        out.append(Finding(
            name=f"{repo.name}: worktree {Path(where).name}", good=True,
            said=f"stale: holds {commit[:7]}, {n} commit(s) behind HEAD" if n else
                 f"holds {commit[:7]}, at HEAD",
            note="" if not n else
                 f"pinned on purpose or forgotten: whatever it measures is measured "
                 f"without the {n} commit(s) since, at {where}"))
    return out


def install_of(repo: Path, *, checkout: Path | None = None,
               python: Path | None = None) -> Finding | None:
    """Where ``import ml_stack`` lands for this repository's interpreter: good only when
    that is under ``checkout``, since a resolve into site-packages is a copy."""
    checkout = CHECKOUT if checkout is None else checkout
    if python is None:
        venv = repo / ".venv" / "bin" / "python"
        python = venv if venv.exists() else Path(sys.executable)
    try:
        got = subprocess.run(
            [str(python), "-c", "import ml_stack, ml_stack.graph.conversation as a; print(a.__file__)"],
            capture_output=True, text=True, timeout=60)
    except Exception as exc:  # noqa: BLE001
        got = None
        error = str(exc)
    else:
        error = (got.stderr.strip().splitlines() or [""])[-1]
    fix = f"{python} -m pip install -e {checkout}"
    if got is None or got.returncode != 0:
        return Finding(name=f"{repo.name}: editable install", good=False,
                       said=f"{python}: import ml_stack fails -- {error or 'no output'}",
                       fix=fix)
    found = Path(got.stdout.strip())
    try:
        under = found.resolve().is_relative_to(checkout.resolve())
    except (OSError, ValueError):
        under = False
    return Finding(
        name=f"{repo.name}: editable install", good=under,
        said=str(found),
        fix="" if under else fix,
        note="" if under else
             f"{python} imports a copy, not {checkout}: what is edited there is not "
             "what runs here")


def _measuring(home: Path) -> Finding | None:
    """What is measuring on this machine, or what ran last, or None when neither.

    A record whose run has ended is what `ml-stack-bench status` names the last run from
    and what `tail` finds its log through, so it is reported, never offered for deletion.
    """
    held = measuring(home)
    if held is not None:
        return Finding(name="bench: measuring", good=True,
                       said=f"pid {held.get('pid')} since {held.get('started') or '?'}: "
                            f"ml-stack-bench {' '.join(held.get('argv') or ())}".rstrip())
    try:
        last = json.loads(measuring_file(home).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(last, dict):
        return None
    return Finding(
        name="bench: measuring", good=True,
        said="nothing is measuring",
        note=f"the last was ml-stack-bench {' '.join(last.get('argv') or ())}, started "
             f"{last.get('started') or '?'}"
             + (f" and ended {last['ended']}" if last.get("ended") else
                f", and pid {last.get('pid')} is gone"))


def _newest_run_at(store: Path) -> tuple[float, int]:
    """When the newest kept run was written, as epoch seconds, and how many there are."""
    from ml_stack.bench import runs

    kept = runs(store) if store.exists() else []
    at = 0.0
    for one in kept:
        try:
            at = max(at, time.mktime(time.strptime(str(one.get("at", "")), "%Y-%m-%dT%H:%M:%S")))
        except ValueError:
            continue
    return at, len(kept)


def _started(log: Path) -> float:
    """When a log began, from the stamp ``detach`` puts in its name; its mtime otherwise."""
    found = _STAMP.search(log.name)
    if found:
        try:
            return time.mktime(time.strptime(found.group(1), "%Y%m%dT%H%M%S"))
        except ValueError:
            pass
    return log.stat().st_mtime


def bench_of(home: Path) -> list[Finding]:
    """The bench store under ``home``: empty runs, a dead lock, a log with no run."""
    out: list[Finding] = []
    if not home.is_dir():
        return out
    store = home / "runs.ladybug"
    try:
        from ml_stack.bench import empties

        hollow = empties(store)
    except Exception as exc:  # noqa: BLE001
        out.append(Finding(name="bench: runs", good=False, said=f"{store} did not open: {exc}"))
        return out
    if hollow:
        out.append(Finding(
            name="bench: runs", good=False,
            said=f"{len(hollow)} run(s) read back as nothing: "
                 + ", ".join(hollow[:3]) + (" ..." if len(hollow) > 3 else ""),
            fix=f"ml-stack-bench forget --empty --kept {store}",
            note="each was a measurement that saved a row of dashes; the table skips them "
                 "and says nothing about why"))

    held = measuring(home)
    running = _measuring(home)
    if running is not None:
        out.append(running)
    live_log = str((held or {}).get("log") or "")

    logs = sorted((home / "logs").glob("*.log"), key=_started) if (home / "logs").is_dir() else []
    newest, count = _newest_run_at(store)
    died = [lg for lg in logs if _started(lg) > newest and str(lg) != live_log]
    if died:
        out.append(Finding(
            name="bench: logs", good=False,
            said=f"{len(died)} log(s) newer than the newest kept run, with no run kept: "
                 + ", ".join(lg.name for lg in died[-3:]),
            note="a measurement that started and kept nothing died before saving -- its "
                 "last lines say where; ml-stack-bench tail reads the latest"))
    elif logs or count:
        out.append(Finding(name="bench: logs", good=True,
                           said=f"{count} run(s) kept; every log has one"))
    return out


def _answers_help(binary: Path) -> bool:
    try:
        got = subprocess.run([str(binary), "--help"], capture_output=True, text=True,
                             timeout=30, env=child_env(binary))
    except Exception:  # noqa: BLE001
        return False
    return got.returncode == 0 and "usage" in (got.stdout + got.stderr).lower()


def _days_old(build_dir: Path) -> int | None:
    built = str(manifest_of(build_dir).get("built_at", ""))
    try:
        then = datetime.fromisoformat(built)
    except ValueError:
        return None
    if then.tzinfo is None:
        then = then.replace(tzinfo=UTC)
    return (datetime.now(UTC) - then).days


def builds_of(current: Path, named: Path, *, stale_days: int = STALE_BUILD_DAYS) -> list[Finding]:
    """``current`` answers ``--help`` and is not stale; the named builds beside it."""
    out: list[Finding] = []
    server = current / server_name()
    if not (current.is_symlink() or current.exists()):
        out.append(Finding(name="llama.cpp: current", good=False,
                           said="not built yet", fix="ml-stack-serve build",
                           note="serving falls back to whatever llama-server is on PATH, "
                                "which lags master by an architecture or two"))
    elif not _answers_help(server):
        out.append(Finding(name="llama.cpp: current", good=False,
                           said=f"{server} does not answer --help",
                           fix="ml-stack-serve build",
                           note="the link is there and the binary behind it is not, or "
                                "cannot load: it will fail the same way at serve time"))
    else:
        commit = manifest_of(current).get("commit", "?")
        age = _days_old(current)
        stale = age is not None and age >= stale_days
        out.append(Finding(
            name="llama.cpp: current", good=not stale,
            said=f"{commit}, " + (f"{age}d old" if age is not None else "age unknown")
                 + ", answers --help",
            fix="ml-stack-serve build" if stale else "",
            note="" if not stale else
                 f"older than {stale_days} days; master has gained an architecture or two "
                 "since, and a model in one of them exits saying only 'unknown model "
                 "architecture'"))
    if named.is_dir():
        kept = [(p.name, p) for p in sorted(named.iterdir())
                if (p.is_symlink() or p.is_dir()) and (p / server_name()).exists()]
        if kept:
            out.append(Finding(
                name="llama.cpp: named builds", good=True,
                said=", ".join(f"{n} ({manifest_of(p).get('commit', '?')})" for n, p in kept),
                note="beside current, not replacing it -- ml-stack-serve up --build NAME "
                     "selects one"))
    return out


def look_checkouts(repos: list[Path] | None = None, *, bench_home: Path | None = None,
                   current: Path | None = None, named: Path | None = None,
                   checkout: Path | None = None) -> list[Finding]:
    """The repositories and the working state, without changing a thing: hooks, the
    working tree, the branch, worktrees, the editable install, the bench store, and the
    managed llama.cpp builds."""
    from ml_stack.bench import home_dir

    out: list[Finding] = []
    seen_python: set[Path] = set()
    for repo in repositories(None) if repos is None else repos:
        if not _git(repo, "rev-parse", "--show-toplevel"):
            out.append(Finding(name=f"{repo.name}: repository", good=False,
                               said=f"{repo} is not a git repository"))
            continue
        hooks = hooks_of(repo)
        if hooks is not None:
            out.append(hooks)
        out.append(status_of(repo))
        out.append(ahead_of(repo))
        out.extend(worktrees_of(repo))
        venv = repo / ".venv" / "bin" / "python"
        python = venv if venv.exists() else Path(sys.executable)
        # By path, not resolved: a venv's python is a link to the same interpreter as
        # sys.executable and imports from a different site-packages all the same.
        if python not in seen_python:
            seen_python.add(python)
            found = install_of(repo, checkout=checkout, python=python)
            if found is not None:
                out.append(found)
    out.extend(bench_of(home_dir() if bench_home is None else bench_home))
    out.extend(builds_of(managed_current() if current is None else current,
                         managed_named() if named is None else named))
    return out


def main(argv: list[str] | None = None) -> int:
    """``ml-stack-doctor`` -- the repositories and the working state, at the start of a
    session. Exit 0 when every finding is good, 1 otherwise."""
    ap = argparse.ArgumentParser(
        prog="ml-stack-doctor",
        description="Check what ml-stack-setup does not: the checkouts (hooks, working "
                    "tree, branch, worktrees, the editable install), the bench store "
                    "(empty runs, a dead lock, a log with no run) and the managed "
                    "llama.cpp. Offers a fix for what has one; never pushes.")
    ap.add_argument("--repo", action="append", metavar="PATH",
                    help="a checkout to look at; may repeat. Default: the current directory, "
                         "~/ai_ceo and the ml-stack checkout, those that are git "
                         "repositories")
    ap.add_argument("--bench-home", metavar="PATH",
                    help="the bench's home (default: where ml-stack-bench keeps its store)")
    ap.add_argument("--yes", action="store_true",
                    help="run every offered fix without asking")
    args = ap.parse_args(argv)
    say("ml-stack: the repositories and the working state\n")
    findings = look_checkouts(
        repositories(args.repo) if args.repo else None,
        bench_home=Path(args.bench_home).expanduser() if args.bench_home else None)
    ask(findings, yes=args.yes)
    return 0 if all(f.good for f in findings) else 1


if __name__ == "__main__":
    raise SystemExit(main())
