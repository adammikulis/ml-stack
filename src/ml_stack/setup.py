"""Whether this machine is ready: what it can do, and what its checkouts are in.

`look()` is the machine facts serving depends on; `look_checkouts()` is the repositories
and the working state. Each is a `Finding`; `ask()` prints each and offers its fix, never
reading a password -- a fix that needs root runs through `sudo`, which prompts on the
terminal itself.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from ml_stack import home
from ml_stack.log import say
from ml_stack.units import human_bytes

__all__ = ["BEHAVIOURS", "CHECKOUT", "HOOKS", "SPEECH_PROTOCOLS", "STALE_BUILD_DAYS",
           "Behaviour", "Finding", "ahead_of", "ask", "bench_of", "builds_of",
           "doctor_main", "hooks_of", "install_of", "look", "look_checkouts", "main",
           "repositories", "status_of", "worktrees_of"]

CHECKOUT = Path("~/Documents/repos/ml-stack").expanduser()
"""Where the editable install must point: the checkout, not a copy of it."""

STALE_BUILD_DAYS = 14
"""A managed llama.cpp older than this is noted."""

HOOKS = ("pre-commit", "commit-msg")
"""The git hooks every repository here installs, from the directory it ships them in."""

_SHIPPED = ("scripts/hooks", "services/hooks")
_STAMP = re.compile(r"(\d{8}T\d{6})\.log$")


@dataclass(frozen=True)
class Behaviour:
    """Something the stack does on its own, said out loud."""

    name: str
    does: str
    setting: str = ""       # how to change it
    why: str = ""


BEHAVIOURS = (
    Behaviour(
        name="a model that is not here",
        does="downloads, the first time it is asked for. An `hf:owner/repo/file.gguf` "
             "reference is fetched by llama.cpp on the spot -- tens of gigabytes, over the "
             "network, with no prompt",
        setting="ml-stack-models files <repo>  first: it says what is already on this "
                "machine, and what each build would cost",
        why="a mistyped name is a download, not an error"),
    Behaviour(
        name="a vision projector",
        does="comes down with the model when it is asked for by `hf:` reference, and "
             "`--mmproj auto` picks the most precise one shipped beside the weights",
        setting="--mmproj PATH, or leave it out to serve as a text model",
        why="without one a multimodal model ignores an image in silence -- nothing errors, "
            "and the answer is confidently about nothing"),
    Behaviour(
        name="a port that is taken",
        does="another is chosen. A lease refused because something else is on that port "
             "moves to a free one rather than failing",
        setting="ml-stack-serve status --port N  to see what actually ended up where",
        why="it is why a server you started can be somewhere you did not expect"),
    Behaviour(
        name="every layer on the GPU",
        does="`-ngl 99` unless told otherwise, so a model that does not fit fails to load "
             "rather than quietly running on the CPU at a tenth of the speed",
        setting="ServerSpec(n_gpu_layers=N), or --on-cpu PATTERN=BUFFER for part of it",
        why="a partial offload is slower than either extreme and looks like neither"),
    Behaviour(
        name="repeated questions",
        does="are answered from a cache, when a graph, model, prompt and tools are all "
             "unchanged. The model is not called at all",
        setting="graph.cache.forget(store), or do not pass a store",
        why="it is why an answer sometimes returns in no time, which reads as a fault"),
    Behaviour(
        name="sampling",
        does="greedy, whatever the model's card recommends. A card is general advice from "
             "a publisher who does not know the task",
        setting="ml-stack-bench --card to test the card's own settings against yours",
        why="gemma-4 asks for temperature 1.0; on tool-calling that measured 15 points "
            "worse, and made every run unrepeatable"),
    Behaviour(
        name="guessing ahead",
        does="off unless asked for. `--draft auto` finds the head a repository ships; "
             "`--spec ngram-*` needs no second model at all",
        setting="--spec TYPE, --draft auto, --draft-ngl N",
        why="a draft left on the CPU is slower than the model it guesses for"),
)


@dataclass
class Finding:
    """One thing about this machine, and what to do if it is wrong."""

    name: str
    good: bool
    said: str
    fix: str = ""            # a shell line the reader may run
    root: bool = False       # whether that line needs sudo
    note: str = ""


def _sysctl(key: str) -> str:
    try:
        got = subprocess.run(["sysctl", "-n", key], capture_output=True, text=True, timeout=5)
        return got.stdout.strip() if got.returncode == 0 else ""
    except Exception:  # noqa: BLE001
        return ""


def look() -> list[Finding]:
    """Everything worth knowing before serving anything, without changing a thing."""
    from ml_stack.hub import room, total_memory

    out: list[Finding] = []

    total = total_memory()
    limit = room()
    if total and limit:
        share = limit / total
        kept = Path("/Library/LaunchDaemons/stack.ml.wired-limit.plist").exists()
        raised = share > 0.78
        out.append(Finding(
            name="memory a model may use",
            good=bool(raised and kept) or not raised,
            said=f"{human_bytes(limit)} of {human_bytes(total)} installed ({share:.0%})",
            note=("raised above the default, but nothing sets it at boot -- it goes back "
                  "to about 75% on the next restart, and a model that fits today will not"
                  if raised and not kept else
                  "this is the default share; raising it lets a larger model fit"
                  if not raised else "raised, and set at boot"),
            fix="ml-stack-serve memory --persist",
            root=False))

    binary = ""
    try:
        from ml_stack.serve.binary import find_binary

        binary = str(find_binary("llama-server") or "")
    except Exception:  # noqa: BLE001
        pass
    if binary:
        arches = _arches(binary)
        out.append(Finding(
            name="llama-server", good=True,
            said=f"{binary}  ({_build_label(binary)})",
            note=("reads " + ", ".join(sorted(arches)[:6]) + (" ..." if len(arches) > 6 else ""))
            if arches else "could not read which architectures it supports"))
        for wanted in ("qwen4exp",) if arches else ():
            out.append(Finding(
                name=f"architecture {wanted}", good=wanted in arches,
                said="supported" if wanted in arches else "not in this build",
                fix="" if wanted in arches else "ml-stack-serve build",
                note="" if wanted in arches else
                     "a release lags master by an architecture or two; ml-stack-serve "
                     "build gets one from llama.cpp's own master, or serve with --binary "
                     "/path/to/a/master/build -- without either the server exits saying "
                     "only 'unknown model architecture'"))
        lacking = _lacking_flags(binary)
        if lacking:
            # Silent when the build answers every flag; an unknown build (no help text)
            # is given no opinion. A flag it lacks fails at the far end of the load.
            out.append(Finding(
                name="flags this build lacks", good=False,
                said=", ".join(flag for flag, _ in lacking),
                fix="ml-stack-serve build",
                note="; ".join(f"no {flag}" + (f", it has {near}" if near else "")
                               for flag, near in lacking)
                     + ". llama.cpp renames flags between releases, and a flag the build "
                       "does not have exits at the end of the load saying only 'invalid "
                       "argument'. ml-stack-serve up refuses before loading; "
                       "ml-stack-serve build gets the current master, or serve with "
                       "--binary /path/to/another/build"))

    try:
        from ml_stack.fleet.models import caches, sized
        from ml_stack.hub import held

        mine = held()
        where = ", ".join(f"{_tilde(path)} {sized(total)}"
                          for path, files, total in caches(home.home())
                          if files)
        hf_home = os.environ.get("HF_HOME", "")
        out.append(Finding(name="models on this machine", good=bool(mine),
                           said=f"{len(mine)} file(s)" + (f": {where}" if where else ""),
                           note=(f"HF_HOME={hf_home} names the Hub cache" if hf_home else "")
                           if mine else "nothing found; ml-stack-models find <words>"))
    except Exception:  # noqa: BLE001
        pass

    out.extend(_speech_findings())
    out.append(_commands_finding())

    from ml_stack.platform import is_windows

    if is_windows():
        out.append(_firewall_finding())
    return out


SPEECH_PROTOCOLS = (
    ("speech recognition", "asr", "pip install 'ml-stack[speech]'"),
    ("speech synthesis", "tts", "pip install 'ml-stack[speech]'"),
    ("voice activity", "vad", ""),
)


def _speech_findings() -> list[Finding]:
    """One finding per speech protocol: which engines probe here, which `auto` picks, and
    what would add one where none does."""
    from ml_stack.speech.service import providers

    found = providers()
    out: list[Finding] = []
    for label, kind, fix in SPEECH_PROTOCOLS:
        table = found.get(kind, {"auto": None, "providers": []})
        working = [one for one in table["providers"] if one["available"]]
        missing = [one for one in table["providers"] if not one["available"]]
        out.append(Finding(
            name=label,
            good=bool(working),
            said=(", ".join(f"{one['name']}" + (f" ({one['model']})" if one["model"] else "")
                            for one in working) if working
                  else "nothing on this machine"),
            fix="" if working else fix,
            note=(f"ml-stack-speech uses {table['auto']} unless told otherwise"
                  + ("; " + "; ".join(f"{one['name']}: {one['detail']}" for one in missing)
                     if missing else "")
                  if working
                  else "; ".join(f"{one['name']}: {one['detail']}" for one in missing))))
    return out


def _tilde(path: Path) -> str:
    """A path with the home directory written as ``~``."""
    try:
        return "~/" + str(Path(path).relative_to(home.user_home()))
    except ValueError:
        return str(path)


def _checkout() -> Path:
    """The checkout this package is imported from, else the one `ml-stack-doctor` checks."""
    here = Path(__file__).resolve().parents[2]
    return here if (here / "pyproject.toml").is_file() else CHECKOUT


def _scripts() -> list[str]:
    """Every command the package installs: the installed metadata's entry points, and the
    `[project.scripts]` of the checkout's pyproject, together."""
    names: set[str] = set()
    try:
        from importlib.metadata import distribution

        names |= {one.name for one in distribution("ml-stack").entry_points
                  if one.group == "console_scripts"}
    except Exception:  # noqa: BLE001
        pass
    try:
        import tomllib

        table = tomllib.loads((_checkout() / "pyproject.toml").read_text(encoding="utf-8"))
        names |= set(table.get("project", {}).get("scripts", {}))
    except Exception:  # noqa: BLE001
        pass
    return sorted(names)


def _commands_finding() -> Finding:
    """Which of the package's commands are on PATH, and the line that puts the rest there."""
    wanted = _scripts()
    missing = [name for name in wanted if not shutil.which(name)]
    return Finding(
        name="commands on PATH",
        good=not missing,
        said=(f"{len(wanted)} command(s) found" if not missing
              else "not found: " + ", ".join(missing)),
        fix="" if not missing else f"pip install -e {_checkout()} && pyenv rehash",
        note=(", ".join(wanted) if not missing else
              "an entry point added to pyproject.toml is not a command until the package "
              "is reinstalled and the shims rehashed; a queue step that names one dies "
              "on 'command not found'"))


def _firewall_rule_present(name: str) -> bool:
    """Whether Windows Defender Firewall has an inbound rule by this name. ``netsh`` exits
    1 and says 'No rules match' when it does not; anything else is read as absent too,
    since a rule that cannot be confirmed is not one to rely on."""
    try:
        got = subprocess.run(
            ["netsh", "advfirewall", "firewall", "show", "rule", f"name={name}"],
            capture_output=True, text=True, timeout=20)
    except Exception:  # noqa: BLE001
        return False
    return got.returncode == 0 and "No rules match" not in (got.stdout or "")


def _firewall_finding() -> Finding:
    """Windows only: the daemon is unreachable and its beacons unheard until the firewall
    lets TCP 8770 and UDP 8771 in. The fix is one line for an administrator's prompt --
    the one `ml-stack-setup --yes` cannot run for you from an ordinary one."""
    from ml_stack.fleet.discovery import windows_firewall_line, windows_firewall_rules

    rules = windows_firewall_rules()
    present = {name: _firewall_rule_present(name) for name, _ in rules}
    missing = [name for name, ok in present.items() if not ok]
    return Finding(
        name="firewall",
        good=not missing,
        said=("inbound rules present: " + ", ".join(present)) if not missing
             else "no inbound rule for " + ", ".join(missing),
        fix="" if not missing else windows_firewall_line(),
        root=True,
        note="" if not missing else
             "Windows blocks inbound TCP 8770 (the daemon) and UDP 8771 (its beacons) "
             "by default, so other machines see nothing in 'ml-stack-peers ls'. Run the "
             "line in a prompt opened as administrator")


def _lacking_flags(binary: str) -> list[tuple[str, str]]:
    """Every flag ``ServerSpec`` can emit that this build's ``--help`` does not list, with
    the nearest it has; empty when the build answers everything or printed no help at all."""
    from ml_stack.serve.backend import (LlamaServerBackend, emitted_flags, flags_of,
                                        unknown_flags)

    try:
        return unknown_flags(emitted_flags(LlamaServerBackend(binary=binary)),
                             flags_of(binary))
    except Exception:  # noqa: BLE001
        return []


def _build_label(binary: str) -> str:
    """Which build this is, and how old -- from ``BUILD.json`` for a build
    ``ml-stack-serve build`` installed, from ``--version`` otherwise."""
    manifest = Path(binary).resolve().parent / "BUILD.json"
    if manifest.is_file():
        try:
            info = json.loads(manifest.read_text())
        except (OSError, ValueError):
            info = {}
        commit = info.get("commit", "?")
        age = _age(str(info.get("built_at", "")))
        return f"managed build {commit}" + (f", {age} old" if age else "")
    return _version(binary) or "version unknown"


def _version(binary: str) -> str:
    try:
        from ml_stack.serve.binary import child_env

        got = subprocess.run([binary, "--version"], capture_output=True, text=True,
                             timeout=10, env=child_env(binary))
    except Exception:  # noqa: BLE001
        return ""
    text = (got.stdout + got.stderr).strip()
    return text.splitlines()[0] if text else ""


def _age(built_at: str) -> str:
    if not built_at:
        return ""
    try:
        then = datetime.fromisoformat(built_at)
    except ValueError:
        return ""
    if then.tzinfo is None:
        then = then.replace(tzinfo=timezone.utc)
    delta = datetime.now(timezone.utc) - then
    if delta.days >= 1:
        return f"{delta.days}d"
    if delta.seconds >= 3600:
        return f"{delta.seconds // 3600}h"
    return f"{max(1, delta.seconds // 60)}m"


def _arches(target: str | Path, *, known: set[str] | None = None) -> set[str]:
    """Which model architectures a build reads, from the names in libllama.

    ``target`` is the server binary (its sibling ``lib/`` and its own directory are
    searched) or a directory holding the dylibs directly. A prefix match alone is
    imprecise -- ``"phi4"`` names a chat template, not an architecture -- so ``known``
    (the real names, from a source checkout) restricts the guess when it is given.
    """
    path = Path(target)
    if path.is_dir():
        dirs: tuple[Path, ...] = (path,)
    else:
        dirs = (path.resolve().parent.parent / "lib", path.resolve().parent)
    found: set[str] = set()
    for where in dirs:
        for name in sorted(where.glob("libllama*.dylib")) + sorted(where.glob("libllama*.so")):
            try:
                got = subprocess.run(["strings", str(name)], capture_output=True, text=True,
                                     timeout=30)
            except Exception:  # noqa: BLE001
                continue
            for line in got.stdout.splitlines():
                word = line.strip()
                if word and word.islower() and 4 <= len(word) <= 20 and word.replace("-", "").isalnum():
                    found.add(word)
            # Keep looking until an *architecture* turns up, not merely until some word
            # does: the first library in the directory is full of ordinary strings and
            # none of the names, and stopping there reported "supports nothing".
    guessed = {w for w in found if any(
        w.startswith(f) for f in ("qwen", "gemma", "llama", "phi", "mistral",
                                  "deepseek", "granite", "olmo", "cohere", "gpt", "glm",
                                  "nemotron", "falcon", "mamba", "rwkv", "exaone"))}
    return guessed & known if known is not None else guessed


# -- the checkouts -----------------------------------------------------------------------

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
    """The checkouts to look at: those named, or the current directory and the two that
    are always here when they exist. Each once, whatever path it was reached by."""
    wanted = [Path(p).expanduser() for p in given] if given else [
        Path.cwd(), Path("~/ai_ceo").expanduser(), CHECKOUT]
    out: list[Path] = []
    for one in wanted:
        if (given or one.is_dir()) and one.resolve() not in [p.resolve() for p in out]:
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
        return f"{shipped}/" in os.readlink(hook)
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
    hooks = Path(common) if os.path.isabs(common) else repo / common
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
            [str(python), "-c", "import ml_stack, ml_stack.graph.ask as a; print(a.__file__)"],
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


def _dead_lock(home: Path) -> Finding | None:
    """A ``measuring.json`` whose pid has gone: the last measurement finished or died,
    and nothing removed its record."""
    from ml_stack.serve.process import pid_exists

    where = home / "measuring.json"
    if not where.exists():
        return None
    try:
        held = json.loads(where.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        held = {}
    pid = held.get("pid") if isinstance(held, dict) else None
    if pid_exists(pid):
        return Finding(name="bench: measuring", good=True,
                       said=f"pid {pid} since {held.get('started', '?')}: "
                            f"ml-stack-bench {' '.join(held.get('argv') or ())}")
    return Finding(
        name="bench: measuring", good=False,
        said=f"stale lock: {where} names pid {pid}, which is not running",
        fix=f"rm -f {where}",
        note=f"the measurement that started {held.get('started', '?')} has ended; "
             "ml-stack-bench status reads this file and says so, but it is the record of "
             "something that is over")


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

    lock = _dead_lock(home)
    if lock is not None:
        out.append(lock)
    live_log = ""
    if lock is not None and lock.good:
        try:
            live_log = str(json.loads((home / "measuring.json").read_text()).get("log", ""))
        except (OSError, ValueError, AttributeError):
            live_log = ""

    logs = sorted((home / "logs").glob("*.log"), key=_started) if (home / "logs").is_dir() else []
    newest, count = _newest_run_at(store)
    died = [lg for lg in logs if _started(lg) > newest and str(lg) != live_log]
    if died:
        out.append(Finding(
            name="bench: logs", good=False,
            said=f"{len(died)} log(s) newer than the newest kept run, with no run kept: "
                 + ", ".join(lg.name for lg in died[-3:]),
            note="a measurement that started and kept nothing died before saving -- its "
                 f"last lines say where; ml-stack-bench tail reads the latest"))
    elif logs or count:
        out.append(Finding(name="bench: logs", good=True,
                           said=f"{count} run(s) kept; every log has one"))
    return out


def _answers_help(binary: Path) -> bool:
    from ml_stack.serve.binary import child_env

    try:
        got = subprocess.run([str(binary), "--help"], capture_output=True, text=True,
                             timeout=30, env=child_env(binary))
    except Exception:  # noqa: BLE001
        return False
    return got.returncode == 0 and "usage" in (got.stdout + got.stderr).lower()


def _days_old(build_dir: Path) -> int | None:
    from ml_stack.serve.build import _manifest_of

    built = str(_manifest_of(build_dir).get("built_at", ""))
    try:
        then = datetime.fromisoformat(built)
    except ValueError:
        return None
    if then.tzinfo is None:
        then = then.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - then).days


def builds_of(current: Path, named: Path, *, stale_days: int = STALE_BUILD_DAYS) -> list[Finding]:
    """``current`` answers ``--help`` and is not stale; the named builds beside it."""
    from ml_stack.serve.build import _manifest_of, _server_name

    out: list[Finding] = []
    server = current / _server_name()
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
        commit = _manifest_of(current).get("commit", "?")
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
                if (p.is_symlink() or p.is_dir()) and (p / _server_name()).exists()]
        if kept:
            out.append(Finding(
                name="llama.cpp: named builds", good=True,
                said=", ".join(f"{n} ({_manifest_of(p).get('commit', '?')})" for n, p in kept),
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
    from ml_stack.serve.binary import MANAGED_CURRENT, MANAGED_NAMED

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
    out.extend(builds_of(MANAGED_CURRENT if current is None else current,
                         MANAGED_NAMED if named is None else named))
    return out


def ask(findings: list[Finding], *, yes: bool = False) -> int:
    """Show what was found and offer each fix, one at a time."""
    worst = 0
    for one in findings:
        mark = "ok  " if one.good else "  ! "
        say(f"{mark}{one.name}: {one.said}")
        if one.note:
            say(f"      {one.note}")
        if one.good or not one.fix:
            continue
        worst = 1
        say(f"      fix: {one.fix}")
        if not yes and not sys.stdin.isatty():
            continue
        answer = "y" if yes else input("      run it now? [y/N] ").strip().lower()
        if answer != "y":
            continue
        # sudo is run so that it prompts on this terminal. The password goes from the
        # keyboard to sudo; nothing here reads it, passes it, or keeps it.
        subprocess.run(one.fix, shell=True, check=False)
    return worst


def explain() -> None:
    """Print what the stack does without being asked."""
    say("\nwhat happens on its own\n")
    for one in BEHAVIOURS:
        say(f"  {one.name}")
        say(f"      {one.does}")
        if one.why:
            say(f"      why it matters: {one.why}")
        if one.setting:
            say(f"      change it: {one.setting}")
        say()


def main(argv: list[str] | None = None) -> int:
    """``ml-stack-setup`` -- what this machine can do, and its checkouts with
    ``--checkouts``. Exit 1 when any finding is not good."""
    ap = argparse.ArgumentParser(
        prog="ml-stack-setup",
        description="Check the handful of machine facts that serving depends on and "
                    "nothing announces; --checkouts adds the repositories and working "
                    "state ml-stack-doctor checks on their own.")
    ap.add_argument("--quiet", action="store_true",
                    help="skip the list of things that happen without being asked")
    ap.add_argument("--checkouts", action="store_true",
                    help="also check the repositories and the working state")
    ap.add_argument("--repo", action="append", metavar="PATH",
                    help="a checkout to look at with --checkouts; may repeat. Default: "
                         "the current directory, ~/ai_ceo and the ml-stack checkout, "
                         "those that exist")
    ap.add_argument("--bench-home", metavar="PATH",
                    help="the bench's home to check with --checkouts (default: where "
                         "ml-stack-bench keeps its store)")
    ap.add_argument("--yes", action="store_true",
                    help="run every offered fix without asking. A fix that needs root will "
                         "still prompt for the password itself")
    args = ap.parse_args(argv)
    say("ml-stack: what this machine can do\n")
    findings = look()
    if args.checkouts:
        findings = findings + look_checkouts(
            repositories(args.repo) if args.repo else None,
            bench_home=Path(args.bench_home).expanduser() if args.bench_home else None)
    ask(findings, yes=args.yes)
    if not args.quiet:
        explain()
    return 0 if all(f.good for f in findings) else 1


def doctor_main(argv: list[str] | None = None) -> int:
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
                         "~/ai_ceo and the ml-stack checkout, those that exist")
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
