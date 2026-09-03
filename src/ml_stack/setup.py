"""A first-run wizard: what this machine can do, and what it should be told to do.

Serving a large model well depends on a handful of machine facts that nothing announces and
everything depends on — how much memory a model may actually use, whether that survives a
reboot, whether the installed llama.cpp reads the architecture you want. Each is a
one-line check, and each is invisible until something fails in a way that does not name it.

So this asks, shows what it found, and offers to fix what it can.

**It never handles a password.** Where a change needs root, `sudo` is run so that it prompts
on the terminal itself — the answer goes from the keyboard to `sudo` and is never seen here,
never passed on a command line, and never stored. Anything that reads a password in order to
pass it along is doing something this deliberately does not.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

__all__ = ["BEHAVIOURS", "Behaviour", "Finding", "ask", "look", "main"]


@dataclass(frozen=True)
class Behaviour:
    """Something the stack does on its own, said out loud.

    Every one of these was once a surprise to somebody: an answer that came back instantly,
    a server that appeared on a port nobody asked for, 90 gigabytes that arrived over the
    network because a name was mistyped. None of them are wrong -- they are what you would
    want -- but a default nobody was told about is indistinguishable from a bug when it
    does something you did not expect.
    """

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
    from ml_stack.hub import _human, room

    out: list[Finding] = []

    total = int(_sysctl("hw.memsize") or 0)
    limit = room()
    if total and limit:
        share = limit / total
        kept = Path("/Library/LaunchDaemons/stack.ml.wired-limit.plist").exists()
        raised = share > 0.78
        out.append(Finding(
            name="memory a model may use",
            good=bool(raised and kept) or not raised,
            said=f"{_human(limit)} of {_human(total)} installed ({share:.0%})",
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
        try:
            from ml_stack.serve import build as build_module

            named = build_module._named_builds()
        except Exception:  # noqa: BLE001
            named = []
        if named:
            described = ", ".join(
                f"{name} ({build_module._manifest_of(link).get('commit', '?')})"
                for name, link in named)
            out.append(Finding(
                name="named builds", good=True, said=described,
                note="beside 'current', not replacing it -- 'ml-stack-serve up --build "
                     "NAME' or $MLSTACK_LLAMA_BUILD=NAME selects one"))

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
        from ml_stack.hub import held

        mine = held()
        out.append(Finding(name="models on this machine", good=bool(mine),
                           said=f"{len(mine)} file(s)",
                           note="" if mine else "nothing found; ml-stack-models find <words>"))
    except Exception:  # noqa: BLE001
        pass

    out.append(_commands_finding())

    from ml_stack.platform import is_windows

    if is_windows():
        out.append(_firewall_finding())
    return out


def _checkout() -> Path:
    """The checkout this package is imported from, else the one `ml-stack-doctor` checks."""
    here = Path(__file__).resolve().parents[2]
    if (here / "pyproject.toml").is_file():
        return here
    from ml_stack.doctor import CHECKOUT

    return CHECKOUT


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
    """Every flag ``ServerSpec`` can emit that this build does not accept, with the nearest.

    Read out of ``--help``, the same cheap way ``_arches`` reads libllama. Empty when the
    build answers everything -- and empty when it printed no help at all, since a build
    that could not be read must not be reported as lacking anything.
    """
    from ml_stack.serve.backend import (LlamaServerBackend, emitted_flags, flags_of,
                                        unknown_flags)

    try:
        return unknown_flags(emitted_flags(LlamaServerBackend(binary=binary)),
                             flags_of(binary))
    except Exception:  # noqa: BLE001
        return []


def _build_label(binary: str) -> str:
    """Which build this is, and how old -- from ``BUILD.json`` for a build
    ``ml-stack-serve build`` installed, from ``--version`` otherwise.

    Read rather than assumed: a build that looks the same in every other way can still be
    six months stale, and that is exactly the fact a release lagging master hides.
    """
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
    """Which model architectures a build reads.

    The names live in libllama, not in the server binary: grepping the executable finds
    nothing and reads as "supports none", which is worse than not looking. ``target`` is
    either the server binary -- its sibling ``lib/`` and its own directory are searched --
    or a directory holding the dylibs directly, the flat shape ``ml-stack-serve build``
    installs into, checked before a new build is trusted enough to switch ``current`` to it.

    A prefix match alone is not precise: measured against a real build, `"phi4"` turned up
    in libllama and reads exactly like an architecture -- but master's own
    ``src/llama-arch.cpp`` defines no ``LLM_ARCH_PHI4`` at all. It names a *chat template*
    (``LLM_CHAT_TEMPLATE_PHI_4`` in ``llama-chat.cpp``) that happens to share the family
    prefix, and a build missing it is not missing an architecture. Pass ``known`` -- the
    real names, from ``build._arches_from_source`` against a source checkout -- to restrict
    the guess to them; without a checkout there is nothing to restrict against, and the
    prefix guess is what there is.
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


def ask(findings: list[Finding], *, yes: bool = False) -> int:
    """Show what was found and offer each fix, one at a time."""
    worst = 0
    for one in findings:
        mark = "ok  " if one.good else "  ! "
        print(f"{mark}{one.name}: {one.said}")
        if one.note:
            print(f"      {one.note}")
        if one.good or not one.fix:
            continue
        worst = 1
        print(f"      fix: {one.fix}")
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
    print("\nwhat happens on its own\n")
    for one in BEHAVIOURS:
        print(f"  {one.name}")
        print(f"      {one.does}")
        if one.why:
            print(f"      why it matters: {one.why}")
        if one.setting:
            print(f"      change it: {one.setting}")
        print()


def main(argv: list[str] | None = None) -> int:
    """``ml-stack-setup`` -- what this machine can do, and what it should be told."""
    ap = argparse.ArgumentParser(
        prog="ml-stack-setup",
        description="Check the handful of machine facts that serving depends on and "
                    "nothing announces.")
    ap.add_argument("--quiet", action="store_true",
                    help="skip the list of things that happen without being asked")
    ap.add_argument("--yes", action="store_true",
                    help="run every offered fix without asking. A fix that needs root will "
                         "still prompt for the password itself")
    args = ap.parse_args(argv)
    print("ml-stack: what this machine can do\n")
    worst = ask(look(), yes=args.yes)
    if not args.quiet:
        explain()
    return worst


if __name__ == "__main__":
    raise SystemExit(main())
