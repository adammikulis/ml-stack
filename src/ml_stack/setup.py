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
import subprocess
import sys
from dataclasses import dataclass
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
            name="llama-server", good=True, said=binary,
            note=("reads " + ", ".join(sorted(arches)[:6]) + (" ..." if len(arches) > 6 else ""))
            if arches else "could not read which architectures it supports"))
        for wanted in ("qwen4exp",) if arches else ():
            out.append(Finding(
                name=f"architecture {wanted}", good=wanted in arches,
                said="supported" if wanted in arches else "not in this build",
                note="" if wanted in arches else
                     "a release lags master by an architecture or two; serve with "
                     "--binary /path/to/a/master/build, or the server exits saying only "
                     "'unknown model architecture'"))

    try:
        from ml_stack.hub import held

        mine = held()
        out.append(Finding(name="models on this machine", good=bool(mine),
                           said=f"{len(mine)} file(s)",
                           note="" if mine else "nothing found; ml-stack-models find <words>"))
    except Exception:  # noqa: BLE001
        pass
    return out


def _arches(binary: str) -> set[str]:
    """Which model architectures a build reads.

    The names live in libllama, not in the server binary: grepping the executable finds
    nothing and reads as "supports none", which is worse than not looking.
    """
    lib = Path(binary).resolve().parent.parent / "lib"
    found: set[str] = set()
    for where in (lib, Path(binary).resolve().parent):
        for name in sorted(where.glob("libllama*.dylib")) + sorted(where.glob("libllama*.so")):
            try:
                got = subprocess.run(["strings", str(name)], capture_output=True, text=True,
                                     timeout=30)
            except Exception:  # noqa: BLE001
                continue
            for line in got.stdout.splitlines():
                word = line.strip()
                if word and word.islower() and 4 <= len(word) <= 20 and word.isalnum():
                    found.add(word)
            # Keep looking until an *architecture* turns up, not merely until some word
            # does: the first library in the directory is full of ordinary strings and
            # none of the names, and stopping there reported "supports nothing".
    return {w for w in found if any(
        w.startswith(f) for f in ("qwen", "gemma", "llama", "phi", "mistral",
                                  "deepseek", "granite", "olmo", "cohere"))}


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
