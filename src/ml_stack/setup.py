"""What this machine can do, and what it does without being asked.

`look()` is the machine facts serving depends on: the memory a model may use, the llama
binary and what it reads, the extras that are installed, the commands on PATH, speech.
Each is a `Finding`. `BEHAVIOURS` is what the stack does on its own.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from ml_stack import home
from ml_stack.checks import Finding, ask, checkout, tilde
from ml_stack.installed import standard
from ml_stack.log import say
from ml_stack.platform import is_windows
from ml_stack.units import human_bytes

__all__ = ["BEHAVIOURS", "SPEECH_PROTOCOLS", "Behaviour", "explain", "look", "main"]


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

    from ml_stack.serve.binary import find_binary

    binary = str(find_binary("llama-server") or "")
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
        from ml_stack.hub import on_disk

        mine = on_disk()
        where = ", ".join(f"{tilde(path)} {sized(total)}"
                          for path, files, total in caches(home.home())
                          if files)
        hf_home = os.environ.get("HF_HOME", "")
        out.append(Finding(name="models on this machine", good=bool(mine),
                           said=f"{len(mine)} file(s)" + (f": {where}" if where else ""),
                           note=(f"HF_HOME={hf_home} names the Hub cache" if hf_home else "")
                           if mine else "nothing found; ml-stack-models find <words>"))
        gguf = [name for name in mine if name.lower().endswith(".gguf")]
        out.append(Finding(
            name="a GGUF on disk", good=bool(gguf),
            said=f"{len(gguf)} file(s)" if gguf else "none",
            fix="" if gguf else "ml-stack-models fetch hf:owner/repo/file.gguf",
            note="" if gguf else "llama-server serves GGUF only; a safetensors checkpoint "
                                 "on disk is not enough to load"))
    except Exception:  # noqa: BLE001
        pass

    out.extend(Finding(name=one.name, good=here,
                       said=f"{one.module} is here" if here else "not installed",
                       fix="" if here else one.fix, note="" if here else one.does)
               for one, here in standard())
    out.extend(_speech_findings())
    out.append(_commands_finding())
    out.extend(_fleet_findings())
    out.append(_store_finding())

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


def _fleet_findings(*, port: int | None = None, discovery_port: int | None = None,
                    cluster_key_path: Path | str | None = None) -> list[Finding]:
    """Whether this machine has joined a cluster, its daemon answers, discovery hears it
    among the peers ``ml-stack-fleet status`` would list, and its two ports are free or
    held by that same daemon.

    Stops naming fleet findings at the first thing that is wrong: a daemon cannot answer
    before a cluster is joined, and discovery cannot hear a daemon that is not running.
    """
    from ml_stack.fleet.discovery import DEFAULT_HTTP_PORT, default_port, memberships
    from ml_stack.fleet.join import already_running, peers

    port = DEFAULT_HTTP_PORT if port is None else port
    discovery_port = default_port() if discovery_port is None else discovery_port
    out: list[Finding] = []
    mine = memberships(cluster_key_path)
    me = already_running(port)
    if not mine:
        out.append(Finding(name="fleet: joined", good=False, said="in no cluster",
                           fix="ml-stack-fleet join --passphrase WORDS"))
    else:
        out.append(Finding(name="fleet: joined", good=True, said=f"cluster '{mine[0].group}'"))
        if me is None:
            out.append(Finding(name="fleet: daemon", good=False,
                               said=f"does not answer on {port}", fix="ml-stack-fleet join"))
        else:
            out.append(Finding(name="fleet: daemon", good=True,
                               said=f"'{me.get('name', '?')}' answers on {port}"))
            try:
                rows = peers(cluster_key_path=cluster_key_path, port=discovery_port,
                            self_machine=str(me.get("machine") or ""))
            except OSError as exc:
                out.append(Finding(name="fleet: seen", good=False,
                                   said=f"discovery failed: {exc}", fix="ml-stack-peers ls"))
            else:
                seen = any(row.get("is_self") for row in rows)
                out.append(Finding(
                    name="fleet: seen", good=seen,
                    said=(f"sees itself among {len(rows)} peer(s)" if seen else
                          f"discovery hears {len(rows)} peer(s), none itself"),
                    fix="" if seen else "ml-stack-peers ls",
                    note="" if seen else
                         "the beacon is unreachable even on loopback -- a firewall or a "
                         "security tool dropping local multicast hides this machine from "
                         "every peer"))
    out.append(_ports_finding(port, discovery_port, me))
    return out


def _store_finding(path: Path | None = None) -> Finding:
    """Whether the default store (bench's own runs.ladybug) opens read-only."""
    from ml_stack.graph.cypher import CypherStore, GraphStoreUnavailable

    path = (home.state("bench") / "runs.ladybug") if path is None else path
    if not path.exists():
        return Finding(name="store", good=True, said=f"no store yet at {tilde(path)}")
    try:
        with CypherStore(path, read_only=True):
            pass
    except GraphStoreUnavailable as exc:
        return Finding(name="store", good=False, said=str(exc),
                       fix="pip install 'ml-stack[store]'")
    except (RuntimeError, OSError) as exc:
        return Finding(name="store", good=False, said=f"{tilde(path)} did not open: {exc}",
                       fix=f"ml-stack-store check {path}")
    return Finding(name="store", good=True, said=f"{tilde(path)} opens")


def _port_free(port: int, *, udp: bool = False) -> bool:
    """Whether a bare bind to ``port`` succeeds on this machine -- false when something
    else already holds it."""
    kind = socket.SOCK_DGRAM if udp else socket.SOCK_STREAM
    with socket.socket(socket.AF_INET, kind) as s:
        try:
            s.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def _ports_finding(http_port: int, discovery_port: int, daemon: dict[str, object] | None
                   ) -> Finding:
    """Whether ``http_port`` and ``discovery_port`` are free, or held by ``daemon`` --
    this machine's own ``/health`` answer, or None when nothing there is ours."""
    if daemon is not None:
        return Finding(name="ports", good=True,
                       said=f"{http_port} and {discovery_port} held by this machine's "
                            "own daemon")
    stuck = [p for p, udp in ((http_port, False), (discovery_port, True))
             if not _port_free(p, udp=udp)]
    return Finding(
        name="ports", good=not stuck,
        said="free" if not stuck else f"held by something else: {', '.join(map(str, stuck))}",
        fix="" if not stuck else f"lsof -nP -iTCP:{http_port} -iUDP:{discovery_port}",
        note="" if not stuck else
             "ml-stack-fleet join needs both free to start the daemon and hear the LAN")


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

        table = tomllib.loads((checkout() / "pyproject.toml").read_text(encoding="utf-8"))
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
        fix="" if not missing else f"pip install -e {checkout()} && pyenv rehash",
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
    from ml_stack.serve.backend import LlamaServerBackend, emitted_flags, flags_of, unknown_flags

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
        then = then.replace(tzinfo=UTC)
    delta = datetime.now(UTC) - then
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
    (the names llama.cpp's source defines) replaces the guess when it is given.
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
                got = subprocess.run(["strings", "-n", "2", str(name)], capture_output=True,
                                     text=True, timeout=30)
            except (OSError, subprocess.SubprocessError):
                continue
            found.update(line.strip() for line in got.stdout.splitlines())
    if known is not None:
        return found & known
    return {w for w in found if w.islower() and 4 <= len(w) <= 20
            and w.replace("-", "").isalnum() and any(
        w.startswith(f) for f in ("qwen", "gemma", "llama", "phi", "mistral",
                                  "deepseek", "granite", "olmo", "cohere", "gpt", "glm",
                                  "nemotron", "falcon", "mamba", "rwkv", "exaone"))}


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
                         "those that are git repositories")
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
        from ml_stack.doctor import look_checkouts, repositories

        findings = findings + look_checkouts(
            repositories(args.repo) if args.repo else None,
            bench_home=Path(args.bench_home).expanduser() if args.bench_home else None)
    ask(findings, yes=args.yes)
    if not args.quiet:
        explain()
    return 0 if all(f.good for f in findings) else 1


if __name__ == "__main__":
    raise SystemExit(main())
