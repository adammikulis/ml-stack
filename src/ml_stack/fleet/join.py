"""``ml-stack-fleet`` -- make this machine a peer in one command, and see what the fleet sees.

``join`` is the whole onboarding: the machine facts serving depends on (`ml_stack.setup.look`),
a llama-server if there is none, a cluster passphrase, the daemon (started now, and at logon
with ``--persist``), and then a listing of every peer that answered on the discovery port --
the same beacons `ml-stack-peers ls` reads, since there is one discovery mechanism and this
adds no second one. ``status`` is that listing on its own, with what each peer serves, its
room, whether it is measuring, and its commit; ``leave`` undoes ``join``.

Every step is a function that takes what it needs, so the app's Join button, the MCP
``fleet_join`` tool and the command line all run this one path.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .discovery import (
    DEFAULT_CLUSTER,
    Beacon,
    DiscoveryError,
    default_port,
    discover,
    in_cluster,
    join as join_cluster,
    key_path,
    leave as leave_cluster,
    memberships,
)
from .launch import HTTP_PORT, already_running, wait_for_health

__all__ = ["Check", "JoinError", "Joined", "DEFAULT_ROOT", "STARTED_FILE", "checks",
           "describe", "join_machine", "leave_machine", "main", "peers", "remember_track",
           "running_code", "start_daemon", "sweep_argv", "table", "updating"]

DEFAULT_ROOT = "~/.ml-stack/traind"
STARTED_FILE = "fleet-daemon.json"
"""Under the root: the pid, log and argv of the daemon ``join`` started, so ``leave`` can
stop it by pid rather than by name."""


class JoinError(RuntimeError):
    """Something ``join`` needs and cannot make: a passphrase, a daemon that answers."""


@dataclass(frozen=True, slots=True)
class Check:
    """One machine fact, and the line that fixes it when it is wrong."""

    name: str
    good: bool
    said: str
    fix: str = ""

    def public(self) -> dict[str, Any]:
        return {"name": self.name, "good": self.good, "said": self.said, "fix": self.fix}


@dataclass
class Joined:
    """What ``join`` did, for the caller to print or return."""

    name: str
    port: int
    root: Path
    group: str
    checks: list[Check] = field(default_factory=list)
    started: bool = False
    daemon_pid: int | None = None
    persisted: bool = False
    persist_note: str = ""
    tracking: str = ""
    peers: list[dict[str, Any]] = field(default_factory=list)

    def public(self) -> dict[str, Any]:
        return {"name": self.name, "port": self.port, "root": str(self.root),
                "group": self.group, "checks": [c.public() for c in self.checks],
                "started": self.started, "daemon_pid": self.daemon_pid,
                "persisted": self.persisted, "persist_note": self.persist_note,
                "tracking": self.tracking, "peers": self.peers}


# -- the checks ------------------------------------------------------------------------
def _machine_findings() -> list[Any]:
    """`ml_stack.setup.look`, behind a name a test can replace: it reads sysctl and libllama."""
    from ml_stack.setup import look

    return look()


def _server_here() -> str:
    """The llama-server this machine would serve with, or ''."""
    try:
        from ml_stack.serve.binary import find_binary

        return str(find_binary("llama-server") or "")
    except Exception:  # noqa: BLE001 - no serve package is no server
        return ""


def checks(root: Path | str, *, ensure: Callable[[Path], Path] | None = None,
           say: Callable[[str], None] = print) -> list[Check]:
    """The facts serving depends on, fixed where fixing is a download and not a decision.

    The memory a model may use and the build's architectures come from `setup.look` and are
    reported, never changed: raising the wiring limit is root's decision. A missing
    llama-server is different -- nothing about it is a choice -- so ``ensure`` (the daemon's
    own `llama.ensure_server`, which downloads a release build) is run, and the line says
    where it landed. ``ml-stack-serve build`` compiles master instead, for an architecture a
    release lags on.
    """
    out: list[Check] = []
    for one in _machine_findings():
        if one.name.startswith("memory") or one.name.startswith("architecture") \
                or one.name.startswith("flags"):
            out.append(Check(one.name, bool(one.good), one.said, one.fix))
    binary = _server_here()
    if binary:
        out.append(Check("llama-server", True, binary))
    else:
        root = Path(root).expanduser()
        if ensure is None:
            from .llama import ensure_server

            ensure = ensure_server
        say("  no llama-server on this machine; getting a release build")
        try:
            got = ensure(root)
            out.append(Check("llama-server", True, f"{got}  (downloaded now; "
                             "'ml-stack-serve build' compiles master instead)"))
        except Exception as exc:  # noqa: BLE001 - reported, and the join goes on
            out.append(Check("llama-server", False, f"none, and could not get one: {exc}",
                             "ml-stack-serve build"))
    return out


# -- the daemon ------------------------------------------------------------------------
def started_file(root: Path | str) -> Path:
    return Path(root).expanduser() / STARTED_FILE


def start_daemon(port: int, root: Path | str, name: str = "") -> int:
    """Start ``ml-stack-traind`` in its own session, its output in a log under ``root``.

    A child of this shell dies with it, and a daemon that dies when the terminal closes is
    a peer that vanishes the moment someone logs out. Returns the pid, which is also
    written to `started_file` for ``leave``.
    """
    root = Path(root).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    log = root / "traind.log"
    argv = [sys.executable, "-m", "ml_stack.fleet.daemon", "--port", str(port),
            "--root", str(root)]
    if name:
        argv += ["--name", name]
    from ml_stack.platform import process_group_kwargs

    with log.open("ab") as out:
        child = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=out,
                                 stderr=subprocess.STDOUT, **process_group_kwargs())
    started_file(root).write_text(json.dumps({
        "pid": child.pid, "argv": argv, "log": str(log),
        "started": time.strftime("%FT%T")}, indent=1), encoding="utf-8")
    return child.pid


def remember_track(root: Path | str, branch: str) -> str:
    """Write the branch this machine follows into the daemon's settings, before it starts.

    Written rather than passed as a flag, because the daemon reads ``settings.json`` at
    startup and the logon service that brings it back after a reboot carries no flags of
    ours. ``off`` (or "") goes back to following releases. Returns what was stored.
    """
    from .settings import Settings

    root = Path(root).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    path = root / "settings.json"
    settings = Settings.load(path)
    wanted = (branch or "").strip()
    settings.track_branch = "" if wanted.lower() in ("", "off", "none") else wanted
    settings.save(path)
    return settings.track_branch


def _started_pid(root: Path | str) -> int | None:
    try:
        held = json.loads(started_file(root).read_text(encoding="utf-8"))
        return int(held["pid"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _enrol_via_daemon(port: int, passphrase: str, group: str) -> None:
    """Add a cluster through the daemon already on ``port``, so it advertises at once.

    Writing the key file underneath a running daemon leaves it announcing on the clusters
    it read at startup; the ``/ui/clusters`` route is what re-reads them. The route is
    loopback-only for a machine in no cluster and needs a session otherwise, which the same
    passphrase opens.
    """
    base = f"http://127.0.0.1:{port}"
    headers = {"X-ML-Stack-UI": "1", "Content-Type": "application/json"}

    def call(path: str, body: dict[str, Any], cookie: str = "") -> tuple[int, dict, str]:
        req = urllib.request.Request(f"{base}{path}", data=json.dumps(body).encode(),
                                     method="POST", headers={**headers,
                                                             **({"Cookie": cookie} if cookie
                                                                else {})})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.status, json.loads(r.read() or b"{}"), r.headers.get("Set-Cookie", "")
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read() or b"{}"), ""
        except (urllib.error.URLError, OSError, ValueError) as e:
            raise JoinError(f"the daemon on port {port} did not take the passphrase: {e}") \
                from None

    cookie = ""
    status, body, set_cookie = call("/ui/session", {"passphrase": passphrase, "group": group})
    if status == 200 and set_cookie:
        cookie = set_cookie.split(";")[0]
    status, body, _ = call("/ui/clusters", {"passphrase": passphrase, "group": group}, cookie)
    if status != 200:
        raise JoinError(f"the daemon on port {port} refused the cluster: "
                        f"{body.get('error', status)}")


# -- what the fleet sees ---------------------------------------------------------------
def describe(beacon: Beacon, *, clusters: Iterable[str] = (), self_name: str = "") -> dict[str, Any]:
    """One peer as a row: what it serves, its room, whether it is busy, its commit.

    Read out of the beacon's device report, defensively: a daemon on an older build
    reports fewer fields, and a row with a '?' in it beats no row.
    """
    d = beacon.device or {}
    served = []
    for one in d.get("serving") or []:
        for model in one.get("models") or []:
            served.append(f"{model}:{one.get('port', '?')}")
    # `fleet.bench.BenchHost.report` puts the memory a model may use, the commit this
    # machine runs and whether it is measuring on the beacon; an older daemon has only
    # the device probe, and its card's memory is the next best answer.
    vram_total, vram_free = d.get("vram_total_gb"), d.get("vram_free_gb")
    if d.get("room_bytes"):
        from ml_stack.hub import _human

        room = _human(int(d["room_bytes"]))
    elif vram_total:
        room = f"{vram_free if vram_free is not None else '?'}/{vram_total} GB"
    elif d.get("ram_gb"):
        room = f"{d.get('ram_gb')} GB ram"
    else:
        room = "?"
    lock = d.get("lock") or ("measuring" if d.get("measuring") else "")
    return {
        "name": beacon.name, "host": beacon.host, "base_url": beacon.base_url,
        "port": beacon.port, "busy": bool(beacon.busy), "free": beacon.free,
        "slots": beacon.slots, "queued": beacon.queued,
        "room": room, "serving": served,
        "models": [str(m.get("name")) for m in (d.get("models") or []) if m.get("name")],
        "lock": str(lock) if lock else "",
        "commit": str(d.get("bench_commit") or d.get("commit") or "?"),
        "commit_age_s": float(d.get("commit_age_s") or 0.0),
        "version": str(d.get("version") or ""),
        "tracking": str(d.get("tracking") or ""),
        "checked_at": float(d.get("update_checked_at") or 0.0),
        "update_error": str(d.get("update_error") or ""),
        "gpu": str(d.get("gpu") or ""),
        "clusters": list(clusters),
        "is_self": bool(self_name) and beacon.name == self_name,
        "device": d,
    }


def peers(*, cluster_key_path: Path | str | None = None, timeout_s: float = 2.0,
          port: int | None = None, self_name: str = "",
          finder: Callable[..., list[Beacon]] = discover) -> list[dict[str, Any]]:
    """Every daemon on the LAN that proves it holds one of this machine's cluster keys.

    One machine in two clusters answers on each; it is listed once with both.
    """
    # Keyed on the daemon, not its address: the same daemon answers each cluster with a
    # beacon of its own, and can be heard on loopback for one and the LAN for the other.
    found: dict[str, dict[str, Any]] = {}
    for member in memberships(cluster_key_path):
        for beacon in finder(member.key, timeout_s=timeout_s, port=port):
            who = f"{beacon.hostname}:{beacon.name}:{beacon.port}"
            row = found.get(who)
            if row is None:
                found[who] = describe(beacon, clusters=[member.group], self_name=self_name)
                continue
            if member.group not in row["clusters"]:
                row["clusters"].append(member.group)
            if beacon.host.startswith("127.") and not row["host"].startswith("127."):
                row["host"], row["base_url"] = beacon.host, beacon.base_url
    return sorted(found.values(), key=lambda r: (not r["is_self"], r["name"]))


def _ago(seconds: float) -> str:
    """A duration a person reads at a glance: 45s, 4m, 3h, 2d. "" for nothing."""
    if seconds <= 0:
        return ""
    if seconds < 90:
        return f"{int(seconds)}s"
    if seconds < 5400:
        return f"{int(seconds / 60)}m"
    if seconds < 172800:
        return f"{int(seconds / 3600)}h"
    return f"{int(seconds / 86400)}d"


def running_code(row: dict[str, Any]) -> str:
    """What a peer runs: the short sha, and how old that commit is."""
    commit = str(row.get("commit") or "?").split()[0][:7] or "?"
    dirty = "*" if "dirty" in str(row.get("commit") or "") else ""
    age = _ago(float(row.get("commit_age_s") or 0.0))
    return f"{commit}{dirty} {age}".strip()


def updating(row: dict[str, Any]) -> str:
    """How a peer keeps current, and when it last looked.

    A fleet half on one commit and half on another is the thing this column exists to make
    visible: 'main 4m' is a machine following a branch that looked four minutes ago,
    'releases' is one waiting for the next published build, 'off' is one that will sit on
    the code it has until somebody visits it.
    """
    mode = str(row.get("tracking") or "") or "?"
    if mode in ("off", "?"):
        return mode
    when = float(row.get("checked_at") or 0.0)
    age = _ago(time.time() - when) if when else ""
    if row.get("update_error"):
        return f"{mode} !"
    return f"{mode} {age}".strip()


def table(rows: Sequence[dict[str, Any]]) -> str:
    """The listing, as text."""
    if not rows:
        return ("no peers answered.\n"
                "  - is the daemon running there?  ml-stack-fleet join\n"
                "  - same LAN, and the same passphrase?")
    lines = [f"{'NAME':<16} {'URL':<28} {'ROOM':<16} {'STATE':<12} {'COMMIT':<12} "
             f"{'UPDATES':<12} SERVING"]
    for r in rows:
        state = "busy" if r["busy"] or r["free"] == 0 else "idle"
        if r.get("queued"):
            state += f" +{r['queued']}"
        if r.get("lock"):
            state = "measuring"
        serving = ", ".join(r["serving"]) or "-"
        lines.append(f"{r['name']:<16} {r['base_url']:<28} {r['room']:<16} {state:<12} "
                     f"{running_code(r):<12} {updating(r):<12} {serving}")
    return "\n".join(lines)


# -- the join ---------------------------------------------------------------------------
def join_machine(*, name: str = "", passphrase: str = "", group: str = DEFAULT_CLUSTER,
                 persist: bool = False, track: str = "", port: int = HTTP_PORT,
                 root: Path | str = DEFAULT_ROOT,
                 cluster_key_path: Path | str | None = None,
                 timeout_s: float = 2.0, wait_s: float = 20.0,
                 say: Callable[[str], None] = print,
                 start: Callable[[int, Path, str], int] = start_daemon,
                 enrol: Callable[[str, str], None] | None = None,
                 ensure: Callable[[Path], Path] | None = None,
                 persist_with: Callable[..., Any] | None = None,
                 finder: Callable[..., list[Beacon]] = discover,
                 discovery_port: int | None = None) -> Joined:
    """Make this machine a peer, and return what the fleet now sees.

    In order: the checks; the cluster (``passphrase`` joins ``group``; a machine already in
    one keeps it); the daemon (reused if one answers on ``port``, else started detached --
    ``enrol`` adds the cluster through a daemon that is already up); ``--persist`` installs
    it at logon; then discovery on the beacon port. ``start``, ``enrol``, ``ensure``,
    ``persist_with`` and ``finder`` are the four things a test replaces with a fake on
    loopback; everything else is what the command does.
    """
    root = Path(root).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    group = (group or "").strip() or DEFAULT_CLUSTER
    joined = Joined(name=name, port=port, root=root, group=group)
    if track:
        joined.tracking = remember_track(root, track)
        say(f"following '{joined.tracking}'" if joined.tracking
            else "following releases")

    say("checking this machine")
    joined.checks = checks(root, ensure=ensure, say=say)
    for c in joined.checks:
        say(f"  {'ok  ' if c.good else '  ! '}{c.name}: {c.said}")
        if not c.good and c.fix:
            say(f"        fix: {c.fix}")

    running = already_running(port)
    if passphrase:
        if running is not None:
            (enrol or (lambda words, g: _enrol_via_daemon(port, words, g)))(passphrase, group)
        else:
            join_cluster(passphrase, group=group, path=cluster_key_path)
        say(f"joined cluster '{group}'")
    elif not in_cluster(cluster_key_path):
        raise JoinError("this machine is in no cluster and no passphrase was given -- "
                        "pass --passphrase WORDS (the same words on every machine)")
    else:
        current = memberships(cluster_key_path)[0].group
        joined.group = current
        say(f"already in cluster '{current}'")

    if running is not None:
        joined.name = str(running.get("name") or name)
        say(f"the daemon is already running as '{joined.name}' on port {port}")
        if joined.tracking:
            say(f"  it reads '{joined.tracking}' at its next start; restart it to follow "
                "the branch now")
    else:
        if persist:
            joined.persisted, joined.persist_note = _persist(persist_with, say)
            if joined.persisted:
                say("waiting for the logon service to bring it up")
                if wait_for_health(port, seconds=wait_s) is not None:
                    running = already_running(port)
        if running is None:
            joined.daemon_pid = start(port, root, name)
            joined.started = True
            say(f"started the daemon (pid {joined.daemon_pid}); waiting for it to answer")
            if wait_for_health(port, seconds=wait_s) is None:
                raise JoinError(f"the daemon did not answer on port {port} within "
                                f"{wait_s:.0f}s; its log is {root / 'traind.log'}")
            running = already_running(port) or {}
        joined.name = str((running or {}).get("name") or name)
    if persist and not joined.persisted and not joined.persist_note:
        joined.persisted, joined.persist_note = _persist(persist_with, say)

    say(f"asking the network (discovery port {discovery_port or default_port()})")
    joined.peers = peers(cluster_key_path=cluster_key_path, timeout_s=timeout_s,
                         port=discovery_port, self_name=joined.name, finder=finder)
    say(table(joined.peers))
    return joined


def _persist(persist_with: Callable[..., Any] | None, say: Callable[[str], None]
             ) -> tuple[bool, str]:
    """Install the daemon at logon. Returns ``(installed, what a person must still run)``."""
    if persist_with is None:
        from . import autostart

        persist_with = autostart.install
    done = persist_with("login")
    if getattr(done, "installed", False):
        say(f"installed at logon ({done.path})")
        return True, ""
    note = f"{done.note}  run: {done.command}".strip() if getattr(done, "command", "") \
        else str(getattr(done, "note", "") or "not installed")
    say(f"  not installed at logon: {note}")
    return False, note


def leave_machine(*, group: str = "", root: Path | str = DEFAULT_ROOT,
                  cluster_key_path: Path | str | None = None, stop: bool = True,
                  say: Callable[[str], None] = print,
                  unpersist: Callable[[], list[Path]] | None = None) -> dict[str, Any]:
    """Undo ``join``: drop the cluster(s), the logon service, and the daemon it started."""
    before = memberships(cluster_key_path)
    left = [m.group for m in before if not group or m.group == group]
    for one in left:
        leave_cluster(one, cluster_key_path)
        say(f"left cluster '{one}'")
    if not left:
        say("in no cluster" if not before else f"not in a cluster called '{group}'")

    if unpersist is None:
        from . import autostart

        unpersist = autostart.uninstall
    removed = [str(p) for p in unpersist()]
    for p in removed:
        say(f"removed the logon service {p}")

    stopped: int | None = None
    pid = _started_pid(root)
    if stop and pid is not None:
        from ml_stack.platform import stop_pid
        from ml_stack.serve.process import pid_exists

        if pid_exists(pid):
            sent = stop_pid(pid)
            stopped = pid
            say(f"stopped the daemon (pid {pid}, {sent})")
        with contextlib.suppress(OSError):
            started_file(root).unlink()
    return {"left": left, "removed": removed, "stopped": stopped}


# -- a sweep over the fleet ----------------------------------------------------------
def sweep_argv(models: Sequence[str], *, peers: Sequence[str] = (), sample: int = 0,
               label: str = "", extra: Sequence[str] = ()) -> list[str]:
    """The ``ml-stack-bench`` line that spreads ``models`` over the fleet, one job each.

    Built here rather than in the page, so the command the page shows is the command the
    daemon runs and the one a person could paste. ``--detach`` is not on it: the caller
    detaches, and hands the log back.
    """
    if not models:
        raise ValueError("a sweep needs at least one model")
    argv = ["sweep", "--fleet"]
    for m in models:
        argv += ["--serve", str(m)]
    if peers:
        argv += ["--peers", ",".join(peers)]
    if sample:
        argv += ["--sample", str(int(sample))]
    if label:
        argv += ["--label", label]
    argv += [str(a) for a in extra]
    return argv


# -- the command -----------------------------------------------------------------------
def _passphrase_from(args: argparse.Namespace, key: Path | str | None) -> str:
    """The words, from the flag, the environment, a terminal, or standard input.

    ``ML_STACK_PASSPHRASE`` is what makes an unattended install possible: a machine being
    set up by a script has no terminal to type at, and prompting one that cannot answer
    hangs the install rather than failing it. The order is deliberate -- an explicit flag
    beats the environment, and both beat asking.
    """
    if args.passphrase:
        return args.passphrase
    told = os.environ.get("ML_STACK_PASSPHRASE", "").strip()
    if told:
        return told
    if in_cluster(key):
        return ""
    if sys.stdin.isatty():
        from .peers import _prompt_passphrase

        print("This machine is in no cluster yet. Type the passphrase every machine shares.")
        return _prompt_passphrase(confirm=True)
    words = sys.stdin.readline().strip()
    return words


def cmd_join(args: argparse.Namespace) -> int:
    words = _passphrase_from(args, args.cluster_key)
    # An install script sets these rather than answering prompts it has no terminal for.
    name = args.name or os.environ.get("ML_STACK_NAME", "").strip()
    group = args.group if args.group != DEFAULT_CLUSTER else (
        os.environ.get("ML_STACK_CLUSTER", "").strip() or DEFAULT_CLUSTER)
    joined = join_machine(name=name, passphrase=words, group=group,
                          persist=args.persist, track=args.track, port=args.port,
                          root=args.root,
                          cluster_key_path=args.cluster_key, timeout_s=args.timeout)
    if args.json:
        print(json.dumps(joined.public(), indent=1))
        return 0
    print()
    print(f"this machine is '{joined.name}' on http://127.0.0.1:{joined.port}"
          + (", starting at logon" if joined.persisted else ""))
    if joined.tracking:
        print(f"  following {joined.tracking}: it updates itself when nothing is running")
    print("  ml-stack-fleet status   -- who is in the fleet, what each serves")
    print("  ml-stack-fleet leave    -- undo this")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    if not memberships(args.cluster_key):
        print(f"in no cluster (no key at {key_path(args.cluster_key)}); "
              "run 'ml-stack-fleet join'", file=sys.stderr)
        return 1
    me = already_running(args.port)
    rows = peers(cluster_key_path=args.cluster_key, timeout_s=args.timeout,
                 self_name=str((me or {}).get("name") or ""))
    if args.json:
        print(json.dumps(rows, indent=1, default=str))
        return 0 if rows else 1
    print(table(rows))
    if me is None:
        print(f"\nthis machine's daemon is not running on port {args.port}; "
              "'ml-stack-fleet join' starts it")
    return 0 if rows else 1


def cmd_leave(args: argparse.Namespace) -> int:
    leave_machine(group=args.group, root=args.root, cluster_key_path=args.cluster_key,
                  stop=not args.keep_running)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="ml-stack-fleet",
        description="Make this machine a peer in one command, and see what the fleet sees.")
    ap.add_argument("--cluster-key", default=None,
                    help="path to the cluster key (default: ~/.ml-stack/cluster.key)")
    ap.add_argument("--root", default=DEFAULT_ROOT,
                    help=f"the daemon's root (default: {DEFAULT_ROOT})")
    ap.add_argument("--port", type=int, default=HTTP_PORT,
                    help=f"the daemon's HTTP port (default: {HTTP_PORT}); discovery is one "
                         f"above it")
    sub = ap.add_subparsers(dest="cmd", required=True)

    join_p = sub.add_parser(
        "join", help="check this machine, start the daemon, announce, and list the fleet")
    join_p.add_argument("--name", default="",
                        help="how this machine identifies itself (default: the hostname)")
    join_p.add_argument("--persist", action="store_true",
                        help="start the daemon at logon as well, so it comes back after "
                             "a restart")
    join_p.add_argument("--passphrase", default="",
                        help="the words every machine shares; prompted for in a terminal "
                             "when the machine is in no cluster yet")
    join_p.add_argument("--group", default=DEFAULT_CLUSTER,
                        help=f"which cluster the words belong to (default: {DEFAULT_CLUSTER})")
    join_p.add_argument("--track", default="",
                        help="follow a branch instead of releases, e.g. 'main': this "
                             "machine pulls, reinstalls if the packaging moved and "
                             "restarts whenever no job, benchmark or model is running. "
                             "Needs a git checkout with an editable install, and runs "
                             "code nobody reviewed. Remembered; '--track off' undoes it.")
    join_p.add_argument("--timeout", type=float, default=2.0,
                        help="seconds to listen for peers (default: 2)")
    join_p.add_argument("--json", action="store_true", help="print what was done as JSON")

    status_p = sub.add_parser("status", help="the peers: what each serves, its room, its "
                                             "lock, its commit")
    status_p.add_argument("--timeout", type=float, default=2.0)
    status_p.add_argument("--json", action="store_true")

    leave_p = sub.add_parser("leave", help="drop the cluster, the logon service and the "
                                           "daemon join started")
    leave_p.add_argument("--group", default="",
                         help="leave only this cluster (default: every one)")
    leave_p.add_argument("--keep-running", action="store_true",
                         help="leave the daemon up, just stop answering as a peer")

    args = ap.parse_args(argv)
    fn = {"join": cmd_join, "status": cmd_status, "leave": cmd_leave}[args.cmd]
    try:
        return fn(args)
    except (JoinError, DiscoveryError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
