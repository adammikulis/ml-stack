"""A training daemon: one GPU box, one job at a time, reachable over the LAN.

`serve_forever` puts the parts together and runs them -- the `fleet.jobs.JobRunner`, the
`fleet.files.Fetcher`, the model store, the bench host, the web interface, the updater and
the beacon -- behind the routes `fleet.api.make_handler` builds. `main` is
``ml-stack-traind``.
"""

from __future__ import annotations

import contextlib
import json
import os
import secrets
import socket
import threading
from collections.abc import Callable, Iterable
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any

from ml_stack.hub import default_roots
from ml_stack.log import say, warn
from ml_stack.platform import on_quit, private_file
from ml_stack.speech import service as speech

from . import autostart, updates as updating
from .api import Daemon, make_handler
from .availability import Availability, parse_window
from .conversations import Conversations
from .device import device_report as default_report, resolve_report, stdlib_device_report
from .discovery import (
    Advertiser,
    Beacon,
    DiscoveryError,
    derive_token,
    key_path,
    load_cluster_key,
    memberships,
)
from .environment import Environment
from .files import Fetcher
from .jobs import JobRunner
from .measuring import BenchHost, bench_home as bench_home_beside
from .models import Downloads, Models
from .pausing import ADOPT_S, adopt_pause, peer_pause
from .serving import Hosting, Serving
from .settings import Settings
from .ui import UI

DEFAULT_PORT = 8770


def load_or_create_token(root: Path, cluster_key: bytes | None = None) -> str:
    """The bearer token: derived from the cluster key, or random and local."""
    p = root / "token"
    if cluster_key is not None:
        tok = derive_token(cluster_key)
        root.mkdir(parents=True, exist_ok=True)
        if not p.exists() or p.read_text().strip() != tok:
            p.write_text(tok)
        private_file(p)
        return tok
    if p.exists():
        return p.read_text().strip()
    root.mkdir(parents=True, exist_ok=True)
    tok = secrets.token_urlsafe(24)
    p.write_text(tok)
    p.chmod(0o600)
    return tok


def serve_forever(root: Path | str = "~/.ml-stack/traind",
                  host: str = "0.0.0.0", port: int = DEFAULT_PORT, *,
                  name: str = "", announce: bool = True,
                  cluster_key_path: Path | str | None = None,
                  device_report: Callable[[], dict[str, Any]] | None = None,
                  slots: int = 1, labels: Iterable[str] = (),
                  fetch_slots: int = 2, web: bool = True,
                  setup_from_lan: bool = False,
                  busy_hours: Iterable[str] = (), free_hours: Iterable[str] = (),
                  on_paused: str = "stop",
                  bench_home: Path | str | None = None,
                  track: str | None = None) -> None:
    """``bench_home`` is where this machine's ``ml-stack-bench`` keeps its measuring
    lock; the ``bench`` beside ``root`` unless given (`fleet.measuring.bench_home`), so a
    daemon rooted in a test's directory never consults the real home.

    ``track`` is a branch this machine follows instead of releases -- ``main`` on a machine
    you trust to run unreviewed code -- and it is remembered, so it is asked for once.
    ``off`` turns it back to releases; None leaves whatever the settings hold."""
    root = Path(root).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    live_token: list[str] = [""]
    files_root = root / "files"
    files_root.mkdir(exist_ok=True)
    key = load_cluster_key(cluster_key_path)
    token = load_or_create_token(root, key)
    live_token[0] = token
    settings_path = root / "settings.json"
    settings = Settings.load(settings_path)
    name = (name or os.environ.get("ML_STACK_PEER_NAME") or settings.name
            or socket.gethostname())
    live_name = [name]
    if slots == 1 and settings.slots != 1:
        slots = settings.slots
    if not labels and settings.labels:
        labels = settings.labels
    if on_paused == "stop" and settings.on_paused != "stop":
        on_paused = settings.on_paused

    schedule_path = root / "availability.json"
    schedule = Availability.load(schedule_path)
    for spec in busy_hours:
        schedule.windows.append(parse_window(spec))
    for spec in free_hours:
        schedule.windows.append(parse_window(spec, busy=False))
    if busy_hours or free_hours:
        schedule.save(schedule_path)

    taken = adopt_pause(schedule, peer_pause(cluster_key_path, timeout_s=ADOPT_S))
    if taken is not None:
        schedule.save(schedule_path)
        say(f"  paused with the cluster: {taken.said()}")

    # What the screen shows has to be what the daemon is doing, so the effective
    # values go back into the settings object whether they came from a flag or a file.
    settings.slots = slots
    settings.labels = [s.strip() for s in labels if s and s.strip()]
    settings.on_paused = on_paused
    if track is not None:
        wanted = track.strip()
        settings.track_branch = "" if wanted.lower() in ("", "off", "none") else wanted
        settings.save(settings_path)

    environment = Environment(root)
    serving = Serving(root / "serving.json")
    hosting = Hosting(root, serving)
    models = Models(default_roots(root), root / "models")
    conversations = Conversations(root / "chats")
    downloads = Downloads(models)
    measuring_home = (Path(bench_home).expanduser() if bench_home is not None
                      else bench_home_beside(root))
    bench_host: list[BenchHost] = []

    def may_start() -> tuple[bool, str]:
        """A measurement holds the GPU as surely as a job does, so nothing starts
        beside one -- whether this daemon started it or someone at the keyboard did."""
        if bench_host and bench_host[0].measuring():
            return False, "a benchmark is measuring on this machine"
        return schedule.may_start()

    runner = JobRunner(root, files_root, slots=slots, gate=may_start,
                       environment=environment)
    bench_host.append(BenchHost(runner, home=measuring_home, name=name))
    fetcher = Fetcher(files_root, key, slots=fetch_slots)
    interface = None
    setup_token = ""
    if web:
        if key is None and setup_from_lan:
            setup_token = secrets.token_urlsafe(9)
        interface = UI(name=name, cluster_key_path=cluster_key_path, peer_port=port,
                       setup_token=setup_token)
    base_report = device_report or default_report
    labels = sorted({s.strip() for s in labels if s and s.strip()})

    heard: list[str] = []

    def probe_speech() -> None:
        heard[:] = speech.working()

    # Off the startup path: a probe imports whisper's dependencies where they are
    # installed, and the beacon carries what has been found by the time it goes out.
    threading.Thread(target=probe_speech, name="speech-probe", daemon=True).start()

    def report() -> dict[str, Any]:
        # `updates.state` puts the version, the commit and how this machine keeps current
        # on the beacon, so `ml-stack-fleet status` can show a fleet that is half-updated
        # rather than everyone guessing.
        return {**base_report(), "labels": labels, **bench_host[0].report(),
                **updating.state(), "speech": list(heard)}
    def every_token() -> set[str]:
        """Every token this machine answers to, one per cluster it is in."""
        return {derive_token(m.key) for m in memberships(cluster_key_path)}

    httpd = ThreadingHTTPServer((host, port),
                                make_handler(Daemon(
                                    runner, files_root, lambda: live_token[0],
                                    name=lambda: live_name[0], report=report, fetcher=fetcher,
                                    ui=interface, schedule=schedule, on_paused=on_paused,
                                    schedule_path=schedule_path, serving=serving, models=models,
                                    cluster_key_path=cluster_key_path, tokens=every_token,
                                    bench=bench_host[0], hosting=hosting)))
    # Keeping this machine current, in one of two modes and never in both. Either way the
    # gate is the same: nothing is replaced over a job, a measurement or a loaded model.
    nothing_running = updating.quiet(
        jobs=lambda: bool(runner.status()["busy"]),
        measuring=lambda: bool(bench_host[0].measuring()),
        leases=lambda: bool(serving.live()))
    tracked = str(getattr(settings, "track_branch", "") or "").strip()
    tracked_from = str(getattr(settings, "track_repo", "") or "") or updating.GIT_URL
    checkout = updating.checkout_here() if tracked else None
    if tracked and checkout is not None:
        say(f"  following {tracked} on {tracked_from} in {checkout}")
        updating.track(tracked_from, tracked, checkout, idle=nothing_running)
    else:
        if tracked:
            say(f"  cannot follow '{tracked}': this copy is not a git checkout. "
                "Releases instead.")
        updating.watch(wanted=lambda: bool(getattr(settings, "auto_update", False)),
                       idle=nothing_running)

    advertiser: Advertiser | None = None
    advertisers: dict[str, Advertiser] = {}

    def refresh(b: Beacon) -> None:
        """Bring the beacon's mutable half up to date before it goes on the wire."""
        status = runner.status()
        available = schedule.public()
        b.busy, b.queued = status["busy"], status["queued"]
        b.slots, b.free = status["slots"], status["free"]
        if not available["available"]:
            b.free = 0
        b.device = {**report(), "availability": available,
                    "serving": serving.public(), **models.beacon()}

    def start_announcing() -> None:
        """Advertise on every cluster this machine is in, and stop on any it left."""
        nonlocal advertiser, key
        if not announce:
            return
        joined = {m.group: m for m in memberships(cluster_key_path)}

        for group in [g for g in advertisers if g not in joined]:
            with contextlib.suppress(Exception):
                advertisers.pop(group).stop()

        for group, member in joined.items():
            if group in advertisers:
                continue
            beacon = Beacon(name=live_name[0], port=port, device=report(),
                            slots=runner.slots, free=runner.slots,
                            machine=bench_host[0].machine)
            try:
                # Not group=: that is the multicast address every cluster shares.
                # Clusters are told apart by the key their beacons are signed with.
                advertisers[group] = Advertiser(beacon, member.key,
                                                refresh=refresh).start()
            except DiscoveryError as exc:
                say(f"  discovery OFF for {group}: {exc}")

        first = next(iter(joined.values()), None)
        if first is not None:
            key = first.key
            fetcher.key = first.key
            live_token[0] = load_or_create_token(root, first.key)
            advertiser = advertisers.get(first.group)

    def rename(called: str) -> str:
        """Give this machine a new name now, on the page and on the network."""
        called = called.strip()
        if not called or called == live_name[0]:
            return live_name[0]
        live_name[0] = called
        bench_host[0].name = called
        if interface is not None:
            interface.name = called
        for one in advertisers.values():
            one.beacon.name = called
        return called

    if interface is not None:
        interface.rename = rename
        interface.on_join = start_announcing
        interface.runner = runner
        interface.schedule = schedule
        interface.settings = settings
        interface.settings_path = settings_path
        interface.schedule_path = schedule_path
        interface.report = report
        interface.environment = environment
        interface.serving = serving
        interface.hosting = hosting
        interface.models = models
        interface.conversations = conversations
        interface.downloads = downloads
        interface.root = root

    if announce:
        start_announcing()
    say(f"ml-stack traind on http://{host}:{port}")
    say(f"  name  {name}")
    say(f"  root  {root}")
    say(f"  bench {measuring_home}")
    say(f"  slots {slots}")
    state = schedule.public()
    for window in schedule.windows:
        say(f"  busy  {window.describe()}" if window.busy
            else f"  free  {window.describe()}")
    if not state["available"]:
        say(f"  NOT TAKING WORK -- {state['unavailable_because']}")
    if labels:
        say(f"  labels {' '.join(labels)}")
    if advertiser is not None:
        say(f"  peers announcing on {advertiser.group}:{advertiser.port} "
            f"(key {key_path(cluster_key_path)})")
        say("  token derived from the cluster key -- peers compute it themselves")
    elif key is None:
        say(f"  token {token}")
        say(f"  discovery OFF: no cluster key at {key_path(cluster_key_path)}")
        if interface is None:
            say("  run 'ml-stack-peers setup' to join one")
    say(f"  device {json.dumps(report())}")
    if slots > 1:
        say(f"  {slots} jobs will run at once. Correct for CPU work; on a GPU box "
            "this makes every job slower.")
    if interface is not None:
        shown = "127.0.0.1" if host in ("0.0.0.0", "") else host
        say(f"  open   http://{shown}:{port}/ui/")
        if key is None:
            say("  this machine has not joined a cluster yet -- open the address "
                "above ON THIS MACHINE to set it up")
            if setup_token:
                say(f"  setup from the LAN with this one-time code: {setup_token}")
    say("  THIS EXECUTES COMMANDS YOU SEND IT. Trusted LAN only.", flush=True)

    def _quit(signum: int, _frame: Any) -> None:
        # SIGTERM (launchd, systemd, kill) and on Windows SIGBREAK take the same exit as
        # Ctrl+C, so the beacon stops and the server closes rather than vanishing.
        raise KeyboardInterrupt(f"signal {signum}")

    on_quit(_quit)
    # A server this machine was told to stop when nobody is using it (`ml-stack-serve
    # limits --idle`). Without one, nothing is watched and nothing is stopped.
    from contextlib import ExitStack

    from ml_stack.limits import read as limits_read
    from ml_stack.serve.reclaim import watching

    idle_s = limits_read().idle_s
    reclaiming = ExitStack()
    if idle_s:
        say(f"  reclaiming a server unused for {idle_s:.0f}s")
        reclaiming.enter_context(watching(older_than=idle_s, say=print))
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        reclaiming.close()
        _stop_advertisers(advertisers)
        if advertiser is not None:
            advertiser.stop()
        runner.shutdown()
        httpd.server_close()


def _stop_advertisers(advertisers: dict[str, Any]) -> None:
    """Stop every cluster's advertiser and empty the mapping."""
    for one in list(advertisers.values()):
        with contextlib.suppress(Exception):
            one.stop()
    advertisers.clear()


def persist(*, slots: int = 1, labels: tuple[str, ...] = (), report: str = "") -> int:
    """``ml-stack-traind --persist``: start at login from now on, the way ``ml-stack-serve
    build --persist`` refreshes weekly. Says what was installed, or what a person must run."""
    done = autostart.install("login", slots=slots, labels=labels, report=report)
    if done.installed:
        say("installed to start at login")
        if done.path is not None:
            say(f"  {done.path}")
        if done.note:
            say(f"  {done.note}")
        say("  starting now; 'ml-stack-peers ls' from another machine should list it")
        return 0
    warn("not installed")
    if done.note:
        warn(f"  {done.note}")
    if done.command:
        warn(f"  run this yourself: {done.command}")
    return 2


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="ml-stack-traind")
    ap.add_argument("--root", default="~/.ml-stack/traind")
    ap.add_argument("--bench-home", default=None, metavar="DIR",
                    help="where this machine's ml-stack-bench keeps its measuring lock "
                         "(default: the 'bench' beside --root, so ~/.ml-stack/bench). "
                         "While that lock is held, queued training waits.")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--name", default="",
                    help="how this box identifies itself to peers "
                         "(default: $ML_STACK_PEER_NAME, else the hostname)")
    ap.add_argument("--cluster-key", default=None,
                    help="path to the cluster key (default: ~/.ml-stack/cluster.key)")
    ap.add_argument("--no-announce", action="store_true",
                    help="serve, but stay invisible to peer discovery")
    ap.add_argument("--busy", action="append", default=[], metavar="WHEN",
                    help="when this machine is NOT available for work, e.g. "
                         "'mon-fri 09:00-17:00' or '22:00-06:00'. Repeatable. Queued "
                         "work waits for the window to close rather than failing.")
    ap.add_argument("--free", action="append", default=[], metavar="WHEN",
                    help="carve an exception out of a busy window, e.g. "
                         "'mon-fri 12:00-13:00'")
    ap.add_argument("--on-paused", choices=("stop", "finish"), default="stop",
                    help="what happens to work already running when the machine is "
                         "paused: stop it (default -- SIGTERM, so a loop that "
                         "checkpoints keeps its progress, and it is requeued) or let "
                         "it finish")
    ap.add_argument("--no-web", action="store_true",
                    help="serve the API but not the web interface")
    ap.add_argument("--setup-from-lan", action="store_true",
                    help="on a machine that has not joined a cluster, allow first-run "
                         "setup from another machine using the one-time code printed "
                         "at startup. For a headless box you cannot open a browser on.")
    ap.add_argument("--fetch-slots", type=int, default=2,
                    help="concurrent peer-to-peer file transfers (default 2). Not job "
                         "slots: a transfer is I/O against another box, not compute, "
                         "and must not occupy a training slot.")
    ap.add_argument("--label", action="append", default=[], metavar="LABEL",
                    help="a role this box declares, e.g. 'prep'. Repeatable. Work can "
                         "require or exclude labels; nothing is inferred from them.")
    ap.add_argument("--report", action="append", default=[], metavar="MODULE:CALLABLE",
                    help="a richer device probe, e.g. "
                         "'ml_stack.train.accelerator:report' on a box with a card. "
                         "Repeatable; later ones win. Without any, the daemon reports "
                         "only what the standard library can see, plus whatever is "
                         "registered under the 'ml_stack.device_report' entry point.")
    ap.add_argument("--slots", type=int,
                    default=int(os.environ.get("ML_STACK_SLOTS") or 1),
                    help="how many jobs to run at once (default 1; raise it only on a "
                         "box whose work does not contend for one accelerator)")
    ap.add_argument("--track", default=None, metavar="BRANCH",
                    help="follow a branch instead of releases: this machine pulls "
                         "BRANCH (e.g. 'main'), reinstalls if the packaging moved and "
                         "restarts, whenever no job, benchmark or model is running. "
                         "Needs a git checkout with an editable install. Remembered, so "
                         "it is asked for once; '--track off' goes back to releases. It "
                         "runs code nobody reviewed -- only on a machine you trust to.")
    ap.add_argument("--persist", action="store_true",
                    help="do not serve; install this daemon (with these --slots, --label "
                         "and --report) to start when you log in -- a LaunchAgent on "
                         "macOS, a user systemd unit on Linux, a Scheduled Task on "
                         "Windows -- and start it now. 'ml-stack-traind --persist' twice "
                         "replaces the first install.")
    a = ap.parse_args(argv)
    if a.persist:
        return persist(slots=a.slots, labels=tuple(a.label), report=a.report[-1]
                       if a.report else "")
    probes = [resolve_report(spec) for spec in a.report]

    def report() -> dict[str, Any]:
        out = stdlib_device_report()
        for probe in probes:
            try:
                out.update(probe() or {})
            except Exception:                         # noqa: BLE001
                pass
        return out

    serve_forever(a.root, a.host, a.port, name=a.name,
                  announce=not a.no_announce, cluster_key_path=a.cluster_key,
                  slots=a.slots, device_report=report if probes else None,
                  labels=a.label or os.environ.get("ML_STACK_LABELS", "").split(","),
                  fetch_slots=a.fetch_slots, web=not a.no_web,
                  setup_from_lan=a.setup_from_lan,
                  busy_hours=a.busy, free_hours=a.free, on_paused=a.on_paused,
                  bench_home=a.bench_home, track=a.track)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
