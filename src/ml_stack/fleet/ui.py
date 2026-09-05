"""The fleet's web interface: routes, and the guard on first-run setup."""

from __future__ import annotations

import contextlib
import json
import time
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any

from ml_stack.http import ServerError, open_stream

from .discovery import (
    DiscoveryError,
    check_passphrase,
    cluster_group,
    derive_token,
    discover,
    in_cluster,
    join_cluster,
    load_cluster_key,
)
from .page import FIT_ONLY
from .routes import ASSETS, UI_HEADER, asset_bytes, routes, write, write_json
from .session import Sessions, Throttle, parse_cookie

__all__ = ["ASSETS", "UI", "UI_HEADER", "asset_bytes", "routes", "serve_page"]


FALLBACK_ROOM = 24 * 1024 ** 3
"""What the fit view is drawn for when nothing said how much room this machine has."""

LOOPBACK = {"127.0.0.1", "::1", "localhost"}
DISCOVER_CACHE_S = 3.0
"""Long enough that three panels refreshing do not each fire a multicast sweep, short"""


SCRAPE_TIMEOUT_S = 4.0
"""A page that is answering a question has one thread busy and this one waiting; four
seconds is long enough for a loopback JSON body and short enough that a view polling it
does not stack up."""


def _scraped(source: str) -> dict[str, Any]:
    """One ``/metrics`` body, fetched from here rather than from the browser.

    ``http`` and ``https`` only, and the URL is whatever a person typed into the view --
    which is the same trust as the address bar of the browser they typed it in, since this
    route is already behind the UI header and a session. A page that is not there, or is
    not JSON, is a note and not a 500: the view says what it could not read and keeps
    polling, because the usual reason is that the page has not been started yet.
    """
    if not source.startswith(("http://", "https://")):
        return {"error": "a /metrics address starts with http:// or https://"}
    try:
        with open_stream(source, timeout=SCRAPE_TIMEOUT_S) as got:
            raw = got.read(1_000_000)
        return {"serving": True, "metrics": json.loads(raw or b"{}")}
    except (ServerError, OSError, ValueError) as exc:
        return {"serving": False, "error": f"{type(exc).__name__}: {str(exc)[:160]}"}



class UI:
    """Route handling for ``/ui/*``. Holds the session store and the login throttle."""

    def __init__(self, *, name: str = "", cluster_key_path: Path | str | None = None,
                 peer_port: int = 8770, setup_token: str = "",
                 on_join: "Any | None" = None) -> None:
        self.runner: Any = None
        self.parts: Any = None
        """The page's components, `page.COMPONENTS` unless a caller leaves some out."""
        self.schedule: Any = None
        self.settings: Any = None
        self.settings_path: Any = None
        self.schedule_path: Any = None
        self.report: Any = None
        self.environment: Any = None
        self.serving: Any = None
        self.models: Any = None
        self.conversations: Any = None
        self.downloads: Any = None
        self.root: Any = None
        self.hosting: Any = None
        self.detach: Any = None
        """How a sweep is started: the bench's own `detach` unless a test hands in a fake."""
        self.answers: Any = None
        """``() -> dict``: this process's own answering telemetry, when it answers anything.
        `AskRoutes.metrics` is that callable. None on a daemon that only serves models,
        which is most of them -- the Telemetry view is then pointed at a page that does."""
        self.discovery_port: int | None = None
        self.persist_with: Any = None
        self.rename: Any = None
        """``(str) -> str``: renames this machine now. Set by the daemon that owns it."""
        self.name = name
        self.on_join = on_join
        self.cluster_key_path = cluster_key_path
        self.peer_port = peer_port
        self.setup_token = setup_token
        self.sessions = Sessions()
        self.throttle = Throttle()
        self._peers: tuple[float, list[dict[str, Any]]] = (0.0, [])

    # -- guards ----------------------------------------------------------
    def host_ok(self, host_header: str) -> bool:
        host = (host_header or "").rsplit(":", 1)[0].strip("[]").lower()
        return host in LOOPBACK or host == "" or not _looks_like_dns(host)

    def may_setup(self, client_ip: str, host_header: str, token: str) -> str:
        """Empty when first-run setup is allowed from here, else the reason it is not."""
        if not self.host_ok(host_header):
            return ("refused: this request arrived addressed to a hostname rather than "
                    "to the machine itself, which is how a web page tries to reach your "
                    "loopback. Open the address bar and type it yourself.")
        if client_ip in LOOPBACK:
            return ""
        if self.setup_token and token and _same(token, self.setup_token):
            return ""
        return ("refused: set this machine up on the machine itself, or over ssh with "
                "'ml-stack-peers setup'. A daemon that has not joined a cluster has no "
                "password to check, so the first person to reach this page would own "
                "the box. Start it with --setup-from-lan to allow this deliberately.")

    def authed(self, cookie_header: str) -> bool:
        return self.sessions.get(parse_cookie(cookie_header)) is not None

    # -- state -----------------------------------------------------------
    def state(self) -> dict[str, Any]:
        joined = in_cluster(self.cluster_key_path)
        done = bool(self.settings and self.settings.setup_done)
        return {"in_cluster": joined, "name": self.name,
                "group": cluster_group(self.cluster_key_path) if joined else None,
                "needs_password": joined,
                "needs_setup": not (joined or done)}

    def setup_finished(self) -> dict[str, Any]:
        """Remember that the wizard was finished, so it is not shown again."""
        if self.settings is not None:
            self.settings.setup_done = True
            if self.settings_path is not None:
                self.settings.save(self.settings_path)
        return self.state()

    def rejoined(self) -> None:
        """The set of clusters changed: advertise on the new one, drop the old."""
        self._peers = (0.0, [])
        if self.on_join is not None:
            with contextlib.suppress(Exception):
                self.on_join()

    def peers(self, *, force: bool = False) -> list[dict[str, Any]]:
        """Everyone on the LAN, cached briefly. The browser cannot do this itself."""
        from .discovery import memberships

        age, cached = self._peers
        if not force and time.time() - age < DISCOVER_CACHE_S:
            return cached
        joined = memberships(self.cluster_key_path)
        if not joined:
            return []
        # One machine in two of your clusters is one machine, listed once, with the
        # clusters you share with it. Each cluster is advertised separately, so the
        # same daemon answers on each with a beacon of its own.
        by_address: dict[str, dict[str, Any]] = {}
        for member in joined:
            for beacon in discover(member.key, timeout_s=1.5, port=self.discovery_port):
                row = by_address.get(beacon.base_url)
                if row is None:
                    row = beacon.public()
                    row["host"] = beacon.host
                    row["base_url"] = beacon.base_url
                    row["is_self"] = beacon.name == self.name
                    row["clusters"] = []
                    by_address[beacon.base_url] = row
                if member.group not in row["clusters"]:
                    row["clusters"].append(member.group)
        found = sorted(by_address.values(),
                       key=lambda r: (not r["is_self"], r["name"]))
        self._peers = (time.time(), found)
        return found

    # -- actions ---------------------------------------------------------
    def join(self, passphrase: str, group: str, source: str) -> tuple[dict[str, Any], str]:
        """Join a cluster, and sign the person in. Returns ``(state, session id)``."""
        held = self.throttle.blocked_for(source)
        if held:
            raise DiscoveryError(f"too many attempts -- wait {held:.0f}s")
        if not self.throttle.acquire():
            raise DiscoveryError("busy deriving another key; try again in a moment")
        try:
            join_cluster(passphrase, group=group or "ml-stack",
                         path=self.cluster_key_path)
        finally:
            self.throttle.release()
        self.throttle.succeeded(source)
        self._peers = (0.0, [])
        if self.on_join is not None:
            try:
                self.on_join()
            except Exception:                         # noqa: BLE001
                pass
        return self.state(), self.sessions.open("setup").sid

    def apply_prefs(self, req: dict[str, Any]) -> dict[str, Any]:
        """Apply the wizard's preference step. Everything takes effect now."""
        from . import autostart as auto

        out: dict[str, Any] = {"applied": [], "manual": ""}
        settings = self.settings
        if settings is None:
            return out

        if "slots" in req and self.runner is not None:
            settings.slots = self.runner.set_slots(1)
            out["applied"].append("one job at a time")
        if "name" in req and str(req["name"]).strip():
            called = str(req["name"]).strip()[:64]
            settings.name = self.rename(called) if callable(self.rename) else called
            self.name = settings.name
            out["applied"].append(f"this machine is called {settings.name}")
        if "labels" in req:
            settings.labels = [str(s) for s in req["labels"] if str(s).strip()]
            out["applied"].append("this machine is for " + (
                " and ".join(settings.labels) or "anything"))
        if "on_paused" in req and req["on_paused"] in ("stop", "finish"):
            settings.on_paused = req["on_paused"]
        if "on_close" in req and req["on_close"] in ("", "background", "quit"):
            settings.on_close = req["on_close"]
        if "auto_update" in req:
            settings.auto_update = bool(req["auto_update"])
        if "autodownload_models" in req:
            settings.autodownload_models = bool(req["autodownload_models"])
        if "context" in req:
            settings.context = max(512, min(1 << 20, int(req["context"])))

        if req.get("work_hours") and self.schedule is not None:
            spec = str(req.get("work_hours_spec") or "mon-fri 09:00-17:00")
            from .availability import parse_window
            try:
                self.schedule.windows.append(parse_window(spec))
                out["applied"].append(f"not taking work {spec}")
            except ValueError as exc:
                out["error"] = str(exc)
        if self.schedule is not None and self.schedule_path is not None:
            self.schedule.save(self.schedule_path)

        mode = str(req.get("autostart") or "")
        if mode in auto.MODES:
            settings.autostart = mode
            done = auto.install(mode, slots=settings.slots,
                                labels=tuple(settings.labels))
            if done.installed:
                out["applied"].append({"boot": "starts with the computer",
                                       "login": "starts when you log in",
                                       "manual": "starts only when you open it"}[mode])
            else:
                out["manual"] = done.command
                out["manual_why"] = done.note

        if self.settings_path is not None:
            settings.save(self.settings_path)
        return out

    def _hosting(self) -> Any:
        if self.hosting is None:
            from .join import DEFAULT_ROOT
            from .serving import Hosting

            self.hosting = Hosting(self.root or DEFAULT_ROOT, self.serving)
        return self.hosting

    def start_serving(self, model: Any) -> Any:
        """Run ``model`` on this machine and tell the network it is here."""
        return self._hosting().start(
            model.path, name=model.name,
            context=int(getattr(self.settings, "context", 0) or 8192))

    def stop_serving(self, port: int) -> None:
        """Stop a model server this machine started."""
        self._hosting().stop(port)

    # -- the fleet, and a sweep over it ----------------------------------
    def fleet(self) -> dict[str, Any]:
        """The peers as `join.describe` rows -- serving, room, busy, commit -- and the
        models they hold between them, which is what a sweep can be built from."""
        from .discovery import Beacon
        from .join import describe

        rows = []
        for row in self.peers():
            beacon = Beacon(name=str(row.get("name") or ""), port=int(row.get("port") or 8770),
                            device=dict(row.get("device") or {}), busy=bool(row.get("busy")),
                            queued=int(row.get("queued") or 0), slots=int(row.get("slots") or 1),
                            free=int(row.get("free", 1)), host=str(row.get("host") or ""),
                            hostname=str(row.get("hostname") or ""))
            rows.append(describe(beacon, clusters=row.get("clusters") or [],
                                 self_name=self.name))
        models = sorted({m for r in rows for m in r["models"]})
        return {"peers": rows, "models": models, "self": self.name,
                "group": cluster_group(self.cluster_key_path), "bench": self.bench_state()}

    def join_fleet(self, *, passphrase: str = "", group: str = "", persist: bool = False,
                   name: str = "") -> dict[str, Any]:
        """The Join button: `join.join_machine`, with this daemon as the one already up."""
        from .discovery import join as join_cluster
        from .join import DEFAULT_ROOT, join_machine

        def enrol(words: str, named: str) -> None:
            join_cluster(words, group=named, path=self.cluster_key_path)
            self.rejoined()

        said: list[str] = []
        joined = join_machine(name=name or self.name, passphrase=passphrase, group=group,
                              persist=persist, port=self.peer_port,
                              root=self.root or DEFAULT_ROOT,
                              cluster_key_path=self.cluster_key_path, enrol=enrol,
                              persist_with=self.persist_with, say=said.append,
                              discovery_port=self.discovery_port)
        self._peers = (0.0, [])
        return {**joined.public(), "said": said}

    def bench_state(self) -> dict[str, Any]:
        """What ``ml-stack-bench status`` says, for the page. The bench's home is
        `ml_stack.bench.home_dir`, the one the command reads."""
        try:
            from ml_stack.bench.run import measuring, status
        except ImportError as exc:
            return {"available": False, "text": f"the bench is not installed here: {exc}",
                    "measuring": None}
        return {"available": True, "text": status(), "measuring": measuring()}

    def bench_history(self, limit: int = 20) -> list[dict[str, Any]]:
        from dataclasses import asdict

        from ml_stack.bench import home_dir
        from ml_stack.bench.history import history

        return [asdict(e) for e in history(home_dir())][::-1][:limit]

    def fit(self, room: int = 0, users: int = 1) -> dict[str, Any]:
        """The measured fit records, seated in a room of this size.

        Everything the Fit view draws, worked out here: `fit.records()` as it comes off
        disk -- measured at load, never estimated -- each with what `Fit.loaded`, `Fit.cost`,
        `Fit.users` and `Fit.longest` say about a machine with ``room`` bytes and ``users``
        on it. ``room`` defaults to `hub.room()`, which is what a model may actually use
        here rather than the installed RAM.
        """
        try:
            from ml_stack.hub import room as room_here
            from ml_stack.serve import fit as fit_mod
        except ImportError as exc:                        # a device-tier install has no serve
            return {"error": f"this install cannot measure or read fits: {exc}",
                    "records": [], "room": 0, "at_room": 0, "name": self.name}
        here = room_here()
        asked = int(room) or here or FALLBACK_ROOM
        people = max(1, int(users))
        rows = []
        for one in fit_mod.records():
            at = one.at_room(asked)
            rows.append({**one.as_dict(), "loaded": at.loaded(), "free": at.free(),
                         "longest": at.longest(people),
                         "seats": [at.users(c) for c in fit_mod.READ_CONTEXTS],
                         "costs": [at.cost(c) for c in fit_mod.READ_CONTEXTS]})
        return {
            "records": rows,
            "room": here,
            "at_room": asked,
            "users": people,
            "name": self.name,
            "vram_gb": list(fit_mod.COMMON_VRAM_GB),
            "contexts": list(fit_mod.PLOT_CONTEXTS),
            "steps": list(fit_mod.SLIDER_CONTEXTS),
            "ladder": list(fit_mod.READ_CONTEXTS),
        }

    def rates(self) -> dict[str, Any]:
        """Every kept bench run as a point: what it scored, what it cost, and where the
        frontier runs -- ``ml-stack-bench show --rates`` as data rather than as a table.

        Nothing here measures anything. `keep._kept` reads the store the command reads,
        `score.derived` turns one run into the rates it already prints, `score.composed`
        adds each model once more as the command does, and `show.pareto` marks the runs
        nothing beats on both axes. The frontier is worked out for *all three* costs and
        sent with each point, so switching the axis on the page costs no round trip -- the
        same reason the fit records carry their two composing numbers rather than answers.
        """
        try:
            from ml_stack.bench import home_dir
            from ml_stack.bench.keep import _kept
            from ml_stack.bench.score import COSTS, NOISE, composed, derived, host_of
            from ml_stack.bench.show import AXES, pareto
        except ImportError as exc:                       # a device-tier install has no bench
            return {"error": f"the bench is not installed here: {exc}",
                    "runs": [], "axes": {}, "keys": {}, "store": ""}
        store = home_dir() / "runs.ladybug"
        kept = _kept(store)
        points = list(kept) + composed(kept)
        # by identity, the way `rates` marks them: two runs can agree on every number and
        # still be two runs
        front = {cost: {id(one) for one in pareto(points, cost=cost)} for cost in AXES}
        rows = []
        for one in points:
            got = derived(one)
            if not got:
                continue
            rows.append({
                "label": str(one.get("label") or ""),
                "host": host_of(one),
                "composed": bool(one.get("composed")),
                "from": str(one.get("from") or ""),
                "front": [cost for cost in AXES if id(one) in front[cost]],
                **{k: v for k, v in got.items()},
            })
        rows.sort(key=lambda r: -r.get("right", 0.0))
        return {"runs": rows, "axes": dict(AXES), "keys": dict(COSTS), "store": str(store),
                "noise": NOISE}

    def telemetry(self, source: str = "") -> dict[str, Any]:
        """What has been answered: this daemon's own record, or another page's ``/metrics``.

        The fleet daemon serves models and does not usually answer questions itself, so
        there is normally nothing local to show -- a host that does answer hangs its
        handler's ``metrics`` on ``ui.answers`` and it appears here. Either way the view
        can be pointed at a page that does answer, by its ``/metrics`` URL, and *this*
        process fetches it: a page on loopback has no reason to allow a cross-origin read,
        and asking the browser to do it would fail silently in exactly the case it is
        wanted. Nothing here measures anything; it reads a number someone else wrote down.
        """
        if source:
            return {"source": source, **_scraped(source)}
        answers = getattr(self, "answers", None)
        if callable(answers):
            try:
                return {"source": "", "serving": True, "metrics": answers()}
            except Exception as exc:  # noqa: BLE001 - a broken counter is not a broken page
                return {"source": "", "serving": True, "error": str(exc)[:200]}
        return {"source": "", "serving": False,
                "note": ("this daemon answers no questions itself, so it has spent nothing. "
                         "Point this at a page that does: its address and /metrics.")}

    def start_sweep(self, argv: list[str]) -> dict[str, Any]:
        """Start ``ml-stack-bench argv`` detached and hand back its log and pid."""
        import shlex

        from ml_stack.bench.run import detach, measuring_file

        log = (self.detach or detach)(argv)
        try:
            held = json.loads(measuring_file().read_text(encoding="utf-8"))
        except (OSError, ValueError):
            held = {}
        return {"log": str(log), "pid": held.get("pid"), "argv": list(argv),
                "command": "ml-stack-bench " + shlex.join(argv)}

    def stop_sweep(self, pid: int) -> str:
        """Stop the detached measurement, but only the one the page was shown."""
        from ml_stack.bench.run import measuring, stop

        held = measuring()
        if held is None or int(held.get("pid") or 0) != int(pid):
            return "nothing is measuring under that pid"
        return stop()

    def install_update(self) -> dict[str, Any]:
        """Put the newest release in place and start it. Returns what happened."""
        from .updates import apply_if_newer, relaunch

        got = apply_if_newer()
        if got.get("installed"):
            got["restarting"] = relaunch()
        return got

    def login(self, source: str, *, passphrase: str = "", group: str = "",
              token: str = "", ticket: str = "") -> str | None:
        """A session id, or None. Raises ``DiscoveryError`` when held off or overloaded."""
        if ticket:
            return self.sessions.open("ticket").sid if self.sessions.spend_ticket(ticket) else None

        key = load_cluster_key(self.cluster_key_path)
        if key is None:
            return None

        if token:
            if _same(token, derive_token(key)):
                self.throttle.succeeded(source)
                return self.sessions.open("token").sid
            self.throttle.failed(source)
            return None

        held = self.throttle.blocked_for(source)
        if held:
            raise DiscoveryError(f"too many attempts -- wait {held:.0f}s")
        if not self.throttle.acquire():
            raise DiscoveryError("busy checking another passphrase; try again")
        try:
            ok = check_passphrase(passphrase, group=group or None,
                                  path=self.cluster_key_path)
        finally:
            self.throttle.release()
        if not ok:
            self.throttle.failed(source)
            return None
        self.throttle.succeeded(source)
        return self.sessions.open("passphrase").sid


def _same(a: str, b: str) -> bool:
    import hmac
    return hmac.compare_digest(a.strip(), b.strip())


def _looks_like_dns(host: str) -> bool:
    """Whether a Host header names something that had to be resolved."""
    if not host or host.replace(".", "").isdigit():
        return False
    return "." in host and not host.endswith(".local")




# ------------------------------------------------------------------ a page on its own

class _Loopback(BaseHTTPRequestHandler):
    """The handler `serve_page` mounts: `routes` and nothing else."""

    ui: "UI"
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:
        if self.path in ("/", ""):
            write(self, 302, b"", "application/octet-stream", {"Location": "/ui/fit"})
            return
        if not routes(self.ui, self):
            write_json(self, 404, {"error": "no such route"})

    do_POST = do_GET

    def log_message(self, *args: Any) -> None:   # a served page is not a log line
        return


def serve_page(*, port: int = 0, name: str = "", host: str = "127.0.0.1") -> Any:
    """A loopback-only server over `routes` alone -- what ``ml-stack-serve fit --ui`` puts up.

    The same ``UI`` object and the same route table the daemon mounts, with nothing else
    on it: no jobs, no models, no peers, and a cluster key path that deliberately does not
    exist, so a machine that *is* in a cluster is not asked to sign in to look at its own
    measurements. It binds loopback, which is what makes that safe: `may_setup` lets this
    machine in and there is no address anyone else could reach.

    Returns the server, not yet serving; the caller decides whether that is a thread or
    this one. ``port=0`` takes whatever is free, and ``server.server_port`` says which.
    """
    import platform as _platform
    import tempfile
    from http.server import ThreadingHTTPServer

    ui = UI(name=name or _platform.node() or "this machine",
            cluster_key_path=Path(tempfile.gettempdir()) / "ml-stack-fit-no-cluster")
    ui.parts = FIT_ONLY
    handler = type("FitHandler", (_Loopback,), {"ui": ui})
    return ThreadingHTTPServer((host, port), handler)
