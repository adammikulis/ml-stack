"""The fleet's web interface: routes, and the guard on first-run setup."""

from __future__ import annotations

import contextlib
import json
import mimetypes
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any

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
from .session import Sessions, Throttle, parse_cookie

__all__ = ["ASSETS", "UI", "asset_bytes", "serve_page"]

ASSETS = Path(__file__).parent / "web"

UI_HEADER = "X-ML-Stack-UI"
"""Required on every UI request. A cross-origin form, image or link cannot set a custom"""

LOOPBACK = {"127.0.0.1", "::1", "localhost"}
DISCOVER_CACHE_S = 3.0
"""Long enough that three panels refreshing do not each fire a multicast sweep, short"""


def asset_bytes(name: str) -> tuple[bytes, str] | None:
    """One file from ``web/``, by exact name."""
    allowed = {p.name: p for p in ASSETS.iterdir() if p.is_file()} if ASSETS.is_dir() else {}
    path = allowed.get(name)
    if path is None:
        return None
    kind, _ = mimetypes.guess_type(name)
    return path.read_bytes(), kind or "application/octet-stream"


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
    import urllib.request

    if not source.startswith(("http://", "https://")):
        return {"error": "a /metrics address starts with http:// or https://"}
    try:
        with urllib.request.urlopen(source, timeout=SCRAPE_TIMEOUT_S) as got:
            raw = got.read(1_000_000)
        return {"serving": True, "metrics": json.loads(raw or b"{}")}
    except (OSError, ValueError) as exc:
        return {"serving": False, "error": f"{type(exc).__name__}: {str(exc)[:160]}"}


def app_location() -> Path | None:
    """The bundle this is running from, which cannot delete itself."""
    import sys
    if not getattr(sys, "frozen", False):
        return None
    here = Path(sys.executable).resolve()
    for parent in here.parents:
        if parent.suffix == ".app":
            return parent
    return here


def _can_serve() -> bool:
    """Whether this install has the code to run a model server itself."""
    from importlib.util import find_spec
    return find_spec("ml_stack.serve") is not None


class UI:
    """Route handling for ``/ui/*``. Holds the session store and the login throttle."""

    def __init__(self, *, name: str = "", cluster_key_path: Path | str | None = None,
                 peer_port: int = 8770, setup_token: str = "",
                 on_join: "Any | None" = None) -> None:
        self.runner: Any = None
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
        self.servers: Any = None
        self._leases: dict[int, Any] = {}
        self.detach: Any = None
        """How a sweep is started: the bench's own `detach` unless a test hands in a fake."""
        self.answers: Any = None
        """``() -> dict``: this process's own answering telemetry, when it answers anything.
        `AskRoutes.metrics` is that callable. None on a daemon that only serves models,
        which is most of them -- the Telemetry view is then pointed at a page that does."""
        self.discovery_port: int | None = None
        self.persist_with: Any = None
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

    def start_serving(self, model: Any) -> Any:
        """Run ``model`` on this machine and tell the network it is here."""
        from .serving import start_model

        started = start_model(
            self.root, model.path, name=model.name,
            context=int(getattr(self.settings, "context", 0) or 8192),
            manager=self.servers, serving=self.serving)
        self.servers = started.manager
        self._leases[started.port] = started.lease
        return started.served

    def stop_serving(self, port: int) -> None:
        """Stop a model server this machine started."""
        from .serving import Started, stop_model

        held = self._leases.pop(port, None)
        if held is None or self.servers is None:
            self.serving.unregister(port)
            return
        stop_model(Started(port=port, lease=held, manager=self.servers),
                   serving=self.serving)

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
        `ml_stack.graph.bench.HOME`, the one the command reads."""
        try:
            from ml_stack.graph.bench.run import measuring, status
        except ImportError as exc:
            return {"available": False, "text": f"the bench is not installed here: {exc}",
                    "measuring": None}
        return {"available": True, "text": status(), "measuring": measuring()}

    def bench_history(self, limit: int = 20) -> list[dict[str, Any]]:
        from dataclasses import asdict

        from ml_stack.graph.bench import HOME
        from ml_stack.graph.bench.history import history

        return [asdict(e) for e in history(HOME)][::-1][:limit]

    def fit(self) -> dict[str, Any]:
        """The measured fit records, this machine's room, and the rooms worth drawing.

        Everything the Fit view needs in one answer: `fit.records()` as it comes off disk
        -- measured at load, never estimated -- plus `hub.room()`, which is what a model
        may actually use here rather than the installed RAM. The page does the arithmetic
        itself, from the same two numbers per record that `Fit.line` composes, so moving
        the room slider costs no round trip.
        """
        try:
            from ml_stack.hub import room
            from ml_stack.serve import fit as fit_mod
        except ImportError as exc:                        # a device-tier install has no serve
            return {"error": f"this install cannot measure or read fits: {exc}",
                    "records": [], "room": 0, "name": self.name}
        return {
            "records": [f.as_dict() for f in fit_mod.records()],
            "room": room(),
            "name": self.name,
            "vram_gb": list(fit_mod.COMMON_VRAM_GB),
            "contexts": list(fit_mod.PLOT_CONTEXTS),
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
            from ml_stack.graph.bench import HOME
            from ml_stack.graph.bench.keep import _kept
            from ml_stack.graph.bench.score import COSTS, NOISE, composed, derived, host_of
            from ml_stack.graph.bench.show import AXES, pareto
        except ImportError as exc:                       # a device-tier install has no bench
            return {"error": f"the bench is not installed here: {exc}",
                    "runs": [], "axes": {}, "keys": {}, "store": ""}
        store = Path(HOME) / "runs.ladybug"
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

        from ml_stack.graph.bench.run import detach, measuring_file

        log = (self.detach or detach)(argv)
        try:
            held = json.loads(measuring_file().read_text(encoding="utf-8"))
        except (OSError, ValueError):
            held = {}
        return {"log": str(log), "pid": held.get("pid"), "argv": list(argv),
                "command": "ml-stack-bench " + shlex.join(argv)}

    def stop_sweep(self, pid: int) -> str:
        """Stop the detached measurement, but only the one the page was shown."""
        from ml_stack.graph.bench.run import measuring, stop

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


def routes(ui: UI, handler: Any) -> bool:
    """Handle one ``/ui/*`` request. Returns True when it did."""
    parsed = urllib.parse.urlparse(handler.path)
    path = parsed.path
    if not path.startswith("/ui"):
        return False

    client_ip = handler.client_address[0]
    host_header = handler.headers.get("Host", "")
    has_ui_header = bool(handler.headers.get(UI_HEADER))
    cookie = handler.headers.get("Cookie", "")
    method = handler.command

    def send(code: int, payload: Any, extra: dict[str, str] | None = None) -> None:
        handler._send(code, payload, headers=extra)

    def body() -> dict[str, Any]:
        length = int(handler.headers.get("Content-Length", "0"))
        try:
            return json.loads(handler.rfile.read(length) or b"{}")
        except ValueError:
            return {}

    # -- static ----------------------------------------------------------
    if path in ("/ui", "/ui/"):
        asset = asset_bytes("index.html")
        if asset is None:
            send(500, {"error": "the UI assets are missing from this install"})
            return True
        raw, kind = asset
        handler._send(200, None, raw=raw, content_type=kind)
        return True

    if path in ("/ui/fit", "/ui/fit/"):
        asset = asset_bytes("fit.html")
        if asset is None:
            send(500, {"error": "the UI assets are missing from this install"})
            return True
        raw, kind = asset
        handler._send(200, None, raw=raw, content_type=kind)
        return True

    if path.startswith("/ui/static/"):
        asset = asset_bytes(path[len("/ui/static/"):])
        if asset is None:
            send(404, {"error": "no such asset"})
            return True
        raw, kind = asset
        handler._send(200, None, raw=raw, content_type=kind,
                      headers={"Cache-Control": "no-cache"})
        return True

    # Everything below is API, and none of it is reachable from a cross-origin page.
    if not has_ui_header:
        send(403, {"error": f"{UI_HEADER} header required"})
        return True

    # -- setup -----------------------------------------------------------
    if path == "/ui/setup" and method == "GET":
        send(200, ui.state())
        return True

    if path == "/ui/setup/join" and method == "POST":
        joined = in_cluster(ui.cluster_key_path)
        if joined and not ui.authed(cookie):
            send(401, {"error": "this machine is already in a cluster -- sign in to "
                                "move it to another one"})
            return True
        if not joined:
            why = ui.may_setup(client_ip, host_header,
                               handler.headers.get("X-ML-Stack-Setup", ""))
            if why:
                send(403, {"error": why})
                return True
        req = body()
        try:
            state, sid = ui.join(str(req.get("passphrase") or ""),
                                 str(req.get("group") or ""), client_ip)
        except DiscoveryError as exc:
            send(429 if "attempts" in str(exc) or "busy" in str(exc) else 400,
                 {"error": str(exc)})
            return True
        session = ui.sessions.get(sid)
        send(200, state,
             {"Set-Cookie": ui.sessions.cookie_header(session)} if session else None)
        return True

    if path == "/ui/setup/suggest" and method == "GET":
        from .settings import suggest
        report = ui.report() if callable(ui.report) else {}
        send(200, {
            "machine": {k: report.get(k) for k in
                        ("gpu", "vendor", "cpus", "ram_gb", "accelerator",
                         "vram_total_gb", "temp_c")},
            "suggest": {k: {"value": s.value, "why": s.why}
                        for k, s in suggest(report).items()},
            "current": ui.settings.public() if ui.settings else {},
        })
        return True

    if path in ("/ui/setup/prefs", "/ui/setup/done") and method == "POST":
        if in_cluster(ui.cluster_key_path):
            if not ui.authed(cookie):
                send(401, {"error": "sign in first"})
                return True
        else:
            why = ui.may_setup(client_ip, host_header,
                               handler.headers.get("X-ML-Stack-Setup", ""))
            if why:
                send(403, {"error": why})
                return True
        if path == "/ui/setup/done":
            send(200, ui.setup_finished())
            return True
        req = body()
        applied = ui.apply_prefs(req)
        send(200, applied)
        return True

    if path == "/ui/setup/peers" and method == "GET":
        send(200, {"peers": ui.peers(force=True)})
        return True

    # -- session ---------------------------------------------------------
    if path == "/ui/session":
        if method == "GET":
            session = ui.sessions.get(parse_cookie(cookie))
            send(200, {"signed_in": session is not None,
                       "expires_at": session.expires_at if session else None})
            return True
        if method == "DELETE":
            ui.sessions.close(parse_cookie(cookie))
            send(200, {"signed_in": False},
                 {"Set-Cookie": ui.sessions.clear_header()})
            return True
        if method == "POST":
            req = body()
            try:
                sid = ui.login(client_ip,
                               passphrase=str(req.get("passphrase") or ""),
                               group=str(req.get("group") or ""),
                               token=str(req.get("token") or ""),
                               ticket=str(req.get("ticket") or ""))
            except DiscoveryError as exc:
                send(429, {"error": str(exc)})
                return True
            if sid is None:
                send(401, {"error": "that passphrase does not match this cluster"})
                return True
            session = ui.sessions.get(sid)
            send(200, {"signed_in": True, "expires_at": session.expires_at},
                 {"Set-Cookie": ui.sessions.cookie_header(session)})
            return True

    # -- everything else needs a session ---------------------------------
    # A machine in no cluster has no password anyone could be asked for, so it answers
    # to itself and to nobody else.
    if not ui.authed(cookie):
        if in_cluster(ui.cluster_key_path):
            send(401, {"error": "sign in first"})
            return True
        why = ui.may_setup(client_ip, host_header,
                           handler.headers.get("X-ML-Stack-Setup", ""))
        if why:
            send(403, {"error": why})
            return True

    if path == "/ui/fit.json" and method == "GET":
        send(200, ui.fit())
        return True

    if path == "/ui/rates.json" and method == "GET":
        send(200, ui.rates())
        return True

    if path == "/ui/telemetry.json" and method == "GET":
        asked = urllib.parse.parse_qs(parsed.query)
        send(200, ui.telemetry(str(asked.get("from", [""])[0] or "")[:400]))
        return True

    if path == "/ui/settings":
        if method == "GET":
            from . import autostart as auto
            from .updates import current_version
            send(200, {
                "settings": ui.settings.public() if ui.settings else {},
                "name": ui.name,
                "group": cluster_group(ui.cluster_key_path),
                "version": current_version(),
                "autostart": auto.status(),
                "schedule": ui.schedule.public() if ui.schedule else {},
                "machine": ui.report() if callable(ui.report) else {},
            })
            return True
        if method == "POST":
            send(200, ui.apply_prefs(body()))
            return True

    if path == "/ui/libraries":
        if ui.environment is None:
            send(501, {"error": "no environment on this daemon"})
            return True
        report = ui.report() if callable(ui.report) else {}
        vendor = str(report.get("vendor") or "cpu")
        if method == "GET":
            send(200, ui.environment.state(vendor))
            return True
        if method == "POST":
            req = body()
            add = [str(s) for s in req.get("install") or []]
            drop = [str(s) for s in req.get("remove") or []]
            out: dict[str, Any] = {}
            try:
                if drop:
                    out.update(ui.environment.uninstall(drop))
                if add:
                    out.update(ui.environment.install(add))
            except EnvironmentError as exc:
                send(400, {"error": str(exc)})
                return True
            send(200, {"changed": out, **ui.environment.state(vendor)})
            return True

    if path == "/ui/models/popular" and method == "GET":
        if ui.models is None:
            send(501, {"error": "no model store on this daemon"})
            return True
        from .models import (
            PER_PAGE, families, how_many, popular, searched_count,
            searched_families)
        asked = urllib.parse.parse_qs(parsed.query)
        page = max(0, int((asked.get("page", ["0"])[0] or "0")))
        rude = asked.get("rude", ["0"])[0] in ("1", "true", "yes")
        query = (asked.get("q", [""])[0] or "").strip()
        ram = float((ui.report() if ui.report else {}).get("ram_gb") or 0)
        free = ui.models.free_gb()
        here = {m.name for m in ui.models.all()}
        found = [x for x in popular(free, ram, page=page, rude=rude, query=query)
                 if x.file not in here]
        if query:
            total = searched_count(query, free, ram, rude=rude)
            names = searched_families(query, free, ram, rude=rude)
        else:
            total = how_many(free, ram, rude=rude)
            # Across the whole list: a family further down still needs a box.
            names = families(free, ram, rude=rude)
        send(200, {"models": [x.public() for x in found], "families": names,
                   "page": page, "pages": max(1, -(-total // PER_PAGE)),
                   "total": total, "q": query})
        return True

    if path == "/ui/models":
        if ui.models is None:
            send(501, {"error": "no model store on this daemon"})
            return True
        from .discovery import load_cluster_key
        from .models import ModelError
        key = load_cluster_key(ui.cluster_key_path)
        auto_models = ui.settings is None or ui.settings.autodownload_models
        if method == "GET":
            here = {m.name for m in ui.models.all()}
            free = ui.models.free_gb()
            elsewhere: dict[str, list[str]] = {}
            for beacon in (ui.peers() if key is not None else []):
                if beacon.get("is_self"):
                    continue
                for row in (beacon.get("device", {}).get("models") or []):
                    elsewhere.setdefault(str(row.get("name")), []).append(
                        str(beacon.get("name")))
            send(200, {
                "here": [m.public() for m in ui.models.all()],
                "elsewhere": [{"name": n, "peers": p}
                              for n, p in sorted(elsewhere.items()) if n not in here],
                "free_gb": free,
                "autodownload": auto_models,
                "unfinished": ui.models.unfinished(),
                "getting": [g.public() for g in ui.downloads.active()]
                           if ui.downloads is not None else [],

            })
            return True
        if method == "DELETE":
            send(200, {"discarded": ui.models.discard(str(body().get("name") or ""))})
            return True
        if method == "POST":
            req = body()
            name = str(req.get("name") or "")
            if not name:
                send(400, {"error": "no model was named"})
                return True
            if ui.downloads is None:
                try:
                    got = ui.models.ensure(
                        name, source=str(req.get("source") or ""), key=key,
                        autodownload=auto_models)
                except (ModelError, ValueError) as exc:
                    send(400, {"error": str(exc)})
                    return True
                send(200, got.public())
                return True
            here = ui.models.find(name)
            if here is not None:
                send(200, here.public())
                return True
            started = ui.downloads.start(name, source=str(req.get("source") or ""),
                                         key=key, autodownload=auto_models,
                                         draft=str(req.get("draft") or ""))
            send(202, started.public())
            return True

    if path == "/ui/serving/install" and method == "POST":
        if ui.root is None:
            send(501, {"error": "this daemon does not know where to keep it"})
            return True
        if not _can_serve():
            send(501, {"error": "this install cannot run a model itself; a machine "
                                "on your network can serve one instead"})
            return True
        from .llama import LlamaError, ensure_server
        try:
            got = ensure_server(ui.root)
        except LlamaError as exc:
            send(400, {"error": str(exc)})
            return True
        send(200, {"ok": True, "server": str(got)})
        return True

    if path == "/ui/serving":
        if ui.serving is None:
            send(501, {"error": "nothing on this daemon can run a model"})
            return True
        if method == "GET":
            send(200, {"running": [x.public() for x in ui.serving.live(force=True)],
                       "can_serve": _can_serve()})
            return True
        if method == "POST":
            if not _can_serve():
                send(501, {"error": "this install cannot run a model itself; a "
                                    "machine on your network can serve one instead"})
                return True
            req = body()
            found = ui.models.find(str(req.get("name") or "")) if ui.models else None
            if found is None:
                send(404, {"error": "no such model on this machine"})
                return True
            try:
                served = ui.start_serving(found)
            except (OSError, RuntimeError, ValueError) as exc:
                send(400, {"error": str(exc)})
                return True
            send(201, served.public())
            return True
        if method == "DELETE":
            ui.stop_serving(int(body().get("port") or 0))
            send(200, {"running": [x.public() for x in ui.serving.live(force=True)]})
            return True

    if path.startswith("/ui/conversations"):
        if ui.conversations is None:
            send(501, {"error": "no chat store on this daemon"})
            return True
        rest = path[len("/ui/conversations"):].strip("/")
        if not rest:
            if method == "GET":
                query = urllib.parse.parse_qs(parsed.query).get("q", [""])[0]
                send(200, {"conversations": [
                    c.public(full=False) for c in ui.conversations.search(query)]})
                return True
            if method == "POST":
                req = body()
                made = ui.conversations.start(model=str(req.get("model") or ""),
                                              title=str(req.get("title") or ""))
                send(201, made.public())
                return True
        else:
            found = ui.conversations.get(rest)
            if found is None:
                send(404, {"error": "no such chat"})
                return True
            if method == "GET":
                send(200, found.public())
                return True
            if method == "DELETE":
                ui.conversations.remove(rest)
                send(200, {"removed": rest})
                return True
            if method == "POST":
                renamed = ui.conversations.rename(rest, str(body().get("title") or ""))
                send(200, renamed.public(full=False))
                return True

    if path == "/ui/chat":
        from .chat import ChatError, find, reply_text, stream, targets
        from .discovery import derive_token, load_cluster_key
        key = load_cluster_key(ui.cluster_key_path)
        available = targets(ui.peers() if key is not None else [],
                            ui.serving, derive_token(key) if key else "")
        if method == "GET":
            send(200, {"models": [t.public() for t in available]})
            return True
        if method == "POST":
            req = body()
            target = find(available, str(req.get("model") or ""))
            if target is None:
                send(503, {"error": "no machine on this network is serving a model"})
                return True
            messages = [m for m in (req.get("messages") or [])
                        if isinstance(m, dict) and m.get("content")]
            if not messages:
                send(400, {"error": "nothing to send"})
                return True
            payload = {"model": target.model, "messages": messages, "stream": True}
            if req.get("temperature") is not None:
                payload["temperature"] = float(req["temperature"])
            try:
                pieces = stream(target, payload)
                first = next(pieces, b"")
            except ChatError as exc:
                send(502, {"error": str(exc)})
                return True

            cid = str(req.get("conversation") or "")
            if ui.conversations is not None and cid:
                ui.conversations.append(cid, "user", str(messages[-1]["content"]))

            handler.send_response(200)
            handler.send_header("Content-Type", "text/event-stream")
            handler.send_header("Cache-Control", "no-store")
            handler.send_header("X-ML-Stack-Peer", target.peer or "")
            handler.send_header("X-ML-Stack-Model", target.model)
            handler.send_header("Connection", "close")
            handler.end_headers()
            said = bytearray(first)
            try:
                handler.wfile.write(first)
                handler.wfile.flush()
                for block in pieces:
                    said += block
                    handler.wfile.write(block)
                    handler.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            if ui.conversations is not None and cid:
                spoken = reply_text(bytes(said))
                if spoken:
                    ui.conversations.append(cid, "assistant", spoken)
            return True

    if path == "/ui/uninstall":
        from .uninstall import plan, remove
        if ui.root is None:
            send(501, {"error": "this daemon does not know where it keeps things"})
            return True
        if method == "GET":
            items = plan(ui.root, key_path=ui.cluster_key_path)
            send(200, {"items": [i.public() for i in items],
                       "app": str(app_location() or "")})
            return True
        if method == "POST":
            req = body()
            keys = [str(k) for k in (req.get("remove") or [])]
            out = remove(ui.root, keys, key_path=ui.cluster_key_path)
            out["app"] = str(app_location() or "")
            send(200, out)
            return True

    if path == "/ui/updates" and method == "GET":
        from .updates import UpdateError, asset_for, check, current_version
        now = current_version()
        try:
            release = check()
        except UpdateError as exc:
            send(200, {"version": now, "checked": False, "error": str(exc)})
            return True
        asset = asset_for(release)
        send(200, {
            "version": now,
            "latest": release.version,
            "newer": release.newer_than(now),
            "known": bool(now),
            "checked": True,
            "notes": release.notes[:2000],
            "url": release.url,
            "download": (asset or {}).get("name"),
            "size": (asset or {}).get("size"),
        })
        return True

    if path == "/ui/updates/install" and method == "POST":
        send(200, ui.install_update())
        return True

    if path == "/ui/clusters":
        from .discovery import join, leave, memberships
        if method == "GET":
            send(200, {"clusters": [m.public() for m in
                                    memberships(ui.cluster_key_path)]})
            return True
        if method == "POST":
            req = body()
            words = str(req.get("passphrase") or "")
            group = str(req.get("group") or "").strip() or "ml-stack"
            try:
                rows = join(words, group=group, path=ui.cluster_key_path)
            except DiscoveryError as exc:
                send(400, {"error": str(exc)})
                return True
            ui.rejoined()
            send(200, {"clusters": [m.public() for m in rows], "joined": group})
            return True
        if method == "DELETE":
            group = str(body().get("group") or "")
            rows = leave(group, ui.cluster_key_path)
            ui.rejoined()
            send(200, {"clusters": [m.public() for m in rows], "left": group})
            return True

    if path == "/ui/peers" and method == "GET":
        send(200, {"peers": ui.peers(), "self": ui.name,
                   "group": cluster_group(ui.cluster_key_path)})
        return True

    # -- the fleet: peers with what they serve, a join, and a sweep across them ------
    if path == "/ui/fleet" and method == "GET":
        send(200, ui.fleet())
        return True

    if path == "/ui/fleet/join" and method == "POST":
        from .join import JoinError
        req = body()
        words = str(req.get("passphrase") or "")
        if not words and not req.get("persist"):
            send(400, {"error": "nothing to do: type the passphrase every machine shares, "
                                "or tick 'start at logon'"})
            return True
        try:
            send(200, ui.join_fleet(passphrase=words, group=str(req.get("group") or ""),
                                    persist=bool(req.get("persist")),
                                    name=str(req.get("name") or "")))
        except (JoinError, DiscoveryError) as exc:
            send(400, {"error": str(exc)})
        return True

    if path == "/ui/bench/sweep" and method == "POST":
        import shlex

        from .join import sweep_argv
        req = body()
        models = [str(m) for m in (req.get("models") or []) if str(m).strip()]
        if not models:
            send(400, {"error": "pick at least one model to measure"})
            return True
        try:
            argv = sweep_argv(models, peers=[str(p) for p in (req.get("peers") or [])],
                              sample=int(req.get("sample") or 0),
                              label=str(req.get("label") or ""),
                              extra=[str(a) for a in (req.get("extra") or [])])
        except ValueError as exc:
            send(400, {"error": str(exc)})
            return True
        if req.get("dry_run"):
            send(200, {"argv": argv, "command": "ml-stack-bench " + shlex.join(argv)})
            return True
        try:
            send(202, ui.start_sweep(argv))
        except ImportError as exc:
            send(501, {"error": f"the bench is not installed here: {exc}"})
        return True

    if path == "/ui/bench/status" and method == "GET":
        send(200, ui.bench_state())
        return True

    if path == "/ui/bench/history" and method == "GET":
        try:
            send(200, {"history": ui.bench_history()})
        except ImportError as exc:
            send(501, {"error": f"the bench is not installed here: {exc}"})
        return True

    if path == "/ui/bench/stop" and method == "POST":
        pid = body().get("pid")
        if not pid:
            send(400, {"error": "say which pid to stop -- the one the page was shown"})
            return True
        send(200, {"stopped": ui.stop_sweep(int(pid))})
        return True

    send(404, {"error": "no such route"})
    return True


# ------------------------------------------------------------------ a page on its own

class _Loopback(BaseHTTPRequestHandler):
    """The handler `serve_page` mounts: `routes` and nothing else.

    ``_send`` is the same shape the daemon's handler gives `routes`, because `routes` is
    what is being run -- the page and its JSON come from one implementation, not two that
    have to be kept saying the same thing.
    """

    ui: "UI"
    protocol_version = "HTTP/1.1"

    def _send(self, code: int, payload: Any, *, raw: bytes | None = None,
              headers: dict[str, str] | None = None, content_type: str = "") -> None:
        body = raw if raw is not None else json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", content_type or (
            "application/octet-stream" if raw is not None else "application/json"))
        self.send_header("Content-Length", str(len(body)))
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def do_GET(self) -> None:      # noqa: N802 - BaseHTTPRequestHandler's spelling
        if self.path in ("/", ""):
            self._send(302, None, raw=b"", headers={"Location": "/ui/fit"})
            return
        if not routes(self.ui, self):
            self._send(404, {"error": "no such route"})

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
    handler = type("FitHandler", (_Loopback,), {"ui": ui})
    return ThreadingHTTPServer((host, port), handler)
