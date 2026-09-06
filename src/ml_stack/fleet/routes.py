"""The ``/ui/*`` route table: one mixin per screen, composed into `Router`.

`Base` holds the request and how to answer it; each mixin answers the paths of one screen
and hands anything else on. `Router.run` puts them in order: the page and its files, then
setup and session, then everything a session is needed for.
"""

from __future__ import annotations

import json
import mimetypes
import urllib.parse
from importlib.util import find_spec
from pathlib import Path
from typing import Any

from .discovery import (
    DiscoveryError,
    cluster_group,
    derive_token,
    in_cluster,
    load_cluster_key,
)
from .page import COMPONENTS, render
from .pausing import minutes_of
from .session import parse_cookie

ASSETS = Path(__file__).parent / "web"

UI_HEADER = "X-ML-Stack-UI"
"""Required on every UI request. A cross-origin form, image or link cannot set a custom"""


def asset_bytes(name: str) -> tuple[bytes, str] | None:
    """One file from ``web/``, by exact name."""
    allowed = {p.name: p for p in ASSETS.iterdir() if p.is_file()} if ASSETS.is_dir() else {}
    path = allowed.get(name)
    if path is None:
        return None
    kind, _ = mimetypes.guess_type(name)
    return path.read_bytes(), kind or "application/octet-stream"


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
    return find_spec("ml_stack.serve") is not None


def _whole(text: str) -> int:
    """A query parameter as a non-negative integer; 0 for anything else."""
    try:
        return max(0, int(float(text)))
    except (TypeError, ValueError):
        return 0


def write(handler: Any, code: int, raw: bytes, content_type: str,
          extra: dict[str, str] | None = None) -> None:
    """Write one response whose body is bytes."""
    handler.send_response(code)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(raw)))
    for key, value in (extra or {}).items():
        handler.send_header(key, value)
    handler.end_headers()
    if handler.command != "HEAD":
        handler.wfile.write(raw)


def write_json(handler: Any, code: int, payload: Any,
               extra: dict[str, str] | None = None) -> None:
    """Write one JSON response."""
    write(handler, code, json.dumps(payload).encode(), "application/json", extra)


class Base:
    """One ``/ui`` request: what it asked for, and how to answer it.

    A mixin overrides one of the three phase methods and hands what it does not know to
    ``super()``. Each returns True once it has answered.
    """

    def __init__(self, ui: Any, handler: Any) -> None:
        parsed = urllib.parse.urlparse(handler.path)
        self.ui = ui
        self.handler = handler
        self.path = parsed.path
        self.query = urllib.parse.parse_qs(parsed.query)
        self.method = handler.command
        self.client_ip = handler.client_address[0]
        self.host_header = handler.headers.get("Host", "")
        self.cookie = handler.headers.get("Cookie", "")

    def header(self, name: str, default: str = "") -> str:
        """One request header."""
        return self.handler.headers.get(name, default)

    def send(self, code: int, payload: Any, extra: dict[str, str] | None = None) -> None:
        """Answer with JSON."""
        write_json(self.handler, code, payload, extra)

    def body(self) -> dict[str, Any]:
        """The request's JSON object, or an empty one."""
        length = int(self.header("Content-Length", "0") or 0)
        try:
            return json.loads(self.handler.rfile.read(length) or b"{}")
        except ValueError:
            return {}

    def asked(self, name: str, fallback: str = "") -> str:
        """One query parameter."""
        return self.query.get(name, [fallback])[0] or fallback

    def open_route(self) -> bool:
        return False

    def public_route(self) -> bool:
        return False

    def route(self) -> bool:
        return False


class PageRoutes:
    """The page and the files it loads: ``/ui``, ``/ui/fit``, ``/ui/static/*``.

    Both paths serve the same page; ``/ui/fit`` opens it on the fit view.
    """

    def open_route(self) -> bool:
        if self.path in ("/ui", "/ui/", "/ui/fit", "/ui/fit/"):
            try:
                page = render(getattr(self.ui, "parts", None) or COMPONENTS)
            except OSError:
                self.send(500, {"error": "the UI assets are missing from this install"})
                return True
            write(self.handler, 200, page.encode("utf-8"), "text/html; charset=utf-8")
            return True
        if self.path.startswith("/ui/static/"):
            asset = asset_bytes(self.path[len("/ui/static/"):])
            if asset is None:
                self.send(404, {"error": "no such asset"})
                return True
            write(self.handler, 200, asset[0], asset[1], {"Cache-Control": "no-cache"})
            return True
        return super().open_route()


class SetupRoutes:
    """First run: what this machine looks like, what to set it to, and who may say so."""

    def _may_setup(self) -> str:
        return self.ui.may_setup(self.client_ip, self.host_header,
                                 self.header("X-ML-Stack-Setup", ""))

    def public_route(self) -> bool:
        if self.path == "/ui/setup" and self.method == "GET":
            self.send(200, self.ui.state())
            return True
        if self.path == "/ui/setup/join" and self.method == "POST":
            return self._join()
        if self.path == "/ui/setup/suggest" and self.method == "GET":
            return self._suggest()
        if self.path in ("/ui/setup/prefs", "/ui/setup/done") and self.method == "POST":
            return self._prefs()
        if self.path == "/ui/setup/peers" and self.method == "GET":
            self.send(200, {"peers": self.ui.peers(force=True)})
            return True
        return super().public_route()

    def _join(self) -> bool:
        ui = self.ui
        joined = in_cluster(ui.cluster_key_path)
        if joined and not ui.authed(self.cookie):
            self.send(401, {"error": "this machine is already in a cluster -- sign in to "
                                     "move it to another one"})
            return True
        if not joined:
            why = self._may_setup()
            if why:
                self.send(403, {"error": why})
                return True
        req = self.body()
        try:
            state, sid = ui.join(str(req.get("passphrase") or ""),
                                 str(req.get("group") or ""), self.client_ip)
        except DiscoveryError as exc:
            self.send(429 if "attempts" in str(exc) or "busy" in str(exc) else 400,
                      {"error": str(exc)})
            return True
        session = ui.sessions.get(sid)
        self.send(200, state,
                  {"Set-Cookie": ui.sessions.cookie_header(session)} if session else None)
        return True

    def _suggest(self) -> bool:
        from .settings import suggest
        ui = self.ui
        report = ui.report() if callable(ui.report) else {}
        self.send(200, {
            "machine": {k: report.get(k) for k in
                        ("gpu", "vendor", "cpus", "ram_gb", "accelerator",
                         "vram_total_gb", "temp_c")},
            "suggest": {k: {"value": s.value, "why": s.why}
                        for k, s in suggest(report).items()},
            "current": ui.settings.public() if ui.settings else {},
        })
        return True

    def _prefs(self) -> bool:
        ui = self.ui
        if in_cluster(ui.cluster_key_path):
            if not ui.authed(self.cookie):
                self.send(401, {"error": "sign in first"})
                return True
        else:
            why = self._may_setup()
            if why:
                self.send(403, {"error": why})
                return True
        if self.path == "/ui/setup/done":
            self.send(200, ui.setup_finished())
            return True
        self.send(200, ui.apply_prefs(self.body()))
        return True


class SessionRoutes:
    """``/ui/session``: who is signed in, signing in, and signing out."""

    def public_route(self) -> bool:
        if self.path != "/ui/session":
            return super().public_route()
        ui = self.ui
        if self.method == "GET":
            session = ui.sessions.get(parse_cookie(self.cookie))
            self.send(200, {"signed_in": session is not None,
                            "expires_at": session.expires_at if session else None})
            return True
        if self.method == "DELETE":
            ui.sessions.close(parse_cookie(self.cookie))
            self.send(200, {"signed_in": False}, {"Set-Cookie": ui.sessions.clear_header()})
            return True
        if self.method == "POST":
            return self._sign_in()
        return super().public_route()

    def _sign_in(self) -> bool:
        ui, req = self.ui, self.body()
        try:
            sid = ui.login(self.client_ip,
                           passphrase=str(req.get("passphrase") or ""),
                           group=str(req.get("group") or ""),
                           token=str(req.get("token") or ""),
                           ticket=str(req.get("ticket") or ""))
        except DiscoveryError as exc:
            self.send(429, {"error": str(exc)})
            return True
        if sid is None:
            self.send(401, {"error": "that passphrase does not match this cluster"})
            return True
        session = ui.sessions.get(sid)
        self.send(200, {"signed_in": True, "expires_at": session.expires_at},
                  {"Set-Cookie": ui.sessions.cookie_header(session)})
        return True


class MeasureRoutes:
    """What has been measured: the fit records, the bench's rates, the answering counts."""

    def route(self) -> bool:
        if self.path == "/ui/fit.json" and self.method == "GET":
            self.send(200, self.ui.fit(room=_whole(self.asked("room")),
                                       users=_whole(self.asked("users", "1")) or 1))
            return True
        if self.path == "/ui/rates.json" and self.method == "GET":
            self.send(200, self.ui.rates())
            return True
        if self.path == "/ui/telemetry.json" and self.method == "GET":
            self.send(200, self.ui.telemetry(self.asked("from")[:400]))
            return True
        return super().route()


class SettingsRoutes:
    """Settings, the libraries this machine has, and taking the install off it."""

    def route(self) -> bool:
        if self.path == "/ui/settings":
            return self._settings()
        if self.path == "/ui/libraries":
            return self._libraries()
        if self.path == "/ui/uninstall":
            return self._uninstall()
        return super().route()

    def _settings(self) -> bool:
        ui = self.ui
        if self.method == "GET":
            from . import autostart as auto
            from .updates import current_version
            self.send(200, {
                "settings": ui.settings.public() if ui.settings else {},
                "name": ui.name,
                "group": cluster_group(ui.cluster_key_path),
                "version": current_version(),
                "autostart": auto.status(),
                "schedule": ui.schedule.public() if ui.schedule else {},
                "machine": ui.report() if callable(ui.report) else {},
            })
            return True
        if self.method == "POST":
            self.send(200, ui.apply_prefs(self.body()))
            return True
        return super().route()

    def _libraries(self) -> bool:
        ui = self.ui
        if ui.environment is None:
            self.send(501, {"error": "no environment on this daemon"})
            return True
        report = ui.report() if callable(ui.report) else {}
        vendor = str(report.get("vendor") or "cpu")
        if self.method == "GET":
            self.send(200, ui.environment.state(vendor))
            return True
        if self.method == "POST":
            return self._change_libraries(vendor)
        return super().route()

    def _change_libraries(self, vendor: str) -> bool:
        req = self.body()
        add = [str(s) for s in req.get("install") or []]
        drop = [str(s) for s in req.get("remove") or []]
        out: dict[str, Any] = {}
        try:
            if drop:
                out.update(self.ui.environment.uninstall(drop))
            if add:
                out.update(self.ui.environment.install(add))
        except OSError as exc:
            self.send(400, {"error": str(exc)})
            return True
        self.send(200, {"changed": out, **self.ui.environment.state(vendor)})
        return True

    def _uninstall(self) -> bool:
        from .uninstall import plan, remove
        ui = self.ui
        if ui.root is None:
            self.send(501, {"error": "this daemon does not know where it keeps things"})
            return True
        if self.method == "GET":
            items = plan(ui.root, key_path=ui.cluster_key_path)
            self.send(200, {"items": [i.public() for i in items],
                            "app": str(app_location() or "")})
            return True
        if self.method == "POST":
            keys = [str(k) for k in (self.body().get("remove") or [])]
            out = remove(ui.root, keys, key_path=ui.cluster_key_path)
            out["app"] = str(app_location() or "")
            self.send(200, out)
            return True
        return super().route()


class ModelRoutes:
    """The models here and elsewhere, what may be downloaded, and what is being served."""

    def route(self) -> bool:
        if self.path == "/ui/models/popular" and self.method == "GET":
            return self._popular()
        if self.path == "/ui/models":
            return self._models()
        if self.path == "/ui/serving/install" and self.method == "POST":
            return self._install_server()
        if self.path == "/ui/serving":
            return self._serving()
        return super().route()

    def _popular(self) -> bool:
        ui = self.ui
        if ui.models is None:
            self.send(501, {"error": "no model store on this daemon"})
            return True
        from .models import PER_PAGE, families, how_many, popular, searched_count, searched_families
        page = max(0, int(self.asked("page", "0")))
        rude = self.asked("rude", "0") in ("1", "true", "yes")
        query = self.asked("q").strip()
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
        self.send(200, {"models": [x.public() for x in found], "families": names,
                        "page": page, "pages": max(1, -(-total // PER_PAGE)),
                        "total": total, "q": query})
        return True

    def _models(self) -> bool:
        ui = self.ui
        if ui.models is None:
            self.send(501, {"error": "no model store on this daemon"})
            return True
        key = load_cluster_key(ui.cluster_key_path)
        auto_models = ui.settings is None or ui.settings.autodownload_models
        if self.method == "GET":
            return self._models_here(key, auto_models)
        if self.method == "DELETE":
            name = str(self.body().get("name") or "")
            self.send(200, {"discarded": ui.models.discard(name)})
            return True
        if self.method == "POST":
            return self._get_model(key, auto_models)
        return super().route()

    def _models_here(self, key: Any, auto_models: bool) -> bool:
        ui = self.ui
        here = {m.name for m in ui.models.all()}
        free = ui.models.free_gb()
        elsewhere: dict[str, list[str]] = {}
        for beacon in (ui.peers() if key is not None else []):
            if beacon.get("is_self"):
                continue
            for row in (beacon.get("device", {}).get("models") or []):
                elsewhere.setdefault(str(row.get("name")), []).append(str(beacon.get("name")))
        self.send(200, {
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

    def _get_model(self, key: Any, auto_models: bool) -> bool:
        from .models import ModelError
        ui, req = self.ui, self.body()
        name = str(req.get("name") or "")
        if not name:
            self.send(400, {"error": "no model was named"})
            return True
        if ui.downloads is None:
            try:
                got = ui.models.ensure(name, source=str(req.get("source") or ""), key=key,
                                       autodownload=auto_models)
            except (ModelError, ValueError) as exc:
                self.send(400, {"error": str(exc)})
                return True
            self.send(200, got.public())
            return True
        here = ui.models.find(name)
        if here is not None:
            self.send(200, here.public())
            return True
        started = ui.downloads.start(name, source=str(req.get("source") or ""),
                                     key=key, autodownload=auto_models,
                                     draft=str(req.get("draft") or ""))
        self.send(202, started.public())
        return True

    def _install_server(self) -> bool:
        ui = self.ui
        if ui.root is None:
            self.send(501, {"error": "this daemon does not know where to keep it"})
            return True
        if not _can_serve():
            self.send(501, {"error": "this install cannot run a model itself; a machine "
                                     "on your network can serve one instead"})
            return True
        from .llama import LlamaError, ensure_server
        try:
            got = ensure_server(ui.root)
        except LlamaError as exc:
            self.send(400, {"error": str(exc)})
            return True
        self.send(200, {"ok": True, "server": str(got)})
        return True

    def _serving(self) -> bool:
        ui = self.ui
        if ui.serving is None:
            self.send(501, {"error": "nothing on this daemon can run a model"})
            return True
        if self.method == "GET":
            self.send(200, {"running": [x.public() for x in ui.serving.live(force=True)],
                            "can_serve": _can_serve()})
            return True
        if self.method == "POST":
            return self._start_serving()
        if self.method == "DELETE":
            ui.stop_serving(int(self.body().get("port") or 0))
            self.send(200, {"running": [x.public() for x in ui.serving.live(force=True)]})
            return True
        return super().route()

    def _start_serving(self) -> bool:
        ui = self.ui
        if not _can_serve():
            self.send(501, {"error": "this install cannot run a model itself; a "
                                     "machine on your network can serve one instead"})
            return True
        req = self.body()
        found = ui.models.find(str(req.get("name") or "")) if ui.models else None
        if found is None:
            self.send(404, {"error": "no such model on this machine"})
            return True
        try:
            served = ui.start_serving(found)
        except (OSError, RuntimeError, ValueError) as exc:
            self.send(400, {"error": str(exc)})
            return True
        self.send(201, served.public())
        return True


class ChatRoutes:
    """The kept conversations, and a question put to whatever is serving a model."""

    def route(self) -> bool:
        if self.path.startswith("/ui/conversations"):
            return self._conversations()
        if self.path == "/ui/chat":
            return self._chat()
        return super().route()

    def _conversations(self) -> bool:
        ui = self.ui
        if ui.conversations is None:
            self.send(501, {"error": "no chat store on this daemon"})
            return True
        rest = self.path[len("/ui/conversations"):].strip("/")
        if rest:
            return self._one_conversation(rest)
        if self.method == "GET":
            self.send(200, {"conversations": [
                c.public(full=False) for c in ui.conversations.search(self.asked("q"))]})
            return True
        if self.method == "POST":
            req = self.body()
            made = ui.conversations.start(model=str(req.get("model") or ""),
                                          title=str(req.get("title") or ""))
            self.send(201, made.public())
            return True
        return super().route()

    def _one_conversation(self, rest: str) -> bool:
        ui = self.ui
        found = ui.conversations.get(rest)
        if found is None:
            self.send(404, {"error": "no such chat"})
            return True
        if self.method == "GET":
            self.send(200, found.public())
            return True
        if self.method == "DELETE":
            ui.conversations.remove(rest)
            self.send(200, {"removed": rest})
            return True
        if self.method == "POST":
            renamed = ui.conversations.rename(rest, str(self.body().get("title") or ""))
            self.send(200, renamed.public(full=False))
            return True
        return super().route()

    def _chat(self) -> bool:
        from .chat import targets
        ui = self.ui
        key = load_cluster_key(ui.cluster_key_path)
        available = targets(ui.peers() if key is not None else [], ui.serving,
                            derive_token(key) if key else "")
        if self.method == "GET":
            self.send(200, {"models": [t.public() for t in available]})
            return True
        if self.method == "POST":
            return self._say(available)
        return super().route()

    def _say(self, available: list) -> bool:
        from .chat import ChatError, find, reply_text, stream
        req = self.body()
        target = find(available, str(req.get("model") or ""))
        if target is None:
            self.send(503, {"error": "no machine on this network is serving a model"})
            return True
        messages = [m for m in (req.get("messages") or [])
                    if isinstance(m, dict) and m.get("content")]
        if not messages:
            self.send(400, {"error": "nothing to send"})
            return True
        payload = {"model": target.model, "messages": messages, "stream": True}
        if req.get("temperature") is not None:
            payload["temperature"] = float(req["temperature"])
        try:
            pieces = stream(target, payload)
            first = next(pieces, b"")
        except ChatError as exc:
            self.send(502, {"error": str(exc)})
            return True
        cid = str(req.get("conversation") or "")
        if self.ui.conversations is not None and cid:
            self.ui.conversations.append(cid, "user", str(messages[-1]["content"]))
        said = self._relay(target, first, pieces)
        if self.ui.conversations is not None and cid:
            spoken = reply_text(said)
            if spoken:
                self.ui.conversations.append(cid, "assistant", spoken)
        return True

    def _relay(self, target: Any, first: bytes, pieces: Any) -> bytes:
        """Stream the answer to the caller as it arrives, and return all of it."""
        handler = self.handler
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
        return bytes(said)


class UpdateRoutes:
    """Which version is out, and installing it."""

    def route(self) -> bool:
        if self.path == "/ui/updates" and self.method == "GET":
            return self._latest()
        if self.path == "/ui/updates/install" and self.method == "POST":
            self.send(200, self.ui.install_update())
            return True
        return super().route()

    def _latest(self) -> bool:
        from .updates import UpdateError, asset_for, check, current_version
        now = current_version()
        try:
            release = check()
        except UpdateError as exc:
            self.send(200, {"version": now, "checked": False, "error": str(exc)})
            return True
        asset = asset_for(release)
        self.send(200, {
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


class ClusterRoutes:
    """The clusters this machine is in, the machines it sees, and joining another."""

    def route(self) -> bool:
        if self.path == "/ui/clusters":
            return self._clusters()
        if self.path == "/ui/peers" and self.method == "GET":
            self.send(200, {"peers": self.ui.peers(), "self": self.ui.name,
                            "group": cluster_group(self.ui.cluster_key_path)})
            return True
        if self.path == "/ui/fleet" and self.method == "GET":
            self.send(200, self.ui.fleet())
            return True
        if self.path == "/ui/fleet/join" and self.method == "POST":
            return self._join_fleet()
        if self.path == "/ui/fleet/pause" and self.method == "POST":
            return self._pause_fleet()
        return super().route()

    def _pause_fleet(self) -> bool:
        req = self.body()
        span = str(req.get("for") or "")
        minutes = minutes_of(span)
        if span and minutes is None:
            self.send(400, {"error": f"{span!r} is not a length of time; "
                                     "try '2h', '90m' or '45s'"})
            return True
        self.send(200, self.ui.pause_fleet(resume=bool(req.get("resume")),
                                           minutes=minutes,
                                           reason=str(req.get("reason") or "")))
        return True

    def _clusters(self) -> bool:
        from .discovery import join, leave, memberships
        ui = self.ui
        if self.method == "GET":
            self.send(200, {"clusters": [m.public() for m in
                                         memberships(ui.cluster_key_path)]})
            return True
        if self.method == "POST":
            req = self.body()
            words = str(req.get("passphrase") or "")
            group = str(req.get("group") or "").strip() or "ml-stack"
            try:
                rows = join(words, group=group, path=ui.cluster_key_path)
            except DiscoveryError as exc:
                self.send(400, {"error": str(exc)})
                return True
            ui.rejoined()
            self.send(200, {"clusters": [m.public() for m in rows], "joined": group})
            return True
        if self.method == "DELETE":
            group = str(self.body().get("group") or "")
            rows = leave(group, ui.cluster_key_path)
            ui.rejoined()
            self.send(200, {"clusters": [m.public() for m in rows], "left": group})
            return True
        return super().route()

    def _join_fleet(self) -> bool:
        from .join import JoinError
        req = self.body()
        words = str(req.get("passphrase") or "")
        if not words and not req.get("persist"):
            self.send(400, {"error": "nothing to do: type the passphrase every machine "
                                     "shares, or tick 'start at logon'"})
            return True
        try:
            self.send(200, self.ui.join_fleet(
                passphrase=words, group=str(req.get("group") or ""),
                persist=bool(req.get("persist")), name=str(req.get("name") or "")))
        except (JoinError, DiscoveryError) as exc:
            self.send(400, {"error": str(exc)})
        return True


class JobRoutes:
    """A measurement sweep: starting it, watching it, stopping it, and what it found."""

    def route(self) -> bool:
        if self.path == "/ui/bench/sweep" and self.method == "POST":
            return self._sweep()
        if self.path == "/ui/bench/status" and self.method == "GET":
            self.send(200, self.ui.bench_state())
            return True
        if self.path == "/ui/bench/history" and self.method == "GET":
            try:
                self.send(200, {"history": self.ui.bench_history()})
            except ImportError as exc:
                self.send(501, {"error": f"the bench is not installed here: {exc}"})
            return True
        if self.path == "/ui/bench/stop" and self.method == "POST":
            return self._stop()
        return super().route()

    def _sweep(self) -> bool:
        import shlex

        from .join import sweep_argv
        req = self.body()
        models = [str(m) for m in (req.get("models") or []) if str(m).strip()]
        if not models:
            self.send(400, {"error": "pick at least one model to measure"})
            return True
        try:
            argv = sweep_argv(models, peers=[str(p) for p in (req.get("peers") or [])],
                              sample=int(req.get("sample") or 0),
                              label=str(req.get("label") or ""),
                              extra=[str(a) for a in (req.get("extra") or [])])
        except ValueError as exc:
            self.send(400, {"error": str(exc)})
            return True
        if req.get("dry_run"):
            self.send(200, {"argv": argv, "command": "ml-stack-bench " + shlex.join(argv)})
            return True
        try:
            self.send(202, self.ui.start_sweep(argv))
        except ImportError as exc:
            self.send(501, {"error": f"the bench is not installed here: {exc}"})
        return True

    def _stop(self) -> bool:
        pid = self.body().get("pid")
        if not pid:
            self.send(400, {"error": "say which pid to stop -- the one the page was shown"})
            return True
        self.send(200, {"stopped": self.ui.stop_sweep(int(pid))})
        return True


class Router(PageRoutes, SetupRoutes, SessionRoutes, MeasureRoutes, SettingsRoutes,
             ModelRoutes, ChatRoutes, UpdateRoutes, ClusterRoutes, JobRoutes, Base):
    """Every screen's routes, in the order a request meets them."""

    def run(self) -> bool:
        """Answer one ``/ui`` request. Returns True when it did."""
        if self.open_route():
            return True
        # Everything below is API, and none of it is reachable from a cross-origin page.
        if not self.header(UI_HEADER):
            self.send(403, {"error": f"{UI_HEADER} header required"})
            return True
        if self.public_route():
            return True
        if not self.signed_in():
            return True
        if self.route():
            return True
        self.send(404, {"error": "no such route"})
        return True

    def signed_in(self) -> bool:
        """Whether the request may go on; sends the refusal when it may not.

        A machine in no cluster has no password anyone could be asked for, so it answers
        to itself and to nobody else.
        """
        ui = self.ui
        if ui.authed(self.cookie):
            return True
        if in_cluster(ui.cluster_key_path):
            self.send(401, {"error": "sign in first"})
            return False
        why = ui.may_setup(self.client_ip, self.host_header,
                           self.header("X-ML-Stack-Setup", ""))
        if why:
            self.send(403, {"error": why})
            return False
        return True


def routes(ui: Any, handler: Any) -> bool:
    """Handle one ``/ui/*`` request. Returns True when it did."""
    if not urllib.parse.urlparse(handler.path).path.startswith("/ui"):
        return False
    return Router(ui, handler).run()
