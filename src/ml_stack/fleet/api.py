"""Every route the daemon answers on the LAN. `make_handler` builds the request handler
the HTTP server runs: jobs, files, models, availability, bench, serving, speech, the
peer-to-peer fetches, and a proxy in front of a model server on this machine's loopback.
The ``/ui/*`` paths are handed to `fleet.routes` through the `fleet.ui.UI` it was given.
"""

from __future__ import annotations

import contextlib
import json
import re
import secrets
import shlex
import ssl
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any

from ml_stack import gate, sealing, sentinel, serverkeys
from ml_stack.files import promote
from ml_stack.macauth import Authenticator, Verdict, parts
from ml_stack.sentinel.adapters import watch_authenticator
from ml_stack.speech import service as speech
from ml_stack.speech.protocols import ProviderError
from ml_stack.speech.service import as_json, transcribe

from . import commands, device_auth, invite_routes, projects as project_routes
from .availability import Availability, parse_window
from .deciding import MAX_REQUEST, Deciding
from .device import device_report
from .discovery import load_cluster_key
from .files import (
    DIGEST_HEADER,
    FILE_CHUNK,
    Fetcher,
    file_digest,
    remember_digest,
    safe_relpath,
)
from .framing import (
    MOST_BODY,
    MOST_UPLOAD,
    Limited,
    Malformed,
    addressed_to_this_machine,
    content_range,
    read_body,
    requested_range,
)
from .jobs import DaemonError, JobRunner
from .measuring import BenchHost, Job as BenchJob, Refused
from .models import Models
from .onboard.joining import API as JOIN_API, Joining
from .serving import Hosting, NoRoom, Serving
from .ui import routes as ui_routes
from .weights import ModelError

INFER_CHUNK = 1 << 13
INFER_TIMEOUT = 600.0
READ_ONLY = ("/slots", "/props", "/metrics", "/models", "/health")
"""Model-server paths the proxy only reads: a write to ``/slots`` saves or erases a cache."""
PROXIED = re.compile(r"/(v1/[A-Za-z0-9/_.-]*|completions?|chat/completions|tokenize|detokenize|"
                     r"embeddings?|infill|apply-template|props|health|models|metrics|slots)"
                     r"(\?[^\s#]*)?")
"""The paths of a model server the proxy passes on; nothing else, and no way up out of them."""


@dataclass
class Daemon:
    """What the daemon's routes answer from. ``bench`` is a `fleet.measuring.BenchHost`:
    with one, the daemon takes bench jobs (``POST /bench``), says what it can measure
    (``GET /bench``) and hands back what it measured (``GET /bench/export``). ``hosting`` is
    a `fleet.serving.Hosting`: with one and ``models``, ``POST /serve`` runs a model this
    machine holds."""

    runner: JobRunner
    files_root: Path
    token: str | Callable[[], str]
    name: str | Callable[[], str] = ""
    report: Callable[[], dict[str, Any]] = device_report
    fetcher: Fetcher | None = None
    ui: Any | None = None
    projects: project_routes.ProjectRegistry | None = None
    workspaces: Any | None = None
    schedule: Availability | None = None
    on_paused: str = "stop"
    schedule_path: Path | None = None
    serving: Serving | None = None
    models: Models | None = None
    cluster_key_path: Path | str | None = None
    tokens: Callable[[], set[str]] | None = None
    devices: Callable[[], list] = lambda: []
    bench: BenchHost | None = None
    hosting: Hosting | None = None
    decide: Deciding | None = None
    ui_from_lan: bool = False
    """Whether the web interface answers other machines. Off: it answers this one alone."""
    joining: Joining | None = None
    """Answers a machine that asks to join with the passphrase; without it the join routes are off."""
    command: Callable[[list[str]], list[str]] = commands.allowed
    """Which argv a ``POST /jobs`` may run, and in what form; raises ValueError to refuse."""


def _count(text: str, fallback: int, most: int = 1_000_000) -> int:
    """A query parameter as a whole number in ``0..most``; ``fallback`` for anything else."""
    return min(int(text), most) if text.isdigit() and len(text) < 12 else fallback


def make_handler(daemon: Daemon) -> type[BaseHTTPRequestHandler]:
    """The request handler class for ``daemon``."""
    runner, files_root, token, name = daemon.runner, daemon.files_root, daemon.token, daemon.name
    report, fetcher, ui, schedule = daemon.report, daemon.fetcher, daemon.ui, daemon.schedule
    on_paused, schedule_path, serving = daemon.on_paused, daemon.schedule_path, daemon.serving
    models, cluster_key_path, tokens = daemon.models, daemon.cluster_key_path, daemon.tokens
    bench, hosting, decide = daemon.bench, daemon.hosting, daemon.decide
    ui_from_lan = daemon.ui_from_lan
    joining, command = daemon.joining, daemon.command

    def secrets_now() -> set[str]:
        """Every secret this machine answers to, read at request time."""
        own = token() if callable(token) else token
        return {one for one in (own, *(tokens() if tokens else ())) if one}

    authenticator = watch_authenticator(Authenticator(secrets_now), sentinel.armed(), Verdict)
    device_authenticator = Authenticator(lambda: [*secrets_now(),
                                                  *(device_auth.secret(device) for device in daemon.devices())])

    class Handler(Limited, BaseHTTPRequestHandler):
        server_version = "ml-stack-traind/0.1"

        def log_message(self, fmt: str, *args: Any) -> None:  # quieter
            pass

        # -- helpers --
        def _token(self) -> str:
            """Read at request time, not captured at startup."""
            return token() if callable(token) else token

        def _name(self) -> str:
            """This machine's name, read at request time."""
            return name() if callable(name) else name

        _opening: tuple[bytes, Verdict, bool, Any] | None = None

        def _sealing(self) -> tuple[bytes, Verdict, bool] | None:
            """What seals this request's answer and opens its body, when it was signed and asked for."""
            held = self._opening
            return held[:3] if held is not None and held[3] is self.headers else None

        def _send(self, code: int, payload: Any, *, raw: bytes | None = None,
                  headers: dict[str, str] | None = None,
                  content_type: str = "") -> None:
            body = raw if raw is not None else json.dumps(payload).encode()
            headers = dict(headers or {})
            opening = self._sealing()
            if opening is not None and opening[2] and self.command != "HEAD":
                key, verdict, _ = opening
                body = sealing.seal(key, body, sealing.response_data(verdict.nonce, code))
                headers[sealing.HEADER] = "1"
            self.send_response(code)
            self.send_header("Content-Type", content_type or (
                "application/octet-stream" if raw is not None else "application/json"))
            self.send_header("Content-Length", str(len(body)))
            for k, v in headers.items():
                self.send_header(k, v)
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _send_file(self, target: Path, start: int = 0,
                       end: int | None = None) -> None:
            """Send a file without ever holding more than FILE_CHUNK of it."""
            size = target.stat().st_size
            if start and start >= size:
                self._send(416, {"error": "range beyond end of file",
                                 "size": size},
                           headers={"Content-Range": f"bytes */{size}"})
                return
            last = size - 1 if end is None else min(end, size - 1)
            length = max(0, last - start + 1)
            ranged = start > 0 or (end is not None and last < size - 1)
            self.send_response(206 if ranged else 200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(length))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header(DIGEST_HEADER, file_digest(target))
            if ranged:
                self.send_header("Content-Range", f"bytes {start}-{last}/{size}")
            self.end_headers()
            if self.command == "HEAD":
                return
            remaining = length
            with target.open("rb") as fh:
                fh.seek(start)
                while remaining > 0:
                    chunk = fh.read(min(FILE_CHUNK, remaining))
                    if not chunk:
                        break
                    try:
                        self.wfile.write(chunk)
                    except (BrokenPipeError, ConnectionResetError):
                        return
                    remaining -= len(chunk)

        def _guard(self, body: bytes | None = None) -> bool:
            """Whether this request is signed by a secret this machine answers to; the
            answer is sent when it is not. ``body`` is what the request carried."""
            self._workspace_device = None
            self._workspace_projects = daemon.projects
            checking = device_authenticator if self.path.split('?')[0].startswith('/workspace/v1/') else authenticator
            verdict = checking.check(self.command, self.path, self.headers, body, self.client_address[0])
            if verdict.ok:
                self._workspace_device = device_auth.identify(daemon.devices(), verdict.secret)
                mode = self.headers.get(sealing.HEADER, "")
                self._opening = (sealing.box_key(verdict.secret), verdict, mode == "2",
                                 self.headers) if mode in ("1", "2") else None
                return True
            if verdict.locked:
                self._send(429, {"error": verdict.reason}, headers={"Retry-After": "60"})
            else:
                self._send(401, {"error": verdict.reason})
            return False

        def _unsealed(self, body: bytes | None) -> bytes | None:
            """What a signed request's body says, or None once a refusal has been sent: a body
            is sealed under the key derived from the secret that signed it."""
            if not body:
                return body
            opening = self._sealing()
            if opening is None:
                self._send(400, {"error": "a request body is sent sealed"})
                return None
            key, verdict, _ = opening
            host, target = parts(f"//{self.headers.get('Host', '')}{self.path}")
            try:
                return sealing.open_(key, body, sealing.request_data(
                    self.command, target, host, verdict.at, verdict.nonce))
            except sealing.SealError:
                self._send(400, {"error": "the body did not authenticate"})
                return None

        def _body(self, most: int = MOST_BODY) -> bytes | None:
            """The request body, or None once a refusal for it has been sent."""
            try:
                return read_body(self, most)
            except Malformed as bad:
                self.close_connection = True
                self._send(bad.status, {"error": bad.message})
                return None

        # -- routes --

        # -- inference proxy --
        def _proxy(self, body: bytes | None) -> bool:
            """Forward /infer/* to a model server on this machine's loopback.

            The model server stays on 127.0.0.1. This route is the only LAN-exposed
            one, and it already requires the bearer token.
            """
            if serving is None or not (self.path == "/infer" or self.path.startswith(
                    ("/infer/", "/infer?"))):
                return False
            if not self._guard(body):
                return True
            body = self._unsealed(body)
            if body is None:
                return True
            rest = self.path[len("/infer"):] or "/"
            rest = "/" + rest if rest.startswith("?") else rest
            route = rest.split("?")[0]
            if (not PROXIED.fullmatch(rest) or ".." in route.split("/")
                    or (route.startswith(READ_ONLY) and self.command not in ("GET", "HEAD"))):
                self._send(403, {"error": "that path is not one the proxy passes on"})
                return True
            parsed = urllib.parse.urlparse(rest)
            model = urllib.parse.parse_qs(parsed.query).get("ml_stack_model", [""])[0]
            port = serving.port_for(model)
            if port is None:
                self._send(503, {"error": "no model server is running on this machine"})
                return True

            upstream = urllib.request.Request(
                f"http://127.0.0.1:{port}{rest}", data=body or None, method=self.command)
            if leased := serverkeys.for_url(upstream.full_url):
                upstream.add_header("Authorization", f"Bearer {leased}")
            for name, value in self.headers.items():
                if name.lower() in ("authorization", "host", "content-length",
                                    "connection", "x-ml-stack-ui"):
                    continue
                upstream.add_header(name, value)
            if body:
                upstream.add_header("Content-Length", str(len(body)))

            where = f"http://127.0.0.1:{port}{rest}"
            try:
                with gate.turn(where) if gate.is_generation(where) else nullcontext():
                    return self._forward(upstream)
            except gate.QueueTimeout as exc:
                self._send(429, {"error": str(exc)})
                return True

        def _forward(self, upstream: urllib.request.Request) -> bool:
            """Relay ``upstream`` to the caller as it is generated."""
            try:
                response = urllib.request.urlopen(upstream, timeout=INFER_TIMEOUT)
            except urllib.error.HTTPError as exc:
                raw = exc.read()
                self._send(exc.code, None, raw=raw,
                           content_type=exc.headers.get("Content-Type",
                                                        "application/json"))
                return True
            except (urllib.error.URLError, OSError) as exc:
                self._send(502, {"error": f"the model server did not answer: {exc}"})
                return True

            kind = response.headers.get("Content-Type", "application/json")
            opening = self._sealing()
            if opening is not None and opening[2] and "event-stream" not in kind:
                with response:
                    self._send(response.status, None, raw=response.read(), content_type=kind)
                return True
            self.send_response(response.status)
            self.send_header("Content-Type", kind)
            # No Content-Length and close framing: a streamed completion must reach the
            # caller as it is generated, not after the last token.
            self.send_header("Connection", "close")
            self.end_headers()
            # read1, not read: read(n) blocks until it has n bytes, and a token is
            # tens of bytes, so the whole completion would arrive at once.
            read = getattr(response, "read1", response.read)
            try:
                while True:
                    block = read(INFER_CHUNK)
                    if not block:
                        break
                    self.wfile.write(block)
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                # The caller hung up. Closing the upstream is what tells llama.cpp to
                # stop generating; without it the card keeps working for nobody.
                pass
            finally:
                response.close()
            return True

        def _ui(self) -> bool:
            if ui is None or not self.path.startswith("/ui"):
                return False
            if not ui_from_lan and not addressed_to_this_machine("localhost",
                                                                 self.client_address[0]):
                self._send(403, {"error": "the web interface answers this machine only; "
                                          "start the daemon with --ui-from-lan to open it"})
                return True
            if not isinstance(self.connection, ssl.SSLSocket) and not addressed_to_this_machine(
                    "localhost", self.client_address[0]):
                self._send(403, {"error": "the web interface is served to other machines over "
                                          "TLS only; it signs in with the passphrase"})
                return True
            return ui_routes(ui, self)

        def _join(self, body: bytes | None) -> bool:
            """Answer a ``/join/v1`` request, which carries no signature; False for any other path."""
            if invite_routes.public(ui, self, body):
                return True
            if joining is None or not self.path.startswith(JOIN_API + "/"):
                return False
            try:
                asked = json.loads(body or b"{}")
            except ValueError:
                asked = None
            if not isinstance(asked, dict):
                self._send(400, {"error": "the body is not a JSON object"})
                return True
            status, answer = joining.handle(self.path.split("?")[0], asked, self.client_address[0])
            self._send(status, answer)
            return True

        def _extension(self, body=None) -> bool:
            for entry in entry_points(group='ml_stack.peer_routes'):
                if self.path.split('?')[0].startswith('/' + entry.name + '/'):
                    return bool(entry.load()(self, body))
            return False

        def do_GET(self) -> None:
            if self.path == "/favicon.ico":
                self.send_response(204)
                self.end_headers()
                return
            if self._proxy(None):
                return
            if self._ui():
                return
            parsed = urllib.parse.urlparse(self.path)
            path, q = parsed.path, urllib.parse.parse_qs(parsed.query)
            here = addressed_to_this_machine(self.headers.get("Host", ""),
                                             self.client_address[0])
            if path == "/health" and "Authorization" not in self.headers and not here:
                self._send(200, {"ok": True})
                return
            if not (path == "/health" and here and "Authorization" not in self.headers) \
                    and not self._guard():
                return
            if self._extension() or project_routes.answer(self, daemon.projects, parsed):
                return
            if path == "/health":
                status = runner.status()
                sched = schedule.public() if schedule is not None else None
                if sched is not None and not sched["available"]:
                    status = {**status, "free": 0}
                self._send(200, {"ok": True, "name": self._name(), **status, **report(),
                                 **({"availability": sched} if sched else {}),
                                 **({"serving": serving.public()} if serving is not None
                                    else {})})
                return
            if path == "/models":
                if models is None:
                    self._send(501, {"error": "no model store on this daemon"})
                    return
                self._send(200, {"models": models.inventory(),
                                 "free_gb": models.free_gb(),
                                 "store": str(models.store)})
                return
            if path.startswith("/models/"):
                if models is None:
                    self._send(501, {"error": "no model store on this daemon"})
                    return
                wanted = urllib.parse.unquote(path[len("/models/"):])
                found = models.find(wanted) or models.find_draft(wanted)
                if found is None:
                    self._send(404, {"error": f"no model called {wanted!r}"})
                    return
                try:
                    start, end = requested_range(self.headers.get("Range", ""))
                except Malformed as bad:
                    self._send(bad.status, {"error": bad.message})
                    return
                self._send_file(found.path, start, end)
                return
            if path == "/availability":
                if schedule is None:
                    self._send(501, {"error": "no schedule on this daemon"})
                    return
                self._send(200, schedule.public())
                return
            if path == "/jobs":
                self._send(200, {"jobs": runner.snapshot()})
                return
            if path == "/speech/providers":
                self._send(200, speech.providers())
                return
            if path == "/bench":
                if bench is None:
                    self._send(501, {"error": "this daemon takes no bench jobs"})
                    return
                self._send(200, {"ok": True, "name": self._name(), **bench.report(),
                                 "jobs": bench.snapshot()})
                return
            if path == "/bench/export":
                if bench is None:
                    self._send(501, {"error": "this daemon takes no bench jobs"})
                    return
                try:
                    out = bench.export(since=q.get("since", [""])[0],
                                       job=q.get("job", [""])[0],
                                       full=q.get("full", ["0"])[0] not in ("", "0"),
                                       anyway=q.get("anyway", ["0"])[0] not in ("", "0"))
                except DaemonError as e:
                    self._send(400, {"error": str(e)})
                    return
                self._send(200, {**out, "host": self._name()})
                return
            m = re.match(r"^/jobs/([^/]+)(/log|/metrics)?$", path)
            if m:
                job = runner.jobs.get(m.group(1))
                if job is None:
                    self._send(404, {"error": "unknown job"})
                    return
                kind = m.group(2)
                if kind is None:
                    self._send(200, job.public())
                    return
                if kind == "/log":
                    n = _count(q.get("tail", ["200"])[0], 200)
                    p = runner.log_path(job.id)
                    text = ""
                    if p.exists():
                        text = "".join(p.read_text(errors="replace").splitlines(True)[-n:])
                    self._send(200, {"log": text})
                    return
                since = _count(q.get("since", ["0"])[0], 0)
                mp = runner.job_dir(job.id) / "metrics.jsonl"
                rows: list[Any] = []
                if mp.exists():
                    for line in mp.read_text(errors="replace").splitlines()[since:]:
                        with contextlib.suppress(json.JSONDecodeError):
                            rows.append(json.loads(line))
                self._send(200, {"metrics": rows, "next": since + len(rows)})
                return
            m = re.match(r"^/fetch/([^/]+)$", path)
            if m:
                if fetcher is None:
                    self._send(501, {"error": "peer-to-peer fetch is not enabled"})
                    return
                fetch = fetcher.fetches.get(m.group(1))
                if fetch is None:
                    self._send(404, {"error": "unknown fetch"})
                    return
                self._send(200, fetch.public())
                return
            if path == "/fetch":
                if fetcher is None:
                    self._send(501, {"error": "peer-to-peer fetch is not enabled"})
                    return
                self._send(200, {"fetches": [f.public() for f in
                                             list(fetcher.fetches.values())],
                                 **fetcher.status()})
                return
            if path.startswith("/files/"):
                try:
                    target = safe_relpath(files_root, path[len("/files/"):])
                except DaemonError as e:
                    self._send(400, {"error": str(e)})
                    return
                if not target.is_file():
                    self._send(404, {"error": "not found"})
                    return
                try:
                    start, end = requested_range(self.headers.get("Range", ""))
                except Malformed as bad:
                    self._send(bad.status, {"error": bad.message})
                    return
                self._send_file(target, start, end)
                return
            self._send(404, {"error": "no such route"})

        do_HEAD = do_GET

        def _decide(self, body: bytes) -> None:
            """Answer ``POST /decide``."""
            if decide is None:
                self._send(501, {"error": "this daemon makes no decisions"})
                return
            if len(body) > MAX_REQUEST:
                self.close_connection = True
                self._send(413, {"error": f"at most {MAX_REQUEST} bytes"})
                return
            try:
                asked = json.loads(body or b"{}")
            except ValueError:
                self._send(400, {"error": "the body is not JSON"})
                return
            status, payload = decide.answer(asked)
            self._send(status, payload)

        def do_POST(self) -> None:
            if self._ui():
                return
            body = self._body()
            if body is None or self._join(body) or self._proxy(body):
                return
            if not self._guard(body):
                return
            body = self._unsealed(body)
            if body is None:
                return
            if self._extension(body):
                return
            try:
                if not self._workspace(body or b"{}"):
                    self._route_post(body or b"{}")
            except Malformed as bad:
                self._send(bad.status, {"error": bad.message})

        def _object(self, body: bytes) -> dict[str, Any]:
            """The JSON object a request body holds; `Malformed` for anything else."""
            try:
                got = json.loads(body or b"{}")
            except ValueError:
                raise Malformed(400, "the body is not JSON") from None
            if not isinstance(got, dict):
                raise Malformed(400, "the body is not a JSON object")
            return got

        def _workspace(self, body: bytes) -> bool:
            match = re.fullmatch(r"/workspace/v1/projects/([a-f0-9]{32})/(join|board|ensure)",
                                 urllib.parse.urlparse(self.path).path)
            if not match:
                return False
            host = daemon.workspaces
            opening = self._sealing()
            if host is None:
                self._send(501, {"error": "project workspace hosting is unavailable"})
            elif opening is None or not opening[2]:
                self._send(403, {"error": "project agent capabilities require sealed fleet requests"})
            else:
                code, reply = host.answer(match[1], match[2], self._object(body),
                                          device=self._workspace_device)
                self._send(code, reply)
            return True

        def _route_post(self, body: bytes) -> None:
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path == "/decide":
                self._decide(body)
                return
            if parsed.path == "/speech/transcribe":
                want = urllib.parse.parse_qs(parsed.query)
                try:
                    heard = transcribe(body,
                                       provider=want.get("provider", [""])[0] or None,
                                       language=want.get("language", [""])[0] or None)
                except ProviderError as e:
                    self._send(503, {"error": str(e)})
                    return
                self._send(200, as_json(heard))
                return
            if parsed.path == "/jobs":
                try:
                    req = self._object(body)
                    if req.get("cwd") or req.get("env"):
                        raise ValueError("a job runs in the daemon's own folder and "
                                         "environment; cwd and env are not accepted")
                    argv = req.get("argv")
                    if isinstance(argv, str):
                        argv = shlex.split(argv)
                    job = runner.submit(req.get("name", ""), command(argv or []),
                                        str(files_root))
                except (DaemonError, ValueError) as e:
                    self._send(400, {"error": str(e)})
                    return
                self._send(201, job.public())
                return
            if parsed.path == "/bench":
                if bench is None:
                    self._send(501, {"error": "this daemon takes no bench jobs"})
                    return
                try:
                    job = bench.submit(BenchJob.from_request(self._object(body)))
                except Refused as e:
                    # 409: the job is well-formed and this peer will not run it now --
                    # its code, its lock or its memory says so, and `refused` says which
                    self._send(409, {"error": str(e), "refused": e.kind})
                    return
                except (DaemonError, ValueError) as e:
                    self._send(400, {"error": str(e)})
                    return
                self._send(201, job.public())
                return
            if parsed.path == "/models/get":
                if models is None:
                    self._send(501, {"error": "no model store on this daemon"})
                    return
                req = self._object(body)
                try:
                    got = models.ensure(str(req.get("name") or ""),
                                        source=str(req.get("source") or ""),
                                        key=load_cluster_key(cluster_key_path),
                                        autodownload=bool(req.get("autodownload", True)))
                except (ModelError, ValueError) as e:
                    self._send(400, {"error": str(e)})
                    return
                self._send(200, got.public())
                return
            if parsed.path == "/availability":
                if schedule is None:
                    self._send(501, {"error": "no schedule on this daemon"})
                    return
                req = self._object(body)
                action = str(req.get("action") or "")
                try:
                    if action == "pause":
                        minutes = req.get("minutes")
                        schedule.pause(minutes=float(minutes) if minutes else None,
                                       reason=str(req.get("reason") or ""))
                        if on_paused == "stop":
                            runner.stop_running()
                    elif action == "resume":
                        schedule.resume()
                    elif action == "reserve":
                        schedule.reserve(str(req.get("holder") or "someone"),
                                         float(req.get("seconds") or 3600),
                                         str(req.get("reason") or ""))
                    elif action == "release":
                        schedule.release(str(req.get("holder") or ""))
                    elif action == "window":
                        schedule.windows.append(
                            parse_window(str(req.get("spec") or ""),
                                         busy=bool(req.get("busy", True))))
                    elif action == "clear_windows":
                        schedule.windows.clear()
                    else:
                        raise DaemonError(f"unknown action {action!r}")
                except (DaemonError, ValueError, PermissionError) as e:
                    self._send(400, {"error": str(e)})
                    return
                if schedule_path is not None:
                    schedule.save(schedule_path)
                self._send(200, schedule.public())
                return
            if parsed.path == "/fetch":
                if fetcher is None:
                    self._send(501, {"error": "peer-to-peer fetch is not enabled"})
                    return
                try:
                    req = self._object(body)
                    if "from_url" in req or "url" in req or "token" in req:
                        self._send(400, {"error": "name the source peer, not a URL: a "
                                                  "route that fetches any URL it is "
                                                  "given is a forgery primitive"})
                        return
                    source = str(req.get("peer") or "")
                    relpath = str(req.get("relpath") or "")
                    if not source or not relpath:
                        raise DaemonError("both 'peer' and 'relpath' are required")
                    fetch = fetcher.start(source=source, relpath=relpath,
                                          to=str(req.get("to") or relpath),
                                          sha256=str(req.get("sha256") or ""))
                except (DaemonError, ValueError) as e:
                    self._send(400, {"error": str(e)})
                    return
                self._send(202, fetch.public())
                return
            if parsed.path == "/serve":
                if hosting is None or models is None:
                    self._send(501, {"error": "this daemon serves no models"})
                    return
                try:
                    req = self._object(body)
                    wanted = str(req.get("model") or "")
                    context = int(req.get("context") or 8192)
                    parallel = max(1, int(req.get("parallel") or 1))
                except (TypeError, ValueError) as e:
                    self._send(400, {"error": str(e)})
                    return
                if not wanted:
                    self._send(400, {"error": "name the model"})
                    return
                found = models.find(wanted)
                if found is None:
                    self._send(404, {"error": f"no model called {wanted!r} on this "
                                              f"machine"})
                    return
                running = hosting.already(found.name)
                if running is not None:
                    self._send(200, running.public())
                    return
                try:
                    served = hosting.start(found.path, name=found.name, context=context,
                                           parallel=parallel,
                                           room=int(report().get("room_bytes") or 0))
                except NoRoom as e:
                    self._send(409, {"error": str(e), "refused": "room"})
                    return
                except (OSError, RuntimeError, ValueError) as e:
                    self._send(502, {"error": str(e)})
                    return
                self._send(201, served.public())
                return
            m = re.match(r"^/jobs/([^/]+)/stop$", parsed.path)
            if m:
                try:
                    job = runner.stop(m.group(1))
                except DaemonError as e:
                    self._send(404, {"error": str(e)})
                    return
                self._send(200, job.public())
                return
            self._send(404, {"error": "no such route"})

        def do_DELETE(self) -> None:
            if self._ui():
                return
            self._send(404, {"error": "no such route"})

        def do_PUT(self) -> None:
            if self._ui():
                return
            body = self._body(MOST_UPLOAD)
            if body is None or not self._guard(body):
                return
            body = self._unsealed(body)
            if body is None:
                return
            parsed = urllib.parse.urlparse(self.path)
            if not parsed.path.startswith("/files/"):
                self._send(404, {"error": "no such route"})
                return
            try:
                target = safe_relpath(files_root, parsed.path[len("/files/"):])
                offset, _ = content_range(self.headers.get("Content-Range", ""), len(body))
            except DaemonError as e:
                self._send(400, {"error": str(e)})
                return
            except Malformed as bad:
                self._send(bad.status, {"error": bad.message})
                return
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_suffix(target.suffix + ".part")
            held = partial.stat().st_size if partial.exists() else 0
            if offset > held:
                self._send(416, {"error": "cannot resume past what this daemon holds",
                                 "held": held, "requested_offset": offset},
                           headers={"Content-Range": f"bytes */{held}"})
                return
            mode = "r+b" if offset else "wb"
            with partial.open(mode) as fh:
                if offset:
                    fh.seek(offset)
                fh.write(body)
            if self.headers.get("X-ML-Stack-Complete", "1") == "1":
                want = self.headers.get(DIGEST_HEADER, "").strip().lower()
                got = file_digest(partial) if want else ""
                if want and not secrets.compare_digest(got, want):
                    partial.unlink(missing_ok=True)
                    self._send(422, {"error": "checksum mismatch, upload discarded",
                                     "expected": want, "got": got})
                    return
                promote(partial, target)
                size = target.stat().st_size
                if got:
                    remember_digest(target, got)
                self._send(200, {"ok": True, "path": str(target), "bytes": size,
                                 "sha256": got or file_digest(target)})
            else:
                self._send(200, {"ok": True, "partial": str(partial),
                                 "bytes": partial.stat().st_size})

    return Handler
