"""Every route the daemon answers on the LAN. `make_handler` builds the request handler
the HTTP server runs: jobs, files, models, availability, bench, serving, speech, the
peer-to-peer fetches, and a proxy in front of a model server on this machine's loopback.
The ``/ui/*`` paths are handed to `fleet.routes` through the `fleet.ui.UI` it was given.
"""

from __future__ import annotations

import json
import re
import secrets
import shlex
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any

from ml_stack.files import promote
from ml_stack.speech import service as speech
from ml_stack.speech.protocols import ProviderError
from ml_stack.speech.service import as_json, transcribe

from .availability import Availability, parse_window
from .device import device_report
from .discovery import load_cluster_key
from .files import (
    DIGEST_HEADER,
    FILE_CHUNK,
    Fetcher,
    byte_range,
    file_digest,
    remember_digest,
    safe_relpath,
)
from .jobs import DaemonError, JobRunner
from .measuring import BenchHost, Job as BenchJob, Refused
from .models import Models
from .serving import Hosting, NoRoom, Serving
from .ui import routes as ui_routes
from .weights import ModelError

INFER_CHUNK = 1 << 13
INFER_TIMEOUT = 600.0


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
    schedule: Availability | None = None
    on_paused: str = "stop"
    schedule_path: Path | None = None
    serving: Serving | None = None
    models: Models | None = None
    cluster_key_path: Path | str | None = None
    tokens: Callable[[], set[str]] | None = None
    bench: BenchHost | None = None
    hosting: Hosting | None = None


def make_handler(daemon: Daemon) -> type[BaseHTTPRequestHandler]:
    """The request handler class for ``daemon``."""
    runner, files_root, token, name = daemon.runner, daemon.files_root, daemon.token, daemon.name
    report, fetcher, ui, schedule = daemon.report, daemon.fetcher, daemon.ui, daemon.schedule
    on_paused, schedule_path, serving = daemon.on_paused, daemon.schedule_path, daemon.serving
    models, cluster_key_path, tokens = daemon.models, daemon.cluster_key_path, daemon.tokens
    bench, hosting = daemon.bench, daemon.hosting

    class Handler(BaseHTTPRequestHandler):
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

        def _authed(self) -> bool:
            got = self.headers.get("Authorization", "")
            if not got.startswith("Bearer "):
                return False
            offered = got[7:]
            # A machine in several clusters answers to each of them.
            for good in {self._token(), *(tokens() if tokens else ())}:
                if good and secrets.compare_digest(offered, good):
                    return True
            return False

        def _send(self, code: int, payload: Any, *, raw: bytes | None = None,
                  headers: dict[str, str] | None = None,
                  content_type: str = "") -> None:
            body = raw if raw is not None else json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", content_type or (
                "application/octet-stream" if raw is not None else "application/json"))
            self.send_header("Content-Length", str(len(body)))
            for k, v in (headers or {}).items():
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

        def _guard(self) -> bool:
            if not self._authed():
                self._send(401, {"error": "bad or missing bearer token"})
                return False
            return True

        # -- routes --

        # -- inference proxy --
        def _proxy(self) -> bool:
            """Forward /infer/* to a model server on this machine's loopback.

            The model server stays on 127.0.0.1. This route is the only LAN-exposed
            one, and it already requires the bearer token.
            """
            if serving is None or not self.path.startswith("/infer"):
                return False
            if not self._guard():
                return True
            rest = self.path[len("/infer"):] or "/"
            parsed = urllib.parse.urlparse(rest)
            model = urllib.parse.parse_qs(parsed.query).get("ml_stack_model", [""])[0]
            port = serving.port_for(model)
            if port is None:
                self._send(503, {"error": "no model server is running on this machine"})
                return True

            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length) if length else None
            upstream = urllib.request.Request(
                f"http://127.0.0.1:{port}{rest}", data=body, method=self.command)
            for name, value in self.headers.items():
                if name.lower() in ("authorization", "host", "content-length",
                                    "connection", "x-ml-stack-ui"):
                    continue
                upstream.add_header(name, value)
            if body is not None:
                upstream.add_header("Content-Length", str(len(body)))

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

            self.send_response(response.status)
            self.send_header("Content-Type",
                             response.headers.get("Content-Type", "application/json"))
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
            return ui_routes(ui, self)

        def do_GET(self) -> None:
            if self.path == "/favicon.ico":
                self.send_response(204)
                self.end_headers()
                return
            if self._proxy():
                return
            if self._ui():
                return
            parsed = urllib.parse.urlparse(self.path)
            path, q = parsed.path, urllib.parse.parse_qs(parsed.query)
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
            if not self._guard():
                return
            if path == "/models":
                if models is None:
                    self._send(501, {"error": "no model store on this daemon"}); return
                self._send(200, {"models": [m.public() for m in models.all()],
                                 "free_gb": models.free_gb(),
                                 "store": str(models.store)})
                return
            if path.startswith("/models/"):
                if models is None:
                    self._send(501, {"error": "no model store on this daemon"}); return
                wanted = urllib.parse.unquote(path[len("/models/"):])
                found = models.find(wanted) or models.find_draft(wanted)
                if found is None:
                    self._send(404, {"error": f"no model called {wanted!r}"}); return
                self._send_file(found.path, *byte_range(self.headers.get("Range", "")))
                return
            if path == "/availability":
                if schedule is None:
                    self._send(501, {"error": "no schedule on this daemon"}); return
                self._send(200, schedule.public()); return
            if path == "/jobs":
                self._send(200, {"jobs": runner.snapshot()})
                return
            if path == "/speech/providers":
                self._send(200, speech.providers()); return
            if path == "/bench":
                if bench is None:
                    self._send(501, {"error": "this daemon takes no bench jobs"}); return
                self._send(200, {"ok": True, "name": self._name(), **bench.report(),
                                 "jobs": bench.snapshot()})
                return
            if path == "/bench/export":
                if bench is None:
                    self._send(501, {"error": "this daemon takes no bench jobs"}); return
                try:
                    out = bench.export(since=q.get("since", [""])[0],
                                       job=q.get("job", [""])[0],
                                       full=q.get("full", ["0"])[0] not in ("", "0"),
                                       anyway=q.get("anyway", ["0"])[0] not in ("", "0"))
                except DaemonError as e:
                    self._send(400, {"error": str(e)}); return
                self._send(200, {**out, "host": self._name()})
                return
            m = re.match(r"^/jobs/([^/]+)(/log|/metrics)?$", path)
            if m:
                job = runner.jobs.get(m.group(1))
                if job is None:
                    self._send(404, {"error": "unknown job"}); return
                kind = m.group(2)
                if kind is None:
                    self._send(200, job.public()); return
                if kind == "/log":
                    n = int(q.get("tail", ["200"])[0])
                    p = runner.log_path(job.id)
                    text = ""
                    if p.exists():
                        text = "".join(p.read_text(errors="replace").splitlines(True)[-n:])
                    self._send(200, {"log": text}); return
                since = int(q.get("since", ["0"])[0])
                mp = runner.job_dir(job.id) / "metrics.jsonl"
                rows: list[Any] = []
                if mp.exists():
                    for line in mp.read_text(errors="replace").splitlines()[since:]:
                        try:
                            rows.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
                self._send(200, {"metrics": rows, "next": since + len(rows)}); return
            m = re.match(r"^/fetch/([^/]+)$", path)
            if m:
                if fetcher is None:
                    self._send(501, {"error": "peer-to-peer fetch is not enabled"})
                    return
                fetch = fetcher.fetches.get(m.group(1))
                if fetch is None:
                    self._send(404, {"error": "unknown fetch"}); return
                self._send(200, fetch.public()); return
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
                    self._send(400, {"error": str(e)}); return
                if not target.is_file():
                    self._send(404, {"error": "not found"}); return
                start, end = 0, None
                rng = self.headers.get("Range", "")
                if rng.startswith("bytes="):
                    spec = rng.split("=", 1)[1].split(",")[0].strip()
                    head, _, tail = spec.partition("-")
                    if not head:
                        self._send(400, {"error": "suffix ranges not supported"})
                        return
                    try:
                        start = int(head)
                        end = int(tail) if tail else None
                    except ValueError:
                        self._send(400, {"error": f"bad Range: {rng!r}"}); return
                self._send_file(target, start, end)
                return
            self._send(404, {"error": "no such route"})

        do_HEAD = do_GET

        def do_POST(self) -> None:
            if self._proxy():
                return
            if self._ui():
                return
            if not self._guard():
                return
            parsed = urllib.parse.urlparse(self.path)
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length) if length else b"{}"
            if parsed.path == "/speech/transcribe":
                want = urllib.parse.parse_qs(parsed.query)
                try:
                    heard = transcribe(body,
                                       provider=want.get("provider", [""])[0] or None,
                                       language=want.get("language", [""])[0] or None)
                except ProviderError as e:
                    self._send(503, {"error": str(e)}); return
                self._send(200, as_json(heard)); return
            if parsed.path == "/jobs":
                try:
                    req = json.loads(body or b"{}")
                    argv = req.get("argv")
                    if isinstance(argv, str):
                        argv = shlex.split(argv)
                    # `or`, not a get() default: the client sends "cwd": "" when the
                    # caller did not set one, so the default never applied and the job
                    # ran wherever the daemon happened to be.
                    job = runner.submit(req.get("name", ""), argv or [],
                                        req.get("cwd") or str(files_root),
                                        req.get("env"))
                except (DaemonError, ValueError) as e:
                    self._send(400, {"error": str(e)}); return
                self._send(201, job.public()); return
            if parsed.path == "/bench":
                if bench is None:
                    self._send(501, {"error": "this daemon takes no bench jobs"}); return
                try:
                    job = bench.submit(BenchJob.from_request(json.loads(body or b"{}")))
                except Refused as e:
                    # 409: the job is well-formed and this peer will not run it now --
                    # its code, its lock or its memory says so, and `refused` says which
                    self._send(409, {"error": str(e), "refused": e.kind}); return
                except (DaemonError, ValueError) as e:
                    self._send(400, {"error": str(e)}); return
                self._send(201, job.public()); return
            if parsed.path == "/models/get":
                if models is None:
                    self._send(501, {"error": "no model store on this daemon"}); return
                req = json.loads(body or b"{}")
                try:
                    got = models.ensure(str(req.get("name") or ""),
                                        source=str(req.get("source") or ""),
                                        key=load_cluster_key(cluster_key_path),
                                        autodownload=bool(req.get("autodownload", True)))
                except (ModelError, ValueError) as e:
                    self._send(400, {"error": str(e)}); return
                self._send(200, got.public()); return
            if parsed.path == "/availability":
                if schedule is None:
                    self._send(501, {"error": "no schedule on this daemon"}); return
                req = json.loads(body or b"{}")
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
                    self._send(400, {"error": str(e)}); return
                if schedule_path is not None:
                    schedule.save(schedule_path)
                self._send(200, schedule.public()); return
            if parsed.path == "/fetch":
                if fetcher is None:
                    self._send(501, {"error": "peer-to-peer fetch is not enabled"})
                    return
                try:
                    req = json.loads(body or b"{}")
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
                    self._send(400, {"error": str(e)}); return
                self._send(202, fetch.public()); return
            if parsed.path == "/serve":
                if hosting is None or models is None:
                    self._send(501, {"error": "this daemon serves no models"}); return
                try:
                    req = json.loads(body or b"{}")
                    wanted = str(req.get("model") or "")
                    context = int(req.get("context") or 8192)
                    parallel = max(1, int(req.get("parallel") or 1))
                except (TypeError, ValueError) as e:
                    self._send(400, {"error": str(e)}); return
                if not wanted:
                    self._send(400, {"error": "name the model"}); return
                found = models.find(wanted)
                if found is None:
                    self._send(404, {"error": f"no model called {wanted!r} on this "
                                              f"machine"}); return
                running = hosting.already(found.name)
                if running is not None:
                    self._send(200, running.public()); return
                try:
                    served = hosting.start(found.path, name=found.name, context=context,
                                           parallel=parallel,
                                           room=int(report().get("room_bytes") or 0))
                except NoRoom as e:
                    self._send(409, {"error": str(e), "refused": "room"}); return
                except (OSError, RuntimeError, ValueError) as e:
                    self._send(502, {"error": str(e)}); return
                self._send(201, served.public()); return
            m = re.match(r"^/jobs/([^/]+)/stop$", parsed.path)
            if m:
                try:
                    job = runner.stop(m.group(1))
                except DaemonError as e:
                    self._send(404, {"error": str(e)}); return
                self._send(200, job.public()); return
            self._send(404, {"error": "no such route"})

        def do_DELETE(self) -> None:
            if self._ui():
                return
            self._send(404, {"error": "no such route"})

        def do_PUT(self) -> None:
            if self._ui():
                return
            if not self._guard():
                return
            parsed = urllib.parse.urlparse(self.path)
            if not parsed.path.startswith("/files/"):
                self._send(404, {"error": "no such route"}); return
            try:
                target = safe_relpath(files_root, parsed.path[len("/files/"):])
            except DaemonError as e:
                self._send(400, {"error": str(e)}); return
            target.parent.mkdir(parents=True, exist_ok=True)
            length = int(self.headers.get("Content-Length", "0"))
            cr = self.headers.get("Content-Range", "")
            offset = 0
            if cr.startswith("bytes "):
                offset = int(cr.split(" ", 1)[1].split("-")[0])
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
                remaining = length
                while remaining > 0:
                    chunk = self.rfile.read(min(1 << 20, remaining))
                    if not chunk:
                        break
                    fh.write(chunk)
                    remaining -= len(chunk)
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
