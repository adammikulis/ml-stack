"""Bounded, durable background installation jobs for the local UI."""
import logging
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from poolhouse.graph.store import GraphStore
from poolhouse.lock import Busy, only_one
from poolhouse.serve.process import pid_exists, started_at

from . import llama
from .session import parse_cookie

ACTIVE = {"queued", "installing"}
PENDING = ACTIVE | {"waiting"}
LIMIT = 32


class Jobs:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._mutex = threading.RLock()
        self._pending = []
        self._worker = None
        self._recover()

    def _store(self):
        return GraphStore(self.root / "setup-jobs.db")

    def _recover(self):
        with self._mutex, only_one(self.root / "setup-jobs.lock"), self._store() as graph:
            for node in graph.nodes("setup-job"):
                row = node["attrs"]
                if row["state"] in ACTIVE and (not pid_exists(row["pid"]) or started_at(row["pid"]) != row["born"]):
                    row.update(state="failed", error="Installation interrupted; retry to continue.", note="Installation interrupted")
                    self._save(graph, row)

    @staticmethod
    def _save(graph, row):
        graph.upsert_node({"id": row["id"], "kind": "setup-job", "label": row["kind"], "attrs": row})

    def all(self):
        with self._mutex, only_one(self.root / "setup-jobs.lock"), self._store() as graph:
            return sorted((node["attrs"] for node in graph.nodes("setup-job")), key=lambda row: row["created"])

    def active(self):
        return any(row["state"] in ACTIVE for row in self.all())

    def start(self, kind, request, operation, *, provenance=None):
        try:
            with only_one(self.root / "runtime-install.lock", wait=False):
                return self._enqueue(kind, request, operation, provenance=provenance)
        except Busy:
            raise ValueError("A runtime update is in progress; retry the installation shortly.") from None

    def _enqueue(self, kind, request, operation, *, provenance=None):
        with self._mutex, only_one(self.root / "setup-jobs.lock"), self._store() as graph:
            rows = [node["attrs"] for node in graph.nodes("setup-job")]
            existing = next((row for row in rows if row["kind"] == kind and row["request"] == request and row["state"] in PENDING), None)
            if existing:
                return existing
            active = [row for row in rows if row["state"] in PENDING]
            if len(active) >= LIMIT:
                raise ValueError("The setup installation queue is full; wait for a job to finish.")
            finished = sorted((row for row in rows if row["state"] not in PENDING), key=lambda row: row["created"])
            for row in finished[:max(0, len(rows) - LIMIT + 1)]:
                graph.query("MATCH (n:Node {id:$id}) DETACH DELETE n", {"id": row["id"]})
            row = {"id": uuid.uuid4().hex, "kind": kind, "request": request, "provenance": provenance or {}, "state": "queued" if operation else "waiting", "note": "Waiting to install", "error": "", "result": {}, "created": time.time(), "pid": os.getpid(), "born": started_at(os.getpid())}
            self._save(graph, row)
            if operation:
                self._pending.append((row, operation))
                if self._worker is None or not self._worker.is_alive():
                    self._worker = threading.Thread(target=self._run, daemon=True, name="setup-installations")
                    self._worker.start()
            return dict(row)

    def _update(self, row, **changes):
        with self._mutex, only_one(self.root / "setup-jobs.lock"), self._store() as graph:
            latest = next((node["attrs"] for node in graph.nodes("setup-job") if node["id"] == row["id"]), None)
            if latest is not None:
                row.update(latest)
            row.update(changes)
            self._save(graph, row)

    def _run(self):
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="setup-operation") as executor:
            while True:
                with self._mutex:
                    if not self._pending:
                        self._worker = None
                        return
                    row, operation = self._pending.pop(0)
                outcome = executor.submit(self._install, row, operation)
                failure = outcome.exception()
                if failure is None:
                    continue
                if not isinstance(failure, Exception):
                    raise failure
                logging.getLogger(__name__).error("Setup installation failed", exc_info=(type(failure), failure, failure.__traceback__))
                self._update(row, state="failed", error=str(failure)[:2000], note="Installation failed")

    def _install(self, row, operation):
        with only_one(self.root / "setup-install.lock"):
            self._update(row, state="installing", note="Preparing installation")
            result = operation(lambda note, current=row: self._update(current, note=str(note)[:1000]))
            errors = [str(value.get("error") or "Installation failed") for value in result.get("changed", {}).values() if not value.get("ok")]
            if errors:
                self._update(row, state="failed", error="; ".join(errors)[:2000], note="Installation failed", result=result)
            else:
                self._update(row, state="done", note="Installation complete", result=result)


def jobs(ui):
    with ui._setup_jobs_lock:
        if ui.setup_jobs is None:
            ui.setup_jobs = Jobs(ui.root)
        return ui.setup_jobs


def libraries(ui, vendor, add, drop, progress):
    changed = {}
    if drop:
        progress("Removing selected libraries")
        changed.update(ui.environment.uninstall(drop))
    if add:
        settings = ui.settings
        if settings is not None and settings.download_sources not in ("internet", "both"):
            raise RuntimeError("Installing libraries requires Internet only or Both download sources.")
        changed.update(ui.environment.install(add, on_progress=progress))
    return {"changed": changed, **ui.environment.state(vendor)}


def server(ui, progress):
    got = llama.ensure_server(ui.root, on_progress=progress, sources=ui.settings.download_sources if ui.settings else "both")
    return {"ok": True, "server": str(got)}


def provenance(route):
    sessions = getattr(route.ui, "sessions", None)
    session = sessions.get(parse_cookie(getattr(route, "cookie", ""))) if sessions else None
    return {"channel": "local-ui", "authentication": session.who if session else "loopback-setup", "client": getattr(route, "client_ip", "")}
