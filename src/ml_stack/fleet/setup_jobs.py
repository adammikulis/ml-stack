"""Bounded, durable background installation jobs for the local UI."""
import logging
import os
import threading
import time
import uuid
from pathlib import Path

from ml_stack.graph.store import GraphStore
from ml_stack.lock import only_one
from ml_stack.serve.process import pid_exists, started_at

ACTIVE = {"queued", "installing"}
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

    def start(self, kind, request, operation, *, provenance=None):
        with self._mutex, only_one(self.root / "setup-jobs.lock"), self._store() as graph:
            rows = [node["attrs"] for node in graph.nodes("setup-job")]
            existing = next((row for row in rows if row["kind"] == kind and row["request"] == request and row["state"] in ACTIVE), None)
            if existing:
                return existing
            active = [row for row in rows if row["state"] in ACTIVE]
            if len(active) >= LIMIT:
                raise ValueError("The setup installation queue is full; wait for a job to finish.")
            finished = sorted((row for row in rows if row["state"] not in ACTIVE), key=lambda row: row["created"])
            for row in finished[:max(0, len(rows) - LIMIT + 1)]:
                graph.query("MATCH (n:Node {id:$id}) DETACH DELETE n", {"id": row["id"]})
            row = {"id": uuid.uuid4().hex, "kind": kind, "request": request, "provenance": provenance or {}, "state": "queued", "note": "Waiting to install", "error": "", "result": {}, "created": time.time(), "pid": os.getpid(), "born": started_at(os.getpid())}
            self._save(graph, row)
            self._pending.append((row, operation))
            if self._worker is None or not self._worker.is_alive():
                self._worker = threading.Thread(target=self._run, daemon=True, name="setup-installations")
                self._worker.start()
            return dict(row)

    def _update(self, row, **changes):
        with self._mutex, only_one(self.root / "setup-jobs.lock"), self._store() as graph:
            row.update(changes)
            self._save(graph, row)

    def _run(self):
        while True:
            with self._mutex:
                if not self._pending:
                    self._worker = None
                    return
                row, operation = self._pending.pop(0)
            try:
                with only_one(self.root / "setup-install.lock"):
                    self._update(row, state="installing", note="Preparing installation")
                    result = operation(lambda note, current=row: self._update(current, note=str(note)[:1000]))
                    errors = [str(value.get("error") or "Installation failed") for value in result.get("changed", {}).values() if not value.get("ok")]
                    if errors:
                        self._update(row, state="failed", error="; ".join(errors)[:2000], note="Installation failed", result=result)
                    else:
                        self._update(row, state="done", note="Installation complete", result=result)
            except Exception as exc:
                logging.getLogger(__name__).exception("Setup installation failed")
                self._update(row, state="failed", error=str(exc)[:2000], note="Installation failed")


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
    from .llama import ensure_server
    got = ensure_server(ui.root, on_progress=progress, sources=ui.settings.download_sources if ui.settings else "both")
    return {"ok": True, "server": str(got)}


def provenance(route):
    from .session import parse_cookie
    sessions = getattr(route.ui, "sessions", None)
    session = sessions.get(parse_cookie(getattr(route, "cookie", ""))) if sessions else None
    return {"channel": "local-ui", "authentication": session.who if session else "loopback-setup", "client": getattr(route, "client_ip", "")}
