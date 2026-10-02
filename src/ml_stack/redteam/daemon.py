"""The fleet daemon's real request handler on a loopback port, in front of a model server."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from ml_stack.fleet.api import Daemon, make_handler
from ml_stack.fleet.daemon import load_or_create_token
from ml_stack.fleet.jobs import JobRunner
from ml_stack.fleet.serving import Serving
from ml_stack.http import Server

__all__ = ["Running", "running"]


@dataclass(frozen=True, slots=True)
class Running:
    """A daemon answering on ``url``; ``token`` is the bearer token it accepts."""

    url: str
    token: str
    files: Path


@contextmanager
def running(root: Path, model_port: int | None = None, name: str = "model") -> Iterator[Running]:
    """A daemon rooted in ``root``; with ``model_port`` its ``/infer`` proxies to that
    server."""
    files = root / "files"
    files.mkdir(parents=True, exist_ok=True)
    token = load_or_create_token(root)
    serving = Serving(root / "serving.json")
    if model_port is not None:
        serving.register(model_port, [name], slots=2)
    runner = JobRunner(root, files)
    httpd = Server(("127.0.0.1", 0), make_handler(
        Daemon(runner, files, token, "redteam", serving=serving)))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        yield Running(f"http://127.0.0.1:{httpd.server_address[1]}", token, files)
    finally:
        runner.shutdown()
        httpd.shutdown()
        httpd.server_close()
