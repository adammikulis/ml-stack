"""Client for a training daemon on another machine.

    from mainspring.train.remote import RemoteTrainer

    rtx = RemoteTrainer("http://rtx:8770", token=...)
    rtx.push("train/data/packed/train.npy", "data/train.npy")
    job = rtx.submit(["python", "-m", "train.cuda_train", "--steps", "30000"],
                     name="doly-full")
    rtx.wait(job["id"], on_metric=print)
    rtx.pull(f"jobs/{job['id']}/ckpt/best/model.safetensors", "local/model.safetensors")

stdlib only. Uploads and downloads resume, because a 2GB dataset over a home
network will be interrupted eventually and restarting from zero each time
turns a five-minute transfer into an afternoon.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

CHUNK = 8 << 20          # 8MB: big enough to be fast, small enough to resume cheaply


class RemoteError(RuntimeError):
    pass


class RemoteTrainer:
    def __init__(self, base_url: str, token: str, *, timeout: float = 60.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout

    # -- plumbing --------------------------------------------------------
    def _request(self, method: str, path: str, *, data: bytes | None = None,
                 headers: dict[str, str] | None = None,
                 timeout: float | None = None) -> tuple[int, bytes, Any]:
        req = urllib.request.Request(f"{self.base_url}{path}", data=data,
                                     method=method)
        req.add_header("Authorization", f"Bearer {self.token}")
        for k, v in (headers or {}).items():
            req.add_header(k, v)
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.timeout) as r:
                return r.status, r.read(), dict(r.headers)
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")
            # The daemon puts the reason in the body; a bare "HTTP 400" from
            # urllib tells you nothing about which field it rejected.
            raise RemoteError(f"{method} {path} -> {e.code}: {body[:400]}") from None
        except urllib.error.URLError as e:
            raise RemoteError(f"{method} {path} -> unreachable: {e.reason}") from None

    def _json(self, method: str, path: str, payload: Any = None) -> Any:
        data = json.dumps(payload).encode() if payload is not None else None
        headers = {"Content-Type": "application/json"} if data else {}
        _, body, _ = self._request(method, path, data=data, headers=headers)
        return json.loads(body or b"{}")

    # -- status ----------------------------------------------------------
    def health(self) -> dict:
        req = urllib.request.Request(f"{self.base_url}/health")
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read())

    def jobs(self) -> list[dict]:
        return self._json("GET", "/jobs")["jobs"]

    def job(self, job_id: str) -> dict:
        return self._json("GET", f"/jobs/{job_id}")

    def log(self, job_id: str, tail: int = 200) -> str:
        return self._json("GET", f"/jobs/{job_id}/log?tail={tail}")["log"]

    def metrics(self, job_id: str, since: int = 0) -> tuple[list[dict], int]:
        out = self._json("GET", f"/jobs/{job_id}/metrics?since={since}")
        return out["metrics"], out["next"]

    # -- jobs ------------------------------------------------------------
    def submit(self, argv: list[str] | str, *, name: str = "",
               cwd: str = "", env: dict[str, str] | None = None) -> dict:
        return self._json("POST", "/jobs", {"argv": argv, "name": name,
                                            "cwd": cwd, "env": env or {}})

    def stop(self, job_id: str) -> dict:
        return self._json("POST", f"/jobs/{job_id}/stop", {})

    def wait(self, job_id: str, *, poll_s: float = 20.0,
             on_metric: Callable[[dict], None] | None = None,
             timeout_s: float | None = None) -> dict:
        """Block until the job leaves the running/queued states."""
        since, deadline = 0, (time.time() + timeout_s) if timeout_s else None
        while True:
            job = self.job(job_id)
            if on_metric is not None:
                rows, since = self.metrics(job_id, since)
                for row in rows:
                    on_metric(row)
            if job["state"] not in ("queued", "running"):
                return job
            if deadline and time.time() > deadline:
                raise RemoteError(f"job {job_id} still {job['state']} after timeout")
            time.sleep(poll_s)

    # -- files -----------------------------------------------------------
    def push(self, local: Path | str, remote: str, *,
             on_progress: Callable[[int, int], None] | None = None) -> dict:
        """Upload, resuming from whatever the daemon already holds."""
        local = Path(local).expanduser()
        total = local.stat().st_size
        sent = 0
        with local.open("rb") as fh:
            while True:
                chunk = fh.read(CHUNK)
                if not chunk:
                    break
                last = fh.tell() >= total
                headers = {
                    "Content-Range": f"bytes {sent}-{sent+len(chunk)-1}/{total}",
                    "X-Mainspring-Complete": "1" if last else "0",
                }
                _, body, _ = self._request("PUT", f"/files/{remote}", data=chunk,
                                           headers=headers, timeout=300)
                sent += len(chunk)
                if on_progress:
                    on_progress(sent, total)
        return json.loads(body or b"{}")

    def pull(self, remote: str, local: Path | str, *,
             on_progress: Callable[[int, int], None] | None = None) -> Path:
        """Download, resuming a partial file rather than restarting it.

        Lands in ``.part`` and is moved with os.replace only when complete, so
        an interrupted pull never leaves a truncated file that a later step
        mistakes for a whole one.
        """
        local = Path(local).expanduser()
        local.parent.mkdir(parents=True, exist_ok=True)
        partial = local.with_suffix(local.suffix + ".part")
        start = partial.stat().st_size if partial.exists() else 0
        headers = {"Range": f"bytes={start}-"} if start else {}
        status, body, hdrs = self._request("GET", f"/files/{remote}",
                                           headers=headers, timeout=600)
        with partial.open("ab" if start else "wb") as fh:
            fh.write(body)
        if on_progress:
            on_progress(partial.stat().st_size, partial.stat().st_size)
        os.replace(partial, local)
        return local
