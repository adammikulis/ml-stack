"""Client for a training daemon on another machine.

    from ml_stack.train.remote import RemoteTrainer

    rtx = RemoteTrainer("http://rtx:8770", token=...)
    rtx.push("train/data/packed/train.npy", "data/train.npy")
    job = rtx.submit(["python", "-m", "train.cuda_train", "--steps", "30000"],
                     name="doly-full")
    rtx.wait(job["id"], on_metric=print)
    rtx.pull(f"jobs/{job['id']}/ckpt/best/model.safetensors", "local/model.safetensors")

Or, with a cluster key in place, without knowing where it is:

    rtx = RemoteTrainer.find_one(require="cuda")

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

from .discovery import Beacon, DiscoveryError, derive_token, discover, key_path, load_cluster_key

CHUNK = 8 << 20          # 8MB: big enough to be fast, small enough to resume cheaply


class RemoteError(RuntimeError):
    pass


class RemoteTrainer:
    def __init__(self, base_url: str, token: str, *, timeout: float = 60.0,
                 beacon: Beacon | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout
        #: Set when this client came from discovery; carries the peer's name
        #: and device report so a caller can choose between several.
        self.beacon = beacon

    @property
    def name(self) -> str:
        return self.beacon.name if self.beacon else self.base_url

    # -- discovery -------------------------------------------------------
    @classmethod
    def discover(cls, *, timeout_s: float = 2.0, key: bytes | None = None,
                 cluster_key_path: Path | str | None = None,
                 timeout: float = 60.0, group: str | None = None,
                 port: int | None = None) -> list["RemoteTrainer"]:
        """Every daemon on the LAN that proves it holds the cluster key.

        No token argument: it is derived from the same key that authenticated
        the peer, so a peer you can find is a peer you can already drive.
        """
        key = key or load_cluster_key(cluster_key_path)
        if key is None:
            raise DiscoveryError(
                f"no cluster key at {key_path(cluster_key_path)} -- "
                "run 'ml-stack-peers init' here and copy the key to each box")
        token = derive_token(key)
        return [cls(b.base_url, token, timeout=timeout, beacon=b)
                for b in discover(key, timeout_s=timeout_s, group=group, port=port)]

    @classmethod
    def find_one(cls, *, require: str = "", name: str = "",
                 timeout_s: float = 2.0, key: bytes | None = None,
                 cluster_key_path: Path | str | None = None,
                 timeout: float = 60.0, group: str | None = None,
                 port: int | None = None) -> "RemoteTrainer":
        """The one peer to use, or an error saying exactly why there isn\'t one.

        ``require`` filters on a backend the peer reports ("cuda", "mps"), and
        ``name`` on its advertised name. Ambiguity is an error rather than a
        silent pick: two GPU boxes on one LAN and a coin-flip between them is
        how a run ends up on the wrong card without anyone noticing.
        """
        peers = cls.discover(timeout_s=timeout_s, key=key,
                             cluster_key_path=cluster_key_path, timeout=timeout,
                             group=group, port=port)
        if not peers:
            raise DiscoveryError(
                "no peers answered -- is traind running there, is it on this "
                "LAN, and does it hold the same cluster key?")
        pool = peers
        if name:
            pool = [p for p in pool if p.beacon and p.beacon.name == name]
        if require:
            pool = [p for p in pool
                    if p.beacon and (p.beacon.device.get(require)
                                     or require in (p.beacon.device.get("backends") or []))]
        if not pool:
            seen = ", ".join(f"{p.name} {sorted(p.beacon.device.get('backends') or [])}"
                             for p in peers if p.beacon)
            raise DiscoveryError(
                f"no peer matches name={name!r} require={require!r}; found: {seen}")
        if len(pool) > 1:
            raise DiscoveryError(
                "several peers match, name one: "
                + ", ".join(p.name for p in pool))
        return pool[0]

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
                    "X-ML-Stack-Complete": "1" if last else "0",
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
