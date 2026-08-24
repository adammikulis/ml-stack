"""Model files this machine holds, and getting one it does not."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["Model", "Models", "ModelError", "default_roots"]

SUFFIXES = (".gguf", ".safetensors", ".bin", ".pt", ".onnx")
CHUNK = 1 << 20
MIN_SIZE = 1 << 20


class ModelError(RuntimeError):
    pass


def _read_stamp(stamp: Path) -> dict[str, Any]:
    """What a half-finished download recorded about where it came from."""
    try:
        raw = json.loads(stamp.read_text())
    except (OSError, ValueError):
        return {}
    return raw if isinstance(raw, dict) else {}


def _write_stamp(stamp: Path, url: str, headers: Any) -> None:
    validator = headers.get("ETag") or headers.get("Last-Modified") or ""
    try:
        stamp.write_text(json.dumps({"url": url, "validator": validator}))
    except OSError:
        pass


@dataclass(frozen=True, slots=True)
class Model:
    name: str
    path: Path
    size: int
    modified: float

    def public(self) -> dict[str, Any]:
        return {"name": self.name, "size": self.size, "modified": self.modified}


def default_roots(root: Path | str) -> list[Path]:
    """Where model files live. The llama.cpp cache is included because a model pulled
    by a server is one this machine already holds."""
    home = Path.home()
    return [
        Path(root).expanduser() / "models",
        home / ".cache" / "llama.cpp",
        home / ".cache" / "huggingface" / "hub",
        home / "models",
    ]


@dataclass
class Models:
    """The model files on this machine."""

    roots: list[Path]
    store: Path
    _digests: dict[tuple[str, int, int], str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.store = Path(self.store).expanduser()
        self.roots = [Path(r).expanduser() for r in self.roots]

    def all(self) -> list[Model]:
        seen: dict[str, Model] = {}
        for root in self.roots:
            if not root.is_dir():
                continue
            for path in sorted(root.rglob("*")):
                if not path.is_file() or path.suffix.lower() not in SUFFIXES:
                    continue
                try:
                    stat = path.stat()
                except OSError:
                    continue
                if stat.st_size < MIN_SIZE:
                    continue
                if path.name not in seen:
                    seen[path.name] = Model(path.name, path, stat.st_size,
                                            stat.st_mtime)
        return sorted(seen.values(), key=lambda m: m.name.lower())

    def find(self, name: str) -> Model | None:
        needle = name.strip().lower()
        for model in self.all():
            if model.name.lower() == needle:
                return model
        for model in self.all():
            if needle in model.name.lower():
                return model
        return None

    def digest(self, model: Model) -> str:
        """sha256, cached against size and mtime so a large file is read once."""
        stat = model.path.stat()
        key = (str(model.path), stat.st_size, stat.st_mtime_ns)
        found = self._digests.get(key)
        if found:
            return found
        h = hashlib.sha256()
        with model.path.open("rb") as fh:
            while True:
                block = fh.read(CHUNK)
                if not block:
                    break
                h.update(block)
        self._digests[key] = h.hexdigest()
        return self._digests[key]

    def public(self, limit: int = 24) -> list[dict[str, Any]]:
        """What goes in the beacon. Names and sizes only -- a beacon is a UDP packet,
        and a full path is needless disclosure."""
        return [m.public() for m in self.all()[:limit]]

    # -- getting one ----------------------------------------------------
    def where(self, name: str, key: bytes, *, timeout_s: float = 2.0
              ) -> list[tuple[str, str, int]]:
        """Machines on this network holding a model. (peer, base_url, size)."""
        from .discovery import discover

        needle = name.strip().lower()
        out = []
        for beacon in discover(key, timeout_s=timeout_s):
            for row in (beacon.device.get("models") or []):
                if needle in str(row.get("name", "")).lower():
                    out.append((beacon.name, beacon.base_url, int(row.get("size") or 0)))
                    break
        return out

    def ensure(self, name: str, *, source: str = "", key: bytes | None = None,
               on_progress: Any = None, autodownload: bool = True) -> Model:
        """Make sure this machine holds a model, preferring one on the network."""
        found = self.find(name)
        if found:
            return found
        if not autodownload:
            raise ModelError(
                f"{name} is not on this machine, and automatic downloading is off")

        if key is not None:
            for peer_name, base_url, _size in self.where(name, key):
                if on_progress:
                    on_progress(f"Copying {name} from {peer_name}")
                try:
                    return self._from_peer(name, base_url, key, on_progress)
                except (ModelError, OSError):
                    continue

        if not source:
            raise ModelError(
                f"no machine on this network has {name}, and no download was given")
        if on_progress:
            on_progress(f"Downloading {name}")
        return self._from_internet(name, source, on_progress)

    def _from_peer(self, name: str, base_url: str, key: bytes,
                   on_progress: Any) -> Model:
        from .discovery import derive_token
        from .remote import Peer

        peer = Peer(base_url, derive_token(key))
        holds = peer.models()
        match = next((m for m in holds
                      if name.strip().lower() in str(m.get("name", "")).lower()), None)
        if match is None:
            raise ModelError(f"{base_url} no longer has {name}")
        self.store.mkdir(parents=True, exist_ok=True)
        target = self.store / str(match["name"])
        peer.pull(str(match["name"]), target, on_progress=on_progress,
                  route="/models/")
        stat = target.stat()
        return Model(target.name, target, stat.st_size, stat.st_mtime)

    def _from_internet(self, name: str, source: str, on_progress: Any) -> Model:
        from .remote import range_total

        url = _resolve(source)
        self.store.mkdir(parents=True, exist_ok=True)
        # The name that was asked for wins: saving it under whatever the URL happened
        # to call it means find() will not match it afterwards.
        wanted = Path(name).name
        if Path(wanted).suffix.lower() not in SUFFIXES:
            wanted = Path(urllib.parse.urlparse(url).path).name or wanted
        target = self.store / wanted
        partial = target.with_suffix(target.suffix + ".part")
        stamp = Path(str(partial) + ".from")

        start = partial.stat().st_size if partial.exists() else 0
        origin = _read_stamp(stamp)
        if start and origin.get("url") not in (None, url):
            partial.unlink(missing_ok=True)
            stamp.unlink(missing_ok=True)
            start, origin = 0, {}

        req = urllib.request.Request(url, headers={"User-Agent": "ml-stack"})
        if start:
            req.add_header("Range", f"bytes={start}-")
            if origin.get("validator"):
                req.add_header("If-Range", str(origin["validator"]))
        try:
            response = urllib.request.urlopen(req, timeout=120)
        except urllib.error.HTTPError as exc:
            if exc.code == 416 and start:
                size = range_total(exc.headers.get("Content-Range", ""))
                if size is not None and size == start:
                    os.replace(partial, target)
                    stamp.unlink(missing_ok=True)
                    stat = target.stat()
                    return Model(target.name, target, stat.st_size, stat.st_mtime)
                partial.unlink(missing_ok=True)
                stamp.unlink(missing_ok=True)
                raise ModelError(
                    f"{name}: the part here is {start} bytes but the file is "
                    f"{size}; discarded it, ask again") from None
            raise ModelError(f"could not download {name}: {exc.code}") from None
        except (urllib.error.URLError, OSError) as exc:
            raise ModelError(f"could not download {name}: {exc}") from None

        # A server that does not honour Range answers 200 with the whole file.
        if start and response.status != 206:
            start = 0
        if start:
            total = range_total(response.headers.get("Content-Range", "")) or 0
        else:
            total = int(response.headers.get("Content-Length") or 0)

        _write_stamp(stamp, url, response.headers)
        done = start
        with response, partial.open("ab" if start else "wb") as fh:
            while True:
                block = response.read(CHUNK)
                if not block:
                    break
                fh.write(block)
                done += len(block)
                if on_progress:
                    on_progress(done, total)
        if total and partial.stat().st_size != total:
            raise ModelError(
                f"{name}: got {partial.stat().st_size} of {total} bytes; "
                f"left {partial.name} to resume from")
        os.replace(partial, target)
        stamp.unlink(missing_ok=True)
        stat = target.stat()
        return Model(target.name, target, stat.st_size, stat.st_mtime)

    def remove(self, name: str) -> bool:
        found = self.find(name)
        if found is None or self.store not in found.path.parents:
            return False
        found.path.unlink(missing_ok=True)
        return True

    def free_gb(self) -> float:
        try:
            self.store.mkdir(parents=True, exist_ok=True)
            return round(shutil.disk_usage(self.store).free / 2**30, 1)
        except OSError:
            return 0.0


def _resolve(source: str) -> str:
    """``hf:owner/repo/file.gguf`` or a plain URL."""
    source = source.strip()
    if source.startswith("hf:"):
        ref = source[3:].strip("/")
        parts = ref.split("/")
        if len(parts) < 3:
            raise ModelError(f"{source!r} should look like hf:owner/repo/file.gguf")
        owner, repo, path = parts[0], parts[1], "/".join(parts[2:])
        return f"https://huggingface.co/{owner}/{repo}/resolve/main/{path}?download=true"
    if source.startswith(("http://", "https://")):
        return source
    raise ModelError(f"{source!r} is not a URL or an hf: reference")
