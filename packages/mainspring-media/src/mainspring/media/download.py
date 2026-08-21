"""Fetch a model asset to disk, once, safely.

Model weights are large and the network is not reliable, so this does four things a
plain ``urlretrieve`` does not: it lands the download in a ``.part`` file and moves it
into place only after verification, it resumes an interrupted transfer via HTTP Range,
it verifies size and digest, and it reports progress.

Progress is a callback rather than a print, so a CLI bar and a server-sent event stream
are both consumers rather than two forks of the download.
"""

from __future__ import annotations

import hashlib
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

_CHUNK = 1 << 16
_PROGRESS_INTERVAL_S = 0.25


class DownloadError(RuntimeError):
    """The asset could not be fetched, or arrived corrupt."""


@dataclass(frozen=True, slots=True)
class Progress:
    name: str
    downloaded: int
    total: int | None
    resumed_from: int = 0

    @property
    def fraction(self) -> float | None:
        if not self.total:
            return None
        return min(1.0, self.downloaded / self.total)


ProgressFn = Callable[[Progress], None]


def _digest(path: Path, algorithm: str) -> str:
    h = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(_CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def _verify(path: Path, *, expect_sha256: str | None, expect_bytes: int | None, name: str) -> None:
    if expect_bytes is not None:
        actual = path.stat().st_size
        if actual != expect_bytes:
            path.unlink(missing_ok=True)
            raise DownloadError(
                f"{name}: expected {expect_bytes} bytes, got {actual}. Removed the partial file."
            )
    if expect_sha256:
        actual = _digest(path, "sha256")
        if actual.lower() != expect_sha256.lower():
            path.unlink(missing_ok=True)
            raise DownloadError(
                f"{name}: sha256 mismatch (expected {expect_sha256}, got {actual}). "
                "Removed the corrupt file."
            )


def fetch(
    url: str,
    target: Path | str,
    *,
    name: str | None = None,
    on_progress: ProgressFn | None = None,
    expect_sha256: str | None = None,
    expect_bytes: int | None = None,
    resume: bool = True,
    timeout: float = 60.0,
) -> Path:
    """Download ``url`` to ``target``, returning the path. Idempotent.

    An existing, verified ``target`` is returned untouched -- this is what makes it safe
    to call on every startup, which is the usual way to use it.

    The download lands in ``target.with_suffix(target.suffix + ".part")`` and is moved
    into place with ``os.replace`` only after verification, so a killed process never
    leaves a half-file that a later run mistakes for a complete one.
    """
    target = Path(target).expanduser()
    label = name or target.name

    if target.exists() and target.stat().st_size > 0:
        if expect_sha256 or expect_bytes:
            _verify(target, expect_sha256=expect_sha256, expect_bytes=expect_bytes, name=label)
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")

    start_at = partial.stat().st_size if (resume and partial.exists()) else 0
    request = urllib.request.Request(url)
    if start_at:
        request.add_header("Range", f"bytes={start_at}-")

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            # A server that ignores Range answers 200 with the whole body; appending to
            # the partial file would then produce a corrupt archive that only fails at
            # decompression time. Detect it and start over instead.
            if start_at and response.status != 206:
                start_at = 0

            remaining = response.headers.get("content-length")
            total: int | None = None
            if remaining is not None:
                try:
                    total = int(remaining) + start_at
                except ValueError:
                    total = None

            mode = "ab" if start_at else "wb"
            downloaded = start_at
            last_report = 0.0

            with partial.open(mode) as handle:
                while True:
                    block = response.read(_CHUNK)
                    if not block:
                        break
                    handle.write(block)
                    downloaded += len(block)

                    if on_progress is not None:
                        now = time.monotonic()
                        if now - last_report >= _PROGRESS_INTERVAL_S:
                            last_report = now
                            on_progress(Progress(label, downloaded, total, start_at))

            if on_progress is not None:
                on_progress(Progress(label, downloaded, total, start_at))

    except urllib.error.HTTPError as exc:
        # A stale .part against a changed file yields 416. Drop it and let the next call
        # start clean rather than wedging forever.
        if exc.code == 416 and partial.exists():
            partial.unlink(missing_ok=True)
            raise DownloadError(
                f"{label}: server rejected the resume range; the partial file was stale "
                "and has been removed. Retry."
            ) from exc
        raise DownloadError(f"{label}: HTTP {exc.code} fetching {url}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise DownloadError(f"{label}: cannot fetch {url} ({exc})") from exc

    _verify(partial, expect_sha256=expect_sha256, expect_bytes=expect_bytes, name=label)
    os.replace(partial, target)
    return target


def bar(width: int = 32) -> ProgressFn:
    """A ``\\r`` progress bar, for a CLI.

    Falls back to a plain byte count when the server sends no content-length, because a
    bar with no denominator is worse than a number.
    """
    import sys

    def render(progress: Progress) -> None:
        fraction = progress.fraction
        if fraction is None:
            sys.stdout.write(f"\r{progress.name}: {progress.downloaded / 1e6:.1f} MB")
        else:
            filled = int(width * fraction)
            sys.stdout.write(
                f"\r{progress.name}: [{'#' * filled}{'.' * (width - filled)}] "
                f"{progress.downloaded / 1e6:.1f}/{(progress.total or 0) / 1e6:.1f} MB"
            )
        if fraction == 1.0:
            sys.stdout.write("\n")
        sys.stdout.flush()

    return render
