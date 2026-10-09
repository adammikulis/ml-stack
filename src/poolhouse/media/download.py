"""Fetch a model asset to disk, once, safely."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from poolhouse import net
from poolhouse.files import sha256_file
from poolhouse.http import ServerError
from poolhouse.httpguard import Refused

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


def _verify(path: Path, *, expect_sha256: str | None, expect_bytes: int | None, name: str) -> None:
    if expect_bytes is not None:
        actual = path.stat().st_size
        if actual != expect_bytes:
            path.unlink(missing_ok=True)
            raise DownloadError(
                f"{name}: expected {expect_bytes} bytes, got {actual}. Removed the partial file."
            )
    if expect_sha256:
        actual = sha256_file(path)
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
    on_progress: ProgressFn | None = None,
    expect_sha256: str | None = None,
    expect_bytes: int | None = None,
) -> Path:
    """Download ``url`` to ``target`` through the net pipeline, returning the path. Idempotent."""
    target = Path(target).expanduser()
    label = target.name

    if target.exists() and target.stat().st_size > 0:
        if expect_sha256 or expect_bytes:
            _verify(target, expect_sha256=expect_sha256, expect_bytes=expect_bytes, name=label)
        return target

    last = {"at": 0.0}

    def progress(done: int, total: int) -> None:
        now = time.monotonic()
        if on_progress is not None and (now - last["at"] >= _PROGRESS_INTERVAL_S or done == total):
            last["at"] = now
            on_progress(Progress(label, done, total or None))

    want = net.Want(sha256=expect_sha256 or "", size=expect_bytes or 0, purpose="document download",
                    max_bytes=2 << 30)
    try:
        net.download(url, target, want, hooks=net.Hooks(progress=progress))
    except net.ChecksumMismatch as exc:
        raise DownloadError(f"{label}: sha256 mismatch ({exc}). The file was set aside.") from exc
    except net.Blocked as exc:
        raise DownloadError(f"{label}: {exc}") from exc
    except net.Truncated as exc:
        raise DownloadError(f"{label}: cannot fetch {url} ({exc})") from exc
    except ServerError as exc:
        raise DownloadError(f"{label}: HTTP {exc.status} fetching {url}") from exc
    except (OSError, Refused) as exc:
        raise DownloadError(f"{label}: cannot fetch {url} ({exc})") from exc
    return target


def bar(width: int = 32) -> ProgressFn:
    """A ``\r`` progress bar, for a CLI."""
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
