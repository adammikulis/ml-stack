"""Bringing a file in: streamed to staging, checked, scanned, then promoted in one step.

Nothing reaches its final path until its size, SHA-256 (when one is pinned), format and scan
have passed. A file that fails is handed to the hold (sentinel's quarantine) and never
promoted. Nothing downloaded is executed or imported here.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from ml_stack import files, http, httpguard
from ml_stack.httpguard import Refused, TooLarge
from ml_stack.net import provenance, sniff
from ml_stack.net.hold import staging_dir
from ml_stack.net.pipeline import Pipeline, bearer, default
from ml_stack.net.policy import host_of
from ml_stack.net.scan import Outcome, scan_file
from ml_stack.safenames import safe_filename

__all__ = ["Blocked", "Cancel", "ChecksumMismatch", "Hooks", "NoDigest", "Progress", "Truncated", "Want", "download", "sweep"]

logger = logging.getLogger(__name__)

MOST_BYTES = 1 << 30
STALE_S = 7 * 86400.0
RANGE = re.compile(r"bytes (\d+)-(\d+)/(\d+|\*)")


class Blocked(Refused):
    """A file that failed a check and was held, not kept. ``held`` is where it is."""

    def __init__(self, message: str, held: str = "") -> None:
        super().__init__(message)
        self.held = held


class ChecksumMismatch(Blocked):
    """The file's SHA-256 is not the pinned one."""


class NoDigest(Refused):
    """A download that must be pinned to a digest was asked for without one."""


class Truncated(Refused):
    """The connection ended before the whole file came. The partial file is kept for a retry."""


@dataclass(frozen=True, slots=True)
class Want:
    """What a download must be: its ``kind`` (from the name when empty), the ``sha256`` it is
    pinned to, whether a digest is ``require_digest``, and how an unscanned file is treated.
    ``verify(path, headers)`` returns why a staged file is not acceptable, or ''. ``rename(final_url,
    headers)`` picks the file name once the answer is known."""

    kind: str = ""
    sha256: str = ""
    size: int = 0
    require_digest: bool = False
    allow_unscanned: bool = False
    max_bytes: int = MOST_BYTES
    deadline_s: float = 3600.0
    token: str = ""
    headers: dict[str, str] = field(default_factory=dict)
    purpose: str = "download"
    resume: bool = True
    admit: bool = True
    verify: Callable[[Path, dict[str, str]], str] | None = None
    rename: Callable[[str, dict[str, str]], str] | None = None


class Cancel:
    """Anything with a ``cancelled`` flag; a UI thread sets it to stop a download."""

    cancelled: bool = False


Progress = Callable[[int, int], None]


@dataclass(frozen=True, slots=True)
class Hooks:
    """A ``cancel`` flag and a ``progress(done, total)`` callback for a download."""

    cancel: Cancel | None = None
    progress: Progress | None = None


NO_HOOKS = Hooks()


def _key(url: str, dest: Path) -> str:
    return hashlib.sha256(f"{url}\n{dest}".encode()).hexdigest()[:20]


def _event(kind: str, severity: str, subject: str, evidence: dict[str, Any]) -> None:
    try:
        from ml_stack import sentinel
        from ml_stack.sentinel.events import Event, Severity

        bus = sentinel.default().bus
        bus.emit(Event(kind, Severity.parse(severity), "net", subject, evidence, time.time()))
    except (OSError, ImportError, ValueError) as exc:
        logger.debug("no sentinel event for %s: %s", kind, exc)


def _meta(part: Path) -> Path:
    return part.with_name(part.name + ".json")


def _resume_headers(part: Path, want: Want) -> tuple[dict[str, str], int]:
    offset = part.stat().st_size if want.resume and part.exists() else 0
    if not offset:
        return {}, 0
    held = files.read_json(_meta(part), {})
    tag = held.get("etag") or held.get("last-modified")
    if not tag:
        return {}, 0
    return {"Range": f"bytes={offset}-", "If-Range": tag}, offset


def _stream_to(pipe: Pipeline, url: str, part: Path, want: Want,
               hooks: Hooks) -> httpguard.Streaming:
    """Append the body to ``part``; the shown answer (without its chunks) is returned."""
    sent, offset = _resume_headers(part, want)
    sent = bearer(url, {**want.headers, **sent}, want.token, pipe.plain())
    limits = pipe.limited(want.purpose, admit=want.admit,
                          max_bytes=max(1, want.max_bytes - offset), deadline_s=want.deadline_s)
    with httpguard.stream(url, headers=sent, limits=limits) as shown:
        if shown.status == 416:
            part.unlink(missing_ok=True)
            raise Truncated("the partial file did not fit the server's; it was discarded, retry")
        if shown.status not in (200, 206):
            raise Refused(f"{http.shown(shown.url)} answered {shown.status}")
        resumed = shown.status == 206
        total = _total(shown, offset if resumed else 0)
        if resumed and not _starts_at(shown, offset):
            part.unlink(missing_ok=True)
            raise Truncated("the server resumed at another offset; the partial file was discarded")
        if total > want.max_bytes:
            raise TooLarge(f"the file is {total} bytes, over {want.max_bytes}")
        got = offset if resumed else 0
        part.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        _meta(part).write_text(json.dumps(
            {k: shown.headers[k] for k in ("etag", "last-modified") if k in shown.headers}),
            encoding="utf-8")
        with part.open("ab" if resumed else "wb") as out:
            part.chmod(0o600)
            for block in shown.chunks:
                if hooks.cancel is not None and hooks.cancel.cancelled:
                    raise Truncated("cancelled; the partial file is kept")
                out.write(block)
                got += len(block)
                if hooks.progress:
                    hooks.progress(got, total)
        if total and got != total:
            raise Truncated(f"{got} of {total} bytes arrived; the partial file is kept")
        return shown


def _total(shown: httpguard.Streaming, offset: int) -> int:
    ranged = RANGE.match(shown.headers.get("content-range", ""))
    if ranged and ranged.group(3).isdigit():
        return int(ranged.group(3))
    length = shown.headers.get("content-length", "")
    return offset + int(length) if length.isdigit() else 0


def _starts_at(shown: httpguard.Streaming, offset: int) -> bool:
    ranged = RANGE.match(shown.headers.get("content-range", ""))
    return bool(ranged and int(ranged.group(1)) == offset)


def _promote(staged: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        files.promote(staged, dest)
    except files.CrossDevice:
        with files.writing(dest) as copy:
            shutil.copyfile(staged, copy)
        staged.unlink(missing_ok=True)


def _reject(pipe: Pipeline, staged: Path, note: provenance.Provenance, reason: str,
            error: type[Blocked] = Blocked) -> Blocked:
    held = pipe.hold.hold(staged, reason, {"url": note.url, "sha256": note.sha256,
                                            "kind": note.kind, "reason": reason})
    provenance.record(replace(note, outcome="held", reason=reason, path=held))
    _event("net.download.held", "warning", f"artifact:{note.sha256 or note.url}",
           {"url": note.url, "reason": reason, "held": held})
    return error(f"{note.url}: {reason}", held)


def download(url: str, dest: Path | str, want: Want | None = None, pipeline: Pipeline | None = None,
             hooks: Hooks = NO_HOOKS) -> provenance.Provenance:
    """``url`` saved at ``dest`` after every check; its `Provenance` is written beside it.

    `NoDigest` before any request when ``want.require_digest`` and no ``sha256`` was given;
    `Truncated` when the connection ended early (the partial file stays in staging and the
    next call resumes it); `ChecksumMismatch` or `Blocked` when the file was held.
    """
    want = want or Want()
    pipe = pipeline or default()
    final = Path(dest)
    name = safe_filename(final.name)
    if want.require_digest and not want.sha256:
        raise NoDigest(f"{name}: this download is refused without a pinned SHA-256")
    host = pipe.policy.admit(url, want.purpose) if want.admit else host_of(url)
    key = _key(url, final)
    part = staging_dir() / f"{key}.part"
    shown = _stream_to(pipe, url, part, want, hooks)
    staged = part.with_name(f"{key}__{name}")
    part.replace(staged)
    _meta(part).unlink(missing_ok=True)
    if want.rename is not None:
        name = safe_filename(want.rename(shown.url, shown.headers))
        final = final.with_name(name)
        renamed = staged.with_name(f"{key}__{name}")
        staged.replace(renamed)
        staged = renamed
    digest, size = files.sha256_file(staged), staged.stat().st_size
    served = shown.headers.get("content-type", "")
    note = provenance.Provenance(
        url=url, final_url=shown.url, path=str(final), sha256=digest, size=size,
        kind=want.kind or sniff.expected_kind(name), fetched_at=provenance.stamp(),
        status=shown.status, redirects=tuple(shown.redirects),
        headers=provenance.subset(shown.headers), expected_sha256=want.sha256,
        host_source=host)
    if want.sha256 and digest != want.sha256.lower():
        raise _reject(pipe, staged, note, "sha256 differs from the pinned one", ChecksumMismatch)
    if want.size and size != want.size:
        raise _reject(pipe, staged, note, f"{size} bytes, expected {want.size}")
    if want.verify is not None and (problem := want.verify(staged, shown.headers)):
        raise _reject(pipe, staged, note, problem)
    verdict = sniff.sniff(staged, want.kind or sniff.expected_kind(name), content_type=served)
    if not verdict.ok:
        raise _reject(pipe, staged, note, "; ".join(verdict.problems))
    summary = scan_file(staged, pipe.scanners)
    keep, why = pipe.scan_policy.decide(verdict.kind, summary, allow_unscanned=want.allow_unscanned)
    if not keep:
        raise _reject(pipe, staged, note, why)
    warnings = (*verdict.warnings, *(w for r in summary.results for w in r.warnings),
                *(r.detail for r in summary.results if r.outcome == Outcome.NO_SCANNER))
    _promote(staged, final)
    done = replace(note, kind=verdict.kind, scan=why,
                   scanners=tuple(r.scanner for r in summary.results),
                   warnings=tuple(dict.fromkeys(warnings)))
    provenance.record(done, beside=final)
    _event("net.download", "info", f"artifact:{digest}", {"url": url, "path": str(final),
                                                          "scan": why})
    return done


def sweep(older_than_s: float = STALE_S) -> int:
    """Delete partial files in staging that have not been written for ``older_than_s``."""
    gone, now = 0, time.time()
    for path in staging_dir().glob("*.part*"):
        try:
            if now - path.stat().st_mtime > older_than_s:
                path.unlink()
                gone += 1
        except OSError:
            continue
    return gone

