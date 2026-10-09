"""What a downloaded file is, read from its first bytes and its structure, never from its name
alone. Nothing here loads tensors, extracts an archive or runs a parser on untrusted code."""

from __future__ import annotations

import json
import struct
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from ml_stack.safenames import Unsafe, safe_join
from ml_stack.tar_libraries import members as library_members

__all__ = ["EXTENSIONS", "Verdict", "audit_archive", "detect", "expected_kind", "sniff"]

HEAD = 4096
MOST_HEADER = 100 * 1024 * 1024
MOST_TENSORS = 200_000
MOST_ARCHIVE_ENTRIES = 20_000
MOST_ARCHIVE_BYTES = 8 << 30
MOST_RATIO = 1000
GGUF_VERSIONS = (1, 2, 3)
DTYPES = frozenset({"F64", "F32", "F16", "BF16", "I64", "I32", "I16", "I8", "U64", "U32", "U16",
                    "U8", "BOOL", "F8_E4M3", "F8_E5M2", "F8_E8M0", "F4", "C64", "C128"})
ARCHIVE_SUFFIXES = (".zip", ".tar", ".gz", ".tgz", ".bz2", ".xz", ".7z", ".rar", ".jar", ".whl")
PICKLE_SUFFIXES = (".pt", ".pth", ".bin", ".ckpt", ".pkl", ".pickle", ".npy", ".npz", ".joblib")

EXTENSIONS: dict[str, str] = {
    ".gguf": "gguf", ".safetensors": "safetensors", ".pdf": "pdf", ".zip": "zip",
    ".tar": "tar", ".tgz": "tar", ".json": "json", ".png": "png", ".jpg": "jpeg",
    ".jpeg": "jpeg", ".step": "step", ".stp": "step", ".txt": "text", ".md": "text",
    ".kicad_mod": "kicad", ".kicad_sym": "kicad", ".exe": "executable", ".dmg": "executable",
    ".pkg": "executable", ".msi": "executable", ".sh": "executable", ".ps1": "executable",
}
"""File-name suffix to the kind of file it claims to be."""

NATIVE_MAGIC = (b"MZ", b"\x7fELF", b"\xfe\xed\xfa\xce", b"\xfe\xed\xfa\xcf", b"\xce\xfa\xed\xfe",
                b"\xcf\xfa\xed\xfe", b"\xca\xfe\xba\xbe", b"#!")


@dataclass(frozen=True, slots=True)
class Verdict:
    """The outcome of looking at one file: whether it is what it claims, and why not."""

    ok: bool
    kind: str
    problems: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


def expected_kind(name: str) -> str:
    """The kind a file name claims; '' when the suffix says nothing."""
    low = name.lower()
    if low.endswith(".tar.gz"):
        return "tar"
    return EXTENSIONS.get(Path(low).suffix, "")


def detect(first: bytes) -> str:
    """The kind the first bytes show: gguf, pdf, zip, gzip, tar, png, jpeg, step, executable,
    json, text or ''."""
    if first.startswith(b"GGUF"):
        return "gguf"
    if first.lstrip()[:5] == b"%PDF-" or b"%PDF-" in first[:1024]:
        return "pdf"
    if first.startswith((b"PK\x03\x04", b"PK\x05\x06")):
        return "zip"
    if first.startswith(b"\x1f\x8b"):
        return "gzip"
    if len(first) > 262 and first[257:262] == b"ustar":
        return "tar"
    if first.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if first.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if first.lstrip().startswith(b"ISO-10303"):
        return "step"
    if first.startswith(NATIVE_MAGIC):
        return "executable"
    stripped = first.lstrip()[:1]
    if stripped in (b"{", b"["):
        return "json"
    try:
        first.decode("utf-8")
    except UnicodeDecodeError:
        return ""
    return "text" if first else ""


def _gguf(path: Path, size: int) -> list[str]:
    with path.open("rb") as handle:
        head = handle.read(24)
    if len(head) < 24:
        return ["a GGUF file is at least 24 bytes"]
    version, tensors, keys = struct.unpack("<IQQ", head[4:24])
    problems = []
    if version not in GGUF_VERSIONS:
        problems.append(f"GGUF version {version} is not one of {GGUF_VERSIONS}")
    if tensors > MOST_TENSORS or keys > MOST_TENSORS:
        problems.append(f"GGUF claims {tensors} tensors and {keys} keys")
    if size < 24 + tensors:
        problems.append("GGUF is shorter than its tensor count")
    return problems


def _safetensors(path: Path, size: int) -> list[str]:
    if size < 8:
        return ["a safetensors file starts with an 8 byte header length"]
    with path.open("rb") as handle:
        (length,) = struct.unpack("<Q", handle.read(8))
        if length == 0 or length > MOST_HEADER or 8 + length > size:
            return [f"the safetensors header length {length} does not fit a {size} byte file"]
        raw = handle.read(length)
    try:
        header = json.loads(raw)
    except ValueError:
        return ["the safetensors header is not JSON"]
    if not isinstance(header, dict):
        return ["the safetensors header is not an object"]
    data = size - 8 - length
    problems: list[str] = []
    spans: list[tuple[int, int]] = []
    tensors = {k: v for k, v in header.items() if k != "__metadata__"}
    if len(tensors) > MOST_TENSORS:
        return [f"the safetensors header names {len(tensors)} tensors"]
    for name, info in tensors.items():
        spans_ok = (isinstance(info, dict) and info.get("dtype") in DTYPES
                    and isinstance(info.get("shape"), list)
                    and isinstance(info.get("data_offsets"), list)
                    and len(info["data_offsets"]) == 2
                    and all(isinstance(n, int) and n >= 0 for n in info["data_offsets"]))
        if not spans_ok:
            problems.append(f"tensor {str(name)[:40]!r} has a malformed entry")
            continue
        begin, end = info["data_offsets"]
        if begin > end or end > data:
            problems.append(f"tensor {str(name)[:40]!r} points outside the data")
        spans.append((begin, end))
    spans.sort()
    if any(spans[i][1] > spans[i + 1][0] for i in range(len(spans) - 1)):
        problems.append("safetensors tensors overlap")
    return problems[:5]


def _pdf(path: Path, size: int) -> tuple[list[str], list[str]]:
    with path.open("rb") as handle:
        head = handle.read(1024)
        handle.seek(max(0, size - 2048))
        tail = handle.read()
    problems = [] if b"%PDF-" in head else ["no %PDF- header"]
    if b"%%EOF" not in tail:
        problems.append("no %%EOF trailer: the PDF is cut short")
    warnings = []
    with path.open("rb") as handle:
        body = handle.read(8 * 1024 * 1024)
    for token in (b"/JavaScript", b"/JS", b"/Launch", b"/OpenAction", b"/EmbeddedFile"):
        if token in body:
            warnings.append(f"the PDF carries {token.decode()}")
    return problems, warnings


def _archive_names(path: Path, library_links: bool = False) -> tuple[list[tuple[str, int, int, bool, bool]], int]:
    """``(name, size, compressed, is_link, is_dir)`` per entry and the archive's own size."""
    rows: list[tuple[str, int, int, bool, bool]] = []
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as zf:
            for info in zf.infolist():
                link = (info.external_attr >> 28) == 0xA
                rows.append((info.filename, info.file_size, info.compress_size, link,
                             info.is_dir()))
    else:
        with tarfile.open(path) as tf:
            if library_links:
                return [(item.name, source.size, source.size, False, item.isdir())
                        for item, source in library_members(tf)], path.stat().st_size
            for member in tf:
                rows.append((member.name, member.size, member.size,
                             member.issym() or member.islnk() or member.isdev(), member.isdir()))
                if len(rows) > MOST_ARCHIVE_ENTRIES:
                    break
    return rows, path.stat().st_size


def audit_archive(path: Path, *, max_entries: int = MOST_ARCHIVE_ENTRIES,
                  max_bytes: int = MOST_ARCHIVE_BYTES, max_ratio: int = MOST_RATIO,
                  library_links: bool = False) -> list[str]:
    """Why an archive must not be unpacked: too many entries, too large once unpacked, a
    compression ratio over ``max_ratio``, a link, an absolute or climbing path, or an archive
    inside it. Empty when it is fine. Nothing is extracted."""
    try:
        rows, own = _archive_names(path, library_links)
    except (zipfile.BadZipFile, tarfile.TarError, OSError, EOFError, Unsafe) as exc:
        return [f"the archive cannot be read: {type(exc).__name__}"]
    problems: list[str] = []
    if len(rows) > max_entries:
        problems.append(f"{len(rows)} entries (most {max_entries})")
    total = sum(size for _n, size, _c, _l, _d in rows)
    if total > max_bytes:
        problems.append(f"unpacks to {total} bytes (most {max_bytes})")
    if own and total / own > max_ratio:
        problems.append(f"compresses {total // own} to 1 (most {max_ratio})")
    seen = {"link": False, "path": False, "nested": False}
    for name, _size, _comp, link, _dir in rows:
        if link and not seen["link"]:
            problems.append(f"{name[:60]!r} is a link or a device")
            seen["link"] = True
        try:
            safe_join(Path("/unpack-root"), name.rstrip("/"))
        except Unsafe:
            if not seen["path"]:
                problems.append(f"{name[:60]!r} is not a plain relative path")
                seen["path"] = True
        if PurePosixPath(name.lower()).suffix in ARCHIVE_SUFFIXES and not seen["nested"]:
            problems.append(f"{name[:60]!r} is an archive inside the archive")
            seen["nested"] = True
    return problems


def sniff(path: Path, kind: str = "", *, content_type: str = "", library_links: bool = False) -> Verdict:
    """Whether ``path`` is the ``kind`` of file it should be (by default what its name claims).

    A ``.gguf`` has GGUF magic, a sane version and counts; a ``.safetensors`` has a header
    that parses and tensor spans inside the data; a PDF has its header and trailer; an
    archive passes `audit_archive`. A file that shows native-executable magic is never a
    data kind. Pickle-bearing suffixes are refused as weights.
    """
    name = path.name.lower()
    claimed = kind or expected_kind(name)
    size = path.stat().st_size
    with path.open("rb") as handle:
        first = handle.read(HEAD)
    shown = detect(first)
    problems: list[str] = []
    warnings: list[str] = []
    if name.endswith(PICKLE_SUFFIXES) and claimed in ("", "gguf", "safetensors", "model"):
        problems.append("pickle-based weights are not accepted: use safetensors or GGUF")
    if shown == "executable" and claimed != "executable":
        problems.append(f"the bytes are a native executable or script, not {claimed or 'data'}")
    if claimed == "gguf":
        problems += _gguf(path, size) if shown == "gguf" else ["no GGUF magic"]
    elif claimed == "safetensors":
        problems += _safetensors(path, size)
    elif claimed == "pdf":
        problems += ["no %PDF- header"] if shown != "pdf" else []
        if shown == "pdf":
            found, warnings = _pdf(path, size)
            problems += found
    elif claimed in ("zip", "tar", "archive"):
        if shown not in ("zip", "gzip", "tar"):
            problems.append(f"the bytes are {shown or 'unrecognised'}, not an archive")
        else:
            problems += audit_archive(path, library_links=library_links)
    elif claimed in ("png", "jpeg", "step") and shown != claimed:
        problems.append(f"the bytes are {shown or 'unrecognised'}, not {claimed}")
    elif claimed == "json":
        try:
            json.loads(path.read_bytes() if size <= 64 * 1024 * 1024 else b"")
        except ValueError:
            problems.append("not JSON")
    served = content_type.split(";")[0].strip().lower()
    if served in ("text/html", "application/xhtml+xml") and claimed not in ("", "text", "html"):
        problems.append(f"served as {served}, not a {claimed} file")
    return Verdict(not problems, claimed or shown, tuple(problems), tuple(warnings))
