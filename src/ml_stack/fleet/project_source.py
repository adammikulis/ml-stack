"""Tracked source snapshots and verified isolated project checkouts."""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import stat
import tempfile
import unicodedata
import zipfile
from pathlib import Path, PurePosixPath

from ml_stack.files import promote, write_json
from ml_stack.home import DEFAULT_NAME
from ml_stack.net import git
from ml_stack.redact.secrets import PATTERNS

MAX_FILE = 8 << 20
MAX_SOURCE = 128 << 20
MAX_FILES = 10_000
MAX_ARCHIVE = MAX_SOURCE + (8 << 20)
EXCLUDED = {".git", ".worktrees", ".aws", ".codex", ".claude", ".agents", ".ssh",
            DEFAULT_NAME, ".venv", "node_modules", "__pycache__"}
SECRET_SUFFIXES = {".pem", ".key", ".p12", ".pfx", ".keystore", ".kdbx"}
CONFIG_SUFFIXES = {".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf", ".txt"}
ID = re.compile(r"[a-f0-9]{32}\Z")
DIGEST = re.compile(r"[a-f0-9]{64}\Z")
RESERVED = {"con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)),
            *(f"lpt{i}" for i in range(1, 10))}


class ProjectError(ValueError):
    """A project source snapshot or checkout was refused."""


def project_id(value: str) -> str:
    """Return a valid project identifier."""
    if not ID.fullmatch(value):
        raise ProjectError("Invalid project identifier")
    return value


def safe_name(name: str) -> str:
    """Return a portable source path without traversal or device names."""
    parts = PurePosixPath(name).parts
    if (not parts or name.startswith("/") or "\\" in name or ":" in name
            or str(PurePosixPath(name)) != name or unicodedata.normalize("NFC", name) != name
            or any(p in {".", ".."} or p.rstrip(" .") != p or any(ord(c) < 32 for c in p)
                   or p.split(".")[0].casefold() in RESERVED for p in parts)):
        raise ProjectError("Unsafe source path")
    return name


def excluded(name: str) -> bool:
    """Whether a source path names credentials, state or a build cache."""
    parts = PurePosixPath(name).parts
    return (any(p.casefold() in EXCLUDED or p.casefold().startswith(DEFAULT_NAME) for p in parts)
            or any(p.casefold().startswith(".env") for p in parts)
            or PurePosixPath(name).suffix.casefold() in SECRET_SUFFIXES
            or parts[-1].casefold() in {"credentials", "credentials.json", "id_rsa", "id_ed25519"})


def sensitive(data: bytes, name: str) -> bool:
    """Whether a source file contains a recognized credential."""
    text = data.decode("utf-8", errors="replace")
    return any(pattern.search(text) for kind, pattern in PATTERNS
               if kind != "assigned-secret" or PurePosixPath(name).suffix.casefold() in CONFIG_SUFFIXES)


def _hash(entries: list[dict]) -> str:
    return hashlib.sha256(json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def build(root: Path, identifier: str) -> tuple[dict, bytes]:
    """Build a manifest and archive from tracked, unignored regular source files."""
    project_id(identifier)
    root = root.resolve()
    ignored = set(git.run(["ls-files", "--cached", "--ignored", "--exclude-standard", "-z"],
                          cwd=root).stdout.split("\0"))
    listed = git.run(["ls-files", "--stage", "-z"], cwd=root).stdout.split("\0")
    entries, contents = [], {}
    selected = []
    total, skipped = 0, 0
    for row in listed:
        if not row:
            continue
        mode, _digest, stage, name = row.replace("\t", " ", 1).split(" ", 3)
        if stage != "0" or mode not in {"100644", "100755"}:
            skipped += 1
            continue
        safe_name(name)
        if name in ignored or excluded(name) or not (root / name).exists():
            skipped += 1
            continue
        selected.append((mode, _digest, name))
    blobs = git.blobs([digest for _, digest, _ in selected], cwd=root, limit=MAX_FILE, total=MAX_SOURCE)
    for mode, digest, name in selected:
        data = blobs[digest]
        if sensitive(data, name):
            skipped += 1
            continue
        total += len(data)
        if total > MAX_SOURCE or len(entries) >= MAX_FILES:
            raise ProjectError("Project source exceeds the snapshot limit")
        entries.append({"path": name, "size": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                        "executable": mode == "100755"})
        contents[name] = data
    entries.sort(key=lambda entry: entry["path"])
    manifest = {"kind": "tracked-source", "project_id": identifier, "files": entries,
                "revision": git.run(["rev-parse", "HEAD"], cwd=root).stdout.strip(),
                "source_hash": _hash(entries), "size_bytes": total, "excluded_files": skipped}
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", json.dumps(manifest, sort_keys=True))
        for entry in entries:
            archive.writestr("files/" + entry["path"], contents[entry["path"]])
    packed = output.getvalue()
    if len(packed) > MAX_ARCHIVE:
        raise ProjectError("Project archive exceeds the snapshot limit")
    return manifest, packed


def verify(packed: bytes, identifier: str, source_hash: str) -> tuple[dict, dict[str, bytes]]:
    """Verify every archive entry and digest before a checkout is created."""
    project_id(identifier)
    if not DIGEST.fullmatch(source_hash) or len(packed) > MAX_ARCHIVE:
        raise ProjectError("Invalid source snapshot size or digest")
    try:
        with zipfile.ZipFile(io.BytesIO(packed)) as archive:
            infos = archive.infolist()
            if len(infos) > MAX_FILES + 1 or len({info.filename for info in infos}) != len(infos):
                raise ProjectError("Duplicate entries or too many source files")
            if any(info.file_size > MAX_FILE or stat.S_IFMT(info.external_attr >> 16) not in {0, stat.S_IFREG}
                   or info.flag_bits & 1 or info.is_dir()
                   for info in infos):
                raise ProjectError("Source archive contains an oversized file or link")
            manifest = json.loads(archive.read("manifest.json"))
            entries = _entries(manifest, identifier, source_hash)
            wanted = {"manifest.json", *("files/" + entry["path"] for entry in entries)}
            if {info.filename for info in infos} != wanted:
                raise ProjectError("Source archive contains unlisted files")
            contents = {}
            for entry in entries:
                data = archive.read("files/" + entry["path"])
                if len(data) != entry["size"] or hashlib.sha256(data).hexdigest() != entry["sha256"]:
                    raise ProjectError("Source file digest mismatch")
                if sensitive(data, entry["path"]):
                    raise ProjectError("Source snapshot contains a credential")
                contents[entry["path"]] = data
            return manifest, contents
    except (zipfile.BadZipFile, KeyError, TypeError, UnicodeError, RuntimeError, NotImplementedError,
            json.JSONDecodeError) as exc:
        raise ProjectError("Malformed source snapshot") from exc


def _entries(manifest: dict, identifier: str, source_hash: str) -> list[dict]:
    if (not isinstance(manifest, dict) or manifest.get("kind") != "tracked-source"
            or manifest.get("project_id") != identifier or manifest.get("source_hash") != source_hash):
        raise ProjectError("Source snapshot belongs to another project or revision")
    entries = manifest.get("files")
    if not isinstance(entries, list) or len(entries) > MAX_FILES or _hash(entries) != source_hash:
        raise ProjectError("Invalid source manifest")
    total, seen = 0, set()
    for entry in entries:
        name = safe_name(entry["path"])
        size = entry["size"]
        if (excluded(name) or name.casefold() in seen or not isinstance(size, int) or isinstance(size, bool)
                or not 0 <= size <= MAX_FILE or not DIGEST.fullmatch(entry["sha256"])
                or not isinstance(entry["executable"], bool)):
            raise ProjectError("Source manifest contains an unsafe file")
        folded = name.casefold()
        if any(folded.startswith(old + "/") or old.startswith(folded + "/") for old in seen):
            raise ProjectError("Source paths collide")
        seen.add(folded)
        total += size
    if total > MAX_SOURCE or manifest.get("size_bytes") != total:
        raise ProjectError("Source manifest exceeds its size limit")
    return entries


def checkout(packed: bytes, identifier: str, source_hash: str, base: Path, *, authority: dict | None = None) -> Path:
    """Create a separate Git checkout from a fully verified source snapshot."""
    manifest, contents = verify(packed, identifier, source_hash)
    base = base.resolve()
    base.mkdir(parents=True, exist_ok=True)
    target = base / (identifier + "-" + os.urandom(8).hex())
    with tempfile.TemporaryDirectory(prefix="project-stage-", dir=base) as temporary:
        stage = Path(temporary) / "checkout"
        stage.mkdir()
        for entry in manifest["files"]:
            path = stage / entry["path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(contents[entry["path"]])
            path.chmod(0o755 if entry["executable"] else 0o644)
        git.run(["init", "--initial-branch=agent"], cwd=stage)
        git.source_index(manifest["files"], cwd=stage)
        git.run(["-c", "user.name=ml-stack", "-c", "user.email=ml-stack@example.invalid",
                 "-c", "commit.gpgsign=false", "commit", "--allow-empty", "-m",
                 "chore: initialize shared project snapshot"], cwd=stage)
        origin = authority or {}
        metadata = {"kind": "project-checkout", "project_id": identifier, "source_hash": source_hash,
                    "authority": {"machine": origin.get("machine", ""), "host": origin.get("host", "")},
                    "source": {"machine": origin.get("source_machine", ""), "host": origin.get("source_host", "")}}
        write_json(stage / ".ml-stack-project.json", metadata)
        promote(stage, target)
    return target
