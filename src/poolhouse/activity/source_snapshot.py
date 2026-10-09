"""Bounded source inventory and raw Git tree digests."""
from __future__ import annotations

import hashlib
import os
import selectors
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from contextlib import ExitStack
from pathlib import Path

from poolhouse import home

if os.name == "posix":
    import grp
from poolhouse.platform import start_process, terminate_process_group

OUTPUT_LIMIT = 8 * 1024 * 1024
INDEX_LIMIT = 32 * 1024 * 1024
FILE_LIMIT = 16 * 1024 * 1024
TOTAL_LIMIT = 256 * 1024 * 1024
GIT_ROOTS = (Path("/opt/homebrew"), Path("/usr/local"))


def validate_storage(base: Path, protected: tuple[Path, ...]) -> Path:
    base = base.resolve(strict=True)
    identities = {(info.st_dev, info.st_ino) for root in protected if root.exists() for info in (root.stat(),)}
    if any(base.is_relative_to(root.resolve()) or root.resolve().is_relative_to(base) for root in protected):
        raise RuntimeError("source snapshot: storage overlaps protected roots")
    for parent in (base, *base.parents):
        info = parent.stat()
        sticky = info.st_uid == 0 and info.st_mode & stat.S_ISVTX
        if ((info.st_dev, info.st_ino) in identities or info.st_uid not in (0, os.getuid())
                or (info.st_mode & 0o022 and not sticky)):
            raise RuntimeError("source snapshot: storage ancestry is foreign, writable or protected")
    return base


def protected_roots(environment: dict[str, str]) -> tuple[Path, ...]:
    roots = {root.resolve() for root in home.account_roots()}
    roots.update(home.expand(environment[key]).resolve()
                 for key in ("POOLHOUSE_HOME", "POOLHOUSE_CACHE") if environment.get(key))
    return tuple(sorted(roots))


def private_namespace(environment: dict[str, str], prefix: str) -> Path:
    base = validate_storage(Path(tempfile.gettempdir()), protected_roots(environment))
    directory = Path(tempfile.mkdtemp(prefix=prefix, dir=base)).resolve()
    directory.chmod(0o700)
    return directory


def git_environment() -> dict[str, str]:
    return {"PATH": os.defpath, "LC_ALL": "C", "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_SYSTEM": os.devnull,
            "GIT_TERMINAL_PROMPT": "0", "GIT_ALLOW_PROTOCOL": ""}


def _git_package(alias: Path, root: Path) -> Path:
    package = root / "opt/git"
    before = package.lstat()
    if not stat.S_ISLNK(before.st_mode) or before.st_uid not in (0, os.getuid()):
        raise RuntimeError("source snapshot: Git package alias is not owned")
    target = package.readlink()
    if len(str(target)) > 4096:
        raise RuntimeError("source snapshot: Git package alias exceeds bound")
    if target.parent not in (Path("../Cellar/git"), root / "Cellar/git"):
        raise RuntimeError("source snapshot: Git package alias has uncontrolled redirects")
    direct = root / "Cellar/git" / target.name
    if direct.resolve(strict=True) != direct:
        raise RuntimeError("source snapshot: Git package alias has uncontrolled redirects")
    resolved = direct / "bin/git"
    if resolved.resolve(strict=True) != resolved:
        raise RuntimeError("source snapshot: Git image has uncontrolled redirects")
    after = package.lstat()
    identities = {(info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_uid, info.st_mode)
                  for info in (before, after)}
    if len(identities) != 1 or package.readlink() != target or alias.resolve(strict=True) != resolved:
        raise RuntimeError("source snapshot: Git package alias changed during verification")
    return resolved


def _git_image(alias: Path, root: Path) -> Path:
    resolved = _git_package(alias, root)
    admin = grp.getgrnam("admin").gr_gid
    for parent in {*resolved.parents, *alias.parents}:
        info = parent.stat()
        entry = parent.lstat()
        if not stat.S_ISDIR(info.st_mode) or (stat.S_ISLNK(entry.st_mode) and parent != root / "opt/git"):
            raise RuntimeError("source snapshot: Git ancestry is not a direct directory")
        index = parent in (root / "opt", root / "Cellar") and info.st_gid == admin
        sticky = parent == Path("/private/tmp") and info.st_uid == 0 and info.st_mode & stat.S_ISVTX
        if (info.st_uid not in (0, os.getuid()) or (info.st_mode & 0o002 and not sticky)
                or (info.st_mode & 0o020 and not index and not sticky)):
            raise RuntimeError("source snapshot: Git ancestry is foreign or writable")
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
    descriptor = os.open(resolved, flags)
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        if (not stat.S_ISREG(before.st_mode) or before.st_uid not in (0, os.getuid())
                or before.st_mode & 0o022 or not before.st_mode & 0o111 or before.st_nlink != 1):
            raise RuntimeError("source snapshot: Git image is not owned and protected")
        if stream.read(4) not in (b"\xcf\xfa\xed\xfe", b"\xce\xfa\xed\xfe", b"\xfe\xed\xfa\xcf",
                                  b"\xfe\xed\xfa\xce", b"\xca\xfe\xba\xbe", b"\xbe\xba\xfe\xca",
                                  b"\xca\xfe\xba\xbf", b"\xbf\xba\xfe\xca"):
            raise RuntimeError("source snapshot: Git image is not Mach-O")
        identities = {(info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns,
                       info.st_uid, info.st_mode, info.st_nlink)
                      for info in (before, os.fstat(stream.fileno()), resolved.stat())}
        if len(identities) != 1 or alias.resolve(strict=True) != resolved:
            raise RuntimeError("source snapshot: Git image changed during verification")
    return resolved


def actual_git() -> Path:
    """Resolve fixed owned Git assets; macOS never executes the Apple selector shim."""
    if sys.platform != "darwin":
        program = shutil.which("git", path=os.defpath)
        if program is None:
            raise RuntimeError("source snapshot: Git unavailable")
        return Path(program).resolve(strict=True)
    for root in GIT_ROOTS:
        alias = root / "opt" / "git" / "bin" / "git"
        if alias.is_symlink() or alias.exists():
            return _git_image(alias, root)
    raise RuntimeError("source snapshot: owned Git unavailable; Apple shim refused")


def git_output(args: list[str], *, cwd: Path, seconds: float = 15) -> bytes:
    program = actual_git()
    if seconds <= 0:
        raise RuntimeError("source snapshot: Git or deadline unavailable")
    command = [str(program), "-c", "core.fsmonitor=false", "-c", "core.hooksPath=" + os.devnull, *args]
    process = start_process(command, cwd=cwd, env=git_environment(), close_fds=True,
                            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    deadline = time.monotonic() + seconds
    value = bytearray()
    try:
        with selectors.DefaultSelector() as ready:
            ready.register(process.stdout, selectors.EVENT_READ)
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not ready.select(remaining):
                    raise TimeoutError("source snapshot: Git inventory deadline exceeded")
                block = os.read(process.stdout.fileno(), 65536)
                if not block:
                    break
                value.extend(block)
                if len(value) > OUTPUT_LIMIT:
                    raise RuntimeError("source snapshot: Git inventory output exceeds bound")
        if process.wait(timeout=max(.001, deadline - time.monotonic())):
            raise RuntimeError("source snapshot: Git inventory failed")
        return bytes(value)
    finally:
        process.stdout.close()
        if process.poll() is None:
            terminate_process_group(process, force=True)
            process.wait(timeout=5)


def metadata_directory(root: Path) -> Path:
    marker = root / ".git"
    info = marker.lstat()
    if info.st_uid != os.getuid() or info.st_mode & 0o022:
        raise RuntimeError("source snapshot: repository metadata is foreign or writable")
    if stat.S_ISDIR(info.st_mode):
        return marker
    if not stat.S_ISREG(info.st_mode) or info.st_size > 4096 or info.st_nlink != 1:
        raise RuntimeError("source snapshot: invalid worktree pointer")
    fd = os.open(marker, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as stream:
        pinned = os.fstat(stream.fileno())
        if (pinned.st_dev, pinned.st_ino, pinned.st_size, pinned.st_mtime_ns) != (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns):
            raise RuntimeError("source snapshot: worktree pointer changed")
        text = stream.read(4097).decode().strip()
    if not text.startswith("gitdir: ") or "\n" in text:
        raise RuntimeError("source snapshot: malformed worktree pointer")
    directory = (root / text[8:]).resolve(strict=True)
    validate_storage(directory, protected_roots(dict(os.environ)))
    info = directory.stat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o022:
        raise RuntimeError("source snapshot: Git index directory is foreign or writable")
    return directory


class SourceSnapshot:
    """Inventory a worktree using private Git metadata without configured filters."""

    def __init__(self, root: Path, storage: Path):
        self.root = validate_storage(root, protected_roots(dict(os.environ)))
        storage = validate_storage(storage, ())
        owner = storage.stat()
        if owner.st_uid != os.getuid() or owner.st_mode & 0o077:
            raise RuntimeError("source snapshot: private owned storage required")
        self.deadline = time.monotonic() + 30
        self.total = 0
        self.directory = Path(tempfile.mkdtemp(prefix="source-git-", dir=storage))
        with ExitStack() as cleanup:
            cleanup.callback(self.close)
            self.prepare()
            cleanup.pop_all()

    def prepare(self) -> None:
        for name in ("objects", "refs"):
            (self.directory / name).mkdir(mode=0o700)
        (self.directory / "HEAD").write_text("ref: refs/heads/source\n")
        (self.directory / "config").write_text("[core]\n repositoryformatversion = 0\n bare = false\n")
        index = metadata_directory(self.root) / "index"
        if index.exists():
            fd = os.open(index, os.O_RDONLY | os.O_NOFOLLOW)
            with os.fdopen(fd, "rb") as stream:
                info = os.fstat(stream.fileno())
                if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                        or info.st_mode & 0o022 or info.st_nlink != 1 or info.st_size > INDEX_LIMIT):
                    raise RuntimeError("source snapshot: invalid bounded owned index")
                value = stream.read(INDEX_LIMIT + 1)
                final = os.fstat(stream.fileno())
                if (final.st_size, final.st_mtime_ns) != (info.st_size, info.st_mtime_ns):
                    raise RuntimeError("source snapshot: index changed during read")
            if (len(value) > INDEX_LIMIT or len(value) < 12 or value[:4] != b"DIRC"
                    or int.from_bytes(value[4:8], "big") not in (2, 3, 4)):
                raise RuntimeError("source snapshot: malformed or oversized index")
            (self.directory / "index").write_bytes(value)
        self.entries = {}
        for row in self.command(["ls-files", "--stage", "-z"]).split(b"\0"):
            if not row:
                continue
            header, name = row.split(b"\t", 1)
            mode, digest, stage = header.split()
            if stage != b"0" or mode not in (b"100644", b"100755", b"120000", b"160000") or len(digest) != 40:
                raise RuntimeError("source snapshot: ambiguous staged source")
            if name in self.entries:
                raise RuntimeError("source snapshot: duplicate staged source")
            self.path(name)
            self.entries[name] = mode, bytes.fromhex(digest.decode())
        self.tracked = tuple(self.entries)
        for name in self.command(["ls-files", "--others", "--exclude-standard", "-z"]).split(b"\0"):
            if name:
                self.path(name)
                self.entries.setdefault(name, (b"100644", b""))

    def command(self, args: list[str]) -> bytes:
        return git_output(["--git-dir=" + str(self.directory), "--work-tree=" + str(self.root), *args],
                          cwd=self.root, seconds=self.deadline - time.monotonic())

    def path(self, name: bytes) -> Path:
        parts = name.split(b"/")
        if not name or any(part in (b"", b".", b"..") for part in parts):
            raise RuntimeError("source snapshot: invalid source pathname")
        path = self.root / os.fsdecode(name)
        if path.parent.resolve() != path.parent:
            raise RuntimeError("source snapshot: redirected source parent")
        parent = path.parent
        while not parent.exists():
            parent = parent.parent
        validate_storage(parent, ())
        return path

    def blob(self, path: Path, info) -> bytes:
        if info.st_size > FILE_LIMIT or self.total + info.st_size > TOTAL_LIMIT:
            raise RuntimeError("source snapshot: source bytes exceed bounds")
        self.total += info.st_size
        digest = hashlib.sha1(f"blob {info.st_size}\0".encode(), usedforsecurity=False)
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(fd, "rb") as stream:
            initial = os.fstat(stream.fileno())
            if (initial.st_dev, initial.st_ino, initial.st_size) != (info.st_dev, info.st_ino, info.st_size):
                raise RuntimeError("source snapshot: source changed before read")
            read = 0
            for block in iter(lambda: stream.read(65536), b""):
                read += len(block)
                if read > info.st_size:
                    raise RuntimeError("source snapshot: source grew during read")
                if time.monotonic() >= self.deadline:
                    raise TimeoutError("source snapshot: source read deadline exceeded")
                digest.update(block)
            final = os.fstat(stream.fileno())
            if (read != info.st_size or (final.st_size, final.st_mtime_ns, final.st_mode, final.st_nlink)
                    != (initial.st_size, initial.st_mtime_ns, initial.st_mode, initial.st_nlink)
                    or (path.lstat().st_dev, path.lstat().st_ino) != (initial.st_dev, initial.st_ino)):
                raise RuntimeError("source snapshot: source changed during read")
        return digest.digest()

    def tree_hash(self) -> str:
        tree = {}
        for name, (mode, digest) in self.entries.items():
            if time.monotonic() >= self.deadline:
                raise TimeoutError("source snapshot: tree deadline exceeded")
            path = self.path(name)
            try:
                info = path.lstat()
            except FileNotFoundError:
                continue
            if info.st_uid != os.getuid() or (info.st_nlink != 1 and not stat.S_ISDIR(info.st_mode)):
                raise RuntimeError("source snapshot: foreign or hardlinked source")
            if stat.S_ISLNK(info.st_mode):
                value = os.fsencode(path.readlink())
                digest = hashlib.sha1(f"blob {len(value)}\0".encode() + value, usedforsecurity=False).digest()
                mode = b"120000"
            elif stat.S_ISREG(info.st_mode):
                if info.st_mode & 0o022:
                    raise RuntimeError("source snapshot: writable source file")
                digest = self.blob(path, info)
                mode = b"100755" if info.st_mode & 0o111 else b"100644"
            elif mode != b"160000" or not stat.S_ISDIR(info.st_mode):
                raise RuntimeError("source snapshot: unsupported source node")
            cursor = tree
            parts = name.split(b"/")
            for part in parts[:-1]:
                cursor = cursor.setdefault(part, {})
            cursor[parts[-1]] = mode, digest
        return tree_digest(tree).hex()

    def close(self) -> None:
        shutil.rmtree(self.directory)


def tree_digest(tree: dict) -> bytes:
    rows = []
    for name, value in sorted(tree.items(), key=lambda item: item[0] + (b"/" if isinstance(item[1], dict) else b"")):
        mode, digest = (b"40000", tree_digest(value)) if isinstance(value, dict) else value
        rows.append(mode + b" " + name + b"\0" + digest)
    value = b"".join(rows)
    return hashlib.sha1(f"tree {len(value)}\0".encode() + value, usedforsecurity=False).digest()
