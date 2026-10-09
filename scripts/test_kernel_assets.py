"""Bounded literal Mach-O loader dependencies for the test interpreter."""
from __future__ import annotations

import os
import selectors
import stat
import subprocess
import sys
import sysconfig
import time
from pathlib import Path

SYSTEM = (Path('/usr/lib'), Path('/System/Library'))


def inspect_tool(images: list[Path], seconds: float, option: str) -> str:
    if option not in ('-L', '-D'):
        raise RuntimeError('test confinement: unsupported loader inspection')
    deadline = time.monotonic() + seconds
    process = subprocess.Popen(['/usr/bin/otool', option, *(str(image) for image in images)],
                               stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL,
                               env={'PATH': '/usr/bin:/bin', 'LC_ALL': 'C'}, close_fds=True)
    chunks = []
    size = 0
    try:
        with selectors.DefaultSelector() as ready:
            ready.register(process.stdout, selectors.EVENT_READ)
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('test confinement: loader inspection deadline exceeded')
                if not ready.select(remaining):
                    raise TimeoutError('test confinement: loader inspection deadline exceeded')
                block = os.read(process.stdout.fileno(), 65536)
                if not block:
                    break
                size += len(block)
                if size > 2 * 1024 * 1024:
                    raise RuntimeError('test confinement: loader inspection output exceeds bound')
                chunks.append(block)
        if process.wait(timeout=max(.001, deadline - time.monotonic())):
            raise RuntimeError('test confinement: loader inspection failed')
        return b''.join(chunks).decode('utf-8', errors='strict')
    finally:
        process.stdout.close()
        if process.poll() is None:
            process.kill()
            process.wait()


def inspect_images(images: list[Path], seconds: float) -> str:
    return inspect_tool(images, seconds, '-L')


def install_names(output: str, images: list[Path]) -> dict[Path, str | None]:
    result = {}
    current = None
    for line in output.splitlines():
        if line.endswith(':') and Path(line[:-1]) in images:
            current = Path(line[:-1])
            if current in result:
                raise RuntimeError('test confinement: duplicate install-name header')
            result[current] = None
        elif (current is None or not line or len(line.encode()) > 4096
              or any(ord(character) < 32 for character in line) or result[current] is not None):
            raise RuntimeError('test confinement: malformed install-name inspection')
        else:
            result[current] = line
    if set(result) != set(images):
        raise RuntimeError('test confinement: incomplete install-name inspection')
    return result


def image_identity(path: Path) -> tuple:
    info = path.lstat()
    return (info.st_dev, info.st_ino, info.st_uid, info.st_mode, info.st_nlink,
            info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def metadata_aliases(path: Path) -> set[Path]:
    aliases = set()
    for part in (path, *path.parents):
        if part.is_symlink():
            aliases.add(part.parent.resolve() / part.name)
    return aliases


def trusted_host_index(path: Path, info) -> bool:
    import grp
    if path not in (Path('/opt/homebrew/Cellar'), Path('/opt/homebrew/opt')):
        return False
    try:
        admin = grp.getgrnam('admin').gr_gid
    except KeyError:
        return False
    return (stat.S_ISDIR(info.st_mode) and info.st_uid in (0, os.getuid())
            and info.st_gid == admin and not info.st_mode & 0o002)


def owned_ancestors(path: Path) -> None:
    for parent in (path, *path.parents):
        info = parent.stat()
        temporary_boundary = parent == Path('/private/tmp') and info.st_uid == 0 and info.st_mode & stat.S_ISVTX
        if info.st_uid not in (0, os.getuid()) or (info.st_mode & 0o022 and not temporary_boundary and not trusted_host_index(parent, info)):
            raise RuntimeError('test confinement: loader ancestry is foreign or writable')


def owned_image(path: Path, protected: tuple[Path, ...], pinned: dict | None = None) -> tuple[Path, set[Path], tuple[int, int, int, int]]:
    resolved = path.resolve()
    eligible = (Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve(),
                Path('/opt/homebrew/Cellar'), Path('/usr/local/Cellar'))
    if not any(resolved.is_relative_to(root) for root in eligible) and resolved not in (pinned or {}):
        raise RuntimeError('test confinement: image is outside interpreter package roots')
    if any(resolved.is_relative_to(root) or root.is_relative_to(resolved) for root in protected):
        raise RuntimeError('test confinement: loader image overlaps protected state')
    info = resolved.stat()
    if resolved in (pinned or {}):
        actual = (info.st_dev, info.st_ino, info.st_uid, info.st_mode, info.st_nlink,
                  info.st_size, info.st_mtime_ns, info.st_ctime_ns)
        if actual != pinned[resolved] or info.st_nlink != 1 or info.st_mode & 0o222:
            raise RuntimeError('test confinement: pinned copied image changed')
    if not stat.S_ISREG(info.st_mode) or info.st_uid not in (0, os.getuid()) or info.st_mode & 0o022:
        raise RuntimeError('test confinement: loader image is not an owned protected file')
    owned_ancestors(resolved.parent)
    aliases = metadata_aliases(path)
    for alias in aliases:
        owned_ancestors(alias.parent)
    descriptor = os.open(resolved, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, 'rb') as stream:
        opened = os.fstat(stream.fileno())
        if (opened.st_dev, opened.st_ino) != (info.st_dev, info.st_ino):
            raise RuntimeError('test confinement: loader image changed during inspection')
        if stream.read(4) not in (b'\xcf\xfa\xed\xfe', b'\xce\xfa\xed\xfe', b'\xfe\xed\xfa\xcf',
                                  b'\xfe\xed\xfa\xce', b'\xca\xfe\xba\xbe', b'\xbe\xba\xfe\xca',
                                  b'\xca\xfe\xba\xbf', b'\xbf\xba\xfe\xca'):
            raise RuntimeError('test confinement: dependency is not a Mach-O image')
    return resolved, aliases, (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)


def image_dependencies(batch, executable: Path, protected, pinned, deadline: float) -> list[Path]:
    dependencies = []
    snapshots = {path: image_identity(path) for path in batch}
    output = inspect_images(batch, deadline - time.monotonic())
    names = install_names(inspect_tool(batch, deadline - time.monotonic(), '-D'), batch)
    for path in batch:
        owned_image(path, protected, pinned)
        if image_identity(path) != snapshots[path]:
            raise RuntimeError('test confinement: image changed during loader inspection')
    current = None
    headers = set()
    first = False
    for line in output.splitlines():
        if not line.startswith('\t'):
            if not line or not line.endswith(':') or Path(line[:-1]) not in batch:
                raise RuntimeError('test confinement: unexpected loader inspection header')
            current = Path(line[:-1])
            if current in headers:
                raise RuntimeError('test confinement: duplicate loader header')
            headers.add(current)
            first = True
            continue
        if current is None or ' (compatibility version ' not in line or not line.endswith(')'):
            raise RuntimeError('test confinement: malformed loader dependency')
        dependency = line.strip().split(' (compatibility version ', 1)[0]
        if first:
            first = False
            if names[current] is not None:
                if dependency != names[current]:
                    raise RuntimeError('test confinement: misplaced image install name')
                continue
        if dependency.startswith('/'):
            dependencies.append(Path(dependency))
        elif dependency.startswith('@loader_path/') and current is not None:
            dependencies.append(current.parent / dependency.removeprefix('@loader_path/'))
        elif dependency.startswith('@executable_path/'):
            dependencies.append(executable.resolve().parent / dependency.removeprefix('@executable_path/'))
        else:
            raise RuntimeError('test confinement: unresolved loader dependency')
    if headers != set(batch):
        raise RuntimeError('test confinement: incomplete loader inspection')
    return dependencies


def loader_assets(images: list[Path], protected: tuple[Path, ...], seconds: float, *, pinned: dict | None = None) -> tuple[list[Path], set[Path], dict[Path, tuple[int, int, int, int]]]:
    deadline = time.monotonic() + seconds
    pending = images
    seen = set()
    files = []
    aliases = set()
    identities = {}
    while pending:
        batch = []
        for path in pending:
            resolved = path.resolve()
            if resolved in seen or any(resolved.is_relative_to(root) for root in SYSTEM):
                continue
            resolved, image_aliases, identity = owned_image(path, protected, pinned)
            identities[resolved] = identity
            seen.add(resolved)
            if len(seen) > 256:
                raise RuntimeError('test confinement: loader image graph exceeds bound')
            batch.append(resolved)
            files.append(resolved)
            aliases.update(image_aliases)
        pending = []
        if not batch:
            break
        pending = image_dependencies(batch, images[0], protected, pinned, deadline)
    return files, aliases, identities


def interpreter_assets(files: list[Path], protected: tuple[Path, ...], seconds: float) -> tuple[list[Path], set[Path], dict[Path, tuple[int, int, int, int]]]:
    images = [Path(sys.executable), *(Path(sysconfig.get_path('stdlib')) / 'lib-dynload').glob('*.so')]
    libraries, aliases, identities = loader_assets(images, protected, seconds)
    return [*files, *libraries], aliases, identities
