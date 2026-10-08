"""Sealed browser images and selector-derived resource receipts."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import platform
import stat
import time
from dataclasses import dataclass, field
from pathlib import Path

MAX_FILES = 8192
MAX_BYTES = 1024 * 1024 * 1024


def identity(path: Path) -> tuple[int, ...]:
    value = path.lstat()
    return (value.st_dev, value.st_ino, value.st_uid, value.st_mode,
            value.st_nlink, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def owned_tree(root: Path, excluded: tuple[Path, ...]) -> list[Path]:
    if root.resolve(strict=True) != root or any(root.is_relative_to(path) for path in excluded):
        raise RuntimeError('browser confinement: redirected or protected asset root')
    for parent in (root, *root.parents):
        info = parent.lstat()
        sticky = parent == Path('/private/tmp') and info.st_uid == 0 and info.st_mode & stat.S_ISVTX
        if (not stat.S_ISDIR(info.st_mode) or info.st_uid not in (0, os.getuid())
                or (info.st_mode & 0o022 and not sticky)):
            raise RuntimeError('browser confinement: replaceable asset ancestry')
    result = []
    total = 0
    pending = [root]
    nodes = 0
    while pending:
        directory = pending.pop()
        with os.scandir(directory) as entries:
            paths = []
            for entry in entries:
                nodes += 1
                if nodes > MAX_FILES:
                    raise RuntimeError('browser confinement: asset node count exceeds bound')
                paths.append(Path(entry.path))
        for path in sorted(paths):
            info = path.lstat()
            if (info.st_uid not in (0, os.getuid()) or info.st_mode & 0o022
                    or not (stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode))):
                raise RuntimeError('browser confinement: foreign, writable or linked asset')
            if stat.S_ISDIR(info.st_mode):
                pending.append(path)
            if stat.S_ISREG(info.st_mode):
                if info.st_nlink != 1:
                    raise RuntimeError('browser confinement: multiply linked asset')
                result.append(path)
                total += info.st_size
                if len(result) > MAX_FILES or total > MAX_BYTES:
                    raise RuntimeError('browser confinement: asset tree exceeds bounds')
    return result


@dataclass
class Assets:
    files: list[Path] = field(default_factory=list)
    directories: set[Path] = field(default_factory=set)
    metadata: set[Path] = field(default_factory=set)
    executables: set[Path] = field(default_factory=set)
    identities: dict[Path, tuple[int, ...]] = field(default_factory=dict)
    environment: dict[str, str] = field(default_factory=dict)
    receipt: dict = field(default_factory=dict)

    def proof(self) -> dict:
        inventory = [(str(path), expected) for path, expected in sorted(self.identities.items())]
        return {**self.receipt, 'images': [(str(path), self.identities[path]) for path in sorted(self.executables)],
                'inventory_sha256': hashlib.sha256(json.dumps(inventory).encode()).hexdigest()}

    def recheck(self) -> None:
        for path, expected in self.identities.items():
            if identity(path) != expected:
                raise RuntimeError('browser confinement: asset identity changed')


def seal(source: Path, destination: Path, excluded: tuple[Path, ...]) -> tuple[list[Path], str]:
    paths = owned_tree(source, excluded)
    destination.mkdir(mode=0o700)
    digest = hashlib.sha256()
    result = []
    total = 0
    deadline = time.monotonic() + 60
    for path in paths:
        before = identity(path)
        target = destination / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(descriptor, 'rb') as incoming, target.open('xb') as outgoing:
            if identity(path) != before:
                raise RuntimeError('browser confinement: source changed before sealing')
            digest.update(str(path.relative_to(source)).encode() + b'\0')
            while block := incoming.read(65536):
                total += len(block)
                if total > MAX_BYTES or time.monotonic() >= deadline:
                    raise RuntimeError('browser confinement: copied asset bytes exceed bound')
                outgoing.write(block)
                digest.update(block)
            if identity(path) != before or identity(path)[:2] != (os.fstat(incoming.fileno()).st_dev, os.fstat(incoming.fileno()).st_ino):
                raise RuntimeError('browser confinement: source changed while sealing')
        target.chmod(0o500 if before[3] & 0o111 else 0o400)
        result.append(target)
    for directory in sorted((p for p in destination.rglob('*') if p.is_dir()), reverse=True):
        directory.chmod(0o500)
    destination.chmod(0o500)
    return result, digest.hexdigest()


def driver_assets(driver: Path, files: list[Path], environment: dict[str, str]) -> Assets:
    node = driver / 'node'
    if node not in files or not os.access(node, os.X_OK):
        raise RuntimeError('browser confinement: driver executable is missing')
    assets = Assets(files=files, executables={node})
    assets.directories = {parent for path in files for parent in path.parents if parent.is_relative_to(driver)}
    assets.identities = {path: identity(path) for path in (*assets.files, *assets.directories)}
    assets.environment = {'PATH': str(driver) + os.pathsep + environment.get('PATH', os.defpath)}
    assets.receipt = {'resource': 'javascript', 'executables': [str(node)], 'files': len(files)}
    assets.recheck()
    return assets


def prepare(resources: frozenset[str], control: Path, environment: dict[str, str],
            protected: tuple[Path, ...], admitted: tuple[Path, ...]) -> Assets | None:
    if not resources & {'browser', 'javascript'}:
        return None
    if platform.system() != 'Darwin' or platform.machine() != 'arm64':
        raise RuntimeError('browser confinement: this adapter requires Darwin arm64')
    spec = importlib.util.find_spec('playwright')
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError('browser confinement: Playwright is not installed')
    package = Path(next(iter(spec.submodule_search_locations))).resolve(strict=True)
    if not any(package.is_relative_to(root) for root in admitted):
        raise RuntimeError('browser confinement: Playwright is outside admitted runtime')
    driver = package / 'driver'
    driver_files = owned_tree(driver, protected)
    driver_receipt = driver_assets(driver, driver_files, environment)
    node = driver / 'node'
    if 'browser' not in resources:
        return driver_receipt
    manifest_path = driver / 'package' / 'browsers.json'
    if manifest_path.stat().st_size > 1024 * 1024:
        raise RuntimeError('browser confinement: installed browser manifest exceeds bound')
    descriptor = os.open(manifest_path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, 'rb') as stream:
        manifest_bytes = stream.read(1024 * 1024 + 1)
    if len(manifest_bytes) > 1024 * 1024:
        raise RuntimeError('browser confinement: installed browser manifest exceeds bound')
    manifest = json.loads(manifest_bytes)
    entries = [value for value in manifest['browsers'] if value['name'] == 'chromium-headless-shell']
    if len(entries) != 1 or not str(entries[0]['revision']).isdecimal():
        raise RuntimeError('browser confinement: invalid installed browser revision')
    revision = str(entries[0].get('revisionOverrides', {}).get('mac-arm64', entries[0]['revision']))
    if not revision.isdecimal() or len(revision) > 16:
        raise RuntimeError('browser confinement: invalid platform browser revision')
    origin = Path.home() / 'Library' / 'Caches' / 'ms-playwright' / f'chromium_headless_shell-{revision}'
    browser_root = control / 'browsers'
    browser_root.mkdir(mode=0o700)
    sealed = browser_root / origin.name
    files, digest = seal(origin, sealed, protected)
    executable = sealed / 'chrome-headless-shell-mac-arm64' / 'chrome-headless-shell'
    if executable not in files:
        raise RuntimeError('browser confinement: browser or driver executable is missing')
    assets = Assets(files=[*driver_files, *files], executables={node, executable})
    for path in assets.files:
        assets.directories.update(parent for parent in path.parents if parent.is_relative_to(driver) or parent.is_relative_to(browser_root))
    assets.metadata.update((browser_root, driver))
    assets.identities = {path: identity(path) for path in (*assets.files, *assets.directories, *assets.metadata)}
    assets.environment = {'PLAYWRIGHT_BROWSERS_PATH': str(browser_root), 'PLAYWRIGHT_NODEJS_PATH': str(node),
                          'PATH': str(driver) + os.pathsep + environment.get('PATH', os.defpath)}
    assets.receipt = {'revision': revision, 'sha256': digest, 'files': len(assets.files),
                      'executables': sorted(map(str, assets.executables)), 'origin': str(origin),
                      'installed_manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest()}
    assets.recheck()
    return assets
