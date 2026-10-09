"""Native world artifact definitions and safe scenario imports."""

import hashlib
import json
import os
from pathlib import Path

from poolhouse.files import write_json
from poolhouse.gym.paths import artifact_root


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def directory(backend, definition):
    encoded = json.dumps(definition, sort_keys=True, separators=(',', ':')).encode()
    path = artifact_root() / 'worlds' / backend / hashlib.sha256(encoded).hexdigest()
    path.mkdir(parents=True, exist_ok=True)
    return path


def record(path, backend, definition, files):
    manifest = {'version': 1, 'backend': backend, 'definition': definition,
                'files': {name: {'path': str(file), 'sha256': digest(file)} for name, file in files.items()}}
    write_json(path / 'world.json', manifest)
    return {**manifest, 'manifest': str(path / 'world.json')}


def imported(value):
    root = os.environ.get('POOLHOUSE_GYM_FILES_ROOT')
    if not root:
        raise ValueError('Manual world import requires the daemon files root (POOLHOUSE_GYM_FILES_ROOT)')
    root = Path(root).resolve(strict=True)
    candidate = Path(value)
    if candidate.is_absolute() or '..' in candidate.parts:
        raise ValueError('World files must use relative paths under the daemon files root')
    candidate = (root / candidate).resolve(strict=True)
    if not candidate.is_relative_to(root) or not candidate.is_file():
        raise ValueError('World file must remain under the daemon files root')
    if candidate.stat().st_size > 20_000_000:
        raise ValueError('World XML must be at most 20 MB')
    return candidate
