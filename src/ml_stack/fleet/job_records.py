"""Private atomic Fleet job records and process ownership checks."""

import json
import math
import os
import stat
import threading
from itertools import islice
from pathlib import Path

import psutil

from ml_stack import lock
from ml_stack.files import write_json

MAX_RECORD_BYTES = 256 * 1024
MAX_RECORDS = 10000
STATES = {'queued', 'launching', 'preparing', 'running', 'done', 'failed', 'stopped', 'interrupted'}


def process_started(pid):
    try:
        process = psutil.Process(pid)
        return (process.create_time() if process.status() != psutil.STATUS_ZOMBIE
                and process.username() == psutil.Process().username() else None)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return None


def ownership(job):
    """Return whether the saved process still exists, or None when it cannot be verified."""
    if not job.pid or not job.process_started:
        return None
    try:
        process = psutil.Process(job.pid)
        if process.status() == psutil.STATUS_ZOMBIE or process.create_time() != job.process_started:
            return False
        return True if process.username() == psutil.Process().username() else None
    except psutil.NoSuchProcess:
        return False
    except psutil.AccessDenied:
        return None


def owns(job):
    return ownership(job) is True


class Records:
    def __init__(self, directory):
        self._io_lock = threading.RLock()
        self.directory = Path(directory)
        if self.directory.is_symlink():
            raise ValueError('Fleet job storage cannot be a symlink')
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        if os.name != 'nt' and self.directory.stat().st_uid != os.getuid():
            raise ValueError('Fleet job storage belongs to another account')
        self.directory.chmod(0o700)
        path = self.directory / '.runner.lock'
        if path.is_symlink():
            raise ValueError('Fleet runner lock cannot be a symlink')
        if path.exists() and os.name != 'nt' and path.stat().st_uid != os.getuid():
            raise ValueError('Fleet runner lock belongs to another account')
        self.fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
        if not lock.take(self.fd):
            os.close(self.fd)
            self.fd = None
            raise ValueError('another Fleet runner owns this durable job queue')

    def close(self):
        with self._io_lock:
            if self.fd is not None:
                lock.release(self.fd)
                os.close(self.fd)
                self.fd = None

    def read(self, path):
        if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_RECORD_BYTES:
            raise ValueError('Fleet job record is not a bounded regular file')
        mode = stat.S_IMODE(path.stat().st_mode)
        if os.name != 'nt' and (mode & 0o022 or path.stat().st_uid != os.getuid()):
            raise ValueError('Fleet job records must be private')
        row = json.loads(path.read_text())
        if not isinstance(row, dict):
            raise ValueError('Fleet job record must be an object')
        if row.get('version') not in (None, 2):
            raise ValueError('Fleet job record version is unsupported')
        if os.name != 'nt' and row.get('version') == 2 and mode & 0o077:
            raise ValueError('Fleet durable job records must be private')
        if type(row.get('capacity_held', False)) is not bool:
            raise ValueError('Fleet capacity reservation must be a boolean')
        for key in ('pid', 'returncode'):
            if row.get(key) is not None and type(row[key]) is not int:
                raise ValueError('Fleet process identifiers and outcomes must be integers')
        if row.get('pid') is not None and row['pid'] <= 0:
            raise ValueError('Fleet process identifiers must be positive')
        for key in ('submitted_at', 'started_at', 'finished_at', 'process_started'):
            value = row.get(key)
            if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or value < 0):
                raise ValueError('Fleet process timestamps must be finite nonnegative numbers')
        if not isinstance(row, dict) or row.get('id') != path.parent.name or row.get('state') not in STATES:
            raise ValueError('Fleet job record has an invalid identity or state')
        if (not isinstance(row.get('argv'), list) or not row['argv']
                or not all(isinstance(argument, str) for argument in row['argv'])
                or not isinstance(row.get('cwd'), str) or not isinstance(row.get('env', {}), dict)
                or not all(isinstance(key, str) and isinstance(value, str)
                           for key, value in row.get('env', {}).items())):
            raise ValueError('Fleet job record has an invalid execution configuration')
        return row

    def load(self):
        paths = list(islice(self.directory.glob('*/job.json'), MAX_RECORDS + 1))
        if len(paths) > MAX_RECORDS:
            raise ValueError('Fleet job recovery exceeds its record limit')
        result = []
        for path in paths:
            if path.parent.is_symlink():
                raise ValueError('Fleet job directories cannot be symlinks')
            result.append(self.read(path))
        return result

    def _write(self, job):
        if self.fd is None:
            return
        if not job.id or Path(job.id).name != job.id or job.id in ('.', '..'):
            raise ValueError('Fleet job identity must be a plain directory name')
        directory = self.directory / job.id
        if directory.is_symlink():
            raise ValueError('Fleet job directories cannot be symlinks')
        directory.mkdir(mode=0o700, exist_ok=True)
        if os.name != 'nt' and directory.stat().st_uid != os.getuid():
            raise ValueError('Fleet job directory belongs to another account')
        directory.chmod(0o700)
        path = directory / 'job.json'
        if path.is_symlink():
            raise ValueError('Fleet job records cannot be symlinks')
        row = {'version': 2, **job.public(), 'state': job.state}
        if len(json.dumps(row).encode()) > MAX_RECORD_BYTES:
            raise ValueError('Fleet job record exceeds its size limit')
        write_json(path, row)
        path.chmod(0o600)

    def write(self, job):
        with self._io_lock:
            self._write(job)
