"""A durable exit code for a Fleet job that outlives the daemon that launched it."""

import json
import os
import stat
import sys
from pathlib import Path

NAME = 'exit.json'
MAX_BYTES = 256

# Runs as the job's process: starts the real command, hands it every stop signal and writes
# its exit status beside the log before leaving. The daemon cannot wait on a child it no longer
# owns after a restart, so without this record a job that finishes meanwhile has no outcome.
# A torn write reads back as no record, which recovery reports as interrupted.
SCRIPT = '''
import json, os, signal, subprocess, sys
path, argv = sys.argv[1], sys.argv[2:]
try:
    child = subprocess.Popen(argv, creationflags=getattr(subprocess, 'CREATE_NEW_PROCESS_GROUP', 0))
except (OSError, ValueError) as exc:
    sys.stderr.write('failed to start: %s\\n' % exc)
    code = -1
else:
    def forward(number, _frame):
        try:
            child.send_signal(signal.CTRL_BREAK_EVENT if os.name == 'nt' else number)
        except OSError:
            pass
    for name in ('SIGTERM', 'SIGINT', 'SIGHUP', 'SIGBREAK'):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), forward)
    code = child.wait()
descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, 'O_NOFOLLOW', 0), 0o600)
with os.fdopen(descriptor, 'w') as stream:
    json.dump({'pid': os.getpid(), 'returncode': code}, stream)
    stream.flush()
    os.fsync(stream.fileno())
os._exit(code if code >= 0 else 128 - code)
'''


def command(directory, argv):
    """The argument vector that runs ``argv`` and records how it ended in ``directory``."""
    return [sys.executable, '-I', '-c', SCRIPT, str(Path(directory) / NAME), *argv]


def clear(directory):
    (Path(directory) / NAME).unlink(missing_ok=True)


def read(directory, pid):
    """The recorded exit code of the job process ``pid``, or None when there is no trusted record."""
    path = Path(directory) / NAME
    try:
        info = path.lstat()
        if (not stat.S_ISREG(info.st_mode) or info.st_size > MAX_BYTES
                or (os.name != 'nt' and (info.st_uid != os.getuid() or info.st_mode & 0o022))):
            return None
        row = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    code = row.get('returncode') if isinstance(row, dict) else None
    if not isinstance(row, dict) or row.get('pid') != pid or type(code) is not int:
        return None
    return code
