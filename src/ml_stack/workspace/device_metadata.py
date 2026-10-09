"""Bounded device provenance for authenticated workspace identities."""
import math
import platform
import re
import socket
import time
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from ml_stack.home import device_id, machine_id

STATES = ('paired', 'agent-reported', 'local-observed', 'inherited', 'unknown')
SOURCES = ('fleet-pairing', 'agent-report', 'local-runtime', 'parent-registry', 'unknown')
TEXT = ('architecture', 'runtime_version')


def _bounded(*values):
    return all(type(text) is str and len(text) <= 128 and not any(ord(char) < 32 for char in text)
               for text in values)


def normalize(value):
    if not isinstance(value, dict):
        raise ValueError('device metadata is an object')
    identity = value.get('device_id')
    peer = value.get('peer_id')
    for field, text in (('device_id', identity), ('peer_id', peer)):
        if text is not None and (type(text) is not str or not re.fullmatch('[0-9a-f]{64}', text)):
            raise ValueError(f'{field} is a stable device fingerprint or null')
    machine = value.get('machine_id')
    if machine is not None and (type(machine) is not str or not re.fullmatch('[0-9a-f]{16}', machine)):
        raise ValueError('machine_id is a stable fleet identifier or null')
    hostname, system = value.get('hostname', ''), value.get('os', '')
    if not _bounded(hostname, system, *(value.get(key, '') for key in TEXT)):
        raise ValueError('device hostname, OS and runtime provenance are bounded text')
    state = value.get('verification', 'agent-reported' if value else 'unknown')
    if state not in STATES or (state == 'paired' and not peer):
        raise ValueError('paired device provenance needs its verified peer fingerprint')
    source = value.get('source', 'agent-report' if value else 'unknown')
    if source not in SOURCES:
        raise ValueError('device source is unsupported')
    peer_state = value.get('peer_verification', 'unknown')
    if peer_state not in ('paired', 'unknown') or (peer_state == 'paired' and not peer):
        raise ValueError('peer verification requires an authenticated peer fingerprint')
    commit = value.get('runtime_commit', '')
    if type(commit) is not str or (commit and not re.fullmatch('[0-9a-f]{40}', commit)):
        raise ValueError('runtime_commit is a full commit or empty')
    parent = value.get('inherited_from', '')
    if not _bounded(parent):
        raise ValueError('inherited device parent is a bounded registered identity')
    parent_state = value.get('parent_verification', 'unknown')
    if parent_state not in STATES:
        raise ValueError('parent device confidence is unsupported')
    if (state == 'inherited' or source == 'parent-registry') and not parent:
        raise ValueError('inherited device provenance requires its registered parent')
    observed = value.get('observed_at', 0.0)
    if type(observed) not in (int, float) or not math.isfinite(observed) or observed < 0:
        raise ValueError('observed_at is a finite nonnegative timestamp')
    return {'device_id': identity, 'hostname': hostname, 'os': system,
            'label': ' · '.join(filter(None, (system, hostname))) or 'Unknown device',
            'peer_id': peer, 'verification': state, 'source': source, 'runtime_commit': commit,
            'inherited_from': parent, 'parent_verification': parent_state,
            'peer_verification': peer_state, 'machine_id': machine,
            **{key: value.get(key, '') for key in TEXT}, 'observed_at': observed}


def reported(value, *, peer=None, observed_at=0.0):
    """An agent's own device report, as the weakest provenance and never as a stable identity.

    The hardware facts it names are kept for display; ``device_id`` and ``machine_id`` are what
    rewards and placement key on, so they are only ever set by this host's own observation. A
    peer fingerprint is the daemon's, taken from the authenticated pairing, never from the report.
    """
    if not isinstance(value, dict):
        raise ValueError('device metadata is an object')
    return normalize({**value, 'device_id': None, 'machine_id': None, 'peer_id': peer,
                      'verification': 'agent-reported', 'source': 'agent-report',
                      'peer_verification': 'paired' if peer else 'unknown',
                      'inherited_from': '', 'parent_verification': 'unknown',
                      'observed_at': observed_at})


def inherited(value, parent):
    """Return parent device facts with explicitly inherited confidence."""
    device = normalize(value)
    return normalize({**device, 'inherited_from': parent, 'verification': 'inherited',
                      'source': 'parent-registry', 'parent_verification': device['verification']})


def runtime_commit():
    """The commit the running ml-stack runtime was built from, or empty when it is unstamped."""
    marker = Path(__file__).resolve().parents[1] / 'fleet' / 'built-from'
    try:
        found = marker.read_text(encoding='utf-8').strip()
    except OSError:
        return ''
    return found if re.fullmatch('[0-9a-f]{40}', found) else ''


@lru_cache(maxsize=1)
def current():
    """The local host observation, cached for this process."""
    try:
        identity = device_id()
    except RuntimeError:
        identity = None
    try:
        runtime_version = version('ml-stack')
    except PackageNotFoundError:
        runtime_version = ''
    wsl = platform.system() == 'Linux' and 'microsoft' in platform.release().lower()
    return normalize({'device_id': identity, 'machine_id': machine_id(), 'hostname': socket.gethostname()[:128],
                      'os': 'Linux (WSL)' if wsl else {'Darwin': 'macOS'}.get(platform.system(), platform.system()),
                      'architecture': platform.machine(), 'runtime_version': runtime_version,
                      'verification': 'local-observed' if identity else 'unknown',
                      'source': 'local-runtime', 'runtime_commit': runtime_commit(),
                      'observed_at': time.time()})
