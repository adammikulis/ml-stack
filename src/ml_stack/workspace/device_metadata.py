"""Bounded device provenance for authenticated workspace identities."""
import platform
import re
import socket
from functools import lru_cache

from ml_stack.home import device_id


def normalize(value):
    if not isinstance(value, dict):
        raise ValueError('device metadata is an object')
    identity = value.get('device_id')
    peer = value.get('peer_id')
    for field, text in (('device_id', identity), ('peer_id', peer)):
        if text is not None and (type(text) is not str or not re.fullmatch('[0-9a-f]{64}', text)):
            raise ValueError(f'{field} is a stable device fingerprint or null')
    hostname, system = value.get('hostname', ''), value.get('os', '')
    if any(type(text) is not str or len(text) > 128 or any(ord(char) < 32 for char in text)
           for text in (hostname, system)):
        raise ValueError('device hostname and OS are bounded text')
    state = value.get('verification', 'agent-reported')
    if state not in ('paired', 'agent-reported', 'local-observed', 'unknown') or (state == 'paired' and not peer):
        raise ValueError('paired device provenance needs its verified peer fingerprint')
    source = value.get('source', 'agent-report')
    if source not in ('fleet-pairing', 'agent-report', 'local-runtime', 'unknown'):
        raise ValueError('device source is unsupported')
    return {'device_id': identity, 'hostname': hostname, 'os': system,
            'label': ' · '.join(filter(None, (system, hostname))) or 'Unknown device',
            'peer_id': peer, 'verification': state, 'source': source}


@lru_cache(maxsize=1)
def current():
    """The local host observation, cached for this process."""
    try:
        identity = device_id()
    except RuntimeError:
        identity = None
    return normalize({'device_id': identity, 'hostname': socket.gethostname()[:128],
                      'os': {'Darwin': 'macOS'}.get(platform.system(), platform.system()),
                      'verification': 'local-observed' if identity else 'unknown',
                      'source': 'local-runtime'})
