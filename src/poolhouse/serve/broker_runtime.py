"""Observed broker runtime identity and wire compatibility."""

import hashlib
import os
import platform
import sys
from pathlib import Path

import psutil

import poolhouse
from poolhouse.net import git
from poolhouse.serve import provenance

PROTOCOL = 1


def snapshot() -> dict:
    """Capture the broker process, loaded package source and Python environment."""
    package = Path(poolhouse.__file__).resolve().parent
    root = next((parent for parent in package.parents
                 if parent / 'src' / 'poolhouse' == package and (parent / '.git').exists()), None)
    commit, dirty = None, None
    if root:
        try:
            commit = git.head(root)
            dirty = bool(git.run(['status', '--porcelain'], cwd=root).stdout.strip())
        except (git.GitFailed, OSError):
            pass
    process = psutil.Process(os.getpid())
    environment = {'interpreter': sys.executable, 'prefix': sys.prefix, 'base_prefix': sys.base_prefix,
                   'python': platform.python_version(), 'package_root': str(package),
                   'package_version': poolhouse.__version__}
    source = package / 'serve' / 'broker_wire.py'
    implementation = hashlib.sha256(source.read_bytes()).hexdigest() if source.is_file() else None
    return {'state': 'observed', 'protocol': PROTOCOL, 'source_commit': commit, 'source_dirty': dirty,
            'implementation_sha256': implementation, 'environment': environment,
            'pid': process.pid, 'pid_started': process.create_time(), 'owner': process.username(),
            'requester': provenance.asked()['requester'], 'requester_authority': 'claimed'}


def reported(value) -> dict:
    """Expose absent provenance as unknown and explicit protocol mismatches as incompatible."""
    if not isinstance(value, dict) or not isinstance(value.get('protocol'), int) \
            or isinstance(value.get('protocol'), bool):
        return {'state': 'unknown', 'protocol': None, 'source_commit': None, 'environment': None,
                'pid': None, 'pid_started': None, 'owner': None, 'requester': None,
                'compatibility': 'unknown',
                'action': 'The running broker does not advertise provenance; verify its grant before use.'}
    compatible = value['protocol'] == PROTOCOL
    return {**value, 'compatibility': 'compatible' if compatible else 'incompatible',
            'action': '' if compatible else 'Upgrade the owned broker at a quiescent boundary; preserve foreign holders.'}
