"""A paired device for the mesh tests: a real daemon answering the journal exchange from its own state root.

Run as ``python mesh_server.py ROOT FINGERPRINT SECRET``: ``FINGERPRINT`` and ``SECRET`` name the
one device it accepts. It prints its port and serves until killed.
"""

import sys
from pathlib import Path

from ml_stack.fleet.api import Daemon, make_handler
from ml_stack.fleet.jobs import JobRunner
from ml_stack.fleet.onboard.requests import Device
from ml_stack.http import Server


def main(root, fingerprint, secret):
    device = Device(fingerprint, 'peer', 'peer-host', '127.0.0.1', 1, mine=True, secret=secret)
    runner = JobRunner(root / 'daemon')
    server = Server(('127.0.0.1', 0), make_handler(Daemon(
        runner, root / 'files', 'unused-fleet-token', devices=lambda: [device])))
    print(server.server_address[1], flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main(Path(sys.argv[1]), sys.argv[2], sys.argv[3])
