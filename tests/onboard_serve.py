"""A peer that serves files, as its own process: ``python onboard_serve.py ROOT MANIFEST PUB
SECRET [TLSDIR|-] [STATE]``. Prints its port on the first line, then serves until its input
closes. STATE is a directory holding ``devices.json`` and ``licences.json``."""

import base64
import sys
from pathlib import Path

from ml_stack.fleet import tls
from ml_stack.fleet.onboard.manifest import verify
from ml_stack.fleet.onboard.requests import Devices
from ml_stack.fleet.onboard.sharing import Licences
from ml_stack.fleet.onboard.transfer import Share, ShareServer, mac_gate


def main() -> None:
    root, manifest_path, pub, secret = sys.argv[1:5]
    tls_dir = sys.argv[5] if len(sys.argv) > 5 and sys.argv[5] != "-" else ""
    state = Path(sys.argv[6]) if len(sys.argv) > 6 else None
    raw = Path(manifest_path).read_bytes()
    manifest = verify(raw, base64.b64decode(pub))
    ident = tls.identity(Path(tls_dir), "peer") if tls_dir else None
    devices = Devices(state / "devices.json").all if state else (lambda: [])
    licences = Licences(state / "licences.json") if state else None
    with ShareServer(Share(Path(root), raw, manifest, licences),
                     authenticate=mac_gate(secret, devices), ident=ident) as s:
        print(s.port, flush=True)
        sys.stdin.read()


if __name__ == "__main__":
    main()
