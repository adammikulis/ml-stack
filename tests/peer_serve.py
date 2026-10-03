"""A device that shares model files, as its own process (the serving half of the peer-first
download tests): ``python peer_serve.py ROOT MANIFEST PUB SECRET TLSDIR STATE``. Prints its
port, then serves until its input closes. It runs the real `ShareServer` with the real
quarantine veto; its own ML_STACK_HOME (the environment) holds its sentinel.

``PEER_LOG`` names a file that gets one line per file request (``bytes=a-b``); ``PEER_CUT_AFTER=N``
makes the process die after N file requests (a cut connection); ``PEER_STALL=1`` makes file
requests hang (a peer too slow to use).
"""

import base64
import os
import sys
import time
from pathlib import Path

from ml_stack.fleet import tls
from ml_stack.fleet.onboard.manifest import verify
from ml_stack.fleet.onboard.peerfirst import quarantine_veto
from ml_stack.fleet.onboard.requests import Devices
from ml_stack.fleet.onboard.sharing import Licences
from ml_stack.fleet.onboard.transfer import API, Share, ShareServer, mac_gate


class Logged(ShareServer):
    served = 0

    def dispatch(self, call):
        if call.path.startswith(f"{API}/files/") and call.method == "GET":
            if os.environ.get("PEER_STALL"):
                time.sleep(120)
            with Path(os.environ["PEER_LOG"]).open("a") as log:
                log.write(call.headers.get("Range", "") + "\n")
            Logged.served += 1
            if Logged.served > int(os.environ.get("PEER_CUT_AFTER", "1000000")):
                os._exit(0)
        return super().dispatch(call)


def main() -> None:
    root, manifest_path, pub, secret, tls_dir, state = sys.argv[1:7]
    raw = Path(manifest_path).read_bytes()
    manifest = verify(raw, base64.b64decode(pub))
    ident = tls.identity(Path(tls_dir), "peer")
    devices = Devices(Path(state) / "devices.json").all
    licences = Licences(Path(state) / "licences.json")
    share = Share(Path(root), raw, manifest, licences, quarantine_veto)
    with Logged(share, authenticate=mac_gate(secret, devices), ident=ident) as s:
        print(s.port, flush=True)
        sys.stdin.read()


if __name__ == "__main__":
    main()
