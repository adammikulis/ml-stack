"""A peer that serves files, as its own process: ``python onboard_serve.py ROOT MANIFEST PUB
SECRET [TLSDIR]``. Prints its port on the first line, then serves until its input closes."""

import base64
import sys
from pathlib import Path

from ml_stack import macauth
from ml_stack.fleet import tls
from ml_stack.fleet.onboard.manifest import verify
from ml_stack.fleet.onboard.transfer import Share, ShareServer, mac_gate


def main() -> None:
    root, manifest_path, pub, secret = sys.argv[1:5]
    raw = Path(manifest_path).read_bytes()
    manifest = verify(raw, base64.b64decode(pub))
    ident = tls.identity(Path(sys.argv[5]), "peer") if len(sys.argv) > 5 else None
    auth = macauth.Authenticator(lambda: [secret])
    with ShareServer(Share(Path(root), raw, manifest), authenticate=mac_gate(auth),
                     ident=ident) as s:
        print(s.port, flush=True)
        sys.stdin.read()


if __name__ == "__main__":
    main()
