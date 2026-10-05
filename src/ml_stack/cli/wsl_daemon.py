"""The Linux daemon held by the Windows launcher's input pipe."""

from __future__ import annotations

import os
import signal
import sys
import threading

from .daemon import main


def _watch() -> None:
    sys.stdin.buffer.read()
    os.kill(os.getpid(), signal.SIGTERM)


if __name__ == "__main__":
    threading.Thread(target=_watch, daemon=True).start()
    raise SystemExit(main())
