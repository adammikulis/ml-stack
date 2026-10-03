"""Installed-interpreter simulation worker protocol."""

import contextlib
import json
import queue
import sys
import threading

from ml_stack.gym.simulation import worker


class Updates:
    """Write sequenced snapshots to the protocol stream."""

    def __init__(self, stream):
        self.stream = stream

    def put_nowait(self, state):
        self.stream.write(json.dumps(state) + "\n")
        self.stream.flush()


def serve_protocol():
    commands = queue.Queue(maxsize=64)

    def read_commands():
        for line in sys.stdin:
            commands.put(json.loads(line))
        commands.put(("close", {}))

    threading.Thread(target=read_commands, daemon=True).start()
    updates = Updates(sys.stdout)
    with contextlib.redirect_stdout(sys.stderr):
        worker(json.loads(sys.argv[1]), commands, updates)


if __name__ == "__main__":
    serve_protocol()
