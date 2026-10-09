"""Installed-interpreter simulation worker protocol."""

import contextlib
import json
import queue
import sys
import threading

from poolhouse.gym.simulation import worker


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
        try:
            while line := sys.stdin.readline(1_000_001):
                if len(line) > 1_000_000:
                    raise ValueError("Simulation command exceeds 1 MB")
                message = json.loads(line)
                if (not isinstance(message, list) or len(message) != 2
                        or not isinstance(message[0], str) or not isinstance(message[1], dict)):
                    raise ValueError("Simulation commands require a command name and payload object")
                commands.put(message)
        except (ValueError, TypeError) as exc:
            commands.put(("protocol-error", {"error": str(exc)}))
        finally:
            commands.put(("close", {}))

    threading.Thread(target=read_commands, daemon=True).start()
    updates = Updates(sys.stdout)
    with contextlib.redirect_stdout(sys.stderr):
        worker(json.loads(sys.argv[1]), commands, updates)


if __name__ == "__main__":
    serve_protocol()
