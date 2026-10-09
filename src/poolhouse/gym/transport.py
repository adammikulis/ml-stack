"""JSONL transport to simulator workers in an installed Python environment."""

import contextlib
import json
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path

from poolhouse.platform import start_process, terminate_process_group


def interpreter(environment=None):
    """Resolve the installed simulator interpreter."""
    choices = json.loads(os.environ.get("POOLHOUSE_GYM_PYTHONS", "{}"))
    if not isinstance(choices, dict):
        raise ValueError("Gym interpreter choices must be an object")
    configured = choices.get(environment) or os.environ.get("POOLHOUSE_GYM_PYTHON")
    if configured and not isinstance(configured, str):
        raise ValueError("Gym interpreter paths must be strings")
    if configured:
        python = Path(configured).expanduser()
        if not python.is_file():
            raise RuntimeError("The configured Gym Python interpreter is missing")
        return str(python)
    if getattr(sys, "frozen", False):
        raise RuntimeError("Install Gym libraries in the managed Python environment")
    return sys.executable


def python_environment():
    """The UTF-8 environment for Python JSONL workers."""
    return {**os.environ, "PYTHONUTF8": "1"}


class Commands:
    """Write control messages to a simulator process."""

    def __init__(self, stream):
        self.stream = stream

    def put(self, message):
        self.stream.write(json.dumps(message) + "\n")
        self.stream.flush()

    def close(self):
        with contextlib.suppress(BrokenPipeError):
            self.stream.close()


class Process:
    """Wrap an external worker and capture its latest snapshots."""

    def __init__(self, settings, updates, log):
        self.settings, self.updates, self.log = settings, updates, log

    def start(self):
        environment = python_environment()
        environment.pop("POOLHOUSE_GYM_PYTHON", None)
        environment.pop("POOLHOUSE_GYM_PYTHONS", None)
        self.handle = start_process([interpreter(self.settings["environment"]), "-m", "poolhouse.gym.worker_entry",
                                        json.dumps(self.settings)], stdin=subprocess.PIPE,
                                       stdout=subprocess.PIPE, stderr=self.log, text=True, bufsize=1, env=environment)
        self.commands = Commands(self.handle.stdin)
        threading.Thread(target=self.read, daemon=True).start()

    def read(self):
        for line in self.handle.stdout:
            try:
                snapshot = json.loads(line)
            except json.JSONDecodeError:
                continue
            try:
                self.updates.put_nowait(snapshot)
            except queue.Full:
                with contextlib.suppress(queue.Empty):
                    self.updates.get_nowait()
                self.updates.put_nowait(snapshot)
        self.handle.stdout.close()

    def is_alive(self):
        return self.handle.poll() is None

    def join(self, timeout):
        try:
            self.handle.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return

    def terminate(self):
        terminate_process_group(self.handle)
