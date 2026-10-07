"""Loopback-only Windows UI connections relayed through WSL standard streams."""

from __future__ import annotations

import contextlib
import socket
import subprocess
import sys
import threading
from collections.abc import Sequence

from ml_stack.platform import start_process

from .wsl_network import NetworkBridge

IDLE_SECONDS = 300


class LocalUIBridge(NetworkBridge):
    """A Windows loopback listener with owned per-connection WSL helpers."""

    def __init__(self, argv: Sequence[str], port: int) -> None:
        super().__init__("127.0.0.1", ("127.0.0.1", port), "", 0, socket.socket)
        self.argv = list(argv)
        self.processes: set[subprocess.Popen] = set()

    def start(self) -> LocalUIBridge:
        try:
            self._listen(self.target[1], self._relay)
            return self
        except OSError:
            self.close()
            raise

    def _relay(self, client: socket.socket) -> None:
        client.settimeout(IDLE_SECONDS)
        child = start_process(self.argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                 stderr=subprocess.DEVNULL)
        with self.lock:
            self.processes.add(child)
            stopped = self.stop.is_set()
        uplink = threading.Thread(target=self._upload, args=(client, child), daemon=True)
        uplink.start()
        try:
            if stopped:
                child.terminate()
            while not self.stop.is_set() and (data := child.stdout.read1(65536)):
                client.sendall(data)
        except OSError:
            pass
        finally:
            with contextlib.suppress(OSError):
                client.shutdown(socket.SHUT_RDWR)
            with contextlib.suppress(OSError):
                child.terminate()
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=5)
            uplink.join(1)
            child.stdout.close()
            with self.lock:
                self.processes.discard(child)

    def _upload(self, client: socket.socket, child: subprocess.Popen) -> None:
        try:
            while not self.stop.is_set() and (data := client.recv(65536)):
                child.stdin.write(data)
                child.stdin.flush()
        except OSError:
            pass
        finally:
            with contextlib.suppress(OSError):
                child.stdin.close()

    def close(self) -> None:
        self.stop.set()
        with self.lock:
            children = list(self.processes)
            for child in children:
                with contextlib.suppress(OSError):
                    child.terminate()
        super().close()
        for child in children:
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=5)


def stdio(port: int) -> None:
    """Relay stdin and stdout through a Linux loopback TCP connection."""
    with socket.create_connection(("127.0.0.1", port), timeout=5) as upstream:
        upstream.settimeout(IDLE_SECONDS)

        def upload() -> None:
            try:
                while data := sys.stdin.buffer.read1(65536):
                    upstream.sendall(data)
            except OSError:
                pass
            finally:
                with contextlib.suppress(OSError):
                    upstream.shutdown(socket.SHUT_WR)

        threading.Thread(target=upload, daemon=True).start()
        while data := upstream.recv(65536):
            sys.stdout.buffer.write(data)
            sys.stdout.buffer.flush()


if __name__ == "__main__":
    stdio(int(sys.argv[1]))
