"""A loopback TCP listener forwarding to a confined model's Unix socket."""

from __future__ import annotations

import contextlib
import selectors
import signal
import socket
import sys
import threading
import time
from pathlib import Path

from poolhouse.platform import launch

__all__ = ["arguments", "supervise"]


def arguments(argv: list[str], port: int, path: str) -> list[str]:
    """The relay supervisor command for an already confined command."""
    return [sys.executable, str(Path(__file__).resolve()), str(port), path, *argv]


def _copy(client: socket.socket, target: str, stopping: threading.Event) -> None:
    with client, socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as model:
        try:
            model.connect(target)
            client.settimeout(1)
            model.settimeout(1)
            with selectors.DefaultSelector() as reader:
                reader.register(client, selectors.EVENT_READ, model)
                reader.register(model, selectors.EVENT_READ, client)
                while reader.get_map():
                    events = reader.select(0.25)
                    if stopping.is_set() and not events:
                        return
                    for ready, _ in events:
                        data = ready.fileobj.recv(65536)
                        if data:
                            ready.data.sendall(data)
                        else:
                            reader.unregister(ready.fileobj)
                            ready.data.shutdown(socket.SHUT_WR)
        except OSError:
            return


def _accept(listener: socket.socket, target: str, stopping: threading.Event,
            workers: list[threading.Thread]) -> None:
    slots = threading.BoundedSemaphore(32)

    def forward(client: socket.socket) -> None:
        try:
            _copy(client, target, stopping)
        finally:
            slots.release()

    listener.settimeout(0.25)
    while not stopping.is_set():
        try:
            client, _ = listener.accept()
        except TimeoutError:
            continue
        except OSError:
            break
        if not slots.acquire(blocking=False):
            client.close()
            continue
        workers[:] = [worker for worker in workers if worker.is_alive()]
        worker = threading.Thread(target=forward, args=(client,), daemon=True)
        workers.append(worker)
        worker.start()


def supervise(argv: list[str] | None = None) -> int:
    """Serve a confined command through loopback until it exits."""
    args = sys.argv[1:] if argv is None else argv
    port, target, *command = args
    stopping = threading.Event()
    workers: list[threading.Thread] = []
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind(("127.0.0.1", int(port)))
            listener.listen(32)
            child = launch(command)

            def stop(_number: int, _frame: object) -> None:
                stopping.set()
                child.terminate()

            signal.signal(signal.SIGTERM, stop)
            signal.signal(signal.SIGINT, stop)
            accepting = threading.Thread(target=_accept, args=(listener, target, stopping, workers), daemon=True)
            accepting.start()
            try:
                return child.wait()
            finally:
                stopping.set()
                listener.close()
                accepting.join(1)
                deadline = time.monotonic() + 1
                for worker in workers:
                    worker.join(max(0, deadline - time.monotonic()))
    finally:
        with contextlib.suppress(OSError):
            Path(target).unlink(missing_ok=True)
            Path(target).parent.rmdir()


if __name__ == "__main__":
    raise SystemExit(supervise())
