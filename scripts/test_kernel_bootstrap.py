"""Isolated precollection kernel probes and supervisor acknowledgement."""
from __future__ import annotations

import errno
import json
import os
import runpy
import signal
import socket
import subprocess
import sys
from pathlib import Path


def denied(operation) -> None:
    try:
        operation()
    except PermissionError:
        return
    raise RuntimeError("test confinement: precollection operation was permitted")


def connect(endpoint: str, unix: bool = False) -> None:
    if unix:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(2)
            connection.connect(endpoint)
    else:
        host, port = endpoint.rsplit(":", 1)
        with socket.create_connection((host, int(port)), timeout=2):
            pass


def probe(manifest: dict) -> list[str]:
    canary = Path(manifest["canary"])
    denied(canary.read_bytes)
    denied(lambda: canary.write_bytes(b"forged"))
    denied(lambda: Path(manifest["escape"]).read_bytes())
    denied(lambda: os.link(canary, Path(manifest["escape"]).parent / "hardlink-canary"))
    denied(lambda: os.link(sys.argv[1], Path(manifest["escape"]).parent / "hardlink-manifest"))
    denied(lambda: connect(manifest["denied_unix"], True))
    denied(lambda: connect(manifest["denied_tcp"]))
    denied(lambda: os.kill(manifest["target_pid"], signal.SIGUSR1))
    code = "from pathlib import Path; import sys\ntry: Path(sys.argv[1]).read_bytes()\nexcept PermissionError: sys.exit(0)\nsys.exit(9)"
    child = subprocess.run([sys.executable, "-I", "-S", "-c", code, str(canary)],
                           stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           close_fds=True, timeout=5, check=False)
    if child.returncode != 0:
        raise RuntimeError("test confinement: descendant read denial failed")
    endpoint = runpy.run_path(manifest["helper"])
    with endpoint["connection"](manifest["admission"], manifest["identity"]) as connection, connection.makefile("rwb") as stream:
        stream.write(json.dumps({"operation": "ping", "token": manifest["token"], "parent": None,
                                 "label": "precollection", "phase": "collection", "heavy": False}).encode() + b"\n")
        stream.flush()
        if json.loads(stream.readline(4097)) != {"ping": True}:
            raise RuntimeError("test confinement: authenticated admission ping failed")
    return ["read", "write", "symlink", "hardlink-canary", "hardlink-manifest", "unix", "tcp", "signal", "descendant", "admission"]


def main() -> None:
    with Path(sys.argv[1]).open("rb") as source:
        raw = source.read(65537)
    if len(raw) > 65536:
        raise RuntimeError("test confinement: bootstrap manifest exceeds bound")
    manifest = json.loads(raw)
    if sys.argv[2] == "retired":
        retired(manifest)
        return
    output, acknowledgement = int(sys.argv[2]), int(sys.argv[3])
    result = {"nonce": manifest["nonce"], "profile": manifest["profile"], "checks": probe(manifest)}
    os.write(output, json.dumps(result).encode() + b"\n")
    os.close(output)
    if os.read(acknowledgement, 1) != b"a":
        raise RuntimeError("test confinement: supervisor refused collection")
    os.close(acknowledgement)
    raise SystemExit(subprocess.run(manifest["argv"], env=os.environ, close_fds=True, check=False).returncode)


def retired(manifest: dict) -> None:
    child = os.fork()
    if child:
        os.waitpid(child, 0)
        return
    os.setsid()
    child = os.fork()
    if child:
        os._exit(0)
    output, acknowledgement = int(sys.argv[4]), int(sys.argv[5])
    try:
        checks = retired_checks(manifest)
        os.write(output, json.dumps({"nonce": sys.argv[3], "pid": os.getpid(), "checks": checks}).encode() + b"\n")
        os.close(output)
        if os.read(acknowledgement, 1) != b"a":
            os._exit(2)
        os.close(acknowledgement)
        os._exit(0)
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as error:
        os.write(output, json.dumps({"nonce": sys.argv[3], "error": str(error)[:256]}).encode() + b"\n")
        os._exit(3)


def retired_checks(manifest: dict) -> list[str]:
    endpoint = runpy.run_path(manifest["helper"])
    for address, identity in ((manifest["admission"], manifest["identity"]),
                               ("unix:" + manifest["terminal"], manifest["terminal_identity"])):
        try:
            endpoint["connection"](address, identity).close()
        except ConnectionRefusedError:
            pass
        else:
            raise RuntimeError("test confinement: retired endpoint remains live")
        denied(lambda address=address: Path(address[5:]).unlink())
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as stream:
            try:
                stream.bind(address[5:])
            except OSError as error:
                if error.errno != errno.EADDRINUSE:
                    raise
            else:
                raise RuntimeError("test confinement: retained endpoint was rebound")
        endpoint["verify_socket"](address[5:], json.loads(identity))
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as stream:
            denied(lambda address=address, stream=stream: stream.bind(str(Path(address[5:]).parent / "rebind-probe.sock")))
    denied(lambda: Path(manifest["canary"]).read_bytes())
    denied(lambda: connect(manifest["denied_tcp"]))
    return ["closed-cpu", "closed-terminal", "unlink", "rebind", "create-denied", "escaped-read", "escaped-tcp"]


if __name__ == "__main__":
    main()
