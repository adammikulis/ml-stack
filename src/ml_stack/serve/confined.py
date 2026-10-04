"""A model server started under the sandbox: loopback only, the GPU, and nothing readable
beyond its binary and the files it was asked to load."""

from __future__ import annotations

import os
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from ml_stack import sandbox
from ml_stack.sandbox import policies
from ml_stack.sandbox.policy import Net

__all__ = ["ENV", "Confined", "confine", "wanted"]

ENV = "ML_STACK_SANDBOX_SERVE"
KEEP = ("DYLD_LIBRARY_PATH", "LD_LIBRARY_PATH", "LLAMA_SERVER_SLOTS_DEBUG", "GGML_METAL_PATH_RESOURCES")


@dataclass(frozen=True, slots=True)
class Confined:
    """The command and environment that start the server confined, and what to ask the
    backend about a failure."""

    argv: list[str]
    env: dict[str, str]
    tag: str
    backend: sandbox.Backend
    began: float
    cwd: str

    def refusals(self) -> str:
        """The sandbox's refusals since the start, one line each."""
        seen = dict.fromkeys(f"  {d['operation']} {d['target']}"
                             for d in self.backend.denials(self.tag, self.began))
        return "\n".join(["the sandbox refused:", *seen]) if seen else ""


def wanted(explicit: bool | None, environ: Mapping[str, str] | None = None) -> bool:
    """Whether to confine: the explicit choice, else ``ML_STACK_SANDBOX_SERVE``."""
    if explicit is not None:
        return explicit
    return (environ if environ is not None else os.environ).get(ENV, "").lower() in {"1", "yes", "on"}


def confine(argv: Sequence[str], env: Mapping[str, str], binary: Path, *,
            writable: Sequence[str] = ()) -> Confined:
    """``argv`` and ``env`` rewritten to run under the ``model_server`` policy. Every absolute
    path in ``argv`` that exists is readable (its directory, for a file); ``writable`` is the
    only place it may write. Raises `SandboxUnavailable` when nothing can hold the policy."""
    reads: set[str] = set()
    for arg in argv[1:]:
        piece = arg.split("=", 1)[-1]
        if piece.startswith("/") and os.path.lexists(piece):
            real = os.path.realpath(piece)
            reads.add(real if Path(real).is_dir() else str(Path(real).parent))
            if real != piece and not Path(piece).is_dir():
                reads.add(os.path.realpath(str(Path(piece).parent)))
    base = policies.model_server(binary, Path(os.path.realpath(binary)).parent)
    argv = list(argv)
    socket_path = ""
    port = 0
    if sys.platform.startswith("linux"):
        host_index = argv.index("--host")
        port = int(argv[argv.index("--port") + 1])
        socket_dir = os.path.realpath(tempfile.mkdtemp(prefix="ml-stack-server-"))
        socket_path = str(Path(socket_dir) / "model.sock")
        writable = (*writable, socket_dir)
        argv[host_index + 1] = socket_path
    held = sandbox.Policy(
        base.name, read=tuple(sorted({*base.read, *reads, *map(os.path.realpath, writable)})),
        write=tuple(os.path.realpath(w) for w in writable), exec=base.exec,
        net=Net.deny() if socket_path else base.net,
        gpu=True, env={**policies.system_env(), **{k: env[k] for k in KEEP if k in env}},
        limits=base.limits)
    began = time.time()
    made = False
    try:
        wrapped, tag, chosen = sandbox.wrapped(argv, held)
        made = True
    finally:
        if socket_path and not made:
            Path(socket_path).parent.rmdir()
    if chosen is None:
        raise sandbox.SandboxUnavailable("no sandbox backend for the model server")
    if socket_path:
        from ml_stack.serve.socket_relay import arguments

        wrapped = arguments(wrapped, port, socket_path)
    return Confined(wrapped, dict(held.env), tag, chosen, began, str(Path(os.path.realpath(binary)).parent))
