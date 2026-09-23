"""Serving a model once at ``-lv 4`` to read what it allocated, then stopping it again."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from ml_stack.serve.backend import LlamaServerBackend, ServerFailed
from ml_stack.serve.loadlog import Measured, parse_load_log
from ml_stack.serve.manager import ServerManager

__all__ = ["measure"]


def _load_log(spec, *, backend=None, timeout: float | None = None) -> str:
    """Serve it once, read the log the backend already writes, stop it again.

    The seam `measure` replaces in a test. Nothing is faked here: the same `ServerManager`
    every other caller leases through, so a spec that a load would refuse is refused the
    same way, by the same preflight, before anything is spawned.
    """
    manager = ServerManager(backend or LlamaServerBackend())
    info = manager.lease(spec, timeout=timeout)
    try:
        if info.adopted:
            raise ServerFailed(
                f"{info.base_url} was already serving that model, and an adopted server's "
                "log is from a load that may not have been asked for -lv 4. Stop it "
                "(`ml-stack-serve down --port %d`) and measure again." % info.port)
        if info.log_path is None:
            raise ServerFailed(f"{info.base_url} kept no log to read")
        return Path(info.log_path).read_text(encoding="utf-8", errors="replace")
    finally:
        manager.release(info)


def measure(spec, *, backend=None, timeout: float | None = None,
            serve: Callable[..., str] | None = None) -> Measured:
    """Serve ``spec`` once at `-lv 4`, read what it allocated, and stop it.

    The verbosity is not decoration: every line this reads is an `LLAMA_LOG_INFO` from the
    library, which `common_log_get_verbosity` maps to LOG_LEVEL_TRACE -- so the server's own
    default of 3 prints the server's lines and none of the model's. A measurement taken
    without it comes back empty and truthfully says so.
    """
    if "-lv" not in spec.extra_args:
        spec = replace(spec, extra_args=tuple(spec.extra_args) + ("-lv", "4"))
    text = (serve or _load_log)(spec, backend=backend, timeout=timeout)
    return parse_load_log(text)
