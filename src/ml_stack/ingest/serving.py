"""The model a run reads with: one lease in its measured shape, held throughout."""

from __future__ import annotations

import time
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

__all__ = ["_alive", "_find_model", "_serving", "_serving_said"]


@contextmanager
def _serving(args: Any, say: Callable[[str], None] = print) -> Any:
    """A client for the run: one lease, held throughout, in the model's measured shape.

    The same branch the extract bench takes, and for the same reason: a model served bare
    when a profile measured it with a build, a head and a cache type is a different program
    from the one the measurement was about.
    """
    from ml_stack.client import Client

    sampling = {k: v for k, v in (("temperature", args.temperature), ("top_p", args.top_p),
                                  ("top_k", args.top_k), ("min_p", args.min_p))
                if v is not None}
    if not args.model:
        yield Client(args.base_url, timeout=args.per_section, n_predict=args.n_predict,
                     **sampling)
        return

    from ml_stack import ingest
    from ml_stack.serve.manager import serve

    found = ingest._find_model(args.model)
    # One slot, one unit at a time. Adam: "we shouldn't be handling parallel requests while
    # extracting. In fact, we should never be splitting the GPU like that" -- and the shelf
    # measured it: one worker read a unit in 86 s, two workers sharing the model averaged
    # 140 s each, slower in aggregate as well as apiece. The whole --context is the one
    # seat's: a 2,500-token unit with four figures through the projector and a reply of
    # several thousand tokens overran a 16k seat on the first night.
    seats = 1
    lease: dict[str, Any] = {"port": args.serve_port, "context": int(args.context),
                             "parallel": seats, "timeout": 900.0, "cache_reuse": 256,
                             "warmup": False}
    manager = None
    if getattr(args, "profile", True):
        from ml_stack.serve.profile import profile_for, said

        measured = profile_for(str(found))
        if measured is not None:
            shape = measured.shape(port=args.serve_port, seats=seats)
            lease = {**lease, **{k: v for k, v in shape.lease().items()
                                 if k not in ("port", "parallel")}}
            lease["context"] = max(int(args.context), int(lease.get("context") or 0))
            manager = shape.manager()
            say(f"    serving in its measured shape: {said(measured)}")
    if not args.images:
        lease.pop("mmproj", None)
    if getattr(args, "n_max", None) is not None:
        # Extraction copies definitions out of the page: the head's guesses were accepted
        # 97% of the time on a biology chapter against ~75% answering questions, so the
        # length that measured best for answering is not the length for this. Measured
        # here, per workload, with the same command that reads the shelf.
        if not lease.get("draft"):
            say("--n-max: no draft head is being served, so there is no draft to lengthen")
        else:
            lease["spec_draft_max"] = int(args.n_max)
            say(f"    draft length {args.n_max} over the profile's")
    began = time.time()
    with serve(found, manager=manager, **lease) as server:
        say(f"    up in {time.time() - began:.0f}s")
        yield Client(server.base_url, timeout=args.per_section, n_predict=args.n_predict,
                     **sampling)


def _alive(client: Any) -> bool:
    """Whether the run's server still answers at all."""
    from ml_stack.client import is_healthy

    base_url = str(getattr(client, "base_url", "") or "")
    return bool(base_url) and is_healthy(base_url, timeout=3.0)


def _serving_said(args: Any) -> str:
    """The measured shape a --model is served in, as one line, for the run record."""
    if not getattr(args, "model", ""):
        return f"base_url {getattr(args, 'base_url', '')}"
    try:
        from ml_stack import ingest
        from ml_stack.serve.profile import profile_for, said

        measured = profile_for(str(ingest._find_model(args.model)))
        return said(measured) if measured is not None else "bare"
    except Exception:  # noqa: BLE001 - a record, never a reason not to read
        return "unknown"


def _find_model(named: str) -> str:
    """A model by name, path or ``hf:`` reference, the way every other command finds one."""
    try:
        from ml_stack.graph.bench.serve import find_model
    except ImportError:  # pragma: no cover - the bench's extras are not required here
        return named
    return find_model(named)
