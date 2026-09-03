"""The model a run reads with: one lease in its measured shape, held throughout."""

from __future__ import annotations

import time
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

__all__ = ["SERVE_EXTRA", "_alive", "_find_model", "_run", "_serving", "_serving_said"]


SERVE_EXTRA: dict[str, Any] = {"timeout": 900.0, "cache_reuse": 256, "warmup": False,
                               "roam": False}
"""What `serve` takes that no `Shape` field names: how long the server is given to come up,
how much of a prompt it may reuse from the last one, that it is not warmed, and that it is
served on the port asked for and no other. The ingest's, not the profile's -- a measurement
of how a model answers says nothing about them."""


def _sampling(args: Any) -> dict[str, Any]:
    """The sampler settings named on the command line, and only those."""
    return {name: value for name, value in
            (("temperature", getattr(args, "temperature", None)),
             ("top_p", getattr(args, "top_p", None)),
             ("top_k", getattr(args, "top_k", None)),
             ("min_p", getattr(args, "min_p", None)))
            if value is not None}


def _said(measured: Any) -> str:
    """One profile as a person reads it. Imported where it is used, so a caller that
    replaced `ml_stack.serve.profile.said` is the one that answers."""
    from ml_stack.serve.profile import said

    return said(measured)


def _run(args: Any, *, resolve: bool = True,
         say: Callable[[str], None] = lambda _line: None) -> tuple[Any, Any]:
    """The whole :class:`~ml_stack.serve.Run` this ingest reads with, and the profile that
    measured it (None when nothing did).

    One object -- the shape to serve the model in, the ways it measured best, the client to
    ask it with -- built here and nowhere else, so the lease that is taken and the serving
    the run record names are the same thing rather than two derivations that drift.

    The command line is laid over the measurement with :meth:`Run.over`, each field going
    to the section that owns it: ``--context`` is the one seat's whole context, ``--n-max``
    the draft's length, ``--per-section`` the cap on one call, ``--n-predict`` the ceiling,
    and the samplers the client's.
    """
    from ml_stack import ingest
    from ml_stack.serve.shape import Run, Shape

    model = str(getattr(args, "model", "") or "")
    found = str(ingest._find_model(model)) if model else ""
    port = int(getattr(args, "serve_port", 8080) or 8080)
    # One slot, one unit at a time. Adam: "we shouldn't be handling parallel requests while
    # extracting. In fact, we should never be splitting the GPU like that" -- and the shelf
    # measured it: one worker read a unit in 86 s, two workers sharing the model averaged
    # 140 s each, slower in aggregate as well as apiece.
    seats = 1
    n_predict = int(getattr(args, "n_predict", None) or 16384)
    timeout = float(getattr(args, "per_section", None) or 300.0)

    measured = None
    if model and getattr(args, "profile", True):
        from ml_stack.serve.profile import profile_for

        measured = profile_for(found)
    if measured is not None:
        run = measured.run(port=port, seats=seats, resolve=resolve,
                           n_predict=n_predict, timeout=timeout)
        say(f"    serving in its measured shape: {_said(measured)}")
    else:
        run = Run(shape=Shape(model=found, port=port, seats=seats)).over(
            n_predict=n_predict, timeout=timeout)

    # The whole --context is the one seat's: a 2,500-token unit with four figures through
    # the projector and a reply of several thousand tokens overran a 16k seat on the first
    # night. A profile that measured a wider seat keeps it.
    run = run.over(seat_context=max(int(getattr(args, "context", 0) or 0),
                                    int(run.shape.seat_context or 0)))
    sampling = _sampling(args)
    if sampling:
        run = run.over(**sampling)
    if not getattr(args, "images", False) and run.shape.mmproj:
        run = run.over(mmproj="")
    if getattr(args, "n_max", None) is not None:
        # Extraction copies definitions out of the page: the head's guesses were accepted
        # 97% of the time on a biology chapter against ~75% answering questions, so the
        # length that measured best for answering is not the length for this. Measured
        # here, per workload, with the same command that reads the shelf.
        if not (run.shape.draft or run.shape.spec_type):
            say("--n-max: no draft head is being served, so there is no draft to lengthen")
        else:
            run = run.over(draft_n_max=int(args.n_max))
            say(f"    draft length {args.n_max} over the profile's")
    return run, measured


@contextmanager
def _serving(args: Any, say: Callable[[str], None] = print) -> Any:
    """A client for the run: one lease, held throughout, in the model's measured shape,
    under the bench's measuring lock so the two never share the GPU.

    The same branch the extract bench takes, and for the same reason: a model served bare
    when a profile measured it with a build, a head and a cache type is a different program
    from the one the measurement was about.
    """
    from ml_stack.graph import bench
    from ml_stack.lock import only_one

    run, _measured = _run(args, say=say)
    with only_one(bench.HOME / "measuring.lock",
                  wait=not getattr(args, "no_queue", False),
                  announce=lambda line: say(f"waiting for the bench -- {line}")):
        if not getattr(args, "model", ""):
            yield run.client(args.base_url)
            return

        from ml_stack.serve.manager import serve

        began = time.time()
        with serve(run.model, manager=run.shape.manager(), **run.lease(),
                   **SERVE_EXTRA) as server:
            say(f"    up in {time.time() - began:.0f}s")
            yield run.client(server.base_url, index=0)


def _alive(client: Any) -> bool:
    """Whether the run's server still answers at all."""
    from ml_stack.client import is_healthy

    base_url = str(getattr(client, "base_url", "") or "")
    return bool(base_url) and is_healthy(base_url, timeout=3.0)


def _serving_said(args: Any) -> str:
    """The measured shape a --model is served in, for the run record.

    Read off the same `Run` `_serving` leases, so a record says what was actually asked for
    -- the seat's context, the draft's length -- and not a second derivation of it that can
    differ from what the server was told.
    """
    if not getattr(args, "model", ""):
        return f"base_url {getattr(args, 'base_url', '')}"
    try:
        run, measured = _run(args, resolve=False)
        lease = run.lease()
        laid = [f"context {lease.get('context')}", f"parallel {lease.get('parallel')}"]
        if lease.get("spec_draft_max") is not None:
            laid.append(f"draft {lease['spec_draft_max']}")
        served = _said(measured) if measured is not None else "bare"
        return f"{served}\n  as served   {', '.join(laid)}"
    except Exception:  # noqa: BLE001 - a record, never a reason not to read
        return "unknown"


def _find_model(named: str) -> str:
    """A model by name, path or ``hf:`` reference, the way every other command finds one."""
    try:
        from ml_stack.graph.bench.serve import find_model
    except ImportError:  # pragma: no cover - the bench's extras are not required here
        return named
    return find_model(named)
