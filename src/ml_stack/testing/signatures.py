"""Compare a fake's signature with the real one it stands in for."""

from __future__ import annotations

import inspect
from typing import Any

__all__ = ["drift", "mirrors"]


_VARIADIC = (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)


def _params(obj: Any) -> dict[str, inspect.Parameter]:
    """``obj``'s parameters, less a leading ``self`` -- so an unbound method and a plain
    function that stands in for it compare on what a caller passes."""
    params = list(inspect.signature(obj).parameters.values())
    if params and params[0].name == "self":
        params = params[1:]
    return {p.name: p for p in params}


def drift(fake: Any, real: Any) -> list[str]:
    """Every way ``fake``'s signature fails to mirror ``real``'s; empty when it does.

    A fake may leave out an *optional* parameter of the real one. It may not take a name
    the real one lacks, give a shared name a different kind or default, leave out a required
    one, or take ``*args``/``**kwargs`` the real one does not -- that last is the one that
    lets a wrong keyword through, and is the reason this module exists.
    """
    mine = _params(fake)
    theirs = _params(real)
    out: list[str] = []
    for name, p in mine.items():
        if p.kind in _VARIADIC:
            if not any(q.kind is p.kind for q in theirs.values()):
                out.append(f"takes {p} where the real one takes nothing of the kind")
            continue
        if name not in theirs:
            out.append(f"takes {name!r}, which the real one does not")
        elif theirs[name].kind is not p.kind:
            out.append(f"{name!r} is {p.kind.description}; the real one's is "
                       f"{theirs[name].kind.description}")
        elif theirs[name].default != p.default:
            out.append(f"{name!r} defaults to {p.default!r}; the real one's to "
                       f"{theirs[name].default!r}")
    for name, q in theirs.items():
        if name not in mine and q.kind not in _VARIADIC and q.default is inspect.Parameter.empty:
            out.append(f"leaves out {name!r}, which the real one requires")
    return out


def mirrors(fake: Any, real: Any) -> bool:
    """True when ``fake`` takes what ``real`` takes -- see `drift` for how it may not."""
    return not drift(fake, real)
