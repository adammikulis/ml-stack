"""Rails between a model and the world, as interventions.

`rails` holds the built-in checks (tool policy, secrets, untrusted-text fencing); they need
nothing installed. `default` adds the model-based screen when one is given and anything else
passed as ``extra``. `start` makes the `Run` a loop asks.

    from poolhouse import guard

    run = guard.start(guard.default(), offered=tool_schemas, task="find a model")
    run.check_call(Call("serve_up", {"model": "x.gguf"})).allowed
    run.screen_result(Call("models_find"), tool_result_text).text
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from typing import Any

from poolhouse import log
from poolhouse.guard.loop import parse_call
from poolhouse.guard.policy import ToolPolicyRail, tool_schemas
from poolhouse.guard.secrets import SecretRail
from poolhouse.guard.untrusted import NOTICE, UntrustedRail
from poolhouse.interventions import Call, Confirm, Context, Run
from poolhouse.taint import TaintRail

__all__ = ["BUILTIN", "NOTICE", "Unguarded", "default", "off", "parse_call", "rails", "start"]

logger = logging.getLogger("poolhouse.guard")
logger.addHandler(logging.NullHandler())

BUILTIN = ("untrusted", "secrets", "tool-policy", "taint")


class Unguarded(list):  # type: ignore[type-arg]
    """The empty list `off` returns: the marker that a loop is meant to run without rails.
    ``because`` is the reason it was given."""

    because: str = ""


def rails(*, without: Iterable[str] = (), because: str = "",
          registries: Mapping[str, Callable[[], Iterable[str]]] | None = None) -> list[Any]:
    """The built-in rails, minus the named ones. Dropping any needs a ``because``, and the
    drop is logged and printed. ``registries`` are the lists of ids the taint rail accepts."""
    dropped = tuple(without)
    unknown = [n for n in dropped if n not in BUILTIN]
    if unknown:
        raise ValueError(f"no built-in rail named {unknown}; they are {list(BUILTIN)}")
    if dropped:
        if not because.strip():
            raise ValueError("turning a rail off needs a because=")
        message = f"guard: {', '.join(dropped)} turned off: {because.strip()}"
        logger.warning(message)
        log.warn(message)
    made: dict[str, Any] = {"untrusted": UntrustedRail(), "secrets": SecretRail(),
                            "tool-policy": ToolPolicyRail(),
                            "taint": TaintRail(registries=registries)}
    return [made[n] for n in BUILTIN if n not in dropped]


def default(extra: Sequence[Any] = (), *, screen: Sequence[Any] = (),
            registries: Mapping[str, Callable[[], Iterable[str]]] | None = None) -> list[Any]:
    """The built-in rails, then the model-based ``screen`` (a list of interventions), then
    ``extra`` ones such as a NeMo rail."""
    return [*rails(registries=registries), *screen, *extra]


def off(because: str) -> Unguarded:
    """No rails at all; needs a ``because`` and is logged and printed. It is the only empty
    list an `Agent` accepts as its ``rails``."""
    marker = Unguarded(rails(without=BUILTIN, because=because))
    marker.because = because.strip()
    return marker


def start(items: Sequence[Any], *, offered: Sequence[Mapping[str, Any]] = (), task: str = "",
          confirm: Callable[[Confirm, Call | None], bool | Awaitable[bool]] | None = None,
          notify: Callable[[Confirm, Call | None], Any] | None = None) -> Run:
    """A `Run` over ``items`` for a loop that offers the tools ``offered`` (OpenAI-style
    definitions) to answer the request ``task``."""
    schemas = tool_schemas(offered)
    for one in items:
        if isinstance(one, ToolPolicyRail):
            one.bind(schemas)
    context = Context(task=task, tools=[dict(t) for t in offered])
    return Run(items, context=context, confirm=confirm, notify=notify)
