"""Transport-independent typed tool schemas without tool registry imports."""

import inspect
import typing
from collections.abc import Callable
from typing import Any

_JSON_TYPES: dict[Any, dict[str, Any]] = {
    str: {"type": "string"}, int: {"type": "integer"}, float: {"type": "number"},
    bool: {"type": "boolean"},
}


def schema_of(fn: Callable[..., Any]) -> dict[str, Any]:
    """A JSON schema for ``fn``'s keyword arguments, read from its type hints.

    ``str``, ``int``, ``float``, ``bool`` and ``list[str]`` are what the tools take; a
    parameter with no default is required. The same hints are what FastMCP reads, so the
    two transports describe every tool identically.
    """
    hints = typing.get_type_hints(fn)
    props: dict[str, Any] = {}
    required: list[str] = []
    for name, param in inspect.signature(fn).parameters.items():
        hint = hints.get(name, str)
        if type(None) in typing.get_args(hint):
            hint = next(part for part in typing.get_args(hint) if part is not type(None))
        if typing.get_origin(hint) is list:
            inner = typing.get_args(hint)[0] if typing.get_args(hint) else str
            prop: dict[str, Any] = {"type": "array",
                                    "items": dict(_JSON_TYPES.get(inner, {"type": "string"}))}
        else:
            prop = dict(_JSON_TYPES.get(hint, {"type": "string"}))
        if param.default is inspect.Parameter.empty:
            required.append(name)
        elif param.default is not None:
            prop["default"] = param.default
        props[name] = prop
    out: dict[str, Any] = {"type": "object", "properties": props}
    if required:
        out["required"] = required
    return out


