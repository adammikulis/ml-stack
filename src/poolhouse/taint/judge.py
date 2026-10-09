"""Where each argument of a call came from: safe, traced to untrusted text, or unproven."""

from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

from poolhouse.taint.ledger import Ledger
from poolhouse.taint.sinks import Arg, Sink

__all__ = ["Finding", "addresses", "judge", "schema_for"]

logger = logging.getLogger("poolhouse.guard")
PREVIEW = 60
Registries = Mapping[str, Callable[[], Iterable[str]]]
INTENT = frozenset({"typed", "vouched", "host"})
"""Reasons that show the person asked for the value, not just that it is harmless."""


@dataclass(frozen=True, slots=True)
class Finding:
    """One value in a call's arguments. ``status`` is ``safe`` (``why`` says how), ``proven``
    (repeats untrusted text from ``origins``) or ``unproven`` (made by the model while untrusted
    text was in its context)."""

    arg: str
    status: str
    why: str
    origins: tuple[str, ...] = ()
    preview: str = field(default="", repr=False)
    digest: str = ""


ADDRESS_NAMES = frozenset({"url", "uri", "href", "link", "endpoint", "host", "hostname", "address",
                           "base_url", "server", "website"})
URL = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*://\S+$")


def addresses(arguments: Mapping[str, Any]) -> dict[str, Any]:
    """The arguments of a call that hold an address: a URL anywhere in them, or a string under a
    name such as ``url`` or ``host``."""
    def holds(name: str, value: Any) -> bool:
        if isinstance(value, str):
            return name.lower() in ADDRESS_NAMES or "://" in value
        if isinstance(value, dict):
            return any(holds(str(k), v) or "://" in str(k) for k, v in value.items())
        if isinstance(value, list | tuple):
            return any(holds(name, v) for v in value)
        return False

    return {k: v for k, v in arguments.items() if holds(k, v)}


def schema_for(tools: Iterable[Mapping[str, Any]], name: str) -> Mapping[str, Any]:
    """The ``parameters`` schema of the tool ``name`` among OpenAI-style definitions."""
    for tool in tools:
        fn = tool.get("function") or tool
        if fn.get("name") == name:
            return fn.get("parameters") or fn.get("inputSchema") or {}
    return {}


def _leaves(value: Any, schema: Mapping[str, Any]) -> Iterable[tuple[Any, Mapping[str, Any]]]:
    if isinstance(value, dict):
        props = schema.get("properties") or {}
        for key, item in value.items():
            yield str(key), {}
            yield from _leaves(item, props.get(key) or {})
    elif isinstance(value, list | tuple):
        for item in value:
            yield from _leaves(item, schema.get("items") or {})
    else:
        yield value, schema


def _in_registry(name: str, value: str, registries: Registries) -> bool:
    fn = registries.get(name)
    if fn is None:
        return False
    try:
        return value in set(fn())
    except (OSError, RuntimeError, ValueError, TypeError):
        logger.warning("taint registry %s failed", name, exc_info=True)
        return False


def _on_allowed_host(url: str, registries: Registries) -> bool:
    fn = registries.get("hosts")
    if fn is None:
        return False
    try:
        host = (urlsplit(url).hostname or "").lower()
        return bool(host) and host in {h.lower() for h in fn()}
    except (OSError, RuntimeError, ValueError, TypeError):
        return False


def _within(value: float, low: float | None, high: float | None) -> bool:
    return low is not None and high is not None and low <= value <= high


def _safe(value: Any, rule: Arg, schema: Mapping[str, Any], ledger: Ledger,
          registries: Registries) -> str:
    """Why ``value`` is safe, or an empty string."""
    text = str(value)
    if value is None or isinstance(value, bool):
        return "boolean"
    if rule.inert:
        return "inert"
    if ledger.is_typed(text):
        return "typed"
    if "enum" in schema and value in schema["enum"]:
        return "enum"
    if isinstance(value, int | float):
        low, high = schema.get("minimum"), schema.get("maximum")
        if _within(value, rule.low, rule.high) or _within(value, low, high):
            return "range"
        return ""
    if URL.match(text) and _on_allowed_host(text, registries):
        return "host"
    if rule.validated and ledger.is_vouched(rule.validated, text):
        return "vouched"
    if rule.registry and _in_registry(rule.registry, text, registries):
        return "registry"
    if rule.pattern and re.fullmatch(rule.pattern, text):
        return "pattern"
    return ""


def judge(arguments: Mapping[str, Any], sink: Sink, schema: Mapping[str, Any], ledger: Ledger,
          registries: Registries) -> list[Finding]:
    """A finding for every value in ``arguments``, nested ones and object keys included."""
    props = schema.get("properties") or {}
    found: list[Finding] = []
    for name, given in arguments.items():
        rule = sink.args.get(name, sink.other)
        for leaf, sub in _leaves(given, props.get(name) or {}):
            text = str(leaf)
            digest = hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()[:12]
            preview = re.sub(r"[\x00-\x1f\x7f]", " ", text)[:PREVIEW]
            why = _safe(leaf, rule, sub, ledger, registries)
            if why:
                found.append(Finding(name, "safe", why, (), preview, digest))
                continue
            origins = tuple(ledger.trace(text))
            found.append(Finding(name, "proven" if origins else "unproven",
                                 "repeats untrusted text" if origins else
                                 "made while untrusted text was in the context",
                                 origins, preview, digest))
    return found
