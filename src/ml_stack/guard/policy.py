"""Authorization for tool calls: allow-list, argument schema, egress, paths, limits."""

from __future__ import annotations

import ipaddress
import re
import time
from collections import Counter, deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

from ml_stack.interventions import Base, Call, Context, Deny, Proceed, Verdict

__all__ = ["SENSITIVE", "Limits", "ToolPolicyRail", "check_arguments", "tool_schemas"]

SENSITIVE = frozenset({
    "serve_up", "serve_down", "serve_escalate", "models_fetch", "fleet_join", "speech_say",
    "world_make", "bench_run", "bench_standard", "bench_speed", "bench_compare", "bench_animate",
})
"""Tools that start processes, write files or download."""

URL = re.compile(r"(?i)\b(?:https?|ftp|wss?)://[^\s\"'<>)]+")
SENSITIVE_PATH = re.compile(
    r"(?i)(?:^|[\s\"'=:/\\~])(?:\.ssh|\.aws|\.gnupg|\.netrc|\.pypirc|\.git-credentials|\.docker/config"
    r"|id_rsa|id_ed25519|\.env(?:\.\w+)?|authorized_keys|known_hosts)(?:$|[\s\"'/\\])"
    r"|/etc/(?:passwd|shadow|sudoers)|/proc/self/environ|\bKeychains?\b")
TRAVERSAL = re.compile(r"(?:^|[\\/])\.\.(?:[\\/]|$)")
CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


@dataclass(frozen=True)
class Limits:
    """Ceilings a run holds to. A call over any of them is denied."""

    calls: int | None = None
    per_minute: int = 120
    repeats: int = 10
    string: int = 8192
    items: int = 64


@dataclass(eq=False)
class ToolPolicyRail(Base):
    """Denies a call unless its tool was offered, its arguments fit the offered schema, and it
    stays inside the limits; text-bearing arguments may not reach outside this machine or
    name a credential file."""

    name = "tool-policy"
    schemas: Mapping[str, dict[str, Any]] = field(default_factory=dict)
    limits: Limits = field(default_factory=Limits)
    allow_hosts: frozenset[str] = frozenset()
    clock: Callable[[], float] = time.monotonic
    open_world: bool = False

    def __post_init__(self) -> None:
        self.count = 0
        self.seen: Counter[str] = Counter()
        self.recent: deque[float] = deque()

    def bind(self, schemas: Mapping[str, dict[str, Any]]) -> None:
        """Take the tools a run offers; a call to any other is denied."""
        self.schemas = dict(schemas)

    def before_tool_call(self, call: Call, context: Context) -> Verdict:
        problem = self._refusal(call)
        if problem:
            return Deny(problem, self.name)
        self.count += 1
        self.recent.append(self.clock())
        return Proceed()

    def _refusal(self, call: Call) -> str:
        return self._shape(call) or self._limits(call) or self._content(call)

    def _shape(self, call: Call) -> str:
        if call.arguments is None:
            return f"arguments of {call.name} are not a JSON object"
        if not self.open_world and call.name not in self.schemas:
            return f"{call.name} is not a tool on offer"
        schema = self.schemas.get(call.name)
        return check_arguments(schema, call.arguments, call.name) if schema else ""

    def _limits(self, call: Call) -> str:
        lim = self.limits
        if lim.calls is not None and self.count >= lim.calls:
            return f"more than {lim.calls} tool calls in one run"
        now = self.clock()
        while self.recent and now - self.recent[0] > 60:
            self.recent.popleft()
        if len(self.recent) >= lim.per_minute:
            return f"more than {lim.per_minute} tool calls in a minute"
        key = f"{call.name}:{sorted((call.arguments or {}).items(), key=repr)!r}"
        self.seen[key] += 1
        if self.seen[key] > lim.repeats:
            return f"{call.name} called with the same arguments more than {lim.repeats} times"
        return ""

    def _content(self, call: Call) -> str:
        lim = self.limits
        for text in _leaves(call.arguments):
            if len(text) > lim.string:
                return f"an argument of {call.name} is longer than {lim.string} characters"
            if CONTROL.search(text):
                return f"an argument of {call.name} holds control characters"
            if TRAVERSAL.search(text):
                return f"an argument of {call.name} climbs out of its directory with .."
            if SENSITIVE_PATH.search(text):
                return f"an argument of {call.name} names a credential file"
            host = self._foreign_host(text)
            if host:
                return f"an argument of {call.name} reaches {host}, which is not this machine"
        return ""

    def _foreign_host(self, text: str) -> str:
        for url in URL.findall(text):
            try:
                host = (urlsplit(url).hostname or "").lower()
            except ValueError:
                return url[:60]
            if host and host not in self.allow_hosts and not _loopback(host):
                return host
        return ""


def _loopback(host: str) -> bool:
    if host == "localhost" or host.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _leaves(value: object) -> Iterable[str]:
    """Every string in a JSON value, keys included."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for k, v in value.items():
            yield str(k)
            yield from _leaves(v)
    elif isinstance(value, list | tuple):
        for v in value:
            yield from _leaves(v)


def tool_schemas(offered: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """``{name: parameters schema}`` from OpenAI-style tool definitions."""
    out: dict[str, dict[str, Any]] = {}
    for item in offered:
        fn = item.get("function") or {}
        if fn.get("name"):
            out[str(fn["name"])] = dict(fn.get("parameters") or {})
    return out


TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,), "integer": (int,), "number": (int, float), "boolean": (bool,),
    "array": (list,), "object": (dict,),
}


def _fits(value: object, schema: Mapping[str, Any]) -> bool:
    kind = schema.get("type")
    if kind is None:
        return True
    if kind in ("integer", "number") and isinstance(value, bool):
        return False
    if not isinstance(value, TYPES.get(str(kind), (object,))):
        return False
    if kind == "array" and isinstance(schema.get("items"), dict):
        return all(_fits(v, schema["items"]) for v in value)
    return "enum" not in schema or value in schema["enum"]


def check_arguments(schema: Mapping[str, Any], arguments: Mapping[str, Any], name: str) -> str:
    """Why ``arguments`` do not fit ``schema`` (an object schema), or empty when they do."""
    props = schema.get("properties") or {}
    for key in arguments:
        if key not in props:
            return f"{name} takes no argument {key!r}"
    for key in schema.get("required") or ():
        if key not in arguments:
            return f"{name} needs the argument {key!r}"
    for key, value in arguments.items():
        if not _fits(value, props[key]):
            return f"argument {key!r} of {name} is not a {props[key].get('type', 'valid value')}"
    return ""
