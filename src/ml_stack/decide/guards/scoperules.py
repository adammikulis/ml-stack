"""Deterministic scope check: which paths and hosts in a tool call leave the project."""

from __future__ import annotations

import json
import posixpath
import re
import shlex
import urllib.parse
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from ml_stack.decide.base import Asked, BaseDecider

URL = re.compile(r"[A-Za-z][A-Za-z0-9+.\-]*://[^\s'\"<>|;)]+")
WINDOWS = re.compile(r"^[A-Za-z]:[\\/]")
PATH_KEYS = frozenset({"path", "src", "dst", "file", "dir", "directory", "cwd", "filename",
                       "target", "source", "dest", "destination", "paths"})
SKIP_KEYS = frozenset({"content", "text", "body", "message", "old", "new", "sql", "query",
                       "pattern", "subject"})


def _walk(value: Any, key: str = "") -> Iterable[tuple[str, str]]:
    if isinstance(value, str):
        yield key, value
    elif isinstance(value, Mapping):
        for k, v in value.items():
            yield from _walk(v, str(k))
    elif isinstance(value, Sequence):
        for v in value:
            yield from _walk(v, key)


def path_outside(path: str, project: str) -> bool:
    """Whether ``path``, taken relative to ``project`` unless absolute, leaves it."""
    if path.startswith("~") or WINDOWS.match(path):
        return True
    root = posixpath.normpath(project)
    full = posixpath.normpath(path if path.startswith("/") else posixpath.join(root, path))
    return not (full == root or full.startswith(root + "/"))


def _pathy(token: str) -> bool:
    return (token.startswith(("/", "~", "./", "../")) or token in (".", "..")
            or "/" in token) and "://" not in token and not token.startswith("-")


def _tokens(command: str) -> list[str]:
    try:
        return shlex.split(command)
    except ValueError:
        return command.split()


def violations(arguments: Mapping[str, Any], project: str, hosts: Iterable[str]) -> list[str]:
    """What in ``arguments`` leaves the project directory or the allowed hosts, one line each."""
    allowed = {h.lower() for h in hosts}
    found: list[str] = []
    for key, text in _walk(arguments):
        if key in SKIP_KEYS:
            continue
        for url in URL.findall(text):
            parts = urllib.parse.urlsplit(url)
            host = (parts.hostname or "").lower()
            if parts.scheme not in ("http", "https") or host not in allowed:
                found.append(f"host {host or parts.scheme + '://'} is not allowed")
        bare = URL.sub(" ", text)
        if key in PATH_KEYS:
            candidates = [bare.strip()] if bare.strip() else []
        else:
            candidates = [t for t in _tokens(bare) if _pathy(t)]
        found.extend(f"path {c} is outside {project}" for c in candidates
                     if path_outside(c, project))
    return list(dict.fromkeys(found))


def parse_state(state: str) -> tuple[str, tuple[str, ...], dict[str, Any]]:
    """The project, hosts and call arguments out of a state written by ``states.scope_state``."""
    project = re.search(r"^Project directory: (.*)$", state, re.M)
    hosts = re.search(r"^Allowed hosts: (.*)$", state, re.M)
    call = re.search(r"^Tool call: [\w.\-]+\((.*)\)$", state, re.M | re.S)
    if not (project and hosts and call):
        raise ValueError("not a scope state: needs Project directory, Allowed hosts and Tool call")
    return (project.group(1).strip(),
            tuple(h.strip() for h in hosts.group(1).split(",") if h.strip()),
            json.loads(call.group(1)))


class ScopeDecider(BaseDecider):
    """The scope question answered by `violations`: ``inside`` or ``outside``, with certainty."""

    name = "scope-rules"
    model = "violations"

    def probabilities(self, asked: Asked) -> tuple[list[float], dict[str, Any]]:
        names = [o.name for o in asked.options]
        if set(names) != {"inside", "outside"}:
            raise ValueError("ScopeDecider decides between 'inside' and 'outside'")
        project, hosts, args = parse_state(asked.state)
        bad = violations(args, project, hosts)
        pick = "outside" if bad else "inside"
        return [1.0 if n == pick else 0.0 for n in names], {"matched": True, "violations": bad}
