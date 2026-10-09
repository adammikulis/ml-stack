"""`decide(..., backend="auto")`: pick the first backend that is available and can take the
question, and keep the ones that are loaded warm."""

from __future__ import annotations

import functools
import os
import threading
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path

from poolhouse.decide import library, registry
from poolhouse.decide.base import Decider, State
from poolhouse.decide.calibrate import Calibration
from poolhouse.decide.embed import EmbedDecider, Head, server_embedder
from poolhouse.decide.logprob import LETTERS, Chat, LogprobDecider, require_decider_host
from poolhouse.decide.pins import STRANDS_V19
from poolhouse.decide.pointer import PointerDecider
from poolhouse.decide.rules import RulesDecider
from poolhouse.decide.sources import local_source, strands_source
from poolhouse.decide.types import BackendUnavailable, DecideError, Decision, Options, options_of
from poolhouse.home import expand
from poolhouse.http import ServerError, request_json

BACKENDS = ("pointer", "logprob", "embed", "rules")
URL_ENV = "POOLHOUSE_DECIDE_URL"
DEFAULT_URL = "http://127.0.0.1:8080"
LIMITS = {"logprob": len(LETTERS)}


@dataclass(frozen=True, slots=True, eq=False)
class Config:
    """Where each backend finds what it needs.

    ``url`` is a chat server for the logprob backend (default ``$POOLHOUSE_DECIDE_URL`` or
    the local llama-server port). ``embed_url`` and ``embed_head`` enable the embed backend.
    ``pointer`` names a trained decider (registered name or directory) instead of the released
    checkpoint. ``rules`` enables the rules backend. ``order`` is the preference for ``auto``. A config
    is compared by identity: deciders built for it stay warm for as long as it is used.
    """

    backend: str = "auto"
    url: str = ""
    model: str = ""
    token: str = ""
    embed_url: str = ""
    embed_head: str = ""
    pointer: str = ""
    calibration: Calibration | None = None
    rules: RulesDecider | None = None
    download: bool = False
    order: tuple[str, ...] = BACKENDS

    @property
    def chat_url(self) -> str:
        """The chat server to use."""
        return self.url or os.environ.get(URL_ENV, DEFAULT_URL)


def _reachable(url: str, token: str) -> str:
    try:
        require_decider_host(url)
    except DecideError as exc:
        return str(exc)
    try:
        request_json(f"{url.rstrip('/')}/v1/models", timeout=1.5, token=token)
    except ServerError as exc:
        return f"{url} did not answer ({exc})"
    return ""


def _trained(name: str) -> Path:
    """A trained decider's directory: a path, a registered name, or the directory of a
    model the library files under kind ``decision``."""
    if expand(name).is_dir():
        return expand(name)
    try:
        return registry.find(name)
    except DecideError:
        found = library.decider_dir(name)
        if found is None:
            raise
        return found


def unavailable(name: str, config: Config) -> str:
    """Why ``name`` cannot run now, or an empty string when it can."""
    if name == "pointer":
        missing = [m for m in ("torch", "transformers", "peft", "safetensors")
                   if find_spec(m) is None]
        if missing:
            return f"needs {', '.join(missing)} (pip install 'poolhouse[decide-pointer]')"
        try:
            if config.pointer:
                local_source(_trained(config.pointer), download=False)
            else:
                strands_source(download=False)
        except (BackendUnavailable, DecideError) as exc:
            return str(exc)
        return ""
    if name == "logprob":
        return _reachable(config.chat_url, config.token)
    if name == "embed":
        if not (config.embed_url and config.embed_head):
            return "needs an embedding server and a trained head (embed_url, embed_head)"
        if find_spec("numpy") is None:
            return "needs numpy (pip install 'poolhouse[decide]')"
        return _reachable(config.embed_url, config.token)
    if name == "rules":
        return "" if config.rules is not None else "no rules were given"
    return f"unknown backend {name!r}; one of {', '.join(BACKENDS)}"


_WARM: dict[tuple[str, int], tuple[Config, Decider]] = {}
DEFAULT = Config()


@functools.cache
def shared(backend: str = "auto", url: str = "") -> Config:
    """The one `Config` for this backend and server, so callers that name only those share a
    warm decider."""
    return Config(backend=backend, url=url)
_LOCK = threading.Lock()


def build(name: str, config: Config) -> Decider:
    """The decider for backend ``name``; the same one is returned on the next call."""
    with _LOCK:
        held = _WARM.get((name, id(config)))
        if held is not None:
            return held[1]
        made: Decider
        if name == "pointer":
            made = PointerDecider(_trained(config.pointer) if config.pointer else STRANDS_V19,
                                  download=config.download, calibration=config.calibration)
        elif name == "logprob":
            made = LogprobDecider(Chat(config.chat_url, config.model, config.token),
                                  calibration=config.calibration)
        elif name == "embed":
            head = Head.load(Path(config.embed_head))
            require_decider_host(config.embed_url)
            made = EmbedDecider(server_embedder(config.embed_url, token=config.token),
                                head=head, calibration=config.calibration)
        elif name == "rules" and config.rules is not None:
            made = config.rules
        else:
            raise BackendUnavailable(unavailable(name, config) or f"unknown backend {name!r}")
        _WARM[(name, id(config))] = (config, made)
        return made


def choose(count: int, config: Config) -> str:
    """The first backend in ``config.order`` that is available and takes ``count`` options."""
    reasons = []
    for name in config.order:
        if count > LIMITS.get(name, 255):
            reasons.append(f"{name}: at most {LIMITS[name]} options")
            continue
        why = unavailable(name, config)
        if not why:
            return name
        reasons.append(f"{name}: {why}")
    raise BackendUnavailable("no decision backend is available: " + "; ".join(reasons))


def decide(question: str, state: State, options: Options, *,
           abstain_below: float | None = None, config: Config | None = None) -> Decision:
    """Decide with ``config.backend``, or with the first available one for ``"auto"``."""
    cfg = config or DEFAULT
    opts = options_of(options)
    name = choose(len(opts), cfg) if cfg.backend == "auto" else cfg.backend
    return build(name, cfg).decide(question, state, opts, abstain_below=abstain_below)
