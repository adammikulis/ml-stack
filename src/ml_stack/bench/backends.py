"""The serving programs a run can be measured against, and what each one reports.

`parse_on` reads a ``--on NAME=URL`` -- ``http://`` is a llama-server, ``ollama://host:port/model``
and ``openai://host/model`` name the program and the model -- and `client_for` builds the
client for it. `served_by` is the record every run carries of what served it: the
program, its version, the weights' format, the runtime and the quantisation, read off the
client when it can say and off ``/props`` and the GGUF name when it is a llama-server.
`describe` and `short` print that record. `processes` is the pids holding the weights.
`timings_of` reads one reply's timings whatever program sent it, with None for every
figure that program does not report.
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from ml_stack.client.http import request_json

# The programs a URL can name ahead of its host, and the api each is spoken to with.
SCHEMES = {"ollama": "ollama", "openai": "openai"}

# Every timing a reply may carry, in the words llama.cpp uses for them.
TIMING_KEYS = ("prompt_ms", "predicted_ms", "prompt_n", "cache_n", "predicted_n", "draft_n",
               "draft_n_accepted", "load_ms")

# Ollama's own names for the same figures, and what each is in: durations in nanoseconds.
_OLLAMA = {"prompt_eval_count": "prompt_n", "eval_count": "predicted_n"}
_OLLAMA_NS = {"prompt_eval_duration": "prompt_ms", "eval_duration": "predicted_ms",
              "load_duration": "load_ms"}


def parse_on(spec: str) -> tuple[str, str, dict[str, Any]]:
    """``(name, url, how)`` from ``NAME=URL``: ``how`` is the ``api`` and ``model`` the URL
    names, empty for a plain http URL."""
    name, _, url = str(spec).partition("=")
    if not name or not url:
        raise ValueError(f"--on wants NAME=URL, got {spec!r}")
    return name.strip(), url.strip(), how_of(url)


def how_of(url: str) -> dict[str, Any]:
    """The ``api`` and ``model`` a URL names, empty for ``http(s)://``."""
    parts = urlsplit(str(url))
    api = SCHEMES.get(parts.scheme.lower())
    if api is None:
        return {}
    model = parts.path.strip("/")
    out: dict[str, Any] = {"api": api}
    if model:
        out["model"] = model
    return out


def http_of(url: str) -> str:
    """The plain ``http://host:port`` a program's URL points at."""
    parts = urlsplit(str(url))
    if parts.scheme.lower() in SCHEMES:
        return f"http://{parts.netloc}"
    return str(url).rstrip("/")


def _accepts(client: Any, name: str) -> bool:
    try:
        params = inspect.signature(client.__init__).parameters
    except (TypeError, ValueError):
        return True
    return name in params or any(p.kind is p.VAR_KEYWORD for p in params.values())


def client_for(url: str, *, client: Any = None, context: int | None = None,
               **settings: Any) -> Any:
    """A client on ``url``: ``Client(url, api=, model=, context=, **settings)`` when the
    client takes those, ``Client(url, **settings)`` when it does not -- and a URL naming
    a program the client cannot speak to is refused by name."""
    if client is None:
        from ml_stack.client import Client

        client = Client
    how = how_of(url)
    asked = dict(settings)
    if how:
        if not _accepts(client, "api"):
            raise ValueError(f"{url}: this client speaks to a llama-server only and cannot "
                             f"take an {how['api']} URL")
        asked.update(how)
    if context is not None and _accepts(client, "context"):
        asked["context"] = int(context)
    # the plain http address with the program and the model said outright, so a client
    # that reads only one scheme off a URL is still told what the others mean
    return client(http_of(url) if how else url, **asked)


def speaks_llama(client: Any) -> bool:
    """Whether the client talks to a llama-server: its ``api`` says so, or it has none."""
    return str(getattr(client, "api", None) or "llama") == "llama"


def served_by(client: Any, base_url: str = "") -> dict[str, Any] | None:
    """What served a run: ``program``, ``version``, ``format``, ``runtime``, ``quant``,
    ``model``, ``weights_bytes`` -- from the client for a program only it can ask, and
    read off ``/props`` and every shard of the GGUF it names for a llama-server. None
    when nothing answers."""
    if hasattr(client, "served_by") and not speaks_llama(client):
        try:
            got = client.served_by()
        except Exception:  # noqa: BLE001 - a record nothing gave is no record
            got = None
        if isinstance(got, Mapping) and got:
            return {key: got.get(key) for key in ("program", "version", "format", "runtime",
                                                   "quant", "model", "weights_bytes")}
    return llama_served_by(base_url or str(getattr(client, "base_url", "") or ""))


def props_of(base_url: str) -> dict[str, Any]:
    """A llama-server's ``/props``, or ``{}`` when nothing on that port answers with one."""
    if not base_url:
        return {}
    try:
        props = request_json(f"{http_of(base_url)}/props", timeout=5.0, method="GET") or {}
    except Exception:  # noqa: BLE001 - nothing on that port
        return {}
    return dict(props) if isinstance(props, Mapping) else {}


def llama_served_by(base_url: str, props: Mapping[str, Any] | None = None
                    ) -> dict[str, Any] | None:
    """A llama-server's record, from ``/props`` (read here unless handed in) and the file
    it names."""
    from ml_stack.client.health import quant_from_model_path

    props = dict(props) if props else props_of(base_url)
    if not props:
        return None
    path = str(props.get("model_path") or "")
    name = path.rsplit("/", 1)[-1]
    out: dict[str, Any] = {
        "program": "llama.cpp",
        "version": str(props.get("build_info") or "") or None,
        "format": "gguf" if name.lower().endswith(".gguf") else None,
        "runtime": None,
        "quant": quant_from_model_path(name) if name else None,
        "model": name or None,
        "weights_bytes": weights_of(path),
    }
    return out


def weights_of(path: str | Path) -> int | None:
    """Every shard of a GGUF on disk, in bytes; None when the file is not here."""
    try:
        where = Path(str(path))
        if not where.is_file():
            return None
        shards = sorted(where.parent.glob(where.name.replace("00001", "*"))) or [where]
        return sum(s.stat().st_size for s in shards if s.is_file())
    except OSError:
        return None


def describe(record: Mapping[str, Any] | None, *, build: str = "") -> str:
    """``ollama 0.33.3 · mlx · nvfp4`` / ``llama.cpp (unsloth) · gguf · Q4_K_XL``: the
    program with its version or its named build, the runtime or the format, the quant."""
    if not record:
        return ""
    program = str(record.get("program") or "")
    if not program:
        return ""
    named = build or str(record.get("build") or "")
    head = program + (f" ({named})" if named else
                      f" {record['version']}" if record.get("version") else "")
    parts = [head]
    runtime, fmt, quant = (record.get(k) for k in ("runtime", "format", "quant"))
    if runtime or fmt:
        parts.append(str(runtime or fmt))
    if quant:
        parts.append(str(quant))
    return " · ".join(parts)


def short(record: Mapping[str, Any] | None) -> str:
    """``ollama·mlx·nvfp4``, one word for a column; ``-`` for a run that kept none."""
    if not record or not record.get("program"):
        return "-"
    bits = [str(record["program"])]
    runtime, fmt, quant = (record.get(k) for k in ("runtime", "format", "quant"))
    if runtime or fmt:
        bits.append(str(runtime or fmt))
    if quant:
        bits.append(str(quant))
    return "·".join(bits).replace(" ", "_")


def processes(client: Any) -> list[int]:
    """The pids holding the weights, as the client says; empty when it cannot."""
    if not hasattr(client, "processes"):
        return []
    try:
        return [int(p) for p in (client.processes() or ())]
    except Exception:  # noqa: BLE001 - a client that cannot say says nothing
        return []


def timings_of(reply: Any) -> dict[str, float | int | None]:
    """One reply's timings under llama.cpp's names, None for each the program did not say.

    A reply carrying ``timings`` is a llama-server's: every figure in it is read, and a
    draft it did not run is 0. A reply carrying Ollama's ``eval_count`` and durations is
    read from those, and it has no cache or draft figure. Anything else says nothing.
    """
    raw = getattr(reply, "raw", None) or {}
    out: dict[str, float | int | None] = {key: None for key in TIMING_KEYS}
    timings = raw.get("timings") if isinstance(raw, Mapping) else None
    if isinstance(timings, Mapping):
        # a key written as null is a figure this program does not measure; a key left
        # out is llama.cpp with nothing to say -- no head, nothing cached -- which is 0
        for key in ("prompt_ms", "predicted_ms", "load_ms"):
            if timings.get(key) is not None:
                out[key] = float(timings[key])
        for key in ("prompt_n", "cache_n", "predicted_n"):
            if timings.get(key) is not None:
                out[key] = int(timings[key])
        for key in ("draft_n", "draft_n_accepted"):
            out[key] = None if (key in timings and timings[key] is None) \
                else int(timings.get(key) or 0)
        if out["cache_n"] is None and "cache_n" not in timings:
            usage = raw.get("usage") or {}
            cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
            out["cache_n"] = int(cached) if cached is not None else 0
        return out
    if isinstance(raw, Mapping) and any(k in raw for k in _OLLAMA):
        for theirs, ours in _OLLAMA.items():
            if raw.get(theirs) is not None:
                out[ours] = int(raw[theirs])
        for theirs, ours in _OLLAMA_NS.items():
            if raw.get(theirs) is not None:
                out[ours] = float(raw[theirs]) / 1e6
        return out
    call = _call_of(reply)
    if call is not None:
        # the client's own record, for a program the bench does not read itself: a field
        # that record leaves None is None here, and one it fills is taken
        for key in TIMING_KEYS:
            got = getattr(call, key, None)
            if got is not None and _reported(call, key):
                out[key] = got
    return out


def _call_of(reply: Any) -> Any | None:
    try:
        from ml_stack.telemetry import Call

        return Call.from_reply(reply, 0.0)
    except Exception:  # noqa: BLE001 - a reply the record cannot read says nothing
        return None


def _reported(call: Any, key: str) -> bool:
    """Whether ``call.key`` is a figure the reply carried: a record whose default for the
    field is None says so by the value; one whose default is 0 cannot, so its zeros are
    not read as measurements."""
    fields = getattr(type(call), "__dataclass_fields__", {})
    field = fields.get(key)
    if field is None:
        return False
    if field.default is None:
        return True
    return bool(getattr(call, key, 0))
