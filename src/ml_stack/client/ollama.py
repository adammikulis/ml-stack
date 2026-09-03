"""Ollama's native API, read into the shapes the rest of the client already uses.

What is relied on, from https://github.com/ollama/ollama/blob/main/docs/api.md:

``POST /api/chat`` takes ``model``, ``messages``, ``tools``, ``think`` (a boolean),
``format`` (``"json"`` or a JSON schema), ``options`` (``num_ctx``, ``num_predict``,
``temperature``, ``top_k``, ``top_p``, ``min_p``, ``seed``), ``keep_alive`` and ``stream``.
The final reply carries ``model``, ``message`` (``content``, ``thinking``, ``tool_calls`` with
``function.name`` and ``function.arguments`` as an object), ``done_reason``, and the durations
``total_duration``, ``load_duration``, ``prompt_eval_duration``, ``eval_duration`` in
nanoseconds with the counts ``prompt_eval_count`` and ``eval_count``. Nothing says how many
prompt tokens were kept from the call before, nor anything about a draft head, so
``cache_n``, ``draft_n`` and ``draft_n_accepted`` are written as ``None`` -- not measured.

``POST /api/show`` with ``{"model": name}`` answers ``details.format``,
``details.quantization_level``, ``details.family``, ``details.parameter_size``,
``model_info`` and ``capabilities``. ``GET /api/tags`` lists ``models[]`` with ``name`` and
``size`` in bytes. ``GET /api/version`` answers ``{"version": ...}``.

``num_ctx`` is only sent when the caller set one: left out, the server uses the model's own
default, which for the Flash-Next tag is 262144 and allocates a cache nobody asked for.
"""

from __future__ import annotations

import json
import re
import sys
from typing import Any

from ml_stack.client.http import request_json

PLATFORM = sys.platform

_QUANT = re.compile(r"^(?:I?Q\d[A-Z0-9_]*|BF16|F16|F32|FP16|FP32)$")

_NS_PER_MS = 1_000_000


def build_body(model: str | None, messages: list[dict[str, Any]], *,
               sampling: dict[str, Any], n_predict: int, context: int | None,
               keep_alive: str | int | None, tools: list[dict[str, Any]] | None,
               think: bool | None, response_format: str | dict[str, Any] | None,
               extra: dict[str, Any]) -> dict[str, Any]:
    """The ``/api/chat`` body for one call. ``extra`` keys that are sampler options go
    under ``options``; anything else is sent as it is."""
    if not model:
        raise ValueError("an Ollama client needs a model tag: Client('ollama://host:port/tag') "
                         "or Client(url, api='ollama', model='tag')")
    options: dict[str, Any] = {**sampling, "num_predict": n_predict}
    if context is not None:
        options["num_ctx"] = int(context)
    rest: dict[str, Any] = {}
    for key, value in extra.items():
        if key in ("id_slot", "cache_prompt", "grammar", "chat_template_kwargs", "tool_choice"):
            continue
        if key == "n_predict":
            options["num_predict"] = value
        elif key in _OPTION_KEYS:
            options[key] = value
        else:
            rest[key] = value
    body: dict[str, Any] = {"model": model, "messages": [outgoing(m) for m in messages],
                            "stream": False, "options": options}
    if tools:
        body["tools"] = tools
    if think is not None:
        body["think"] = bool(think)
    if keep_alive is not None:
        body["keep_alive"] = keep_alive
    if response_format is not None:
        body["format"] = _format(response_format)
    body.update(rest)
    return body


_OPTION_KEYS = frozenset({
    "temperature", "top_k", "top_p", "min_p", "seed", "stop", "repeat_penalty",
    "repeat_last_n", "typical_p", "presence_penalty", "frequency_penalty", "num_keep",
    "num_ctx", "num_predict", "num_batch", "num_gpu", "num_thread"})


def _format(response_format: str | dict[str, Any]) -> Any:
    if isinstance(response_format, str):
        return "json" if response_format in ("json", "json_object") else response_format
    kind = response_format.get("type")
    if kind == "json_schema":
        return (response_format.get("json_schema") or {}).get("schema") or {}
    if kind == "json_object":
        return "json"
    return response_format


def outgoing(message: dict[str, Any]) -> dict[str, Any]:
    """One message as Ollama reads it: tool-call arguments as objects, not JSON strings."""
    calls = message.get("tool_calls")
    if not calls:
        return message
    out = dict(message)
    out["tool_calls"] = [_call_out(one) for one in calls]
    return out


def _call_out(call: dict[str, Any]) -> dict[str, Any]:
    fn = dict(call.get("function") or {})
    args = fn.get("arguments")
    if isinstance(args, str):
        try:
            fn["arguments"] = json.loads(args) if args.strip() else {}
        except ValueError:
            fn["arguments"] = {"_unparsed": args}
    return {**call, "function": fn}


def to_openai(payload: dict[str, Any]) -> dict[str, Any]:
    """An ``/api/chat`` reply as a ``/v1/chat/completions`` payload, with llama.cpp's
    ``timings`` keys filled from the durations and the native reply kept under ``ollama``."""
    message = dict(payload.get("message") or {})
    out_message: dict[str, Any] = {"role": "assistant", "content": message.get("content") or ""}
    if message.get("thinking"):
        out_message["reasoning_content"] = message["thinking"]
    calls = message.get("tool_calls") or []
    if calls:
        out_message["tool_calls"] = [_call_in(i, one) for i, one in enumerate(calls)]
    prompt_n = int(payload.get("prompt_eval_count") or 0)
    predicted_n = int(payload.get("eval_count") or 0)
    return {
        "model": payload.get("model"),
        "choices": [{"index": 0, "message": out_message,
                     "finish_reason": _finish(payload.get("done_reason"), calls)}],
        "usage": {"prompt_tokens": prompt_n, "completion_tokens": predicted_n,
                  "total_tokens": prompt_n + predicted_n},
        "timings": {
            "prompt_ms": _ms(payload.get("prompt_eval_duration")),
            "predicted_ms": _ms(payload.get("eval_duration")),
            "load_ms": _ms(payload.get("load_duration")),
            "total_ms": _ms(payload.get("total_duration")),
            "prompt_n": prompt_n,
            "predicted_n": predicted_n,
            "cache_n": None,
            "draft_n": None,
            "draft_n_accepted": None,
        },
        "ollama": payload,
    }


def _ms(nanoseconds: Any) -> float:
    return float(nanoseconds or 0) / _NS_PER_MS


def _finish(done_reason: Any, calls: list[Any]) -> str:
    if done_reason == "length":
        return "length"
    if calls:
        return "tool_calls"
    return str(done_reason or "stop")


def _call_in(index: int, call: dict[str, Any]) -> dict[str, Any]:
    fn = call.get("function") or {}
    args = fn.get("arguments")
    return {"id": call.get("id") or f"call_{index}", "type": "function",
            "function": {"name": str(fn.get("name") or ""),
                         "arguments": args if isinstance(args, str)
                         else json.dumps(args if args is not None else {})}}


# ------------------------------------------------------------------ what is serving

def runtime_for(format: str | None, platform: str = PLATFORM) -> str:
    """What runs a model of ``format`` on ``platform``: Ollama runs a GGUF in a
    ``llama-server`` child everywhere and a safetensors model only on its MLX runner, which
    exists on macOS alone."""
    if (format or "").lower() == "gguf":
        return "llama.cpp"
    if (format or "").lower() == "safetensors" and platform == "darwin":
        return "mlx"
    return "unknown"


def served_by(base_url: str, model: str | None, *, timeout: float = 5.0) -> dict[str, Any]:
    """``/api/show`` + ``/api/version`` + ``/api/tags`` as one record."""
    shown = request_json(f"{base_url}/api/show", payload={"model": model}, timeout=timeout) \
        if model else {}
    details = (shown or {}).get("details") or {}
    fmt = details.get("format")
    version = (request_json(f"{base_url}/api/version", timeout=timeout) or {}).get("version")
    tags = (request_json(f"{base_url}/api/tags", timeout=timeout) or {}).get("models") or []
    size = next((int(t["size"]) for t in tags
                 if isinstance(t, dict) and t.get("name") == model and t.get("size")), None)
    return {"program": "ollama", "version": version, "format": fmt,
            "runtime": runtime_for(fmt, PLATFORM), "quant": details.get("quantization_level"),
            "model": model, "weights_bytes": size}


def quant_in_name(name: str) -> str | None:
    """The quantisation token in a GGUF filename: ``Q4_K_XL`` out of ``thing-UD-Q4_K_XL.gguf``,
    ``IQ4_XS``, ``BF16``; ``None`` when the name carries none."""
    stem = name.rsplit("/", 1)[-1]
    if stem.lower().endswith(".gguf"):
        stem = stem[:-5]
    for part in reversed(stem.replace(".", "-").split("-")):
        if _QUANT.match(part.upper()):
            return part.upper()
    return None


# ------------------------------------------------------------------ the processes

def listeners(port: int) -> list[Any]:
    """The psutil processes listening on ``port``."""
    import psutil

    found = []
    for process in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            connections = process.net_connections(kind="inet")
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        for conn in connections:
            if conn.status == psutil.CONN_LISTEN and conn.laddr and conn.laddr.port == port:
                found.append(process)
                break
    return found


def processes(port: int) -> list[int]:
    """The pids holding the weights behind an Ollama on ``port``: every descendant of the
    listener -- a ``llama-server`` child for a GGUF, ``ollama runner --mlx-engine`` for a
    safetensors model -- and the listener itself while nothing is loaded."""
    import psutil

    pids: list[int] = []
    for process in listeners(port):
        try:
            children = process.children(recursive=True)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            children = []
        pids.extend(child.pid for child in children)
        if not children:
            pids.append(process.pid)
    return sorted(set(pids))
