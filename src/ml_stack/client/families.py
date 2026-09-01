"""What differs between model families in an OpenAI-compatible chat reply.

A ``Family`` says where the answer lives, where the thinking lives, whether thinking is
delimited in-band, how tool calls arrive, and what a request sends to turn thinking on
or off. ``normalize`` and ``gather_stream`` both read a reply through one.

The only family signal an OpenAI-compatible endpoint offers is the model id it reports:
in ``model`` on every completion and chunk, and at ``/v1/models``.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

THINK = re.compile(r"<think>(.*?)</think>\s*", re.DOTALL | re.IGNORECASE)

_OPEN = "<think>"
_CLOSE = "</think>"


# ------------------------------------------------------------------ tool calls

def openai_tool_calls(message: Mapping[str, Any]) -> list[dict[str, Any]] | None:
    """Tool calls off a message, lifting a legacy ``function_call`` into the modern shape."""
    calls = message.get("tool_calls") or None
    if not calls and message.get("function_call"):
        return [{"id": "call_0", "type": "function", "function": message["function_call"]}]
    return calls


def openai_tool_delta(calls: dict[int, dict[str, Any]], delta: Mapping[str, Any]) -> None:
    """Fold one streamed delta's tool-call fragments into ``calls``, keyed by index."""
    for one in delta.get("tool_calls") or []:
        slot = calls.setdefault(int(one.get("index") or 0), {
            "id": "", "type": "function", "function": {"name": "", "arguments": ""}})
        if one.get("id"):
            slot["id"] = one["id"]
        function = one.get("function") or {}
        if function.get("name"):
            slot["function"]["name"] = function["name"]
        if function.get("arguments"):
            slot["function"]["arguments"] += function["arguments"]


# ------------------------------------------------------------------ think switches

def enable_thinking(on: bool) -> dict[str, Any]:
    """``chat_template_kwargs`` for a template with an ``enable_thinking`` flag."""
    return {"enable_thinking": on}


def reasoning_effort(on: bool) -> dict[str, Any]:
    """``chat_template_kwargs`` for a harmony template, which has no thinking flag and
    reads ``reasoning_effort`` instead."""
    return {"reasoning_effort": "high" if on else "low"}


# ------------------------------------------------------------------ the family

@dataclass(frozen=True, slots=True)
class Sampling:
    """The sampler settings a model's own card asks for. **Informational, not applied.**

    A card is general advice from a publisher who does not know what you are doing with the
    model. gemma-4's asks for temperature 1.0 / top_p 0.95 / top_k 64 "across all use cases";
    measured on a graph-answering task where the model must call tools with exact ids, that
    scored 55% against 70% greedy and took 258s against 140s, because sampling noise there
    becomes a wrong argument rather than a livelier sentence.

    So this is where a benchmark *starts*, never what a client silently sends. Read it with
    `ml-stack-models card <repo>`, try it with `ml-stack-bench --card`, and ship whatever the
    measurement actually favoured.

    Not a house style and not a guess: only what the publisher wrote down. A field left None
    is one the card did not speak about, and nothing here invents a number to fill it —
    gpt-oss's cards say nothing about sampling at all.
    """

    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    why: str = ""                  # where these came from, for anyone who doubts them

    def asked(self) -> dict[str, Any]:
        """Only the settings the card actually named."""
        named = {"temperature": self.temperature, "top_p": self.top_p,
                 "top_k": self.top_k, "min_p": self.min_p}
        return {k: v for k, v in named.items() if v is not None}


@dataclass(frozen=True, slots=True)
class Family:
    """How one model family shapes a chat reply, and what a request tells it about thinking."""

    name: str
    model_ids: tuple[str, ...] = ()
    content_field: str = "content"
    thinking_fields: tuple[str, ...] = ("reasoning_content", "reasoning")
    inline_think: bool = True
    tool_calls: Callable[[Mapping[str, Any]], list[dict[str, Any]] | None] = openai_tool_calls
    tool_delta: Callable[[dict[int, dict[str, Any]], Mapping[str, Any]], None] = openai_tool_delta
    think_kwargs: Callable[[bool], dict[str, Any]] = enable_thinking
    # what the publisher recommends, for a benchmark to start from -- never sent
    # by a client on its own; see `Sampling`
    card: Sampling = field(default_factory=Sampling)


GENERIC = Family(
    name="generic",
    thinking_fields=("reasoning_content", "reasoning"),
    inline_think=True,
    think_kwargs=enable_thinking,
)

GPT_OSS = Family(
    name="gpt-oss",
    model_ids=("gpt-oss", "gpt_oss", "gptoss", "harmony"),
    thinking_fields=("reasoning_content", "reasoning"),
    inline_think=False,
    think_kwargs=reasoning_effort,
)

QWEN = Family(
    name="qwen",
    model_ids=("qwen",),
    thinking_fields=("reasoning_content", "reasoning"),
    inline_think=True,
    think_kwargs=enable_thinking,
)

GEMMA = Family(
    name="gemma",
    model_ids=("gemma",),
    thinking_fields=("reasoning_content", "reasoning"),
    inline_think=True,
    think_kwargs=enable_thinking,
    # "Use the following standardized sampling configuration across all use cases" --
    # the gemma-4 card, which names all three. Read it again with
    # `ml-stack-models card unsloth/gemma-4-E4B-it-qat-GGUF` rather than trusting this line.
    card=Sampling(temperature=1.0, top_p=0.95, top_k=64,
                  why="gemma-4 card, Best Practices / Sampling Parameters"),
)

KNOWN: tuple[Family, ...] = (GPT_OSS, QWEN, GEMMA)


def by_name(name: str) -> Family:
    """The family called ``name``. Raises ``ValueError`` for a name nothing answers to."""
    wanted = name.strip().lower().replace("_", "-")
    for family in (*KNOWN, GENERIC):
        if family.name == wanted:
            return family
    known = ", ".join(f.name for f in (*KNOWN, GENERIC))
    raise ValueError(f"unknown model family {name!r}; known families are {known}")


def for_model_id(model_id: Any) -> Family:
    """The family a served model id belongs to, or ``GENERIC`` when nothing matches."""
    if not isinstance(model_id, str):
        return GENERIC
    bare = model_id.rsplit("/", 1)[-1].lower()
    for family in KNOWN:
        if any(needle in bare for needle in family.model_ids):
            return family
    return GENERIC


def for_model_ids(model_ids: Any) -> Family:
    """The first recognised family among several served model ids."""
    for one in model_ids or ():
        family = for_model_id(one)
        if family is not GENERIC:
            return family
    return GENERIC


def resolve(family: Family | str | None) -> Family | None:
    """A ``Family`` from a family, a family name, or ``None``."""
    if family is None or isinstance(family, Family):
        return family
    return by_name(family)


# ------------------------------------------------------------------ reading a reply

def split_inline(text: str | None) -> tuple[str | None, str | None]:
    """Split ``<think>`` blocks out of a reply. Returns ``(visible, thinking)``."""
    if not text:
        return text, None
    blocks = [block.strip() for block in THINK.findall(text)]
    if not blocks:
        return text, None
    return THINK.sub("", text).strip(), "\n".join(blocks)


def split(family: Family, message: Mapping[str, Any]) -> tuple[str | None, str | None]:
    """``(content, thinking)`` for one assistant message, by ``family``'s conventions."""
    content = message.get(family.content_field)
    thinking: str | None = None
    if family.inline_think:
        content, thinking = split_inline(content)

    reasoned = ""
    for key in family.thinking_fields:
        reasoned = str(message.get(key) or "").strip()
        if reasoned:
            break
    if reasoned:
        thinking = f"{reasoned}\n{thinking}" if thinking else reasoned
    return content, thinking


def unread_text_fields(choice: Mapping[str, Any], message: Mapping[str, Any],
                       family: Family) -> list[str]:
    """Names of the string fields in a choice that hold text ``family`` does not read."""
    read = {"role", family.content_field, *family.thinking_fields}
    names = [key for key, value in message.items()
             if key not in read and isinstance(value, str) and value.strip()]
    names += [key for key, value in choice.items()
              if key not in ("finish_reason", "index") and isinstance(value, str)
              and value.strip()]
    return names


# ------------------------------------------------------------------ reading a stream

def inline_splitter() -> Callable[..., list[tuple[str, str]]]:
    """A ``feed(text, final=False) -> [(channel, text)]`` that routes streamed text either
    side of ``<think>`` markers, holding back a tail that could still grow into one."""
    state: dict[str, Any] = {"buffer": "", "inside": False}

    def feed(text: str, final: bool = False) -> list[tuple[str, str]]:
        state["buffer"] += text
        out: list[tuple[str, str]] = []
        while True:
            buffer = state["buffer"]
            marker = _CLOSE if state["inside"] else _OPEN
            at = buffer.lower().find(marker)
            if at >= 0:
                head = buffer[:at]
                if head:
                    out.append(("thinking" if state["inside"] else "content", head))
                state["buffer"] = buffer[at + len(marker):]
                state["inside"] = not state["inside"]
                continue
            hold = 0 if final else _marker_tail(buffer)
            head = buffer[: len(buffer) - hold] if hold else buffer
            if head:
                out.append(("thinking" if state["inside"] else "content", head))
            state["buffer"] = buffer[len(buffer) - hold:] if hold else ""
            return out

    return feed


def _marker_tail(buffer: str) -> int:
    """How many characters at the end of ``buffer`` could still become a think marker."""
    low = buffer.lower()
    for size in range(min(len(low), len(_CLOSE)), 0, -1):
        tail = low[-size:]
        if _OPEN.startswith(tail) or _CLOSE.startswith(tail):
            return size
    return 0
