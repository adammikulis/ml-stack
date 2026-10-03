"""The logprob backend: one next-token distribution from any chat server that reports
log-probabilities, read over the letters that label the options."""

from __future__ import annotations

import ipaddress
import math
import re
import string
import unicodedata
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

from ml_stack import httpguard
from ml_stack.client import Client
from ml_stack.decide.base import Asked, BaseDecider
from ml_stack.decide.calibrate import Calibration
from ml_stack.decide.types import DecideError, Option
from ml_stack.http import ServerError, request_json

LETTERS = string.ascii_uppercase
SYSTEM = ("You answer multiple-choice questions about a situation. "
          "Reply with only the letter of the chosen option.")
MIN_MASS = 0.5
"""The share of the first token's probability that must sit on the option letters. A model that
puts most of it elsewhere (a flooded or confused prompt) gave no answer, and renormalising what
is left of it would turn noise into a verdict."""
LOOPBACK_NAMES = frozenset({"localhost", "ip6-localhost"})


STATE_TAGS = re.compile(r"</?\s*(?:state|options?|question)\b[^>\n]*>?", re.I)
HEADERS = re.compile(
    r"^([ \t>|]*)((?:options?|question|answer|response|reply|verdict|decision|judge|system|"
    r"assistant|user|human|state|tool result|user request|tool call)\s*:|[A-Za-z][.):]\s)",
    re.I | re.M)
CONTROLS = re.compile(r"[^\S\n\t ]|[\x00-\x08\x0b-\x1f\x7f]")


def closed(text: str) -> str:
    """``text`` that cannot close the ``<state>`` block it is put in: look-alike characters
    folded to the ones they imitate (NFKC), invisible and control characters removed, and the
    prompt's own tags replaced."""
    folded = unicodedata.normalize("NFKC", text)
    folded = "".join(c for c in folded if unicodedata.category(c) != "Cf")
    return STATE_TAGS.sub("[tag removed]", CONTROLS.sub(" ", folded))


def defang(text: str) -> str:
    """``closed`` text from outside, in which a line that starts like the prompt's own labels
    (``Options:``, ``A.``, ``Answer:``) is marked with a leading ``|`` so it cannot pass for
    one."""
    return HEADERS.sub(lambda m: f"{m.group(1)}| {m.group(2)}", closed(text))


def render(question: str, state: str, options: tuple[Option, ...]) -> str:
    """The user message: the state, the question and the lettered options."""
    lines = [f"{LETTERS[i]}. {o.name}" + (f" - {' '.join(o.description.split())}"
                                          if o.description else "")
             for i, o in enumerate(options)]
    return (f"<state>\n{closed(state)}\n</state>\n\nQuestion: {question}\n\nOptions:\n"
            + "\n".join(lines) + "\n\nAnswer with the letter only.")


def letter_probabilities(top: list[dict[str, Any]], count: int) -> list[float]:
    """Probability mass the first token puts on each of the first ``count`` letters.

    A letter's mass is the sum over its spellings (``A``, `` A``, ``a``) among the
    returned candidates; a letter not among them gets none.
    """
    mass = [0.0] * count
    for entry in top:
        token = str(entry.get("token", "")).strip().upper()
        logprob = entry.get("logprob")
        if len(token) == 1 and token in LETTERS[:count] and logprob is not None:
            mass[LETTERS.index(token)] += math.exp(float(logprob))
    return mass


def require_decider_host(url: str) -> None:
    """A decider judges tool calls and what they touch, so it talks only to this machine unless
    the operator has named the host (``ML_STACK_FETCH_ALLOW_HOSTS``, the same list that approves
    other private or remote hosts). Raises `DecideError` otherwise."""
    host = (urlsplit(url).hostname or "").lower()
    if host in LOOPBACK_NAMES or host in httpguard.allowed_hosts():
        return
    try:
        if ipaddress.ip_address(host).is_loopback:
            return
    except ValueError:
        pass
    raise DecideError(f"decider server {host or url!r} is not on this machine; to allow it, name it "
                      f"in {httpguard.ALLOW_ENV} (a remote decider sees every call it judges)")


@dataclass(frozen=True, slots=True)
class Chat:
    """The chat server a `LogprobDecider` asks: its address, model, key and limits."""

    url: str = "http://127.0.0.1:8080"
    model: str = ""
    key: str = ""
    timeout: float = 60.0
    top_logprobs: int = 20
    system: str = SYSTEM


class LogprobDecider(BaseDecider):
    """Asks a served chat model for one token and normalises over the option letters.

    Works with any server that returns ``top_logprobs`` on ``/v1/chat/completions``
    (llama.cpp, vLLM, OpenAI). ``top_logprobs`` must be large enough to catch every letter;
    a server that returns fewer is an error rather than a guess.
    """

    name = "logprob"

    def __init__(self, chat: Chat | str | None = None, *, calibration: Calibration | None = None
                 ) -> None:
        self.chat = Chat(chat) if isinstance(chat, str) else chat or Chat()
        require_decider_host(self.chat.url)
        self.base_url = self.chat.url.rstrip("/")
        self.model = self.chat.model
        self.calibration = calibration
        self._client = Client(self.base_url, model=self.chat.model or None)

    def body(self, asked: Asked) -> dict[str, Any]:
        """The chat-completions request for one question."""
        body: dict[str, Any] = {
            "messages": [{"role": "system", "content": self.chat.system},
                         {"role": "user", "content": render(asked.question, asked.state,
                                                            asked.options)}],
            "max_tokens": 1, "temperature": 0, "logprobs": True,
            "top_logprobs": self.chat.top_logprobs, "cache_prompt": True,
            "chat_template_kwargs": self._client.family.think_kwargs(False)}
        if self.model:
            body["model"] = self.model
        return body

    def probabilities(self, asked: Asked) -> tuple[list[float], dict[str, Any]]:
        count = len(asked.options)
        if count > len(LETTERS):
            raise DecideError(f"the logprob backend labels at most {len(LETTERS)} options, "
                              f"got {count}")
        try:
            reply = request_json(f"{self.base_url}/v1/chat/completions",
                                 payload=self.body(asked), timeout=self.chat.timeout,
                                 token=self.chat.key)
        except ServerError as exc:
            raise DecideError(f"{self.base_url} did not answer: {exc}") from exc
        try:
            top = reply["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
        except (KeyError, IndexError, TypeError) as exc:
            raise DecideError(f"{self.base_url} returned no log-probabilities; the server must "
                              "support `logprobs` on /v1/chat/completions") from exc
        mass = letter_probabilities(top, count)
        total = sum(mass)
        if total < MIN_MASS:
            first = top[0].get("token") if top else None
            raise DecideError("the model's first token was not one of the option letters "
                              f"(it was {first!r})")
        self.model = self.model or str(reply.get("model", ""))
        return mass, {"letter_mass": total}
