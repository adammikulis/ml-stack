"""The logprob backend: one next-token distribution from any chat server that reports
log-probabilities, read over the letters that label the options."""

from __future__ import annotations

import math
import string
from dataclasses import dataclass
from typing import Any

from ml_stack.client import Client
from ml_stack.decide.base import Asked, BaseDecider
from ml_stack.decide.calibrate import Calibration
from ml_stack.decide.types import DecideError, Option
from ml_stack.http import ServerError, request_json

LETTERS = string.ascii_uppercase
SYSTEM = ("You answer multiple-choice questions about a situation. "
          "Reply with only the letter of the chosen option.")
MIN_MASS = 1e-4


def render(question: str, state: str, options: tuple[Option, ...]) -> str:
    """The user message: the state, the question and the lettered options."""
    lines = [f"{LETTERS[i]}. {o.name}" + (f" - {' '.join(o.description.split())}"
                                          if o.description else "")
             for i, o in enumerate(options)]
    return (f"<state>\n{state}\n</state>\n\nQuestion: {question}\n\nOptions:\n"
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
