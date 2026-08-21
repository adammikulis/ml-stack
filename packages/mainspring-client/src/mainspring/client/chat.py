"""One client for a local OpenAI-compatible model server.

It smooths over the differences between llama.cpp's OpenAI-compatible surface and the
hosted one, and a few local-server behaviours that are easy to get wrong:

* ``id_slot`` + ``cache_prompt``, so a conversation lives in one KV slot and coexists
  with another caller on a different slot.
* ``build_body`` split out from ``chat``, so the request shape is unit-testable with no
  server running.
* A grammar tripwire, and one retry at double the token budget with a fresh seed when a
  constrained generation hits the ceiling mid-structure.
* ``<think>...</think>`` split out of the reply and kept, rather than discarded.
* ``"json"`` normalised to ``"json_object"``, and llama.cpp-only sampler keys dropped
  when the base URL is the hosted API, which rejects them.
* A legacy ``function_call`` lifted up into ``tool_calls``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from mainspring.client.http import ServerError, request_json

# The group is load-bearing: `findall` yields it, so the trace can be kept rather than
# discarded, while `sub` still removes the whole tagged block from the visible content.
_THINK = re.compile(r"<think>(.*?)</think>\s*", re.DOTALL | re.IGNORECASE)

# Sampler keys llama.cpp understands but the hosted OpenAI API rejects outright.
_OPENAI_UNSUPPORTED = ("top_k", "min_p", "typical_p", "repeat_penalty",
                       "repeat_last_n", "mirostat", "mirostat_tau", "mirostat_eta",
                       "n_predict", "cache_prompt", "id_slot", "grammar")


class GrammarBudgetError(ServerError):
    """A grammar-constrained generation ran out of tokens mid-structure.

    Distinct from a plain truncation because the remedy is different: the output is not
    merely short, it is *unparseable*, and retrying with the same budget will produce the
    same unparseable output.
    """


class GrammarUnsupportedError(ServerError):
    """The server ignored a grammar it was given.

    Raised by ``assert_grammar_support``. A server that silently ignores GBNF returns
    plausible free text where structured output was required, which downstream code
    parses into nonsense rather than failing.
    """


@dataclass(frozen=True, slots=True)
class Reply:
    content: str | None
    tool_calls: list[dict[str, Any]] | None = None
    finish_reason: str | None = None
    thinking: str | None = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def truncated(self) -> bool:
        return self.finish_reason == "length"


def strip_thinking(text: str | None) -> tuple[str | None, str | None]:
    """Split ``<think>`` blocks out of a reply. Returns ``(visible, thinking)``.

    Kept rather than discarded: a reasoning trace is the most useful thing in the
    response when a tool call comes back wrong.
    """
    if not text:
        return text, None
    blocks = [block.strip() for block in _THINK.findall(text)]
    if not blocks:
        return text, None
    return _THINK.sub("", text).strip(), "\n".join(blocks)


class Client:
    """A local model server, over HTTP. Standard library only."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080",
        *,
        slot: int | None = None,
        temperature: float = 0.0,
        n_predict: int = 512,
        timeout: float = 180.0,
        tries: int = 1,
        api_key: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.slot = slot
        self.temperature = temperature
        self.n_predict = n_predict
        self.timeout = timeout
        self.tries = tries
        self.api_key = api_key

    # --------------------------------------------------------------- request shape

    @property
    def _is_hosted_openai(self) -> bool:
        return "api.openai.com" in self.base_url

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    def build_body(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str = "auto",
        stream: bool = False,
        response_format: str | dict[str, Any] | None = None,
        **extra: Any,
    ) -> dict[str, Any]:
        """Build the ``/v1/chat/completions`` body.

        Separate from ``chat`` on purpose: this is the part worth testing, and testing it
        must not need a model loaded.
        """
        body: dict[str, Any] = {
            "messages": messages,
            "temperature": self.temperature,
            "n_predict": self.n_predict,
            "stream": stream,
        }
        if self.slot is not None:
            # llama.cpp extension: pin this conversation to one KV slot, and reuse that
            # slot's cache across turns -- which is the entire point of having slots.
            body["id_slot"] = self.slot
            body["cache_prompt"] = True
        if tools:
            body["tools"] = tools
            body["tool_choice"] = tool_choice
        if response_format is not None:
            # llama-server accepts {"type": "json_object"}; callers habitually pass "json".
            if isinstance(response_format, str):
                kind = "json_object" if response_format == "json" else response_format
                body["response_format"] = {"type": kind}
            else:
                body["response_format"] = response_format

        body.update(extra)

        if self._is_hosted_openai:
            body["max_tokens"] = body.pop("n_predict", None)
            for key in _OPENAI_UNSUPPORTED:
                body.pop(key, None)

        return body

    # --------------------------------------------------------------- calls

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str = "auto",
        timeout: float | None = None,
        **extra: Any,
    ) -> Reply:
        """One chat completion."""
        body = self.build_body(messages, tools=tools, tool_choice=tool_choice, **extra)
        payload = request_json(
            f"{self.base_url}/v1/chat/completions",
            payload=body,
            timeout=timeout or self.timeout,
            tries=self.tries,
            headers=self._headers(),
        )
        return self.normalize(payload)

    def complete(
        self,
        prompt: str,
        *,
        grammar: str | None = None,
        n_predict: int | None = None,
        timeout: float | None = None,
        retry_on_budget: bool = True,
        **extra: Any,
    ) -> str:
        """Raw ``/completion``, the endpoint to use when there is no chat template.

        A model whose tokenizer ships no chat template must use this path:
        ``/v1/chat/completions`` would either fail outright or re-wrap the prompt in a
        template the model was never trained on.

        On a grammar-constrained generation that stops at the token ceiling, retries once
        at double the budget with a fresh seed. The fresh seed matters: the same seed
        re-walks the same path into the same ceiling.
        """
        budget = n_predict if n_predict is not None else self.n_predict
        body: dict[str, Any] = {
            "prompt": prompt,
            "temperature": self.temperature,
            "n_predict": budget,
            "stream": False,
        }
        if grammar:
            body["grammar"] = grammar
        if self.slot is not None:
            body["id_slot"] = self.slot
            body["cache_prompt"] = True
        body.update(extra)

        payload = self._completion(body, timeout)
        text = (payload.get("content") or "").strip()

        hit_ceiling = payload.get("stopped_limit") or payload.get("truncated")
        if grammar and hit_ceiling and retry_on_budget:
            retry = dict(body, n_predict=budget * 2, seed=_fresh_seed(body.get("seed")))
            payload = self._completion(retry, timeout)
            text = (payload.get("content") or "").strip()
            if payload.get("stopped_limit"):
                raise GrammarBudgetError(
                    f"grammar-constrained generation still hit the ceiling at "
                    f"{budget * 2} tokens; the output is structurally incomplete"
                )
        return text

    def _completion(self, body: dict[str, Any], timeout: float | None) -> dict[str, Any]:
        payload = request_json(
            f"{self.base_url}/completion",
            payload=body,
            timeout=timeout or self.timeout,
            tries=self.tries,
            headers=self._headers(),
        )
        return payload if isinstance(payload, dict) else {}

    def assert_grammar_support(self) -> None:
        """Fail now if constrained decoding is broken on this server.

        A server that ignores GBNF does not error -- it returns fluent prose where a
        single token was required, and every downstream parse then produces confident
        nonsense. Cheap to check once at startup; very expensive to discover from the
        output.
        """
        grammar = 'root ::= "ok"'
        try:
            answer = self.complete("", grammar=grammar, n_predict=8, retry_on_budget=False)
        except ServerError as exc:
            raise GrammarUnsupportedError(
                f"grammar tripwire could not run against {self.base_url}: {exc}"
            ) from exc
        if answer.strip() != "ok":
            raise GrammarUnsupportedError(
                f"grammar tripwire failed: constrained to the literal 'ok', "
                f"{self.base_url} returned {answer!r}. Constrained decoding is not working."
            )

    # --------------------------------------------------------------- response shape

    @staticmethod
    def normalize(payload: Any) -> Reply:
        """Flatten the first choice, extract tool calls, split off the thinking."""
        if not isinstance(payload, dict):
            raise ServerError(f"unexpected response shape: {type(payload).__name__}")

        choices = payload.get("choices") or [{}]
        choice = choices[0] if choices else {}
        message = choice.get("message") or {}

        tool_calls = message.get("tool_calls") or None
        if not tool_calls and message.get("function_call"):
            # Some llama.cpp builds emit tool calls only in the legacy `function_call`
            # field. Normalise it up so callers see one shape.
            tool_calls = [
                {"id": "call_0", "type": "function", "function": message["function_call"]}
            ]

        content, thinking = strip_thinking(message.get("content"))
        return Reply(
            content=content,
            tool_calls=tool_calls,
            finish_reason=choice.get("finish_reason"),
            thinking=thinking,
            raw=payload,
        )


def _fresh_seed(previous: Any) -> int:
    """A different seed from last time, without importing ``random``.

    Determinism is not wanted here -- the point is to leave the path that hit the ceiling.
    """
    import time

    base = int(previous) if isinstance(previous, int) and previous >= 0 else 0
    return (base + int(time.monotonic_ns() % 1_000_003) + 1) % 2**31
