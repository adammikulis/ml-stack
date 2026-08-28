"""One client for a local OpenAI-compatible model server."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from collections.abc import Sequence
from typing import Any

from ml_stack.client.http import ServerError, request_json

_THINK = re.compile(r"<think>(.*?)</think>\s*", re.DOTALL | re.IGNORECASE)

# Sampler keys llama.cpp understands but the hosted OpenAI API rejects outright.
_OPENAI_UNSUPPORTED = ("top_k", "min_p", "typical_p", "repeat_penalty",
                       "repeat_last_n", "mirostat", "mirostat_tau", "mirostat_eta",
                       "n_predict", "cache_prompt", "id_slot", "grammar")


class GrammarBudgetError(ServerError):
    """A grammar-constrained generation ran out of tokens mid-structure."""


class GrammarUnsupportedError(ServerError):
    """The server ignored a grammar it was given."""


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
    """Split ``<think>`` blocks out of a reply. Returns ``(visible, thinking)``."""
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
        """Build the ``/v1/chat/completions`` body."""
        body: dict[str, Any] = {
            "messages": messages,
            "temperature": self.temperature,
            "n_predict": self.n_predict,
            "stream": stream,
        }
        if self.slot is not None:
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
        """Raw ``/completion``, the endpoint to use when there is no chat template."""
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

    def extract(self, text: str, schema: dict[str, Any], *, instructions: str = "",
                n_predict: int | None = None) -> dict[str, Any]:
        """``text`` as a JSON document matching ``schema``. Retries once on bad JSON."""
        from ml_stack.contracts.jsonschema import grammar_for

        grammar = grammar_for(schema)
        prompt = f"{instructions}\n\nText:\n{text}\n\nJSON:\n"

        raw = self.complete(prompt, grammar=grammar, n_predict=n_predict)
        try:
            return json.loads(raw)
        except ValueError:
            pass

        raw = self.complete(prompt, grammar=grammar, n_predict=n_predict,
                            seed=_fresh_seed(None))
        try:
            return json.loads(raw)
        except ValueError:
            raise ServerError(
                f"twice the model returned something that is not JSON: {raw[:200]!r}"
            ) from None

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
        """Fail now if constrained decoding is broken on this server."""
        grammar = 'root ::= "ok"'
        try:
            answer = self.complete("Reply:", grammar=grammar, n_predict=8, retry_on_budget=False)
        except ServerError as exc:
            raise GrammarUnsupportedError(
                f"grammar tripwire could not run against {self.base_url}: {exc}"
            ) from exc
        if answer.strip() != "ok":
            raise GrammarUnsupportedError(
                f"grammar tripwire failed: constrained to the literal 'ok', "
                f"{self.base_url} returned {answer!r}. Constrained decoding is not working."
            )

    def tokenize(self, text: str, *, with_pieces: bool = False) -> list[Any]:
        """Token ids the SERVER assigns to ``text``, via llama.cpp's /tokenize."""
        payload = request_json(
            f"{self.base_url}/tokenize",
            payload={"content": text, "with_pieces": with_pieces},
            timeout=self.timeout,
            tries=self.tries,
            headers=self._headers(),
        )
        tokens = payload.get("tokens") if isinstance(payload, dict) else None
        return list(tokens) if tokens is not None else []

    def detokenize(self, tokens: Sequence[int]) -> str:
        """The round trip. A tokenizer can encode consistently and still fail to"""
        payload = request_json(
            f"{self.base_url}/detokenize",
            payload={"tokens": list(tokens)},
            timeout=self.timeout,
            tries=self.tries,
            headers=self._headers(),
        )
        content = payload.get("content") if isinstance(payload, dict) else None
        return str(content or "")

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
    """A different seed from last time, without importing ``random``."""
    import time

    base = int(previous) if isinstance(previous, int) and previous >= 0 else 0
    return (base + int(time.monotonic_ns() % 1_000_003) + 1) % 2**31
