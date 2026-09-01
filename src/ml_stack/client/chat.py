"""One client for a local OpenAI-compatible model server."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from collections.abc import Callable, Sequence
from typing import Any

from ml_stack.client import families
from ml_stack.client.families import Family
from ml_stack.client.health import reported_models
from ml_stack.client.http import ServerError, request_json, request_stream

logger = logging.getLogger(__name__)

_Extractor = tuple[Callable[[Any], str], Callable[[str, list[str]], None]]
"""``(ask(seed) -> raw, reject(raw, objections))`` for one extraction run."""

EXTRACT_INSTRUCTIONS = "Return only a JSON document that matches the schema. No prose."

# Chat templates end a prompt with the opener of the turn the model is to fill in.
_ASSISTANT_OPENERS = (
    "<|im_start|>assistant\n",
    "<|start_header_id|>assistant<|end_header_id|>\n\n",
    "[/INST]",
    "<start_of_turn>model\n",
)

# Sampler keys llama.cpp understands but the hosted OpenAI API rejects outright.
_OPENAI_UNSUPPORTED = ("top_k", "min_p", "typical_p", "repeat_penalty",
                       "repeat_last_n", "mirostat", "mirostat_tau", "mirostat_eta",
                       "n_predict", "cache_prompt", "id_slot", "grammar")


# What each server turned out to be serving, so it is asked once rather than per client.
# Cleared by `forget_families()` when a server is restarted with a different model.
_FAMILY_BY_URL: dict[str, Any] = {}


def forget_families() -> None:
    """Forget which family each server was serving. Call after restarting one."""
    _FAMILY_BY_URL.clear()


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
    return families.split_inline(text)


class Client:
    """A local model server, over HTTP. Standard library only."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080",
        *,
        slot: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        min_p: float | None = None,
        n_predict: int = 16384,
        timeout: float = 180.0,
        tries: int = 1,
        api_key: str | None = None,
        family: Family | str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.slot = slot
        # None means "whatever this model's card asks for", resolved once the family is
        # known; a number given here is the caller overriding its publisher, which is
        # sometimes right — a benchmark wants 0.0 so a run can be repeated — and is never
        # guessed at on their behalf.
        self.asked_temperature = temperature
        self.asked_top_p = top_p
        self.asked_top_k = top_k
        self.asked_min_p = min_p
        # A ceiling, not a budget: nothing is spent that is not generated, so a high one
        # costs nothing and a low one truncates. 512 was set when a reply was a sentence;
        # a thinking model spends most of a turn reasoning before it writes anything, and
        # measured here gemma-4 filled 220 tokens with thought and returned empty content.
        # What a low ceiling cuts is always the answer, never the thinking.
        self.n_predict = n_predict
        self.timeout = timeout
        self.tries = tries
        self.api_key = api_key
        self.pinned_family = families.resolve(family)
        self._probed: Family | None = None

    # --------------------------------------------------------------- request shape

    @property
    def family(self) -> Family:
        """The family this client shapes requests for: the one pinned at construction,
        else the one the model id at ``/v1/models`` names, else generic. Probed once."""
        if self.pinned_family is not None:
            return self.pinned_family
        if self._probed is None:
            # Once per server, not once per client. A served model does not change while the
            # server is up, and callers build a client per request — so probing per client is
            # a round trip per request to be told the same thing again.
            known = _FAMILY_BY_URL.get(self.base_url)
            if known is None:
                known = families.for_model_ids(
                    reported_models(self.base_url, timeout=min(self.timeout, 5.0)))
                _FAMILY_BY_URL[self.base_url] = known
            self._probed = known
        return self._probed

    @property
    def sampling(self) -> dict[str, Any]:
        """What this request will ask for: what the caller chose, and nothing else.

        A model card is deliberately *not* consulted here. It is general advice from a
        publisher who does not know the task, and applying it silently would mean a library
        overruling a caller who measured. gemma-4's card asks for temperature 1.0; on a
        tool-calling task that measured 15 points worse than greedy. The card is worth
        reading — `ml-stack-models card <repo>`, or `Client.card` — and worth trying —
        `ml-stack-bench --card` — but what ships is what the benchmark favoured.

        Greedy by default, because a caller who has not chosen wants the repeatable answer.
        """
        out: dict[str, Any] = {}
        for name, value in (("temperature", self.asked_temperature),
                            ("top_p", self.asked_top_p), ("top_k", self.asked_top_k),
                            ("min_p", self.asked_min_p)):
            if value is not None:
                out[name] = value
        out.setdefault("temperature", 0.0)
        return out

    @property
    def card(self) -> dict[str, Any]:
        """What this model's publisher recommends. A starting point for a benchmark."""
        return dict(self.family.card.asked())

    @property
    def temperature(self) -> float:
        """What this client will send. Kept as a name because callers read it."""
        return float(self.sampling.get("temperature", 0.0))

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
            **self.sampling,
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
        on_delta: Callable[[str, str], None] | None = None,
        **extra: Any,
    ) -> Reply:
        """One chat completion.

        With ``on_delta`` the completion is streamed: it is called with
        ``("thinking", text)`` and ``("content", text)`` as pieces arrive, and the
        assembled Reply is returned as usual.
        """
        body = self.build_body(messages, tools=tools, tool_choice=tool_choice,
                               stream=on_delta is not None, **extra)
        if on_delta is None:
            payload = request_json(
                f"{self.base_url}/v1/chat/completions",
                payload=body,
                timeout=timeout or self.timeout,
                tries=self.tries,
                headers=self._headers(),
            )
            return self.normalize(payload, self.pinned_family)
        chunks = request_stream(
            f"{self.base_url}/v1/chat/completions",
            payload=body,
            timeout=timeout or self.timeout,
            headers=self._headers(),
        )
        assembled = gather_stream(chunks, on_delta, self.pinned_family)
        return self.normalize(assembled, self.pinned_family)

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
            **self.sampling,
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
                n_predict: int | None = None,
                check: Callable[[dict[str, Any]], list[str]] | None = None,
                tries: int = 2, prompt: str | None = None,
                messages: list[dict[str, Any]] | None = None,
                think: bool = False,
                schema_name: str = "extraction") -> dict[str, Any]:
        """``text`` as a JSON document matching ``schema``, re-prompted while ``check``
        objects. Objections left after ``tries`` calls land under ``"_objections"``."""
        if tries < 1:
            raise ValueError(f"tries must be at least 1, got {tries}")

        if prompt is not None:
            ask, reject = self._raw_extractor(prompt, schema, n_predict)
        else:
            ask, reject = self._chat_extractor(
                text, schema, instructions, messages, think, schema_name, n_predict)

        seed: int | None = None
        raw = ""
        parsed: Any = None
        objections: list[str] = []

        for attempt in range(tries):
            raw = ask(seed)
            try:
                answer = json.loads(raw)
            except ValueError:
                objections = ["reply was not valid JSON"]
            else:
                parsed = answer
                objections = list(check(answer)) if check else []
                if not objections:
                    return answer

            if attempt + 1 < tries:
                seed = _fresh_seed(seed)
                reject(raw, objections)

        if parsed is None:
            raise ServerError(
                f"the model returned something that is not JSON {tries} times: {raw[:200]!r}"
            )
        if isinstance(parsed, dict):
            return dict(parsed, _objections=objections)
        return parsed

    def _raw_extractor(self, prompt: str, schema: dict[str, Any],
                       n_predict: int | None) -> _Extractor:
        """``(ask, reject)`` driving ``/completion`` under a grammar built from ``schema``."""
        from ml_stack.contracts.jsonschema import grammar_for

        grammar = grammar_for(schema)
        state = {"prompt": prompt}

        def ask(seed: int | None) -> str:
            extra = {} if seed is None else {"seed": seed}
            return self.complete(state["prompt"], grammar=grammar,
                                 n_predict=n_predict, **extra)

        def reject(raw: str, objections: list[str]) -> None:
            state["prompt"] = _with_rejection(prompt, objections)

        return ask, reject

    def _chat_extractor(self, text: str, schema: dict[str, Any], instructions: str,
                        messages: list[dict[str, Any]] | None, think: bool,
                        schema_name: str, n_predict: int | None) -> _Extractor:
        """``(ask, reject)`` driving ``/v1/chat/completions`` under a JSON schema."""
        convo = list(messages) if messages is not None else [
            {"role": "system", "content": instructions or EXTRACT_INSTRUCTIONS},
            {"role": "user", "content": text},
        ]
        response_format = {
            "type": "json_schema",
            "json_schema": {"name": schema_name, "schema": strict_schema(schema)},
        }

        def ask(seed: int | None) -> str:
            extra: dict[str, Any] = {} if seed is None else {"seed": seed}
            if n_predict is not None:
                extra["n_predict"] = n_predict
            reply = self.chat(
                convo,
                response_format=response_format,
                chat_template_kwargs=self.family.think_kwargs(think),
                **extra,
            )
            return (reply.content or "").strip()

        def reject(raw: str, objections: list[str]) -> None:
            convo.append({"role": "assistant", "content": raw})
            convo.append({"role": "user", "content": _rejection(objections)})

        return ask, reject

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
    def normalize(payload: Any, family: Family | str | None = None) -> Reply:
        """Flatten the first choice, extract tool calls, split off the thinking.

        Without ``family`` the adapter comes from the ``model`` id in the payload.
        """
        if not isinstance(payload, dict):
            raise ServerError(f"unexpected response shape: {type(payload).__name__}")

        choices = payload.get("choices") or [{}]
        choice = choices[0] if choices else {}
        message = choice.get("message") or {}

        served = payload.get("model")
        adapter = families.resolve(family) or families.for_model_id(served)
        tool_calls = adapter.tool_calls(message)
        content, thinking = families.split(adapter, message)
        _warn_if_nothing_read(choice, message, adapter, served, content, thinking, tool_calls)

        return Reply(
            content=content,
            tool_calls=tool_calls,
            finish_reason=choice.get("finish_reason"),
            thinking=thinking,
            raw=payload,
        )


def _warn_if_nothing_read(choice: dict[str, Any], message: dict[str, Any],
                          adapter: Family, served: Any, content: str | None,
                          thinking: str | None,
                          tool_calls: list[dict[str, Any]] | None) -> None:
    """Say so when a reply carried text and nothing came back. Names fields, never text."""
    if (content or "").strip() or (thinking or "").strip() or tool_calls:
        return
    stray = families.unread_text_fields(choice, message, adapter)
    if stray:
        logger.warning(
            "reply came back empty while carrying text in %s, which the %s adapter does "
            "not read; the server reports model %r",
            ", ".join(stray), adapter.name, served,
        )


def gather_stream(chunks: Any, on_delta: Callable[[str, str], None],
                  family: Family | str | None = None) -> dict[str, Any]:
    """Assemble streamed chat chunks into the completion payload they add up to,
    reporting each piece.

    ``on_delta`` is called with ``("thinking", text)`` for reasoning pieces and
    ``("content", text)`` for answer pieces, in arrival order. Without ``family`` the
    adapter comes from the ``model`` id the chunks carry.
    """
    pinned = families.resolve(family)
    content: list[str] = []
    thinking: list[str] = []
    calls: dict[int, dict[str, Any]] = {}
    finish: str | None = None
    model: str | None = None
    adapter: Family | None = pinned
    inline = families.inline_splitter()
    unread: dict[str, list[str]] = {}

    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        if isinstance(chunk.get("model"), str) and not model:
            model = chunk["model"]
        if adapter is None:
            adapter = families.for_model_id(model) if model else families.GENERIC

        choices = chunk.get("choices") or []
        if not choices:
            continue
        choice = choices[0] or {}
        finish = choice.get("finish_reason") or finish
        delta = choice.get("delta") or {}

        for key in adapter.thinking_fields:
            piece = delta.get(key)
            if piece:
                thinking.append(str(piece))
                on_delta("thinking", str(piece))
                break

        piece = delta.get(adapter.content_field)
        if piece:
            content.append(str(piece))
            if adapter.inline_think:
                for channel, text in inline(str(piece)):
                    on_delta(channel, text)
            else:
                on_delta("content", str(piece))

        adapter.tool_delta(calls, delta)

        for key in families.unread_text_fields({}, delta, adapter):
            unread.setdefault(key, []).append(str(delta[key]))

    adapter = adapter or pinned or families.GENERIC
    if adapter.inline_think:
        for channel, text in inline("", final=True):
            on_delta(channel, text)

    message: dict[str, Any] = {"role": "assistant", adapter.content_field: "".join(content)}
    if thinking:
        message[adapter.thinking_fields[0]] = "".join(thinking)
    if calls:
        message["tool_calls"] = [calls[i] for i in sorted(calls)]
    for key, pieces in unread.items():
        message[key] = "".join(pieces)
    assembled: dict[str, Any] = {"choices": [{"message": message, "finish_reason": finish}]}
    if model:
        assembled["model"] = model
    return assembled


def _rejection(objections: list[str]) -> str:
    """The block appended to a prompt to re-ask after a checker refused the answer."""
    lines = "".join(f"- {objection}\n" for objection in objections)
    return f"\n\nYour previous answer was rejected:\n{lines}Return corrected JSON.\n"


def strict_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """A copy where every object lists all its properties as required and admits no
    others, the way ``grammar_for`` already reads a schema."""
    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            out = {k: walk(v) for k, v in node.items()}
            if out.get("type") == "object" and isinstance(out.get("properties"), dict):
                out.setdefault("required", list(out["properties"].keys()))
                out.setdefault("additionalProperties", False)
            return out
        if isinstance(node, list):
            return [walk(v) for v in node]
        return node
    return walk(schema)


def _with_rejection(prompt: str, objections: list[str]) -> str:
    """``prompt`` with the rejection block ahead of any assistant-turn opener."""
    block = _rejection(objections)
    for opener in _ASSISTANT_OPENERS:
        if prompt.endswith(opener):
            return prompt[: -len(opener)] + block + opener
    return prompt + block


def _fresh_seed(previous: Any) -> int:
    """A different seed from last time, without importing ``random``."""
    import time

    base = int(previous) if isinstance(previous, int) and previous >= 0 else 0
    return (base + int(time.monotonic_ns() % 1_000_003) + 1) % 2**31
