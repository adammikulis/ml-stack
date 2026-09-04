"""One client for a model server: llama.cpp, a hosted OpenAI endpoint, or Ollama.

``api`` says which -- ``"llama"`` (the default), ``"openai"`` (inferred from
``api.openai.com``) or ``"ollama"`` (inferred from an ``ollama://host:port/tag`` URL, which
names the model too). One ``chat()`` and one ``Reply`` shape whichever it is.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from collections.abc import Callable, Sequence
from pathlib import Path
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
                       "n_predict", "cache_prompt", "id_slot", "grammar",
                       "chat_template_kwargs")

APIS = ("llama", "openai", "ollama")


def parse_url(base_url: str, api: str | None) -> tuple[str, str, str | None]:
    """``(base_url, api, model)`` from a server URL: ``ollama://host:port/tag`` names all
    three, ``api.openai.com`` names the second, anything else is llama.cpp's."""
    url = base_url.rstrip("/")
    model: str | None = None
    if url.startswith("ollama://"):
        rest = url[len("ollama://"):]
        host, _, tag = rest.partition("/")
        url = f"http://{host}"
        model = tag or None
        api = api or "ollama"
    elif "api.openai.com" in url:
        api = api or "openai"
    api = api or "llama"
    if api not in APIS:
        raise ValueError(f"unknown api {api!r}; one of {', '.join(APIS)}")
    return url, api, model


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
        api: str | None = None,
        model: str | None = None,
        context: int | None = None,
        keep_alive: str | int | None = None,
    ) -> None:
        self.base_url, self.api, found = parse_url(base_url, api)
        # The model tag, where the server wants one named per request (openai, ollama).
        self.model = model or found
        # Ollama's num_ctx. Unset means the server's own default for the model, which for
        # the Flash-Next tag is 262144 -- a cache nobody asked for.
        self.context = context
        self.keep_alive = keep_alive
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
        self._carded: dict[str, Any] | None = None

    # --------------------------------------------------------------- request shape

    @property
    def family(self) -> Family:
        """The family this client shapes requests for: the one pinned at construction,
        else the one the model id at ``/v1/models`` names, else generic. Probed once."""
        if self.pinned_family is not None:
            return self.pinned_family
        if self._probed is None and self.model:
            self._probed = families.for_model_id(self.model)
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
        """What this model's publisher recommends. A starting point for a benchmark.

        Read from the served model's own GGUF when it can be reached -- `general.sampling.*`
        is written into the file, so it cannot drift from the weights and needs no prose
        parsed out of a README. The family's table is the fallback for a server whose model
        is not on this disk.

        It is a recommendation and nothing sends it. `Client.sampling` is what goes out, and
        that is the caller's choice: the publisher does not know the task.
        """
        found = self._from_gguf()
        return found or dict(self.family.card.asked())

    def _from_gguf(self) -> dict[str, Any]:
        """The served model's own recommendation, asked of /props for where it lives."""
        if self._carded is None:
            self._carded = {}
            if self.api != "llama":
                return {}
            try:
                from ml_stack.client.http import request_json
                from ml_stack.hub import in_gguf

                props = request_json(f"{self.base_url}/props", timeout=5.0, method="GET") or {}
                where = str(props.get("model_path") or "")
                if where:
                    self._carded = in_gguf(where)
            except Exception:  # noqa: BLE001 - a server that will not say leaves the family
                self._carded = {}
        return dict(self._carded)

    @property
    def temperature(self) -> float:
        """What this client will send. Kept as a name because callers read it."""
        return float(self.sampling.get("temperature", 0.0))

    @property
    def _is_hosted_openai(self) -> bool:
        return self.api == "openai"

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
        """The request body: ``/v1/chat/completions`` for llama.cpp and OpenAI, ``/api/chat``
        for Ollama."""
        if self.api == "ollama":
            from ml_stack.client import ollama

            return ollama.build_body(self.model, messages, sampling=self.sampling,
                                     n_predict=self.n_predict, context=self.context,
                                     keep_alive=self.keep_alive, tools=tools,
                                     think=extra.pop("think", None),
                                     response_format=response_format, extra=extra)
        body: dict[str, Any] = {
            "messages": messages,
            **self.sampling,
            "n_predict": self.n_predict,
            "stream": stream,
        }
        if self.model:
            body["model"] = self.model
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

        # `think` is the caller's word; the server's is the family's chat-template flag
        # (`enable_thinking` for a qwen template, `reasoning_effort` for harmony). Merged
        # into the body it was an unknown key the server ignored, and Flash-Next thought
        # through every call that asked it not to (2026-09-02, ~400 characters a call).
        think = extra.pop("think", None)
        if think is not None:
            asked = self.family.think_kwargs(bool(think)) if self.family.think_kwargs else {}
            if asked:
                body["chat_template_kwargs"] = {**(extra.pop("chat_template_kwargs", None) or {}),
                                                **asked}
        body.update(extra)

        if self._is_hosted_openai:
            body["max_tokens"] = body.pop("n_predict", None)
            # The hosted API has no template flags; harmony's `reasoning_effort` is the one
            # thinking switch it reads, so only that one survives.
            effort = (body.get("chat_template_kwargs") or {}).get("reasoning_effort")
            for key in _OPENAI_UNSUPPORTED:
                body.pop(key, None)
            if effort is not None:
                body["reasoning_effort"] = effort

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
        if self.api == "ollama" and on_delta is not None:
            raise NotImplementedError(
                "streaming (on_delta) is not implemented for the Ollama api yet; call "
                "chat() without on_delta and read the whole Reply")
        body = self.build_body(messages, tools=tools, tool_choice=tool_choice,
                               stream=on_delta is not None, **extra)
        if self.api == "ollama":
            from ml_stack.client import ollama

            payload = request_json(
                f"{self.base_url}/api/chat",
                payload=body,
                timeout=timeout or self.timeout,
                tries=self.tries,
                headers=self._headers(),
            )
            if not isinstance(payload, dict):
                raise ServerError(f"unexpected response shape: {type(payload).__name__}")
            return self.normalize(ollama.to_openai(payload), self.pinned_family or self.family)
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
        if self.api == "ollama":
            raise NotImplementedError(
                "a raw completion under a grammar is llama.cpp's /completion; the Ollama api "
                "has no grammar -- use chat() with a json_schema response_format")
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
                schema_name: str = "extraction",
                cache_dir: str | Path | None = None, cache_version: str = "",
                cache_extra: str = "") -> dict[str, Any]:
        """``text`` as a JSON document matching ``schema``, re-prompted while ``check``
        objects. Objections left after ``tries`` calls land under ``"_objections"``.

        With ``cache_dir``, an extraction already done is not done again: the answer is kept
        as a file per key under it and read back instead of asking the model. The key is
        ``cache_version`` + the schema + ``text`` + ``cache_extra``, and deliberately *not*
        the instructions -- :mod:`ml_stack.client.cache` says why, and what belongs in
        ``cache_extra``. Only a clean answer is kept: one the model never got past ``check``
        is asked again next run rather than cached as settled.
        """
        if tries < 1:
            raise ValueError(f"tries must be at least 1, got {tries}")

        key: str | None = None
        if cache_dir is not None:
            from ml_stack.client.cache import extraction_key, read_cached

            key = extraction_key(text, schema, version=cache_version, extra=cache_extra)
            done = read_cached(cache_dir, key)
            if done is not None:
                return done

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
                    if key is not None:
                        from ml_stack.client.cache import write_cached

                        write_cached(cache_dir, key, answer)  # type: ignore[arg-type]
                    return answer

            if attempt + 1 < tries:
                seed = _fresh_seed(seed)
                reject(raw, objections)

        if parsed is None:
            if getattr(self, "_last_finish", None) == "length":
                # not a model that cannot write JSON: a reply cut off mid-object, by the
                # context the slot had or by n_predict. Say which knob rather than "not JSON"
                raise ServerError(
                    f"the reply was cut off (finish_reason=length) after {len(raw)} characters "
                    f"and is not whole JSON: the slot's context or n_predict ended it -- serve "
                    f"more context per seat, raise n_predict, or split the text: {raw[:120]!r}",
                    body=raw)     # the whole reply rides on the error, so a caller can keep it
            raise ServerError(
                f"the model returned something that is not JSON {tries} times: {raw[:200]!r}",
                body=raw)
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
            self._last_finish = reply.finish_reason
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

    # --------------------------------------------------------------- what is serving

    def served_by(self) -> dict[str, Any]:
        """What program serves this URL, and what it holds: ``program`` (``llama.cpp`` or
        ``ollama``), ``version``, ``format`` (``gguf``/``safetensors``), ``runtime``
        (``llama.cpp``/``mlx``), ``quant``, ``model`` and ``weights_bytes`` -- ``None`` for
        anything the server does not say."""
        from ml_stack.client import ollama

        if self.api == "ollama":
            return ollama.served_by(self.base_url, self.model, timeout=min(self.timeout, 10.0))
        props = request_json(f"{self.base_url}/props", timeout=min(self.timeout, 10.0),
                             method="GET", headers=self._headers()) or {}
        where = str(props.get("model_path") or "")
        name = where.rsplit("/", 1)[-1] or self.model
        size: int | None = None
        if where:
            try:
                size = Path(where).expanduser().stat().st_size
            except OSError:
                size = None
        return {"program": "llama.cpp", "version": props.get("build_info"),
                "format": "gguf" if where.lower().endswith(".gguf") or not where else None,
                "runtime": "llama.cpp", "quant": ollama.quant_in_name(name) if name else None,
                "model": name or None, "weights_bytes": size}

    def processes(self) -> list[int]:
        """The pids holding the weights behind this URL, for a memory sampler: the
        ``llama-server`` whose command line carries this port, or every process under the
        Ollama listener on it."""
        from ml_stack.client import ollama

        port = _port_of(self.base_url)
        if self.api == "ollama":
            return ollama.processes(port)
        import psutil

        pids: list[int] = []
        for process in psutil.process_iter(["pid", "name", "cmdline"]):
            info = getattr(process, "info", None) or {}
            if "llama-server" not in str(info.get("name") or ""):
                continue
            argv = [str(a) for a in (info.get("cmdline") or [])]
            if not argv or _is_zombie(process):
                continue
            if _port_in(argv) == port:
                pids.append(int(info.get("pid") or process.pid))
        return sorted(pids)

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
    tail: dict[str, Any] = {}

    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        if isinstance(chunk.get("model"), str) and not model:
            model = chunk["model"]
        for key in ("timings", "usage"):
            if isinstance(chunk.get(key), dict):
                tail[key] = chunk[key]
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
    assembled.update(tail)
    return assembled


def _port_of(base_url: str) -> int:
    """The port a server URL names, or the scheme's default."""
    from urllib.parse import urlsplit

    parts = urlsplit(base_url)
    if parts.port:
        return int(parts.port)
    return 443 if parts.scheme == "https" else 80


def _is_zombie(process: Any) -> bool:
    import psutil

    try:
        return process.status() == psutil.STATUS_ZOMBIE
    except (psutil.NoSuchProcess, psutil.AccessDenied, AttributeError):
        return False


def _port_in(argv: list[str]) -> int:
    """The port a ``llama-server`` command line listens on: ``--port N``, else 8080."""
    for i, arg in enumerate(argv):
        if arg == "--port" and i + 1 < len(argv) and argv[i + 1].isdigit():
            return int(argv[i + 1])
        if arg.startswith("--port="):
            tail = arg.split("=", 1)[1]
            if tail.isdigit():
                return int(tail)
    return 8080


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
