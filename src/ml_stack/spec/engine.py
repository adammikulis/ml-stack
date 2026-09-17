"""A loaded target, its drafter and a persistent session: one ``chat`` per completion."""

from __future__ import annotations

import json
import threading
import uuid
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx_lm import load

from ml_stack import home
from ml_stack.spec.accept import Rule
from ml_stack.spec.cost import load_curve
from ml_stack.spec.decode import Asked, Decoded, Pass, Session, decode
from ml_stack.spec.drafters import Budget, Drafter
from ml_stack.spec.drafters.dflash import DFlashDrafter, calibration_for, load_dflash
from ml_stack.spec.drafters.mtp import MtpDrafter, load_head
from ml_stack.spec.drafters.ngram import NgramDrafter
from ml_stack.spec.layout import Layout, layout_for
from ml_stack.spec.sample import Sampling

__all__ = ["DRAFTERS", "Engine", "EngineConfig", "Reply", "Request", "weights"]

DRAFTERS = ("dflash", "mtp", "ngram", "none")

#: the model's own recommendations: thinking, then answering directly
SAMPLING = {True: Sampling(1.0, 20, 0.95), False: Sampling(0.7, 20, 0.8)}


def weights(name: str | Path) -> Path:
    """A local weights directory, or a Hub repository id brought into the Hub cache."""
    where = home.expand(name)
    if where.is_dir():
        return where
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(str(name)))


@dataclass(frozen=True)
class EngineConfig:
    """What to load: the target, the drafter kind and its weights, and the tree budget."""

    model: str
    drafter: str = "ngram"
    drafter_model: str = ""
    max_nodes: int = 32
    keep: int = 2
    recalibrate: bool = False


@dataclass(frozen=True)
class Request:
    """How one completion is asked for; ``sampling`` None takes the model's recommendation."""

    max_tokens: int = 4096
    sampling: Sampling | None = None
    thinking: bool = True
    rule: Rule = field(default_factory=Rule)
    seed: int | None = None
    template: Mapping[str, Any] = field(default_factory=dict)
    tools: Sequence[Mapping[str, Any]] = ()


@dataclass
class Reply:
    """The answer, the reasoning before it, and what decoding it took."""

    content: str
    reasoning: str
    decoded: Decoded
    reasoning_tokens: int = 0
    phases: dict[str, list[Pass]] = field(default_factory=dict)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


def make_drafter(config: EngineConfig, layout: Layout, budget: Budget) -> Drafter:
    """The drafter ``config`` names, loaded against ``layout``."""
    if config.drafter == "dflash":
        path = weights(config.drafter_model)
        model, dflash = load_dflash(path)
        return DFlashDrafter(layout, model, dflash, budget,
                             calibration=calibration_for(path.name))
    if config.drafter == "mtp":
        return MtpDrafter(layout, load_head(weights(config.drafter_model), layout.args), budget)
    if config.drafter in ("ngram", "none"):
        return NgramDrafter(budget, depth=8 if config.drafter == "ngram" else 0)
    raise ValueError(f"unknown drafter {config.drafter!r}; choose from {', '.join(DRAFTERS)}")


class Engine:
    """Decoding runs on one worker thread, one completion at a time, whatever thread asks."""

    def __init__(self, config: EngineConfig) -> None:
        self._worker = ThreadPoolExecutor(max_workers=1)
        self._worker.submit(self._load, config).result()

    def _load(self, config: EngineConfig) -> None:
        """Load on the worker thread, which is the thread MLX's streams belong to."""
        mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
        path = weights(config.model)
        self.model, self.tokenizer = load(str(path))
        self.layout = layout_for(self.model)
        curve = load_curve(self.layout, path.name, again=config.recalibrate)
        budget = Budget(max_nodes=min(32, max(1, config.max_nodes)), cost=curve)
        self.drafter = make_drafter(config, self.layout, budget)
        self.session = Session(self.layout, self.drafter, keep=config.keep)
        self.eos = set(self.tokenizer.eos_token_ids)
        self.think_end = self.tokenizer.convert_tokens_to_ids("</think>")

    def chat(self, messages: Sequence[Mapping[str, Any]], request: Request,
             on_text: Callable[[str, bool], None] | None = None,
             stop: Callable[[], bool] | None = None) -> Reply:
        """The completion for OpenAI-style ``messages``; ``on_text(piece, thinking)`` streams it."""
        cancelled = threading.Event()
        future = self._worker.submit(self._chat, messages, request, on_text,
                                     lambda: cancelled.is_set() or bool(stop and stop()))
        try:
            return future.result()
        finally:
            cancelled.set()

    def prompt(self, messages: Sequence[Mapping[str, Any]], request: Request) -> list[int]:
        """The chat template over ``messages``, with tool-call arguments as the template reads them."""
        extra = {"tools": list(request.tools)} if request.tools else {}
        return list(self.tokenizer.apply_chat_template(
            [_templated(m) for m in messages], add_generation_prompt=True,
            enable_thinking=request.thinking, **extra, **dict(request.template)))

    def tool_calls(self, text: str, tools: Sequence[Mapping[str, Any]]) -> tuple[str, list[dict]]:
        """``text`` without its tool-call blocks, and those blocks as OpenAI tool calls."""
        start, end = self.tokenizer.tool_call_start, self.tokenizer.tool_call_end
        if not (tools and start and start in text):
            return text, []
        kept, calls = [text.split(start, 1)[0]], []
        for block in text.split(start)[1:]:
            body, _, rest = block.partition(end) if end else (block, "", "")
            parsed = self.tokenizer.tool_parser(body.strip(), list(tools))
            for one in parsed if isinstance(parsed, list) else [parsed]:
                arguments = one.get("arguments", {})
                calls.append({"id": f"call_{uuid.uuid4().hex[:12]}", "type": "function",
                              "function": {"name": one["name"], "arguments": arguments
                                           if isinstance(arguments, str)
                                           else json.dumps(arguments)}})
            kept.append(rest)
        return "".join(kept).strip(), calls

    def _chat(self, messages: Sequence[Mapping[str, Any]], request: Request,
              on_text: Callable[[str, bool], None] | None, stop: Callable[[], bool]) -> Reply:
        if request.seed is not None:
            mx.random.seed(request.seed)
        sampling = request.sampling or SAMPLING[request.thinking]
        held = self.tokenizer.tool_call_start if request.tools else ""
        writer = _Writer(self.tokenizer, (self.eos, self.think_end, held), request.thinking,
                         on_text)
        asked = Asked(request.max_tokens, sampling, request.rule, frozenset(self.eos))
        decoded = decode(self.session, self.prompt(messages, request), asked, writer.add, stop)
        writer.finish()
        answer_at = writer.answer_pass if writer.answer_pass is not None else len(decoded.passes)
        content, calls = self.tool_calls("".join(writer.parts[False]).strip(), request.tools)
        return Reply(content, "".join(writer.parts[True]).strip(), decoded,
                     writer.reasoning_tokens,
                     {"reasoning": decoded.passes[:answer_at],
                      "answer": decoded.passes[answer_at:]}, calls)


def _templated(message: Mapping[str, Any]) -> dict[str, Any]:
    """``message`` with each tool call's JSON arguments decoded, as chat templates expect."""
    out = dict(message)
    if out.get("tool_calls"):
        calls = []
        for call in out["tool_calls"]:
            function = dict(call.get("function") or {})
            if isinstance(function.get("arguments"), str):
                function["arguments"] = json.loads(function["arguments"] or "{}")
            calls.append({**call, "function": function})
        out["tool_calls"] = calls
    return out


class _Writer:
    """Detokenises as passes arrive, splitting reasoning inside the think block from the answer."""

    def __init__(self, tokenizer: Any, marks: tuple[set[int], int, str], thinking: bool,
                 on_text: Callable[[str, bool], None] | None) -> None:
        self.detok = tokenizer.detokenizer
        self.detok.reset()
        self.eos, self.think_end, self.held_from = marks
        self.thinking, self.on_text = thinking, on_text
        self.holding = False
        self.parts: dict[bool, list[str]] = {True: [], False: []}
        self.started = False
        self.passes = 0
        self.answer_pass: int | None = None
        self.reasoning_tokens = 0

    def _emit(self, piece: str) -> None:
        if not piece:
            return
        self.parts[self.thinking].append(piece)
        if not self.thinking and self.held_from and self.held_from in piece:
            self.holding = True
        if self.on_text is not None and not (self.holding and not self.thinking):
            self.on_text(piece, self.thinking)

    def add(self, tokens: list[int]) -> None:
        if not self.thinking and self.answer_pass is None:
            self.answer_pass = self.passes
        self.passes += 1
        for token in tokens:
            if token in self.eos:
                continue
            if self.thinking and token == self.think_end:
                self.detok.finalize()
                self._emit(self.detok.last_segment)
                self.detok.reset()
                self.thinking = False
                continue
            if self.thinking:
                self.reasoning_tokens += 1
            self.detok.add_token(token)
            piece = self.detok.last_segment
            if not self.thinking and not self.started:
                piece = piece.lstrip()
                self.started = bool(piece)
            self._emit(piece)

    def finish(self) -> None:
        self.detok.finalize()
        self._emit(self.detok.last_segment)
