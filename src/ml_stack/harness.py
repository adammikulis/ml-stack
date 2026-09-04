"""The Claude Agent SDK on a model this machine serves -- the harness, inside ml-stack.

Adam: "do we just include the harness library itself? ... in ml-stack I mean, integrate
it." The SDK (`claude-agent-sdk`, the ``claude`` extra) drives the same Claude Code binary
`ml_stack.claude` launches, and reads the same environment, so this is the same lease and
the same wiring with a Python face: :func:`session` leases the model in its measured shape
and yields a :class:`Harness` whose :meth:`Harness.ask` runs one agentic task and returns
what it said and what it spent, and whose :meth:`Harness.stream` hands the SDK's messages
through as they come. Every result's usage lands in a `Spent`-shaped record so the page,
the bench and an agent run are read the same way.

    from ml_stack.harness import session

    with session("Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf") as agent:
        answer = agent.ask("Read README.md and say what this repository is for.")
        print(answer.text, answer.spent)
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from collections.abc import AsyncIterator, Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from ml_stack.claude import DEFAULT_PORT, DEFAULT_SEATS, alias_of, environment

__all__ = ["Answer", "Harness", "Usage", "main", "session"]


def sdk() -> Any:
    """The SDK module, or a plain sentence about the extra that installs it."""
    try:
        import claude_agent_sdk
    except ImportError as why:  # pragma: no cover - depends on the extra
        raise ImportError(
            "the Claude Agent SDK is not installed: `pip install \"ml-stack[claude]\"` "
            "(claude-agent-sdk, which bundles the Claude Code binary)") from why
    return claude_agent_sdk


@dataclass
class Usage:
    """What one agentic task spent, in the shape `Spent` reads: tokens read, written and
    served from cache, turns, seconds. Cost is zero on a served model and is not pretended."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    turns: int = 0
    seconds: float = 0.0
    session_id: str = ""
    subtype: str = ""

    def said(self) -> str:
        return (f"{self.turns} turn(s) in {self.seconds:.1f}s; {self.input_tokens} read "
                f"({self.cache_read_tokens} from cache), {self.output_tokens} written")


@dataclass
class Answer:
    text: str
    spent: Usage
    messages: list[Any] = field(default_factory=list)
    is_error: bool = False


class Harness:
    """One served model, one SDK configuration, any number of tasks."""

    def __init__(self, base_url: str, alias: str, *, offline: bool = True,
                 options: Mapping[str, Any] | None = None) -> None:
        self.base_url = base_url
        self.alias = alias
        self.env = environment(base_url, alias, offline=offline, base={})
        self.options = dict(options or {})

    def configured(self, **over: Any) -> Any:
        """A `ClaudeAgentOptions` for this model: the environment that points every call
        at the server, the served alias as the model, and whatever the caller adds."""
        return sdk().ClaudeAgentOptions(model=self.alias, **{**self.options, **over,
                                        "env": {**self.env, **dict(over.get("env") or {})}})

    async def stream(self, prompt: str, **over: Any) -> AsyncIterator[Any]:
        """The SDK's messages for ``prompt``, as they arrive."""
        async for message in sdk().query(prompt=prompt, options=self.configured(**over)):
            yield message

    async def ask_async(self, prompt: str, **over: Any) -> Answer:
        began = time.time()
        texts: list[str] = []
        kept: list[Any] = []
        usage = Usage()
        async for message in self.stream(prompt, **over):
            kept.append(message)
            name = type(message).__name__
            if name == "AssistantMessage":
                for block in getattr(message, "content", []) or []:
                    text = getattr(block, "text", None)
                    if isinstance(text, str):
                        texts.append(text)
            elif name == "ResultMessage":
                usage = _usage_of(message)
                if getattr(message, "result", None) and not texts:
                    texts.append(str(message.result))
        usage.seconds = round(time.time() - began, 2)
        return Answer(text="\n".join(texts).strip(), spent=usage, messages=kept,
                      is_error=bool(getattr(kept[-1], "is_error", False)) if kept else False)

    def ask(self, prompt: str, **over: Any) -> Answer:
        """One agentic task, start to finish, and what it spent."""
        return asyncio.run(self.ask_async(prompt, **over))


def _usage_of(message: Any) -> Usage:
    raw = getattr(message, "usage", None) or {}
    get = (lambda k: raw.get(k)) if isinstance(raw, Mapping) else (lambda k: getattr(raw, k, None))
    return Usage(input_tokens=int(get("input_tokens") or 0),
                 output_tokens=int(get("output_tokens") or 0),
                 cache_read_tokens=int(get("cache_read_input_tokens") or 0),
                 cache_creation_tokens=int(get("cache_creation_input_tokens") or 0),
                 turns=int(getattr(message, "num_turns", 0) or 0),
                 seconds=float(getattr(message, "duration_ms", 0) or 0) / 1000.0,
                 session_id=str(getattr(message, "session_id", "") or ""),
                 subtype=str(getattr(message, "subtype", "") or ""))


@contextmanager
def session(model: str, *, port: int = DEFAULT_PORT, seats: int = DEFAULT_SEATS,
            profile: bool = True, offline: bool = True,
            say: Callable[[str], None] = lambda _line: None, **options: Any) -> Iterator[Harness]:
    """Lease ``model`` in its measured shape and yield a :class:`Harness` on it; the server
    goes when the block ends. ``options`` are `ClaudeAgentOptions` fields (cwd,
    allowed_tools, permission_mode, max_turns, system_prompt, mcp_servers, hooks...)."""
    from ml_stack.graph.bench.serve import find_model
    from ml_stack.serve.manager import serve
    from ml_stack.serve.profile import profile_for, said

    found = str(find_model(model))
    measured = profile_for(found) if profile else None
    if measured is not None:
        run = (measured.alone(port=port, model=found) if seats == 1
               else measured.run(port=port, seats=seats, model=found))
        say(f"serving in its measured shape: {said(measured)}")
    else:
        from ml_stack.serve.shape import Run, Shape

        run = Run(shape=Shape(model=found, port=port, seats=seats))
    with serve(run.model, manager=run.shape.manager(), **run.lease(), timeout=900.0,
               cache_reuse=256, warmup=False) as server:
        alias = alias_of(server.base_url, found)
        say(f"the harness on {server.base_url} as {alias!r}")
        yield Harness(server.base_url, alias, offline=offline, options=options)


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="ml-stack-agent",
        description="One agentic task through the Claude Agent SDK on a model this machine "
                    "serves, in its measured shape; prints what it said and what it spent.")
    ap.add_argument("prompt")
    ap.add_argument("--model", required=True)
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--seats", type=int, default=DEFAULT_SEATS)
    ap.add_argument("--cwd", default="")
    ap.add_argument("--max-turns", type=int, default=None)
    ap.add_argument("--allow", action="append", default=[], metavar="TOOL",
                    help="a tool to allow without asking; repeatable")
    ap.add_argument("--permission-mode", default=None)
    ap.add_argument("--no-profile", action="store_true")
    ap.add_argument("--online", action="store_true")
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(list(sys.argv[1:] if argv is None else argv))
    options: dict[str, Any] = {}
    if args.cwd:
        options["cwd"] = args.cwd
    if args.max_turns is not None:
        options["max_turns"] = args.max_turns
    if args.allow:
        options["allowed_tools"] = list(args.allow)
    if args.permission_mode:
        options["permission_mode"] = args.permission_mode
    with session(args.model, port=args.port, seats=args.seats, profile=not args.no_profile,
                 offline=not args.online, say=print, **options) as agent:
        answer = agent.ask(args.prompt)
    print(answer.text)
    print(f"spent: {answer.spent.said()}")
    return 1 if answer.is_error else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
