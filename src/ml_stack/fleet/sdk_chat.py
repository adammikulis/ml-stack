"""Plain chat SDK streaming in the Fleet SSE format."""

from __future__ import annotations

import asyncio
import json
import select
import socket
from collections.abc import Iterator
from typing import Any

from agents import Agent, ModelSettings, RunConfig, Runner

from ml_stack.client.sdk import local_client

from .chat import ChatError


def _disconnected(connection: Any) -> bool:
    if connection is None or not select.select([connection], [], [], 0)[0]:
        return False
    return not connection.recv(1, socket.MSG_PEEK)


async def _events(target: Any, payload: dict[str, Any], connection: Any):
    from agents import OpenAIChatCompletionsModel

    client = local_client(target, payload["model"])
    agent = Agent(name="Chat", model=OpenAIChatCompletionsModel(payload["model"], client),
                  model_settings=ModelSettings(temperature=payload.get("temperature")))
    result = Runner.run_streamed(agent, input=payload["messages"],
                                 run_config=RunConfig(tracing_disabled=True,
                                                      trace_include_sensitive_data=False))
    try:
        events = result.stream_events()
        while True:
            pending = asyncio.create_task(anext(events))
            while not pending.done():
                await asyncio.wait([pending], timeout=0.1)
                if _disconnected(connection):
                    pending.cancel()
                    await asyncio.gather(pending, return_exceptions=True)
                    return
            try:
                event = pending.result()
            except StopAsyncIteration:
                break
            if event.type == "raw_response_event" and event.data.type == "response.output_text.delta":
                frame = {"choices": [{"delta": {"content": event.data.delta}}]}
                yield b"data: " + json.dumps(frame).encode() + b"\n\n"
        yield b"data: [DONE]\n\n"
    finally:
        result.cancel()
        await client.close()


def stream(target: Any, payload: dict[str, Any], *, connection: Any = None) -> Iterator[bytes]:
    """Run one SDK turn and close generation when its consumer disconnects."""
    loop = asyncio.new_event_loop()
    events = _events(target, payload, connection)
    try:
        while True:
            try:
                yield loop.run_until_complete(anext(events))
            except StopAsyncIteration:
                break
    except Exception as exc:
        raise ChatError(str(exc)) from exc
    finally:
        loop.run_until_complete(events.aclose())
        pending = asyncio.all_tasks(loop)
        for task in pending:
            task.cancel()
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()
