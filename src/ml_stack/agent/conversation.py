"""Conversation execution through the Agents SDK with the chat's role rails."""

from __future__ import annotations

import asyncio
import json
from typing import Any
from uuid import uuid4

from agents import Agent, FunctionTool, Model, ModelSettings, RunConfig, Runner
from agents.agent import ToolsToFinalOutputResult
from agents.exceptions import MaxTurnsExceeded
from agents.models.interface import ModelResponse
from agents.usage import Usage
from openai.types.responses import (
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputText,
)

from ml_stack.agent.sdk_runtime import input_items


class Conversation(Model):
    """An SDK model and tools bound to one guarded conversation."""

    def __init__(self, chat: Any, schemas: list, run_by: dict, outcome: Any) -> None:
        self.chat, self.schemas, self.run_by, self.outcome = chat, schemas, run_by, outcome
        self.calls: dict[str, dict] = {}
        self.steps = 0
        self.agent: Any = None
        self.lock = asyncio.Lock()

    async def get_response(self, *args: Any, **kwargs: Any) -> ModelResponse:
        said, calls = await asyncio.to_thread(self.chat._say, self.schemas)
        self.steps += 1
        output: list[Any] = []
        if calls:
            self.outcome.rounds += 1
            for call in calls:
                call['id'] = f"{call.get('id') or 'call'}_{uuid4().hex}"
                self.calls[call['id']] = call
                name = call['function']['name']
                output.append(ResponseFunctionToolCall(type='function_call', call_id=call['id'],
                              name=name if name in self.run_by else 'ml_stack_unknown_tool',
                              arguments=call['function'].get('arguments') or '{}'))
            self.chat.messages.append({'role': 'assistant', 'content': said, 'tool_calls': calls})
        else:
            self.chat.messages.append({'role': 'assistant', 'content': said})
            output.append(ResponseOutputMessage(type='message', id=uuid4().hex, status='completed',
                          role='assistant', content=[ResponseOutputText(type='output_text', text=said,
                                                                     annotations=[])]))
        return ModelResponse(output=output, usage=Usage(requests=1), response_id=None)

    async def stream_response(self, *args: Any, **kwargs: Any):
        raise NotImplementedError('Conversation text is streamed through the person interface')
        yield

    async def invoke(self, context: Any, arguments: str) -> str:
        async with self.lock:
            if self.chat.person.finished or self.chat.person.left:
                return json.dumps({'stopped': True})
            await asyncio.to_thread(self.chat._answer, self.calls[context.tool_call_id],
                                    self.run_by, self.outcome)
            return self.chat.messages[-1]['content']

    async def finished(self, context: Any, results: Any) -> ToolsToFinalOutputResult:
        person = self.chat.person
        return ToolsToFinalOutputResult(is_final_output=person.finished or person.left,
                                       final_output=person.summary)


async def _execute(chat: Any, schemas: list, run_by: dict, out: Any, nudge: str) -> bool:

    model = Conversation(chat, schemas, run_by, out)
    tools = [FunctionTool(name=s['function']['name'], description=s['function'].get('description', ''),
                          params_json_schema=s['function'].get('parameters') or {'type': 'object'},
                          on_invoke_tool=model.invoke, strict_json_schema=False) for s in schemas]
    tools.append(FunctionTool(name='ml_stack_unknown_tool', description='',
                              params_json_schema={'type': 'object'}, on_invoke_tool=model.invoke,
                              strict_json_schema=False))
    model.agent = Agent(name='local-chat', model=model, tools=tools,
                        model_settings=ModelSettings(parallel_tool_calls=False),
                        tool_use_behavior=model.finished)
    try:
        await Runner.run(model.agent, input_items(chat.messages), max_turns=chat.rounds,
                         run_config=RunConfig(tracing_disabled=True))
        if chat.task and not (chat.person.finished or chat.person.left):
            if chat.rounds is not None and model.steps >= chat.rounds:
                return True
            chat.messages.append({'role': 'user', 'content': nudge})
            await Runner.run(model.agent, input_items(chat.messages),
                             max_turns=None if chat.rounds is None else chat.rounds - model.steps,
                             run_config=RunConfig(tracing_disabled=True))
        return False
    except MaxTurnsExceeded:
        return True


def execute(chat: Any, schemas: list, run_by: dict, out: Any, nudge: str) -> bool:
    """Run one chat turn and return whether its turn budget was exhausted."""
    return asyncio.run(_execute(chat, schemas, run_by, out, nudge))
