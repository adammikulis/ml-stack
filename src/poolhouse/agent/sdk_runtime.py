"""Agents SDK execution over the maintained local client and intervention rails."""

from __future__ import annotations

import asyncio
import json
from typing import Any
from uuid import uuid4

from agents import Agent as SdkAgent, FunctionTool, Model, RunConfig, Runner
from agents.exceptions import MaxTurnsExceeded
from agents.models.chatcmpl_converter import Converter
from agents.models.interface import ModelResponse
from agents.usage import Usage
from openai.types.responses import (
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputText,
)

from poolhouse.agent.events import Denied, Done, Repair, ToolCall, ToolResult
from poolhouse.agent.loop import _completion_tokens, _inject, _overflowed, _task_of
from poolhouse.agent.schema import from_mcp, index_by_name
from poolhouse.guard import start
from poolhouse.http import ServerError
from poolhouse.interventions import Call


class Stopped(Exception):
    """A run reached an intervention or resource limit."""

    def __init__(self, reason: str, text: str = "") -> None:
        self.reason, self.text = reason, text


def input_items(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert kept chat messages to SDK input items."""
    out = []
    for message in messages:
        if message['role'] == 'tool':
            out.append({'type': 'function_call_output', 'call_id': message['tool_call_id'],
                        'output': message.get('content', '')})
            continue
        if message.get('content'):
            out.append({'role': message['role'], 'content': message['content']})
        for call in message.get('tool_calls') or []:
            fn = call['function']
            out.append({'type': 'function_call', 'call_id': call['id'],
                        'name': fn['name'], 'arguments': fn['arguments']})
    return out


class Runtime(Model):
    """Translate SDK model turns and tools through the application's rails."""

    def __init__(self, owner: Any, messages: list, schemas: list, run: Any) -> None:
        self.owner, self.messages, self.schemas, self.run = owner, messages, schemas, run
        self.index = index_by_name(schemas)
        self.steps = self.calls = self.tokens = self.repairs = 0
        self.pending: dict[str, Any] = {}
        self.names: dict[str, str] = {}
        self.agent: Any = None
        self.consumed = 0
        self.results: dict[str, Any] = {}
        self.batch: list[Any] = []
        self.used_ids: set[str] = set()
        self.gate = asyncio.Semaphore(max(1, owner.budget.parallel))

    def emit(self, event: Any) -> None:
        self.owner._asked.put_nowait(event)

    async def decide(self, hook: str) -> None:
        owner = self.owner
        if owner.watch:
            owner.watch.enter()
        gate = await self.run.decide(hook, self.run.context)
        why = '' if gate.allowed else getattr(gate.verdict, 'reason', '')
        why = why or (owner.watch.frozen() if owner.watch else '')
        if why:
            raise Stopped('denied', why)

    async def invoke(self, context: Any, arguments: str) -> str:
        owner = self.owner
        one = self.pending[context.tool_call_id]
        async with self.gate:
            answers = await owner._dispatch([one])
        answer = answers[0]
        text = answer.text
        if not (one.errors or one.denied):
            if owner.watch:
                owner.watch.enter()
            screened = await self.run.after_tool(Call(one.name, one.args, one.id), text)
            text = owner.watch.shown(one.name, text, screened) if owner.watch else screened.text
        self.results[one.id] = ToolResult(one.id, one.name, text, answer.is_error)
        if len(self.results) == len(self.batch):
            for call in self.batch:
                self.emit(self.results[call.id])
        self.names[one.id] = one.name
        return text

    def tool(self, name: str, schema: dict[str, Any]) -> FunctionTool:
        return FunctionTool(name=name, description=schema.get('description', ''),
                            params_json_schema=schema.get('parameters', {'type': 'object'}),
                            on_invoke_tool=self.invoke, strict_json_schema=False)

    async def vet(self, pending: list) -> None:
        owner = self.owner
        for one in pending:
            if one.errors:
                continue
            if owner.watch and (why := owner.watch.refuses(one.name, one.args)):
                one.denied = f'Denied: {why}'
                continue
            gate = await self.run.before_tool(Call(one.name, one.args, one.id))
            one.denied = '' if gate.allowed else getattr(gate.verdict, 'reason', '')
            if one.denied and owner.watch:
                owner.watch.denied(one.name, one.args, getattr(gate.verdict, 'by', ''),
                                   one.denied, logged=gate.confirmed is None)

    async def get_response(self, *args: Any, **kwargs: Any) -> ModelResponse:
        input = args[1] if len(args) > 1 else kwargs['input']
        owner = self.owner
        incoming = Converter.items_to_messages(input[self.consumed:])
        if self.consumed:
            self.messages.extend(incoming)
        self.consumed = len(input)
        if owner.budget.max_repairs is not None and self.repairs > owner.budget.max_repairs:
            raise Stopped('repairs_exhausted')
        if owner.budget.max_tokens is not None and self.tokens >= owner.budget.max_tokens:
            raise Stopped('max_tokens', getattr(self, 'last_text', ''))
        for message in self.messages:
            for call in message.get('tool_calls') or []:
                if call['id'] in self.names:
                    call['function']['name'] = self.names[call['id']]
            if message['role'] == 'tool' and message.get('tool_call_id') in self.names:
                message['name'] = self.names[message['tool_call_id']]
        self.run.context.step, self.run.context.tool_calls = self.steps + 1, self.calls
        await self.decide('before_model_call')
        _inject(self.run, self.messages)
        if owner.auto:
            for event in await owner.auto.before(self.messages, self.schemas):
                self.emit(event)
        reply = await self.ask()
        self.steps += 1
        count = _completion_tokens(reply)
        self.tokens += count
        pending = owner._pending(reply, self.index)
        text = reply.content or ''
        text = owner.watch.said(text) if owner.watch else text
        output: list[Any] = []
        if text and not pending:
            output.append(ResponseOutputMessage(id=uuid4().hex, role='assistant', status='completed',
                type='message', content=[ResponseOutputText(type='output_text', text=text, annotations=[])]))
        if pending:
            if owner.budget.max_tool_calls is not None and self.calls + len(pending) > owner.budget.max_tool_calls:
                raise Stopped('max_tool_calls', text)
            self.calls += len(pending)
            self.run.context.tool_calls = self.calls
            for one in pending:
                if one.id in self.used_ids:
                    one.id = f'{one.id}_{self.steps}'
                self.used_ids.add(one.id)
            await self.vet(pending)
            self.batch, self.results = pending, {}
            self.repairs = self.repairs + 1 if all(one.errors for one in pending) else 0
            for one in pending:
                self.pending[one.id] = one
                self.names[one.id] = one.name
                self.emit(Repair(one.id, one.name, one.errors) if one.errors else
                          Denied(one.id, one.name, one.denied) if one.denied else
                          ToolCall(one.id, one.name, one.args or {}))
                output.append(ResponseFunctionToolCall(type='function_call', call_id=one.id,
                              name=one.name if one.name in self.index else 'poolhouse_repair',
                              arguments=json.dumps(one.args or {})))
        self.last_text = text
        return ModelResponse(output=output, usage=Usage(requests=1, output_tokens=count,
                                                       total_tokens=count), response_id=None)

    async def ask(self) -> Any:
        owner = self.owner
        reply = None
        for attempt in (0, 1):
            try:
                async for piece in owner._ask(self.messages, self.schemas):
                    if hasattr(piece, 'delta'):
                        self.emit(piece)
                    else:
                        reply = piece
                break
            except ServerError as exc:
                if attempt or not owner.auto or not _overflowed(exc):
                    raise
                for event in await owner.auto.before(self.messages, self.schemas, force=True):
                    self.emit(event)
        return reply

    async def stream_response(self, *args: Any, **kwargs: Any):
        raise NotImplementedError('Local model deltas are emitted through the application event queue')
        yield


async def execute(owner: Any, task: Any):
    """Yield application events from an SDK run."""
    messages = [{'role': 'user', 'content': task}] if isinstance(task, str) else task
    schemas = from_mcp(listed := await owner.tools.list_tools(), owner.budget.profile)
    rail = start(owner._items(listed), offered=schemas, task=_task_of(messages),
                 confirm=owner.confirm, notify=owner._notify)
    rail.context.messages = messages
    runtime = Runtime(owner, messages, schemas, rail)
    runtime.agent = SdkAgent(name='local-agent', model=runtime,
                            tools=[runtime.tool(s['function']['name'], s['function']) for s in schemas]
                                  + [runtime.tool('poolhouse_repair', {})])

    async def drive():
        try:
            await runtime.decide('before_invocation')
            result = await Runner.run(runtime.agent, input_items(messages),
                                      max_turns=owner.budget.max_steps,
                                      run_config=RunConfig(tracing_disabled=True))
            messages.extend(Converter.items_to_messages(result.to_input_list()[runtime.consumed:]))
            text = str(result.final_output or '')
            return Done('answer', owner.watch.said(text) if owner.watch else text,
                        runtime.steps, runtime.calls, runtime.tokens, messages)
        except MaxTurnsExceeded:
            return Done('max_steps', '', runtime.steps, runtime.calls, runtime.tokens, messages)
        except Stopped as stop:
            return Done(stop.reason, stop.text, runtime.steps, runtime.calls, runtime.tokens, messages)

    async for event in owner._watching(drive()):
        yield event
