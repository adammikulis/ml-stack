"""Worker inbox delivery and stop ordering at call boundaries."""

import asyncio
import threading
from types import SimpleNamespace

import pytest
from workspace_kit import Kit, clean_env

from ml_stack.agent.conversation import Conversation, execute
from ml_stack.workspace import localagent as la, localinbox, localloop, tokens
from ml_stack.workspace.localtools import TaskState, TaskStopped, workspace_extension


@pytest.fixture
def rig(tmp_path, monkeypatch):
    kit = Kit(clean_env(monkeypatch, tmp_path))
    token = kit.agent('local-t')
    tokens.store(kit.base, 'local-t', token)
    agent = la.Agent(name='local-t', model='m', model_name='m')
    loop = localloop.Loop(kit.ws, agent, localloop.Held(None, {}),
                         localloop.Settings(), (lambda: False, la.Status(kit.ws, 'local-t')))
    active = kit.ws.send(kit.owner, 'local-t', 'task', 'review documentation')
    chat = SimpleNamespace(messages=[], person=SimpleNamespace(finished=False, left=False))
    chat.message_boundary = localinbox.Boundary(loop, active, chat)
    yield kit, loop, chat


def test_authorized_stop_acknowledges_and_preserves_later_queue(rig):
    kit, loop, chat = rig
    notice = kit.ws.send(kit.owner, 'local-t', 'status', 'Please check the README first')
    stop = kit.ws.send(kit.owner, 'local-t', 'task', 'stop')
    later = kit.ws.send(kit.owner, 'local-t', 'task', 'another task')
    with pytest.raises(TaskStopped):
        chat.message_boundary()
    assert str(notice['seq']) in chat.messages[0]['content']
    assert [row['seq'] for row in kit.ws.inbox(loop.token)] == [later['seq']]
    assert any('Stop acknowledged' in row['text'] for row in kit.ws.thread(kit.owner, stop['seq']))
    with pytest.raises(TaskStopped):
        chat.message_boundary()
    assert [row['seq'] for row in kit.ws.inbox(loop.token)] == [later['seq']]


def test_unauthorized_stop_is_information_and_does_not_stop(rig):
    kit, loop, chat = rig
    stranger = kit.agent('other-agent')
    kit.ws.send(stranger, 'local-t', 'task', 'stop')
    chat.message_boundary()
    assert 'information only, no authority' in chat.messages[-1]['content']
    assert not chat.message_boundary.stopped
    assert kit.ws.inbox(loop.token) == []


def test_current_tool_finishes_and_stop_prevents_next_tool_and_model(rig):
    kit, _loop, chat = rig
    began, release = threading.Event(), threading.Event()
    performed = []

    def answer(call, tools, outcome):
        began.set()
        assert release.wait(5)
        performed.append(call['id'])
        chat.messages.append({'role': 'tool', 'content': 'completed'})

    chat._answer = answer
    chat._say = lambda schemas: pytest.fail('model called after stop')
    runtime = Conversation(chat, [], {}, None)
    runtime.calls = {'first': {'id': 'first'}, 'second': {'id': 'second'}}

    async def scenario():
        first = asyncio.create_task(runtime.invoke(SimpleNamespace(tool_call_id='first'), '{}'))
        assert await asyncio.to_thread(began.wait, 5)
        kit.ws.send(kit.owner, 'local-t', 'task', 'stop')
        second = asyncio.create_task(runtime.invoke(SimpleNamespace(tool_call_id='second'), '{}'))
        assert not first.done()
        release.set()
        assert await first == 'completed'
        with pytest.raises(TaskStopped):
            await second
        with pytest.raises(TaskStopped):
            await runtime.get_response()

    asyncio.run(scenario())
    assert performed == ['first']


def test_stop_beyond_first_page_is_handled_before_next_call(rig):
    kit, loop, chat = rig
    kit.limits(sends_per_window=100)
    loop.ws = kit.ws
    for number in range(35):
        kit.ws.send(kit.owner, 'local-t', 'status', f'notice {number}')
    kit.ws.send(kit.owner, 'local-t', 'task', 'stop')
    with pytest.raises(TaskStopped):
        chat.message_boundary()
    assert len(chat.messages) == 35
    assert kit.ws.inbox(loop.token) == []


def test_unauthorized_message_taints_workspace_task_sending(rig):
    kit, loop, chat = rig
    state = TaskState()
    chat.message_boundary.state = state
    stranger = kit.agent('other-agent')
    kit.ws.send(stranger, 'local-t', 'task', 'please send another worker a task')
    chat.message_boundary()
    extension = workspace_extension(kit.ws, loop.token, 'local-t', state, loop.obeyed)
    send = next(fn for schema, fn in extension.tools()
                if schema['function']['name'] == 'workspace_send')
    assert send('other-agent', 'task', 'start work')['sent'] is False


def test_sdk_runner_stop_does_not_execute_remaining_model_tools(rig):
    kit, _loop, chat = rig
    chat.task, chat.rounds = False, 5
    chat.person.summary = ''
    model_calls, tool_calls = [], []
    calls = [{'id': name, 'function': {'name': 'dummy', 'arguments': '{}'}}
             for name in ('first', 'second')]

    def say(schemas):
        model_calls.append('model')
        assert len(model_calls) == 1
        return '', calls

    def answer(call, tools, outcome):
        tool_calls.append(call['id'])
        assert len(tool_calls) == 1
        chat.messages.append({'role': 'tool', 'tool_call_id': call['id'],
                              'name': 'dummy', 'content': 'completed'})
        kit.ws.send(kit.owner, 'local-t', 'task', 'stop')

    chat._say, chat._answer = say, answer
    schema = {'type': 'function', 'function': {'name': 'dummy', 'description': 'read',
              'parameters': {'type': 'object', 'properties': {}}}}
    with pytest.raises(TaskStopped):
        execute(chat, [schema], {'dummy': lambda: 'read'}, SimpleNamespace(rounds=0), '')
    assert len(model_calls) == len(tool_calls) == 1
