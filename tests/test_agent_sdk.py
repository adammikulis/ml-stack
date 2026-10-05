"""SDK local-endpoint isolation and pre-inference refusal accounting."""

import asyncio

from ml_stack import guard
from ml_stack.agent import Agent, Done, FunctionTools
from ml_stack.client import Client
from ml_stack.interventions import Deny
from ml_stack.testing.tool_server import ToolCallingServer, Turn


def test_local_sdk_run_ignores_hosted_defaults(monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY', 'not-a-key')
    monkeypatch.setenv('OPENAI_BASE_URL', 'https://unselected.invalid')
    async def forbid_cloud(*args, **kwargs):
        raise AssertionError('local agent attempted an SDK HTTP request')
    monkeypatch.setattr('httpx.AsyncClient.send', forbid_cloud)
    server = ToolCallingServer([Turn(text=('local answer',))])
    agent = Agent(Client(server.base_url), FunctionTools([]), interventions=guard.default())
    try:
        result = asyncio.run(_collect(agent))[-1]
        assert isinstance(result, Done) and result.text == 'local answer'
        assert len(server.bodies) == 1
    finally:
        agent.close()
        server.close()


class RefuseModel:
    def before_model_call(self, context):
        return Deny('not allowed to generate')


def test_refused_sdk_model_turn_never_generates_or_counts_a_turn():
    server = ToolCallingServer([Turn(text=('must not appear',))])
    agent = Agent(Client(server.base_url), FunctionTools([]),
                  interventions=[*guard.default(), RefuseModel()])
    try:
        result = asyncio.run(_collect(agent))[-1]
        assert (result.reason, result.steps, result.tokens) == ('denied', 0, 0)
        assert not server.bodies
    finally:
        agent.close()
        server.close()


async def _collect(agent):
    return [event async for event in agent.run('say hello')]
