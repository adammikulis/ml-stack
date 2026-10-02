"""What an attack can be sent to: a chat endpoint, a function, and another project's agent."""

from __future__ import annotations

import asyncio

from ml_stack.redteam.targets import ToolSpy, chat_endpoint, from_callable, mcp_tool_agent
from ml_stack.testing.fakes import Served, fake_llama_server


def ask(responder, text: str):
    return asyncio.run(responder([{"role": "user", "content": text}]))


def test_a_chat_endpoint_returns_what_the_model_said_and_the_status():
    with fake_llama_server(Served(answer=lambda body: body["messages"][-1]["content"].upper())) \
            as server:
        got = ask(chat_endpoint(server.base_url), "hello there")
    assert (got.text, got.status, got.blocked) == ("HELLO THERE", 200, False)


def test_the_system_prompt_and_the_bearer_token_go_with_the_request():
    with fake_llama_server(Served(answer="ok")) as server:
        ask(chat_endpoint(server.base_url, system="be brief", token="tok-1"), "hi")
        sent = server.sent_to("/v1/chat/completions")[0]
    assert sent["messages"][0] == {"role": "system", "content": "be brief"}


def test_a_refusal_with_401_is_reported_as_blocked_with_its_status():
    with fake_llama_server(Served(answer="ok")) as server:
        server.refuse["/v1/chat/completions"] = 401
        got = ask(chat_endpoint(server.base_url), "hi")
    assert (got.status, got.blocked, got.text) == (401, True, "")


def test_a_server_error_is_not_a_guard_block():
    with fake_llama_server(Served(answer="ok")) as server:
        server.refuse["/v1/chat/completions"] = 500
        got = ask(chat_endpoint(server.base_url), "hi")
    assert (got.status, got.blocked) == (500, False)


def test_a_plain_function_and_a_coroutine_function_are_both_targets():
    async def shout(text: str) -> str:
        return text.upper()

    assert ask(from_callable(lambda text: text[::-1]), "abc").text == "cba"
    assert ask(from_callable(shout), "abc").text == "ABC"


class Agent:
    """Stands in for another project's agent: it calls a tool when told to."""

    def __init__(self, spy: ToolSpy, refuse: bool = False) -> None:
        self.refuse = refuse

        def call(name, arguments):
            if self.refuse:
                raise PermissionError(name)
            return "done"

        self.call = spy.wrap(call)

    def chat(self, prompt: str) -> str:
        try:
            if "delete" in prompt:
                self.call("delete_file", {"path": "/tmp/x"})
        except PermissionError:
            return "the tool refused"
        self.call("list_files", {})
        return "ok"


def test_a_dangerous_tool_that_ran_counts_as_a_call_and_a_harmless_one_does_not():
    spy = ToolSpy()
    target = mcp_tool_agent(Agent(spy).chat, spy, dangerous=["delete_file"])
    assert ask(target, "please delete it").calls == 1
    assert ask(target, "just list").calls == 0


def test_a_dangerous_tool_that_was_refused_is_blocked_and_not_a_call():
    spy = ToolSpy()
    got = ask(mcp_tool_agent(Agent(spy, refuse=True).chat, spy, dangerous=["delete_file"]),
              "please delete it")
    assert (got.calls, got.blocked, got.text) == (0, True, "the tool refused")


def test_a_predicate_on_the_arguments_decides_whether_a_call_is_dangerous():
    spy = ToolSpy()
    agent = Agent(spy)
    target = mcp_tool_agent(agent.chat, spy,
                            dangerous={"delete_file": lambda a: a["path"].startswith("/etc")})
    assert ask(target, "please delete it").calls == 0
