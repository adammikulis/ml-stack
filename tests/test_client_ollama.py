"""The client against an Ollama server and against a hosted OpenAI one, on a fake socket.

Every reply here is shaped exactly as the Ollama API documents it
(https://github.com/ollama/ollama/blob/main/docs/api.md): ``/api/chat`` with the
nanosecond durations, ``/api/show`` with ``details``, ``/api/ps``, ``/api/tags`` and
``/api/version``. No request reaches port 11434 or 8080; the fake server answers on an
ephemeral port and psutil is faked in place.
"""

from __future__ import annotations

import json
import types

import pytest
from conftest import json_reply
from ml_stack.client import Client
from ml_stack.client import ollama
from ml_stack.client.spent import Spent
from ml_stack.telemetry import Call

MODEL = "qwen3.8-flash-next:125b-mlx"


def ollama_chat_reply(*, content="ok", thinking=None, tool_calls=None, done_reason="stop",
                      prompt_eval_count=300, eval_count=40, prompt_eval_duration=100_000_000,
                      eval_duration=400_000_000, load_duration=2_000_000_000,
                      total_duration=2_600_000_000):
    """One final ``/api/chat`` response, durations in nanoseconds as Ollama sends them."""
    message = {"role": "assistant", "content": content}
    if thinking is not None:
        message["thinking"] = thinking
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return {"model": MODEL, "created_at": "2026-09-03T00:00:00Z", "message": message,
            "done": True, "done_reason": done_reason, "total_duration": total_duration,
            "load_duration": load_duration, "prompt_eval_count": prompt_eval_count,
            "prompt_eval_duration": prompt_eval_duration, "eval_count": eval_count,
            "eval_duration": eval_duration}


def an_ollama(server, reply=None, *, show=None, tags=None, ps=None, version="0.33.3"):
    """A fake Ollama: the native routes, ``/v1/models``, and no ``/props``."""
    reply = reply if reply is not None else ollama_chat_reply()

    def handled(method, path, body):
        if path == "/api/chat":
            return json_reply(reply)
        if path == "/api/show":
            return json_reply(show or {"details": {"format": "safetensors",
                                                   "quantization_level": "nvfp4",
                                                   "family": "qwen3", "parameter_size": "125B"},
                                       "capabilities": ["completion", "tools", "thinking"],
                                       "model_info": {"general.architecture": "qwen3"}})
        if path == "/api/tags":
            return json_reply(tags or {"models": [{"name": MODEL, "model": MODEL,
                                                   "size": 70_000_000_000,
                                                   "details": {"format": "safetensors"}}]})
        if path == "/api/ps":
            return json_reply(ps or {"models": []})
        if path == "/api/version":
            return json_reply({"version": version})
        if path == "/v1/models":
            return json_reply({"object": "list", "data": [{"id": MODEL, "object": "model"}]})
        return 404, b'{"error": "not found"}'

    return server(handled)


# -- which API a client speaks ------------------------------------------------------------
class TestWhichApi:
    def test_llama_is_the_default(self):
        assert Client("http://127.0.0.1:8080").api == "llama"

    def test_the_hosted_openai_host_is_inferred(self):
        assert Client("https://api.openai.com/v1").api == "openai"

    def test_an_ollama_url_names_the_api_the_host_and_the_model(self):
        c = Client(f"ollama://127.0.0.1:11434/{MODEL}")
        assert c.api == "ollama" and c.base_url == "http://127.0.0.1:11434" and c.model == MODEL

    def test_an_ollama_url_without_a_model_leaves_it_unset(self):
        c = Client("ollama://127.0.0.1:11434")
        assert c.api == "ollama" and c.model is None

    def test_the_api_can_be_chosen_explicitly(self):
        assert Client("http://127.0.0.1:11434", api="ollama", model=MODEL).api == "ollama"
        assert Client("http://127.0.0.1:8080", api="openai", model="m").api == "openai"

    def test_an_unknown_api_is_refused(self):
        with pytest.raises(ValueError, match="api"):
            Client("http://x", api="banana")


# -- the body for a hosted OpenAI server ----------------------------------------------------
class TestOpenAIBody:
    def test_the_model_is_named_and_the_ceiling_is_max_tokens(self):
        body = Client("https://api.openai.com/v1", model="gpt-x",
                      n_predict=512).build_body([{"role": "user", "content": "hi"}])
        assert body["model"] == "gpt-x" and body["max_tokens"] == 512
        assert "n_predict" not in body

    def test_llamacpp_only_keys_never_go_out(self):
        body = Client("https://api.openai.com/v1", model="gpt-x", slot=2, top_k=40,
                      min_p=0.05).build_body([], grammar="root ::= x",
                                             chat_template_kwargs={"enable_thinking": False})
        for key in ("top_k", "min_p", "cache_prompt", "id_slot", "grammar",
                    "chat_template_kwargs"):
            assert key not in body, key

    def test_think_becomes_reasoning_effort_only_where_the_family_says_so(self):
        harmony = Client("https://api.openai.com/v1", model="gpt-oss-x",
                         family="gpt-oss").build_body([], think=False)
        assert harmony["reasoning_effort"] == "low" and "chat_template_kwargs" not in harmony
        qwen = Client("https://api.openai.com/v1", model="q",
                      family="qwen").build_body([], think=False)
        assert "reasoning_effort" not in qwen and "chat_template_kwargs" not in qwen
        assert "think" not in qwen

    def test_a_llamacpp_body_is_untouched(self):
        body = Client("http://127.0.0.1:8080", slot=2, n_predict=512).build_body([], top_k=40)
        assert body["n_predict"] == 512 and body["top_k"] == 40 and body["id_slot"] == 2
        assert "model" not in body


# -- the body for Ollama ---------------------------------------------------------------------
class TestOllamaBody:
    def test_it_is_the_native_chat_shape(self):
        tool = {"type": "function", "function": {"name": "ping", "parameters": {}}}
        c = Client(f"ollama://127.0.0.1:11434/{MODEL}", n_predict=4096, top_k=20, min_p=0.05,
                   temperature=0.0, context=32768, keep_alive="10m")
        body = c.build_body([{"role": "user", "content": "hi"}], tools=[tool], think=False,
                            seed=7)
        assert body["model"] == MODEL and body["messages"] == [{"role": "user", "content": "hi"}]
        assert body["tools"] == [tool] and body["think"] is False and body["stream"] is False
        assert body["keep_alive"] == "10m"
        assert body["options"] == {"temperature": 0.0, "top_k": 20, "min_p": 0.05,
                                   "num_predict": 4096, "num_ctx": 32768, "seed": 7}
        for key in ("n_predict", "id_slot", "cache_prompt", "chat_template_kwargs",
                    "tool_choice", "max_tokens", "seed"):
            assert key not in body, key

    def test_the_context_is_only_sent_when_the_caller_set_one(self):
        body = Client(f"ollama://127.0.0.1:11434/{MODEL}").build_body([])
        assert "num_ctx" not in body["options"] and "keep_alive" not in body

    def test_think_is_not_sent_unless_asked(self):
        assert "think" not in Client(f"ollama://127.0.0.1:11434/{MODEL}").build_body([])
        assert Client(f"ollama://127.0.0.1:11434/{MODEL}").build_body([], think=True)["think"] is True

    def test_a_json_schema_response_format_becomes_format(self):
        schema = {"type": "object", "properties": {"a": {"type": "string"}}}
        body = Client(f"ollama://127.0.0.1:11434/{MODEL}").build_body(
            [], response_format={"type": "json_schema", "json_schema": {"name": "s", "schema": schema}})
        assert body["format"] == schema
        assert Client(f"ollama://127.0.0.1:11434/{MODEL}").build_body(
            [], response_format="json")["format"] == "json"

    def test_assistant_tool_call_arguments_go_out_as_objects(self):
        """Ollama's message schema carries ``arguments`` as an object; a conversation
        replayed from the OpenAI shape carries the string the model wrote."""
        history = [{"role": "assistant", "content": "", "tool_calls": [
            {"id": "call_0", "type": "function",
             "function": {"name": "ping", "arguments": '{"id": "person:iris"}'}}]},
            {"role": "tool", "tool_call_id": "call_0", "content": "{}"}]
        body = Client(f"ollama://127.0.0.1:11434/{MODEL}").build_body(history)
        assert body["messages"][0]["tool_calls"][0]["function"]["arguments"] == {"id": "person:iris"}
        assert body["messages"][1]["role"] == "tool"

    def test_a_client_without_a_model_cannot_build_a_body(self):
        with pytest.raises(ValueError, match="model"):
            Client("ollama://127.0.0.1:11434").build_body([])


# -- a chat round trip ------------------------------------------------------------------------
class TestOllamaChat:
    def test_it_posts_to_the_native_route_and_reads_the_reply(self, server):
        fake = an_ollama(server, ollama_chat_reply(content="Iris surveys land.",
                                                   thinking="who surveys?"))
        c = Client(f"ollama://127.0.0.1:{fake.port}/{MODEL}")
        reply = c.chat([{"role": "user", "content": "who surveys?"}], think=True)
        method, path, body = fake.requests[-1]
        assert (method, path) == ("POST", "/api/chat")
        assert json.loads(body)["model"] == MODEL and json.loads(body)["stream"] is False
        assert reply.content == "Iris surveys land." and reply.thinking == "who surveys?"
        assert reply.finish_reason == "stop" and reply.raw["model"] == MODEL

    def test_tool_calls_come_back_in_the_openai_shape(self, server):
        fake = an_ollama(server, ollama_chat_reply(content="", tool_calls=[
            {"function": {"name": "look_up", "arguments": {"word": "surveying"}}}]))
        c = Client(fake.base_url, api="ollama", model=MODEL)
        reply = c.chat([{"role": "user", "content": "hi"}])
        assert reply.tool_calls is not None
        call = reply.tool_calls[0]
        assert call["type"] == "function" and call["id"]
        assert call["function"]["name"] == "look_up"
        assert json.loads(call["function"]["arguments"]) == {"word": "surveying"}

    def test_the_usage_and_timings_are_in_the_llamacpp_shape(self, server):
        fake = an_ollama(server, ollama_chat_reply())
        reply = Client(fake.base_url, api="ollama", model=MODEL).chat([])
        assert reply.raw["usage"] == {"prompt_tokens": 300, "completion_tokens": 40,
                                      "total_tokens": 340}
        timings = reply.raw["timings"]
        assert timings["prompt_ms"] == 100.0 and timings["predicted_ms"] == 400.0
        assert timings["prompt_n"] == 300 and timings["predicted_n"] == 40
        assert timings["load_ms"] == 2000.0
        assert timings["cache_n"] is None and timings["draft_n"] is None
        assert reply.raw["ollama"]["done_reason"] == "stop"

    def test_a_reply_cut_by_the_ceiling_is_length(self, server):
        fake = an_ollama(server, ollama_chat_reply(done_reason="length"))
        assert Client(fake.base_url, api="ollama", model=MODEL).chat([]).truncated

    def test_streaming_is_refused_with_a_reason(self, server):
        fake = an_ollama(server)
        with pytest.raises(NotImplementedError, match="stream"):
            Client(fake.base_url, api="ollama", model=MODEL).chat(
                [], on_delta=lambda channel, text: None)

    def test_a_raw_completion_is_refused(self, server):
        fake = an_ollama(server)
        with pytest.raises(NotImplementedError, match="completion"):
            Client(fake.base_url, api="ollama", model=MODEL).complete("x", grammar="root ::= x")


# -- the record, off an Ollama reply -------------------------------------------------------
class TestCallFromAnOllamaReply:
    def test_the_durations_become_milliseconds_and_the_counts_are_read(self, server):
        fake = an_ollama(server, ollama_chat_reply())
        reply = Client(fake.base_url, api="ollama", model=MODEL).chat([])
        one = Call.from_reply(reply, 0.8)
        assert (one.prompt_ms, one.predicted_ms, one.load_ms) == (100.0, 400.0, 2000.0)
        assert (one.prompt_n, one.predicted_n) == (300, 40)
        assert (one.prompt_tokens, one.completion_tokens) == (300, 40)
        assert one.model == MODEL

    def test_what_ollama_cannot_report_is_none_and_not_zero(self, server):
        fake = an_ollama(server, ollama_chat_reply())
        one = Call.from_reply(Client(fake.base_url, api="ollama", model=MODEL).chat([]), 0.8)
        assert one.cache_n is None and one.draft_n is None and one.draft_n_accepted is None
        assert one.held is None
        assert json.dumps(one.public())

    def test_a_llamacpp_reply_still_reads_zero_for_what_it_did_not_say(self):
        from ml_stack.client.chat import Reply

        bare = Reply(content="ok", raw={"usage": {"prompt_tokens": 120, "completion_tokens": 8},
                                        "timings": {"prompt_n": 100, "predicted_n": 8}})
        one = Call.from_reply(bare, 0.2)
        assert one.cache_n == 0 and one.draft_n == 0 and one.held == 108

    def test_spent_says_not_measured_rather_than_zero(self, server):
        fake = an_ollama(server, ollama_chat_reply())
        s = Spent()
        s.note(Client(fake.base_url, api="ollama", model=MODEL).chat([]), 0.8)
        s.note(Client(fake.base_url, api="ollama", model=MODEL).chat([]), 0.4)
        assert s.read_tokens == 600 and s.completion_tokens == 80
        assert s.cached_tokens is None and s.draft_tokens is None and s.draft_taken is None
        assert s.acceptance is None and s.drafted is False
        assert s.context_peak is None and s.context_last is None
        assert s.prompt_tokens_per_second == 3000.0
        got = s.public()
        assert got["cached_tokens"] is None and got["context_peak"] is None
        assert json.dumps(got)
        totals = Spent.totals([got])
        assert totals["cached_tokens"] is None and totals["context_peak"] is None
        assert totals["read_tokens"] == 600 and totals["acceptance"] is None


# -- what is serving ---------------------------------------------------------------------------
class TestServedBy:
    def test_ollama_says_program_format_quant_runtime_and_size(self, server, monkeypatch):
        monkeypatch.setattr(ollama, "PLATFORM", "darwin")
        fake = an_ollama(server)
        got = Client(fake.base_url, api="ollama", model=MODEL).served_by()
        assert got == {"program": "ollama", "version": "0.33.3", "format": "safetensors",
                       "runtime": "mlx", "quant": "nvfp4", "model": MODEL,
                       "weights_bytes": 70_000_000_000}
        shown = [json.loads(body) for method, path, body in fake.requests if path == "/api/show"]
        assert shown == [{"model": MODEL}]

    def test_a_gguf_on_ollama_runs_on_llamacpp(self, server):
        fake = an_ollama(server, show={"details": {"format": "gguf",
                                                   "quantization_level": "Q4_K_M"}},
                         tags={"models": []})
        got = Client(fake.base_url, api="ollama", model=MODEL).served_by()
        assert got["format"] == "gguf" and got["runtime"] == "llama.cpp"
        assert got["quant"] == "Q4_K_M" and got["weights_bytes"] is None

    def test_safetensors_off_a_mac_is_not_called_mlx(self):
        assert ollama.runtime_for("safetensors", "darwin") == "mlx"
        assert ollama.runtime_for("safetensors", "linux") == "unknown"
        assert ollama.runtime_for("gguf", "linux") == "llama.cpp"

    def test_llamacpp_reads_props_and_the_filename(self, server, tmp_path):
        weights = tmp_path / "thing-125B-UD-Q4_K_XL.gguf"
        weights.write_bytes(b"GGUF" + b"\0" * 96)

        def handled(method, path, body):
            if path == "/props":
                return json_reply({"model_path": str(weights), "build_info": "b7100-abc123",
                                   "total_slots": 1})
            return 404, b"{}"

        fake = server(handled)
        got = Client(fake.base_url).served_by()
        assert got == {"program": "llama.cpp", "version": "b7100-abc123", "format": "gguf",
                       "runtime": "llama.cpp", "quant": "Q4_K_XL",
                       "model": "thing-125B-UD-Q4_K_XL.gguf", "weights_bytes": 100}

    @pytest.mark.parametrize("name, quant", [
        ("thing-UD-Q4_K_XL.gguf", "Q4_K_XL"), ("thing-Q4_K_M.gguf", "Q4_K_M"),
        ("thing-IQ4_XS.gguf", "IQ4_XS"), ("thing-UD-IQ2_M.gguf", "IQ2_M"),
        ("thing-BF16.gguf", "BF16"), ("thing-Q8_0-00001-of-00003.gguf", "Q8_0"),
        ("thing.gguf", None)])
    def test_the_quant_token_is_read_off_a_gguf_name(self, name, quant):
        assert ollama.quant_in_name(name) == quant


# -- the processes holding the weights ------------------------------------------------------
def _proc(pid, argv, *, listening=(), children=(), state="running", name=None):
    """One psutil process: its argv, the ports it listens on, its children, its state."""
    conns = [types.SimpleNamespace(status="LISTEN",
                                   laddr=types.SimpleNamespace(ip="127.0.0.1", port=p))
             for p in listening]
    return types.SimpleNamespace(
        pid=pid, info={"pid": pid, "name": name or argv[0].rsplit("/", 1)[-1],
                       "cmdline": argv},
        status=lambda: state,
        net_connections=lambda kind="inet": conns,
        children=lambda recursive=False: list(children))


class TestProcesses:
    def test_llamacpp_is_the_server_whose_command_line_carries_the_port(self, monkeypatch):
        import psutil

        fakes = [_proc(11, ["/x/llama-server", "--port", "8081", "-m", "/m/a.gguf"]),
                 _proc(12, ["/x/llama-server", "-m", "/m/b.gguf", "--port", "8082"]),
                 _proc(13, ["/usr/bin/python3", "-m", "something", "--port", "8082"])]
        monkeypatch.setattr(psutil, "process_iter", lambda attrs=None: fakes)
        monkeypatch.setattr(psutil, "CONN_LISTEN", "LISTEN")
        assert Client("http://127.0.0.1:8082").processes() == [12]
        assert Client("http://127.0.0.1:8090").processes() == []

    def test_a_zombie_with_no_command_line_is_not_the_default_port(self, monkeypatch):
        """psutil reports an empty argv for a defunct llama-server; read as "no --port"
        that matched 8080 on this machine (2026-09-03) beside the live one."""
        import psutil

        fakes = [_proc(11, [], state=psutil.STATUS_ZOMBIE, name="llama-server"),
                 _proc(12, ["/x/llama-server", "--port", "8080", "-m", "/m/b.gguf"])]
        monkeypatch.setattr(psutil, "process_iter", lambda attrs=None: fakes)
        assert Client("http://127.0.0.1:8080").processes() == [12]

    def test_ollama_is_every_child_of_the_listener(self, monkeypatch):
        import psutil

        runner = _proc(31, ["/Applications/Ollama.app/Contents/Resources/ollama", "runner",
                            "--mlx-engine", "--port", "54321"])
        helper = _proc(32, ["/x/lib/ollama/llama-server", "--port", "54322"])
        serve = _proc(30, ["/Applications/Ollama.app/Contents/Resources/ollama", "serve"],
                      listening=(11434,), children=(runner, helper))
        app = _proc(29, ["/Applications/Ollama.app/Contents/MacOS/Ollama"], children=(serve,))
        monkeypatch.setattr(psutil, "process_iter", lambda attrs=None: [app, serve, runner, helper])
        monkeypatch.setattr(psutil, "CONN_LISTEN", "LISTEN")
        assert Client("http://127.0.0.1:11434", api="ollama", model=MODEL).processes() == [31, 32]

    def test_a_listener_with_nothing_loaded_is_the_only_candidate(self, monkeypatch):
        import psutil

        serve = _proc(30, ["/x/ollama", "serve"], listening=(11434,))
        monkeypatch.setattr(psutil, "process_iter", lambda attrs=None: [serve])
        monkeypatch.setattr(psutil, "CONN_LISTEN", "LISTEN")
        assert Client("http://127.0.0.1:11434", api="ollama", model=MODEL).processes() == [30]


# -- the family, on a server with no /props -----------------------------------------------
class TestFamilyOnOllama:
    def test_the_tag_names_the_family_without_a_probe(self, server):
        fake = an_ollama(server)
        c = Client(fake.base_url, api="ollama", model=MODEL)
        assert c.family.name == "qwen"
        assert fake.requests == []

    def test_without_a_model_the_served_ids_are_asked_for(self, server):
        fake = an_ollama(server)
        from ml_stack.client.chat import forget_families

        forget_families()
        c = Client(fake.base_url, api="ollama")
        assert c.family.name == "qwen"
        assert [path for _, path, _ in fake.requests] == ["/v1/models"]

    def test_the_card_does_not_ask_for_props(self, server):
        fake = an_ollama(server)
        c = Client(fake.base_url, api="ollama", model=MODEL)
        assert isinstance(c.card, dict)
        assert "/props" not in [path for _, path, _ in fake.requests]
