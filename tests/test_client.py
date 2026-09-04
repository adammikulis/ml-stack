"""The client, against a real HTTP server on a real socket.

Nothing here mocks ``urlopen``. The failures these modules exist to prevent are
transport-shaped -- a server that answers /health while still loading, one that 500s on a
concurrent request, one that ignores a grammar -- and a mocked transport cannot produce
any of them.
"""

from __future__ import annotations

import json
import time

import pytest
from conftest import json_reply
from ml_stack.client import (
    Client,
    EmbeddingError,
    GrammarUnsupportedError,
    ServerError,
    ServerUnreachable,
    embed,
    estimate_tokens,
    family_by_name,
    family_for_model_id,
    heuristic_tokens,
    is_healthy,
    quant_from_model_path,
    reported_models,
    serving_params,
    strip_thinking,
    wait_for_health,
)


def _chat_reply(content: str, **message: object):
    return json_reply(
        {"choices": [{"message": {"role": "assistant", "content": content, **message},
                      "finish_reason": "stop"}]}
    )


class TestBuildBody:
    """Split out from ``chat`` precisely so it is testable with no server running."""

    def test_pins_the_kv_slot_when_one_is_given(self):
        body = Client("http://x", slot=3).build_body([{"role": "user", "content": "hi"}])
        assert body["id_slot"] == 3
        assert body["cache_prompt"] is True, "pinning a slot without reusing its cache "\
                                             "is the cost with none of the benefit"

    def test_omits_slot_keys_when_unpinned(self):
        body = Client("http://x").build_body([])
        assert "id_slot" not in body and "cache_prompt" not in body

    def test_normalises_the_json_response_format_alias(self):
        """Callers habitually pass "json"; llama-server wants "json_object"."""
        body = Client("http://x").build_body([], response_format="json")
        assert body["response_format"] == {"type": "json_object"}

    def test_passes_through_an_explicit_response_format(self):
        schema = {"type": "json_schema", "json_schema": {"name": "s"}}
        assert Client("http://x").build_body([], response_format=schema)["response_format"] == schema

    def test_strips_llamacpp_only_keys_for_hosted_openai(self):
        """api.openai.com rejects top_k outright rather than ignoring it."""
        body = Client("https://api.openai.com/v1", slot=1,
                      n_predict=512).build_body([], top_k=40)
        assert "top_k" not in body
        assert "n_predict" not in body and body["max_tokens"] == 512
        assert "id_slot" not in body

    def test_keeps_them_for_a_local_server(self):
        body = Client("http://127.0.0.1:8080", n_predict=512).build_body([], top_k=40)
        assert body["top_k"] == 40 and body["n_predict"] == 512

    def test_the_token_ceiling_defaults_high(self):
        """A ceiling is not a budget: nothing is spent that is not generated, so a high one
        costs nothing and a low one truncates -- and what it truncates is the answer, never
        the thinking. Measured, gemma-4 filled a 220-token ceiling with reasoning and
        returned empty content."""
        assert Client("http://x").build_body([])["n_predict"] >= 8192

    def test_tools_only_appear_when_supplied(self):
        tool = {"type": "function", "function": {"name": "ping", "parameters": {}}}
        assert "tools" not in Client("http://x").build_body([])
        assert Client("http://x").build_body([], tools=[tool])["tool_choice"] == "auto"


class TestNormalize:
    def test_lifts_a_legacy_function_call_into_tool_calls(self):
        """Some llama.cpp builds emit tool calls only in the legacy field. Callers must
        see one shape, not two."""
        reply = Client.normalize(
            {"choices": [{"message": {"function_call": {"name": "ping", "arguments": "{}"}}}]}
        )
        assert reply.tool_calls is not None
        assert reply.tool_calls[0]["function"]["name"] == "ping"
        assert reply.tool_calls[0]["type"] == "function"

    def test_prefers_real_tool_calls_over_the_legacy_field(self):
        reply = Client.normalize(
            {"choices": [{"message": {
                "tool_calls": [{"id": "a", "type": "function",
                                "function": {"name": "real", "arguments": "{}"}}],
                "function_call": {"name": "legacy", "arguments": "{}"}}}]}
        )
        assert reply.tool_calls[0]["function"]["name"] == "real"

    def test_truncation_is_visible(self):
        reply = Client.normalize({"choices": [{"message": {"content": "cut"},
                                               "finish_reason": "length"}]})
        assert reply.truncated

    def test_rejects_a_non_dict_payload(self):
        with pytest.raises(ServerError):
            Client.normalize(["not", "a", "response"])


    def test_a_reasoning_field_is_carried_as_thinking(self):
        reply = Client.normalize({"choices": [{"message": {
            "role": "assistant", "content": "",
            "reasoning_content": "count the people first"}, "finish_reason": "stop"}]})
        assert reply.content == "" and reply.thinking == "count the people first"

    def test_reasoning_and_think_tags_both_survive(self):
        reply = Client.normalize({"choices": [{"message": {
            "role": "assistant", "content": "<think>tags</think>42",
            "reasoning": "field"}, "finish_reason": "stop"}]})
        assert reply.content == "42"
        assert reply.thinking == "field\ntags"


class TestStripThinking:
    def test_separates_the_trace_from_the_answer(self):
        visible, thinking = strip_thinking("<think>weigh it up</think>The answer is 4.")
        assert visible == "The answer is 4."
        assert thinking == "weigh it up"

    def test_keeps_the_trace_rather_than_discarding_it(self):
        """It is the most useful thing in the response when a tool call comes back wrong."""
        _, thinking = strip_thinking("<think>a</think>x<think>b</think>y")
        assert thinking == "a\nb"

    def test_leaves_ordinary_text_untouched(self):
        assert strip_thinking("plain") == ("plain", None)
        assert strip_thinking(None) == (None, None)


GPT_OSS_ID = "/models/gpt-oss-120b-mxfp4-00001-of-00003.gguf"
QWEN_ID = "/models/Qwen3-4B-Instruct-Q4_K_M.gguf"
UNKNOWN_ID = "/models/mystery-1b-Q4_K_M.gguf"


def _completion(model, content, **message):
    """A whole ``/v1/chat/completions`` payload, the way a served model returns one."""
    return {"model": model,
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": content, **message}}]}


def _sse(chunks):
    """The chunks as an SSE body, one ``data:`` frame each, terminated."""
    return ("".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n").encode()


TIMINGS = {"prompt_n": 12, "prompt_ms": 30.5, "predicted_n": 4, "predicted_ms": 80.0}
USAGE = {"prompt_tokens": 12, "completion_tokens": 4, "total_tokens": 16}


def _stream_of(model, deltas, finish="stop"):
    """Deltas as chunks carrying the served model id, the last one finishing and carrying
    the timings and usage the way llama-server's last chunk does."""
    out = []
    for index, delta in enumerate(deltas):
        chunk = {"model": model,
                 "choices": [{"index": 0, "delta": delta, "finish_reason": None}]}
        if index == len(deltas) - 1:
            chunk["choices"][0]["finish_reason"] = finish
            chunk["timings"], chunk["usage"] = TIMINGS, USAGE
        out.append(chunk)
    return out


def _both_ways(server, model, payload, deltas, *, family=None, finish="stop"):
    """The same reply fetched whole and streamed. Returns ``(whole, streamed, seen)``."""
    whole_server = server(lambda m, p, b: json_reply({**payload, "timings": TIMINGS,
                                                       "usage": USAGE}))
    whole = Client(whole_server.base_url, family=family).chat([{"role": "user", "content": "x"}])

    body = _sse(_stream_of(model, deltas, finish=finish))
    stream_server = server(lambda m, p, b: (200, body))
    seen: list[tuple[str, str]] = []
    streamed = Client(stream_server.base_url, family=family).chat(
        [{"role": "user", "content": "x"}], on_delta=lambda k, t: seen.append((k, t)))
    return whole, streamed, seen


def _same(whole, streamed):
    assert whole.content == streamed.content
    assert whole.thinking == streamed.thinking
    assert whole.tool_calls == streamed.tool_calls
    assert whole.finish_reason == streamed.finish_reason
    assert whole.raw.get("timings") == streamed.raw.get("timings") == TIMINGS
    assert whole.raw.get("usage") == streamed.raw.get("usage") == USAGE


class TestFamilies:
    def test_a_model_id_names_its_family(self):
        assert family_for_model_id(GPT_OSS_ID).name == "gpt-oss"
        assert family_for_model_id(QWEN_ID).name == "qwen"

    def test_an_unrecognised_id_falls_back_to_generic(self):
        assert family_for_model_id(UNKNOWN_ID).name == "generic"
        assert family_for_model_id(None).name == "generic"

    def test_a_family_can_be_named(self):
        assert family_by_name("gpt-oss") is family_for_model_id(GPT_OSS_ID)
        assert family_by_name("QWEN").name == "qwen"
        with pytest.raises(ValueError, match="unknown model family"):
            family_by_name("nonesuch")


class TestGptOssReplies:
    def test_the_answer_and_the_reasoning_both_come_through(self, server):
        payload = _completion(GPT_OSS_ID, "4", reasoning_content="user asks 2+2, so 4")
        instance = server(lambda m, p, b: json_reply(payload))
        reply = Client(instance.base_url).chat([{"role": "user", "content": "2+2?"}])
        assert reply.content == "4"
        assert reply.thinking == "user asks 2+2, so 4"

    def test_an_unfinished_turn_keeps_its_reasoning(self, server):
        """Harmony emits the analysis channel before the final one, so a turn cut short
        has text in reasoning_content and nothing in content."""
        payload = _completion(GPT_OSS_ID, "", reasoning_content="counting the people")
        payload["choices"][0]["finish_reason"] = "length"
        instance = server(lambda m, p, b: json_reply(payload))
        reply = Client(instance.base_url).chat([])
        assert reply.content == ""
        assert reply.thinking == "counting the people"
        assert reply.truncated

    def test_a_literal_think_tag_in_the_answer_is_left_alone(self, server):
        """Harmony carries no in-band delimiter, so the tag is the model's own text."""
        payload = _completion(GPT_OSS_ID, "write <think> to open a block",
                              reasoning_content="explain the syntax")
        instance = server(lambda m, p, b: json_reply(payload))
        reply = Client(instance.base_url).chat([])
        assert reply.content == "write <think> to open a block"
        assert reply.thinking == "explain the syntax"

    def test_streaming_lands_where_the_whole_reply_does(self, server):
        whole, streamed, seen = _both_ways(
            server, GPT_OSS_ID,
            _completion(GPT_OSS_ID, "42", reasoning_content="add them up"),
            [{"role": "assistant"}, {"reasoning_content": "add "},
             {"reasoning_content": "them up"}, {"content": "4"}, {"content": "2"}],
        )
        assert whole.content == "42" and whole.thinking == "add them up"
        _same(whole, streamed)
        assert seen == [("thinking", "add "), ("thinking", "them up"),
                        ("content", "4"), ("content", "2")]


class TestQwenReplies:
    def test_an_in_band_block_splits_and_does_not_leak(self, server):
        payload = _completion(QWEN_ID, "<think>weigh it up</think>The answer is 4.")
        instance = server(lambda m, p, b: json_reply(payload))
        reply = Client(instance.base_url).chat([])
        assert reply.content == "The answer is 4."
        assert reply.thinking == "weigh it up"
        assert "<think>" not in reply.content and "weigh" not in reply.content

    def test_streaming_splits_a_block_broken_across_chunks(self, server):
        whole, streamed, seen = _both_ways(
            server, QWEN_ID,
            _completion(QWEN_ID, "<think>weigh it up</think>The answer is 4."),
            [{"role": "assistant"}, {"content": "<th"}, {"content": "ink>weigh "},
             {"content": "it up</thi"}, {"content": "nk>The answer "},
             {"content": "is 4."}],
        )
        assert whole.content == "The answer is 4." and whole.thinking == "weigh it up"
        _same(whole, streamed)
        assert seen == [("thinking", "weigh "), ("thinking", "it up"),
                        ("content", "The answer "), ("content", "is 4.")]

    def test_the_block_never_reaches_the_answer_channel(self, server):
        _, _, seen = _both_ways(
            server, QWEN_ID, _completion(QWEN_ID, "<think>secret</think>said"),
            [{"content": "<think>sec"}, {"content": "ret</think>sa"}, {"content": "id"}],
        )
        answered = "".join(text for kind, text in seen if kind == "content")
        assert answered == "said"
        assert "<think>" not in answered and "secret" not in answered


class TestGenericReplies:
    def test_a_plain_openai_reply_is_untouched(self, server):
        instance = server(lambda m, p, b: json_reply(_completion(UNKNOWN_ID, "The answer is 4.")))
        reply = Client(instance.base_url).chat([])
        assert reply.content == "The answer is 4."
        assert reply.thinking is None
        assert reply.tool_calls is None

    def test_streaming_a_plain_reply_matches(self, server):
        whole, streamed, seen = _both_ways(
            server, UNKNOWN_ID, _completion(UNKNOWN_ID, "The answer is 4."),
            [{"role": "assistant"}, {"content": "The answer "}, {"content": "is 4."}],
        )
        assert whole.content == "The answer is 4."
        _same(whole, streamed)
        assert seen == [("content", "The answer "), ("content", "is 4.")]

    def test_an_unknown_model_still_gives_up_its_reasoning_field(self, server):
        """Detection is not required for correctness: the fallback reads both conventions."""
        instance = server(lambda m, p, b: json_reply(
            _completion(UNKNOWN_ID, "4", reasoning_content="add them up")))
        reply = Client(instance.base_url).chat([])
        assert reply.content == "4" and reply.thinking == "add them up"

    def test_an_unknown_model_still_gives_up_an_in_band_block(self, server):
        instance = server(lambda m, p, b: json_reply(
            _completion(UNKNOWN_ID, "<think>add them up</think>4")))
        reply = Client(instance.base_url).chat([])
        assert reply.content == "4" and reply.thinking == "add them up"

    def test_a_reply_with_no_model_id_at_all_still_reads(self, server):
        payload = _completion(UNKNOWN_ID, "<think>a</think>b", reasoning_content="c")
        payload.pop("model")
        instance = server(lambda m, p, b: json_reply(payload))
        reply = Client(instance.base_url).chat([])
        assert reply.content == "b" and reply.thinking == "c\na"


class TestFamilyOverride:
    def test_a_pinned_family_beats_the_model_id(self, server):
        """gpt-oss carries no in-band delimiter, so pinning it keeps the tag as text."""
        instance = server(lambda m, p, b: json_reply(_completion(UNKNOWN_ID, "<think>a</think>b")))
        assert Client(instance.base_url).chat([]).content == "b"
        pinned = Client(instance.base_url, family="gpt-oss").chat([])
        assert pinned.content == "<think>a</think>b" and pinned.thinking is None

    def test_a_pinned_family_beats_the_model_id_while_streaming(self, server):
        body = _sse(_stream_of(UNKNOWN_ID, [{"content": "<think>a</think>"}, {"content": "b"}]))
        instance = server(lambda m, p, b: (200, body))
        seen: list[tuple[str, str]] = []
        pinned = Client(instance.base_url, family="gpt-oss").chat(
            [], on_delta=lambda k, t: seen.append((k, t)))
        assert pinned.content == "<think>a</think>b"
        assert seen == [("content", "<think>a</think>"), ("content", "b")]

    def test_a_pinned_family_shapes_the_request(self, server):
        """A harmony template has no enable_thinking; it reads reasoning_effort."""
        seen = {}

        def handler(method, path, body):
            seen["body"] = json.loads(body)
            return json_reply(_completion(GPT_OSS_ID, "{}"))

        instance = server(handler)
        Client(instance.base_url, family="gpt-oss").extract(
            "x", {"type": "object", "properties": {}})
        assert seen["body"]["chat_template_kwargs"] == {"reasoning_effort": "low"}

    def test_the_served_model_id_shapes_the_request_when_nothing_is_pinned(self, server):
        seen = {}

        def handler(method, path, body):
            if path == "/v1/models":
                return json_reply({"data": [{"id": GPT_OSS_ID}]})
            seen["body"] = json.loads(body)
            return json_reply(_completion(GPT_OSS_ID, "{}"))

        instance = server(handler)
        Client(instance.base_url).extract("x", {"type": "object", "properties": {}})
        assert seen["body"]["chat_template_kwargs"] == {"reasoning_effort": "low"}

    def test_a_generic_server_keeps_the_enable_thinking_switch(self, server):
        seen = {}

        def handler(method, path, body):
            if path == "/v1/models":
                return json_reply({"data": [{"id": UNKNOWN_ID}]})
            seen["body"] = json.loads(body)
            return json_reply(_completion(UNKNOWN_ID, "{}"))

        instance = server(handler)
        Client(instance.base_url).extract("x", {"type": "object", "properties": {}})
        assert seen["body"]["chat_template_kwargs"] == {"enable_thinking": False}


class TestEmptyAnswerIsLoud:
    """A response that carried text and yielded nothing is a defect in this client, and
    names the fields it did not read."""

    def test_text_in_an_unrecognised_field_is_announced(self, server, caplog):
        payload = _completion(UNKNOWN_ID, "")
        payload["choices"][0]["message"]["output_text"] = "the whole answer"
        instance = server(lambda m, p, b: json_reply(payload))
        with caplog.at_level("WARNING", logger="ml_stack.client.chat"):
            reply = Client(instance.base_url).chat([])
        assert not (reply.content or "").strip()
        said = [r.getMessage() for r in caplog.records]
        assert len(said) == 1 and "output_text" in said[0]
        assert "the whole answer" not in said[0]

    def test_text_in_a_choice_level_field_is_announced(self, server, caplog):
        payload = _completion(UNKNOWN_ID, "")
        payload["choices"][0]["text"] = "the whole answer"
        instance = server(lambda m, p, b: json_reply(payload))
        with caplog.at_level("WARNING", logger="ml_stack.client.chat"):
            Client(instance.base_url).chat([])
        said = [r.getMessage() for r in caplog.records]
        assert len(said) == 1 and "text" in said[0]
        assert "the whole answer" not in said[0]

    def test_a_recovered_reply_stays_quiet(self, server, caplog):
        payload = _completion(GPT_OSS_ID, "", reasoning_content="counting the people")
        instance = server(lambda m, p, b: json_reply(payload))
        with caplog.at_level("WARNING", logger="ml_stack.client.chat"):
            reply = Client(instance.base_url).chat([])
        assert reply.thinking == "counting the people"
        assert caplog.records == []

    def test_a_genuinely_empty_reply_stays_quiet(self, server, caplog):
        instance = server(lambda m, p, b: json_reply(_completion(UNKNOWN_ID, "")))
        with caplog.at_level("WARNING", logger="ml_stack.client.chat"):
            Client(instance.base_url).chat([])
        assert caplog.records == []

    def test_a_tool_call_with_no_prose_stays_quiet(self, server, caplog):
        payload = _completion(UNKNOWN_ID, "", tool_calls=[
            {"id": "c1", "type": "function",
             "function": {"name": "look_up", "arguments": "{}"}}])
        instance = server(lambda m, p, b: json_reply(payload))
        with caplog.at_level("WARNING", logger="ml_stack.client.chat"):
            reply = Client(instance.base_url).chat([])
        assert reply.tool_calls is not None
        assert caplog.records == []

    def test_an_unread_streamed_field_is_announced_too(self, server, caplog):
        """Streaming says the same thing the whole reply does, and keeps the text in raw."""
        body = _sse(_stream_of(UNKNOWN_ID, [
            {"role": "assistant"}, {"output_text": "the whole "}, {"output_text": "answer"}]))
        instance = server(lambda m, p, b: (200, body))
        with caplog.at_level("WARNING", logger="ml_stack.client.chat"):
            reply = Client(instance.base_url).chat([], on_delta=lambda k, t: None)
        said = [r.getMessage() for r in caplog.records]
        assert len(said) == 1 and "output_text" in said[0]
        assert "the whole answer" not in said[0]
        assert reply.raw["choices"][0]["message"]["output_text"] == "the whole answer"


class TestStreamedToolCallsAcrossFamilies:
    def test_a_call_split_across_chunks_matches_the_whole_reply(self, server):
        call = {"id": "c9", "type": "function",
                "function": {"name": "look_up", "arguments": '{"text": "iron"}'}}
        payload = _completion(GPT_OSS_ID, "", tool_calls=[call])
        payload["choices"][0]["finish_reason"] = "tool_calls"
        whole, streamed, _ = _both_ways(
            server, GPT_OSS_ID, payload,
            [{"role": "assistant"},
             {"tool_calls": [{"index": 0, "id": "c9",
                              "function": {"name": "look_up", "arguments": '{"te'}}]},
             {"tool_calls": [{"index": 0, "function": {"arguments": 'xt": "iron"}'}}]}],
            finish="tool_calls",
        )
        assert whole.tool_calls == [call]
        _same(whole, streamed)

    def test_a_call_alongside_an_in_band_block_matches(self, server):
        call = {"id": "c1", "type": "function",
                "function": {"name": "look_at", "arguments": '{"ids": []}'}}
        payload = _completion(QWEN_ID, "<think>look first</think>", tool_calls=[call])
        payload["choices"][0]["finish_reason"] = "tool_calls"
        whole, streamed, seen = _both_ways(
            server, QWEN_ID, payload,
            [{"content": "<think>look "}, {"content": "first</think>"},
             {"tool_calls": [{"index": 0, "id": "c1",
                              "function": {"name": "look_at", "arguments": '{"ids'}}]},
             {"tool_calls": [{"index": 0, "function": {"arguments": '": []}'}}]}],
            finish="tool_calls",
        )
        assert whole.thinking == "look first" and whole.tool_calls == [call]
        _same(whole, streamed)
        assert [kind for kind, _ in seen] == ["thinking", "thinking"]


class TestHealth:
    def test_healthy_server_is_detected(self, server):
        instance = server(lambda m, p, b: json_reply({"status": "ok"}))
        assert is_healthy(instance.base_url)

    def test_dead_port_is_not(self):
        assert not is_healthy("http://127.0.0.1:1", timeout=0.5)

    def test_wait_returns_as_soon_as_the_server_answers(self, server):
        ready_at = time.monotonic() + 0.5

        def handler(method: str, path: str, body: bytes):
            if time.monotonic() < ready_at:
                return 503, b"loading"
            return json_reply({"status": "ok"})

        instance = server(handler)
        assert wait_for_health(instance.base_url, timeout=10.0)

    def test_process_death_ends_the_wait_immediately(self):
        """A bad model path makes a server exit in under a second. Polling a dead port
        for two minutes turns 'no such file' into 'timed out', which sends the reader
        looking for a slow load that never happened."""
        started = time.monotonic()
        alive = False

        assert not wait_for_health(
            "http://127.0.0.1:1", timeout=30.0, is_alive=lambda: alive
        )
        assert time.monotonic() - started < 2.0, "kept polling a dead process"

    def test_wait_times_out_when_nothing_ever_answers(self):
        assert not wait_for_health("http://127.0.0.1:1", timeout=0.6)

    def test_serving_params_reads_back_what_actually_ran(self, server):
        instance = server(lambda m, p, b: json_reply({
            "model_path": "/models/Qwen3-32B-Q5_K_M.gguf",
            "total_slots": 4,
            "default_generation_settings": {"n_ctx": 8192, "seed": 1234},
        }))
        params = serving_params(instance.base_url)
        assert params is not None
        assert (params.n_ctx, params.total_slots, params.seed) == (8192, 4, 1234)
        assert params.quant == "Q5_K_M"

    def test_sentinel_seed_is_discarded(self, server):
        """Reading -1 back as a real seed makes a run look reproducible when it is not."""
        instance = server(lambda m, p, b: json_reply(
            {"default_generation_settings": {"n_ctx": 512, "seed": -1}}))
        assert serving_params(instance.base_url).seed is None

    def test_no_server_is_none_but_an_odd_shape_is_not(self, server):
        """Collapsing these two turns a llama.cpp version bump into a silent loss of
        provenance."""
        assert serving_params("http://127.0.0.1:1", timeout=0.5) is None

        instance = server(lambda m, p, b: json_reply({"unfamiliar": "shape"}))
        params = serving_params(instance.base_url)
        assert params is not None and params.n_ctx is None

    def test_reported_models_enables_adoption(self, server):
        instance = server(lambda m, p, b: json_reply(
            {"data": [{"id": "qwen3"}, {"id": "gemma4"}, {"no_id": 1}]}))
        assert reported_models(instance.base_url) == ["qwen3", "gemma4"]

    @pytest.mark.parametrize(
        "path,expected",
        [("/m/Qwen3-32B-Q5_K_M.gguf", "Q5_K_M"),
         ("model-Q8_0.gguf", "Q8_0"),
         ("weights-BF16.gguf", "BF16"),
         ("plain.gguf", None)],
    )
    def test_quant_is_recovered_from_the_filename(self, path, expected):
        """The server does not report quantisation, and it is the single most important
        thing to record when comparing two benchmark runs."""
        assert quant_from_model_path(path) == expected


class TestRetry:
    def test_retries_a_local_500_and_succeeds(self, server):
        """Local servers commonly answer 500 to a request that overlaps another."""
        state = {"calls": 0}

        def handler(method: str, path: str, body: bytes):
            state["calls"] += 1
            if state["calls"] < 3:
                return 500, b"busy"
            return json_reply({"data": [{"embedding": [0.1, 0.2]}]})

        instance = server(handler)
        vectors = embed("hello", base_url=instance.base_url, tries=3)
        assert vectors == [[0.1, 0.2]]
        assert state["calls"] == 3

    def test_never_retries_a_4xx(self, server):
        """The request is wrong and will stay wrong; hammering only delays the report."""
        state = {"calls": 0}

        def handler(method: str, path: str, body: bytes):
            state["calls"] += 1
            return 400, b'{"error": "bad request"}'

        instance = server(handler)
        with pytest.raises(ServerError):
            embed("x", base_url=instance.base_url, tries=5)
        assert state["calls"] == 1

    def test_unreachable_raises_the_specific_error(self):
        with pytest.raises(ServerUnreachable):
            embed("x", base_url="http://127.0.0.1:1", tries=1, timeout=0.5)


class TestChat:
    def test_round_trip(self, server):
        instance = server(lambda m, p, b: _chat_reply("hello there"))
        reply = Client(instance.base_url).chat([{"role": "user", "content": "hi"}])
        assert reply.content == "hello there"

    def test_thinking_is_split_off_the_content(self, server):
        instance = server(lambda m, p, b: _chat_reply("<think>hmm</think>42"))
        reply = Client(instance.base_url).chat([])
        assert reply.content == "42" and reply.thinking == "hmm"

    def test_the_request_actually_carries_the_slot(self, server):
        instance = server(lambda m, p, b: _chat_reply("ok"))
        Client(instance.base_url, slot=7).chat([{"role": "user", "content": "x"}])
        _, path, body = instance.requests[-1]
        assert path == "/v1/chat/completions"
        assert json.loads(body)["id_slot"] == 7


    def test_streaming_deltas_are_reported_and_assembled(self, server):
        chunks = [
            {"choices": [{"delta": {"reasoning_content": "hmm "}}]},
            {"choices": [{"delta": {"reasoning_content": "ok"}}]},
            {"choices": [{"delta": {"content": "4"}}]},
            {"choices": [{"delta": {"content": "2"}, "finish_reason": "stop"}]},
        ]
        body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
        instance = server(lambda m, p, b: (200, body.encode()))
        got: list[tuple[str, str]] = []
        reply = Client(instance.base_url).chat(
            [{"role": "user", "content": "x"}], on_delta=lambda k, t: got.append((k, t)))
        assert reply.content == "42" and reply.thinking == "hmm ok"
        assert reply.finish_reason == "stop"
        assert got == [("thinking", "hmm "), ("thinking", "ok"),
                       ("content", "4"), ("content", "2")]
        assert json.loads(instance.requests[-1][2])["stream"] is True

    def test_streamed_tool_calls_are_assembled_across_chunks(self, server):
        chunks = [
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c9", "function": {
                "name": "look_up", "arguments": "{\"te"}}]}}]},
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {
                "arguments": "xt\": \"iron\"}"}}]}, "finish_reason": "tool_calls"}]},
        ]
        body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
        instance = server(lambda m, p, b: (200, body.encode()))
        reply = Client(instance.base_url).chat([], on_delta=lambda k, t: None)
        assert reply.tool_calls == [{"id": "c9", "type": "function", "function": {
            "name": "look_up", "arguments": '{"text": "iron"}'}}]


class TestGrammarTripwire:
    def test_passes_when_the_grammar_is_honoured(self, server):
        instance = server(lambda m, p, b: json_reply({"content": "ok"}))
        Client(instance.base_url).assert_grammar_support()

    def test_sends_a_non_empty_prompt(self, server):
        """llama-server predicts zero tokens for an empty prompt, grammar or not."""
        instance = server(lambda m, p, b: json_reply({"content": "ok"}))
        Client(instance.base_url).assert_grammar_support()
        _, _, body = instance.requests[-1]
        assert json.loads(body)["prompt"].strip()

    def test_fails_loudly_when_the_grammar_is_ignored(self, server):
        """A server that ignores GBNF does not error -- it returns fluent prose where a
        single token was required, and every downstream parse produces confident nonsense."""
        instance = server(lambda m, p, b: json_reply(
            {"content": "Certainly! I would be happy to help you with that."}))
        with pytest.raises(GrammarUnsupportedError, match="not working"):
            Client(instance.base_url).assert_grammar_support()

    def test_fails_when_the_server_is_absent(self):
        with pytest.raises(GrammarUnsupportedError, match="could not run"):
            Client("http://127.0.0.1:1", timeout=0.5).assert_grammar_support()

    def test_budget_exhaustion_retries_once_at_double_with_a_fresh_seed(self, server):
        """The fresh seed matters: the same seed re-walks the same path into the same
        ceiling."""
        seen: list[dict] = []

        def handler(method: str, path: str, body: bytes):
            payload = json.loads(body)
            seen.append(payload)
            if len(seen) == 1:
                return json_reply({"content": "{partial", "stopped_limit": True})
            return json_reply({"content": '{"done": true}'})

        instance = server(handler)
        out = Client(instance.base_url).complete("x", grammar="root ::= .+", n_predict=64)

        assert out == '{"done": true}'
        assert len(seen) == 2
        assert seen[1]["n_predict"] == 128, "did not double the budget"
        assert seen[1].get("seed") != seen[0].get("seed"), "reused the seed that hit the ceiling"

    def test_no_retry_without_a_grammar(self, server):
        """Plain truncation is short, not unparseable -- a different problem."""
        calls = {"n": 0}

        def handler(method: str, path: str, body: bytes):
            if path.endswith("/completion"):     # the family probe is not a retry
                calls["n"] += 1
            return json_reply({"content": "cut off", "stopped_limit": True})

        instance = server(handler)
        Client(instance.base_url).complete("x", n_predict=16)
        assert calls["n"] == 1


class TestEmbeddings:
    def test_dimension_mismatch_raises_rather_than_padding(self, server):
        """Padding or trimming to the expected width so an insert can never fail turns a
        model mismatch into a store full of garbage vectors that still cosine-compare."""
        instance = server(lambda m, p, b: json_reply({"data": [{"embedding": [0.1] * 384}]}))
        with pytest.raises(EmbeddingError, match="corrupt the index"):
            embed("x", base_url=instance.base_url, expect_dim=768, tries=1)

    def test_unwraps_a_nested_pooled_vector(self, server):
        instance = server(lambda m, p, b: json_reply({"data": [{"embedding": [[1.0, 2.0]]}]}))
        assert embed("x", base_url=instance.base_url) == [[1.0, 2.0]]

    def test_a_short_batch_is_an_error_not_a_silent_truncation(self, server):
        instance = server(lambda m, p, b: json_reply({"data": [{"embedding": [1.0]}]}))
        with pytest.raises(EmbeddingError, match="asked for 2"):
            embed(["a", "b"], base_url=instance.base_url, tries=1)

    def test_empty_input_does_not_hit_the_network(self, server):
        instance = server(lambda m, p, b: json_reply({"data": []}))
        assert embed([], base_url=instance.base_url) == []
        assert not instance.requests

    def test_cosine_of_identical_vectors_is_one(self):
        from ml_stack.client import cosine

        assert cosine([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)

    def test_cosine_rejects_a_dimension_mismatch(self):
        from ml_stack.client import cosine

        with pytest.raises(EmbeddingError):
            cosine([1.0], [1.0, 2.0])

    def test_rank_pairs_returns_every_pair_once_best_first(self):
        from ml_stack.client import rank_pairs

        vectors = [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]]
        pairs = rank_pairs(vectors)
        assert [(i, j) for i, j, _ in pairs] == [(0, 1), (1, 2), (0, 2)]
        assert pairs[0][2] > pairs[-1][2]

    def test_rank_pairs_breaks_ties_by_index_so_the_order_is_stable(self):
        from ml_stack.client import rank_pairs

        vectors = [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]
        assert [(i, j) for i, j, _ in rank_pairs(vectors)] == [(0, 1), (0, 2), (1, 2)]

    def test_rank_pairs_limit_keeps_the_best(self):
        from ml_stack.client import rank_pairs

        vectors = [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]]
        assert [(i, j) for i, j, _ in rank_pairs(vectors, limit=1)] == [(0, 1)]

    def test_rank_pairs_of_one_vector_is_empty(self):
        from ml_stack.client import rank_pairs

        assert rank_pairs([[1.0, 0.0]]) == []


class TestTokenEstimate:
    def test_over_counts_rather_than_under(self):
        """A prompt the assembler believes fits must always fit, so the estimate has to
        err high. Real Qwen BPE on this sentence is around 14 tokens."""
        text = "The quick brown fox jumps over the lazy dog and keeps running."
        assert estimate_tokens(text) >= len(text.split())

    def test_empty_is_zero(self):
        assert estimate_tokens("") == 0

    def test_punctuation_soup_costs_more_than_prose_of_the_same_length(self):
        assert heuristic_tokens("!!!???<<<>>>...") > heuristic_tokens("aaaaaaaaaaaaaaa")

    def test_an_installed_counter_is_used_and_can_be_removed(self):
        from ml_stack.client import set_token_counter

        try:
            set_token_counter(lambda _: 99)
            assert estimate_tokens("anything") == 99
            assert heuristic_tokens("anything") != 99, "the heuristic must stay reachable"
        finally:
            set_token_counter(None)
        assert estimate_tokens("anything") != 99


class TestTokenize:
    """The server's tokenizer is the one the model actually reads with. When it
    disagrees with the tokenizer the model was TRAINED with, nothing errors --
    generation stays fluent while every sequence means something else."""

    def test_returns_the_servers_ids(self, server):
        instance = server(lambda m, p, b: json_reply({"tokens": [1, 2087, 9]}))
        assert Client(instance.base_url).tokenize("go") == [1, 2087, 9]

    def test_with_pieces_is_forwarded_so_a_mismatch_can_be_explained(self, server):
        seen = {}

        def handler(method, path, body):
            seen["path"] = path
            seen["body"] = json.loads(body)
            return json_reply({"tokens": [{"id": 2087, "piece": "go"}]})

        instance = server(handler)
        out = Client(instance.base_url).tokenize("go", with_pieces=True)
        assert seen["path"] == "/tokenize"
        assert seen["body"]["with_pieces"] is True
        assert out == [{"id": 2087, "piece": "go"}]

    def test_a_response_without_tokens_is_empty_not_a_crash(self, server):
        instance = server(lambda m, p, b: json_reply({}))
        assert Client(instance.base_url).tokenize("go") == []

    def test_detokenize_round_trips_through_content(self, server):
        instance = server(lambda m, p, b: json_reply({"content": "<user> go"}))
        assert Client(instance.base_url).detokenize([1, 2, 3]) == "<user> go"

    def test_detokenize_posts_the_ids_it_was_given(self, server):
        seen = {}

        def handler(method, path, body):
            seen["path"] = path
            seen["body"] = json.loads(body)
            return json_reply({"content": "x"})

        instance = server(handler)
        Client(instance.base_url).detokenize((4, 5))
        assert seen["path"] == "/detokenize"
        assert seen["body"]["tokens"] == [4, 5]


class TestExtract:
    """A schema in, a dict out, over a real socket. The default path goes through the
    server's chat template; ``prompt=`` keeps the raw endpoint under a grammar."""

    SCHEMA = {
        "type": "object",
        "properties": {
            "people": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"name": {"type": "string"},
                                   "role": {"type": "string"}},
                },
            },
        },
    }

    @staticmethod
    def chat_reply(content):
        return json_reply({"choices": [{"message": {"role": "assistant",
                                                    "content": content}}]})

    def test_a_document_comes_back_as_a_dict(self, server):
        found = {"people": [{"name": "Ada", "role": "engineer"}]}
        instance = server(lambda m, p, b: self.chat_reply(json.dumps(found)))

        out = Client(instance.base_url).extract("Ada is an engineer.", self.SCHEMA)
        assert out == found

    def test_it_asks_the_chat_endpoint_under_the_schema_with_thinking_off(self, server):
        seen = {}

        def handler(method, path, body):
            seen["path"] = path
            seen["body"] = json.loads(body)
            return self.chat_reply('{"people": []}')

        instance = server(handler)
        out = Client(instance.base_url).extract("Ada is an engineer.", self.SCHEMA,
                                                instructions="Pull out the people.",
                                                n_predict=256)

        assert out == {"people": []}
        assert seen["path"] == "/v1/chat/completions"
        assert seen["body"]["response_format"]["type"] == "json_schema"
        from ml_stack.client.chat import strict_schema
        assert seen["body"]["response_format"]["json_schema"]["schema"] == strict_schema(self.SCHEMA)
        assert seen["body"]["response_format"]["json_schema"]["name"] == "extraction"
        assert seen["body"]["chat_template_kwargs"]["enable_thinking"] is False
        assert seen["body"]["n_predict"] == 256
        assert seen["body"]["messages"] == [
            {"role": "system", "content": "Pull out the people."},
            {"role": "user", "content": "Ada is an engineer."},
        ]

    def test_the_default_system_message_asks_for_json_and_no_prose(self, server):
        seen = {}

        def handler(method, path, body):
            seen["body"] = json.loads(body)
            return self.chat_reply("{}")

        instance = server(handler)
        Client(instance.base_url).extract("Ada is an engineer.", self.SCHEMA)

        assert seen["body"]["messages"][0] == {
            "role": "system",
            "content": "Return only a JSON document that matches the schema. No prose.",
        }

    def test_thinking_can_be_turned_back_on(self, server):
        seen = {}

        def handler(method, path, body):
            seen["body"] = json.loads(body)
            return self.chat_reply("{}")

        instance = server(handler)
        Client(instance.base_url).extract("Ada is an engineer.", self.SCHEMA, think=True)

        assert seen["body"]["chat_template_kwargs"]["enable_thinking"] is True

    def test_a_schema_name_reaches_the_response_format(self, server):
        seen = {}

        def handler(method, path, body):
            seen["body"] = json.loads(body)
            return self.chat_reply("{}")

        instance = server(handler)
        Client(instance.base_url).extract("Ada is an engineer.", self.SCHEMA,
                                          schema_name="people")

        assert seen["body"]["response_format"]["json_schema"]["name"] == "people"

    def test_supplied_messages_are_sent_word_for_word(self, server):
        seen: list[dict] = []
        convo = [{"role": "system", "content": "You index people."},
                 {"role": "user", "content": "Ada is an engineer."},
                 {"role": "assistant", "content": "Understood."}]

        def handler(method, path, body):
            seen.append(json.loads(body))
            return self.chat_reply("{}")

        instance = server(handler)
        Client(instance.base_url).extract("ignored", self.SCHEMA,
                                          instructions="ignored", messages=convo)

        assert seen[0]["messages"] == convo

    def test_malformed_json_is_retried_once_with_a_fresh_seed(self, server):
        seen: list[dict] = []

        def handler(method, path, body):
            seen.append(json.loads(body))
            if len(seen) == 1:
                return self.chat_reply("Here you go: {people:")
            return self.chat_reply('{"people": []}')

        instance = server(handler)
        out = Client(instance.base_url).extract("Ada is an engineer.", self.SCHEMA)

        assert out == {"people": []}
        assert len(seen) == 2
        assert seen[1].get("seed") != seen[0].get("seed")

    def test_two_malformed_replies_raise_with_what_came_back(self, server):
        instance = server(lambda m, p, b: self.chat_reply("I cannot do that."))

        with pytest.raises(ServerError, match="I cannot do that."):
            Client(instance.base_url).extract("Ada is an engineer.", self.SCHEMA)

    def test_the_raw_text_in_the_error_is_truncated(self, server):
        instance = server(lambda m, p, b: self.chat_reply("no " * 500))

        with pytest.raises(ServerError) as exc:
            Client(instance.base_url).extract("x", self.SCHEMA)
        assert len(str(exc.value)) < 300

    def test_an_accepted_answer_is_returned_without_asking_again(self, server):
        found = {"people": [{"name": "Ada", "role": "engineer"}]}
        seen: list[dict] = []

        def handler(method, path, body):
            seen.append(json.loads(body))
            return self.chat_reply(json.dumps(found))

        instance = server(handler)
        out = Client(instance.base_url).extract("Ada is an engineer.", self.SCHEMA,
                                                check=lambda obj: [])

        assert out == found
        assert "_objections" not in out
        assert len(seen) == 1

    def test_a_rejected_answer_comes_back_as_a_turn_of_its_own(self, server):
        seen: list[dict] = []

        def handler(method, path, body):
            seen.append(json.loads(body))
            if len(seen) == 1:
                return self.chat_reply('{"people": []}')
            return self.chat_reply('{"people": [{"name": "Ada"}]}')

        instance = server(handler)
        out = Client(instance.base_url).extract(
            "Ada is an engineer.", self.SCHEMA,
            check=lambda obj: [] if obj["people"] else ["no people were found"])

        assert out == {"people": [{"name": "Ada"}]}
        assert "_objections" not in out
        assert len(seen) == 2

        messages = seen[1]["messages"]
        assert messages[0]["role"] == "system"
        assert messages[:2] == seen[0]["messages"]
        assert messages[2] == {"role": "assistant", "content": '{"people": []}'}
        assert messages[3]["role"] == "user"
        assert "no people were found" in messages[3]["content"]
        assert "no people were found" not in messages[2]["content"]

    def test_objections_that_survive_every_try_come_back_on_the_object(self, server):
        seen: list[dict] = []

        def handler(method, path, body):
            seen.append(json.loads(body))
            return self.chat_reply('{"people": []}')

        instance = server(handler)
        out = Client(instance.base_url).extract(
            "Ada is an engineer.", self.SCHEMA, tries=3,
            check=lambda obj: ["no people were found", "role is missing"])

        assert len(seen) == 3
        assert out["people"] == []
        assert out["_objections"] == ["no people were found", "role is missing"]

    def test_one_try_returns_the_rejected_answer_without_asking_again(self, server):
        seen: list[dict] = []

        def handler(method, path, body):
            seen.append(json.loads(body))
            return self.chat_reply('{"people": []}')

        instance = server(handler)
        out = Client(instance.base_url).extract(
            "Ada is an engineer.", self.SCHEMA, tries=1,
            check=lambda obj: ["no people were found"])

        assert len(seen) == 1
        assert out["_objections"] == ["no people were found"]

    def test_a_supplied_prompt_asks_the_completion_endpoint_with_the_grammar(self, server):
        from ml_stack.contracts import grammar_for

        seen: list[dict] = []

        def handler(method, path, body):
            seen.append((path, json.loads(body)))
            return json_reply({"content": "{}"})

        instance = server(handler)
        Client(instance.base_url).extract("Ada is an engineer.", self.SCHEMA,
                                          instructions="ignored", n_predict=256,
                                          prompt="<|user|>find the people<|assistant|>")

        assert seen[0][0] == "/completion"
        assert seen[0][1]["prompt"] == "<|user|>find the people<|assistant|>"
        assert seen[0][1]["grammar"] == grammar_for(self.SCHEMA)
        assert seen[0][1]["n_predict"] == 256

    def test_the_rejection_goes_before_the_assistant_opener(self, server):
        seen: list[dict] = []

        def handler(method, path, body):
            seen.append(json.loads(body))
            return json_reply({"content": '{"people": []}'})

        instance = server(handler)
        base = "<|im_start|>user\nfind the people<|im_end|>\n<|im_start|>assistant\n"
        Client(instance.base_url).extract(
            "Ada is an engineer.", self.SCHEMA, prompt=base,
            check=lambda obj: ["no people were found"])

        retried = seen[1]["prompt"]
        assert len(seen) == 2
        assert retried.endswith("<|im_start|>assistant\n")
        assert "no people were found" in retried
        assert retried.index("no people were found") < retried.index(
            "<|im_start|>assistant\n")
        assert retried.startswith("<|im_start|>user\nfind the people<|im_end|>\n")

    def test_a_rejection_with_no_assistant_opener_is_appended(self, server):
        seen: list[dict] = []

        def handler(method, path, body):
            seen.append(json.loads(body))
            return json_reply({"content": '{"people": []}'})

        instance = server(handler)
        Client(instance.base_url).extract(
            "Ada is an engineer.", self.SCHEMA, prompt="find the people\nJSON:\n",
            check=lambda obj: ["no people were found"])

        assert seen[1]["prompt"].startswith(seen[0]["prompt"])
        assert "no people were found" in seen[1]["prompt"]

    def test_a_schema_it_cannot_constrain_never_reaches_the_server(self, server):
        from ml_stack.contracts import ContractError

        calls = {"n": 0}

        def handler(method, path, body):
            calls["n"] += 1
            return json_reply({"content": "{}"})

        instance = server(handler)
        with pytest.raises(ContractError):
            Client(instance.base_url).extract(
                "x", {"type": "object", "properties": {"a": {"type": "date"}}},
                prompt="find it\nJSON:\n")
        assert calls["n"] == 0



class TestStrictSchema:
    def test_every_object_requires_all_its_properties_and_nothing_else(self, server):
        """Without ``required`` the server's schema-to-grammar makes every key optional,
        and a model that skips ``title`` produces a record nothing downstream can use."""
        from ml_stack.client.chat import strict_schema
        schema = {"type": "object", "properties": {"items": {"type": "array", "items": {
            "type": "object", "properties": {"title": {"type": "string"}, "kind": {"type": "string"}}}}}}
        out = strict_schema(schema)
        assert out["required"] == ["items"] and out["additionalProperties"] is False
        inner = out["properties"]["items"]["items"]
        assert inner["required"] == ["title", "kind"] and inner["additionalProperties"] is False
        assert "required" not in schema  # the caller's schema is left alone

    def test_the_chat_request_carries_the_hardened_schema(self, server):
        instance = server(lambda m, p, b: json_reply({"choices": [{"message": {"content": "{\"a\": \"x\"}"}}]}))
        Client(instance.base_url).extract("t", {"type": "object", "properties": {"a": {"type": "string"}}})
        _, _, body = instance.requests[-1]
        sent = json.loads(body)["response_format"]["json_schema"]["schema"]
        assert sent["required"] == ["a"] and sent["additionalProperties"] is False


def test_a_model_card_informs_but_is_never_sent_on_its_own():
    """A card is general advice from a publisher who does not know the task.

    Applying it silently would mean the library overruling a caller who measured: gemma-4's
    card asks for temperature 1.0, which on a tool-calling task measured 15 points worse
    than greedy. So it is readable, and it is never what goes out.
    """
    from ml_stack.client.chat import Client
    from ml_stack.client.families import GEMMA, GPT_OSS

    gemma = Client("http://nowhere.invalid", family=GEMMA)
    assert gemma.card == {"temperature": 1.0, "top_p": 0.95, "top_k": 64}
    assert gemma.sampling == {"temperature": 0.0}          # greedy, whatever the card says
    assert "top_p" not in gemma.build_body([{"role": "user", "content": "x"}])

    # a caller who chooses gets what they chose, and only that
    chosen = Client("http://nowhere.invalid", family=GEMMA, temperature=0.7, top_k=40)
    assert chosen.sampling == {"temperature": 0.7, "top_k": 40}

    # and a card that says nothing leaves an empty record rather than an invented one
    oss = Client("http://nowhere.invalid", family=GPT_OSS)
    assert oss.card == {}
    assert oss.sampling == {"temperature": 0.0}


def test_the_card_comes_from_the_served_model_before_the_family(monkeypatch, tmp_path):
    """A family table is a guess about a whole lineage; the GGUF is the file being served.
    Qwen3.8-Flash-Next asks for top_k 20 where gemma-4 asks for 64 -- one table cannot
    hold both, and the file already knows."""
    import struct

    from ml_stack.client.chat import Client
    from ml_stack.client.families import GEMMA

    blob = bytearray(b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0)
                     + struct.pack("<Q", 1))
    key = b"general.sampling.top_k"
    blob += struct.pack("<Q", len(key)) + key + struct.pack("<I", 6) + struct.pack("<f", 20.0)
    where = tmp_path / "served.gguf"
    where.write_bytes(bytes(blob))

    import ml_stack.client.chat as chat
    monkeypatch.setattr(chat, "request_json", lambda *a, **k: {"model_path": str(where)},
                        raising=False)

    served = Client("http://nowhere.invalid", family=GEMMA)
    monkeypatch.setattr(served, "_from_gguf", lambda: {"top_k": 20.0})
    assert served.card == {"top_k": 20.0}, "the file being served wins"
    assert served.sampling == {"temperature": 0.0}, "and none of it is sent"

    # a server that will not say falls back to what the family knows
    quiet = Client("http://nowhere.invalid", family=GEMMA)
    monkeypatch.setattr(quiet, "_from_gguf", lambda: {})
    assert quiet.card == {"temperature": 1.0, "top_p": 0.95, "top_k": 64}


def test_think_becomes_the_familys_template_flag_and_never_a_body_key():
    """`chat(think=False)` must reach the server as the chat template's own switch;
    as a body key it was ignored and the model thought anyway (Flash-Next, 2026-09-02)."""
    from ml_stack.client import families
    from ml_stack.client.chat import Client

    client = Client("http://127.0.0.1:1", family="qwen")
    body = client.build_body([{"role": "user", "content": "hi"}], think=False)
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert "think" not in body
    body = client.build_body([{"role": "user", "content": "hi"}], think=True,
                             chat_template_kwargs={"other": 1})
    assert body["chat_template_kwargs"] == {"other": 1, "enable_thinking": True}
    plain = Client("http://127.0.0.1:1", family=families.GENERIC.name)
    body = plain.build_body([{"role": "user", "content": "hi"}], think=False)
    assert "think" not in body      # whatever the family's switch is, the word itself never goes
