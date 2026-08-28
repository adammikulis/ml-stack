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
        body = Client("https://api.openai.com/v1", slot=1).build_body([], top_k=40)
        assert "top_k" not in body
        assert "n_predict" not in body and body["max_tokens"] == 512
        assert "id_slot" not in body

    def test_keeps_them_for_a_local_server(self):
        body = Client("http://127.0.0.1:8080").build_body([], top_k=40)
        assert body["top_k"] == 40 and body["n_predict"] == 512

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
    """A schema in, a dict out, over a real socket. The grammar the server is sent is
    the one ``grammar_for`` built; nothing here reimplements it."""

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

    def test_a_document_comes_back_as_a_dict(self, server):
        found = {"people": [{"name": "Ada", "role": "engineer"}]}
        instance = server(lambda m, p, b: json_reply({"content": json.dumps(found)}))

        out = Client(instance.base_url).extract("Ada is an engineer.", self.SCHEMA)
        assert out == found

    def test_it_asks_the_completion_endpoint_with_the_grammar(self, server):
        from ml_stack.contracts import grammar_for

        seen = {}

        def handler(method, path, body):
            seen["path"] = path
            seen["body"] = json.loads(body)
            return json_reply({"content": "{}"})

        instance = server(handler)
        Client(instance.base_url).extract("Ada is an engineer.", self.SCHEMA,
                                          instructions="Pull out the people.",
                                          n_predict=256)

        assert seen["path"] == "/completion"
        assert seen["body"]["grammar"] == grammar_for(self.SCHEMA)
        assert seen["body"]["n_predict"] == 256
        assert seen["body"]["prompt"] == (
            "Pull out the people.\n\nText:\nAda is an engineer.\n\nJSON:\n")

    def test_malformed_json_is_retried_once_with_a_fresh_seed(self, server):
        """A server that ignores the grammar returns prose. The same seed returns the
        same prose."""
        seen: list[dict] = []

        def handler(method, path, body):
            seen.append(json.loads(body))
            if len(seen) == 1:
                return json_reply({"content": "Here you go: {people:"})
            return json_reply({"content": '{"people": []}'})

        instance = server(handler)
        out = Client(instance.base_url).extract("Ada is an engineer.", self.SCHEMA)

        assert out == {"people": []}
        assert len(seen) == 2
        assert seen[1].get("seed") != seen[0].get("seed")

    def test_two_malformed_replies_raise_with_what_came_back(self, server):
        instance = server(lambda m, p, b: json_reply({"content": "I cannot do that."}))

        with pytest.raises(ServerError, match="I cannot do that."):
            Client(instance.base_url).extract("Ada is an engineer.", self.SCHEMA)

    def test_the_raw_text_in_the_error_is_truncated(self, server):
        instance = server(lambda m, p, b: json_reply({"content": "no " * 500}))

        with pytest.raises(ServerError) as exc:
            Client(instance.base_url).extract("x", self.SCHEMA)
        assert len(str(exc.value)) < 300

    def test_an_accepted_answer_is_returned_without_asking_again(self, server):
        found = {"people": [{"name": "Ada", "role": "engineer"}]}
        seen: list[dict] = []

        def handler(method, path, body):
            seen.append(json.loads(body))
            return json_reply({"content": json.dumps(found)})

        instance = server(handler)
        out = Client(instance.base_url).extract("Ada is an engineer.", self.SCHEMA,
                                                check=lambda obj: [])

        assert out == found
        assert "_objections" not in out
        assert len(seen) == 1

    def test_a_rejected_answer_is_asked_again_with_the_objection(self, server):
        seen: list[dict] = []

        def handler(method, path, body):
            seen.append(json.loads(body))
            if len(seen) == 1:
                return json_reply({"content": '{"people": []}'})
            return json_reply({"content": '{"people": [{"name": "Ada"}]}'})

        instance = server(handler)
        out = Client(instance.base_url).extract(
            "Ada is an engineer.", self.SCHEMA,
            check=lambda obj: [] if obj["people"] else ["no people were found"])

        assert out == {"people": [{"name": "Ada"}]}
        assert "_objections" not in out
        assert len(seen) == 2
        assert "no people were found" in seen[1]["prompt"]
        assert seen[1]["prompt"].startswith(seen[0]["prompt"])

    def test_objections_that_survive_every_try_come_back_on_the_object(self, server):
        seen: list[dict] = []

        def handler(method, path, body):
            seen.append(json.loads(body))
            return json_reply({"content": '{"people": []}'})

        instance = server(handler)
        out = Client(instance.base_url).extract(
            "Ada is an engineer.", self.SCHEMA, tries=3,
            check=lambda obj: ["no people were found", "role is missing"])

        assert len(seen) == 3
        assert out["people"] == []
        assert out["_objections"] == ["no people were found", "role is missing"]

    def test_a_supplied_prompt_is_sent_word_for_word(self, server):
        seen: list[dict] = []

        def handler(method, path, body):
            seen.append(json.loads(body))
            return json_reply({"content": "{}"})

        instance = server(handler)
        Client(instance.base_url).extract("Ada is an engineer.", self.SCHEMA,
                                          instructions="ignored",
                                          prompt="<|user|>find the people<|assistant|>")

        assert seen[0]["prompt"] == "<|user|>find the people<|assistant|>"

    def test_one_try_returns_the_rejected_answer_without_asking_again(self, server):
        seen: list[dict] = []

        def handler(method, path, body):
            seen.append(json.loads(body))
            return json_reply({"content": '{"people": []}'})

        instance = server(handler)
        out = Client(instance.base_url).extract(
            "Ada is an engineer.", self.SCHEMA, tries=1,
            check=lambda obj: ["no people were found"])

        assert len(seen) == 1
        assert out["_objections"] == ["no people were found"]

    def test_a_schema_it_cannot_constrain_never_reaches_the_server(self, server):
        from ml_stack.contracts import ContractError

        calls = {"n": 0}

        def handler(method, path, body):
            calls["n"] += 1
            return json_reply({"content": "{}"})

        instance = server(handler)
        with pytest.raises(ContractError):
            Client(instance.base_url).extract(
                "x", {"type": "object", "properties": {"a": {"type": "date"}}})
        assert calls["n"] == 0
