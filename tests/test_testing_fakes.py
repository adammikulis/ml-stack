"""The shared fakes carry the real signatures, and the diff that keeps them there.

A fake with ``**kwargs`` accepts what the real thing refuses; the suite went green on a
keyword that took an 87G load down. Every fake in `ml_stack.testing.fakes` is diffed here
against what it stands in for, and the diff itself is tested on fakes built to drift.
"""

from __future__ import annotations

import inspect
import json

import pytest
from ml_stack.client import Client, Reply
from ml_stack.serve import ServerFailed, ServerInfo, ServerSpec
from ml_stack.serve.preflight import Report
from ml_stack.testing import (
    MIRRORED,
    FakeClient,
    FakePreflight,
    FakeReport,
    FakeServe,
    ScriptedModel,
    drift,
    fake_serve,
    mirrors,
)


# ---------------------------------------------------------------- the diff

@pytest.mark.parametrize("label, fake, real", MIRRORED, ids=[m[0] for m in MIRRORED])
def test_every_fake_mirrors_the_real_signature(label, fake, real):
    """Names, kinds and defaults agree; a required parameter is never left out; nothing
    variadic is taken that the real one does not take. Mutation: add ``**kw`` to a fake."""
    assert mirrors(fake, real), f"{label}: " + "; ".join(drift(fake, real))


def test_every_fake_in_the_module_is_in_the_table():
    """A fake added to the module and not to `MIRRORED` is a fake nothing checks."""
    from ml_stack.testing import fakes

    listed = {label.split(".")[0] for label, _, _ in MIRRORED}
    public = {name for name in fakes.__all__
              if name[0].isupper() or name.startswith("fake_")}
    # FakeReport is a Report, not a stand-in for one: its own test is `isinstance`.
    assert public - {"FakeReport"} == listed


def _real(base_url, *, timeout=None, slot=None):
    pass


def test_a_fake_taking_kwargs_the_real_one_lacks_is_drift():
    """The failure this module exists for."""
    def fake(base_url, **kwargs):
        pass

    assert not mirrors(fake, _real)
    assert any("**kwargs" in line for line in drift(fake, _real))


def test_a_fake_taking_a_name_the_real_one_lacks_is_drift():
    def fake(base_url, *, timeout=None, tight=False):
        pass

    assert drift(fake, _real) == ["takes 'tight', which the real one does not"]


def test_a_fake_with_a_different_kind_or_default_is_drift():
    def positional(base_url, timeout=None):
        pass

    def other_default(base_url, *, timeout=30.0):
        pass

    assert "keyword-only" in drift(positional, _real)[0]
    assert "defaults to 30.0" in drift(other_default, _real)[0]


def test_a_fake_may_leave_out_an_optional_parameter_but_not_a_required_one():
    def fewer(base_url):
        pass

    def none():
        pass

    assert mirrors(fewer, _real)
    assert drift(none, _real) == ["leaves out 'base_url', which the real one requires"]


def test_a_fake_may_take_kwargs_when_the_real_one_does():
    def real(messages, **extra):
        pass

    def fake(messages, **anything):
        pass

    assert mirrors(fake, real)


def test_an_unbound_method_is_compared_on_what_a_caller_passes():
    """``self`` is not a parameter of the call; a plain function stands in for a method."""
    class Thing:
        def go(self, x, *, y=1):
            pass

    def go(x, *, y=1):
        pass

    assert mirrors(Thing.go, go)
    assert mirrors(go, Thing.go)


# ---------------------------------------------------------------- FakeClient

def test_the_fake_client_refuses_what_the_real_one_refuses():
    with pytest.raises(TypeError, match="tight"):
        FakeClient("http://127.0.0.1:1", tight=True)
    with pytest.raises(TypeError, match="tight"):
        Client("http://127.0.0.1:1", tight=True)
    assert inspect.signature(FakeClient.__init__) == inspect.signature(Client.__init__)


def test_the_fake_client_samples_and_cards_as_the_real_one_shapes_them():
    c = FakeClient("http://127.0.0.1:1/", top_k=40)
    assert c.base_url == "http://127.0.0.1:1"
    assert c.sampling == {"temperature": 0.0, "top_k": 40}
    assert c.temperature == 0.0
    assert c.card == {}
    Carded = FakeClient.scripted([], card={"temperature": 1.0, "top_k": 64})
    assert Carded().card == {"temperature": 1.0, "top_k": 64}
    assert Carded().card is not Carded.card_says, "a copy, so a caller cannot edit the card"


def test_a_script_is_spent_in_order_and_its_last_reply_repeats():
    Scripted = FakeClient.scripted([("look_up", {"texts": ["compilers"]}), "a compiler person"])
    c = Scripted()
    first = c.chat([{"role": "user", "content": "who?"}], tools=[])
    assert isinstance(first, Reply)
    assert first.tool_calls[0]["function"]["name"] == "look_up"
    assert json.loads(first.tool_calls[0]["function"]["arguments"]) == {"texts": ["compilers"]}
    assert c.chat([], tools=[]).content == "a compiler person"
    assert c.chat([], tools=[]).content == "a compiler person"
    assert len(c.seen) == 3
    assert [call["method"] for call in c.calls] == ["chat"] * 3
    assert c.calls[0]["tools"] == []


def test_a_callable_script_is_asked_each_time_with_what_was_offered():
    def answer(messages, tools):
        offered = [t["function"]["name"] for t in tools or []]
        return f"offered {offered}"

    c = FakeClient.scripted(answer)()
    reply = c.chat([], tools=[{"type": "function", "function": {"name": "look_up"}}])
    assert reply.content == "offered ['look_up']"


def test_every_built_client_is_recorded_on_the_scripted_class_alone():
    One = FakeClient.scripted("one")
    Two = FakeClient.scripted("two")
    a, b = One("http://127.0.0.1:1"), One("http://127.0.0.1:2", temperature=0.5)
    assert One.built == [a, b]
    assert Two.built == []
    assert [x.base_url for x in One.built] == ["http://127.0.0.1:1", "http://127.0.0.1:2"]
    assert b.sampling == {"temperature": 0.5}


def test_extract_answers_from_the_script_and_reports_objections():
    c = FakeClient.scripted([{"name": "Ada"}, {"name": ""}])()
    schema = {"type": "object", "properties": {"name": {"type": "string"}}}
    assert c.extract("Ada wrote it", schema) == {"name": "Ada"}
    checked = c.extract("nobody wrote it", schema,
                        check=lambda a: ["name is empty"] if not a["name"] else [])
    assert checked == {"name": "", "_objections": ["name is empty"]}
    assert [call["method"] for call in c.calls] == ["extract", "extract"]
    with pytest.raises(ValueError, match="tries"):
        c.extract("", schema, tries=0)


def test_on_delta_receives_the_thinking_and_the_answer_as_whole_pieces():
    c = FakeClient.scripted(Reply(content="Ada.", thinking="weigh it"))()
    heard = []
    c.chat([], on_delta=lambda kind, text: heard.append((kind, text)))
    assert heard == [("thinking", "weigh it"), ("content", "Ada.")]


def test_told_is_what_the_tools_said():
    c = FakeClient()
    c.chat([{"role": "user", "content": "who?"},
            {"role": "tool", "content": "person:ada"},
            {"role": "tool", "content": "person:bea"}])
    assert c.told() == "person:ada person:bea"


# ---------------------------------------------------------------- ScriptedModel

def _offer(*names):
    return [{"type": "function", "function": {"name": n}} for n in names]


def test_the_scripted_model_calls_only_what_it_was_offered_then_answers():
    m = ScriptedModel([("look_up", {"text": "Ada"}), ("look_at", {"ids": ["person:ada"]})],
                      answer="Ada.")
    first = m.chat([], tools=_offer("look_up", "look_at"))
    assert first.tool_calls[0]["function"]["name"] == "look_up"
    # look_at is next but not on offer: words, and the call is not spent
    assert m.chat([], tools=_offer("look_up")).content == "Ada."
    assert m.script == [("look_at", {"ids": ["person:ada"]})]
    assert m.chat([], tools=_offer("look_at")).tool_calls[0]["function"]["name"] == "look_at"
    assert m.chat([], tools=_offer("look_at")).content == "Ada."
    assert len(m.seen) == 4


def test_the_scripted_model_takes_what_the_tool_loop_passes():
    """The loop passes ``think=``; a fake whose ``chat`` lacked ``**extra`` would refuse it."""
    m = ScriptedModel()
    assert m.chat([], think=False, tools=_offer("look_up")).content == ScriptedModel.ANSWER


# ---------------------------------------------------------------- serving

def test_fake_serve_yields_a_real_server_info_on_the_port_asked():
    with fake_serve("tiny.gguf", port=4242, context=8192) as info:
        assert isinstance(info, ServerInfo)
        assert (info.base_url, info.port) == ("http://127.0.0.1:4242", 4242)
    with fake_serve("tiny.gguf") as info:
        assert info.port == 1


def test_fake_serve_refuses_a_spec_keyword_the_real_one_would():
    with pytest.raises(TypeError, match="tight"):
        with fake_serve("tiny.gguf", tight=True):
            pass


def test_a_recording_fake_serve_keeps_every_lease_and_release():
    serving = FakeServe(load_s=12.5, warmup_s=1.2, pid=7)
    with serving("tiny.gguf", port=9, draft="head.gguf", timeout=30.0) as info:
        assert (info.load_s, info.warmup_s, info.pid) == (12.5, 1.2, 7)
        assert serving.released == []
    assert isinstance(serving.leased[0], ServerSpec)
    assert (serving.leased[0].port, serving.leased[0].draft) == (9, "head.gguf")
    assert serving.timeouts == [30.0]
    assert serving.released == [info]


def test_a_fake_serve_can_refuse_a_model_by_name():
    class Refused(RuntimeError):
        pass

    serving = FakeServe(refuse=("huge",), raising=Refused)
    with pytest.raises(Refused, match="huge-Q8"):
        with serving("huge-Q8.gguf"):
            pass
    assert len(serving.leased) == 1, "the lease was asked for before it was refused"
    with pytest.raises(ServerFailed):
        with FakeServe(refuse=("huge",))("huge.gguf"):
            pass


# ---------------------------------------------------------------- preflight

def test_a_fake_report_is_a_report():
    ok = FakeReport(weights_bytes=5 * 2**30, kv_estimate_bytes=2**30)
    assert isinstance(ok, Report)
    assert ok.ok and ok.weights_bytes == 5 * 2**30 and ok.kv_estimate_bytes == 2**30
    assert ok.said().startswith("ok    shards")
    bad = FakeReport(ok=False, model="huge.gguf")
    assert not bad.ok
    assert "FAIL  shards: missing or empty: huge.gguf" in bad.said()


def test_a_fake_preflight_records_what_it_was_asked_and_refuses_by_name():
    preflight = FakePreflight(refuse=("huge",), weights_bytes=1, kv_estimate_bytes=2)
    spec = ServerSpec(model="tiny.gguf", draft="head.gguf")
    report = preflight(spec, binary="/bin/llama-server", limit_bytes=2**30)
    assert report.ok and (report.weights_bytes, report.kv_estimate_bytes) == (1, 2)
    assert not preflight(ServerSpec(model="huge.gguf"), binary="x").ok
    assert [s.model for s in preflight.seen] == ["tiny.gguf", "huge.gguf"]
    assert preflight.seen[0].draft == "head.gguf"
