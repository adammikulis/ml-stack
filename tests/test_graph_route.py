"""Which tool a question wants, decided by an embedder.

The tests use a stand-in embedder, so they run with no server. The real thing was checked
against embeddinggemma-300M and the numbers are in the comments where they matter.
"""

from __future__ import annotations

import pytest

from ml_stack.graph.ask import CHAT, TOOL_PROMPTS, TOOLS, prompts_for, tools_for
from ml_stack.graph.route import Routed, narrow, rank

GRAPH = {"nodes": [], "edges": [], "messages": {}}


def words(text: str) -> set[str]:
    return {w.strip(".,?!") for w in text.casefold().split()}


def fake_embedder(texts, **kw):
    """A bag of words as a vector: crude, but it prefers the same sentence to a different
    one, which is all these tests are about."""
    vocab = sorted({w for t in texts for w in words(t)})
    return [[1.0 if w in words(t) else 0.0 for w in vocab] for t in texts]


def routed(question: str, **kw):
    return rank(question, TOOL_PROMPTS, base_url="http://nowhere.invalid",
                model="pretend", embedder=fake_embedder, **kw)


def test_the_examples_are_questions_and_never_reach_the_model():
    """Two consumers, two texts. A chat model wants prose about what a tool does; an
    embedder wants the questions that should route there, because a question against
    questions is like-to-like and that is where the signal is."""
    assert {t["function"]["name"] for t in TOOLS} <= set(TOOL_PROMPTS)
    assert CHAT in TOOL_PROMPTS, "there must be somewhere for a greeting to go"

    said = " ".join(t["function"]["description"] for t in TOOLS)
    for name, examples in TOOL_PROMPTS.items():
        assert examples, f"{name} has no examples to match against"
        for example in examples:
            # whole examples, not fragments: "hi" is inside "which", and a two-word
            # greeting appearing by coincidence is not a leak
            if len(example.split()) < 3:
                continue
            assert example not in said, \
                "an example question must not be in the description the model reads"
    assert prompts_for("look_up") and prompts_for("nothing-called-this") == ()


def test_a_question_routes_to_the_tool_whose_examples_it_resembles():
    assert routed("who fixes machines?").order[0] == "look_up"
    assert routed("tell me about Iris Bellweather").order[0] == "look_at"
    assert routed("how are these two connected?").order[0] == "path_between"
    assert routed("highlight those on the graph").order[0] == "show"


def test_a_greeting_wants_no_graph_at_all():
    """Without somewhere for these to go, a greeting is matched against four search tools
    and wins one of them: measured against embeddinggemma, "hi" scored 0.900 against
    "highlight them on the graph"."""
    for small in ("hi", "tell me a joke", "what is the capital of France?"):
        assert routed(small).chat, f"{small!r} should want no graph"
        assert narrow(tools_for(GRAPH), routed(small)) == [], \
            "a message wanting no graph is offered no tools -- that is the whole saving"


def test_a_graph_question_is_never_mistaken_for_small_talk():
    """The worst failure available: answered without looking anything up, which reads as a
    confident answer and is about nothing."""
    for real in ("who fixes machines?", "tell me about Iris Bellweather",
                 "how are these two connected?", "who could introduce me to a lawyer?"):
        assert not routed(real).chat
        assert narrow(tools_for(GRAPH), routed(real)), f"{real!r} was left with no tools"


def test_nothing_is_narrowed_unless_the_router_was_sure():
    """A tool hidden from a model that needed it produces a wrong answer with no visible
    cause, which is worse than a longer prompt."""
    unsure = Routed(order=["look_up", "look_at"], scores={"look_up": .5, "look_at": .49},
                    clear=False)
    assert len(narrow(tools_for(GRAPH), unsure)) == len(TOOLS)
    assert len(narrow(tools_for(GRAPH), unsure, keep=1)) == len(TOOLS)


def test_show_survives_every_narrowing():
    """A turn that cannot say what its answer is about lights nothing, whatever else it
    got right."""
    sure = Routed(order=["path_between", "look_at"], scores={"path_between": .9}, clear=True)
    kept = {t[0]["function"]["name"] for t in narrow(tools_for(GRAPH), sure, keep=1)}
    assert kept == {"path_between", "show"}
    assert CHAT not in kept, "chat is not a tool and can never be offered as one"


def test_an_embedder_that_will_not_answer_routes_nothing():
    """A router that cannot embed must not narrow: no answer is not the same as 'chat'."""
    def refuses(texts, **kw):
        raise OSError("no embedding server")

    out = rank("who fixes machines?", TOOL_PROMPTS, base_url="http://nowhere.invalid",
               model="pretend", embedder=refuses)
    assert out.order == [] and not out.clear and not out.chat
    assert len(narrow(tools_for(GRAPH), out)) == len(TOOLS)


def test_an_empty_question_routes_nothing():
    assert not routed("").order and not routed("   ").chat


def test_a_tool_scores_as_its_best_example_not_its_average():
    """A tool with one example that fits exactly and four that do not is the right tool;
    averaging buries it under one whose examples are all vaguely close."""
    prompts = {"right": ("who fixes machines?", "aaa", "bbb", "ccc", "ddd"),
               "wrong": ("who fixes bicycles?", "who fixes boats?", "who fixes cars?")}
    out = rank("who fixes machines?", prompts, base_url="http://nowhere.invalid",
               model="pretend", embedder=fake_embedder)
    assert out.order[0] == "right"


@pytest.mark.parametrize("margin,expect", [(0.0, False), (10.0, False)])
def test_the_margin_gates_acting_on_a_routing(margin, expect):
    """Zero turns the gate off, which must not mean 'always sure'; a huge margin means
    never sure. Neither may quietly narrow."""
    out = routed("who fixes machines?", margin=margin)
    assert out.clear is expect


def test_a_question_about_what_kinds_of_thing_are_here_routes_to_listing_them():
    """Nothing in a graph is labelled "company", so "which companies are here?" is not a
    search however it is worded. It wants every entry of one kind, and that is a different
    tool -- kept apart from look_up's examples, which ask for a particular thing."""
    assert routed("which companies are here?").order[0] == "list_kind"
    assert routed("what kinds of places are there?").order[0] == "list_kind"
    assert routed("which employers are represented here?").order[0] == "list_kind"
    # asking for one particular thing is still a search
    assert routed("who fixes machines?").order[0] == "look_up"
    assert not any(example in TOOL_PROMPTS["look_up"] for example in TOOL_PROMPTS["list_kind"])
