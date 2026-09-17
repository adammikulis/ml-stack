"""Ranking a set of named documents against a query, and the declaration they come from."""

from __future__ import annotations

import pytest

from ml_stack.client.embed import TASK, VectorMismatch
from ml_stack.client.select import Ranking, Selector, fingerprint
from ml_stack.client.tools import (
    NOTHING,
    ToolSpec,
    documents,
    function_schemas,
    in_category,
    runnable,
    spec_by_name,
)
from ml_stack.testing.embedding import bag_of_words_embedder

DOCS = {
    "look_up": ("who fixes machines?", "who can sell things?"),
    "look_at": ("tell me about Iris Bellweather",),
    "no_tool": ("hello", "tell me a joke"),
}


embedder = bag_of_words_embedder(
    *DOCS.values(), ["who fixes engines? good morning nothing like these at all"])


def a_selector(**kw):
    kw.setdefault("model", "pretend")
    kw.setdefault("abstain", "no_tool")
    return Selector(DOCS, **kw)


def ranked(query: str, **kw):
    return a_selector(**kw).rank(query, base_url="http://nowhere.invalid",
                                 embedder=embedder)


# --- ranking ---------------------------------------------------------------


def test_a_query_ranks_the_document_it_resembles_first():
    assert ranked("who fixes machines?").best == "look_up"
    assert ranked("tell me about Iris Bellweather").best == "look_at"


def test_a_tool_scores_as_its_best_text_not_its_average():
    """A name with one text that fits exactly and four that do not is still the right name."""
    docs = {"right": ("who fixes machines?", "aaa", "bbb", "ccc"),
            "wrong": ("who fixes engines?", "who fixes engines?")}
    out = Selector(docs, model="pretend").rank(
        "who fixes machines?", base_url="http://nowhere.invalid", embedder=embedder)
    assert out.best == "right"


def test_abstaining_is_a_document_winning_not_a_threshold():
    """A greeting reaches the abstain document, and `chose` then says to act on nothing."""
    out = ranked("hello")
    assert out.best == "no_tool"
    assert out.abstained
    assert out.chose() is None


def test_a_real_query_does_not_abstain():
    out = ranked("who fixes machines?")
    assert not out.abstained
    assert out.chose() == "look_up"


def test_among_masks_the_candidates_but_never_the_abstain_document():
    """Masking is how a caller hides a tool it cannot run; it may not hide abstention,
    or a turn that needs no tool has nowhere to land."""
    out = a_selector().rank("hello", base_url="http://nowhere.invalid",
                            embedder=embedder, among=["look_at"])
    assert set(out.order) == {"look_at", "no_tool"}
    assert out.abstained


def test_top_leaves_the_abstain_document_out_of_a_shortlist():
    """A shortlist is tools to offer a model; abstaining is not one of them."""
    out = ranked("hello")
    assert "no_tool" not in out.top(3)


def test_an_empty_query_ranks_nothing():
    assert ranked("").order == []
    assert ranked("   ").chose() is None


def test_the_margin_gate_is_off_rather_than_always_sure_at_zero():
    """Turning the gate off must not read as confidence: nothing was tested."""
    assert not ranked("who fixes machines?", margin=0.0).clear
    assert not ranked("who fixes machines?", margin=10.0).clear
    assert ranked("who fixes machines?", margin=0.01).clear


def test_a_flat_ranking_is_not_clear():
    """A query resembling nothing in particular leaves a flat field."""
    assert not ranked("nothing like these at all", margin=0.2).clear


def test_chose_can_require_the_ranking_to_stand_out():
    out = ranked("who fixes machines?", margin=10.0)
    assert out.chose() == "look_up"
    assert out.chose(require_clear=True) is None


# --- caching ---------------------------------------------------------------


def test_the_documents_are_embedded_once_and_the_query_every_time():
    """A steady process pays for the catalogue once; a turn pays for its own message."""
    batches: list[list[str]] = []

    def counting(texts, **kw):
        batches.append(list(texts))
        return embedder(texts, **kw)

    picked = Selector(DOCS, model=f"count-{id(batches)}", abstain="no_tool")
    for _ in range(3):
        picked.rank("who fixes machines?", base_url="http://nowhere.invalid",
                    embedder=counting)
    assert len(batches[0]) == sum(len(t) for t in DOCS.values())
    assert [len(b) for b in batches[1:]] == [1, 1, 1]


def test_editing_a_document_changes_the_key_and_a_steady_one_does_not():
    """The key is the texts, so a description edit invalidates it and a rerun does not."""
    one = Selector(DOCS, model="pretend")
    same = Selector(dict(DOCS), model="pretend")
    edited = Selector({**DOCS, "look_at": ("tell me about someone else",)}, model="pretend")
    assert one.key() == same.key()
    assert one.key() != edited.key()


def test_the_key_separates_models_prefixes_and_servers():
    assert fingerprint(DOCS, model="a", query_prefix=TASK, document_prefix=TASK) \
        != fingerprint(DOCS, model="b", query_prefix=TASK, document_prefix=TASK)
    assert fingerprint(DOCS, model="a", query_prefix=TASK, document_prefix=TASK) \
        != fingerprint(DOCS, model="a", query_prefix="other: ", document_prefix=TASK)
    assert fingerprint(DOCS, model="a", query_prefix=TASK, document_prefix=TASK) \
        != fingerprint(DOCS, model="a", query_prefix=TASK, document_prefix=TASK,
                       base_url="http://elsewhere")


def test_the_prefixes_reach_the_embedder():
    seen: list[str] = []

    def watching(texts, **kw):
        seen.extend(texts)
        return embedder(texts, **kw)

    Selector(DOCS, model=f"prefix-{id(seen)}", prefixes=("Q: ", "D: ")).rank(
        "who fixes machines?", base_url="http://nowhere.invalid", embedder=watching)
    assert any(t.startswith("D: ") for t in seen)
    assert any(t.startswith("Q: ") for t in seen)


# --- failure ---------------------------------------------------------------


def test_incomparable_vectors_are_raised_not_ranked():
    """An embedder handing back the wrong width is a bug, and a bug that returns a flat
    ranking instead of raising is indistinguishable from a question about nothing."""
    def wrong_width(texts, **kw):
        return [[1.0] * (3 + len(t) % 2) for t in texts]

    with pytest.raises(VectorMismatch):
        Selector(DOCS, model=f"bad-{id(wrong_width)}").rank(
            "who fixes machines?", base_url="http://nowhere.invalid", embedder=wrong_width)


# --- declarations ----------------------------------------------------------


SPECS = (
    ToolSpec(name="look_up", description="Find a thing by what it does.",
             schema={"type": "object", "properties": {"q": {"type": "string"}},
                     "required": ["q"]},
             examples=("who fixes machines?",), category="search", needs=("graph",)),
    ToolSpec(name="show", description="Light nodes up on the graph.", category="display"),
)


def test_a_declaration_carries_the_schema_a_model_is_offered():
    (offered,) = function_schemas(SPECS[:1])
    assert offered["function"]["name"] == "look_up"
    assert offered["function"]["description"] == SPECS[0].description
    assert offered["function"]["parameters"]["required"] == ["q"]


def test_a_declaration_carries_the_documents_a_selector_embeds():
    assert documents(SPECS)["look_up"] == ("who fixes machines?",)


def test_a_tool_with_no_examples_falls_back_to_its_description():
    """Worse than examples and never silent: the description is what is left to match on."""
    assert documents(SPECS)["show"] == ("show: Light nodes up on the graph.",)


def test_a_tool_that_takes_nothing_still_declares_an_object_schema():
    assert SPECS[1].schema == NOTHING
    assert SPECS[1].required() == ()
    assert SPECS[0].required() == ("q",)


def test_two_tools_may_not_share_a_name():
    with pytest.raises(ValueError, match="look_up"):
        spec_by_name([SPECS[0], SPECS[0]])


def test_category_groups_and_needs_gates_and_they_are_different_questions():
    """A category is what a selector may choose between; needs is what a tool cannot run
    without. One word for both is how "which tools?" became two questions."""
    assert [s.name for s in in_category(SPECS, "search")] == ["look_up"]
    assert [s.name for s in runnable(SPECS, ["graph"])] == ["look_up", "show"]
    assert [s.name for s in runnable(SPECS, [])] == ["show"]


def test_a_ranking_over_declarations_is_the_same_ranking():
    """The selector takes declarations, not a second hand-written table beside them."""
    specs = [ToolSpec(name=n, description=n, examples=tuple(t)) for n, t in DOCS.items()]
    out = Selector(documents(specs), model="pretend", abstain="no_tool").rank(
        "who fixes machines?", base_url="http://nowhere.invalid", embedder=embedder)
    assert out.best == "look_up"


def test_a_ranking_with_no_documents_at_all():
    assert Ranking([], {}, False).best is None
    assert Ranking([], {}, False).chose() is None
