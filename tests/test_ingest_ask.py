"""Asking a shelf a question, and scoring a set of them.

The model is `ml_stack.testing.ScriptedModel` -- the tool loop's own fake, which answers
with the calls it was told to and then with words. No port is opened and no model is served.
Every concept and question here is invented.
"""

from __future__ import annotations

import json

import pytest
from test_ingest import a_part_read_book, a_read, said

from ml_stack import ingest
from ml_stack.testing import ScriptedModel

pytest.importorskip("ladybug")

OPEN_TEXTS = "velthorne-open-texts"


def a_store(tmp_path):
    """One book folded into a store, and a hidden run node beside it."""
    rows = [a_read(f"{OPEN_TEXTS}:1:1.1",
                   extracted=said("vault", "vault current",
                                  relations=[("vault current", "part_of", "vault")])),
            a_read(f"{OPEN_TEXTS}:1:1.2", section="1.2", pages=(4, 5),
                   extracted=said("seam wall", "vault",
                                  relations=[("seam wall", "part_of", "vault")]))]
    store = a_part_read_book(tmp_path, rows=rows)
    ingest.fold(store, say=lambda _: None)
    ingest.write_run(store, {"id": "run:20260903T090000", "model": "a-model",
                             "started": "2026-09-03T09:00:00"})
    return store


def a_model(*calls, answer="The vault current runs inside the vault."):
    return ScriptedModel(list(calls), answer=answer)


def found_then_shown(*ids, answer="The vault current runs inside the vault."):
    """A model that looks a thing up, reads it, and says the answer is about ``ids``."""
    return a_model(("look_up", {"texts": ["vault"]}), ("look_at", {"ids": list(ids)}),
                   ("show", {"ids": list(ids)}), answer=answer)


class _one_each:
    """One scripted model per question, handed out in the order the questions are asked.

    A `ScriptedModel` spends its script, so a set of questions asked of one would answer
    everything after the first from nothing. A turn carrying no answer and no tool result
    is the first of a new question.
    """

    def __init__(self, models):
        self.models = list(models)
        self.given = None

    def chat(self, messages, **kw):
        fresh = not any(m.get("role") in ("assistant", "tool") for m in messages)
        if fresh and self.models:
            self.given = self.models.pop(0)
        return self.given.chat(messages, **kw)


# -- the store as a graph a question can be asked of -----------------------------------------


def test_the_store_reads_out_as_nodes_and_edges_with_the_run_left_out(tmp_path):
    store = a_store(tmp_path)

    graph = ingest.graph_of(store)

    ids = {n["id"] for n in graph["nodes"]}
    assert {"book:velthorne-open-texts", "concept:vault", "concept:vault-current"} <= ids
    assert not [i for i in ids if i.startswith("run:")], "the run is the record, not the graph"
    assert all(e["source"] in ids and e["target"] in ids for e in graph["edges"])


def test_a_concept_is_read_out_with_the_definition_the_book_gave_it(tmp_path):
    """`look_at` is where a model gets the facts it answers from, and a concept's facts are
    its definition."""
    from ml_stack.graph.ask import look_at

    graph = ingest.graph_of(a_store(tmp_path))

    read = look_at(graph, ["concept:vault"])

    assert "vault" in read
    assert 'defined: "What a vault is."' in read


# -- one question ----------------------------------------------------------------------------


def test_a_question_is_answered_with_the_tools_it_called_and_what_it_spent(tmp_path, capsys):
    graph = ingest.graph_of(a_store(tmp_path))
    model = found_then_shown("concept:vault-current")

    answer = ingest.ask(graph, "what is a vault current?", model)

    said_out = capsys.readouterr().out
    assert "The vault current runs inside the vault." in said_out
    assert "tools: " in said_out and "looked up 'vault'" in said_out
    assert "about: concept:vault-current" in said_out
    assert "Spent: " in said_out and "call(s)" in said_out
    assert answer.show == ["concept:vault-current"]


def test_the_model_is_told_what_the_book_said_rather_than_being_asked_from_nothing(tmp_path):
    graph = ingest.graph_of(a_store(tmp_path))
    model = found_then_shown("concept:vault-current")

    ingest.ask(graph, "what is a vault current?", model, say=lambda _: None)

    assert "What a vault current is." in model.told()


def test_an_empty_store_is_not_asked_about(tmp_path, capsys):
    from ml_stack.graph.store import GraphStore

    store = tmp_path / "empty.ladybug"
    with GraphStore(store) as held:
        held.write({"nodes": [], "edges": []})

    assert ingest.main(["ask", "--out", str(store), "what is a vault?"]) == 2
    assert "nothing in" in capsys.readouterr().err


# -- a set of questions, scored ----------------------------------------------------------------


def a_gold_file(tmp_path, questions):
    path = tmp_path / "asked.json"
    path.write_text(json.dumps(questions), encoding="utf-8")
    return path


def test_a_gold_set_of_questions_is_read_back_and_an_empty_one_is_refused(tmp_path):
    path = a_gold_file(tmp_path, [{"question": "what is a vault?", "expected": ["vault"]}])

    assert ingest.read_asked(path) == [{"question": "what is a vault?", "expected": ["vault"]}]
    assert ingest.read_asked(a_gold_file(tmp_path, {"questions": [{"question": "?"}]}))

    with pytest.raises(ValueError):
        ingest.read_asked(a_gold_file(tmp_path, []))


def test_what_a_question_expects_may_be_a_label_and_is_resolved_to_the_id(tmp_path):
    graph = ingest.graph_of(a_store(tmp_path))

    assert ingest._ids_for(graph, ["vault current", "concept:vault", "Seam Wall"]) == [
        "concept:vault-current", "concept:vault", "concept:seam-wall"]
    assert ingest._ids_for(graph, ["nothing here"]) == ["nothing here"]


def test_a_perfect_answer_scores_one_and_a_wrong_one_is_named(tmp_path, capsys):
    graph = ingest.graph_of(a_store(tmp_path))
    asked = [{"label": "right", "question": "what runs in the vault?",
              "expected": ["vault current"]},
             {"label": "wrong", "question": "what holds the vault up?",
              "expected": ["seam wall"]}]
    models = [found_then_shown("concept:vault-current"), found_then_shown("concept:vault")]
    rows = ingest.score_asked(graph, _one_each(models), asked, log=print)

    assert [r.label for r in rows] == ["right", "wrong"]
    assert rows[0].recall == 1.0 and rows[0].precision == 1.0 and rows[0].hit == 1.0
    assert rows[1].hit == 0.0
    lines = ingest.asked_lines(rows)
    assert "2 question(s) -- recall 50%, precision 50%, F1 50%" in lines[0]
    assert any("wrong: F1 0%; missed concept:seam-wall; also showed concept:vault" in line
               for line in lines)
    assert "recall 100%" in capsys.readouterr().out


def test_the_score_is_the_benchs_own_and_not_a_second_one(tmp_path):
    """A number measured two ways is two numbers: `graph.bench.score.Row` scores both."""
    from ml_stack.graph.bench.score import Row

    graph = ingest.graph_of(a_store(tmp_path))
    asked = [{"question": "what is in the vault?",
              "expected": ["vault current", "seam wall"]}]
    rows = ingest.score_asked(graph, found_then_shown("concept:vault-current"), asked)

    assert isinstance(rows[0], Row)
    assert rows[0].recall == 0.5 and rows[0].precision == 1.0


def test_a_question_expecting_nothing_is_not_averaged_in(tmp_path):
    graph = ingest.graph_of(a_store(tmp_path))
    asked = [{"question": "hello", "expected": []},
             {"question": "what runs in the vault?", "expected": ["vault current"]}]
    rows = ingest.score_asked(graph, _one_each([found_then_shown("concept:vault"),
                                                found_then_shown("concept:vault-current")]),
                              asked)

    assert ingest.asked_f1(rows) == 1.0
    assert "1 question(s) -- recall 100%" in ingest.asked_lines(rows)[0]




# -- through the command -------------------------------------------------------------------


def serving(monkeypatch, model):
    """`_serving` yielding a scripted model rather than leasing one."""
    import contextlib

    @contextlib.contextmanager
    def held(args, say=print):
        yield model

    monkeypatch.setattr(ingest, "_serving", held)


def test_ask_through_the_command_prints_the_answer_and_what_it_cost(tmp_path, monkeypatch,
                                                                   capsys):
    store = a_store(tmp_path)
    serving(monkeypatch, found_then_shown("concept:vault-current"))

    assert ingest.main(["ask", "--out", str(store), "what is a vault current?"]) == 0

    said_out = capsys.readouterr().out
    assert "node(s)" in said_out
    assert "The vault current runs inside the vault." in said_out
    assert "Spent:" in said_out


def test_ask_through_the_command_scores_a_gold_set(tmp_path, monkeypatch, capsys):
    store = a_store(tmp_path)
    path = a_gold_file(tmp_path, [{"question": "what runs in the vault?",
                                   "expected": ["vault current"]}])
    serving(monkeypatch, found_then_shown("concept:vault-current"))

    assert ingest.main(["ask", "--out", str(store), "--gold", str(path)]) == 0

    said_out = capsys.readouterr().out
    assert "asking 1 question(s)" in said_out
    assert "1 question(s) -- recall 100%, precision 100%, F1 100%" in said_out


def test_fail_under_gates_the_questions(tmp_path, monkeypatch, capsys):
    store = a_store(tmp_path)
    path = a_gold_file(tmp_path, [{"question": "what holds the vault up?",
                                   "expected": ["seam wall"]}])
    serving(monkeypatch, found_then_shown("concept:vault"))

    assert ingest.main(["ask", "--out", str(store), "--gold", str(path),
                        "--fail-under", "0.5"]) == 1
    assert "is under 0.50" in capsys.readouterr().err


def test_ask_with_neither_a_question_nor_a_gold_set_is_an_error(tmp_path, capsys):
    store = a_store(tmp_path)

    assert ingest.main(["ask", "--out", str(store)]) == 2
    assert "ask needs a question" in capsys.readouterr().err


def test_ask_on_a_store_that_is_not_there_says_so(tmp_path, capsys):
    assert ingest.main(["ask", "--out", str(tmp_path / "nothing.ladybug"), "what?"]) == 2
    assert "no store at" in capsys.readouterr().err


def test_ask_needs_a_store(capsys):
    assert ingest.main(["ask"]) == 2
    assert "ask needs --out STORE" in capsys.readouterr().err
