"""The task prefixes and the margin gate the embedder is asked for."""

from __future__ import annotations

from ml_stack.client.embed import DOCUMENT, QUERY, TASK


def test_the_question_and_document_prefixes_differ():
    """A question and a paragraph are not the same kind of text, and are not told they are."""
    assert QUERY != DOCUMENT
    assert TASK not in (QUERY, DOCUMENT)


def test_stands_out_separates_a_question_from_a_greeting():
    """The gate reads the shape of the results, not how high the best score is.

    These are the numbers that were measured: a greeting scores higher than a real question
    and is still the flatter field, which is why a threshold on the score cannot work.
    """
    from ml_stack.client.embed import stands_out

    greeting = [0.754, 0.751, 0.744, 0.739, 0.731, 0.728]     # "hi"
    question = [0.740, 0.681, 0.652, 0.640, 0.633, 0.629]     # "someone who can sell things"

    assert greeting[0] > question[0]                          # the score says the wrong thing
    assert not stands_out(greeting)
    assert stands_out(question)


def test_stands_out_on_nothing_and_with_the_gate_off():
    from ml_stack.client.embed import stands_out

    assert not stands_out([])
    assert stands_out([], margin=0)                # off means everything passes, even nothing
    assert stands_out([0.7, 0.7, 0.7], margin=-1)
    assert not stands_out([0.9])                   # one result is its own mean: no margin
