"""Every request raised, answered or withdrawn is an activity record that names the request, its kind,
who raised it and the way it was answered, and never the words it was about."""

from __future__ import annotations

from ml_stack import activity, requests
from tests.activity_support import entries, person, ring
from tests.requests_support import CANARY, ask

__all__ = ["person", "ring"]


def test_a_raise_an_answer_and_a_withdrawal_each_leave_one_record_without_the_subject(person):
    activity.mirror_requests()
    one = requests.raise_request(ask(CANARY, agent="alpha", project="proj"))
    requests.answer(one.id, "allow-once", one.fingerprint, "ui")
    two = requests.raise_request(ask("other", agent="alpha"))
    two.withdraw()
    rows = [e for e in entries() if e.kind.startswith("request.")]
    assert [(e.kind, e.subject, e.outcome) for e in rows] == [
        ("request.raised", one.id, "pending"), ("request.answered", one.id, "approved"),
        ("request.raised", two.id, "pending"), ("request.cancelled", two.id, "cancelled")]
    answered = rows[1]
    assert answered.actor == "person" and answered.meta["via"] == "ui" and answered.meta["choice"] == "allow-once"
    assert answered.refs["agent"] == "alpha" and answered.refs["project"] == "proj"
    assert CANARY not in repr(rows)
