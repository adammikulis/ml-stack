"""The agent-guard benchmark: hand-written tool calls labelled for three guard questions.

``guard_cases()`` builds the cases in a fixed order; `poolhouse.decide.guards.states` holds the
question and option text each one is asked with.
"""

from __future__ import annotations

from poolhouse.decide.cases import Case
from poolhouse.decide.guards.destructive import DESTRUCTIVE, REVERSIBLE, SAFE
from poolhouse.decide.guards.grounding import GROUNDED, INJECTED
from poolhouse.decide.guards.scope import INSIDE, OUTSIDE
from poolhouse.decide.guards.states import (
    QUESTIONS,
    call_text,
    destructive_state,
    grounded_state,
    scope_state,
)


def guard_cases() -> list[Case]:
    """Every case: destructive, then grounded, then scope; ids are ``<question>-<number>``."""
    cases: list[Case] = []
    for label, pool in (("safe", SAFE), ("reversible", REVERSIBLE), ("destructive", DESTRUCTIVE)):
        question, options = QUESTIONS["destructive"]
        for tool, args, requests in pool:
            for request in requests:
                cases.append(Case(question, destructive_state(request, tool, args), options,
                                  label, id=f"destructive-{len(cases) + 1:03d}",
                                  group=call_text(tool, args), tags=("destructive", label)))
    for label, pool in (("grounded", GROUNDED), ("injected", INJECTED)):
        question, options = QUESTIONS["grounded"]
        for request, output, tool, args in pool:
            cases.append(Case(question, grounded_state(request, output, tool, args), options,
                              label, id=f"grounded-{len(cases) + 1:03d}",
                              group=request, tags=("grounded", label)))
    for label, pool in (("inside", INSIDE), ("outside", OUTSIDE)):
        question, options = QUESTIONS["scope"]
        for request, tool, args in pool:
            cases.append(Case(question, scope_state(request, tool, args), options,
                              label, id=f"scope-{len(cases) + 1:03d}",
                              group=call_text(tool, args), tags=("scope", label)))
    return cases
