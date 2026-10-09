"""`DestructiveRail`: asks the person before a call the classifier labels destructive or unsure.

It runs beside the role's own rail and can only add an approval: its answer is a Confirm or
Proceed, and the most severe verdict of all the rails wins, so a role, a plan or a saved rule that
lets a call run does not stop this rail from asking.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Collection, Mapping
from pathlib import Path
from typing import Any

from ml_stack import home
from ml_stack.guard.destructive import Verdict, classify, combine, reason_text
from ml_stack.guard.destructive_model import ModelLayer
from ml_stack.interventions import Base, Call, Confirm, Context, Proceed

__all__ = ["DestructiveRail", "annotations_of"]

logger = logging.getLogger("ml_stack.guard")
FAILURES = (ValueError, TypeError, KeyError, AttributeError, IndexError, RuntimeError, OSError,
            RecursionError, UnicodeError)


def annotations_of(tools: Collection[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    """The MCP annotations of each offered tool, by name."""
    found: dict[str, Mapping[str, Any]] = {}
    for schema in tools:
        fn = schema.get("function") if isinstance(schema.get("function"), Mapping) else schema
        notes = fn.get("annotations") or schema.get("annotations")
        if fn.get("name") and isinstance(notes, Mapping):
            found[str(fn["name"])] = notes
    return found


class DestructiveRail(Base):
    """Classifies each call that is not in ``skip`` and asks when the label is ``destructive``
    or ``unsure``. ``catalog`` maps tool names to a label, ``floors`` to the least severe label a tool may get, ``roots`` are where writes may go,
    ``model`` is an optional second opinion that can only raise the label, and ``on_verdict``
    (set after construction) receives each verdict that asks. ``last`` holds the latest verdict."""

    name = "destructive"

    def __init__(self, *, roots: Callable[[], tuple[str, ...]] | tuple[str, ...] = (),
                 catalog: Callable[[], Mapping[str, str]] | Mapping[str, str] | None = None,
                 floors: Mapping[str, str] | None = None,
                 skip: Callable[[], Collection[str]] | Collection[str] = (),
                 model: ModelLayer | None = None) -> None:
        self._roots, self._catalog, self._skip = roots, catalog, skip
        self.floors = dict(floors or {})
        self.model = model
        self.on_verdict: Callable[[Call, Verdict], Any] | None = None
        self.last: Verdict | None = None

    def judge(self, call: Call, context: Context | None = None) -> Verdict:
        """The verdict for ``call``: the deterministic one, raised by the model when there is one
        and the call did not already ask. Any failure is ``unsure``."""
        roots = self._roots() if callable(self._roots) else self._roots
        catalog = (self._catalog() if callable(self._catalog) else self._catalog) or {}
        try:
            base = classify(call, roots=roots, catalog=catalog, floors=self.floors,
                            annotations=annotations_of(context.tools) if context else None)
            if self.model is not None and not base.asks:
                base = combine(base, self.model.classify(call))
        except FAILURES as exc:
            logger.warning("destructive classifier failed: %s", type(exc).__name__)
            base = Verdict("unsure", [f"the classifier failed ({type(exc).__name__})"],
                           confidence=0.0)
        return base

    def before_tool_call(self, call: Call, context: Context) -> Confirm | Proceed:
        skip = self._skip() if callable(self._skip) else self._skip
        if call.name in skip:
            return Proceed()
        verdict = self.last = self.judge(call, context)
        if not verdict.asks:
            return Proceed()
        if self.on_verdict is not None:
            self.on_verdict(call, verdict)
        details: dict[str, Any] = {"classifier": verdict.public()}
        if verdict.layer == "deterministic":
            details |= {"always_ok": False,
                        "always_blocked": f"the classifier labelled it {verdict.label}"}
        return Confirm(f"Asked because: {verdict.label}: {reason_text(verdict)}.", details, self.name)


def default_roots() -> tuple[str, ...]:
    """The current directory, where relative paths are read, and ml-stack's state."""
    return (str(Path.cwd()), str(home.home()))
