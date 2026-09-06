"""The ways a question is asked: one record, and what each caller of `converse` takes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["Asking"]


@dataclass(frozen=True)
class Asking:
    """One model, asked one way: the argument :func:`ml_stack.graph.ask.converse` takes.

    Every field changes what the model is asked, and none of them is anything a client or
    a server has ever heard of.
    """

    tight: bool = True
    batch: bool = False
    single: bool = False                 # one entry to a read, more turns -- batch's opposite
    few: bool = False                    # three tools offered, not eight
    kinds: bool = False
    summary: bool = False                # the `summarise` tool, named as the bench is
    rich: bool = False
    terse: bool = False                  # `tools_for`'s schemas, not `converse`'s
    reach: int | None = None             # tokens one tool result may carry
    rounds: int | None = None            # tool-calling turns one question may spend
    constrain_ids: bool = False          # id arguments held to the graph's ids by grammar

    @classmethod
    def for_model(cls, model: str, *, workload: str = "") -> Asking:
        """The way a named model measured best at ``workload`` -- the graph asking when
        none is named -- or the default way when nothing measured it."""
        from ml_stack.serve.profile import profile_for

        found = profile_for(str(model), workload=workload)
        return found.asked() if found is not None else cls()

    def tools(self) -> dict[str, Any]:
        """The keyword arguments :func:`~ml_stack.graph.ask.tools_for` takes about the
        asking -- the terse set is chosen outside `converse` and handed in."""
        return {"tight": bool(self.tight), "reach": self.reach,
                "batch": bool(self.batch), "single": bool(self.single),
                "few": bool(self.few), "summary": bool(self.summary)}

    def said(self) -> dict[str, Any]:
        """The way itself, in the words the bench keeps beside its rows: ``summary`` under
        its own name, ``terse`` said outright, and nothing that was not asked for."""
        out: dict[str, Any] = {"tight": bool(self.tight), "terse": bool(self.terse)}
        for way in ("rich", "batch", "kinds", "summary", "single", "few", "constrain_ids"):
            if getattr(self, way):
                out[way] = True
        if self.reach:
            out["reach"] = int(self.reach)
        if self.rounds:
            out["rounds"] = int(self.rounds)
        return out
