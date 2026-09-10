"""How a question is asked: one record, and what each caller of `converse` takes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["ASKING", "Asking"]


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
    cite: bool = False                   # where every entry was read, and the `quote` tool
    reach: int | None = None             # tokens one tool result may carry
    rounds: int | None = None            # tool-calling turns one question may spend
    constrain_ids: bool = False          # id arguments held to the graph's ids by grammar

    @classmethod
    def for_model(cls, model: str, *, workload: str = "") -> Asking:
        """The asking a named model scored best with at ``workload`` -- the graph asking
        when none is named -- or the default when nothing measured it."""
        from ml_stack.serve.profile import profile_for

        found = profile_for(str(model), workload=workload)
        return found.asked() if found is not None else cls()

    def tools(self) -> dict[str, Any]:
        """The keyword arguments :func:`~ml_stack.graph.ask.tools_for` takes about the
        asking -- the terse set is chosen outside `converse` and handed in."""
        return {"asking": self, "cite": bool(self.cite)}

    def said(self) -> dict[str, Any]:
        """The asking itself, in the words the bench keeps beside its rows: ``summary``
        under its own name, ``terse`` said outright, and nothing that was not asked for."""
        out: dict[str, Any] = {"tight": bool(self.tight), "terse": bool(self.terse)}
        for flag in ("rich", "batch", "kinds", "summary", "single", "few", "cite",
                     "constrain_ids"):
            if getattr(self, flag):
                out[flag] = True
        if self.reach:
            out["reach"] = int(self.reach)
        if self.rounds:
            out["rounds"] = int(self.rounds)
        return out


#: the way a question is asked when nobody says
ASKING = Asking()
