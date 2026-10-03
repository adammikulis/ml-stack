"""Whether a request asks the model to think, decided per use from the person's setting."""

from __future__ import annotations

import os

__all__ = ["AGENT", "DECISION", "ENV", "MODES", "REASONING", "SHORT", "describe", "policy", "resolve", "setting"]

ENV = "ML_STACK_THINK"
MODES = ("off", "on", "auto")
DECISION, AGENT, SHORT, REASONING = "decision", "agent", "short", "reasoning"

# A prompt that contains one of these asks for reasoning under ``auto``.
_ASKS = ("think step by step", "think carefully", "reason through", "show your reasoning",
         "step by step")


def setting(asked: str = "") -> str:
    """``asked`` when it is a mode, else ``$ML_STACK_THINK`` when that is one, else ``auto``."""
    for one in (asked, os.environ.get(ENV, "")):
        if one in MODES:
            return one
    return "auto"


def resolve(use: str, *, asked: str = "", prompt: str = "") -> bool:
    """Whether a request of kind ``use`` thinks. Decisions never do. ``on`` and ``off`` apply
    to every other use; ``auto`` thinks only for the ``reasoning`` use or when ``prompt``
    contains a phrase asking for reasoning."""
    if use == DECISION:
        return False
    chosen = setting(asked)
    if chosen != "auto":
        return chosen == "on"
    return use == REASONING or any(ask in prompt.lower() for ask in _ASKS)


def describe(use: str, *, asked: str = "", prompt: str = "") -> str:
    """The line a header shows: ``thinking off (auto, agent)``."""
    on = resolve(use, asked=asked, prompt=prompt)
    return f"thinking {'on' if on else 'off'} ({'always off' if use == DECISION else setting(asked)}, {use})"


def policy(asked: str = "") -> str:
    """The policy in one line: the mode and what each use gets."""
    uses = ", ".join(f"{use} {'on' if resolve(use, asked=asked) else 'off'}"
                     for use in (DECISION, AGENT, SHORT))
    return f"thinking {setting(asked)}: {uses}; on when the person or the task asks"
