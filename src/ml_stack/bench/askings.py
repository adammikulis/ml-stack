"""How one served model is asked, read off the command line.

`_askings` is the ways one load is measured -- what was asked for, plus each ``--also``;
`halves` is the plain and shortlisted halves a sweep asks of each model and `_asked`
crosses the two; `asking_from` and `sampling_from` read the asking and the sampler
overrides off the parsed line, and `with_card` puts a model's own card over a client.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Any

# The package is the namespace the tests and `selfcheck` patch -- `bench._askings` -- so
# anything patchable is looked up there at call time, never bound here at import.
from ml_stack import bench
from ml_stack.asking import Asking
from ml_stack.log import warn

__all__ = ["REACH", "_asked", "_askings", "asking_from", "halves", "sampling_from",
           "with_card"]


# What `--also reach` gives one tool result, in tokens, when `--reach` did not say. See
# `_askings`: a neighbourhood read whole, which a 256k window does not notice.
REACH = 8000


def _askings(args: Any) -> list[dict[str, Any]]:
    """The askings to make of one served model: what was asked for, plus each --also.

    Separating these from the serving is where the time goes. A model load is minutes; an
    asking is minutes too, and repeating the load for a question about the *asking* pays it
    twice for nothing.
    """
    asked = asking_from(args)
    first: dict[str, Any] = {"terse": asked.terse, **sampling_from(args)}
    out = [first]
    for also in getattr(args, "also", []) or []:
        if also == "terse":
            out.append({"label": "terse", "terse": True, **sampling_from(args)})
        elif also == "greedy":
            out.append({"label": "greedy", "terse": first["terse"], "temperature": 0.0})
        elif also == "card":
            # the card's own settings are read from the served model at ask time
            out.append({"label": "card", "terse": first["terse"], "_card": True})
        elif also == "rich":
            # look_up results carry a score and why they matched, and a topic hit brings
            # the people joined to it -- a question about the asking, so one load
            out.append({"label": "rich", "terse": first["terse"], "rich": True,
                        **sampling_from(args)})
        elif also == "reach":
            # What Flash-Next is for. Measured 2026-09-02: 256k of context at 48K bytes a
            # token, a tool result read back at ~390 tok/s against ~35 tok/s written, and
            # 5-9 calls a question -- so half the wall clock was reading and the way to
            # spend less of it is fewer, fatter calls. `look_around` is the fat call and
            # `reach` is what lets a result be worth making; 8000 tokens is a page of
            # neighbourhood, which is nothing to a 256k window and too much for E2B's.
            out.append({"label": "reach", "terse": first["terse"],
                        "reach": asked.reach or REACH, **sampling_from(args)})
        elif also == "loose":
            # the control: show told to name what the answer is about with no cap and no
            # closing rule -- what every run before 2026-09-02 measured. Tight is the
            # default asking now; Flash-Next went from 43% to 83% precision on it.
            out.append({"label": "loose", "terse": first["terse"], "tight": False,
                        **sampling_from(args)})
        elif also == "batch":
            # All the lookups in one turn. Measured 2026-09-02, Qwen3.8-Flash-Next spent
            # about seven tool calls and 25 seconds a question, and those calls were one
            # question asked one entry at a time -- nothing in the prompt said the ids are
            # a list. The system text says it, each searching tool is shown a three-entry
            # call, and a turn that reads one entry while more are still unread is told
            # once to read the rest in one call. What it should move is the `calls` column.
            out.append({"label": "batch", "terse": first["terse"], "batch": True,
                        **sampling_from(args)})
        elif also == "kinds":
            # The question word already says what kind the answer is, and the precision
            # misses were mostly right-adjacent: 65% precision against 85% recall, with a
            # topic lit beside the people for a "who" question. Drops from `show` what the
            # question did not ask for, and nothing at all where it named several kinds or
            # none.
            out.append({"label": "kinds", "terse": first["terse"], "kinds": True,
                        **sampling_from(args)})
        elif also == "summary":
            # The broad question -- "what is this group about?" -- has no name in it to
            # look up, so a search answers it with whatever the words happened to hit.
            # `summarise` is the whole graph at a glance, computed without a model.
            out.append({"label": "summary", "terse": first["terse"], "summary": True,
                        **sampling_from(args)})
        elif also == "single":
            # `batch` turned around, and here for the opposite model. A fat tool result is
            # a long thing to hold in mind: a small model handed a dozen entries in one
            # message answers about the last one or about none of them. One entry to a
            # read, more turns, each result short enough to still be in view when the
            # answer is written -- measured against `--also batch` on the same load, since
            # which trade a model wants is a number and not a taste.
            out.append({"label": "single", "terse": first["terse"], "single": True,
                        **sampling_from(args)})
        elif also == "few":
            # Three tools -- look_up, look_at, show -- and no other way of looking, for the
            # model whose tool choice degrades with the number of schemas rather than with
            # the question. Nothing is faked to cover what went: look_up's description says
            # the offer has no path tool and no listing tool, and says how to answer those
            # questions by reading, which is what the loop then does.
            out.append({"label": "few", "terse": first["terse"], "few": True,
                        **sampling_from(args)})
        elif also == "tight":
            warn("note: tight is the default asking now; --also tight measures nothing new "
                 "(--also loose is the old asking, as a control)")
    # `--reach`, `--rounds`, `--batch`, `--kinds`, `--summary` and `--constrain-ids` are
    # not askings of their own: each rides on every asking, so the hundred-question run of
    # "everything that held" is one asking and not four.
    riders: dict[str, Any] = {name: True for name in
                              ("batch", "kinds", "summary", "constrain_ids")
                              if getattr(asked, name)}
    for name in ("reach", "rounds"):
        if getattr(asked, name):
            riders[name] = getattr(asked, name)
    for one in out:
        for name, value in riders.items():
            one.setdefault(name, value)
    return out


def halves(args: Any, model: str = "") -> list[tuple[str, int]]:
    """The ``(suffix, shortlist)`` halves a sweep asks of one model: plain, and shortlisted.

    Every model gets its plain half. The shortlist half goes to every model too, unless
    ``--shortlist-for`` names substrings of the models that should have it -- `e2b,e4b` --
    in which case a model matching none of them is measured plain only. ``--plain-only``
    still means no shortlist half for anything. Matched case aside, against the name the
    model was asked for by and the file it resolved to, so `e2b` finds `gemma-4-E2B-it`.
    """
    if getattr(args, "plain_only", False):
        return [("plain", 0)]
    wanted = [w.strip().lower() for w in str(getattr(args, "shortlist_for", "") or "").split(",")
              if w.strip()]
    if wanted and not any(w in str(model).lower() for w in wanted):
        return [("plain", 0)]
    return [("plain", 0), ("shortlist", int(getattr(args, "shortlist", 0) or 0))]


def _asked(args: Any, parts: Sequence[tuple[str, int]]) -> list[dict[str, Any]]:
    """Every asking one served model is measured with, both halves in one load: each half of
    ``parts`` crossed with each `_askings` variant, labelled ``plain``, ``plain-terse``...

    Loading the model once per half was how the sweep began, and a load is minutes that
    say nothing about the asking. Whether a shortlist is handed over is a question about
    the asking, so it rides on the asking like `terse` does and the server is put up once.
    """
    out: list[dict[str, Any]] = []
    for suffix, shortlist in parts:
        for one in bench._askings(args):
            tag = str(one.get("label", "") or "")
            out.append({**one, "label": f"{suffix}-{tag}" if tag else suffix,
                        "shortlist": shortlist})
    return out


def asking_from(args: Any) -> Asking:
    """The asking given on the command line, and nothing else."""
    return Asking(terse=bool(getattr(args, "terse", False)),
                  reach=int(getattr(args, "reach", 0) or 0) or None,
                  rounds=int(getattr(args, "rounds", 0) or 0) or None,
                  batch=bool(getattr(args, "batch", False)),
                  kinds=bool(getattr(args, "kinds", False)),
                  summary=bool(getattr(args, "summary", False)),
                  constrain_ids=bool(getattr(args, "constrain_ids", False)))


def sampling_from(args: Any) -> dict[str, Any]:
    """The sampler overrides asked for on the command line, and nothing else.

    A setting not given is left out entirely rather than defaulted here, so the client falls
    through to the model's own card. Sweeping them is the point: "is gemma-4 better at the
    temperature its publisher asks for than at 0?" is a question about this graph and these
    questions, and nobody else can answer it for you.
    """
    named = {"n_predict": getattr(args, "n_predict", None),
             "temperature": getattr(args, "temperature", None),
             "top_p": getattr(args, "top_p", None), "top_k": getattr(args, "top_k", None),
             "min_p": getattr(args, "min_p", None)}
    return {k: v for k, v in named.items() if v is not None}


def with_card(client: Any, args: Any) -> Any:
    """The same client, asking with what its model's card recommends, when --card was given.

    This is the only place a card is ever applied. A publisher's advice is a hypothesis about
    a task they have not seen; making it easy to test and impossible to ship by accident is
    the whole arrangement.
    """
    if not getattr(args, "card", False):
        return client
    asked = dict(client.card)
    asked.update(sampling_from(args))          # an explicit flag still beats the card
    if not asked:
        warn(f"note: {client.base_url} serves a model whose card names no sampler settings")
        return client
    return type(client)(client.base_url, model=client.model, family=client.pinned_family,
                        request=replace(client.request, **asked), transport=client.transport)
