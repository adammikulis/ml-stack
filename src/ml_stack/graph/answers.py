"""What a question found: the `Answer` record, what each tool call writes onto it, and
cutting `show` down to the entries the answer is about."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ml_stack.asking import Asking
from ml_stack.client.spent import Spent
from ml_stack.graph.looking import ids_in, kind_of
from ml_stack.vision.payloads import build_message


@dataclass
class Answer:
    """What to say, what to light up, and what was done to find out.

    ``found`` holds what look_up and list_kind returned, ``read`` what look_at was given,
    ``path`` what path_between traversed. ``ids`` is their union — read first, then path, then found —
    capped at converse's ``limit``.

    ``show`` is different in kind from all three: they record what the tools *touched*,
    which is the working, while ``show`` is what the answer is *about*, because the model
    said so. Reading an entry to describe someone else is not writing about it, and a
    person named from a quote was never read at all — so lighting up ``read`` lights the
    search and hides the answer. Empty when the model never said; the caller decides what
    to fall back to.
    """

    content: str = ""
    ids: list[str] = field(default_factory=list)
    found: list[str] = field(default_factory=list)
    listed: list[str] = field(default_factory=list)  # ids a list_kind returned; the cap spares them
    read: list[str] = field(default_factory=list)
    path: list[str] = field(default_factory=list)
    show: list[str] = field(default_factory=list)
    steps: list[str] = field(default_factory=list)
    # How many turns of the loop asked for a tool. Not the same as `spent.tool_calls`: a
    # turn that names three entries in one look_at is one round and three calls, and a turn
    # that reads them one at a time is three rounds. The round is what costs the wall clock,
    # because it is a round trip through the model, so it is what a way like `batch` moves.
    rounds: int = 0
    # which model answered and what it spent -- see `Spent`; `model` is the short form
    spent: Spent = field(default_factory=Spent)

    @property
    def model(self) -> str:
        return self.spent.model

    @property
    def why(self) -> str:
        return "; ".join(self.steps)


def noted(into: list[str], ids: Sequence[str], known: set[str]) -> None:
    """Add to ``into``, in order and without repeating, the ids the graph holds."""
    for one in ids:
        if str(one) in known and str(one) not in into:
            into.append(str(one))


def _read_ids(args: Mapping[str, Any], known: set[str]) -> tuple[list[str], int]:
    """The ids a call named, and how many of them the graph holds."""
    ids = [str(i) for i in (args.get("ids") or ())]
    return ids, sum(1 for i in ids if i in known)


def _voters(rows: Any) -> str:
    """`` by label, words, meaning`` -- the ways those hits were found, in the order they
    first appear; "" when the hits do not say. See `search.hybrid`'s ``matched``."""
    seen: list[str] = []
    for row in rows if isinstance(rows, list) else ():
        for way in (row.get("matched") or ()) if isinstance(row, Mapping) else ():
            if str(way) not in seen:
                seen.append(str(way))
    return " by " + ", ".join(seen) if seen else ""


def recorded(out: Answer, name: str, args: Mapping[str, Any], result: Any,
              known: set[str]) -> Any:
    """``result`` as the model is told it, with what the call found, read and traced
    written onto ``out`` and a step for what it did."""
    if name == "look_up":
        noted(out.found, [r["id"] for r in result] if isinstance(result, list) else [], known)
        asked = [str(x) for x in (args.get("texts") or ())] or [str(args.get("text") or "")]
        out.steps.append("looked up " + ", ".join(repr(x) for x in asked) + _voters(result))
    elif name == "look_at":
        # one guard, in `noted`: an id the model made up is neither read nor lit up
        ids, real = _read_ids(args, known)
        noted(out.read, ids, known)
        result = result or "nothing on those"
        out.steps.append(f"read {real} entr" + ("y" if real == 1 else "ies"))
    elif name == "look_around":
        # The centres are read; everything the result named -- read back off the text, so
        # a neighbourhood the budget cut short is not counted -- is found. Between them
        # that is what look_at's ids and look_up's hits are, so the score, the cap and
        # tight's "did a tool return this" all treat it the same.
        ids, real = _read_ids(args, known)
        noted(out.read, ids, known)
        result = result or "nothing is joined to those"
        noted(out.found, ids_in(str(result)), known)
        out.steps.append(f"looked around {real} entr" + ("y" if real == 1 else "ies"))
    elif name == "path_between":
        noted(out.path, result.get("path") or [], known)
        out.steps.append("traced a path" if result.get("path") else "found no path")
    elif name == "quote":
        ids, real = _read_ids(args, known)
        noted(out.read, ids, known)
        out.steps.append(f"quoted {real} entr" + ("y" if real == 1 else "ies"))
    elif name == "summarise":
        # everything it named is bracketed, as look_around brackets its neighbours, so an
        # entry the summary read out counts as found and may be selected
        noted(out.found, ids_in(str(result)), known)
        out.steps.append("read the graph at a glance")
    elif name == "list_kind":
        listed = result.get("entries") if isinstance(result, Mapping) else None
        if listed is None:
            out.steps.append(f"found no kind {str(args.get('kind') or '')!r}")
        else:
            noted(out.found, [r["id"] for r in listed], known)
            noted(out.listed, [r["id"] for r in listed], known)
            out.steps.append(f"listed {len(listed)} of kind {result.get('kind')!r}")
    elif name == "show":
        # the same guard as look_at: an id the model made up is not lit up
        ids, real = _read_ids(args, known)
        noted(out.show, ids, known)
        out.steps.append(f"selected {real} entr" + ("y" if real == 1 else "ies"))
    else:
        out.steps.append(f"used {name}")
    return result


def without_images(name: str, result: Any, out: Answer) -> tuple[Any, dict[str, Any] | None, int]:
    """``result`` without its images, the message carrying them, and how many went.

    llama.cpp cannot carry an image inside a tool result, so what `web.tools(vision=True)`
    returns under ``_images`` comes out before the result is encoded and goes in as a user
    message of its own. A picture that cannot be prepared is said in the steps and not
    sent: a model told to look at nothing answers about nothing, confidently.
    """
    if not (isinstance(result, Mapping) and "_images" in result):
        return result, None, 0
    result = dict(result)
    pictures = list(result.pop("_images") or ())
    if not pictures:
        return result, None, 0
    seen, report = build_message(f"What {name} returned for the call above, as seen:",
                                 pictures)
    kept = sum(1 for p in seen["content"] if p.get("type") == "image_url")
    if kept:
        return result, seen, kept
    why = "; ".join(report.warnings) or "no reason given"
    out.steps.append(f"{name} returned {len(pictures)} image"
                     + ("" if len(pictures) == 1 else "s")
                     + f" and none could be shown: {why}")
    return result, None, 0


def result_count(result: Any) -> int:
    if isinstance(result, (list, tuple)):
        return len(result)
    if isinstance(result, Mapping) and "entries" in result:
        return len(result.get("entries") or ())
    if isinstance(result, Mapping) and "path" in result:
        return len(result.get("path") or ())
    if isinstance(result, str):
        return len(result.splitlines()) if result.strip() else 0
    return 1 if result else 0


# How many entries a tight answer may light. Six is three times what a good answer wants
# and a third of LIT; past it, the ids the prose names are kept and the rest are cut.
LIT_TIGHT = 6


# ------------------------------------------------------------------ what kind was asked for
#
# The precision miss that is not a retrieval failure. Measured 2026-09-02 on
# Qwen3.8-Flash-Next: 85% recall against 65% precision, and what it lit beside the answer
# was mostly right-adjacent -- the topic the people share, selected for a question that
# asked *who*. The question word already says what kind the answer is, so a `who` question
# keeps the people and drops the topic it found them through.
#
# Only where the question word is unambiguous. A question that asks for two kinds ("who and
# what"), or names no kind at all ("how is X connected to Y", "tell me about X"), filters
# nothing: a filter that guesses wrong empties the selection, and an empty selection is
# worse than a loose one.
_SEVERAL = re.compile(
    r"\bwho and what\b|\bwhat and who\b"
    r"|\b(?:people|persons?) and (?:companies|organisations?|organizations?|orgs?|firms?"
    r"|places|towns?|topics?|subjects?|events?)\b"
    r"|\b(?:companies|organisations?|organizations?|orgs?|firms?|places|towns?|topics?"
    r"|subjects?|events?) and (?:people|persons?)\b")
# (kind, what a question asking for it says). Read against the *asking* clause -- the last
# sentence -- because "Somebody is selling a machine and needs help. What do they need?"
# asks about a subject and merely mentions a person on the way.
#
# Checked over the bench's own 110 questions before it was measured on a model: 72 are
# filtered to the kind their expected answer actually is, 31 are left alone, and exactly one
# is filtered wrongly -- "Who is offering work keeping a production line running?", whose
# answer is the company rather than a person. That is the shape of the trade, and it is why
# a question naming several kinds or none filters nothing.
ASKED_FOR: tuple[tuple[str, str], ...] = (
    # "who" only where it opens the asking, or follows a comma. In the middle of a clause
    # it is a relative pronoun and says nothing about the answer: "Which places do the
    # people *who* do repair live in?" asks for places, and reading that `who` as the
    # question word filtered the places out (measured against the bench's own set).
    ("person", r"(?:^|,\s*)(?:whos?|whom|whose)\b"
               r"|\bsomebody\b|\bsomeone\b|\banybody\b|\banyone\b"
               r"|\bwhich (?:person|people|members)\b|\bwhat people\b"),
    ("org", r"\bwhich (?:company|companies|organisations?|organizations?|orgs?|firms?"
            r"|business|businesses|employers?)\b"
            r"|\bwhat (?:company|companies|organisations?|organizations?|firms?|employers?)\b"
            r"|\bwhere (?:do|does|did)\b[^?]*\bworks?\b"),
    ("place", r"\bwhich (?:place|places|towns?|cit(?:y|ies))\b"
              r"|\bwhat (?:place|places|towns?|cit(?:y|ies))\b"
              r"|\bwhere\b[^?]*\b(?:based|live|lives|lived|located)\b"),
    ("topic", r"\bwhich (?:topics?|subjects?)\b|\bwhat (?:topics?|subjects?)\b"
              r"|\btalk about\b|\btalking about\b|\bdiscuss(?:ed|ing)?\b"),
    ("event", r"\bwhich events?\b|\bwhat events?\b"),
    ("opportunity", r"\bopenings?\b|\bopportunit(?:y|ies)\b|\bvacanc(?:y|ies)\b"
                    r"|\bwork going\b|\bany work\b"),
)
_ASKED = tuple((kind, re.compile(pattern)) for kind, pattern in ASKED_FOR)
_SENTENCE = re.compile(r"(?<=[.?!])\s+")


def asking_clause(question: str) -> str:
    """The sentence of a question that actually asks it: the last one.

    "I need two people to build a prototype. Who?" asks for people, and "Somebody is
    selling a machine. What do they need?" does not, though both say a person word. What
    comes last is what is being asked.
    """
    parts = [p for p in _SENTENCE.split(" ".join((question or "").split())) if p.strip()]
    return parts[-1] if parts else ""


def asked_kinds(question: str) -> set[str] | None:
    """The kind of entry this question asks for, or ``None`` for "it did not say".

    ``None`` means filter nothing, and is the answer whenever the question named several
    kinds or none: only a question whose own words settle the kind is acted on.

        asked_kinds("Who fixes machines?")        # {"person"}
        asked_kinds("Which company does surveying?")  # {"org"}
        asked_kinds("Tell me about Otto Vance.")  # None
    """
    clause = asking_clause(question).casefold()
    if not clause or _SEVERAL.search(clause):
        return None
    found = {kind for kind, pattern in _ASKED if pattern.search(clause)}
    return found if len(found) == 1 else None


_NOT_A_WORD = re.compile(r"[^\w]+")


def _words(text: Any) -> str:
    return " " + " ".join(_NOT_A_WORD.sub(" ", str(text or "").casefold()).split()) + " "


def _joined_to(graph: Mapping[str, Any], ids: Sequence[str]) -> set[str]:
    """Every id on the other end of an edge from one of `ids`: what a look_at showed the model
    beside the entry itself, and so a name it read rather than guessed."""
    wanted = set(ids)
    out: set[str] = set()
    for edge in graph.get("edges") or ():
        a, b = str(edge.get("source") or ""), str(edge.get("target") or "")
        if a in wanted:
            out.add(b)
        if b in wanted:
            out.add(a)
    return out


def _named_counts(graph: Mapping[str, Any], text: str, ids: Sequence[str]) -> dict[str, int]:
    """How many times the prose names each of those entries, by label, as whole words.

    Case does not matter; a label shorter than three characters is never matched, as on
    the page, because "Al" is in "already". An entry the prose never names is absent.
    """
    said = _words(text)
    if not said.strip():
        return {}
    label = {str(n["id"]): str(n.get("label") or "") for n in (graph.get("nodes") or ())}
    out: dict[str, int] = {}
    for one in ids:
        name = label.get(one, "")
        key = _words(name).strip()
        if len(name) < 3 or not key:
            continue
        hits = len(re.findall(r"(?<= )" + re.escape(key) + r"(?= )", said))
        if hits:
            out[one] = hits
    return out


def _of_the_kind(out: Answer, graph: Mapping[str, Any], question: str) -> None:
    """Drop from `show` what is not the kind the question asked for.

    The question word said what kind the answer is, so what is not of that kind was found
    on the way rather than asked for -- the topic that led to the people, for a question
    that asked *who*. A listing is exempt, as it is from the cap; so is a question that
    named several kinds or none, which `asked_kinds` answers None for.
    """
    wanted = asked_kinds(question)
    if not wanted:
        return
    listed = set(out.listed)
    of_kind = {str(n["id"]): kind_of(n) for n in (graph.get("nodes") or ())}
    kept = [i for i in out.show
            if i in listed or of_kind.get(i, "") in wanted or i not in of_kind]
    # A filter that empties the selection lights nothing, which is worse than lighting the
    # wrong kind: the reader is left looking at a blank graph.
    if kept and len(kept) < len(out.show):
        out.steps.append(f"dropped {len(out.show) - len(kept)} of another kind from show, "
                         f"which asked for " + "/".join(sorted(wanted)))
        out.show = kept


def selected(out: Answer, graph: Mapping[str, Any], question: str, asking: Asking,
              start: Sequence[str]) -> None:
    """Cut `show` down to what the answer is about: a name no tool ever returned, a kind
    the question did not ask for, and anything over `LIT_TIGHT`."""
    named = _named_counts(graph, out.content, out.show)
    if asking.tight:
        # The `made` case: a name in the prose that look_at never read is a guess, and
        # lighting it endorses the guess. Known is what any tool returned -- found by
        # look_up, listed by list_kind, read, traced, handed over -- and what is joined to
        # something read: measured 2026-09-02, `place:calderwick` was listed and joined to
        # the people read, and dropping it as "unread" cut a right answer. `out.found` is
        # where a `look_around` neighbourhood lands, so a neighbour the model only ever saw
        # indented under something it asked for still counts as read.
        seen = (set(out.read) | set(start) | set(out.found) | set(out.path)
                | _joined_to(graph, out.read))
        guessed = [i for i in out.show if named.get(i) and i not in seen]
        if guessed:
            out.show = [i for i in out.show if i not in guessed]
            out.steps.append(f"dropped {len(guessed)} unread from show")
    if asking.kinds and out.show:
        _of_the_kind(out, graph, question)
    if asking.tight:
        # The cap never cuts a listing: "which companies are here?" has as many right
        # answers as there are companies, and cutting thirteen to six alphabetically threw
        # away three expected ones (measured 2026-09-02). It cuts among the rest, keeping
        # what the prose names first.
        listed = set(out.listed)
        rest = [i for i in out.show if i not in listed]
        if len(rest) > LIT_TIGHT:
            whole = len(out.show)
            kept = sorted(rest, key=lambda i: -named.get(i, 0))[:LIT_TIGHT]
            out.show = [i for i in out.show if i in listed] + kept
            out.steps.append(f"cut {whole - len(out.show)} of {whole} lit")
