"""One section of a document through the model: the vocabulary, the instructions,
the prompt, and what one extraction cost."""

from __future__ import annotations

import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ml_stack.ingest.reads import Read

__all__ = ["IMAGES_PER_SECTION", "INSTRUCTIONS", "PER_SECTION", "VERBS", "WITH_IMAGES",
           "extract_unit", "prompt_for", "schema"]


PER_SECTION = 1200.0    # a ceiling, not a budget: a legitimate unit writes 6k tokens at ~50 tok/s


IMAGES_PER_SECTION = 4
"""How many of a section's figures are shown to the model at once. A section of a biology
textbook has a dozen plates; a dozen images is a prompt of images with a paragraph in it."""


VERBS: dict[str, str] = {
    "is_a": "a kind of (mitochondria is_a organelle)",
    "part_of": "belongs inside a larger whole (nucleus part_of cell)",
    "has_part": "the whole, naming a part (cell has_part nucleus)",
    "causes": "brings about (mutation causes variation)",
    "produces": "makes or yields (mitochondria produces ATP)",
    "consumes": "uses up (photosynthesis consumes carbon dioxide)",
    "regulates": "controls the rate or amount of (insulin regulates blood glucose)",
    "located_in": "found in a place or region (mitochondria located_in cytoplasm)",
    "measured_in": "the unit a quantity takes (force measured_in newton)",
    "defined_by": "fixed by a law, equation or definition (momentum defined_by mass times velocity)",
    "example_of": "one instance of a general thing (glucose example_of monosaccharide)",
    "contrasts_with": "set against, as the text opposes them (prokaryote contrasts_with eukaryote)",
    "precedes": "comes before, in a sequence or in time (prophase precedes metaphase)",
    "requires": "cannot happen without (respiration requires oxygen)",
    "converts": "turns one thing into another (the sun converts hydrogen, fuses it, into helium)",
    "created_by": "written, proposed, discovered or built by a person or body (the declaration created_by its author)",
    "adopted_by": "enacted, ratified or taken up by a body (the declaration adopted_by the congress)",
    "member_of": "one of a group, class or body (a delegate member_of the congress)",
}
"""The closed relation vocabulary, each verb glossed with the sense the model should take.

The schema's enum is this list; a test keeps them equal. The gloss is what moved
precision on the Slack graph -- a model told what a verb means uses it for that and
nothing else -- so every verb here has one, and a verb without a gloss is not added."""


def _verbs_line() -> str:
    return "The verb phrases, and what each means: " + "; ".join(
        f"{verb} -- {gloss}" for verb, gloss in VERBS.items()) + ".\n"


INSTRUCTIONS = (
    "You are reading one section of a textbook into a knowledge graph. List the concepts "
    "the section names, how they stand to one another, what its figures show, and the "
    "terms it defines.\n"
    "Invent nothing. Every concept, relation and definition must be stated in the text "
    "you were given; a fact you know from elsewhere does not belong here.\n"
    "A definition is the book's own words, cut to one line, and only when the section "
    "defines the thing. When it does not, the definition is an empty string -- an empty "
    "string is always better than a definition you wrote yourself.\n"
    "`aliases` are other names this same section uses for the same thing: a plural, an "
    "abbreviation, a symbol. Not synonyms you happen to know.\n"
    "A relation joins two concept names from your own `concepts` list, using one of the "
    "verb phrases the schema allows and no other. State only what the section states. "
    "Both ends are concept names -- never a clause or a phrase such as 'lights the "
    "system'. When no verb says what the text says, leave the relation out: a relation "
    "with the wrong verb is worse than none.\n"
    + _verbs_line() +
    "A caption is marked in the text as [Figure 2.9]. For each figure, `shows` is what "
    "the picture shows in one line, and `concepts` are only those the caption or the "
    "surrounding text says it illustrates -- never a concept guessed from the picture.\n"
    "Return only JSON matching the schema."
)

WITH_IMAGES = (
    "\nThe section's figures follow the text as pictures. Use them to say what each figure "
    "shows; still take the concepts a figure illustrates from what the caption and the text "
    "say, not from the picture alone."
)


def schema() -> dict[str, Any]:
    """The document extraction shape, read from the contracts."""
    from ml_stack.contracts import load

    return dict(load("extraction-document.schema.json"))


class _Recording:
    """A client that keeps a `Call` for every reply it gets, and is otherwise the client.

    `Client.extract` calls the client's own `chat`, so binding `extract` here puts every
    call an extraction makes through the recording one -- the same trick, and for the same
    reason, as the bench's `_Extracting`.
    """

    def __init__(self, client: Any, *, host: str = "", port: int = 0) -> None:
        self._client = client
        self._host, self._port = host, port
        self.calls: list[Any] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)

    def chat(self, *args: Any, **kwargs: Any) -> Any:
        from ml_stack.telemetry import Call

        began = time.time()
        reply = self._client.chat(*args, **kwargs)
        self.calls.append(Call.from_reply(reply, time.time() - began, tool="extract",
                                          args={}, host=self._host, port=self._port))
        return reply

    def extract(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        from ml_stack.client.chat import Client

        return Client.extract(self, *args, **kwargs)  # type: ignore[arg-type]

    def _chat_extractor(self, *args: Any, **kwargs: Any) -> Any:
        from ml_stack.client.chat import Client

        return Client._chat_extractor(self, *args, **kwargs)  # type: ignore[arg-type]

    def _raw_extractor(self, *args: Any, **kwargs: Any) -> Any:
        from ml_stack.client.chat import Client

        return Client._raw_extractor(self, *args, **kwargs)  # type: ignore[arg-type]


def prompt_for(unit: Any, *, images: bool = False,
               most: int = IMAGES_PER_SECTION) -> tuple[list[dict[str, Any]], int]:
    """The turns one section is extracted from, and how many pictures went with it.

    The section is named before its text -- a model reading "2.1 Atoms, Isotopes, Ions and
    Molecules" knows what the pronouns in the first paragraph refer to. With ``images`` the
    figures that rendered go in as a user message of their own after the text, which is the
    `_images` convention `graph.ask` uses for a tool that brings pictures back: llama.cpp
    cannot carry an image inside anything but a user turn.
    """
    where = " / ".join(x for x in (unit.book_title, f"Chapter {unit.chapter}"
                                   if unit.chapter else "", unit.chapter_title) if x)
    head = f"{where}\n{unit.section} {unit.section_title}".strip()
    if unit.parts > 1:
        head += f" (part {unit.part} of {unit.parts})"
    terms = ("\n\nTerms this section sets in bold: " + ", ".join(unit.key_terms)
             if unit.key_terms else "")
    turns: list[dict[str, Any]] = [
        {"role": "system", "content": INSTRUCTIONS + (WITH_IMAGES if images else "")},
        {"role": "user", "content": f"{head}\n\n{unit.text}{terms}"},
    ]
    if not images:
        return turns, 0
    pictures = [f.png for f in unit.figures if f.png][:most]
    if not pictures:
        return turns, 0
    from ml_stack.vision.payloads import build_message

    seen, report = build_message("The figures of this section, in order:", list(pictures))
    kept = sum(1 for part in seen["content"] if part.get("type") == "image_url")
    if not kept:
        # a picture that cannot be prepared is not sent and not claimed: a model told to
        # look at nothing answers about nothing, confidently
        return turns, 0
    turns.append(seen)
    return turns, kept


def extract_unit(client: Any, unit: Any, shape: Mapping[str, Any], *, images: bool = False,
                 per_section: float = PER_SECTION, cache_dir: str | Path | None = None) -> Read:
    """One unit through ``client.extract``, and what it cost.

    A failure is a result, not the end of the run: the row keeps the error and the next
    section is read. ``think=False`` -- reading a page is a reading, not a reasoning, and
    the thinking channel is where a ceiling gets spent.
    """
    row = Read(unit=unit.id, source=unit.source, chapter=unit.chapter, section=unit.section,
               title=unit.section_title, pages=[unit.first_page, unit.last_page])
    turns, shown = prompt_for(unit, images=images)
    row.images = shown
    recording = _Recording(client)
    began = time.time()
    try:
        got = recording.extract(unit.text, dict(shape), messages=turns, think=False, tries=1,
                                cache_dir=cache_dir,
                                cache_extra=f"document/{unit.id}/{int(images)}")
        row.extracted = got if isinstance(got, dict) else {}
    except Exception as exc:  # noqa: BLE001 - one bad section does not end a run
        row.error = f"{type(exc).__name__}: {exc}"[:200]
        # a unit that ran to the ceiling twice on the first night left nothing to read
        # but 120 characters; the whole reply is kept beside the unit, so the next person
        # can see whether it looped or rambled without spending ten minutes of GPU again
        row.raw = str(getattr(exc, "body", "") or "")
    row.seconds = round(time.time() - began, 2)
    if per_section and row.seconds >= per_section and row.error:
        row.timed_out = True
    row.calls = [call.public() for call in recording.calls]
    got = row.extracted
    row.concepts = len(got.get("concepts") or ())
    row.relations = len(got.get("relations") or ())
    row.figures = len(got.get("figures") or ())
    return row
