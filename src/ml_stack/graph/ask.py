"""Asking a model a question about a graph.

Handing a model the whole graph does not scale and handing it a pre-chosen slice makes the
choosing the answer. This gives it five things it can do instead — find entries by name, read
what is held on them, read a whole neighbourhood at once, trace how two of them connect, list
everything of one kind — and lets it
decide which to use. What it touched comes back with the answer, which is what a caller needs
to show its working.

The graph is a mapping with ``nodes`` and ``edges``; nothing here cares what a project calls
its kinds or relations.

    reply = converse("how are Ada and Bea connected?", graph, client)
    reply.content   # what to say
    reply.ids       # what to light up
    reply.steps     # what it did to find out
"""

from __future__ import annotations

import copy
import json
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ml_stack.client.spent import Spent
from ml_stack.client.tokens import estimate_tokens
from ml_stack.graph.search import MATCHED_BY_RANK

SYSTEM = (
    "You are answering a question about a graph. You cannot see it; you read it with the tools "
    "you have been given. Look up the names in the question to get their ids, read what is held "
    "on them, and when the question is about how two things relate, trace the path between "
    "them. When some entries are named as currently highlighted, read them so your answer "
    "knows what the reader is looking at; that is context, not subject matter, and an entry "
    "belongs in your answer only if you have something to say about it.\n\n"
    "Then write the answer. Do not narrate what you looked up — the reader can see that "
    "already. Say what the entries add up to: what they have in common, where they differ, "
    "what connects them, what a reader should do with it. Quote the words that make your point "
    "when the graph holds them. Four to eight sentences of plain prose, no bullet points and no "
    "headings, naming the things you mean rather than their ids.\n\n"
    "Last, call show once with the ids of the entries your answer is about, so the reader can "
    "see them on the graph. Every one you named belongs in it — including any you named from a "
    "quote rather than by reading it. What you opened on the way and did not write about does "
    "not.\n\n"
    "Everything you say comes from what the tools returned. Say plainly when the graph does not "
    "answer the question, and never invent an entry the tools did not show you."
)

# How many tool calls a question may spend. Five was enough to look two names up and read
# them; it is not enough to staff a project, which means looking up each skill, reading the
# people behind them, and then saying which of them the answer is about. Measured against a
# real graph: every staffing question spent all five on searching and never reached `show`.
ROUNDS = 10
# How many times a model may ask for something it has already asked for before the tools are
# taken away. Two is a stumble; three is a loop, and a loop always ends in no answer at all.
REPEATS = 2

# The tools that look for something, and so are taken away on the last turn; what is left
# then is the tools that act (show, and whatever a caller adds -- a change request, say).
# The web tools are searches too: a model that may still search will search instead of
# answering, which is the failure the last turn exists to end.
SEARCHING = frozenset({"look_up", "look_at", "look_around", "path_between", "list_kind",
                       "summarise", "web_search", "web_read", "web_look"})
# Openings that mean the model is planning rather than answering. gpt-oss puts its analysis
# in a channel of its own, but when a turn spends its whole budget deciding what to do the
# reply that comes back IS the analysis — not empty, so nothing caught it, and the reader
# was shown "We need to answer: ... Search for X again maybe missing." as the answer.
WORKING = ("we need to", "i need to", "need to", "need ", "let me", "let's", "lets ", "from data:",
           "the user asks", "the user wants", "first, i", "i should", "we should", "okay,",
           "ok,", "so the question", "search for", "we must", "i'll produce", "we can use",
           "we need", "look up ", "maybe ")


# A sentence end, including the one with no space after it: the leak arrives welded to the
# answer as "…tool search.Grace Hopper brings…", and splitting on ". " alone keeps them
# as one sentence, so cutting the notes cuts the answer with it.
_ENDS = re.compile(r"(?<=[.!?])(?=\s|[A-Z\u201c\u2018\"'])")


def _sentences(said: str) -> list[str]:
    return [x.strip() for x in _ENDS.split((said or "").replace("\n", " ")) if x.strip()]


def _is_note(one: str) -> bool:
    head = " ".join((one or "").strip().split()).casefold()
    return any(head.startswith(x) for x in WORKING)


def without_notes(said: str) -> str:
    """``said`` with the planning it opens with removed.

    Measured against a real model: a good answer arrives with one sentence of the analysis
    channel stuck to the front — "We need to use tool search.Grace Hopper brings decades
    of healthcare experience…". Throwing the whole reply away over that opening loses a real
    answer and shows the reader "the model did not finish an answer", which is worse than
    the leak. So the leak is cut off and the answer kept.
    """
    parts = _sentences(said)
    while parts and _is_note(parts[0]):
        parts.pop(0)
    return " ".join(parts).strip()


# A tool call written out as prose. On the last turn the tools are taken away so the model
# answers in words — and a model that wanted to call `show` writes the call instead, on the
# end of an otherwise good answer: "…could enhance Dan's swarm projects.
# show({"ids":["person:ada","person:bea"]})". It is not an answer, so it is cut; but it is
# also the model saying exactly what it meant, so the ids are kept.
_SPOKEN = re.compile(r"\s*(?:<[^>]*>)?\s*show\s*\(\s*(\{.*\})\s*\)\s*(?:<[^>]*>)?\s*$",
                     re.DOTALL | re.IGNORECASE)


def spoken_show(said: str) -> tuple[str, list[str]]:
    """``said`` without a trailing written-out ``show`` call, and the ids it named."""
    found = _SPOKEN.search(said or "")
    if not found:
        return (said or "").strip(), []
    try:
        args = json.loads(found.group(1))
    except ValueError:
        return said[:found.start()].strip(), []
    ids = [str(i) for i in (args.get("ids") or ())] if isinstance(args, Mapping) else []
    return said[:found.start()].strip(), ids


def is_working(said: str) -> bool:
    """Whether a reply is notes and nothing else.

    Length is not the test — "Nobody in the graph does both." is a short answer, not a note.
    What decides it is whether anything survives taking the planning off the front.
    """
    return bool((said or "").strip()) and not without_notes(said)


FOUND = 12
JOINED = 12
# How many people a rich look_up hit brings with it. A topic found is only halfway to who
# has it, and measured against a real graph every staffing question spent rounds guessing
# spellings because nothing joined a skill to its people in one call. Eight is enough to
# staff from and few enough that a hit with its people is still a line, not a page.
JOINED_HITS = 8
SAID = 2
SAID_CHARS = 220
LIT = 25
# How many entries list_kind reads out of one kind. Enough for every organisation a
# community of a few hundred people works for; a kind bigger than that is read most
# mentioned first, and the total is stated so the model knows what it did not see.
LISTED = 40
# How many joined entries `look_around` reads out beside each entry it was given, and how
# many entries it will read out in all. Twelve is `JOINED` -- what look_at already shows as
# names -- said properly, with each one's kind, id and a line of its own words, so a model
# can select a neighbour it never looked up.
AROUND = 12
AROUND_ENTRIES = 24
# The flat cut on one tool result: the characters a tool message has always been trimmed to.
# A conversation given a `reach` cuts by tokens instead -- see `_cut`.
CUT = 6000

# Every example below is invented, and deliberately shares no name with the community the
# bench asks its questions of: an example that used the bench's own people would be teaching
# the answers rather than the calling convention, and the score would stop meaning anything.
#
# The examples are here because they are what the small models need. Measured over the
# invented community at 32k: gemma-4-E4B scored 17%, and six of its nine failures were the
# same shape — two model calls, a hundred characters of prose, no tool call at all. It was
# not reasoning badly, it was answering from nothing. gpt-oss failed the opposite way, with
# *more* calls on a wrong answer than a right one. A description that says what a tool is
# leaves a small model to infer that it should be called; one that shows a call does not.
TOOLS = [
    {"type": "function", "function": {
        "name": "look_up",
        "description": "Search the graph for entries whose name or attached words match some "
                       "text. Call this first, before answering anything: you have never seen "
                       "this graph, nothing in it is in your memory, and an entry that this "
                       "tool did not return does not exist. Pass every word you want in one "
                       "call — it costs the same as one, and asking one at a time is how a "
                       "question runs out of turns. If the question's own words find "
                       "nothing, search the idea behind them instead of asking again with "
                       "more of them: a cracked kiln is looked for as \"ceramics\", "
                       "\"firing\", \"studio\", not as \"cracked kiln\" again. Example: for "
                       "\"who runs a pottery studio in "
                       "Ambleford?\" call look_up with "
                       "{\"texts\": [\"pottery\", \"studio\", \"Ambleford\"]}, not three "
                       "separate calls.",
        "parameters": {"type": "object", "properties": {
            "text": {"type": "string", "description": "one thing to look for, e.g. \"glassblowing\""},
            "texts": {"type": "array", "items": {"type": "string"},
                      "description": "several things to look for, in one call, e.g. "
                                     "[\"pottery\", \"studio\", \"Ambleford\"]"}},
            "required": []}}},
    {"type": "function", "function": {
        "name": "look_at",
        "description": "Read what the graph holds on entries you found: their attributes, what "
                       "they are joined to, and a line or two of what was actually said. "
                       "look_up gives you names only — this is where the facts to answer from "
                       "come from, so call it on anything you intend to write about. Example: "
                       "look_at with {\"ids\": [\"person:wren\", \"topic:ceramics\"]}.",
        "parameters": {"type": "object", "properties": {
            "ids": {"type": "array", "items": {"type": "string"},
                    "description": "entry ids exactly as look_up returned them, e.g. "
                                   "[\"person:wren\", \"org:tinsley\"]"}},
            "required": ["ids"]}}},
    # The call that replaces four of them. Measured 2026-09-02 on Qwen3.8-Flash-Next, a
    # hybrid recurrent model with 256k of context at 48K bytes a token: recall 89-95%, and a
    # question cost 5-9 tool calls at about 2k new tokens each, half the wall clock spent
    # reading results back at ~390 tok/s and the other half writing at ~35 tok/s. Reading is
    # eleven times cheaper than writing for that model, so the way to make a question
    # cheaper is fewer, fatter calls -- one look_up and one look_around instead of a look_up
    # and five look_at's. A small model has the opposite profile and is not made to use it.
    {"type": "function", "function": {
        "name": "look_around",
        "description": "Read a whole neighbourhood at once: the entries you name — what is "
                       "held on them and what they said — and, under each, everything joined "
                       "to it, with the relation, that entry's kind, its id and a line of its "
                       "own words. One call where reading each neighbour separately would "
                       "take five. Call it when the question is about who or what is around "
                       "something rather than about one entry on its own — who could help "
                       "with a topic, who a place holds, what an organisation is tied to. "
                       "Everything it returns you have read: you may write about a neighbour "
                       "and select it by the id in brackets without looking it up. Example: "
                       "having found \"topic:ceramics\", call look_around with "
                       "{\"ids\": [\"topic:ceramics\"]}; pass {\"ids\": "
                       "[\"topic:ceramics\"], \"hops\": 2} to take in the neighbours' "
                       "neighbours too.",
        "parameters": {"type": "object", "properties": {
            "ids": {"type": "array", "items": {"type": "string"},
                    "description": "entry ids to read the neighbourhood of, e.g. "
                                   "[\"topic:ceramics\", \"place:ambleford\"]"},
            "hops": {"type": "integer",
                     "description": "how far out to read: 1 for the entries and what is "
                                    "joined to them (the default), 2 to go one further"}},
            "required": ["ids"]}}},
    {"type": "function", "function": {
        "name": "path_between",
        "description": "Trace how two entries are connected, as the chain of entries between "
                       "them. Use it when the question is about a relationship rather than a "
                       "fact — who could introduce two people, how someone relates to a "
                       "company — and they are not already joined directly. Example: "
                       "path_between with {\"from_id\": \"person:wren\", "
                       "\"to_id\": \"org:tinsley\"}.",
        "parameters": {"type": "object", "properties": {
            "from_id": {"type": "string", "description": "an entry id, e.g. \"person:wren\""},
            "to_id": {"type": "string", "description": "another entry id, e.g. \"person:hollis\""}},
            "required": ["from_id", "to_id"]}}},
    # The one question no description of look_up reaches: "which companies do people here
    # work for?" Nothing in the graph is *labelled* company, so a small model searches
    # "company", then "organization", finds nothing, and gives up. The capability was
    # absent, not badly described -- the large models got it right only by reading enough
    # of the graph to have seen every organisation on the way.
    {"type": "function", "function": {
        "name": "list_kind",
        "description": "List every entry of one kind the graph holds — all the organisations, "
                       "all the topics, all the places — most mentioned first. Use it when "
                       "the question asks what is represented here rather than for a name: "
                       "look_up matches words, and no entry is labelled \"company\", so "
                       "searching for one finds nothing. Companies, organisations, businesses "
                       "and employers are all the kind \"org\". Examples: for \"Which "
                       "companies do people here work for?\" call list_kind with "
                       "{\"kind\": \"org\"}; for \"What topics come up here?\" call list_kind "
                       "with {\"kind\": \"topic\"}. When the kind you guessed does not exist, "
                       "the result names the kinds that do — call it again with one of those.",
        "parameters": {"type": "object", "properties": {
            "kind": {"type": "string",
                     "description": "one kind of entry, e.g. \"org\", \"topic\", \"place\""}},
            "required": ["kind"]}}},
    {"type": "function", "function": {
        "name": "show",
        "description": "Say which entries your answer is about, so they are selected on the graph. "
                       "Every answer ends with this call. An answer without it selects nothing "
                       "and the reader is left looking at an empty graph, however good the "
                       "words were. Pass everyone and everything you actually wrote about — "
                       "including anyone you named from a quote — and nothing you merely "
                       "opened on the way. A question that asks *who* is answered by people: "
                       "show the people, not the subject they have in common. Example: having "
                       "written \"Wren Halloway fires "
                       "the kiln and Hollis Fen runs the studio\", call show with "
                       "{\"ids\": [\"person:wren\", \"person:hollis\"]}.",
        "parameters": {"type": "object", "properties": {
            "ids": {"type": "array", "items": {"type": "string"},
                    "description": "entry ids the answer is about, e.g. "
                                   "[\"person:wren\", \"person:hollis\"]"}},
            "required": ["ids"]}}},
]


# The same six tools, said briefly. What a model needs to be told depends entirely on the
# model: the worked examples above took gemma-4-E4B from 17% to 70% recall, and cost
# gpt-oss-120b twenty points over the same questions. A model that already reaches for a
# tool does not need telling to, and being told anyway spends its attention on instructions
# instead of on the question.
#
# So both exist and the caller chooses. `ml-stack-bench --terse` measures which a given
# model wants, because there is no answering that from first principles.
TERSE = [
    {"type": "function", "function": {
        "name": "look_up",
        "description": "Find entries in the graph whose name or attached words match some "
                       "text. Several words in one call cost the same as one.",
        "parameters": {"type": "object", "properties": {
            "text": {"type": "string", "description": "what to look for"},
            "texts": {"type": "array", "items": {"type": "string"},
                      "description": "several things to look for, in one call"}},
            "required": []}}},
    {"type": "function", "function": {
        "name": "look_at",
        "description": "What the graph holds on some entries: their attributes, what they "
                       "are joined to, and a line or two of what was actually said.",
        "parameters": {"type": "object", "properties": {
            "ids": {"type": "array", "items": {"type": "string"},
                    "description": "entry ids, as returned by look_up"}},
            "required": ["ids"]}}},
    {"type": "function", "function": {
        "name": "look_around",
        "description": "The neighbourhood of some entries in one call: each entry, and "
                       "everything joined to it with the relation, kind, id and a line of "
                       "its words. One call instead of reading each neighbour separately.",
        "parameters": {"type": "object", "properties": {
            "ids": {"type": "array", "items": {"type": "string"},
                    "description": "entry ids, as returned by look_up"},
            "hops": {"type": "integer", "description": "1 for what is joined to them, 2 to "
                                                       "go one further"}},
            "required": ["ids"]}}},
    {"type": "function", "function": {
        "name": "path_between",
        "description": "How two entries are connected, as the chain of entries between "
                       "them. For a question about how two things relate.",
        "parameters": {"type": "object", "properties": {
            "from_id": {"type": "string"}, "to_id": {"type": "string"}},
            "required": ["from_id", "to_id"]}}},
    {"type": "function", "function": {
        "name": "list_kind",
        "description": "Every entry of one kind -- org, topic, place -- most mentioned first. "
                       "A kind that does not exist is answered with the kinds that do.",
        "parameters": {"type": "object", "properties": {
            "kind": {"type": "string", "description": "one kind of entry"}},
            "required": ["kind"]}}},
    {"type": "function", "function": {
        "name": "show",
        "description": "The entries your answer is about, to select on the graph. Call it "
                       "once, last, with what you actually wrote about -- not everything "
                       "you opened on the way.",
        "parameters": {"type": "object", "properties": {
            "ids": {"type": "array", "items": {"type": "string"},
                    "description": "entry ids the answer is about"}},
            "required": ["ids"]}}},
]


# What look_up says when its hits carry why they matched and who is joined to them. It is
# added to a *copy*: the base descriptions are what the answer cache fingerprints and what
# the bench is measuring, so with the flag off they stay byte for byte as they were.
RICH_SENTENCE = ("Each hit says why it matched and, for a topic or a place, who is joined to "
                 "it — read those people rather than searching for them.")


def _rich(schemas: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Those schemas, copied, with look_up told what its hits now carry."""
    out = copy.deepcopy(list(schemas))
    for schema in out:
        if schema["function"]["name"] == "look_up":
            schema["function"]["description"] += " " + RICH_SENTENCE
    return out


RICH = _rich(TOOLS)
RICH_TERSE = _rich(TERSE)


# The tight asking: what `show` says, and what the model is told, when a model that finds
# nearly everything then lights far more than the answer. Measured over the invented
# community, 34 questions at 32k: Qwen3.8-Flash-Next reached 92% recall and 44% precision,
# lighting and naming entries it had looked at on the way, and named 70 entries its tools
# never found, read or showed. A good answer lights about two. E4B, at 60% recall, was at
# 47% precision -- so the words about `show` are all this changes, and nothing about the
# searching.
#
# Said on copies, for the same reason as RICH: the base descriptions, the nudge and SYSTEM
# are what the answer cache fingerprints and what the ranking measured, so under
# `tight=False` -- the loose asking, kept as the control -- they stay byte for byte as they
# were. Tight is what everything asks otherwise (Adam, 2026-09-02: "let's not ever use
# plain then").
TIGHT_SENTENCE = ("Select only the entries that answer the question — the people or things "
                  "the asker would act on — never the ones you looked at on the way, and "
                  "never every name in a quote. Most questions have one to three; a question "
                  "asking which of a kind there are lists them all, and a question asking how "
                  "two entries are connected shows the whole chain between them.")
TIGHT_SHOW = ("Select the entries your answer is about — the reader acts on that selection. "
              "Every answer ends with this call. " + TIGHT_SENTENCE + " A question that asks "
              "*who* is answered by people: select the people, not the subject they have in "
              "common. "
              "Example: having written \"Wren Halloway fires the kiln and Hollis Fen runs "
              "the studio\", call show with {\"ids\": [\"person:wren\", \"person:hollis\"]}.")
TIGHT_SHOW_TERSE = ("Select the entries your answer is about. Call it once, last. "
                    + TIGHT_SENTENCE)
# What the model is asked when it answered without saying what to light. The first is the
# text the ranking runs were measured with and what `tight=False` still sends, so it stays
# as it is; the second is the one every other run gets.
SHOW_NUDGE = ("Now call show once with the ids of the entries your answer is about — everyone "
              "and everything you named in it, including any you named from a quote. Nothing "
              "you opened and did not write about.")
TIGHT_NUDGE = ("Now call show once with only the entries that answer the question — the ones "
               "the asker would act on. Not what you read on the way, not every name you "
               "mentioned. Usually one to three; all of them for a which-of-a-kind question, "
               "and the whole chain for a how-are-they-connected question.")
# Added to SYSTEM by the tight asking, and so absent only under `tight=False`: the `made`
# case is a name in the prose that no tool ever returned, and the remedy is to say so rather
# than guess.
# The base system prompt tells the model every name it wrote belongs in show; tight says the
# opposite, so its copy of the prompt swaps that paragraph rather than arguing with it.
SHOW_PARAGRAPH = ("Every one you named belongs in it — including any you named from a "
                  "quote rather than by reading it. What you opened on the way and did not "
                  "write about does not.")
TIGHT_SHOW_PARAGRAPH = ("Only the entries that answer the question belong in it — the ones the "
                        "asker would act on, usually one to three; every one of a kind for a "
                        "which-of-a-kind question, the whole chain for a connection. Not what "
                        "you read on the way, not every name you mentioned, not a name from a "
                        "quote.")

TIGHT_SYSTEM_SENTENCE = ("Name in your answer only entries a tool returned — found, listed, "
                         "read, or joined to something you read; say plainly when you did not "
                         "find something, rather than guessing a name.")
# How many entries a tight answer may light. Six is three times what a good answer wants
# and a third of LIT; past it, the ids the prose names are kept and the rest are cut.
LIT_TIGHT = 6


def _tight(schemas: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Those schemas, copied, with show told to light only what answers the question.

    The terse set gets the terse wording, told by show's description being TERSE's; any
    other show -- TOOLS', RICH's, a caller's -- gets the full one, example and all.
    """
    brief = _schema("show", TERSE)["function"]["description"]
    out = copy.deepcopy(list(schemas))
    for schema in out:
        if schema["function"]["name"] == "show":
            was = schema["function"]["description"]
            schema["function"]["description"] = TIGHT_SHOW_TERSE if was == brief else TIGHT_SHOW
    return out


# ------------------------------------------------------------------ all the lookups at once
#
# The asking that costs the least, for the model that answers best. Measured 2026-09-02 on
# Qwen3.8-Flash-Next: 25 seconds a question over about seven tool calls, and a tool call is
# a whole round trip through the slow half of a model that reads eleven times faster than it
# writes. The calls were not seven different questions -- they were one question asked one
# entry at a time, because nothing in the prompt said the ids are a list. Every one of these
# tools already takes several; this is the asking that says so, in the system text, in the
# descriptions, and in a nudge the moment a turn reads one entry when more were found.
#
# Said on copies, for the same reason as RICH and TIGHT: with the flag off the descriptions
# and the system prompt stay byte for byte as they were, so a run before this one is still
# comparable with a run after it.
BATCH_SYSTEM_SENTENCE = (
    "Ask for everything you want in one call. look_up takes a list of words, and look_at "
    "and look_around take a list of ids -- reading three entries in one call costs one turn "
    "where three calls cost three, and a question that reads one entry at a time runs out of "
    "turns before it answers. Name every entry you mean to read in a single call, and never "
    "read them one at a time.")
# What a turn is told when it read one entry and left the rest of what it found unread.
BATCH_NUDGE = ("You read one entry, and more were found. Read the rest in one call: put "
               "every id you still want into a single look_at or look_around, not one call "
               "each.")
# The worked example each searching tool gains: three things in one call, written out, since
# a description that says a parameter is a list leaves a small model to infer that it should
# fill one, and one that shows three ids in it does not.
BATCH_EXAMPLES = {
    "look_up": " Three words is still one call: {\"texts\": [\"pottery\", \"firing\", "
               "\"Ambleford\"]}, never three calls of one word.",
    "look_at": " ids is a list, so read everything you mean to write about at once: "
               "{\"ids\": [\"person:wren\", \"person:hollis\", \"topic:ceramics\"]} is one "
               "call for three entries. Reading them one at a time costs three turns and "
               "tells you nothing more.",
    "look_around": " ids is a list here too: {\"ids\": [\"topic:ceramics\", "
                   "\"place:ambleford\", \"org:tinsley\"]} reads three neighbourhoods in "
                   "one call.",
}


def _batched(schemas: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Those schemas, copied, with each searching tool shown a three-entry call."""
    out = copy.deepcopy(list(schemas))
    for schema in out:
        more = BATCH_EXAMPLES.get(str(schema["function"]["name"]))
        if more:
            schema["function"]["description"] += more
    return out


# ------------------------------------------------------------------ one entry at a time
#
# The opposite of `batch`, and it is here for the opposite model. A fat tool result is a
# long thing to hold in mind, and a small model handed a dozen entries in one message
# answers about the last one it read or about none of them -- the thread goes, and what
# comes back is fluent and about nothing. So this asks for the other trade: one entry per
# read, more turns, each result short enough to still be in view when the answer is
# written.
#
# Which of the two a model wants is not a matter of taste and is not one answer for every
# model. It is a number in a store: `ml-stack-bench --also single` measures it against
# `--also batch` on one load, and the profile keeps whichever won.
#
# Said on copies, for the same reason as RICH, TIGHT and BATCH: with the flag off the
# descriptions and the system prompt stay byte for byte as they were.
SINGLE_SYSTEM_SENTENCE = (
    "Read one entry at a time. Put a single id in look_at and a single id in look_around, "
    "then call again for the next one; look one word up at a time in the same way. You "
    "have turns to spend, and a short result you can still see when you write the answer "
    "is worth more than a long one you had to skim.")
# What a turn is told when it read several entries in one call.
SINGLE_NUDGE = ("You read several entries in one call. Read them one at a time instead: put "
                "a single id in the next look_at or look_around, and call it again for the "
                "next one.")
# The worked example each searching tool gains -- one thing in the list, written out, for
# the same reason BATCH_EXAMPLES writes out three.
SINGLE_EXAMPLES = {
    "look_up": " One word to a call: {\"texts\": [\"pottery\"]}, then another call for "
               "\"firing\". Two words in one call give you two sets of hits to hold at once.",
    "look_at": " One id to a call: {\"ids\": [\"person:wren\"]}, then another call for "
               "{\"ids\": [\"person:hollis\"]}. What comes back is then short enough to "
               "still be in front of you when you write the answer.",
    "look_around": " One id to a call here too: {\"ids\": [\"topic:ceramics\"]}, then "
                   "another call for the next neighbourhood.",
}


def _singled(schemas: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Those schemas, copied, with each searching tool shown a one-entry call."""
    out = copy.deepcopy(list(schemas))
    for schema in out:
        more = SINGLE_EXAMPLES.get(str(schema["function"]["name"]))
        if more:
            schema["function"]["description"] += more
    return out


def _all_at_once(reply: Any) -> bool:
    """Whether that reply read more than one entry in a single reading call.

    The shape `single` exists to stop, and the mirror of `_one_by_one`: one `look_at`
    carrying three ids is the fat result the thread gets lost in, however few calls the
    turn made.
    """
    for call in (getattr(reply, "tool_calls", None) or []):
        fn = call.get("function") or {}
        if str(fn.get("name") or "") not in ("look_at", "look_around"):
            continue
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except ValueError:
            continue
        if isinstance(args, Mapping) and len(list(args.get("ids") or ())) > 1:
            return True
    return False


# ------------------------------------------------------------------ three tools, not eight
#
# Every schema in the offer is a decision the model makes before it reaches the one that
# matters, and for some models the choosing itself is what goes wrong: eight descriptions
# in, it picks `list_kind` for a question about two people. `few` offers three -- look_up,
# look_at, show -- and takes away every other way of looking.
#
# Nothing is renamed and nothing is faked. There is no path tool and no listing tool in
# this offer, so what look_up's description gains is how to answer those questions with
# what is actually there: look both ends up, read them, and read on to whatever they are
# joined to. Telling the model to "ask for a path as 'path A to B'" would be a tool that
# does not exist, and a model that believed it would spend every turn it has on a call
# nothing answers.
FEW_TOOLS = ("look_up", "look_at", "show")
FEW_SENTENCE = (
    "This offer holds three tools and no others -- look_up, look_at and show -- so nothing "
    "here traces a path or lists a kind for you. A question about how two entries are "
    "connected is answered by looking both of them up, reading them, and reading on to "
    "whatever they are joined to; a question about which entries of a kind there are is "
    "answered by looking the kind's own words up and reading what comes back.")
# The clause of SYSTEM that names a tool this offer does not have, and what it becomes.
# Swapped rather than argued with, exactly as tight swaps the show paragraph.
PATH_CLAUSE = ("and when the question is about how two things relate, trace the path "
               "between them.")
FEW_PATH_CLAUSE = ("and when the question is about how two things relate, read both ends "
                   "and follow what each of them is joined to.")
FEW_SYSTEM_SENTENCE = (
    "You have three tools and no others: look_up finds entries by name, look_at reads what "
    "is held on them and what they are joined to, and show says what your answer is about. "
    "There is no path tool and no listing tool here; read your way to both of those.")


def _few(schemas: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Those schemas, copied, with every way of looking but look_up and look_at taken away.

    A tool that does not search is kept, whoever added it: this is about what the choosing
    costs, and `show` -- or a caller's change request -- is not a choice between ways to
    look. `look_up` is told what the offer no longer has, and how to answer those questions
    with what it does.
    """
    out: list[dict[str, Any]] = []
    for schema in copy.deepcopy(list(schemas)):
        name = str((schema.get("function") or {}).get("name") or "")
        if name in SEARCHING and name not in FEW_TOOLS:
            continue
        if name == "look_up":
            schema["function"]["description"] += " " + FEW_SENTENCE
        out.append(schema)
    return out


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


# ------------------------------------------------------------------ the graph at a glance
#
# The broad question -- "what is this community about?" -- has no name in it to look up, so
# a search answers it by finding whatever the words happen to hit, and the answer is about
# five arbitrary entries. What it wants is the shape of the whole graph, and that is
# arithmetic over the nodes and edges: no model, no round trip, the same text every time.
SUMMARY_TOP = 10          # entries read out per kind, most mentioned first
SUMMARY_RELATIONS = 8     # how many relations are named as the busiest
SUMMARY_SAID = 140        # characters of one entry's own words


def summarise(graph: Mapping[str, Any], *, top: int = SUMMARY_TOP,
              relations: int = SUMMARY_RELATIONS) -> str:
    """What the whole graph holds, in one text: counts per kind, the most-mentioned
    entries of each kind with a line of their own words, and the busiest relations.

    Computed from the graph and nothing else, so it costs no model call and reads the same
    way twice. Ids are in brackets, as `look_around` writes them, so a model may select an
    entry the summary named without looking it up.
    """
    nodes = list(graph.get("nodes") or ())
    messages = graph.get("messages") or {}
    by_kind: dict[str, list[Mapping[str, Any]]] = {}
    for node in nodes:
        by_kind.setdefault(_kind_of(node) or "entry", []).append(node)
    order = sorted(by_kind, key=lambda k: (-len(by_kind[k]), k))
    edges = list(graph.get("edges") or ())
    counted = ", ".join(f"{len(by_kind[k])} {k}" for k in order)
    lines = [f"This graph holds {len(nodes)} entries and {len(edges)} joins"
             + (f": {counted}." if counted else ".")]
    for kind in order:
        rows = sorted(by_kind[kind], key=lambda n: (-int(n.get("mentions") or 0),
                                                    str(n.get("label") or "")))
        lines.append(f"{kind} ({len(rows)}), most mentioned first:")
        for node in rows[:top]:
            attrs = node.get("attrs") or {}
            line = f"  - {node.get('label')} [{node['id']}]"
            for key in ("role", "type", "location"):
                if str(attrs.get(key) or "").strip():
                    line += f", {attrs[key]}"
            said = _said_lines(node, messages, most=1, chars=SUMMARY_SAID, indent="")
            if said:
                line += " " + said[0]
            lines.append(line)
        if len(rows) > top:
            lines.append(f"  ... and {len(rows) - top} more {kind}")
    busiest: dict[str, int] = {}
    for edge in edges:
        rel = str(edge.get("rel") or "joined to")
        busiest[rel] = busiest.get(rel, 0) + 1
    if busiest:
        best = sorted(busiest.items(), key=lambda kv: (-kv[1], kv[0]))[:relations]
        lines.append("Busiest joins: " + ", ".join(f"{rel} ({n})" for rel, n in best) + ".")
    return "\n".join(lines)


SUMMARY_SCHEMA: dict[str, Any] = {"type": "function", "function": {
    "name": "summarise",
    "description": "Read the whole graph at a glance: how many entries of each kind there "
                   "are, the most-mentioned entries of each kind with a line of their own "
                   "words and their ids in brackets, and which relations are busiest. Call "
                   "it for a broad question -- what this community is about, what it does, "
                   "what the main subjects are -- where no particular name is being looked "
                   "for and a search would only find whatever the words happened to hit. "
                   "One call answers it; then select the entries your answer is about. "
                   "Example: for \"what does this group do?\" call summarise with {}.",
    "parameters": {"type": "object", "properties": {}, "required": []}}}
# Its questions, for a router. Kept out of `TOOL_PROMPTS` because that mapping is also read
# as "every tool the model has" -- `ml-stack-train-tools` refuses a prompts key naming a
# tool the schemas lack, and `summarise` is offered only when a caller asks for it.
SUMMARY_PROMPTS: tuple[str, ...] = (
    "what does this community do?",
    "what is this group about?",
    "what are the main subjects here?",
    "give me an overview of the whole thing",
    "summarise this graph for me",
)


# What a *question* looks like when it wants each tool -- for an embedder to match against,
# and never sent to the chat model.
#
# Two consumers want two different texts. A chat model wants to know what a tool does, in
# prose: "search the graph for entries whose name or attached words match some text". An
# embedder wants the opposite. Comparing a user's question to prose *about a capability* is
# comparing unlike things, and embeddinggemma is measurably poor at it -- the same asymmetry
# that made DOCUMENT and QUERY prefixes necessary in `graph.vectors`, where telling it which
# side was which moved a robotics technician from unplaced to second.
#
# So these are questions, not descriptions, because a question against questions is
# like-to-like and that is where the signal is. Putting them in the description instead
# would serve neither: it lengthens the text the chat model reads, which already measured
# too long for a large model, to help something that is not reading it.
# The name for "this question wants no tool at all". Not a tool, so it can never be called;
# it exists so a router has somewhere to put a greeting other than the nearest search.
CHAT = "chat"

TOOL_PROMPTS: dict[str, tuple[str, ...]] = {
    "look_up": (
        "who knows about robotics?",
        "is there anyone here who does marketing?",
        "find me people working on healthcare",
        "somebody who can sell things",
        "who fixes machines?",
    ),
    # "Which companies are here?" is not a search: nothing is labelled company, so look_up
    # finds nothing however it is worded. These are the questions that ask what kinds of
    # thing the graph holds, kept apart from look_up's, which ask for a particular one.
    "list_kind": (
        "which companies are here?",
        "what kinds of organisations are represented?",
        "list all the topics people talk about",
        "what places do people live in?",
        "which employers are represented here?",
    ),
    "look_at": (
        "tell me about Iris Bellweather",
        "what is she good at?",
        "what does that company do?",
        "what has this person actually said?",
        "more detail on those two",
    ),
    # Not look_at's: "tell me about her" wants one entry read properly, and these want
    # everyone standing next to something. Routed to look_at, a staffing question spends a
    # call per neighbour and finds the neighbours by guessing their spellings first.
    "look_around": (
        "who is around Otto Vance?",
        "everyone joined to that topic",
        "who could help with a cracked kiln?",
        "which people does that place hold?",
        "what is attached to the foundry?",
    ),
    "path_between": (
        "how are these two connected?",
        "who could introduce me to a lawyer?",
        "what links the foundry and the survey firm?",
        "is there anyone in common between them?",
        "how do I reach that person?",
    ),
    "show": (
        "highlight those on the graph",
        "show me who you mean",
        "light up the people in that answer",
    ),
    # Not a tool: the questions that want *no* graph at all. Without somewhere for these to
    # go, a greeting is matched against four search tools and wins one of them -- "hi"
    # scored 0.900 against "highlight them on the graph", because everything is close to
    # everything in embedding space and the question is only ever "close to what".
    #
    # A greeting is not a failed search, it is a different kind of message, and answering it
    # costs one turn instead of six.
    CHAT: (
        "hi",
        "hello there",
        "thanks, that is helpful",
        "tell me a joke",
        "what can you do?",
        "how does this work?",
        "who are you?",
        "what is the capital of France?",
        "write me a haiku about rain",
        "never mind",
    ),
}


def prompts_for(name: str) -> tuple[str, ...]:
    """Example questions that should route to ``name``, for an embedder. May be empty."""
    if name == "summarise":
        return SUMMARY_PROMPTS
    return TOOL_PROMPTS.get(name, ())


def routing_prompts(*, summary: bool = False) -> dict[str, tuple[str, ...]]:
    """What to route a question against: `TOOL_PROMPTS`, and `summarise`'s when it is offered.

        routed = rank(question, routing_prompts(summary=True), base_url=..., model=...)

    A tool the model was not given must not be routed to, so the summary's questions are
    added only when a caller asked for the tool -- otherwise a broad question would be sent
    to something that is not on the table.
    """
    return {**TOOL_PROMPTS, "summarise": SUMMARY_PROMPTS} if summary else dict(TOOL_PROMPTS)


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


def look_up(graph: Mapping[str, Any], text: str, *, limit: int = FOUND,
            rich: bool = False) -> list[dict[str, Any]]:
    """Entries whose name, attributes or own words carry that text, best match first.

    Characters only. For a search that also stems and also knows what a word means, pass
    ``finder=`` to converse — see ``ml_stack.graph.search.hybrid``. With ``rich``, each hit
    also carries ``"score"`` (the rank it was found at) and ``"matched"`` (what found it:
    ``label``, ``attribute`` or ``said``), so an exact name is told apart from one word in
    one quote. Off, a hit is ``{"id", "label", "kind"}`` and nothing else.
    """
    want = " ".join((text or "").split()).casefold()
    if not want:
        return []
    messages = graph.get("messages") or {}
    scored: list[tuple[int, int, Mapping[str, Any]]] = []
    for node in graph.get("nodes") or ():
        label = str(node.get("label") or "").casefold()
        attrs = node.get("attrs") or {}
        if label == want:
            score = 4
        elif want in label:
            score = 3
        elif any(want in str(v).casefold() for v in attrs.values()):
            score = 2
        elif any(want in str((messages.get(mid) or {}).get("text") or "").casefold()
                 for mid in (node.get("messages") or ())[:20]):
            score = 1
        else:
            continue
        scored.append((score, int(node.get("mentions") or 0), node))
    scored.sort(key=lambda row: (-row[0], -row[1], str(row[2].get("label") or "")))
    rows: list[dict[str, Any]] = []
    for score, _, n in scored[:limit]:
        row: dict[str, Any] = {"id": str(n["id"]), "label": str(n.get("label") or ""),
                               "kind": str(n.get("kind") or "")}
        if rich:
            row["score"] = score
            row["matched"] = [MATCHED_BY_RANK[score]]
        rows.append(row)
    return rows


def joined_people(graph: Mapping[str, Any], node_id: str, *,
                  limit: int = JOINED_HITS) -> list[dict[str, str]]:
    """The people joined to an entry by any edge, in either direction, most mentioned first.

    For the staffing question: a topic found is halfway to who has it, and reading the
    people off the topic in the same call is what saves the rounds spent guessing their
    spellings. A person is whatever the graph calls ``person``, by kind or by id prefix.
    """
    by_id = {str(n["id"]): n for n in (graph.get("nodes") or ())}
    people: dict[str, Mapping[str, Any]] = {}
    for edge in graph.get("edges") or ():
        source, target = str(edge.get("source") or ""), str(edge.get("target") or "")
        other = target if source == node_id else source if target == node_id else ""
        if not other or other == node_id:
            continue
        node = by_id.get(other)
        if node is not None and _kind_of(node) == "person":
            people[other] = node
    rows = sorted(people.values(), key=lambda n: (-int(n.get("mentions") or 0),
                                                  str(n.get("label") or ""), str(n["id"])))
    return [{"id": str(n["id"]), "label": str(n.get("label") or "")} for n in rows[:limit]]


def _enriched(graph: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """look_up's hits with, on each that is not a person, the people joined to it.

    A finder of the caller's own may already say why a hit matched (``score``,
    ``matched``); what it did not say is left out rather than guessed, and ``joined`` is
    read from the graph either way. The caller's rows are copied, never written to.
    """
    by_id = {str(n["id"]): n for n in (graph.get("nodes") or ())}
    out: list[dict[str, Any]] = []
    for r in rows:
        row = dict(r)
        node = by_id.get(str(row.get("id") or ""))
        kind = str(row.get("kind") or "") or (_kind_of(node) if node is not None else "")
        if kind != "person":
            row["joined"] = joined_people(graph, str(row.get("id") or ""))
        out.append(row)
    return out


def _packed(blocks: Sequence[tuple[int, str, str]], budget: int | None) -> str:
    """Those blocks as one text, and how much of them a ``budget`` of tokens will carry.

    Without a budget every block goes, in the order it was built, which is what this always
    did. With one, and only when they do not all fit, the most-mentioned go first and each
    goes *whole* -- its quotes with it -- rather than every entry going with its words cut
    out. The quote is the evidence; four entries entire answer a question that twelve
    clipped ones do not.

    Packing rather than stopping at the first block that will not fit: a small entry after a
    large one still fits, and the budget is there to be spent. Ties are broken by id, so the
    same graph and the same budget read out the same way twice.
    """
    if budget is None:
        return "\n".join(text for _mentions, _node_id, text in blocks)
    whole = "\n".join(text for _mentions, _node_id, text in blocks)
    if not whole or estimate_tokens(whole) <= budget:
        return whole
    kept: list[str] = []
    left = int(budget)
    for mentions, node_id, text in sorted(blocks, key=lambda b: (-b[0], b[1])):
        cost = estimate_tokens(text) + 1
        if kept and cost > left:
            continue
        left -= cost
        kept.append(text)
    return "\n".join(kept)


def _said_lines(node: Mapping[str, Any], messages: Mapping[str, Any], *, most: int,
                chars: int = SAID_CHARS, indent: str = "    ") -> list[str]:
    """What an entry said, at most ``most`` of them, each on one line."""
    out: list[str] = []
    for mid in (node.get("messages") or ())[:most]:
        text = str((messages.get(mid) or {}).get("text") or "")
        if text:
            out.append(f'{indent}said: "{" ".join(text.split())[:chars]}"')
    return out


def look_at(graph: Mapping[str, Any], ids: Sequence[str], *,
            budget: int | None = None) -> str:
    """What the graph holds on those entries, as text a model can answer from.

    ``budget`` is a conversation's `reach`: a ceiling in tokens on what one tool result may
    carry. Without one nothing is packed and nothing is dropped -- what a caller asked for
    is what it gets. See `_packed`.
    """
    by_id = {str(n["id"]): n for n in (graph.get("nodes") or ())}
    messages = graph.get("messages") or {}
    blocks: list[tuple[int, str, str]] = []
    for node_id in ids:
        node = by_id.get(str(node_id))
        if node is None:
            continue
        attrs = node.get("attrs") or {}
        joined = [f"{e.get('rel', 'joined to')} {by_id[e['target']].get('label')}"
                  for e in (graph.get("edges") or ())
                  if e.get("source") == node_id and e.get("target") in by_id]
        joined += [f"{by_id[e['source']].get('label')} {e.get('rel', 'joined to')} this"
                   for e in (graph.get("edges") or ())
                   if e.get("target") == node_id and e.get("source") in by_id]
        line = f"- {node.get('label')} ({attrs.get('type') or node.get('kind') or 'entry'})"
        for key in ("role", "location"):
            if attrs.get(key):
                line += f", {attrs[key]}"
        if joined:
            line += ": " + "; ".join(joined[:JOINED])
        lines = [line, *_said_lines(node, messages, most=SAID)]
        blocks.append((int(node.get("mentions") or 0), str(node_id), "\n".join(lines)))
    return _packed(blocks, budget)


def _edges_around(graph: Mapping[str, Any], node_id: str) -> list[tuple[str, str, float]]:
    """Everything joined to ``node_id``: ``(other id, how it reads, weight)``, best first.

    Both directions, and the direction is kept in the reading -- ``-> works_at`` against
    ``<- employs`` -- because "Wren employs the foundry" and "the foundry employs Wren" are
    not the same fact and a model given the wrong one writes it down.

    Heaviest edge first, then by id: a neighbourhood that reads out in a different order
    each time cannot have two answers about it compared with each other.
    """
    best: dict[str, tuple[float, str]] = {}
    for edge in graph.get("edges") or ():
        source, target = str(edge.get("source") or ""), str(edge.get("target") or "")
        if source == node_id:
            other, reads = target, f"-> {edge.get('rel') or 'joined to'}"
        elif target == node_id:
            other, reads = source, f"<- {edge.get('rel') or 'joined to'}"
        else:
            continue
        if not other or other == node_id:
            continue
        weight = float(edge.get("weight") or 0)
        if other not in best or weight > best[other][0]:
            best[other] = (weight, reads)
    return sorted(((one, reads, weight) for one, (weight, reads) in best.items()),
                  key=lambda row: (-row[2], row[0]))


def look_around(graph: Mapping[str, Any], ids: Sequence[str], *, hops: int = 1,
                joined: int = AROUND, entries: int = AROUND_ENTRIES,
                budget: int | None = None) -> str:
    """The neighbourhood of those entries, read out in one call.

    Each entry as `look_at` gives it -- label, kind, what is held on it, what it said -- and
    then, indented under it, everything joined to it: the relation and its direction, the
    other entry's label, its id in brackets, its kind, and one line of its own words. The id
    is the point of the brackets: a model may write about a neighbour and select it without
    ever having looked it up, which is the four calls this replaces.

    ``hops`` of 2 makes each neighbour an entry in its own right and reads its neighbourhood
    too; ``entries`` caps how many entries are read out in all, whatever the hops.

    For a model whose context is cheap and whose reading is fast, this is the shape that
    suits it: one fat result instead of five thin ones, since a tool call costs a round trip
    through the slow half of the model and reading the answer back costs the fast half.
    ``budget`` -- a conversation's `reach` -- is what keeps that from being unbounded.
    """
    by_id = {str(n["id"]): n for n in (graph.get("nodes") or ())}
    messages = graph.get("messages") or {}
    wanted: list[str] = []
    for one in ids:
        if str(one) in by_id and str(one) not in wanted:
            wanted.append(str(one))
    blocks: list[tuple[int, str, str]] = []
    seen = set(wanted)
    frontier = wanted
    for _ in range(max(1, int(hops or 1))):
        after: list[str] = []
        for node_id in frontier:
            if len(blocks) >= entries:
                break
            node = by_id[node_id]
            attrs = node.get("attrs") or {}
            head = (f"- {node.get('label')} [{node_id}] "
                    f"({attrs.get('type') or node.get('kind') or 'entry'})")
            for key in sorted(attrs):
                if key != "type" and str(attrs[key]).strip():
                    head += f", {key}: {attrs[key]}"
            lines = [head, *_said_lines(node, messages, most=SAID)]
            for other, reads, _weight in _edges_around(graph, node_id)[:joined]:
                near = by_id.get(other)
                if near is None:
                    continue
                kind = str((near.get("attrs") or {}).get("type")
                           or near.get("kind") or "entry")
                lines.append(f"    {reads} {near.get('label')} [{other}] ({kind})")
                lines += _said_lines(near, messages, most=1, indent="        ")
                if other not in seen:
                    seen.add(other)
                    after.append(other)
            blocks.append((int(node.get("mentions") or 0), node_id, "\n".join(lines)))
        frontier = after
    return _packed(blocks, budget)


def path_between(graph: Mapping[str, Any], start: str, goal: str) -> dict[str, Any]:
    """The chain of entries from one to another, and how it reads."""
    from ml_stack.entities.paths import between, shortest_path

    edges = list(graph.get("edges") or ())
    ids = between(edges, start, goal)
    if not ids:
        return {"path": [], "why": "nothing in the graph joins those two"}
    label = {str(n["id"]): str(n.get("label") or "") for n in (graph.get("nodes") or ())}
    steps = shortest_path(edges, start, goal)
    return {"path": ids, "rels": [e.get("rel", "") for e in steps],
            "reads": " → ".join(label.get(i, i) for i in ids)}


def _kind_of(node: Mapping[str, Any]) -> str:
    """A node's kind: what it says, else the ``kind:`` prefix of its id, else nothing."""
    kind = str(node.get("kind") or "")
    if kind:
        return kind
    head, sep, _rest = str(node.get("id") or "").partition(":")
    return head if sep else ""


def _singular(word: str) -> tuple[str, ...]:
    """The kind names a word might be asking for: itself, and what it is the plural of."""
    ways = [word]
    if word.endswith("ies"):
        ways.append(word[:-3] + "y")
    if word.endswith("es"):
        ways.append(word[:-2])
    if word.endswith("s"):
        ways.append(word[:-1])
    return tuple(ways)


def list_kind(graph: Mapping[str, Any], kind: str, *, limit: int = LISTED,
              budget: int | None = None) -> dict[str, Any]:
    """Every entry of one kind, most mentioned first — or the kinds there are.

    For the question a search cannot reach: "which companies do people here work for?" is
    answered by every ``org`` in the graph, and no word finds those, because nothing is
    labelled "company". ``kind`` is matched without regard to case or number, so ``Orgs``
    lists ``org``. A kind the graph does not have comes back as ``{"none": ..., "kinds":
    {name: count}}``, so a model that guessed wrong learns the real names on its first miss
    rather than guessing again.

    ``budget`` -- a conversation's `reach` -- replaces ``limit`` rather than joining it: a
    model that reads cheaply wants every organisation there is, and 40 was only ever a
    guess at what a result may cost. With one, as many entries as that many tokens will
    carry go, most mentioned first; ``total`` still says how many there were.
    """
    nodes = [n for n in graph.get("nodes") or () if not (n.get("attrs") or {}).get("hidden")]
    counts: dict[str, int] = {}
    for node in nodes:
        named = _kind_of(node)
        if named:
            counts[named] = counts.get(named, 0) + 1
    by_fold = {k.casefold(): k for k in counts}
    want = " ".join(str(kind or "").split()).casefold()
    found = next((by_fold[w] for w in _singular(want) if w in by_fold), None)
    if found is None:
        kinds = dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))
        return {"none": f"no kind {str(kind or '')!r}", "kinds": kinds}
    rows = [n for n in nodes if _kind_of(n) == found]
    rows.sort(key=lambda n: (-int(n.get("mentions") or 0), str(n.get("label") or "")))
    entries = [{"id": str(n["id"]), "label": str(n.get("label") or ""),
                "mentions": int(n.get("mentions") or 0)}
               for n in (rows if budget is not None else rows[:limit])]
    if budget is not None:
        entries = _within(entries, budget)
    return {"kind": found, "total": len(rows), "entries": entries}


def _within(entries: Sequence[Mapping[str, Any]], budget: int) -> list[dict[str, Any]]:
    """As many of those entries as ``budget`` tokens will carry, in the order given.

    They arrive most-mentioned first, so this cuts the tail rather than choosing: a listing
    read out of order is not a listing. One always goes, however long it is, because an
    empty result reads to a model as "there are none".
    """
    kept: list[dict[str, Any]] = []
    left = int(budget)
    for row in entries:
        left -= estimate_tokens(json.dumps(row, ensure_ascii=False)) + 1
        if kept and left < 0:
            break
        kept.append(dict(row))
    return kept


def _cut(text: str, reach: int | None) -> str:
    """One tool result, cut to what the conversation will carry it in.

    Without a ``reach`` this is the flat `CUT` characters a tool message has always been
    trimmed to -- byte for byte what it was, so nothing measured before moves. With one it
    is that many tokens, found by halving rather than by a characters-per-token guess,
    because the guess is wrong in both directions on the text a graph returns: ids and
    punctuation cost more than prose, and prose costs less.
    """
    if not reach:
        return text[:CUT]
    if estimate_tokens(text) <= reach:
        return text
    low, high = 0, len(text)
    while low < high:
        mid = (low + high + 1) // 2
        if estimate_tokens(text[:mid]) <= reach:
            low = mid
        else:
            high = mid - 1
    return text[:low]


_BRACKETED = re.compile(r"\[([^\[\]\s]{1,120})\]")


def _ids_in(text: str) -> list[str]:
    """The bracketed ids a `look_around` result read out, in the order it read them.

    Read back off the text rather than recomputed, so what the answer counts as found is
    exactly what the model was shown -- a neighbourhood the budget cut short did not happen.
    Anything that is not an id in this graph is dropped by the caller.
    """
    out: list[str] = []
    for one in _BRACKETED.findall(text or ""):
        if one not in out:
            out.append(one)
    return out


def _schema(name: str, among: Sequence[Mapping[str, Any]] = TOOLS) -> dict[str, Any]:
    """The tool schema called ``name``.

    By name, never by position: the order of ``TOOLS`` is the order a model reads them in,
    which changes when a tool is added, and ``TOOLS[3]`` was ``show`` right up until it
    was not.
    """
    for schema in among:
        if (schema.get("function") or {}).get("name") == name:
            return schema
    raise KeyError(name)


def tools_for(graph: Mapping[str, Any], *, finder: Any = None,
              terse: bool = False, rich: bool = False,
              tight: bool = True, reach: int | None = None,
              batch: bool = False, single: bool = False, few: bool = False,
              summary: bool = False) -> list[tuple[dict[str, Any], Any]]:
    """The built-in tools over that graph, as ``(schema, callable)`` pairs.

    Each callable takes the parsed arguments mapping. ``finder`` replaces how look_up looks:
    it takes the text and returns ``[{"id", "label", "kind"}, ...]``.

    ``rich`` is a retrieval change still being measured, so it is off unless asked for:
    each look_up hit then says why it matched (``score``, ``matched``) and, when it is not a
    person, who is joined to it (``joined``), and the look_up description says so. Off,
    nothing observable changes -- the same descriptions byte for byte, the same result
    shape -- so a sweep of the current behaviour stays comparable with one before it.

    ``tight`` is how these are asked for -- on by default: show's description, on a copy,
    says to light only what answers the question, see `TIGHT_SHOW`. The rest of what tight
    does (the nudge, the system sentence, the cap on what is lit) is `converse`'s.
    ``tight=False`` is the loose asking kept as a control: the sets themselves, unchanged.

    ``reach`` is how much one tool result may carry, in tokens: `look_at`, `look_around`
    and `list_kind` pack up to it instead of stopping at a fixed number of entries. Without
    it they behave exactly as they did, which is what a model with a small window and an
    expensive cache wants; with it a model whose context is cheap takes more per call and
    makes fewer calls. See `converse`.

    ``batch`` shows each searching tool a three-entry call, on a copy -- the rest of what
    it does is `converse`'s. ``single`` is its opposite and shows each of them a one-entry
    call, for the model that loses the thread of a long result. ``few`` takes away every
    way of looking but `look_up` and `look_at`, and tells look_up what the offer no longer
    has, for the model whose tool choice degrades with the number of schemas; a tool that
    does not search is kept whoever added it. ``summary`` adds the `summarise` tool: the
    whole graph at a glance, computed without a model, for the broad question no search
    reaches -- and `few`, offering three, takes it away again.
    """
    def find(args: Mapping[str, Any]) -> Any:
        # one word or several: a staffing question needs a lookup per skill, and doing them
        # one round at a time is what spent every turn a question had
        wanted = [str(x) for x in (args.get("texts") or ()) if str(x).strip()]
        if not wanted and str(args.get("text") or "").strip():
            wanted = [str(args["text"])]
        rows, seen = [], set()
        for text in wanted:
            for r in (finder(text) if finder is not None else look_up(graph, text, rich=rich)):
                if r["id"] not in seen:
                    seen.add(r["id"])
                    rows.append(r)
        if rich:
            rows = _enriched(graph, rows)
        # an empty list reads as "try again"; saying nothing matched reads as "move on"
        return rows or {"none": f"Nothing in the graph matches {', '.join(map(repr, wanted))}. "
                                "Try different words, or answer with what you already have."}

    def read(args: Mapping[str, Any]) -> str:
        return look_at(graph, [str(i) for i in (args.get("ids") or ())], budget=reach)

    def around(args: Mapping[str, Any]) -> str:
        try:
            hops = int(args.get("hops") or 1)
        except (TypeError, ValueError):
            hops = 1                  # a model that wrote "one" meant one, not an error
        return look_around(graph, [str(i) for i in (args.get("ids") or ())],
                           hops=max(1, min(hops, 3)), budget=reach)

    def trace(args: Mapping[str, Any]) -> dict[str, Any]:
        return path_between(graph, str(args.get("from_id") or ""), str(args.get("to_id") or ""))

    def listing(args: Mapping[str, Any]) -> dict[str, Any]:
        return list_kind(graph, str(args.get("kind") or ""), budget=reach)

    def light(args: Mapping[str, Any]) -> str:
        # the ids are the whole result; the model is told they arrived so it stops calling it
        return f"selected {len(list(args.get('ids') or ()))} on the graph"

    def glance(args: Mapping[str, Any]) -> str:
        return summarise(graph)

    does = {"look_up": find, "look_at": read, "look_around": around, "path_between": trace,
            "list_kind": listing, "show": light, "summarise": glance}
    if rich:
        schemas = RICH_TERSE if terse else RICH
    else:
        schemas = TERSE if terse else TOOLS
    if summary:
        schemas = [*schemas, SUMMARY_SCHEMA]
    if batch:
        schemas = _batched(schemas)
    if single:
        schemas = _singled(schemas)
    if few:
        # after the worked calls and before tight: what is taken away is taken away
        # whatever was written on it, and tight is about `show`, which stays
        schemas = _few(schemas)
    if tight:
        schemas = _tight(schemas)
    # in the order the schemas are written, which is the order a model reads them in
    return [(schema, does[str(schema["function"]["name"])]) for schema in schemas]


# What each way of asking is when nobody says: a profile fills in only what is still this.
_UNSAID = {"rich": False, "tight": True, "reach": None, "kinds": False, "batch": False,
           "summary_tool": False, "single": False, "few": False, "rounds": ROUNDS}


def _under(profile: Any, given: dict[str, Any]) -> dict[str, Any]:
    """``given``, with a measured profile's ways filling in whatever nobody said.

    A keyword still at its default takes the record's; anything said outright is kept, so a
    caller overruling a measurement on purpose is not overruled back. A value typed out that
    happens to equal the default cannot be told from one not typed at all -- argparse cannot
    either -- and it does not matter: what it asked for is what it got.
    """
    from ml_stack.serve.profile import Profile, profile_for

    found = profile if isinstance(profile, Profile) else profile_for(str(profile))
    if found is None:
        return given
    out = dict(given)
    for name, value in found.asking().items():
        if out.get(name) == _UNSAID.get(name):
            out[name] = value
    return out


def converse(question: str, graph: Mapping[str, Any], client: Any, *,
             turns: Sequence[Mapping[str, str]] = (), system: str = SYSTEM,
             rounds: int = ROUNDS, limit: int = LIT,
             tools: Sequence[tuple[Mapping[str, Any], Any]] | None = None,
             finder: Any = None, held: Sequence[str] = (),
             opening: Sequence[str] = (), rich: bool = False,
             tight: bool = True, reach: int | None = None, summary: Any = None,
             recalled: Sequence[Any] = (), kinds: bool = False, batch: bool = False,
             single: bool = False, few: bool = False,
             summary_tool: bool = False, profile: Any = None) -> Answer:
    """One question, answered with the graph in hand.

    ``client`` is anything with ``chat(messages, tools=...)`` returning a reply carrying
    ``content`` and ``tool_calls`` — ``ml_stack.client.Client`` does. ``tools`` is
    ``[(schema, callable), ...]``, each callable taking the parsed arguments mapping;
    ``tools_for(graph)`` by default. ``finder`` replaces just look_up's callable.
    ``held`` names entries already highlighted for the reader: they are told to the model
    by label and id, and enter ``ids`` only if a tool call touches them. ``rich`` is
    ``tools_for``'s: look_up hits that say why they matched and who is joined to them.

    ``tight`` is the asking, and on by default, because a model that finds nearly
    everything then lights far more than the answer. `show`'s description and the closing
    nudge say to light only what answers the question, the system prompt says to name only
    what a tool returned, ``show`` is capped at `LIT_TIGHT` -- the ids the prose names kept
    first -- and an id the prose names that no tool returned is dropped from it. Each of
    those is a line in ``steps``. ``tight=False`` is the loose asking the ranking runs and
    the answer cache were fingerprinted on, kept as a control: with it, nothing observable
    changes from what this did before, byte for byte.

    ``reach`` is how much one tool result may carry, in tokens, for this whole
    conversation. ``None`` -- the default -- is the flat `CUT` characters a tool message
    has always been trimmed to, and nothing else changes. An int is that many tokens per
    result, and `look_at`, `look_around` and `list_kind` pack up to it: whole entries with
    their quotes, most-mentioned first, rather than every entry with its words clipped.
    Measured 2026-09-02: Qwen3.8-Flash-Next reads a tool result back at about 390 tok/s and
    writes at about 35, so one fat result is far cheaper than the five thin ones it
    replaces; a small model with an expensive cache has the opposite profile and is left
    on the default.

    ``turns`` is the window: the last few turns, chosen by recency, sent whole and in
    order. ``summary`` -- a thread's rolling summary, as a ``Turn``, a mapping with
    ``text`` or ``content``, or a string -- goes first after the system prompt, as one
    message reading "Earlier in this conversation: ...", ahead of anything that changes
    per question so it stays inside the model's cached prefix. ``recalled`` -- earlier
    turns outside the window that match this question, oldest first, the same shapes --
    follow it, each marked as recalled, before the window. With neither, the messages are
    byte for byte what they were.

    Five ways of asking, each off by default and each measured by `ml-stack-bench --also`.
    None of them is a default waiting to be promoted: they exist so that one model can be
    asked the way *it* measured best while another is asked the opposite, and which is
    which lives in a profile (see ``profile`` below), never in a habit:

    ``batch`` is all the lookups in one turn. The system prompt gains a sentence saying the
    ids are a list, each searching tool's description gains a worked three-entry call, and a
    turn that reads one entry while more are still unread is told once to read the rest in
    one call. What it saves is *rounds* -- `Answer.rounds` -- and a round is a round trip
    through the model, which is where the wall clock goes.

    ``kinds`` drops from ``show`` the entries whose kind the question did not ask for: a
    question that asks *who* keeps the people and drops the topic it found them through.
    Only when the question word settles the kind -- see `asked_kinds`, which answers
    ``None`` for a question naming several kinds or none, and then nothing is dropped. A
    listing (`list_kind`) is exempt, as it is from the cap, and a filter that would empty
    the selection is not applied.

    ``summary_tool`` offers `summarise`: the whole graph at a glance -- counts per kind, the
    most-mentioned entries of each kind, the busiest relations -- computed without a model,
    for the broad question ("what is this group about?") that no search reaches.

    ``single`` is `batch` turned around, for the model that loses the thread of a long tool
    result rather than running out of turns: the system prompt says to read one entry at a
    time, each searching tool is shown a one-entry call, and a turn that reads several
    entries at once is told once to read them one at a time. It buys short results and
    spends rounds, which is exactly the trade `batch` makes in the other direction.

    ``few`` offers three tools -- `look_up`, `look_at`, `show` -- and takes away every other
    way of looking, for the model whose tool choice degrades with the number of schemas. No
    tool is faked to cover what went: look_up's description says the offer has no path tool
    and no listing tool, and says how to answer those questions by reading, which is what
    the loop then does. Anything that does not search -- a caller's change request -- is
    kept, because it is not a choice between ways to look.

    ``rounds`` is not a way but is measured beside them: how many tool-calling turns a
    question may spend before it must answer. `few` and `single` both want more of them and
    `batch` wants fewer, which is the whole reason they are measured together.

    ``profile`` is a :class:`~ml_stack.serve.Profile` or a model name, and supplies those
    ways from what that model *measured* best rather than from what a caller remembered:
    Flash-Next wants batch, kinds and summary together and gemma-4 wants none of them, and
    which is which is a number in a store, not a habit. One asking per model is the point --
    a model that answers better on three tools and twenty rounds is asked that way, and the
    one that answers better on eight tools and six is not asked to match it. It fills in
    only what this call left unsaid -- see `_under`.
    """
    if profile is not None:
        said = _under(profile, {"rich": rich, "tight": tight, "reach": reach,
                                "kinds": kinds, "batch": batch,
                                "summary_tool": summary_tool, "single": single,
                                "few": few, "rounds": rounds})
        rich, tight, reach = said["rich"], said["tight"], said["reach"]
        kinds, batch, summary_tool = said["kinds"], said["batch"], said["summary_tool"]
        single, few, rounds = said["single"], said["few"], int(said["rounds"])
    return _converse(question, graph, client, turns=turns, system=system, rounds=rounds,
                     limit=limit, tools=tools, finder=finder, held=held, emit=None,
                     opening=opening, rich=rich, tight=tight, reach=reach, summary=summary,
                     recalled=recalled, kinds=kinds, batch=batch, single=single, few=few,
                     summary_tool=summary_tool)


def converse_stream(question: str, graph: Mapping[str, Any], client: Any, *,
                    on_event: Any,
                    turns: Sequence[Mapping[str, str]] = (), system: str = SYSTEM,
                    rounds: int = ROUNDS, limit: int = LIT,
                    tools: Sequence[tuple[Mapping[str, Any], Any]] | None = None,
                    finder: Any = None, held: Sequence[str] = (),
                    opening: Sequence[str] = (), rich: bool = False,
                    tight: bool = True, reach: int | None = None, summary: Any = None,
                    recalled: Sequence[Any] = (), kinds: bool = False, batch: bool = False,
                    single: bool = False, few: bool = False,
                    summary_tool: bool = False) -> Answer:
    """converse, reporting what is happening to ``on_event`` as it happens.

    ``on_event`` gets one mapping per event: ``{"event": "thinking", "text"}`` as the
    model thinks, ``{"event": "tool", "name", "detail"}`` when it calls a tool,
    ``{"event": "tool_result", "name", "count"}`` with how much came back,
    ``{"event": "answer", "text"}`` as the answer arrives, then ``{"event": "done"}``.
    A client whose ``chat`` takes ``on_delta`` streams the text a piece at a time;
    any other client's text arrives whole.
    """
    return _converse(question, graph, client, turns=turns, system=system, rounds=rounds,
                     limit=limit, tools=tools, finder=finder, held=held, emit=on_event,
                     opening=opening, rich=rich, tight=tight, reach=reach, summary=summary,
                     recalled=recalled, kinds=kinds, batch=batch, single=single, few=few,
                     summary_tool=summary_tool)


def _call_detail(name: str, args: Mapping[str, Any]) -> str:
    if name == "look_up":
        return repr(str(args.get("text") or ""))
    if name in ("look_at", "show"):
        ids = list(args.get("ids") or ())
        return f"{len(ids)} id" + ("" if len(ids) == 1 else "s")
    if name == "look_around":
        ids = list(args.get("ids") or ())
        hops = int(args.get("hops") or 1)
        return (f"{len(ids)} id" + ("" if len(ids) == 1 else "s")
                + ("" if hops == 1 else f", {hops} hops"))
    if name == "path_between":
        return f"{args.get('from_id')} → {args.get('to_id')}"
    if name == "list_kind":
        return repr(str(args.get("kind") or ""))
    return json.dumps(args, ensure_ascii=False)[:80]


def _plain(value: Any) -> str:
    """What the JSON encoder says about a value it cannot encode, instead of raising.

    A tool result is whatever a caller's tool returned, and one that carries raw bytes --
    a picture the `_images` convention did not cover, a path -- must not take the whole
    answer down with a TypeError. The bytes themselves are never sent: a model reads
    nothing from them and they are the size of a picture.
    """
    if isinstance(value, (bytes, bytearray, memoryview)):
        return f"<{len(value)} bytes>"
    return repr(value)[:80]


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


def _one_by_one(reply: Any) -> bool:
    """Whether that reply read exactly one entry, in exactly one reading call.

    The shape `batch` exists to stop: one `look_at` with one id, when a `look_up` has
    already handed over several. Two calls in a turn is already the habit being asked for,
    however few ids each carried.
    """
    reading = [call for call in (getattr(reply, "tool_calls", None) or [])
               if str((call.get("function") or {}).get("name") or "")
               in ("look_at", "look_around")]
    if len(reading) != 1:
        return False
    try:
        args = json.loads((reading[0].get("function") or {}).get("arguments") or "{}")
    except ValueError:
        return False
    return isinstance(args, Mapping) and len(list(args.get("ids") or ())) == 1


def _result_count(result: Any) -> int:
    if isinstance(result, (list, tuple)):
        return len(result)
    if isinstance(result, Mapping) and "entries" in result:
        return len(result.get("entries") or ())
    if isinstance(result, Mapping) and "path" in result:
        return len(result.get("path") or ())
    if isinstance(result, str):
        return len(result.splitlines()) if result.strip() else 0
    return 1 if result else 0


# How the summary and a recalled turn announce themselves. The summary is one user message,
# placed before everything that changes per question; a recalled turn keeps its role and
# says it was recalled, so the model does not read an old answer as the one just given.
EARLIER = "Earlier in this conversation: "
RECALLED = "Recalled from earlier in this conversation: "
SAID_CHARS_BACK = 4000


def _spoken(turn: Any) -> tuple[str, str]:
    """``(role, text)`` of a turn given as a ``Turn``, a mapping, or bare text."""
    if turn is None:
        return "user", ""
    if isinstance(turn, str):
        return "user", turn
    if isinstance(turn, Mapping):
        role = turn.get("role")
        text = turn.get("content") if turn.get("content") is not None else turn.get("text")
    else:
        role = getattr(turn, "role", None)
        text = getattr(turn, "text", None)
    return ("assistant" if role == "assistant" else "user"), str(text or "")


def _converse(question: str, graph: Mapping[str, Any], client: Any, *,
              turns: Sequence[Mapping[str, str]], system: str, rounds: int, limit: int,
              tools: Sequence[tuple[Mapping[str, Any], Any]] | None,
              finder: Any, held: Sequence[str], emit: Any, opening: Sequence[str] = (),
              rich: bool = False, tight: bool = True, reach: int | None = None,
              summary: Any = None, recalled: Sequence[Any] = (), kinds: bool = False,
              batch: bool = False, single: bool = False, few: bool = False,
              summary_tool: bool = False) -> Answer:
    given = tools is not None
    if tools is None:
        tools = tools_for(graph, finder=finder, rich=rich, tight=tight, reach=reach,
                          batch=batch, single=single, few=few, summary=summary_tool)
    elif finder is not None:
        def _found(args: Mapping[str, Any]) -> Any:
            wanted = [str(x) for x in (args.get("texts") or ()) if str(x).strip()]
            if not wanted and str(args.get("text") or "").strip():
                wanted = [str(args["text"])]
            rows, seen = [], set()
            for text in wanted:
                for r in finder(text):
                    if r["id"] not in seen:
                        seen.add(r["id"])
                        rows.append(r)
            return _enriched(graph, rows) if rich else rows

        tools = [(schema, fn) if (schema.get("function") or {}).get("name") != "look_up"
                 else (schema, _found)
                 for schema, fn in tools]
    if batch and given:
        # the same as tight: a caller's own tools are described on a copy, never in place
        told = _batched([schema for schema, _ in tools])
        tools = [(schema, fn) for schema, (_, fn) in zip(told, tools, strict=True)]
    if single and given:
        told = _singled([schema for schema, _ in tools])
        tools = [(schema, fn) for schema, (_, fn) in zip(told, tools, strict=True)]
    if few and given:
        # this one takes schemas away rather than rewriting them, so the callables are
        # matched back by name: a caller's own acting tool keeps the function it came with
        kept = {str((schema.get("function") or {}).get("name") or ""): schema
                for schema in _few([schema for schema, _ in tools])}
        tools = [(kept[name], fn) for schema, fn in tools
                 if (name := str((schema.get("function") or {}).get("name") or "")) in kept]
    if tight and given:
        # a caller's own tools carry a show of their own; tight is about what it says
        told = _tight([schema for schema, _ in tools])
        tools = [(schema, fn) for schema, (_, fn) in zip(told, tools, strict=True)]
    schemas = [schema for schema, _ in tools]
    run = {str((schema.get("function") or {}).get("name") or ""): fn for schema, fn in tools}

    known = {str(n["id"]) for n in (graph.get("nodes") or ())}
    if tight:
        system = system.replace(SHOW_PARAGRAPH, TIGHT_SHOW_PARAGRAPH) + " " + TIGHT_SYSTEM_SENTENCE
    if batch:
        system = system + "\n\n" + BATCH_SYSTEM_SENTENCE
    if single:
        system = system + "\n\n" + SINGLE_SYSTEM_SENTENCE
    if few:
        # the prompt names `path_between` in its second sentence, and this offer has no
        # such tool: swapped for what to do instead, never left to be reached for
        system = system.replace(PATH_CLAUSE, FEW_PATH_CLAUSE) + "\n\n" + FEW_SYSTEM_SENTENCE
    lit = [str(h) for h in held if str(h) in known]
    if lit:
        label = {str(n["id"]): str(n.get("label") or "") for n in (graph.get("nodes") or ())}
        system = (system + "\n\nCurrently highlighted: "
                  + ", ".join(f"{label[h]} ({h})" for h in lit))
    out = Answer()

    def note(into: list[str], ids: Sequence[str]) -> None:
        for one in ids:
            if str(one) in known and str(one) not in into:
                into.append(str(one))

    said = [{"role": ("assistant" if t.get("role") == "assistant" else "user"),
             "content": str(t.get("content") or "")[:SAID_CHARS_BACK]}
            for t in turns if str(t.get("content") or "").strip()]
    # In this order: the summary, then what was recalled, then the window. The summary is
    # the thing that changes least, so it goes first and the cached prefix covers it; the
    # window is the thing that must be complete, so nothing here is taken out of it.
    ahead: list[dict[str, Any]] = []
    _, told = _spoken(summary)
    if told.strip():
        ahead.append({"role": "user", "content": EARLIER + " ".join(told.split())})
    for one in recalled:
        role, text = _spoken(one)
        if text.strip():
            ahead.append({"role": role, "content": RECALLED + text[:SAID_CHARS_BACK]})
    messages: list[dict[str, Any]] = [{"role": "system", "content": system}, *ahead, *said]
    # what filled the prompt, by part -- with a window, a summary and recall the length of
    # the conversation says nothing about what the slot holds; this does, and `spent`
    # keeps the server's exact count beside it
    out.spent.part("system", system)
    out.spent.part("tools", json.dumps(schemas))
    for one in ahead:
        text = str(one.get("content") or "")
        out.spent.part("summary" if text.startswith(EARLIER) else "recalled", text)
    for one in said:
        out.spent.part("window", one.get("content"))

    # A search has already been run — cheaply, by a small model that only measures meaning —
    # and what it found is read out before the first turn. Most questions are then answered
    # without calling anything, which is the point: every look_up is a whole round trip
    # through a large model, and the small one costs milliseconds. It is a suggestion, not
    # an answer: the tools are all still there, and the prompt says to go looking when this
    # is not what the question was about.
    #
    # It goes *before* the question, as candidates to check, and not after it as the
    # answer's raw material. Measured on gemma-4-E4B: eight likely entries handed over as
    # the last message after the question, phrased "use them if they answer it", took it
    # from 58% F1 to 33% -- it echoed the list rather than selecting from it. What comes
    # last is what a small model answers about; the question has to be the last thing it
    # reads, and the list has to arrive as something to verify rather than something to say.
    start = [str(i) for i in opening if str(i) in known][:FOUND]
    if start:
        material = (run.get("look_at") or (lambda a: look_at(graph, a["ids"])))({"ids": start})
        note(out.found, start)
        out.steps.append(f"was handed {len(start)} to start from")
        messages.append({"role": "user", "content":
                         "A search turned up these entries; some may be irrelevant. Look at "
                         "the ones that seem to answer the question before trusting them, "
                         "and ignore the rest:\n" + str(material)})
        out.spent.part("shortlist", material)
        if emit is not None:
            emit({"event": "tool", "name": "shortlist", "detail": f"{len(start)} to start from"})
    messages.append({"role": "user", "content": question})
    out.spent.part("question", question)

    # The tools that go looking. Taking these away is how a turn is made to stop searching
    # and answer; taking away *every* tool is how it was also made unable to act. A message
    # asking for an entry to be changed came in, the search went round in circles, the loop
    # stopped — and the one call left had no `request_change` to reach for, so a member's
    # request became "the model did not finish an answer". Stop the searching, not the rest.
    searching = set(SEARCHING)
    quiet = [x for x in schemas if str((x.get("function") or {}).get("name")) not in searching]

    def step(with_tools: bool) -> Any:
        offer = schemas if with_tools else quiet
        kw: dict[str, Any] = {"tools": offer} if offer else {}
        sent = time.monotonic()
        if emit is None:
            reply = client.chat(messages, think=False, **kw)
            out.spent.note(reply, time.monotonic() - sent)
            return reply
        streamed = {"thinking": False, "answer": False}

        def on_delta(kind: str, text: str) -> None:
            if not text:
                return
            name = "thinking" if kind == "thinking" else "answer"
            streamed[name] = True
            emit({"event": name, "text": text})

        reply = client.chat(messages, think=False, on_delta=on_delta, **kw)
        out.spent.note(reply, time.monotonic() - sent)
        if not streamed["thinking"]:
            trace = (getattr(reply, "thinking", "") or "").strip()
            if trace:
                emit({"event": "thinking", "text": trace})
        if not streamed["answer"] and not (getattr(reply, "tool_calls", None) or []):
            whole = (getattr(reply, "content", "") or "")
            if whole.strip():
                emit({"event": "answer", "text": whole})
        return reply

    def _searched(reply: Any) -> bool:
        """Whether that reply asked for anything other than showing."""
        return any((call.get("function") or {}).get("name") in searching
                   for call in (getattr(reply, "tool_calls", None) or []))

    spent = False
    spent_on: set[tuple[str, str]] = set()
    repeats = 0
    reply = None
    def dispatch(reply: Any) -> bool:
        """Run whatever the reply asked for. False when it asked for nothing."""
        nonlocal spent, repeats
        calls = getattr(reply, "tool_calls", None) or []
        if not calls:
            return False
        spent = True
        # One round, however many calls it carried. A reply may ask for several at once --
        # llama-server returns them in one message -- and every one of them is run below,
        # in order, each answered with a tool message of its own.
        out.rounds += 1
        messages.append({"role": "assistant", "content": reply.content or "", "tool_calls": calls})
        for call in calls:
            fn = call.get("function") or {}
            name = fn.get("name") or ""
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except ValueError:
                args = {}
            if emit is not None:
                emit({"event": "tool", "name": name, "detail": _call_detail(name, args)})
            do = run.get(name)
            again = (name, json.dumps(args, sort_keys=True, ensure_ascii=False))
            if do is None:
                result: Any = {"error": f"no such tool: {name}"}
            elif again in spent_on:
                # Asking the same thing twice is how a budget disappears: measured against a
                # real graph, one question spent six of its ten rounds looking up one word,
                # over and over, and never answered. Being told does not stop it — the tools
                # are taken away below, which does.
                repeats += 1
                result = {"already": "You have asked this exactly before and got the same "
                                     "answer. Use what it gave you, look for something else, "
                                     "or answer the question."}
                out.steps.append(f"asked {name} the same thing again")
            elif name == "look_up":
                result = do(args)
                note(out.found, [r["id"] for r in result] if isinstance(result, list) else [])
                asked = [str(x) for x in (args.get("texts") or ())] or [str(args.get("text") or "")]
                out.steps.append("looked up " + ", ".join(repr(x) for x in asked))
            elif name == "look_at":
                # one guard, in note: an id the model made up is neither read nor lit up
                ids = [str(i) for i in (args.get("ids") or ())]
                note(out.read, ids)
                result = do(args) or "nothing on those"
                real = sum(1 for i in ids if i in known)
                out.steps.append(f"read {real} entr" + ("y" if real == 1 else "ies"))
            elif name == "look_around":
                # The centres are read; everything the result named -- read back off the
                # text, so a neighbourhood the budget cut short is not counted -- is found.
                # Between them that is what look_at's ids and look_up's hits are, so the
                # score, the cap and tight's "did a tool return this" all treat it the same.
                ids = [str(i) for i in (args.get("ids") or ())]
                note(out.read, ids)
                result = do(args) or "nothing is joined to those"
                note(out.found, _ids_in(str(result)))
                real = sum(1 for i in ids if i in known)
                out.steps.append(f"looked around {real} entr" + ("y" if real == 1 else "ies"))
            elif name == "path_between":
                result = do(args)
                note(out.path, result.get("path") or [])
                out.steps.append("traced a path" if result.get("path") else "found no path")
            elif name == "summarise":
                result = do(args)
                # everything it named is bracketed, as look_around brackets its neighbours,
                # so an entry the summary read out counts as found and may be selected
                note(out.found, _ids_in(str(result)))
                out.steps.append("read the graph at a glance")
            elif name == "list_kind":
                result = do(args)
                listed = result.get("entries") if isinstance(result, Mapping) else None
                if listed is None:
                    out.steps.append(f"found no kind {str(args.get('kind') or '')!r}")
                else:
                    note(out.found, [r["id"] for r in listed])
                    note(out.listed, [r["id"] for r in listed])
                    out.steps.append(f"listed {len(listed)} of kind {result.get('kind')!r}")
            elif name == "show":
                # the same guard as look_at: an id the model made up is not lit up
                ids = [str(i) for i in (args.get("ids") or ())]
                note(out.show, ids)
                result = do(args)
                real = sum(1 for i in ids if i in known)
                out.steps.append(f"selected {real} entr" + ("y" if real == 1 else "ies"))
            else:
                result = do(args)
                out.steps.append(f"used {name}")
            spent_on.add(again)
            # A tool may bring pictures back for a vision model -- `web.tools(vision=True)`
            # returns what a page looked like under `_images`. llama.cpp cannot carry an
            # image inside a tool result, so they come out before the result is encoded
            # and go in as a user message of their own, right after it.
            seen: dict[str, Any] | None = None
            kept = 0
            if isinstance(result, Mapping) and "_images" in result:
                result = dict(result)
                pictures = list(result.pop("_images") or ())
                if pictures:
                    from ml_stack.vision.payloads import build_message

                    seen, report = build_message(
                        f"What {name} returned for the call above, as seen:", pictures)
                    kept = sum(1 for p in seen["content"] if p.get("type") == "image_url")
                    if not kept:
                        # a picture that cannot be prepared is said, not sent: a model
                        # told to look at nothing answers about nothing, confidently
                        seen = None
                        why = "; ".join(report.warnings) or "no reason given"
                        out.steps.append(f"{name} returned {len(pictures)} image"
                                         + ("" if len(pictures) == 1 else "s")
                                         + f" and none could be shown: {why}")
            count = _result_count(result)
            if kept and isinstance(result, Mapping) and not ({"entries", "path"} & set(result)):
                count = kept
            if emit is not None:
                emit({"event": "tool_result", "name": name, "count": count})
            messages.append({"role": "tool", "tool_call_id": call.get("id") or name,
                             "name": name,
                             "content": _cut(json.dumps(result, ensure_ascii=False,
                                                        default=_plain), reach)})
            out.spent.part("tool_results", messages[-1]["content"])
            if seen is not None:
                messages.append(seen)
        return True

    answered = False
    nudged = False
    nudged_single = False
    for _ in range(rounds):
        if repeats >= REPEATS:
            # going in circles: the loop ends here and the answer is asked for below
            out.steps.append("stopped searching in circles")
            break
        reply = step(True)
        if not dispatch(reply):
            answered = True          # it stopped calling tools, so this reply is the answer
            break
        # It read one entry and left the rest of what it found unread: the round it is
        # about to spend on the second one buys nothing the first call could not have
        # carried. Said once -- a model that ignores it twice is not going to be told into
        # it, and the reminder costs a message in every prompt after it.
        if batch and not nudged and _one_by_one(reply) and any(i not in out.read
                                                               for i in out.found):
            nudged = True
            messages.append({"role": "user", "content": BATCH_NUDGE})
            out.spent.part("question", BATCH_NUDGE)
            out.steps.append("asked it to read the rest in one call")
        # And the same thing from the other end: it read several entries at once, which is
        # the fat result `single` exists to avoid. Said once, for the same reason.
        if single and not nudged_single and _all_at_once(reply):
            nudged_single = True
            messages.append({"role": "user", "content": SINGLE_NUDGE})
            out.spent.part("question", SINGLE_NUDGE)
            out.steps.append("asked it to read one at a time")
        # Once it has said what its answer is about, more searching cannot improve that --
        # `show` is the last thing a turn does, and a round after it is a round trip spent
        # to be told the same. Only when the round did nothing else: a turn that showed and
        # kept looking in the same breath has not finished looking.
        if out.show and not _searched(reply):
            out.steps.append("said what to light, so the searching stopped")
            break
    if spent and not answered:
        # The searching is over, one way or another. What is left on the table are the tools
        # that act — saying what to light, asking for the graph to be changed — and they are
        # run, not ignored: a member's request arrived on exactly this turn and was dropped
        # because nothing here executed it.
        reply = step(False)
        if dispatch(reply):
            reply = step(False)
    # a thinking model can stop calling tools and still say nothing, and it can run out of
    # rounds the same way. Either silence gets one plain instruction to answer; one that
    # only ever searched also gets the top finds read to it first
    if spent and not (getattr(reply, "content", "") or "").strip():
        nudge = "Answer the question now, in plain words."
        if not out.read and out.found:
            top = out.found[:FOUND]
            do = run.get("look_at")
            material = (do({"ids": top}) if do is not None else look_at(graph, top)) or ""
            note(out.read, top)
            out.steps.append(f"read the top {len(top)} find" + ("" if len(top) == 1 else "s"))
            if emit is not None:
                emit({"event": "tool", "name": "look_at", "detail": f"{len(top)} ids"})
                emit({"event": "tool_result", "name": "look_at",
                      "count": _result_count(material)})
            nudge = ("What the graph holds on what you found:\n" + str(material)
                     + "\n\n" + nudge)
        messages.append({"role": "user", "content": nudge})
        reply = step(False)
    out.content = (getattr(reply, "content", "") or "").strip()
    if out.content:
        out.content, meant = spoken_show(out.content)
        if meant:
            # what it wrote down is what it meant, and it is a tighter set than asking again
            note(out.show, meant)
            out.steps.append("said what to light in words, so it was not asked again")
    if out.content and not is_working(out.content):
        trimmed = without_notes(out.content)
        if trimmed != out.content:
            out.steps.append("trimmed the notes off the front of the answer")
            out.content = trimmed
    if out.content and is_working(out.content):
        # It answered with its notes. Same remedy as saying nothing at all — the notes are
        # the material, and what is wanted is the answer they were working towards.
        out.steps.append("answered with its notes, so was asked again")
        messages.append({"role": "assistant", "content": out.content})
        out.content = ""
    if not out.content:
        # Thinking is a scratchpad — "Actually the look_at shows… need to check X" — and
        # showing it as the answer reads as a broken machine. It is offered back as
        # material and asked for the answer it was working towards; only prose that reads
        # like an answer is used, never the working itself.
        trace = (getattr(reply, "thinking", "") or "").strip()
        if trace or messages[-1].get("role") == "assistant":
            messages.append({"role": "user", "content":
                             "Write the answer your notes were working towards, in plain "
                             "prose for someone who cannot see them. Do not narrate what you "
                             "looked up."})
            reply = client.chat(messages, think=False)
            said = (getattr(reply, "content", "") or "").strip()
            out.content = "" if is_working(said) else without_notes(said)
            if out.content and emit is not None:
                emit({"event": "answer", "text": out.content})
    if not out.content:
        # Why there is no answer, from the last reply, because the assumption is always the
        # token budget and it is almost never that. Measured: the same failing question at
        # n_predict 2048 and 6144 came back `finish_reason: stop` both times, with 628 and
        # 767 characters of reasoning and nothing after it. Nothing printed that, so the
        # ceiling was raised again, which proved nothing. Now the reader is told.
        why = getattr(reply, "finish_reason", None) or "unknown"
        thought = len((getattr(reply, "thinking", "") or "").strip())
        wrote = len((getattr(reply, "content", "") or "").strip())
        out.steps.append(f"no answer: finish_reason={why}, thinking {thought} chars, "
                         f"answer {wrote} chars")
        out.content = ("The model did not finish an answer. What it opened is lit up on the "
                       "graph; asking again, or more narrowly, usually gets one.")
        if emit is not None:
            emit({"event": "answer", "text": out.content})
    # A model that answered without touching the graph answered from nothing, and nothing is
    # what it knows: this graph was not in its training data. Measured over the invented
    # community, six of gemma-4-E4B's nine failures were exactly this — two model calls, a
    # hundred characters of prose, no search. The larger models never do it, which is why it
    # went unseen until a small one was measured. It gets one chance to go and look, with the
    # searching tools back on the table; if it declines again, its answer stands as it is.
    if out.content and not (out.read or out.found or out.path) and "look_up" in run:
        messages.append({"role": "assistant", "content": out.content})
        messages.append({"role": "user", "content":
                         "You answered without searching the graph. You have not seen this "
                         "graph before and cannot answer it from memory. Call look_up now "
                         "with the words from the question, then answer from what it returns."})
        if emit is not None:
            # a reader watching this stream has the first answer on screen already, and the
            # page appends what arrives; without being told to start over it would show the
            # two answers run together
            emit({"event": "restart", "why": "answered without searching"})
        reply = step(True)
        if dispatch(reply):
            reply = step(False)
        said = (getattr(reply, "content", "") or "").strip()
        if said:
            # whatever it said last is what the reader is looking at, so it is the answer
            out.content, meant = spoken_show(said)
            note(out.show, meant)
        if out.read or out.found or out.path:
            out.steps.append("answered without looking, so it was sent to look")

    # What the answer is about is the one thing only the model knows, and a turn that spends
    # its budget searching never reaches `show` on its own — measured against a real graph,
    # not one staffing question in six did. So it is asked outright, once, with the answer it
    # just wrote in front of it and nothing else to call.
    if out.content and not out.show and (out.read or out.found or out.path):
        messages.append({"role": "assistant", "content": out.content})
        messages.append({"role": "user", "content": TIGHT_NUDGE if tight else SHOW_NUDGE})
        offer = _schema("show", _tight(TOOLS)) if tight else _schema("show")
        last = client.chat(messages, think=False, tools=[offer])
        for call in (getattr(last, "tool_calls", None) or []):
            fn = call.get("function") or {}
            if (fn.get("name") or "") != "show":
                continue
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except ValueError:
                continue
            ids = [str(i) for i in (args.get("ids") or ())]
            note(out.show, ids)
        if out.show:
            out.steps.append(f"selected {len(out.show)} entr" + ("y" if len(out.show) == 1 else "ies"))
    if (tight or kinds) and out.show:
        # The `made` case: a name in the prose that look_at never read is a guess, and
        # lighting it endorses the guess. What was handed over at the start was read to the
        # model too, so it counts as read. Then the cap: what the prose names is kept, most
        # named first, and what it merely lit goes.
        named = _named_counts(graph, out.content, out.show)
        if tight:
            # Known is what any tool returned -- found by look_up, listed by list_kind,
            # read, traced, handed over -- and what is joined to something read: measured
            # 2026-09-02, `place:calderwick` was listed and joined to the people read, and
            # dropping it as "unread" cut a right answer. Only a name no tool ever returned
            # is a guess. `out.found` is where a `look_around` neighbourhood lands, so a
            # neighbour the model only ever saw indented under something it asked for still
            # counts as read.
            seen = (set(out.read) | set(start) | set(out.found) | set(out.path)
                    | _joined_to(graph, out.read))
            guessed = [i for i in out.show if named.get(i) and i not in seen]
            if guessed:
                out.show = [i for i in out.show if i not in guessed]
                out.steps.append(f"dropped {len(guessed)} unread from show")
        if kinds and out.show:
            # The question word said what kind the answer is, so what is not of that kind
            # was found on the way rather than asked for -- the topic that led to the
            # people, for a question that asked *who*. A listing is exempt, as it is from
            # the cap below; so is a question that named several kinds or none, which
            # `asked_kinds` answers None for.
            wanted = asked_kinds(question)
            if wanted:
                listed = set(out.listed)
                of_kind = {str(n["id"]): _kind_of(n) for n in (graph.get("nodes") or ())}
                kept = [i for i in out.show
                        if i in listed or of_kind.get(i, "") in wanted or i not in of_kind]
                # A filter that empties the selection lights nothing, which is worse than
                # lighting the wrong kind: the reader is left looking at a blank graph.
                if kept and len(kept) < len(out.show):
                    out.steps.append(f"dropped {len(out.show) - len(kept)} of another kind "
                                     f"from show, which asked for "
                                     + "/".join(sorted(wanted)))
                    out.show = kept
        # The cap never cuts a listing: "which companies are here?" has as many right
        # answers as there are companies, and cutting thirteen to six alphabetically threw
        # away three expected ones (measured 2026-09-02). It cuts among the rest, keeping
        # what the prose names first.
        if tight:
            listed = set(out.listed)
            rest = [i for i in out.show if i not in listed]
            if len(rest) > LIT_TIGHT:
                whole = len(out.show)
                kept = sorted(rest, key=lambda i: -named.get(i, 0))[:LIT_TIGHT]
                out.show = [i for i in out.show if i in listed] + kept
                out.steps.append(f"cut {whole - len(out.show)} of {whole} lit")
    for one in (*out.read, *out.path, *out.found):
        if one not in out.ids:
            out.ids.append(one)
    out.ids = out.ids[:limit]
    out.show = out.show[:limit]
    if emit is not None:
        emit({"event": "done"})
    return out
