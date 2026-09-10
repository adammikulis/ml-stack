"""Every string a question is asked with: the system prompt, the tool schemas, the
sentences each way of asking adds, and the questions a router matches against."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import Any

from ml_stack.asking import Asking

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

# What a turn answered under a schema is told, on a copy of the system prompt: the two
# shapes the grammar allows, so the model reaches for the right one rather than being
# steered into it token by token.
CONSTRAINED_SYSTEM_SENTENCE = (
    'When you are asked to reply as JSON, reply with one object: {"name": "<tool>", '
    '"arguments": {...}} calls one tool, and {"answer": "..."} is your answer, written out '
    'in full inside it.')


# The tools that look for something, and so are refused on the last turn; what still runs
# then is the tools that act (show, and whatever a caller adds -- a change request, say).
# The web tools are searches too: a model that may still search will search instead of
# answering, which is the failure the last turn exists to end. Refused, never taken away:
# the tools block is rendered before every message, so a turn offering a different list
# loses the prompt cache from the first byte and re-reads the whole conversation.
SEARCHING = frozenset({"look_up", "look_at", "look_around", "path_between", "list_kind",
                       "summarise", "quote", "web_search", "web_read", "web_look"})
# What the last turn is told as the user, and what a search called on it is answered with.
OVER = "The searching is over. Answer now with what you have."


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


QUOTE_SCHEMA: dict[str, Any] = {"type": "function", "function": {
    "name": "quote",
    "description": "Read the source's own words behind entries you have looked at: the "
                   "passage itself, and the source, section and pages it was read at. Call "
                   "it before you quote or claim anything, and write only what the passage "
                   "says. Quotation marks belong only around a passage that came back with "
                   "\"verbatim\": true; one without it is the graph's own wording, and an "
                   "entry with no passage at all has nothing behind it -- say so rather "
                   "than filling the gap. Example: quote with "
                   "{\"ids\": [\"concept:glimmer-node\"]}.",
    "parameters": {"type": "object", "properties": {
        "ids": {"type": "array", "items": {"type": "string"},
                "description": "entry ids to read the passage behind, e.g. "
                               "[\"concept:glimmer-node\"]"}},
        "required": ["ids"]}}}
# What a model is told when it is answering with citations: the tools carry where every
# entry was read, and an answer says it.
CITE_SYSTEM_SENTENCE = (
    "Every entry you read says where it was read: the source, the section and the pages. "
    "Name that source in your answer for each thing you say, and call quote to read the "
    "passage behind an entry before you put its words in quotation marks. An entry that "
    "says where it came from can be cited; one that does not cannot, and a claim you "
    "cannot cite does not go in the answer.")

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


def schema_for(name: str, among: Sequence[Mapping[str, Any]] = TOOLS) -> dict[str, Any]:
    """The tool schema called ``name``.

    By name, never by position: the order of ``TOOLS`` is the order a model reads them in,
    which changes when a tool is added, and ``TOOLS[3]`` was ``show`` right up until it
    was not.
    """
    for schema in among:
        if (schema.get("function") or {}).get("name") == name:
            return schema
    raise KeyError(name)


_TERSE_SHOW = schema_for("show", TERSE)["function"]["description"]


def as_asked(schemas: Sequence[Mapping[str, Any]], asking: Asking) -> list[dict[str, Any]]:
    """Those schemas, copied, with everything ``asking`` asks for written onto each.

    One walk over the set. ``rich`` says what a look_up hit carries; ``batch`` and
    ``single`` show each searching tool a three-entry and a one-entry call; ``few`` drops
    every way of looking but `look_up` and `look_at` and tells look_up what is gone;
    ``tight`` replaces what `show` says, in the terse wording for a terse `show` and the
    full one otherwise. The schemas themselves come back when the asking asks for none of
    it.
    """
    if not (asking.rich or asking.batch or asking.single or asking.few or asking.tight):
        return list(schemas)
    out: list[dict[str, Any]] = []
    for schema in copy.deepcopy(list(schemas)):
        name = str((schema.get("function") or {}).get("name") or "")
        if asking.few and name in SEARCHING and name not in FEW_TOOLS:
            continue
        more = ""
        if asking.rich and name == "look_up":
            more += " " + RICH_SENTENCE
        if asking.batch:
            more += BATCH_EXAMPLES.get(name, "")
        if asking.single:
            more += SINGLE_EXAMPLES.get(name, "")
        if asking.few and name == "look_up":
            more += " " + FEW_SENTENCE
        if more:
            schema["function"]["description"] += more
        if asking.tight and name == "show":
            was = schema["function"]["description"]
            schema["function"]["description"] = (
                TIGHT_SHOW_TERSE if was == _TERSE_SHOW else TIGHT_SHOW)
        out.append(schema)
    return out


# How the summary and a recalled turn announce themselves. The summary is one user message,
# placed before everything that changes per question; a recalled turn keeps its role and
# says it was recalled, so the model does not read an old answer as the one just given.
EARLIER = "Earlier in this conversation: "
RECALLED = "Recalled from earlier in this conversation: "


# What one turn is told when it asked for the same thing twice, when it said nothing,
# when it answered from memory, and what the reader is told when nothing answered at all.
ALREADY = ("You have asked this exactly before and got the same answer. Use what it gave "
           "you, look for something else, or answer the question.")
SHORTLIST = ("A search turned up these entries; some may be irrelevant. Look at the ones "
             "that seem to answer the question before trusting them, and ignore the "
             "rest:\n")
ANSWER_NOW = "Answer the question now, in plain words."
FROM_THE_NOTES = ("Write the answer your notes were working towards, in plain prose for "
                  "someone who cannot see them. Do not narrate what you looked up.")
GO_AND_LOOK = ("You answered without searching the graph. You have not seen this graph "
               "before and cannot answer it from memory. Call look_up now with the words "
               "from the question, then answer from what it returns.")
NO_ANSWER = ("The model did not finish an answer. What it opened is lit up on the graph; "
             "asking again, or more narrowly, usually gets one.")


DRAFT_SYSTEM = (
    "You write a short note introducing people to each other, for the person who runs "
    "this group to send. You are given what the graph holds on each of them, in their own "
    "words.\n\n"
    "Write the message itself and nothing else — no preamble, no subject line, no sign-off, "
    "no explanation of what you are doing. Address them by name. Say in one sentence each "
    "what the other should know about them, using what they actually said rather than "
    "adjectives, then say plainly why they are being introduced and suggest one concrete "
    "first step. Three to five sentences. Warm and ordinary, the way a person writes to "
    "people they know — not a press release.\n\n"
    "Everything you say comes from what you were given. Never invent a fact about anyone. If "
    "what you were given does not support an introduction, say so in one sentence instead of "
    "writing one."
)
