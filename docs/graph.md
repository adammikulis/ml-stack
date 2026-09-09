# Working with a graph

A graph here is a mapping with `nodes` and `edges` and nothing else agreed in advance —
what a project calls its kinds and its relations is the project's business.

```python
from ml_stack.graph import GraphStore, replace, converse, render, hybrid

replace("graph.ladybug", graph)              # safely: see below
with GraphStore("graph.ladybug") as store:
    store.set_embedding("person:ada", vector)
    store.similar(vector)                    # nearest by meaning
    store.search("compiler")                 # stemmed, so it finds "compilers"
    store.shortest_path("person:ada", "person:bea")

hybrid(graph, "who fixes machines", store=store, vector=asked)   # all three at once
converse("how are these two connected?", graph, client)          # the model, with tools
open("page.html", "w").write(render(graph, title="Who knows what"))
```

**The model is given six things it can do, not the graph.** `look_up` finds entries by
name or by the words attached to them, `look_at` reads what is held on them, `look_around`
reads a whole neighbourhood at once, `path_between`
traces how two connect, `list_kind` reads out everything of one kind, and `show` says what
the answer is about. `list_kind` exists for the question no search reaches: "which companies
do people here work for?" is answered by every `org` in the graph, and no word finds those,
because nothing is *labelled* company — a small model searched "company", then
"organization", found nothing and gave up. A kind the graph does not have comes back as the
kinds it does, with counts, so a wrong guess costs one call rather than a turn. An answer
that comes back empty says why in `steps` — `no answer: finish_reason=stop, thinking 628
chars, answer 0 chars` — because the assumption is always the token budget and, measured,
it almost never is. A tool of the caller's own that returns pictures (`_images` in its
result) has them shown to the model as a message of their own, since a tool result cannot
carry an image; a picture that cannot be prepared is a line in `steps`, not a crash.

**Fewer, fatter calls, for a model that reads far faster than it writes.** Measured
2026-09-02 over the invented community, Qwen3.8-Flash-Next — hybrid recurrent, 256k of
context at 48K bytes a token — found nearly everything (89-95% recall, and 76-83% precision
once the asking was tight), and spent 5-9 tool calls a question at about 2k new tokens each:
half the wall clock reading results back at ~390 tok/s, the other half writing at ~35. For
that model a round trip is the expensive part and reading is nearly free, so the way to make
a question cheaper is to take more per call. `look_around(ids, hops=1)` is the fat call: the
entries you name, and under each of them everything joined to it with the relation, the
neighbour's kind, **its id in brackets** and a line of its own words — so "who could help
with X" is one `look_up` and one `look_around` where it used to be five `look_at`s, and the
answer may select a neighbour it never looked up. `Asking(reach=N)` is the budget
that makes such a result safe to ask for: N tokens per tool result instead of the flat 6000
characters, with `look_at`, `look_around` and `list_kind` packing **whole entries with their
quotes**, most-mentioned first, rather than every entry with its words clipped — because the
quote is the evidence, and `list_kind`'s fixed forty was only ever a guess at what a result
may cost. Both are off by default (`reach=None`), because the small models have the opposite
profile — E4B and E2B reach about 60% recall with cheap decoding and an expensive cache, and
a fatter result is a worse trade for them. `ml-stack-bench --also reach` measures the fat
asking against the default on one load, and `--reach N` sets the size on every way.

**`look_up`'s first hits are put in the order the vectors mean.** Fusion decides which
entries come back — a reciprocal rank is a vote count, and three ways agreeing beats one way
being certain — but it cannot say which of them the asker meant, because a vote is not a
distance. The vectors can, and they are already in the store, so `hybrid` re-orders its first
six hits by cosine to the question whenever the question arrived embedded. Membership never
changes and a hit the vectors have never seen keeps its place rather than sinking, so an
exact label match cannot be pushed down the page by something the embedder merely likes;
`rerank=False` is the fused order exactly as it was.

**A node with no words of its own is still found by what its neighbours mean.** Vectors come
from text, and plenty of entries have none -- a person nobody described, a unit that is only
ever mentioned. `graph.smooth` spreads every vector over the graph: rounds of `D^-1/2 (A + I)
D^-1/2` through the same message passing a GCN layer uses, each result scaled to unit length,
so an entry with nothing written about it ends up pointing where its neighbours point and an
entry with plenty is nudged towards the company it keeps. `remember(..., smooth_hops=N)`
writes the smoothed vectors as it embeds, and `ml-stack-store embed PATH --smooth N` spreads
the ones a store already holds without asking a model anything.


**A hit that says why it matched, and who is joined to it, is behind a flag until it is
measured.** `look_up` returns `{id, label, kind}` and nothing else, so a model cannot tell an
exact label from one word in one quote, and a topic it finds is only halfway to the people
who have it — measured against a real graph, every staffing question spent rounds guessing
spellings. `tools_for(graph, rich=True)` (and `Asking(rich=True)`, `hybrid(...,
rich=True)`) adds `score`, `matched` — which of `label`, `attribute`, `said`, `words`,
`meaning` found it — and, on anything that is not a person, `joined`: the eight
most-mentioned people on any edge to it. The look_up description gains one sentence saying
so, on a copy. With the flag off nothing observable changes, byte for byte, because the
answer cache fingerprints those descriptions and a sweep of the current behaviour has to stay
comparable with the one that measures this.

**Lighting only what answers the question is the asking, not a variant.** A model that
finds nearly everything can then light nearly everything: measured over the invented
community, 34 questions at 32k, Qwen3.8-Flash-Next reached 92% recall at 44% precision, and
named 70 entries its tools never found, read or showed, where a good answer lights about
two. So `converse` and `tools_for` ask tight by default — the words about `show` change, on
a copy: light only the entries that answer the question, the ones the asker would act on,
never what was looked at on the way, usually one to three — the closing nudge says the same,
one sentence is added to the system prompt (name only what a tool returned; say when
something was not found rather than guess a name), `show` is capped at `LIT_TIGHT` (six, the
ids the prose names kept first, `cut N of M lit` in `steps`), and an entry the prose names
that no tool ever returned is dropped (`dropped N unread from show`). `tight=False` is the
loose asking kept as a **control** — the words the ranking runs and the answer cache
fingerprinted, the same schema objects, byte for byte as they were — and
`ml-stack-bench --also loose` measures it against the default on the same load.

**Three more askings, each off until it is measured: `batch`, `kinds`, `summary`.**
Measured 2026-09-02 over the invented community, Qwen3.8-Flash-Next answered at 70% F1 —
85% recall, 65% precision — and spent 25 seconds a question over about seven tool calls.
`Asking(batch=True)` is for the seconds: the calls were one question asked one entry
at a time, because nothing said the ids are a list, so the system prompt says it, each
searching tool's description gains a worked three-entry call, and a turn that reads one
entry while more are still unread is told once to *read the rest in one call*. What it
should move is `Answer.rounds` — a round is a round trip through the model, and a reply that
asks for three tools at once is one round, all three of them run before the next turn.
`Asking(kinds=True)` is for the precision: the misses were mostly right-adjacent —
the topic lit beside the people for a question that asked *who* — and the question word
already says what kind the answer is, so `asked_kinds` reads it off the asking clause and
`show` keeps only that kind. It filters nothing when the question named several kinds or
none (`how is X connected to Y`, `tell me about X`), a listing is exempt as it is from the
cap, and a filter that would empty the selection is not applied; over the bench's own 110
questions it filters 72 to the right kind, leaves 31 alone and gets exactly one wrong.
`Asking(summary=True)` is for the broad question no search reaches — "what is
this group about?" has no name in it to look up — and adds `summarise`: counts per kind, the
ten most-mentioned entries of each kind with a line of their own words and their ids in
brackets, and the busiest relations, computed from the graph with no model call at all.
`routing_prompts(summary=True)` is what to route against when it is offered.
`ml-stack-bench --also batch --also kinds --also summary` measures all three against the
default on one load.

**Two more, pulling the other way: `single` and `few`.** `Asking(single=True)` is
`batch` turned around, and it is here for the opposite model. A fat tool result is a long
thing to hold in mind: a small model handed a dozen entries in one message answers about the
last one it read, or about none of them, and what comes back is fluent and about nothing. So
the system prompt says to read one entry at a time, each searching tool's description gains a
worked *one*-entry call, and a turn that reads several at once is told once to read them one
at a time. It buys short results and spends rounds — exactly the trade `batch` makes in the
other direction. `Asking(few=True)` offers three tools — `look_up`, `look_at`, `show`
— and takes away every other way of looking, for the model whose tool choice degrades with
the number of schemas rather than with the question. **Nothing is faked to cover what went.**
There is no path tool and no listing tool in that offer, so look_up's description and the
system prompt say so, and say how to answer those questions with what is there: look both
ends up, read them, and read on to whatever they are joined to. A description telling the
model to "ask for a path as `path A to B`" would be a tool that does not exist, and a model
that believed it would spend every turn it has on a call nothing answers. Anything that does
not search survives `few` — `show`, and a caller's own change request — because it is not a
choice between ways to look. `Asking(rounds=N)` is the ceiling those two trade
against: `--rounds N` rides on every way a sweep asks, the way `--reach` does.

**One asking per model.** These ways exist to be *chosen per model by measurement*, never
picked once and applied to everything. Flash-Next wants batch, kinds and summary together;
a 2B that loses the thread of a long result wants `single` and more rounds; a model whose
tool choice degrades with the offer wants `few` and more rounds still — and which is which
is a number in a store, not a taste. The same goes for sampling: a model is measured at the
temperature, top-p and top-k that suit *it*, not at one setting shared by all. So the choice
lives in the model's **profile** (`ml_stack/data/profiles.json`,
`ml_stack.serve.profile`, and [the shape a model measured best in](serving.md)):
`ml-stack-bench report --profile`
writes, per model and workload, the asking and the sampling of the fastest row whose F1 the questions
could not tell apart from the best — F1 alone would trade real seconds for a hundredth of a
point it cannot see — with the label of the row that set it. `converse(question, graph, client,
asking=Asking.for_model(MODEL))` then asks that way, and `ml-stack-serve profile MODEL`
reads it out: `ask with tight + few + rounds 20 at temperature 1.0 / top-p 0.95 /
top-k 20`.

**Where people say they are becomes a point on the map.** A node that carries a place in its
attributes -- and a `place` node, which carries one in its name -- goes through
`ml_stack.geo`, and comes back with `lat` and `lon` on it, which is what the page's map
draws. `ml-stack-graph geocode --graph FILE --cache FILE.json` is the command, and the cache
means a place is asked about once however many times a run is repeated. `--near K` joins each
placed entry to its `K` closest with a `near` edge weighted `1 / (1 + kilometres)`, so
"who else is out there" is a question the graph itself answers, and the layout draws the
people in one city together.

**The page is components, and a page is the list of them.** `graph.page.render` assembles
the page out of one file per component under `graph/web/components/` -- a custom element
apiece, each holding its own style, markup and script, sharing one model -- so a caller can
leave a component out (`parts=`) or add one of its own file; `ml_stack.ui.assemble` is the
assembler, for any page built this way. The ask pane, the review queue, the change-request
form, the refresh button and the note drafter each have a route behind them, and
`ml_stack.graph.serve` has one mixin per route: `AskRoutes` streams answers from
`/ask/stream`, falls back to `/ask`, and reopens a conversation from `/thread/<name>`;
`ReviewRoutes` lists and acts on a `graph.review.Queue`; `RequestRoutes` keeps a request
on disk before saying so; `RefreshRoutes` streams the stages a subclass's `stages()`
yields; `DraftRoutes` hands ids to a `drafter`. A subclass says how a question is answered
(`asker`), where conversations are kept (`threads`) and what each other route is given,
and hangs its own journal off `answered`; each route is a 404 until it is given its thing,
and the ones that change the machine refuse a request that came through a proxy.
`Handler` composes them all; who may ask is still the project's policy.

**A store cannot be lost to a bad rebuild.** A pipeline that read nothing produces an empty
graph, and an empty graph looks exactly like "remove everything". `replace` refuses a write
that would take most of a store, and leaves a verified snapshot when it would take a tenth.
`snapshot` and `roll_back` are there directly, and a restore saves what is there first.

**A store checks itself.** Every `put_doc` reads its document back by key and raises
`StoreMismatch` when what comes back is not what went in; a node is read back by id the same
way, an edge from its own `RETURN`, and `replace` counts what it wrote before committing.
Measured 2026-09-01: twelve bench runs read back empty through a scan of `Doc.value` while a
lookup by key returned them whole, so `ml-stack-store check PATH` reads every document, node
and edge by key *and* by scan and prints one line per disagreement (exit 1 on any);
`--fix` rewrites a document the scan lost and checks again rather than announcing a repair,
and `ml-stack-store docs PATH` lists the documents with their sizes.

**How much memory a store takes is yours to set.** Left alone the engine claims a share of
the machine's, which is what one store on one machine wants and four at once do not:
`GraphStore(path, buffer_pool_size=N)`, or `$MLSTACK_STORE_MEMORY=N` for every store a
process opens, caps it at N bytes.

**Two processes cannot corrupt one.** The database's own lock stops the second writer with an
IO error; what `ml_stack.graph.access` adds is knowing whose lock it is, waiting for a turn,
and letting go of a read handle when a writer wants in.

**The files around a graph are the same in every project.** `ml_stack.files.write_json`
writes beside the file and renames over it, so whatever is reading the graph while a
pipeline rewrites it sees the old one or the new one and never half of either;
`prune_orphans` deletes the per-record files (an extraction per message) whose record the
log has since dropped. `ml_stack.geo.geocode_all` turns the places people write — "Raleigh",
"MD", "sf" — into points through Nominatim, cached to a JSON file, one request a second,
picking the answer that is actually *called* what was asked rather than the county Nominatim
ranks first; pass your own `user_agent`, as its usage policy asks. `ml_stack.redact.names_in`
reads every name a graph and its message log hold, for a `Redactor` to keep out of anything
printed.

**A vocabulary the model coined drifts, and folds back.** A model asked to read prose into
(subject, relation, object) invents the relation as it goes, so one relationship arrives as
`works_at`, `worksat` and `worked_at` and the graph is split three ways.
`ml_stack.entities.fold.fold_edges` keeps whichever spelling the graph already uses more and
folds the rest into it -- `entities.close` decides what counts as the same word -- but only
while one of them is rare: past `ESTABLISHED` weight both are names people keep choosing, and
neither folds without a written entry saying which is right. Every fold is logged *and*
returned, so a wrong one is something a test can point at. `dead_keys` is the other half:
the entries of a hand-written map that nothing produces any more, which fail silently
otherwise.

**An extraction already done is not done again.** `Client.extract(..., cache_dir=...)` keeps
each answer as a file there and reads it back instead of asking the model. The key is
`cache_version` + the schema + the text + `cache_extra` (the rest of the prompt that varies
per record -- the thread a message replies to, the vocabulary offered) and deliberately *not*
the instructions: wording those is iterative, and a pipeline that re-reads its whole corpus
because a sentence was rephrased is one where nobody rephrases anything. `cache_version` is
the knob for a change nobody should be allowed to skip. Only an answer that passed `check` is
kept, so a run that gave up is asked again rather than remembered as settled.

## A hierarchy read out of prose or a picture

```python
from ml_stack.graph.tree import FAMILY, ORG, read, to_graph

rows = read(client, ORG, images=[chart], reader=document_model)   # or text=...
graph = to_graph(rows, ORG)                                       # entries and links
```

An org chart, a family tree, a subject taxonomy and a parts breakdown are one object with
four shapes: named things, and a link from each to the one above it. A `Shape` carries what
an entry is called, what the link means and how many parents it may have — a family tree
keeps two, an org chart one, because a second manager there is a misreading.

`reader` is a second model that reads the picture first, which is how a document model gets
used for what it is good at: it transcribes, `client` structures. A picture needs a server
started with its projector (`--mmproj auto`), and a picture that cannot be prepared raises
rather than being dropped — otherwise a model is asked to read a chart with no chart
attached, and answers confidently about nothing.

## Which tool a question wants

```python
from ml_stack.graph.ask import TOOL_PROMPTS, tools_for
from ml_stack.graph.route import narrow, rank

routed = rank(question, TOOL_PROMPTS, base_url=embedder, model=name)
tools = narrow(tools_for(graph), routed)      # [] when the question wants no graph
```

`graph.route` asks a small embedder which tool a question resembles, by comparing it to
**example questions** rather than to the tools' descriptions. That distinction is the whole
of it: a question against prose describing a capability is comparing unlike things, and a
question against questions is like-to-like. The examples live in `ask.TOOL_PROMPTS` and are
never sent to the chat model, which wants the opposite text — what a tool *does*.

Both sides carry the same embedding prefix, because both are questions. Using the
asymmetric `QUERY`/`DOCUMENT` pair here scored "tell me about Otto Vance" at 0.409 against
an example reading "tell me about Iris Bellweather", which is the same sentence; with the
symmetric prefix it is 0.83.

The useful case is the one that is not a tool at all. `CHAT` collects greetings, jokes and
asides, and a message routed there is offered **no tools whatsoever** — one model call
instead of the six a graph question takes. Without somewhere for those to go, a greeting is
matched against four search tools and wins one of them: "hi" scored 0.900 against
"highlight them on the graph", because everything is close to everything and the only
question is close to *what*.

"Which companies are here?" is the other question that is not a search. Nothing in a graph
is labelled company, so `look_up` finds nothing however it is worded, and a question that
asks what *kinds* of thing are represented routes to `list_kind` instead. Its examples are
kept apart from `look_up`'s on purpose: one asks for a particular thing, the other for
everything of one sort.

Nothing narrows unless the routing was clear, `show` survives every narrowing, and an
embedder that will not answer routes nothing rather than defaulting to chat — a real
question mistaken for small talk is answered without looking anything up, which reads as a
confident answer and is about nothing.

When a search has already been run for the question — a shortlist, from the word index and
the vectors — what it found is read to the model *before* the question, as candidates to
check, and never after it as material to answer from. Measured on gemma-4-E4B: eight likely
entries handed over as the last message, phrased "use them if they answer it", took it from
58% F1 to 33%, because it echoed the list rather than selecting from it. What comes last is
what a small model answers about, so the question is the last thing it reads.

## Answering the same question twice

```python
from ml_stack.graph.ask import Answer, converse
from ml_stack.graph.cache import asked, digest, forget

out, again = asked(store, question, lambda: converse(question, graph, client),
                   kind=Answer, graph=graph, model=name, system=SYSTEM, tools=tools)
```

`asked` hands back the answer already given when nothing that shaped it has changed, and
calls the model only on a miss — measured at 27.9s against 0.00s for the repeat. What makes
that safe is the fingerprint, which covers the graph, the model, the system prompt, **the
tool schemas including their descriptions**, the shortlist and whatever `context=` the
caller adds (the turns before this one, most obviously). Rewording a tool changes what the
model does with it, so it misses — which is right: rewording them moved every score in the
bench.

`keep=` refuses an answer that should not be served twice — one whose turn also *did*
something, like filing a change request. Its answer is a receipt, and handing it out again
would tell the next person their request was filed when it was not.

A rebuilt graph misses on its own, but does not sweep on its own: pass
`forget(store, keeping=digest(graph))` after a rebuild or the store keeps every answer it
ever gave. Entries live under keys beginning `_`, which `GraphStore.docs` skips, so a cache
in the same store as a graph never leaks into `read()`.

## A conversation of any length

```python
from functools import partial
from ml_stack.graph.thread import WINDOW, latest_summary, recall, recent, summarise, write_summary

turns = recent(store, thread, turns=WINDOW)                       # the last ten, always
summary = latest_summary(store, thread)                           # one paragraph, rarely changed
recalled = recall(store, thread, question, embedder=embed)        # two or three older turns
out = converse(question, graph, client, turns=turns, summary=summary, recalled=recalled)
...
summarise(store, thread, partial(write_summary, client))          # every EVERY turns
```

What goes back with a question, in this order: the system prompt; the latest **summary**, as
one message reading "Earlier in this conversation: …"; the turns **recalled** for this
question, oldest first, each marked as recalled; the last `WINDOW` (ten) turns, whole and in
order; the shortlist, if any; the question. The window is chosen by recency and nothing
else — a follow-up ("and where is she based?") resolves from it alone — and neither the
summary nor the recall ever takes a turn out of it. With no summary and nothing recalled the
messages are byte for byte what they were, which the ranking runs and the answer cache rest
on; pinned in `tests/test_graph_ask.py`.

`recall` is the word index over what was said, fused with the turn vectors when the same
`embedder` the turns were remembered with is given (`remember_turn(..., embedder=)`, kept
under the thread's own name), the way `search.hybrid` fuses. It never returns a turn inside
the window or a summary. `summarise` rolls the summary forward when `every` (eight) ordinary
turns have been said since the last, handing the writer the previous paragraph and those
turns with what they drew on; the paragraph is a `Turn` of role `"summary"`, joined to every
id it names that those turns rested on, out of `follow`'s ordinary window and read by
`latest_summary`. `AskRoutes` does all of this for a page: `history` returns a `History` —
the window as messages, with `.summary` and `.recalled` on it — and `remember` embeds each
turn and calls `summarise` when the subclass returns a `summariser()`.

What it costs, per question, with everything on: one embedding call for the question and
one per turn written (two), one word-index and one vector query, and every eight turns one
more short model call — over eight turns of text plus the previous paragraph, made after the
answer has gone out. Prompt tokens per question are the summary (a paragraph), up to three
recalled turns, the ten-turn window and the question; because the summary sits ahead of
everything that changes per question, it is inside the cached prefix and re-read for free
until it changes. Measured on the fake model: a fact stated at turn one is in front of the
model at turn two hundred twice, once in the summary and once recalled, and the prefix is
identical across turns 193–200. Measure the `cached` share per turn with `ml-stack-bench
concurrent` after changing `EVERY`; a summary that changes too often shows up there.

A fact stated in conversation reaches the *graph* through the change-request path, not
through any of this. The summary and the recall keep it in the model's view for this
thread; only an entry makes the tools find it next time, in any thread.

## Searching the web

The graph's tools see the graph and nothing else. `ml_stack.web` adds the web in the same
`(schema, callable)` shape, so a model can be handed both:

```python
from ml_stack.graph.ask import converse, tools_for
from ml_stack.web import PROMPTS, tools as web_tools

converse(question, graph, client, tools=tools_for(graph) + web_tools())
# and, for routing: rank(question, {**TOOL_PROMPTS, **PROMPTS}, ...)
```

`web_search(query)` returns titles, links and a line each; `web_read(url)` returns one
page as text, cut at a sentence, and falls through to a real browser when the plain fetch
came back thin. `web_tools(vision=True)` adds `web_look(url)` — a full-page screenshot and
the page's largest pictures, returned under `_images` for the ask loop to hand to a model
that can see. The descriptions carry worked examples, for the reason [Measuring](bench.md)
gives.

**A model cannot read the machine it runs on.** `web_read` and `web_look` refuse anything
that is not http(s) and any host that resolves to a loopback, private or link-local
address — `file:`, `localhost`, `127/8`, `10/8`, `192.168/16` and their kin — before a
byte is fetched or a browser navigates. The browser uses its own profile
(`MLSTACK_WEB_PROFILE`, default `~/.ml-stack/web`), never the scraper's signed-in one.

`MLSTACK_SEARCH` picks the engine. `ddgs` (the default; `pip install 'ml-stack[web]'`) is
keyless, fronts several engines, and is rate-limited by them: a refusal comes back to the
model as `{"none": "search unavailable: ..."}` rather than an empty list, so it moves on
instead of asking again. `searxng` is the robust option when the questions are many: a
self-hosted instance at `SEARXNG_URL` with its JSON format enabled, reached through the
stdlib, and nobody rate-limits it but you.

