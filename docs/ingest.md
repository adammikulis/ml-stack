# Documents into a graph

A book is not a document to a model. A thousand-page PDF has no prompt that fits it, and the
obvious cut -- a page, a fixed number of characters -- goes through the middle of a
definition, so the sentence that says what a thing *is* arrives without the name it defines.
The unit that survives being read alone is a **section**: the book itself decided it was one
idea, it names itself, and it carries its own figures.

```sh
ml-stack-ingest textbook.pdf --out ./sources.ladybug --model Qwen3.8-Flash-Next --chapter 2
ml-stack-ingest ~/texts/*.pdf --out ./sources.ladybug --model Qwen3.8-Flash-Next --resume --detach
ml-stack-ingest status  --out ./sources.ladybug     # how far, what failed, how long is left
ml-stack-ingest show    --out ./sources.ladybug     # what each source was read as
ml-stack-ingest sources --out ./sources.ladybug     # what every source holds together
ml-stack-ingest fold    --out ./sources.ladybug     # every source so far into the store
ml-stack-ingest import ./extraction --out ./sources.ladybug --dry-run   # a pair somebody else extracted
ml-stack-ingest ask     --out ./sources.ladybug "how is heart rate controlled"   # ask the store
ml-stack-ingest ask     --out ./sources.ladybug --gold ./questions.json  # score the answers, not the reading
ml-stack-ingest retry   --out ./sources.ladybug   # read again the units that gave up
ml-stack-ingest tidy    --out ./sources.ladybug   # the hygiene pass; --apply to write it
ml-stack-ingest migrate --out ./sources.ladybug   # a store from the books era becomes a sources store
ml-stack-ingest stop                              # end the run, after it folds what it read
```

`ml_stack.sources.pdf` does the reading. `read(path)` gives a `Document` of `Chapter`s of
`Section`s: a publisher's PDF carries an outline (`doc.get_toc()`) and that is believed, and
a book printed to PDF by a browser has none, so the headings are found by the way they are
set -- a section heading is numbered `N.M` and set larger than the body, a chapter opens with
`CHAPTER N` above the largest line on its page. Which reading was used is on the document as
`how`, because "the sections look wrong" is answered by knowing which. The text is cleaned
the way reading it aloud would clean it: a word broken across a line is put back together,
and the running head, the running foot and the page number are dropped -- found by
*repetition*, a margin line that says almost the same thing on a fifth of the pages, rather
than by matching any particular wording. The two things worth keeping that are not prose are
kept and labelled: a figure's caption, as `[Figure 2.9] ...`, and the terms the book sets in
bold. `units()` is the last cut, splitting a section over ~2,500 tokens on paragraph
boundaries and never inside one. `is_openstax()` reads the licence page, because a file
renamed by whoever downloaded it says nothing.

`ml-stack-ingest` is the other half. Each unit goes through `Client.extract` against
`contracts/extraction-document.schema.json` -- concepts with a kind and a one-line definition
*in the book's words or empty*, relations whose verb phrase is one of nineteen glossed core
ones or, where none of those says what the page says, one the model names itself in the same
snake_case shape, what each figure shows and which concepts it illustrates, and the key terms
-- and the extractions are folded into one graph per source with `entities.fold`, so
`has_part` and `haspart` are one relationship and a plural folds into the spelling the
source uses more. The core verbs and kinds are what every source shares; a verb from
outside them is marked `extension` on its edge and a kind `extension_kind` on its node --
the same mark an imported predicate carries -- so a source's own vocabulary can be read
back. `ml-stack-ingest sources` prints it, and `--core-only` reads a source with the core
lists and nothing else.

The vocabulary grows as the store is read into. What earlier sections named for themselves
is kept as `ingest:vocabulary`, with a use count and the unit that first said it, and the
forty most-used are named in the next section's prompt so the reader takes a word already
in use rather than coining a third spelling of it. It goes in the prompt and never in the
schema, so a store that has grown a word does not change the shape earlier reads were
answered under: the extraction cache and `--resume` are untouched by it. Nodes
and edges go into one `GraphStore`, every one of them *pointing at* the units it was read
from -- `provenance` is unit ids and nothing else, the unit document holds the source,
chapter, section and pages, and points in turn at the hidden `run` node that read it: the
model, its build and head, sampling, the schema and instructions hashes, the version, the
host, when. `located()` and `origin()` walk the pointers back to a page and a model, so a
claim in a knowledge graph always has a page and a model behind it, without a string
copied onto every node. Each extraction's `ml_stack.telemetry.Call` is kept, so "the run
took nine hours" breaks down into which source, which section and how much of it was prompt.

The model is served the way the bench serves one: `--model` takes a lease for the whole run
in the shape its profile measured (`--no-profile` serves it bare), or `--base-url` uses a
server that is already up. `--images` hands the model each section's rendered figures as
pictures rather than only their captions -- the `_images` convention `graph.ask` uses -- and
without a projector the captions are all it gets, which it says rather than pretending
otherwise. A run is hours, so `--detach` runs it in its own session with a log under
`~/.ml-stack/ingest/logs`, a progress file beside the store records every unit that finished,
`--resume` skips those, and `status` says how many sections of how many sources are done, at
what rate, what is in the store, and how long the rest will take.

## A graph somebody else already extracted

`ml-stack-ingest import DIR --out STORE` takes a `nodes.csv`/`edges.csv` pair another
extractor wrote and puts it in the store as one source. The pair becomes this library's own
reads -- one per section, in the document schema's shape -- and goes in through the same
fold a read source does, so it has the same node and edge shape, the same `source:<slug>`
and `read_from` edges, the same `<source>:<chapter>:<section>` unit ids behind every claim,
and `sources`, `show`, `tidy` and `ask` work over it unchanged. The model and run id on the
rows become a `run` node the units point at, so `origin()` still says which model said this.

The two vocabularies are joined by a table. This library sets nineteen verbs itself; an
extractor free to choose its own writes thousands -- one anatomy textbook carries 2,110
distinct predicates. A predicate with a counterpart is normalised onto that verb, subject
and object swapped where the natural reading is the inverse: `includes` is `has_part`,
`defines` is `defined_by`, `enables` is `requires`, `caused_by` is `causes` the other way
round. Every other predicate comes in as it stands, its edges carrying `extension`, so a
reader and a query can tell a verb this library set from a verb the extractor chose;
`--core-only` takes the first kind alone. What each predicate became is counted by name,
printed, and kept in the store as `ingest:predicates:<source>`.

Some of what an open vocabulary writes stands in for a relation rather than being one.
`related_to`, `describes` and `supports` mostly say the two things were named near one
another -- 3,861 of the anatomy textbook's 21,922 relations, under 92 predicates that
`ingest.VAGUE` lists. Mostly, not always: `associated_with` is the exact claim an
epidemiological source means, and reading it as `causes` would say more than the source
did. Telling the deliberate hedge from the shrug takes the passage, and an import has no
passage -- it has another extractor's output, where the two look alike. So those relations
are counted by name and not carried across, and `--keep-vague` takes them anyway, their
edges marked `vague` beside `extension`. What is carried across either way is the
`has_ir_absorption_at` kind of edge, which says something real this library has no verb
for. `--dry-run` prints all of it
and writes nothing; `--confidence` takes rows at a level and above, `--no-provisional`
leaves the ones the extractor left provisional, and `--slug` names the source.

## A source is readable before it is finished

A few thousand sections is days at eighty-odd seconds a section, and a source that is only
in the store once it is finished is one nobody can ask about until then. Each unit's
extraction lands in `<store>.<slug>.reads.json` the moment it comes back, and the source so
far is folded and written into the store as the run goes -- at a chapter's end once
twenty-five sections have gone by since the last fold, and inside a chapter longer than
fifty. Writing a source is an upsert and nothing more -- a node the store lacks is added,
one it has takes the fold's mentions, aliases, definition and provenance, an edge likewise,
and nothing is merged or removed: a knowledge graph is updated by adding to it. Joining
duplicates is a separate pass (`ml_stack.graph.tidy`), and `fold --rebuild` -- the source's
own nodes and edges out, then the full fold from its reads -- is the one path that removes
anything, for after a fix that changed what a read means. `fold --dry-run` says what a fold
would add and writes nothing.

The interval is a measured cost rather than a formality. The fold is `entities.fold`
comparing every concept name against every other, so it grows with the square of the
vocabulary: 400 invented sections of a twelve-word vocabulary fold and write in 3.8 s, and
300 sections of a 2,700-word one take 44 s to fold and 9 s to write.

`ml-stack-ingest fold --out STORE [--source SLUG]` does the same from the reads on demand,
and is idempotent. `show` prints what each source was read as -- concepts with their kind
and definition, relations with their verb and the page behind them, the spellings and
plurals the fold joined, how many figures -- and says which sources are partial. `sources`
prints them together: what the store holds for each, the concepts more than one of them
names, the names `tidy` joined across sources, and the relations joining one source's
vocabulary to another's. `stop` ends a detached run: it raises inside the section being
read, folds the source so far, and exits, and the command waits for it and says whether the
fold landed. `ask` questions the store with the same tools a page asks with:
`ask --out STORE "question"` answers from the sources read so far, and `ask --gold FILE`
runs a set of questions -- `{"question", "expected": [ids or labels]}` -- and prints F1 the
way the bench prints it, so *answerable* is a number and `--fail-under` gates that too.
`retry` re-reads the units `status` counts as given up, after the fix that should save
them; `tidy` is the hygiene pass over the store -- dry unless `--apply`, and it refuses to
run beside a detached ingest, because one job is on the GPU.

`Sources` is the same thing for an application:

```python
from ml_stack.ingest import Sources

view = Sources("./sources.ladybug")
for one in view.sources():
    print(one.slug, one.read, "of", one.wanted, "partial" if one.partial else "")

graph = view.graph("velthorne-open-texts")   # folded from the reads: no store, no PDF
with view.store() as store:                  # read-only, beside the running writer
    store.nodes(kind="concept")
```

A unit that failed contributes nothing to the fold -- what a cut-off reply wrote is kept for
reading, not for believing -- and every file beside the store is written through a rename,
so a kill in the middle of one leaves the file that was there.

## Whether it does a good job

```sh
ml-stack-ingest --gold tests/fixtures/extraction-gold.json --model Qwen3.8-Flash-Next --fail-under 0.7
```

`--gold FILE` is the measurement, not an opinion. The file holds passages with the triples
they state -- `{"subject", "predicate", "object"}` and, for each, the other names a right
answer may use -- and every passage goes through *the same* extraction a source is read
with: the same prompt, the same schema, the same sampling. Subjects and objects are matched
through their aliases and `entities.close`, predicates through theirs, and what comes back is
recall, precision, F1 and every triple that was missed or invented, listed. Aliases are the
point: an extractor writing the singular where the gold writes the plural is right, and a
scorer without them reports a failure that is not one. `--fail-under` turns the score into a
gate.

`tests/fixtures/extraction-gold.json` is the shipped set: twenty invented textbook passages,
three to six sentences each, with every relation each one states written down as a triple in
the core vocabulary -- every core verb, and each verb with an inverse at least twice. With everything a passage states listed, precision measures the model: a
relation it says that the passage does not is an invention. A triple matches as written or
the other way round through `ingest.INVERSES`, so `sheath has_part gland` and `gland part_of
sheath` are the same fact.

