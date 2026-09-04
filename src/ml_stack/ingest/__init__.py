"""Documents into a graph: a shelf of textbooks read section by section into one store.

`ml_stack.sources.pdf` turns a book into units an extractor can take. This is the other
half: each unit through `Client.extract` against `contracts/extraction-document.schema.json`,
the extractions folded into one graph per book (`ml_stack.entities.fold`, so a relation the
model spelled two ways is one edge), and nodes, edges and the raw extractions written into a
`GraphStore`. Every node and edge carries where it was read from -- book, chapter, section,
page -- because a claim in a knowledge graph with no page behind it is a claim nobody can
check.

Three things here are not obvious and all three were paid for elsewhere in this repo:

*A run is hours, so it is resumable and it detaches.* A progress file beside the store
records every unit that finished; `--resume` skips those, `--detach` re-runs the command in
its own session with a log under ``~/.ml-stack/ingest/logs`` (a child of a shell dies with
the shell, and a ranking sweep was killed that way thirty minutes in), and
``ml-stack-ingest status`` says how many sections of how many books are done and at what
rate.

*What it cost is on record, per section.* Each extraction keeps a `ml_stack.telemetry.Call`
and the run keeps their `Spent`, so "the ten books took nine hours" can be broken down
into which book, which section, and how much of it was prompt.

*Whether it does a good job is measured, not asserted.* ``--gold FILE`` runs a set of
passages with known triples through the same extraction and scores recall and precision,
matching subjects and objects through their aliases and `entities.close` and predicates
through theirs, and lists what was missed. ``--fail-under`` makes that a gate.

*A half-read book is readable.* Each unit's extraction lands in
``<store>.<slug>.reads.json`` as it finishes, and the book so far is folded into the store
as the run goes -- see `FOLD_EVERY` -- so a shelf that will take days can be asked
questions today. `Shelf` is how an application reads one::

    shelf = Shelf("./shelf.ladybug")
    for book in shelf.books():
        print(book.slug, book.units, "of", book.wanted, "partial" if book.partial else "")
    graph = shelf.graph("velthorne-open-texts")     # folded from the reads, no store needed
    with shelf.store() as store:                    # read-only, beside the running writer
        store.nodes(kind="concept")

``ml-stack-ingest fold --out STORE`` does the same fold from the shelf into the store on
demand, ``show`` prints what a book holds, ``shelf`` prints what the books hold together --
the concepts more than one of them names and the relations joining their vocabularies, see
`Shelf.shared` -- and ``stop`` ends a detached run after folding what it has read.

*A shelf is asked questions the way any graph is.* ``ml-stack-ingest ask --out STORE
"question"`` puts the store's graph through `ml_stack.graph.ask.converse` with the model
served in its measured shape, and ``ask --gold FILE`` scores a set of questions against the
entries each answer should have selected, using the bench's own scorer.

Nothing here is about any one book: it reads a PDF, it asks a model, it writes a graph.

The modules: `reads` (a unit's `Read`, and the files beside the store), `extract` (one
section through the model), `fold` (extractions into a graph, and into the store),
`progress` (how far a run has got), `shelf` (what has been read), `judge` (the run record
and the judge a fold hands close spellings to), `gold` (the extraction scored), `ask` (the
shelf asked questions), `serving` (the model a run reads with), `run` (the read run) and
`cli` (the command). Everything a caller needs is re-exported here.
"""

from ml_stack.ingest.ask import (
    ask as ask,
    asked_f1 as asked_f1,
    asked_lines as asked_lines,
    graph_of as graph_of,
    read_asked as read_asked,
    score_asked as score_asked,
    spent_line as spent_line,
    _ids_for as _ids_for,
)
from ml_stack.ingest.cli import (
    HOME as HOME,
    KIND as KIND,
    STOP_WAIT as STOP_WAIT,
    detach as detach,
    main as main,
    parser as parser,
    retry as retry,
    stop as stop,
    wait as wait,
    _WINDOWS_DETACHED as _WINDOWS_DETACHED,
    _ask_run as _ask_run,
    _gold_run as _gold_run,
    _out_of as _out_of,
    _parsed as _parsed,
    _recorded_alive as _recorded_alive,
)
from ml_stack.ingest.extract import (
    IMAGES_PER_SECTION as IMAGES_PER_SECTION,
    INSTRUCTIONS as INSTRUCTIONS,
    PER_SECTION as PER_SECTION,
    VERBS as VERBS,
    WITH_IMAGES as WITH_IMAGES,
    extract_unit as extract_unit,
    prompt_for as prompt_for,
    schema as schema,
    _Recording as _Recording,
)
from ml_stack.ingest.fold import (
    build as build,
    fold as fold,
    fold_book as fold_book,
    fold_into as fold_into,
    plurals as plurals,
    write as write,
    _apply as _apply,
    _drop_book as _drop_book,
    _missing_from as _missing_from,
    _texts_of as _texts_of,
    _unit_docs as _unit_docs,
)
from ml_stack.ingest.gold import (
    INVERSES as INVERSES,
    Scored as Scored,
    gold_lines as gold_lines,
    gold_score as gold_score,
    read_gold as read_gold,
    sayable as sayable,
    vocabulary as vocabulary,
    _matches as _matches,
    _names as _names,
    _passage_unit as _passage_unit,
    _same as _same,
)
from ml_stack.ingest.judge import (
    located as located,
    origin as origin,
    run_record as run_record,
    sources_for as sources_for,
    write_run as write_run,
    _judge as _judge,
)
from ml_stack.ingest.progress import (
    GIVE_UP as GIVE_UP,
    Progress as Progress,
    status as status,
    _book_in_store as _book_in_store,
    _folded_at as _folded_at,
    _for_long as _for_long,
)
from ml_stack.ingest.reads import (
    Read as Read,
    reads_path as reads_path,
    tokens_of as tokens_of,
    unit_of as unit_of,
    units_of as units_of,
    _Unit as _Unit,
    _keep_reads as _keep_reads,
    _read_json as _read_json,
    _slug as _slug,
    _write_json as _write_json,
)
from ml_stack.ingest.run import (
    FOLD_EVERY as FOLD_EVERY,
    FOLD_SECONDS as FOLD_SECONDS,
    Stopped as Stopped,
    read_unit as read_unit,
    _call_of as _call_of,
    _fold_interval as _fold_interval,
    _read_run as _read_run,
    _rows as _rows,
    _stopping as _stopping,
    _time_to_fold as _time_to_fold,
)
from ml_stack.ingest.serving import (
    _alive as _alive,
    _find_model as _find_model,
    _run as _run,
    _serving as _serving,
    _serving_said as _serving_said,
)
from ml_stack.ingest.shelf import (
    Book as Book,
    Shelf as Shelf,
    shelf as shelf,
    show as show,
    _decisions_in as _decisions_in,
    _label as _label,
    _run_said as _run_said,
)

__all__ = ["FOLD_EVERY", "FOLD_SECONDS", "HOME", "INSTRUCTIONS", "PER_SECTION", "VERBS",
           "Book", "Progress", "Scored", "Shelf", "Stopped", "ask", "asked_lines", "build",
           "detach", "extract_unit", "fold", "fold_book", "fold_into", "gold_score",
           "graph_of", "main", "read_asked", "read_gold", "sayable", "schema", "score_asked",
           "shelf", "vocabulary",
           "show", "status", "unit_of", "units_of", "write"]
