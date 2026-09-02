"""What a change to the asking costs and whether it was worth it.

A graph answers questions through a large model, and every tool call it makes is a whole
round trip. Any change to that — a different prompt, a search run before the model instead of
by it — has to be shown to be an improvement rather than asserted, on wall clock, on tokens,
and on whether the answers were right. Runs are kept, so two of them can be compared later.

One package, one command (``ml-stack-bench``), six modules so that each agent is told
"touch only this one":

- `keep`: where runs are kept, `save` and the read-back, `SHORT` and `SMOKE`;
- `score`: F1 per answer, the rates, the baseline a draft head is measured against, the
  ranking and the export;
- `measure`: the questions, `asking`, one question through the client with its bill,
  `measure` and `concurrent`, and what the server costs;
- `show`: the table, `--detail`, `--rates`, the frontier, the plot, the `drafts` summary;
- `serve`: a model put up, asked every way on one load, and taken down;
- `run`: the parser, every subcommand, the lock, `--detach`, `status`/`tail`/`stop`;
- and beside them `extract` (reading a graph out of messages), `selfcheck` (the dry run
  `main` makes before the lock) and `history` (the day, read out of the logs).

This module is the namespace: everything is imported here, and every call between the
modules to something a test or `selfcheck` patches -- `served`, `measure`, `footprint`,
`runs`, `HOME` -- goes through ``bench.<name>`` at call time, so patching it here patches it
everywhere. ``python -m ml_stack.graph.bench`` is what `detach` re-runs.
"""

from __future__ import annotations

import platform  # noqa: F401 - `bench.platform.system` is what the detach tests patch

from ml_stack.graph.bench.keep import (  # noqa: F401
    HOME,
    SHORT,
    SMOKE,
    RunNotKept,
    _kept,
    _plain,
    empties,
    forget,
    prepared,
    read_back,
    resumable,
    runs,
    save,
)
from ml_stack.graph.bench.score import (  # noqa: F401
    NOISE,
    Choice,
    Row,
    _build,
    _exportable,
    _flat,
    _head_of,
    _hit,
    _over_invented,
    _precision,
    _recall,
    _score,
    _times,
    _total,
    _which,
    _with_rates,
    _words,
    baseline,
    choices,
    composed,
    derived,
    export,
    invented_digest,
    per_question,
    ranking,
    speedup,
    unread_named,
    wall_of,
)
from ml_stack.graph.bench.measure import (  # noqa: F401
    PER_QUESTION,
    Counting,
    QuestionTimedOut,
    _Peak,
    _ask_once,
    _how_many,
    _idle,
    ask_from,
    asking,
    beyond_weights,
    busy,
    concurrent,
    finding,
    footprint,
    measure,
    read_questions,
    sample,
    slot_count,
)
from ml_stack.graph.bench.show import (  # noqa: F401
    AXES,
    _shown,
    at_once,
    compare,
    drafted,
    drafting,
    kv_short,
    made,
    missed,
    pareto,
    plot,
    rates,
    sampled,
    shape,
    table,
    timeouts,
)
from ml_stack.graph.bench.serve import (  # noqa: F401
    SmokeFailed,
    drafts,
    find_model,
    prefetch,
    references_in,
    served,
    smoked,
)
from ml_stack.graph.bench.run import (  # noqa: F401
    MEASURING,
    _asked,
    _commit,
    _last_line,
    _latest_log,
    _main,
    _named_in,
    _parser,
    _run,
    _stop_on_sigterm,
    _ways,
    checking,
    detach,
    halves,
    main,
    measuring,
    measuring_file,
    sampling_from,
    smoke_first,
    status,
    stop,
    tail,
    wants_smoke,
    with_card,
)
from ml_stack.graph.vectors import MARGIN, stands_out  # noqa: F401 - imported from here too
from ml_stack.paths import repo_root  # noqa: F401

__all__ = ["Counting", "HOME", "NOISE", "PER_QUESTION", "QuestionTimedOut", "Row", "SHORT",
           "SMOKE", "SmokeFailed", "baseline", "beyond_weights", "choices", "composed",
           "drafted", "export", "ranking", "ask_from", "asking", "compare", "concurrent",
           "detach", "empties", "finding", "footprint", "forget", "halves", "kv_short", "main",
           "measure", "measuring", "prefetch", "prepared", "read_questions", "references_in",
           "runs", "save", "slot_count", "speedup", "status", "stop", "table", "tail",
           "unread_named"]
