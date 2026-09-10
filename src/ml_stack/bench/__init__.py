"""What a change to the asking costs and whether it was worth it.

A graph answers questions through a large model, and every tool call it makes is a whole
round trip. Any change to that -- a different prompt, a search run before the model instead
of by it -- has to be shown to be an improvement rather than asserted, on wall clock, on
tokens, and on whether the answers were right. Runs are kept, so two of them can be
compared later.

One package, one command (``ml-stack-bench``). `keep` is where runs are kept and `score`
is what a run is worth; `questions`, `counting`, `measure` and `holding` ask a set of
questions and count what the server spent on them; `serve`, `run` and `options` put a
model up and drive the subcommands; `show`, `detail`, `frontier`, `gathered`, `profiles`
and `report` read the runs back; and `truth`, `folding` and `extract` are the other half
of the bench, reading a graph out of messages.

This module is the namespace: everything is imported here, and every call between the
modules to something a test or `selfcheck` patches -- `served`, `measure`, `footprint`,
`runs`, `home_dir` -- goes through ``bench.<name>`` at call time, so patching it here
patches it everywhere. ``python -m ml_stack.bench`` is what `detach` re-runs.
"""

from __future__ import annotations

import platform  # noqa: F401 - `bench.platform.system` is what the detach tests patch

from ml_stack.bench import backends  # noqa: F401
from ml_stack.bench.askings import (  # noqa: F401
    REACH,
    _asked,
    _askings,
    asking_from,
    halves,
    sampling_from,
    with_card,
)
from ml_stack.bench.backends import (  # noqa: F401
    client_for,
    describe,
    parse_on,
    served_by,
    timings_of,
)
from ml_stack.bench.counting import (  # noqa: F401
    PER_QUESTION,
    TRACE_CAP,
    TRACE_ENV,
    TRACE_TEXT_CAP,
    Counting,
    QuestionTimedOut,
    wants_trace,
)
from ml_stack.bench.detail import (  # noqa: F401
    missed,
    shape,
    transcript,
)
from ml_stack.bench.estimate import (  # noqa: F401
    CEILING_MIN,
    Estimate,
    ModelEstimate,
    ceiling_default,
    estimate,
    span,
)
from ml_stack.bench.frontier import (  # noqa: F401
    AXES,
    pareto,
    plot,
    rates,
)
from ml_stack.bench.gathered import (  # noqa: F401
    ASKINGS,
    across,
    answering,
    asking_of,
    by_model,
    cache_of,
    fit_for,
    model_of,
    recommended_head,
    thinking_of,
)
from ml_stack.bench.holding import (  # noqa: F401
    SAMPLE_EVERY,
    Watching,
    _idle,
    _Peak,
    _rusage_footprint,
    beyond_weights,
    busy,
    footprint,
    footprint_of,
    machine_memory,
    process_tree,
    serving_pids,
    serving_process,
    slot_count,
    watched,
    watching,
)
from ml_stack.bench.keep import (  # noqa: F401
    SHORT,
    SMOKE,
    RunNotKept,
    _commit,
    _kept,
    _plain,
    asked_with,
    empties,
    forget,
    home_dir,
    prepared,
    read_back,
    resumable,
    runs,
    save,
    stamped,
)
from ml_stack.bench.measure import (  # noqa: F401
    _ask_once,
    ask_from,
    asking,
    concurrent,
    finding,
    measure,
)
from ml_stack.bench.ops import (  # noqa: F401
    Refused,
    fleet_jobs,
    kept_for,
    measured_run,
    newest,
    serving_fields,
    swept,
)
from ml_stack.bench.options import checking  # noqa: F401
from ml_stack.bench.progress import (  # noqa: F401
    _latest_log,
    beside_on_the_card,
    note_beside_the_run,
    results_since,
    serving_lines,
    status,
    stop,
    tail,
)
from ml_stack.bench.questions import (  # noqa: F401
    _how_many,
    filed,
    mix,
    read_questions,
    sample,
)
from ml_stack.bench.record import (  # noqa: F401
    Measured,
    Spread,
    of,
    prompt_digest,
)
from ml_stack.bench.report import (  # noqa: F401
    Doc,
    fits_named,
    report,
)
from ml_stack.bench.run import (  # noqa: F401
    COMMANDS,
    HANDED_OVER,
    _estimated,
    _fleet_sweep,
    _main,
    _parser,
    _run,
    _stop_on_sigterm,
    main,
    smoke_first,
    wants_smoke,
)
from ml_stack.bench.score import (  # noqa: F401
    BOOTSTRAP,
    BOOTSTRAP_SEED,
    NOISE,
    Choice,
    Row,
    _exportable,
    _flat,
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
    band,
    bands,
    baseline,
    choices,
    composed,
    derived,
    export,
    half_band,
    held_up,
    host_of,
    hosts_of,
    invented_digest,
    per_question,
    prefix_hits,
    prefix_kept,
    ranking,
    separated,
    speedup,
    unread_named,
    wall_of,
)
from ml_stack.bench.serve import (
    EMBEDDED,  # noqa: F401
    SmokeFailed,
    drafted_by,
    drafts,
    prefetch,
    references_in,
    served,
    smoked,
)
from ml_stack.bench.show import (  # noqa: F401
    SHOWN_FLAGS,
    _shown,
    asked_as,
    at_once,
    band_of,
    by_serving,
    cache_turns,
    compare,
    drafted,
    drafting,
    head_short,
    kv_short,
    measured,
    prefixed,
    sampled,
    served_as,
    serving_of,
    table,
    timeouts,
    wired_of,
)
from ml_stack.bench.underway import (  # noqa: F401
    MEASURING,
    _last_line,
    _named_in,
    detach,
    measuring,
    measuring_file,
)
from ml_stack.graph.vectors import MARGIN, stands_out  # noqa: F401 - imported from here too
from ml_stack.paths import repo_root  # noqa: F401

__all__ = [
    "NOISE",
    "PER_QUESTION",
    "SHORT",
    "SMOKE",
    "Counting",
    "Estimate",
    "Measured",
    "QuestionTimedOut",
    "Row",
    "SmokeFailed",
    "Spread",
    "Watching",
    "ask_from",
    "asked_as",
    "asking",
    "band",
    "bands",
    "baseline",
    "beyond_weights",
    "by_serving",
    "choices",
    "compare",
    "composed",
    "concurrent",
    "detach",
    "drafted",
    "drafted_by",
    "empties",
    "estimate",
    "export",
    "finding",
    "footprint",
    "footprint_of",
    "forget",
    "halves",
    "held_up",
    "home_dir",
    "kv_short",
    "machine_memory",
    "main",
    "measure",
    "measuring",
    "prefetch",
    "prefix_hits",
    "prefix_kept",
    "prepared",
    "prompt_digest",
    "ranking",
    "read_questions",
    "references_in",
    "report",
    "runs",
    "save",
    "separated",
    "served_as",
    "serving_of",
    "serving_process",
    "slot_count",
    "speedup",
    "status",
    "stop",
    "table",
    "tail",
    "transcript",
    "unread_named",
    "wants_trace",
    "watching",
    "wired_of",
]
