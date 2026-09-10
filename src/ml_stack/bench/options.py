"""Every ``ml-stack-bench`` subcommand's flags, built when a parser asks for them.

One function per subcommand, plus the four sets more than one takes: `asking_options` is
what a measuring command is asked with, `reaching_options` how much one tool result may
carry, `measuring_options` what every command that holds the GPU takes, and `checking` the
checks made before a measurement starts. The defaults name this machine's bench home, so
each set is built at the moment a parser is.
"""

from __future__ import annotations

import argparse

# The package is the namespace the tests and `selfcheck` patch -- `bench.home_dir()` -- so
# anything patchable is looked up there at call time, never bound here at import.
from ml_stack import bench
from ml_stack.bench.askings import REACH
from ml_stack.bench.estimate import ceiling_default
from ml_stack.bench.keep import SHORT, SMOKE
from ml_stack.bench.measure import PER_QUESTION
from ml_stack.bench.score import NOISE
from ml_stack.command import Option, flag
from ml_stack.graph.vectors import MARGIN

__all__ = ["animate_options", "asking_options", "checking", "concurrent_options",
           "drafts_options", "forget_options", "measuring_options", "prepare_options",
           "queue_options", "reaching_options", "report_options", "run_options",
           "show_options", "sweep_options", "tail_options", "wait_options"]



def checking() -> tuple[Option, ...]:
    """The flags every measuring subcommand has for the checks it makes before it measures:
    the self-check on no GPU, the estimate against the ceiling, and the smoke on the real
    one."""
    return (
        flag("--ceiling", type=float, default=ceiling_default(), metavar="MINUTES",
             help="refuse to start when the estimate -- seconds per question from "
                  "the runs kept of each model, else a guess from its weights, "
                  "times the questions, the askings and the models, plus a load each "
                  "-- is over this many minutes, unless --yes (default: "
                  "%(default)s, or MLSTACK_BENCH_CEILING). A --smoke run is never "
                  "refused. No more eight-hour tests"),
        flag("--yes", action="store_true",
             help="run it even when the estimate is over --ceiling"),
        flag("--no-selfcheck", action="store_true",
             help="skip the dry run made first, before the lock is taken: this exact "
                  "command through the whole path with a scripted model, no server "
                  "and no GPU, into a scratch store read back -- what catches a flag "
                  "the client does not take before a load is paid for it. For a run "
                  "you are deliberately repeating, whose path the last one proved. "
                  "Read before the rest of the line is parsed, like --no-queue"),
        flag("--no-smoke", action="store_true",
             help=f"skip the {SMOKE}-question smoke a real run makes first on the "
                  f"real server and the real store, read back, before its own "
                  f"questions. Without this every run that is not itself --smoke "
                  f"smokes first -- on the same load, where the model is served -- "
                  f"and a smoke that fails ends the run before anything else starts"),
    )


def asking_options() -> tuple[Option, ...]:
    """The flags `run`, `sweep` and `concurrent` share: what is asked, how many and how."""
    return (
        flag("--trace", action=argparse.BooleanOptionalAction, default=None,
             help="keep each question's transcript -- every call with its arguments, "
                  "what came back and what it cost -- beside the totals, for "
                  "`show --trace` and `ml-stack-train-tools from-bench`. Default: on "
                  f"at {SHORT} questions or fewer, off for a full run, where the "
                  "transcripts are tens of megabytes in a store nothing backs up"),
        flag("--sample", type=int, default=0, metavar="N",
             help="ask only N of the questions, keeping every kind of answer. "
                  "For a comparison where accuracy is not the variable -- draft "
                  "heads, sampling, serving flags -- most of a full run is "
                  "spent re-establishing a score that cannot move"),
        flag("--short", dest="short", action="store_true",
             help=f"the same as --sample {SHORT}: every kind still asked about "
                  f"and the same mean number of answers expected, at about half "
                  f"the time. Each question is worth more, so a small difference "
                  f"is noise on a short run and signal on a full one"),
        flag("--smoke", action="store_true",
             help=f"ask only {SMOKE} questions, to prove the whole path works "
                  f"before spending the GPU on it. Serving, asking, scoring, "
                  f"measuring and saving all happen, so anything that would "
                  f"raise at the end raises here instead. The score means "
                  f"nothing at this size -- run it first, then run it properly"),
        flag("--temperature", type=float, default=None,
             help="override the sampling temperature; the default is whatever "
                  "the model's own card asks for (gemma-4: 1.0)"),
        flag("--top-p", type=float, default=None, help="override top_p"),
        flag("--top-k", type=int, default=None, help="override top_k"),
        flag("--min-p", type=float, default=None, help="override min_p"),
        flag("--n-predict", type=int, default=16384,
             help="tokens each turn may write -- thinking, tool calls and the "
                  "answer together. A thinking model spends most of a turn "
                  "reasoning, so a low ceiling truncates the answer rather than "
                  "the thought (default: %(default)s)"),
        flag("--anyway", action="store_true",
             help="measure even when the server is already busy; the wall clock "
                  "will then be two runs sharing a GPU, not one run"),
        flag("--card", action="store_true",
             help="ask with what the model's own card recommends, to see whether "
                  "it suits this task -- it is not what a client sends otherwise"),
    )


def reaching_options() -> tuple[Option, ...]:
    """The flags `run` and `sweep` share: how much one tool result carries, and each --also."""
    return (
        flag("--reach", type=int, default=0, metavar="TOKENS",
             help="how much one tool result may carry, in tokens, on every way "
                  "asked. Off by default, which is the flat character cut every "
                  "run so far measured. With one, look_at, look_around and "
                  "list_kind pack whole entries with their quotes up to it "
                  "instead of stopping at a fixed count -- for a model whose "
                  "context is cheap and whose reading is eleven times faster "
                  "than its writing, which is what makes fewer, fatter calls "
                  f"the cheaper question (--also reach uses {REACH})"),
        flag("--rounds", type=int, default=0, metavar="N",
             help="how many tool-calling turns one question may spend before "
                  "it must answer, on every way asked (converse's `rounds`; "
                  "unset is the library default). A three-tool offer and a "
                  "one-entry-at-a-time read both want more of them and a "
                  "batched read wants fewer, so this is measured beside "
                  "--also few and --also single rather than fixed for all"),
        flag("--also", action="append", default=[],
             choices=("terse", "card", "greedy", "rich", "tight", "loose",
                      "reach", "batch", "kinds", "summary", "single", "few"),
             help="ask the same served model another way as well. Whether the "
                  "tools are described briefly, what sampling is used, "
                  "whether look_up says why it matched (rich), and whether "
                  "show is told to name what the answer is about with no cap "
                  "(loose -- the old asking, kept as a control against the "
                  "tight one every run uses now), and how much one tool result "
                  "may carry (reach -- fat results and look_around, for a model "
                  "that reads faster than it writes), whether every lookup is "
                  "asked for in one turn (batch -- fewer rounds, which is where "
                  "the wall clock goes), whether show keeps only the kind the "
                  "question asked for (kinds -- a who question is answered by "
                  "people, not by the topic they share) and whether the whole "
                  "graph can be read at a glance (summary -- for the broad "
                  "question no search reaches), whether every read takes one "
                  "entry and more turns (single -- batch turned around, for a "
                  "model that loses the thread of a long result) and whether "
                  "only three tools are offered (few -- look_up, look_at and "
                  "show, for a model whose tool choice degrades with the "
                  "number of schemas) are questions about the "
                  "asking, not the serving, so ten of them cost one load "
                  "rather than ten. There is no one asking every model wants: "
                  "measure them per model, and `report --profile` writes the "
                  "winner into that model's record. Repeatable"),
    )


def measuring_options() -> tuple[Option, ...]:
    """The flags every command that holds the GPU takes: the per-question cap, the queue,
    the background and the prefetch."""
    return (
        flag("--per-question", type=float, default=PER_QUESTION,
             metavar="SECONDS",
             help="the most one question may take before it is recorded as "
                  "timed out -- no answer, scored wrong, the cap as its wall "
                  "clock -- and the next is asked (default: %(default)s). The "
                  "table counts them under t/o and --detail names them. "
                  "Measured: a 26B thinking model spent 505 s on one question "
                  "under a 16k ceiling, and a run that waits for that is not "
                  "a run"),
        flag("--no-queue", action="store_true",
             help="fail at once if another measurement holds the GPU, rather "
                  "than queue behind it. Read before the rest of the line is "
                  "parsed, and listed here so that --help says it exists"),
        flag("--detach", action="store_true",
             help="run this in the background, owned by nobody's terminal: the "
                  "command re-runs itself in a new session with its output in "
                  f"a log under {bench.home_dir() / 'logs'}, prints the log's path and "
                  "returns at once. `status` says what is measuring, `tail -f` "
                  "follows the log, `stop` ends it and takes its server down. "
                  "Read before the rest of the line is parsed, like --no-queue"),
        flag("--no-prefetch", action="store_true",
             help="do not download the hf: models and heads named here before "
                  "the measuring lock is taken. Without this every reference "
                  "is fetched first, one line each with its size, because a "
                  "download inside the timed window is a timing of the network"),
    )


def run_options() -> tuple[Option, ...]:
    """``run``: ask every question once and keep what it cost."""
    return (
        flag("label", help="what this run is, e.g. with-shortlist"),
        flag("--kept", default=str(bench.home_dir() / "runs.ladybug"),
             help="where to keep the run (default: %(default)s)"),
        flag("--base-url", default="http://127.0.0.1:8080",
             help="the model answering (default: %(default)s)"),
        flag("--graph", default="",
             help="a graph as JSON (default: the invented community that ships here)"),
        flag("--questions", default="",
             help="one per line: a question, or {\"q\":..., \"expect\":[ids]} "
                  "(default: the ones that go with the invented community)"),
        flag("--shortlist", type=int, default=0, metavar="N",
             help="hand the model the N likeliest entries before it starts, found by "
                  "search rather than by asking it to look (default: 0, it looks)"),
        flag("--store", default=bench.prepared(),
             help="a graph store with the word index and vectors: look_up searches it "
                  "as the application does, and --shortlist reads it first "
                  "(default: what `prepare` built, when it has been)"),
        flag("--embed-url", default="",
             help="a server that embeds, for --shortlist to search by meaning"),
        flag("--embed-model", default="", help="the model that embedded the graph"),
        flag("--margin", type=float, default=MARGIN,
             help="how far the best match must stand above the rest before a "
                  "shortlist is worth handing over (default: %(default)s)"),
        flag("--ask", default="",
             help="module:function taking (question, client) — for asking some other "
                  "way than the ordinary one"),
        flag("--client", default="",
             help="module:function returning the model client, instead of --base-url"),
        *asking_options(), *reaching_options(), *measuring_options(), *checking(),
    )


def drafts_options() -> tuple[Option, ...]:
    """``drafts``: one model served with each draft head in turn."""
    return (
        flag("model", help="the model to serve: a name, a path, or an hf: "
                              "reference. A name is looked up on this machine"),
        flag("--draft", action="append", default=[], metavar="PATH",
             help="a draft head to measure; repeat for each. Pass '' for the "
                  "baseline with no draft, which every other row must beat"),
        flag("--reasoning-budget", type=int, default=None, metavar="N",
             help="tokens the model may spend thinking on each turn, on every arm "
                  "(llama-server --reasoning-budget; 0 turns thinking off, -1 is "
                  "unlimited). A head and no thinking is the serving worth "
                  "measuring together, since drafting pays most where the tokens "
                  "are. Every label ends -rbN"),
        flag("--n-max", action="append", type=int, default=[], metavar="N",
             help="how many tokens a head guesses ahead per pass "
                  "(--spec-draft-n-max); repeat to measure each, labelled "
                  "draft:<head>@nN. Without it, once at the build's own default"),
        flag("--server-per-depth", action="store_true",
             help="load the model again for each --n-max, rather than asking one "
                  "load for each depth. A build carrying the per-request "
                  "speculative fields is asked per depth unless this says not to"),
        flag("--serve-kv", default="", metavar="TYPE",
             help="how the served model's KV cache is stored (q8_0 unless said; "
                  "f16, q4_0); a label ends -kv-TYPE for anything but q8_0 and "
                  "the table's ctx column shows it"),
        flag("--port", type=int, default=8099),
        flag("--context", type=int, default=32768),
        flag("--parallel", type=int, default=1, metavar="N",
             help="slots for a --serve'd model (default: %(default)s)"),
        flag("--binary", default="", help="a llama-server that reads this model"),
        flag("--kept", default=str(bench.home_dir() / "runs.ladybug")),
        flag("--questions", default=""),
        flag("--store", default=bench.prepared(),
             help="a graph store with the word index and vectors: look_up searches "
                  "it as the application does, so a head is measured against the "
                  "look_up the ranking measures (default: what `prepare` built, "
                  "when it has been)"),
        flag("--embed-url", default="",
             help="a server that embeds, for look_up to search by meaning"),
        flag("--embed-model", default="", help="the model that embedded the graph"),
        flag("--sample", type=int, default=SHORT, metavar="N",
             help="how many questions to ask each head (default: %(default)s). "
                  "A draft cannot change an answer -- the large model verifies "
                  "every token -- so what is being measured is acceptance and "
                  "wall clock, and a full run spends most of itself proving a "
                  "score that must come out the same"),
        flag("--smoke", action="store_true",
             help=f"ask only {SMOKE} questions of each head, to prove the whole "
                  f"path -- serve, ask, save, and read the run back -- before "
                  f"spending the GPU on it"),
        *measuring_options(), *checking(),
    )


def concurrent_options() -> tuple[Option, ...]:
    """``concurrent``: N conversations of T turns each at once."""
    return (
        flag("label", help="what this run is, e.g. e2b-4x3"),
        flag("--conversations", type=int, default=4, metavar="N",
             help="how many conversations are in flight together (default: "
                  "%(default)s). More than the server has slots, and the turns "
                  "queue -- which is the thing worth measuring"),
        flag("--turns", type=int, default=3, metavar="T",
             help="how many questions each conversation asks in turn, the earlier "
                  "ones carried (default: %(default)s)"),
        flag("--kept", default=str(bench.home_dir() / "runs.ladybug"),
             help="where to keep the run (default: %(default)s)"),
        flag("--base-url", default="http://127.0.0.1:8080",
             help="the model answering (default: %(default)s)"),
        flag("--graph", default="",
             help="a graph as JSON (default: the invented community that ships here)"),
        flag("--questions", default="",
             help="one per line, as for run (default: the invented community's)"),
        flag("--store", default=bench.prepared(),
             help="a graph store with the word index and vectors, so look_up "
                  "searches as the application does (default: what `prepare` "
                  "built, when it has been)"),
        flag("--embed-url", default="", help="a server that embeds, for the store"),
        flag("--embed-model", default="", help="the model that embedded the graph"),
        flag("--client", default="",
             help="module:function returning the model client, instead of --base-url"),
        *asking_options(), *measuring_options(), *checking(),
    )


def prepare_options() -> tuple[Option, ...]:
    """``prepare``: a graph in a store, indexed and embedded."""
    return (
        flag("--store", default=str(bench.home_dir() / "graph.ladybug"),
             help="the store to build (default: %(default)s)"),
        flag("--graph", default="",
             help="a graph as JSON (default: the invented community)"),
        flag("--embed-url", default="",
             help="a server that embeds; without one only the word index is built"),
        flag("--embed-model", default="", help="what to file the vectors under"),
        flag("--mix", action="store_true",
             help="print how many questions ask for each kind of answer, and "
                  "build nothing: what a full run measures, and what a short one "
                  "draws from"),
        flag("--questions", default="",
             help="the set --mix reports on (default: the invented community's)"),
    )


def sweep_options() -> tuple[Option, ...]:
    """``sweep``: every model, with and without a shortlist."""
    return (
        flag("--on", action="append", metavar="NAME=URL", default=[],
             help="a model to measure, e.g. e4b=http://127.0.0.1:8083; repeatable"),
        flag("--kept", default=str(bench.home_dir() / "runs.ladybug"),
             help="where to keep the runs (default: %(default)s)"),
        flag("--graph", default="", help="a graph as JSON (default: the invented one)"),
        flag("--questions", default="", help="(default: the ones that go with it)"),
        flag("--shortlist", type=int, default=8, metavar="N",
             help="how many to hand over in the second run (default: %(default)s)"),
        flag("--store", default=bench.prepared(),
             help="the indexed and embedded graph, for look_up and the shortlist "
                  "(default: what `prepare` built, when it has been)"),
        flag("--embed-url", default="", help="a server that embeds, for --shortlist"),
        flag("--embed-model", default="", help="the model that embedded the graph"),
        flag("--margin", type=float, default=MARGIN),
        flag("--serve", action="append", default=[], metavar="MODEL",
             help="a model to put up, measure and take down again, one at a time: "
                  "a name, a path, or an hf: reference. A name is looked up on "
                  "this machine. "
                  "Repeat for each. Without this, --on measures servers somebody "
                  "else started -- which leaves the starting, stopping and waiting "
                  "to a shell loop that dies with its terminal"),
        flag("--context", type=int, default=0, metavar="N",
             help="total context for a --serve'd model (default: 32768 per slot)"),
        flag("--parallel", type=int, default=1, metavar="N",
             help="slots for a --serve'd model (default: %(default)s)"),
        flag("--serve-port", type=int, default=8099,
             help="the port each served model gets (default: %(default)s)"),
        flag("--serve-draft", action="append", default=[], metavar="PATH_OR_AUTO",
             help="a draft head for the matching --serve, positionally; 'auto' "
                  "finds the one shipped with it, '' serves without"),
        flag("--binary", default="", metavar="PATH",
             help="a llama-server that reads these models"),
        flag("--serve-kv", default="", metavar="TYPE",
             help="how each --serve'd model's KV cache is stored (q8_0 unless "
                  "said; f16, q4_0): a label ends -kv-TYPE for anything but "
                  "q8_0 and the table's ctx column shows it, since a cache "
                  "stored differently is another configuration and not "
                  "another model"),
        flag("--serve-kv-unified", action=argparse.BooleanOptionalAction,
             default=None,
             help="serve each --serve'd model with one cache pool for every slot "
                  "(or, --no-serve-kv-unified, a cache per slot); unset leaves "
                  "the build's default"),
        flag("--profile", action=argparse.BooleanOptionalAction, default=True,
             help="serve each model in the settings that scored best from ml-stack's profiles "
                  "-- the head at the length that measured best, its build, cache "
                  "type, thinking budget, raw flags and asking -- for every flag "
                  "this sweep leaves unset; --no-profile serves it bare"),
        flag("--serve-label", default="", metavar="NAME",
             help="what a --serve'd model's runs are labelled, instead of the "
                  "first 14 characters of its file's name: flash, so its runs "
                  "read flash-plain beside an --on flash-ollama=... run's"),
        flag("--no-draft", action="store_true",
             help="serve each --serve'd model without the draft head its profile "
                  "measured best with, everything else as measured; every label "
                  "carries -nodraft, so the head's worth is two labels apart"),
        flag("--label-suffix", default="", metavar="TEXT",
             help="appended to every label this sweep keeps, so a run that varies a "
                  "serving knob (--serve-arg, --serve-mlock, --serve-mmproj) is "
                  "told apart from the plain one in the table, e.g. -ub2048"),
        flag("--serve-arg", action="append", default=[], metavar="ARG",
             help="a raw llama-server argument for each --serve'd model, repeatable "
                  "(e.g. --serve-arg=--spec-draft-p-min --serve-arg=0.5): the knob "
                  "the bench has no flag for, measured before it gets one"),
        flag("--serve-mlock", action="store_true",
             help="pin the weights in memory rather than page them in on touch"),
        flag("--serve-no-flash-attn", action="store_true",
             help="serve without flash attention, to measure what it is worth"),
        flag("--serve-mmproj", default="", metavar="PATH_OR_AUTO",
             help="a vision projector to load beside each --serve'd model, as the "
                  "page does; measures what sight costs a text question"),
        *(flag(f"--{name}", action="store_true",
               help=f"{said} -- on every way this sweep asks, the way --reach is")
          for name, said in
          (("batch", "every read in one call, with the nudge when it is not"),
           ("kinds", "keep only the kind the question asked for"),
           ("summary", "offer the summarise tool for the broad questions"),
           ("constrain-ids", "answer every turn that offers look_at, show or "
                             "path_between under a grammar in which an id can "
                             "only be one the graph holds"))),
        flag("--n-max", type=int, default=None, metavar="N",
             help="how far the served head guesses ahead (--spec-draft-n-max), the "
                  "length `drafts` found best -- 4 for Flash-Next"),
        flag("--reasoning-budget", type=int, default=None, metavar="N",
             help="tokens each --serve'd model may spend thinking before it must "
                  "answer (llama-server --reasoning-budget; -1 is unlimited). A "
                  "ceiling (--n-predict) cuts the answer; this is the budget that "
                  "stops the thinking. Every label ends -rbN and the table's ctx "
                  "column shows /rb, since it is another configuration"),
        flag("--plain-only", action="store_true",
             help="skip the shortlist half, just measure each model as it is"),
        flag("--shortlist-for", default="", metavar="A,B",
             help="substrings of the models that get the shortlist half as well "
                  "(e2b,e4b); the rest are measured plain only. Both halves of "
                  "a model are asked of one load"),
        flag("--resume", action="store_true",
             help="skip any model and way already kept since --since with this "
                  "many questions at this context and these slots, so a sweep "
                  "killed on its third model costs the third model and not all "
                  "three. Says which it skipped and when each was kept"),
        flag("--since", default="", metavar="WHEN",
             help="with --resume, how old a kept run may be and still count: an "
                  "ISO date or date-time (default: the start of today)"),
        flag("--fleet", action="store_true",
             help="spread the --serve models over the fleet instead of this "
                  "machine: one job per model, this same line with that one "
                  "--serve, planned over the peers, dispatched, waited for and "
                  "gathered into --kept, then shown. Every peer must be on this "
                  "checkout's commit; a peer that is not is refused, here and by "
                  "its daemon"),
        flag("--peers", default="", metavar="NAME,...",
             help="with --fleet, only these peers (default: whichever the fleet "
                  "plans over)"),
        *asking_options(), *reaching_options(), *measuring_options(), *checking(),
    )


def show_options() -> tuple[Option, ...]:
    """``show``: compare two runs, or list what is kept."""
    return (
        flag("--trace", nargs="?", const="", default=None, metavar="LABEL",
             help="print a traced question as a conversation, one line per call: the "
                  "newest run, or the run with this label; --question narrows it"),
        flag("--question", default="", metavar="SUBSTRING",
             help="with --trace: only the question containing this"),
        flag("--by", default="", choices=("serving",), metavar="serving",
             help="group the rows by identical serving and asking, one line "
                  "per group with the mean and the band"),
        flag("--last", type=int, default=0, metavar="N",
             help="only the newest N runs kept"),
        flag("--since", default="", metavar="ISO",
             help="only runs kept at or after this time (e.g. 2026-09-02T14:29)"),
        flag("--kept", default=str(bench.home_dir() / "runs.ladybug"),
             help="the store the runs are in (default: %(default)s)"),
        flag("--compare", nargs=2, metavar=("BEFORE", "AFTER"), default=None),
        flag("--extract", action="store_true",
             help="only the extraction runs (ml-stack-bench extract), in their own "
                  "table; without it they print under the answering table"),
        flag("--speed", action="store_true",
             help="the speed runs (ml-stack-bench speed), one table per label: "
                  "prefill and decode tokens/s and the first token by prompt size "
                  "and streams"),
        flag("--detail", nargs="?", const="", default=None, metavar="LABEL",
             help="the questions themselves, not the totals: what each one wanted, "
                  "what it showed, and what it missed. A label narrows it to one run"),
        flag("--all", action="store_true",
             help="with --detail, every question and not only the ones that missed"),
        flag("--shape", action="store_true",
             help="what the question set is made of -- which kinds its answers "
                  "want, and how many want no person -- so its bias is visible"),
        flag("--rates", action="store_true",
             help="what each run cost to be right -- accuracy over time, tokens and "
                  "memory -- with the Pareto frontier marked"),
        flag("--anyway-export", action="store_true", dest="export_anyway",
             help="with --export, include runs measured over some other graph. "
                  "Not into a repository: those may carry a real community's "
                  "questions"),
        flag("--export", default="", metavar="FILE.json",
             help="write every run as JSON. The store lives under ~/.ml-stack and "
                  "nothing backs it up: a day of measuring is on one disk, and the "
                  "results are all of an invented community, so they can be kept "
                  "beside the code that produced them"),
        flag("--rank", default="", metavar="FILE.md",
             help="write which model answers best, as a conclusion rather than as "
                  "evidence: one line per model -- accuracy from its largest run, "
                  "cost per question from its fastest run that held that accuracy, "
                  "a draft head or a fork included -- and which run that was. This "
                  "is the part worth keeping in a repository -- the raw runs are "
                  "not, since they describe one machine and one build"),
        flag("--noise", type=float, default=NOISE * 100, metavar="PTS",
             help="how far a run's F1 may fall under its model's largest run and "
                  "still supply the cost, in points (default: %(default)s -- one "
                  "question of a short run). A run outside it is listed as rejected"),
        flag("--plot", default="", metavar="FILE.html",
             help="write the runs as a scatter of accuracy against --cost, with "
                  "the frontier joined; opens with no network and no packages"),
        flag("--cost", default="seconds",
             choices=("seconds", "paid_tokens", "kv_bytes"),
             help="which cost the frontier is drawn against (default: %(default)s)"),
    )


def report_options() -> tuple[Option, ...]:
    """``report``: everything measured so far as one document."""
    return (
        flag("--kept", default=str(bench.home_dir() / "runs.ladybug"),
             help="the store the runs are in (default: %(default)s)"),
        flag("--since", default="", metavar="ISO",
             help="only runs kept at or after this time (e.g. 2026-09-02T14:29)"),
        flag("--last", type=int, default=0, metavar="N",
             help="only the newest N runs kept"),
        flag("--model", action="append", default=[], metavar="SUBSTRING",
             help="only models whose file or label contains this; repeatable"),
        flag("--min-n", type=int, default=6, metavar="N", dest="min_n",
             help="a run of fewer scored questions than this is counted in a "
                  "footnote rather than tabled -- it proves the path works "
                  "rather than how well the model answers (default: %(default)s)"),
        flag("--full-n", type=int, default=0, metavar="N", dest="full_n",
             help="the floor for the across-models table, so every model is "
                  "read at the same number of questions (default: each model's "
                  "own largest run)"),
        flag("--room", action="append", default=[], metavar="SIZE",
             help="also answer the memory table for a machine with this much "
                  "room (24G, 24GiB, 24576M); repeatable"),
        flag("--at", type=int, default=32768, metavar="TOKENS",
             help="the per-user context the memory and serving lines answer at "
                  "(default: %(default)s)"),
        flag("--md", default="", metavar="FILE",
             help="write the document here instead of to stdout"),
        flag("--text", action="store_true",
             help="plain text with fixed-width columns instead of Markdown"),
        flag("--open", action="store_true",
             help="with --md, open the file with whatever this desktop opens "
                  "files with"),
        flag("--profile", action="store_true",
             help="write each model's best settings into profiles.json -- the "
                  "build, head, cache, thinking and asking of its best row -- so "
                  "`ml-stack-serve up --profile` and `serve.profile.asking_for` "
                  "serve and ask what was measured. Writes that and nothing else"),
        flag("--profiles", default="", metavar="FILE",
             help="with --profile, write the records here instead of the shipped "
                  "profiles.json (or this machine's own, when the package cannot "
                  "be written to)"),
    )


def forget_options() -> tuple[Option, ...]:
    """``forget``: delete kept runs, the empty ones or one label's."""
    return (
        flag("label", nargs="?", default="",
             help="delete every run kept under this label (needs --yes)"),
        flag("--empty", action="store_true",
             help="delete every run that reads back as nothing"),
        flag("--yes", action="store_true",
             help="really delete a label's runs; without it they are only listed"),
        flag("--kept", default=str(bench.home_dir() / "runs.ladybug"),
             help="the store the runs are in (default: %(default)s)"),
    )


def tail_options() -> tuple[Option, ...]:
    """``tail``: the log of the current measurement, or the latest."""
    return (
        flag("-n", type=int, default=20, metavar="N",
             help="how many lines from the end (default: %(default)s)"),
        flag("-f", action="store_true", dest="follow",
             help="keep printing as the log grows, until the measurement ends"),
    )


def wait_options() -> tuple[Option, ...]:
    """``wait``: block until the detached measurement has ended."""
    return (
        flag("--every", type=float, default=60.0, metavar="SECONDS",
             help="how often to say it is still running (default: %(default)s)"),
    )


def queue_options() -> tuple[Option, ...]:
    """``queue``: an evening of measurements from a file."""
    return (
        flag("label", metavar="FILE",
             help="the queue file: one `ml-stack-bench` invocation per line, "
                  "`#` comments, `set NAME=VALUE` with ${NAME} (the environment "
                  "when nothing sets it), and a `smoke:` line whose failure "
                  "skips the `then:` under it. See docs/examples/"),
        flag("--dry-run", action="store_true",
             help="print the steps as they would be run -- expanded, each one "
                  "checked against this parser, with the label --resume would "
                  "match -- and run none of them"),
        flag("--resume", action="store_true",
             help="skip every step whose label the runs store already holds "
                  "since this queue's start, so a queue stopped half-way does "
                  "not measure the first half again"),
        flag("--yes", action="store_true",
             help="the go-ahead, passed to every step that takes one, so a run "
                  "over its ceiling is not refused into a log nobody is watching"),
        flag("--ceiling", type=float, default=0.0, metavar="MINUTES",
             help="pass this ceiling to every step that does not name its own "
                  "(default: each step's own)"),
        flag("--detach", action="store_true",
             help="run the whole queue in the background the way a measurement "
                  f"detaches: a log under {bench.home_dir() / 'logs'}, `status` for "
                  "the step it is on and what is left, `tail -f` for the log, "
                  "`stop` to end the queue and the step inside it"),
    )


def animate_options() -> tuple[Option, ...]:
    """``animate``: a comparison document as an animated graphic."""
    return (
        flag("comparison", help="the comparison document (JSON)"),
        flag("--out", required=True, help="the .mp4 to write"),
        flag("--png", help="also write the last frame as a still"),
        flag("--quality", choices=("l", "m", "h"), default="h",
             help="l 480p15, m 720p30, h 1080p60 (default h)"),
        flag("--seconds", type=float, default=50,
             help="the length of the whole cut (default 50)"),
        flag("--only", help="comma-separated scene keys to render alone"),
        flag("--work", help="where manim keeps its partial renders"),
        flag("--dry-run", action="store_true",
             help="print the scene plan and write nothing"),
    )
