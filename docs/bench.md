# Measuring

## What this measured, and what it changed

Architectures that behave unlike a dense transformer when served have their own notes
under [`docs/architectures/`](architectures/README.md): what the header says, what it
meant when measured. Flash-Next's 51B-parameter n-gram table is there.

**IQ quantisations are the slow choice on Apple silicon.** Measured 2026-09-02, the same
ten questions, the same fork build, the same 32k x 2 slots, Qwen3.8-Flash-Next answering
plain with thinking on: `UD-IQ4_XS` (87 GB) took 70 s a question at 54% F1; `UD-Q4_K_XL`
(104 GB) took 44 s at 64%. The IQ formats decode through lookup tables that Metal runs
markedly slower than the K-quant kernels, so on a Mac the smaller file is the slower
model, and at ten questions the accuracy gap is suggestive rather than settled. Rule: on
macOS take a K-quant (`Q4_K_M`, unsloth's `UD-Q4_K_XL`) and spend the memory; take an IQ
build only when a K-quant does not fit at all. `ml-stack-models files` marks IQ builds on
a Mac so the choice is made knowingly. On a CUDA card the IQ kernels are fine, and the
memory saved is worth having.

Answering a graph is the case this was built for, so the numbers are here rather than in
somebody's notes. Ten questions over the invented community, 32k per slot, greedy, one run
at a time on an otherwise idle machine. **F1** over the entries an answer lights, with the
pair behind it, because the pair is what says how a run was wrong:

| run | F1 | recall | precision | lit per question | wall | KV + runtime |
| --- | --- | --- | --- | --- | --- | --- |
| gptoss-shortlist | **62%** | 70% | 59% | 2.0 | 198s | 11.89G |
| gptoss-plain | 61% | 65% | 62% | 2.0 | 127s | 11.89G |
| e4b-plain | 58% | 70% | 51% | 2.0 | 159s | 7.75G |
| e2b-plain | 41% | 55% | 39% | 2.3 | 59s | 3.21G |
| e4b-shortlist | 33% | 70% | 25% | 4.5 | 300s | 8.34G |
| e2b-shortlist | 29% | 80% | 18% | 6.1 | 64s | 3.50G |

A good answer lights about **1.7** entries. Read the last column against that and the table
explains itself.

**Scoring on recall alone said the opposite, and it was wrong.** Under it, `e2b-shortlist`
was the most accurate run there was at 80%, beating the 120B — because showing more costs
nothing under recall, and it lit six entries where fewer than two were wanted. A model that
lit every entry in the graph on every question scored 100%. That metric survived a day and
twenty-four green tests, none of which asked what the degenerate strategy would score. One
does now.

**A shortlist handed to a small model is echoed, not selected from.** It is the single
largest effect here: E4B 58% → 33%, E2B 41% → 29%, precision halving while recall rises.
The same shortlist does nothing for the 120B either way. The idea is sound and the machinery
is worth keeping; what is missing is teaching a model that a shortlist is somewhere to look.

**The table above was measured with the loose asking, which is no longer how anything
asks.** Every run in it let `show` name what the answer was about, uncapped — what
`tight=False` still does, and what `--also loose` now measures as the control. Telling
`show` instead to light only the entries that answer the question moved Qwen3.8-Flash-Next
from 43% to 83% precision over the invented community (2026-09-02) with nothing about the
searching changed, so tight is what `converse` does when it is asked nothing. Re-measure
before reading a row of that table as current.

**What the tool descriptions changed was real.** E4B answered 17% before they carried a
worked call and 70% recall after, on the same weights and the same questions. Six of its
nine failures had been the identical shape: two model calls, a hundred characters of prose,
no search at all. The large models never needed telling, which is why this went unseen until
a small one was measured.

Two mistakes cost the earlier numbers their meaning. Timings were taken while other runs
shared the GPU, which is why `sweep` and `run` refuse a busy server now. And every model had
512 tokens for thinking, tool calls and answer together, because `n_predict` defaulted low:
a thinking model fills that with reasoning and returns an empty answer.

None of this survives a new model release. Re-run it.

The runs themselves are in a graph store under `~/.ml-stack/bench`, which nothing backs up.
`ml-stack-bench show --export PATH` writes them out, so a day of GPU time is not on one disk.
**That file does not belong in a repository, and `--export` refuses one**: the numbers
describe one machine and one llama.cpp build, they go stale with the next model release, and
`run --graph` takes any graph, including a real community's. Back it up somewhere outside a
working tree.

What is worth keeping here is the conclusion, not the evidence. `ml-stack-bench show --rank
FILE.md` writes one line per model -- its best run, and what that run cost -- because that is
what the defaults in this library are set from, and a default with no recorded reason is a
default nobody can argue with. Both it and `--export` carry only runs whose recorded graph
fingerprint is the community that ships with this package, and refuse a run from before that
marker existed, because not knowing which graph a run read is not the same as knowing it was
invented. The ranking also ignores anything shorter than a short run, so a `--smoke` run
cannot rank a model on two questions.

## Measuring a change to the asking

```
ml-stack-bench prepare --embed-url http://127.0.0.1:8081 --embed-model embeddinggemma-300M-Q8_0.gguf
ml-stack-bench sweep --on gptoss=http://127.0.0.1:8080 --on e4b=http://127.0.0.1:8083 \
    --embed-url http://127.0.0.1:8081 --embed-model embeddinggemma-300M-Q8_0.gguf
ml-stack-bench show
```

`sweep` measures each model twice — as it is, and with a search run before it — and prints
them all in one table. `run` does one of those on its own, and `show --compare A B` puts two
side by side with the difference.

`sweep` and `run` **refuse a server that is already working**, because a timing taken while
another run has the same GPU is not a timing. It is a real failure and not a hypothetical:
several sweeps left running in the background against one server produced wall clocks that
were two runs sharing a machine, and nothing in the numbers said so. A server that will not
answer `/slots` is reported as unknown rather than assumed idle. `--anyway` proceeds on
purpose.

Every measuring command **estimates itself before it starts** -- after the self-check, before
a download or the lock -- from what is kept: seconds per question from the newest run of
each model (at the same context when one is kept there), else a guess from its weights on
disk, times the questions, the ways one load is asked and the models, plus a load each,
printed as `estimate:` lines that `history` reads back beside the actual. Over `--ceiling`
minutes (30, or `MLSTACK_BENCH_CEILING`) it refuses with exit 5 and says what to shorten;
`--yes` runs it anyway, and a `--smoke` is never refused. No more eight-hour tests.

Serve every model being compared with the **same context and the same number of slots**, or
the comparison is of two configurations rather than two models: a model at 8k per slot is
faster and holds a smaller cache than the same model at 32k. The table prints `ctx` on every
line so a mismatch is visible rather than silent.

`look_up` is measured **as the application ships it**. With a store -- `prepare` builds one,
and `run` and `sweep` take it as their default once it exists -- every `look_up` the model
makes is `ml_stack.graph.search.hybrid`: the characters, the store's word index and, given
`--embed-url`, its vectors, fused by rank. Without one it is character matching alone, which
is what the bench measured for months while the application ran the other thing, so every
ranking it wrote ranked a `look_up` nobody used. The table prints `find` on every line --
`chars`, `words` or `meaning` -- beside `draft`, and for the same reason as `ctx`: a run
with one finder against a run with another is two measurements, not a comparison.

The questions are asked of an invented community that ships with this package, so a number
means the same thing on any machine and no real person's details are involved. Each question
may carry the ids a good answer names, which is what makes accuracy measurable rather than
impressionistic. Runs are kept in a graph store under `~/.ml-stack/bench`, so one can be
compared with another a week later.

There are two question sets, and a ranking should be read on both. The curated set,
`ml_stack.graph.community.QUESTIONS` -- a hundred scored, ten whose right answer is nobody
-- is written for nuance: two people who share a surname, a false premise about a real
person, a role nobody has but one person nearly does, a count scored as the people counted,
an answer two hops away from the person the question describes, and things only somebody's
own words say. Rather than write all hundred by hand, half of the second fifty were drawn by
handing this graph to [the generator](world.md) and reading what it produced; `ml-stack-bench
prepare --mix` prints how many questions ask for each kind of answer, which is what says
whether it still measures the whole page or has drifted into being about people. The generated set, `ml-stack-world questions --world DIR --n 200`,
is derived from an invented world's truth for breadth -- hundreds of questions over
thousands of people, tagged by `kind`, with `--kinds aggregate,twohop,trap,quote` to draw
only those -- and reaches sizes the hand-written set never will. Both are fed to
`--questions`; a model that is good on one and not the other is telling you which of the two
it was tuned on.

`show` reports wall clock, model calls, prompt tokens **and how many of them were cached** —
a conversation re-sends itself every turn, so the tokens shown and the tokens actually read
are different numbers and only the second is a cost.

`pfx` is that cache per turn. A question is several calls, and the second should pay for
the tool result and the model's reply and nothing before them; when its cached tokens fall
short of the previous call's whole prompt, the system prompt and every tool schema were
read again. The table prints the share of turns that kept the prefix, `--detail` says
`cache 3/4 turns` per question, and a change to the asking that breaks the prefix -- the
cheapest speed lever there is -- shows here where the totals hide it. Blank on a run from
before it was counted, because not counted is not none.

It also reports what the server costs to keep up beyond its weights (`kv+run`: the KV cache
and the runtime around it, measured as resident memory minus the weights on disk) and that
figure per 1k of held context (`per 1k`). The second is the one to compare: it says what one
more conversation costs, whatever context each server happened to be given, and it is what
decides how many conversations a machine can hold at once.

`show --rates` puts accuracy over each of the three scarcities — time, tokens, and the
memory a conversation holds — because a score alone cannot choose between a model that is
better and one that is cheaper. It marks the **Pareto frontier**: the runs nothing else
beats on both accuracy and cost, which are the only ones there is ever a reason to choose.
`--cost seconds|paid_tokens|kv_bytes` redraws it against whichever is scarce, and
`--plot out.html` writes it as a scatter with the frontier joined — hand-built SVG, no
library and no network, so it opens on any machine.

`show --rank FILE.md` composes each model's line rather than reading it off one run, because
a draft head cannot change an answer -- the target verifies every token -- only the wall
clock and the memory. Accuracy comes from the model's largest run (the full sweep, undrafted,
on mainline; the newest on a tie), and cost, printed per question so a twenty-question
`drafts` run compares with a hundred, from its fastest run of at least `SHORT` questions
whose F1 held within five points of that -- a head, a draft length and a fork included -- with
the last column naming which run and which build it was (`--noise` widens or tightens the
five). A run that fell outside the noise is listed under the table as `rejected`, so a head
that hurt accuracy is seen rather than skipped, and a smoke run supplies neither accuracy nor
cost. `--rates` and `--plot` carry the same composed point per model, marked `=` and drawn as
a ring, beside the runs themselves. The question that made it was how to rank a model whose
draft head was not yet settled: before this, the ranking took each model's best-F1 run and
reported that run's cost, which ranked a drafted model at its undrafted speed, or not at all
when the drafted run was short.

`--n-predict` is a **ceiling, not a budget**: nothing is spent that is not generated, so a
high one costs nothing and a low one truncates. It defaults high on purpose. A thinking
model spends most of a turn reasoning before it writes anything — measured, gemma-4 filled
a 220-token ceiling entirely with thought and returned empty content — so what a low
ceiling cuts is always the answer, never the thinking.

`--card` asks with what the model itself recommends, which is the only place a
recommendation is ever applied. It is read from the served model's **GGUF metadata** where
that exists — `general.sampling.temp`, `.top_k`, `.top_p` are written into the file, so they
cannot drift from the weights and need no prose parsed out of a README — and from the card
otherwise. The two agree where both exist: gemma-4 says temperature 1.0 / top_p 0.95 /
top_k 64 in each. They are per model, not per family: Qwen3.8-Flash-Next asks for top_k 20. A publisher's advice is a hypothesis about a task they have not seen: gemma-4
asks for temperature 1.0 across all use cases, and on this one — calling tools with exact
ids, where sampling noise becomes a wrong argument rather than a livelier sentence — greedy
measured better on the plain path. Read the card with `ml-stack-models card <repo>`, test it
with `--card`, and ship what the measurement favoured.

A score is only worth acting on when you can see which questions made it, so
`show --detail` prints the questions themselves — what each one wanted, what the answer
showed, what it missed, and what it cost — with `--detail LABEL` for one run and `--all` for
every question rather than only the ones that fell short. It is what turns a number into a
diagnosis: a model whose wrong answers took *more* calls than its right ones searched hard
and missed, while one whose wrong answers took *fewer* never reached for the tools at all,
and those two failures are fixed by opposite things.

F1 scores what was lit, and nothing else scored the prose. An answer can light the right
entries and still name one the model never found, read or showed -- a plausible name it
made up, or half-remembered from the question before -- and F1 is none the wiser. So every
row also counts the entry labels that appear in the answer's text (whole words, case aside,
as the page matches them) that no tool call produced, and `show` prints the total per run
as `made`, beside the scores; `--detail` names them, and `--rank` carries the column. Blank
on a run from before it was counted, because not counted is not none.

`sweep --serve` asks the same served model in several ways for one load: `--also terse`
describes the tools briefly, `--also card` asks with the model's own sampling, `--also
greedy` at temperature 0, `--also rich` has `look_up` say what matched and why, with a
topic hit bringing the people joined to it, `--also loose` asks the old way — `show`
told to name what the answer is about, uncapped — as a control against the tight asking
every run now uses, `--also reach` gives one tool result a page of neighbourhood rather than
a flat character cut, `--also batch` asks for every read in one call, `--also single` asks
for the opposite (one entry a read, more turns), `--also few` offers three tools and no
other way of looking, `--also kinds` keeps only the kind the question asked for, and `--also
summary` offers the whole graph at a glance. (`--also tight` is what the first way already
does, and says so.) `--reach N` and `--rounds N` are not ways of their own: they ride on
every way, as `--batch`, `--kinds` and `--summary` do. There is no asking every model wants,
which is the point of measuring ten of them on one load — `report --profile` then writes the
winner into that model's record.

```
ml-stack-bench sweep --serve gemma-4-E2B-it --also terse --also card --detach
ml-stack-bench status
ml-stack-bench tail -f
ml-stack-bench stop
ml-stack-bench sweep --serve gemma-4-E2B-it --also terse --also card --resume
```

A measurement is hours, and a child of a shell -- `nohup`, `&`, a redirect into a scratch
directory -- dies with the shell, or with the agent that opened it; a ranking sweep was
killed that way half an hour in. So `--detach` on `run`, `sweep`, `drafts` and `concurrent`
has the command re-run itself in a session of its own, with its output in a log under
`~/.ml-stack/bench/logs/`, and gives the shell back at once. `status` says what is measuring,
since when, and the last line of its log; `tail -f` follows the log; `stop` sends the pid
SIGTERM -- never a name -- which the child takes as an exit, so a model it put up comes
down with it. `sweep --resume` then skips every model and way already kept today with the
same questions, context and slots, so the killed sweep costs the model it died on and not
the ones before it.

### An evening as a file: `ml-stack-bench queue`

A night of measurements is not one command, it is nine — a fairness sample, a knob matrix
smoked one knob at a time, the hundred-question runs, the extraction runs, then the ranking
and the report. That was a zsh script in a scratch directory, rewritten nine times in one
evening (2026-09-02), with `&&` between each smoke and the run it guarded and a `--yes`
typed onto every long line; nothing could say what was running or what was left.
`ml-stack-bench queue FILE` is that evening as a file:

```
# the restart: every improvement smoked, compared on ten, then the hundred
set FX=hf:unsloth/Some-Model-GGUF/UD-Q4_K_XL/Some-Model-UD-Q4_K_XL.gguf
set BEST=--serve ${FX} --serve-draft auto --serve-kv q8_0 --context 65536

smoke: sweep ${BEST} --label-suffix=-v2 --smoke
then:  sweep ${BEST} --label-suffix=-v2 --sample 10
sweep ${BEST} --label-suffix=-v2
show --rank docs/model-ranking.md
```

```
ml-stack-bench queue docs/examples/flash-next-restart.queue --dry-run
ml-stack-bench queue docs/examples/flash-next-restart.queue --yes --detach
ml-stack-bench status          # step 3/9, what is left, and what it has kept so far
ml-stack-bench stop            # the queue, and the step inside it
```

One `ml-stack-bench` invocation per line, `#` comments, `${VAR}` from a `set` line or from
the environment, and a `smoke:` whose failure skips the `then:` under it and says so — the
`&&` kept, so a shape that will not load is never measured on a hundred questions while the
rest of the evening still happens. Every line is checked against this parser as the file is
read, so `--sampel` on the last line is refused before the first model loads rather than
after the eighth measurement, and an unset `${FX}` is refused rather than expanded to
nothing and measured as the default model for six hours.

It is not a second scheduler. Each step is its own `ml-stack-bench` process, so it brings
the measuring lock, the self-check, the estimate and the smoke it already has, and a step
of a queue and a run started by hand still wait for each other; the queue holds no lock and
is only the thing that waits. `--yes` and `--ceiling` are given once at the top and passed
to every step that takes them (never to `show`), `--resume` skips every step whose label the
runs store already holds since the queue started, `--detach` puts the whole evening in the
background the way one run does — one log, named after the queue file — and `status` grows a
`queue` block naming the step in flight, the tally so far and what is left. A step that
fails on its own does not end the queue, and the exit code is 1 if any step failed. Every
summary line is one shape, so the log can be read by eye or by `grep`:

```
=== 21:41:07 step 3/9: sweep --serve hf:unsloth/Some-Model-GGUF -- ok (612s)
=== 21:41:07 step 4/9: sweep --serve hf:unsloth/Some-Model-GGUF -- skipped (0s): its smoke (step 3) failed
```

`docs/examples/flash-next-restart.queue` is the seventh of those nine scripts, as a file.

`history` answers "how much GPU time did that day cost, and how much of it kept nothing"
from the logs directory alone, one line per detached measurement, oldest first:
`started  sub  model/label  est  actual  exit  kept`. The start comes from the log's header
or its filename stamp, the end from the log's last write, the exit from what the log says --
`done`, `killed`, `crashed: <the exception line>`, `running` while `measuring.json` names a
pid that is alive -- the estimate from an `estimate:` line beside what it actually took,
and `kept` is every run in the store whose `at` falls inside that window. The last line is
the total: runs, GPU hours, and the hours that produced no kept run, which is the number to
be embarrassed by. `--since today|24h|7d|<date>` narrows it, `--json` dumps the entries,
`--home` and `--kept` point it elsewhere.

The same sweep kept twelve runs as nothing: the store took them and gave back an empty
string for each, and the smoke run had passed because the summary was printed from memory.
`save` now reads every run back the way `show` reads it before it returns, and refuses to
if what comes back is not what went in; a `--smoke` run's summary is read from the store
for the same reason. `show` counts any run that still reads back empty, and `forget --empty`
removes them.

**The runner checks itself before it spends the GPU, and smokes before it measures.** Every
measuring command -- `run`, `sweep`, `drafts`, `concurrent`, `extract` -- first drives the
exact command line it was given through the whole path with no server and no GPU: a
scripted model that takes exactly what `Client` takes, a served model that never starts, a
preflight that reads nothing, the invented community and two of its questions, into a
scratch store it reads back. It prints `selfcheck: ok (2.1 s)` and goes on, or refuses with
exit 4 and the traceback -- before the lock is taken and before anything is fetched;
`--no-selfcheck` skips it, for a run you are deliberately repeating. Then, unless the run
*is* a `--smoke` run or is told `--no-smoke`, it smokes for real: two questions (three
messages, for `extract`) through the real server and the real store, read back, before its
own questions -- on the same load where a model is served, so a sweep smokes each model as
it comes up and pays for it once -- and a smoke where every question fails ends the run with
exit 1 and the reason before anything else starts. The day this was written a new `--also
tight` way reached `Client.__init__` as a keyword and took an 87G load down with it, because
the smoke was a step in a plan and the test's fake client accepted anything; now the fake is
the runner's own -- `bench.selfcheck.ScriptedModel`, bound against the real signature, and
the tests use it too -- and the check is not a step anyone has to remember.

**A load is fetched, checked and timed before it is measured.** Every `hf:` reference a
measuring command names -- the models `--serve` puts up, the heads `--serve-draft` and
`--draft` name -- is downloaded through `hub.fetch` *before the measuring lock is taken*,
one line each with its size, because a download inside the timed window is a timing of the
network and holding the lock through it makes the next run wait for the Hub;
`--no-prefetch` skips it. Then each served model is preflighted -- shards present,
architecture read by this build, weights plus an estimated KV cache under what this machine
may wire, every flag one the build accepts -- and the report is printed under the `up in`
line, so the estimate sits beside what `kv+run` then measures; it is kept on the run as
`server.preflight`. A refused preflight prints the reason and moves to the next model
instead of ending the sweep: a sweep of five must not die on the one that does not fit. The
lease's own `load_s` and `warmup_s` are kept on the run too -- not a stopwatch around the
serve, which also holds an adopted server's nothing -- and `show` prints `load` next to
`wall`, blank for a run from before it was recorded; `--detail`'s header names it and
`--rank` carries it.

```
ml-stack-bench sweep --serve gemma-4-E2B-it --serve gemma-4-E4B-it --serve gpt-oss-120b \
    --shortlist-for e2b,e4b --serve-kv q8_0
ml-stack-bench drafts gemma-4-E4B-it --draft '' --draft auto --n-max 4 --n-max 8 --n-max 16
```

`sweep --shortlist-for e2b,e4b` gives the shortlist half only to the models whose name holds
one of those; the rest are measured plain. Either way both halves of a model are asked of
**one load** -- the shortlist is a question about the asking, like `--also terse`, and
loading the model twice to answer it measured nothing about the asking. `--plain-only`
still means no shortlist half for anyone.

Every served model's KV cache is stored as q8_0 unless `--serve-kv` says otherwise; a run
served with another type has a label ending `-kv-TYPE`, and the `ctx` column reads
`32k x1/q8` or `32k x1/f16`, because a run with one cache type against one with another
is a comparison of configurations, and the column is what stops it being read as a
comparison of models. `drafts --n-max N`, repeated, serves each head once per value --
`--spec-draft-n-max` is bound when the server starts, like the head itself -- labelled
`draft:<head>@n8`, so the table shows acceptance and wall clock per (head, n-max). Without
it, once at the build's own default. The baseline with no head is measured once. What a
head was worth is then a number rather than a division done by hand: `show` prints
`speed` beside `draft` -- the newest undrafted run of the same model, build and size,
per question, over this one, as `1.42x` -- `--rank` carries it into `cost from`, and
`drafts` ends with its own table, one row per (head, n-max) with acceptance, speedup
and how far F1 moved, and names the fastest configuration whose F1 held within the noise.

```
ml-stack-bench concurrent e2b-4x3 --conversations 4 --turns 3 --base-url http://127.0.0.1:8080
```

Everything above asks one question at a time, which is right for timing a model and wrong
for the question a server is actually asked: how many people can talk to it at once, and
what that costs each of them. `concurrent` runs N conversations of T turns each on threads
against one server, each a chain of questions with the earlier turns carried, and records
per turn the wall clock, the time until the server began generating, and what the turn
spent waiting -- its wall clock less what the server itself reports reading and generating,
which is the queueing once N exceeds the slots `/slots` reports. For the run it keeps the
wall clock over all of them (not the sum of the turns), the most the server held while they
were in flight, sampled rather than read afterwards, and F1 as usual, so a setting that
answers faster by answering worse is visible. `show` marks such a run `4x3` in the `conc`
column; the flags a build has for holding conversations -- `--kv-unified`, `--cache-ram`,
`--cache-idle-slots`, `--slot-prompt-similarity`, `--slot-save-path` -- are typed on
`ServerSpec`, so they can be varied and the build asked whether it has them before a load.
It takes the same lock as `run` and `sweep`, and `--smoke` runs two conversations of one
turn to prove the path.

### Measuring the reading, not the asking

```
ml-stack-world make --kind community --size small --seed 3 --out ./world
ml-stack-bench extract flash-next --world ./world --serve Qwen3.8-Flash-Next --smoke
ml-stack-bench extract flash-next --world ./world --serve Qwen3.8-Flash-Next --twice
ml-stack-bench show --extract
```

Everything above measures a graph that already exists. Before it exists it has to be read
out of what people said, one message at a time, and which model reads best was never
measured, because nobody knows the truth behind a real message. An invented world does:
the simulation writes each message *from* the graph, and since 2026-09-02 it writes down
what each message asserts as it goes -- ``attrs["asserts"]``, the ids of the people,
organisations, topics, places and other entries the writer put into that sentence and the
relations it stated -- so the gold is a record, not an inference. The template writer's
record is exact; the model writer's is the opening it was grounded in plus what its answer
drew on, a lower bound, and is scored separately with its coverage read as "against a
lower bound". The asserts ride through ``messages.jsonl`` and the scraper-shaped rows
``emit`` writes as an extra key the readers ignore.

`extract` samples N messages (forty by default, three under `--smoke`), stratified so an
arc's thread and every kind of routine chatter both appear -- an arc is a handful of
threads in a fortnight and a plain draw would miss it -- has the model read each into
`contracts/extraction.schema.json` (people with an optional role, organisation and place;
organisations; topics; places; relations as ``from / rel / to``) under a grammar with the
sender named as context and thinking off, folds the extractions into one graph by name
(case aside, near-spellings joined by `entities.spelling.close`, a first name joined to
its full name), and scores that against the union of what those messages assert. A world
without messages is simulated for a few days with the template writer first, and the
command says so. The estimate is printed before the clock starts: forty messages at the
guessed fifteen seconds each is ten minutes, or whatever an earlier run of the same model
measured.

The table keeps coverage and precision as separate columns per kind, because an F1 alone
cannot say whether a model missed things or made them up and those are fixed by opposite
changes to the asking; `invented` is the count and rate of extracted people and
organisations that match nothing in the gold, the reading-side twin of `made`; relations
match loosely on the name (case, underscores and spaces aside, then near-spellings) and
strictly on the ends. Under each row: the folded graph's connected components and the
share of nodes in the largest against the gold's own, so a model that scores well on
triples and builds a fragmented graph is seen; conformance, how many relations used the
world's own vocabulary; fact survival, the share of each message's assertions still
present after the fold, which catches a fold that merged two people into one; and
resolution, extracted nodes per gold node (`splits`) and gold nodes per extracted node
(`merges`), 1.00 each when perfect. `--twice` reads the sample again with the model's own
card and reports the Jaccard similarity of the two graphs: a model that gives a different graph each
run is a finding. An entry naming something the messages asserted under `others` -- a
project, a department, which the generic schema has no word for -- is neither found nor
invented. The runs sit in the same store as the answering runs, marked `kind: "extract"`;
`show` prints them in their own table under the answering one, `show --extract` alone.
`extract` takes the measuring lock like `run`, and `--detach`, `status`, `tail` and `stop`
work as they do there.

Every run records which machine measured it (`server["host"]`) and which code
(`server["commit"]`, the short sha, `(dirty)` when the tree had changes). `show` adds a
`host` column only when a store holds more than one; the ranking never composes one host's
accuracy with another's cost -- a cost run from another machine is listed as
`rejected: other host` -- and `--rates` and `--plot` name points by host. `sweep --fleet
[--peers NAME,...]` spreads the `--serve` models over the fleet: one job per model with
the same line otherwise, planned, dispatched, waited for and gathered into `--kept`, then
shown; a peer on another commit is refused before anything is dispatched.

