# An invented world

A demo of a graph read out of a community needs a community, and a real one cannot be shown.
`ml_stack.world` invents one from a seed: an organised group with people who have reasonable
jobs, a voice each, and a memory -- the graph -- so that when they talk (`world.simulate`)
what they say makes sense. A company is one kind of organised group; anything that
communicates in an organised way is another, and five are built in, all producing the same
schema `ml_stack.graph.community` uses so the store, the bench, the page and the ask loop
take them unchanged:

| kind | structure |
| --- | --- |
| `company` | departments under a CEO, reporting lines with spans of five to nine, customers, partners, products, projects |
| `community` | a Slack community of professionals: day jobs at *different* invented organisations, interest groups with moderators, no reporting lines |
| `university` | departments of labs, each led by a principal investigator who `advises` postdocs and students; grants, seminars |
| `open-source` | one project of many repositories; lead and core maintainers `maintain`, contributors `contribute_to`; releases, sponsors |
| `nonprofit` | programmes under an executive director, a board that `advises`, volunteers, funders |

```sh
ml-stack-world make --kind company --size medium --seed 3 --out ./world --json
ml-stack-world questions --world ./world --n 40 --out questions.jsonl
ml-stack-bench run <model> --graph ./world/graph.json --questions questions.jsonl
ml-stack-world simulate --world ./world --out ./talk --days 20 --mix 0.3 --model-url http://127.0.0.1:8080
ml-stack-world emit --from ./talk --as slack-export --out ./export
```

`make` writes `graph.json`, `personas.json`, an empty `calendar.json` and `world.json`;
`simulate` (`world.simulate.run`) has the people talk for some working days -- arcs from
`world.story` for the kind, launches or defences or releases, and routine chatter along
whatever relations the graph holds -- templated unless `--mix` hands a share of threads to
a model at `--model-url`; `emit` writes `messages.jsonl` the way Slack, a mail client, Teams
or a scraper exports it (`--as slack-export|mbox|teams|rows`), so `ml_stack.sources` reads
the invented corpus exactly as it reads a real one.

```python
from ml_stack.world.organisation import make, summary
from ml_stack.world.questions import questions

world = make("community", "small", seed=0)   # the same world every time for a seed
world.graph                                  # nodes, edges, messages: the community schema
world.personas[world.people[0]]              # {"voice", "system", "knows": [ids]}
questions(world, 40)                         # [{"q", "expect": [ids]}, ...] for the bench
```

Sizes are `small`, `medium` and `large` -- 50, 500 and 5,000 people -- and the large one is
made in well under a second (a test holds it under ten). Every person is `part_of` a unit (a department, group, lab,
repository or programme), `works_at` an organisation, is `experienced_in` a few `topic`s
drawn from the unit's own, is `based_in` a real city or remote, `works_on` a handful of
cross-unit projects, and `works_with` the people on their team and their projects; a few are
mentored. Each has a title with a level (IC1 to IC5, manager, director, VP, C-level for a
company; faculty, postdoc and student for a university; lead, core, maintainer and
contributor for a project), a responsibility in a sentence, a start date and tenure, one to
three things they would say about their work (so `look_up`'s "said" voter has something),
and a persona: a voice in a sentence, a system prompt built from their node, and `knows`,
the graph two hops out stepping through people and projects, plus everything public.

Names are assembled from syllable tables at the moment they are asked for -- six sound
families, so five thousand people do not read as one culture -- and organisations from
word stems, so nothing here is, or can recognise, a real person. The only real things are
the cities.

The questions are generated from the truth that made the world -- who reports to whom, who
works on what, who is where -- spread over the same kinds of answer the bench's own set
covers: people mostly, then organisations, places, subjects, units, paths between two
people, events, work going spare, and a few whose right answer is nobody. A kind that lacks
a relation (a community has no `reports_to`) simply asks no such question. Four more are kinds
of *question* rather than of answer, because the bench's own set was short of them:
`aggregate` (a count, scored as the people counted; the unit or employer with the most
people, only when that is not a tie), `twohop` (who works with whoever knows a subject, or
which units or places those people are in -- the far end, never the middle), `trap` (a
false premise about a real person, whose right answer is the place exactly as the graph
has it) and `quote` (answerable from what a person said and from nothing else). Every
question carries its `kind`, which the bench ignores, and `--kinds` draws only some.

## Days that produce conversations

```python
import random
from ml_stack.world.simulate import model_writer, run, simulate, template_writer
from ml_stack.world.story import calendar

world.calendar = calendar(world, days=20, rng=random.Random(world.seed))
for message in simulate(world, days=20, writer=None, rng=random.Random(1)):   # no model
    ...
run("world/", "out/", days=20, mix=0.1, model_url="http://127.0.0.1:8080", seed=1)
```

An invented organisation is a graph until its people talk, and talk with nothing behind it
reads as noise. `world.story.calendar` lays **arcs** over the days -- for a company a
launch, an incident, a new hire's first week, an escalation, an offsite, a quarterly
review, a reorg, a deadline slip; for a community an introduction, a question that gets
answered, a meetup, a job post, a recommendation, an intro between two members; a
university has paper deadlines, grants, seminars, defences and lab moves; an open-source
project releases, bug fixes, RFCs, first pull requests and advisories; a nonprofit
fundraisers, programme launches, volunteer drives and board meetings. `World.kind` picks
the table. Who is in an arc comes from the graph: a group is any node with people joined
to it, named by the words in its label, so an incident is whoever is in "engineering" and
"support" whatever the graph calls their kinds, and a community with no `reports_to` is
scheduled from `works_with`, `part_of` and `moderates` because the sampler only ever uses
the relations it finds. Each arc says where it happens -- a Slack channel, an email
subject, a Teams chat -- and is deterministic from the seed.

`simulate` is the clock. Each working day it takes the arcs alive that day plus routine
chatter, a Poisson-ish number per person along their real relations: people who work
together talk in their team channel or a DM, a reporting line is a 1:1 DM or an email,
two people with no group in common get email or Teams. Every thread is two to eight
`Message`s with timestamps inside work hours in each sender's office timezone
(`attrs.timezone` on their place, else UTC), monotone within the thread. Who says what is
a **writer**, `(persona, prompt, context) -> str`. `template_writer` needs no model:
sentences per conversation kind and organisation kind, filled with the graph's own names
-- the speaker's project, place and subject, the group, the person they are talking to --
and never the same sentence twice in a thread. `model_writer` has a persona speak through
`converse` over the subgraph it `knows`, with its own `system` prompt, the thread so far as
turns, and its earlier threads of the same arc read back from the store as memory, so what
it said last week is what it says this week. `mix` is the share of threads the model
writes, and the arcs get it first because an arc is where consistency is noticed.

**Outcomes write back.** An arc's end leaves one typed edge in `world.graph` -- `decision`,
`moved_to`, `now_works_with` or `joined` -- carrying `attrs.said_in`, the message it was
said in, so the next conversation and the truth agree, and the graph handed in is the
graph after. **What a message costs** is two model calls: the answer, then `converse`
asking what the answer was about, which is kept because its ids are exactly the `Drew`
edges the thread memory wants. A persona is handed what its thread is about as the
`opening` -- its own entry, the project, the place, the others in the thread -- which
grounds it and stops `converse` sending an answer that touched nothing back to look. `run`
reads `graph.json`, `personas.json` and any `calendar.json` from a directory, writes
`messages.jsonl`, the graph after and the calendar used, holds `simulate.lock` while a
model is in use, and returns the counts, including `messages_per_model_call`.

## Message formats

**A corpus is one list, whichever product it came from.** `ml_stack.world.Message` is the
shape every reader returns and every emitter writes: a world id, a `source`, a `channel`, a
`sender`, a Slack-style `ts`, the text, and `thread` naming the root. `ml_stack.sources.read`
looks at a path and reads a Slack export directory, an mbox, a Microsoft Graph
`chatMessage` dump or the rows a Slack scraper writes -- each also there by name
(`sources.slack_export`, `sources.mbox`, `sources.teams`, `sources.rows`). Given the
world's people (`id -> {"label", "email"?, "handle"?}`) a reader puts `person:` ids back on
every `U0…`, address and Graph uuid; without them the product's id stays in `sender` and
`attrs["sender_kind"]` says whose it is.

```python
from ml_stack import sources
from ml_stack.world.emit import slack_export, mbox, teams, rows

slack_export(messages, people, "demo/slack", domain="pellard.example")  # users.json, channels.json, dms.json, <channel>/<day>.json, dms/<D0…>/
mbox(messages, people, "demo/mail.mbox")                              # From/To/Cc/Date/Subject/Message-ID/In-Reply-To/References
teams(messages, people, "demo/teams.json")                            # {"value": [chatMessage, ...], "channels": [...], "chats": [...]}
log = rows(messages, people)                                          # the Slack scraper's rows: channel, channelId, ts, sender, text, threadTs, scrapedAt, permalink

back = sources.read("demo/slack", people)      # sniffed; equal to `messages` up to attrs
```

The emitters write what each product actually exports -- Slack's per-day files cut at
midnight UTC with `thread_ts`, `reply_count` and `reactions`; mail that any client threads,
written through `mailbox.mbox`; Graph's `from.user`, `body.content`, `replyToId` and
`channelIdentity` or `chatId` -- with product ids minted deterministically from the world's,
and written back into each message's `attrs`. The world's id rides in the one slot each
product has for it (`client_msg_id`, Teams' `id`, an `X-World-Id` header beside
`X-World-Ts`, since `Date:` has no fraction of a second); scraper rows have none, so that
reader mints `<channelId>-<ts>` the way the scraper's pipeline does. `rows` exists so a demo
profile drops into a pipeline built on those rows with no adapter between them.
