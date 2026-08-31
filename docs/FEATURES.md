# What ml-stack does

Every claim here has a check in `docs/verify_release.py`. Run it:

```
python docs/verify_release.py
```

---

## Setting up

Install it on each machine and type the same passphrase. The key is derived from those
words, so two machines that heard the same phrase agree on it without exchanging
anything — there is no key to copy and no address to write down.

- Machines that derived their key from different words are invisible to each other, so
  several groups share a network without any of them being configured to.
- The group name separates two clusters that happened to choose the same passphrase.
- The passphrase is stretched with scrypt before it becomes a key. Everyone on the
  network can hear the beacons, so a weak phrase would otherwise be worth grinding
  guesses against offline.
- At least eight characters.

Beacons are signed. A peer's address is taken from the packet it arrived in, never from
anything the packet claims about itself.

A machine can be in several clusters at once, and join or leave any of them from the
Cluster screen without touching the others. It answers to each: work sent from any
cluster it belongs to reaches it, and its models and served endpoints are offered to all
of them. The one it answers as, when something has to pick, is the first in the list.
Leaving takes effect at once, and a machine that has left stops answering to that
cluster's credential.

Settings has a button to walk through first-run setup again, for renaming the machine or
changing its passphrase.

## Running work

Each machine runs a daemon that accepts jobs, moves files, and reports what it is.

- **Slots.** One job at a time by default, because two jobs on one card contend for
  memory and both get slower. A machine whose work is preparing data has no such
  contention and can be told to run several.
- **Files.** Uploads and downloads resume from where they stopped and are verified by
  digest, in both directions.
- **Stopping.** A stop is SIGTERM first, so a loop that checkpoints on it keeps its
  progress. Stopping one job leaves the others alone.
- **Jobs run in the file root**, so a relative path in a job's arguments points at the
  files you pushed.

## Choosing where work runs

- **What a machine reports** — CUDA, ROCm, Apple silicon, free VRAM, RAM, cores. These
  are three different questions: PyTorch's ROCm build answers yes to every CUDA
  question, so a Radeon would otherwise satisfy a CUDA-only requirement and only reveal
  itself once the job failed.
- **What a machine declares** — labels like `train` or `prep`, set by whoever runs it. A
  machine cannot prove it has no GPU, so this half is declared rather than detected.
- **How fast it has been** — measured per kind of work, from jobs that ran. There is no
  table of which card is faster.
- **A machine nobody has measured is tried, not skipped.** It is scored as typical and
  given work, because being given work is the only way it stops being unmeasured. A new
  machine also benchmarks itself once when it joins, so the first choice is not a
  coin-flip.

Work that no machine can run fails immediately, naming every machine and the reason:

```
gpubox:  has 23.0 GB VRAM, needs 80.0
pi-rack: does not report 'cuda'; has no backends
radeon:  this machine is in use (mon tue wed thu fri 09:00-17:00); work resumes Mon 17:00
```

Work that fails is retried on a different machine. A machine that fails several jobs in
a row is set aside for a growing cooldown rather than draining the queue.

## Keeping your machine yours

- **Working hours.** Block out times when a machine is somebody's desk. Windows are
  local wall-clock and may cross midnight. Work submitted during a blocked window waits
  rather than failing.
- **Pause.** Stops the machine taking work now. What is already running is stopped and
  requeued, so the machine comes back to you immediately and the run resumes from its
  last checkpoint. A pause survives a restart.
- **Reservations.** One machine can hold another for a while, with a ceiling so a
  forgotten reservation cannot take a machine out of the cluster permanently.

## Models

Every model file this machine holds, and every one held by the others.

- **Popular models are listed for you**, asked of Hugging Face each time rather than
  written into the app, so the list is what people are downloading now and not what was
  current when it shipped. Ranked by where a model sits in both the last month's
  downloads and what is trending, so a release from last week reaches the first page.
  Only builds that fit this machine appear.
- **Searching narrows the same list.** Type into the box above it and the hub is
  searched; the results carry the same sizes, families and symbols. Paste a reference
  instead and it is fetched directly.
- **Pages.** Twelve at a time, with how many there are in total.
- **Uncensored builds are left out** unless the box is ticked. Abliterated, heretic and
  uncensored variants are published with their refusals removed, and are a large share
  of what is most downloaded.
- **A draft model is taken too** when the publisher ships one, and the server is told
  to guess ahead with it.
- **Each one shows** its download size, how many parameters it has, how many are active
  if it is a mixture of experts, and what it reads and writes: 💬 text, 🖼 images,
  🔊 audio, 🎬 video.
- **Families are checkboxes**, all ticked, covering the whole list rather than the page
  in front of you, so a family further down still has a box.
- **Getting one** takes a Hugging Face reference or a link. Naming a repository and no
  file gets its Q4 build, which is the one that fits in memory. If another machine on your
  network already has that file, it is copied across the network instead of downloaded
  again.
- **Getting a model happens in the background**, so a large one does not hold the
  screen. It is listed while it arrives, with how much has come and how much is left.
- **A download that stopped** is listed on the Models screen with how much of it
  arrived, to resume by asking for the model again or to discard.
- **A copy in progress** resumes where it stopped. A part-file is only finished into a
  real model once its length matches what the far end says the file is, and a part left
  behind by a different file is discarded rather than continued.
- **Between machines**, the reassembled file is checked against a digest the sending
  machine computed, so a copy that arrived wrong is thrown away rather than kept.
- **Automatic downloading** can be turned off in Settings, leaving the network as the
  only source.

## Chat

Talk to a model from the interface, whichever machine is running it.

- **The model list** covers models running here and models running on any machine on
  your network. One that a different machine is serving is used exactly like a local
  one.
- **A machine that installs nothing extra can still chat**, as long as something on the
  network is serving. Reaching another machine needs no address and no key exchange —
  the passphrase both machines already share is enough.
- **Replies arrive as they are written**, not in one piece at the end.
- **Conversations are kept** between runs and searched by anything said in them.
- **Running a model here** is a button beside it in the Models list. The llama.cpp
  server is downloaded the first time and checked against the digest GitHub publishes
  for it.

The model server itself listens only on this machine. The one address exposed to the
network is the daemon's, which already requires the shared credential.

## Training

`Trainer` runs the loop on PyTorch or MLX. The framework comes from the model, so the
same call works on a Mac and on a CUDA box.

- Checkpoints are written atomically; a half-written one is ignored rather than resumed
  from.
- A resume restores weights **and** optimizer state, or refuses. Weights alone is a warm
  restart, and shows up later as a loss spike.
- `steps` is a total, so re-running the same call after a crash finishes the run.
- A run whose loss goes non-finite is stopped before the update reaches the weights.
- Learning-rate schedules are plain functions returning floats.
- Leak-safe splits: contiguous tail, by group, or stratified by label.

### Recipes

Training without writing code. A recipe is a JSON contract describing a form, the data it
needs and what it requires from a machine; a builder turns the answers into a model.

| Recipe | What it learns |
|---|---|
| `text-lm` | Continues a pile of writing — a house style, a character voice, a domain's jargon |
| `classify-text` | Sorts labelled documents into their categories |

Both are byte-level, so any text works with no vocabulary file.

- A setting the recipe does not declare is refused rather than ignored, so a config
  cannot quietly disagree with what was trained.
- The model size is checked against the memory the chosen machine reports.
- `--dry-run` trains twenty steps and writes nothing.

## What each machine can train with

Training needs numpy, safetensors and a framework, and those are too large and too
platform-specific to carry inside the app. So the app builds an environment of its own,
separate from any Python already on the machine, and runs training jobs with it.

Settings lists what can go in it, with what each is for and what it costs to download:

| | |
|---|---|
| Training essentials | Arrays, checkpoint files, and ml-stack's own training code |
| PyTorch | The build that matches the card — NVIDIA, AMD ROCm, or processor-only |
| MLX | Apple silicon |
| Images | Reading and resizing pictures |
| Hugging Face models | Starting from a downloaded model rather than from scratch |
| Temperature and clocks | Reporting this machine's temperature and GPU clock |

The right PyTorch is offered for the card that is there: the NVIDIA and AMD builds come
from different indexes, and installing the wrong one gives a machine that never uses its
card.

If the machine has no Python new enough to build with — macOS ships 3.9 — the app
downloads one. Nothing is installed into the system Python.

## The interface

A native window — WKWebView on macOS, WebView2 on Windows, WebKitGTK on Linux. Each
download also carries a headless binary for a machine with no screen, which serves the
same interface to a browser.

- **First run** asks what to call the machine, then which clusters it belongs to, then
  what the machine is for. The settings are pre-filled from the hardware, with the reason
  shown beside each, so a wrong guess is visible rather than silent.
- **Clusters are optional and there can be several.** The same page lists the ones this
  machine is in, joins another with a passphrase, and leaves one. A machine that joins
  none runs models and trains on its own; with no passphrase there is nothing to check,
  so it answers to that machine and to nobody else.
- **Starting up** — with the computer, when you log in, or only when you open it. The
  boot option asks for permission through the operating system's own password dialog.
- **The cluster view** shows every machine with its vendor, how many jobs are
  running of how many it will take, how much of its memory is in use, how busy its
  processors are, and — where a probe reports them — temperature, clocks, power,
  video memory and whether it is throttling.
- **Closing** asks once whether to keep running so the others can still send it work, or
  quit. The answer is remembered if you leave the box ticked.

Setup on a machine that has not joined a cluster is refused from anywhere but that
machine, because until it joins there is no credential to check and the first person to
reach it would own it. A headless machine can be set up over the network with a one-time
code printed on its console.

The cluster key never enters the browser. Signing in is the passphrase, checked by
deriving the key again and comparing.

## Removing it

Settings has a Remove section listing everything on this machine with what it takes up.

- **Your models and your own files are not ticked.** Everything ml-stack made for
  itself is: the settings, the key, the chats, the training environment, the model
  server. A model takes as long to download the second time as it did the first, so it
  is only removed if you say so.
- **Each line says what it is and what going without it costs**, so a decision is not
  made for you.
- **Removing needs two clicks**, and says what it took and how much came back.
- Starting up with the computer is undone whether or not anything else is.

## Keeping itself up to date

Turned on in Settings, and off in the same place.

- **Checked once a day.** A newer release is downloaded, verified against the digest
  GitHub publishes for it, and unpacked over the running copy, which is kept until the
  new one is in place.
- **Never while a job is running.** A machine part way through training is left alone
  until it is not, so an update cannot cost a run.
- **It starts the new copy itself** and stops the old one, rather than waiting to be
  opened again.
- Installed with pip instead? It says so, and leaves pip to it.

## Telemetry

Temperature, clocks, power draw, utilisation and throttle state.

| Machine | Source |
|---|---|
| NVIDIA | `nvidia-smi` |
| AMD | `rocm-smi` |
| Apple silicon | `darwin-perf`, no `sudo` |

The vendor tools are read by the daemon itself, so a machine with a card and no
framework installed still reports it. They also see memory held by other processes,
which a framework's own reading does not.

A machine with no probe reports what the standard library can see — cores, architecture,
memory — and says nothing about accelerators rather than guessing.

## Installing

Twelve packages, installed separately. Four have no dependencies at all: `fleet`,
`client`, `media` and `contracts`, so the daemon installs on a small board as fast as on
a workstation.

```
python packaging/build.py            # wheels
python packaging/build.py --bundle   # and a standalone app for this platform
```

## Graphs

A graph is a mapping with `nodes` and `edges`. What a project calls its kinds and its
relations is the project's business; nothing here has an opinion about either.

**Where it lives.** An embedded property graph in one file — no server, native Cypher,
shortest paths worked out by the engine rather than by a loop. Nodes and edges go in as plain
dictionaries and come back as the same ones: whatever a caller carries beyond the columns, the
messages a node was read from, a flag of its own, rides along untouched. Anything about the
graph as a whole — what it counts, when it was built — is kept beside it.

**It cannot be lost to a bad rebuild.** A pipeline that read nothing produces an empty graph,
and an empty graph looks exactly like "remove everything" to anything that trusts it. A write
that would take most of a store raises rather than runs, and says what it thinks went wrong
upstream. One that would take a tenth leaves a verified copy behind on the way past. Copies
are verified by reopening them on a fresh handle and counting — a copy nobody opened is not a
backup — and a restore saves what is there first, because restoring the wrong one must not be
the second unrecoverable act of the day. On a filesystem with copy-on-write this costs
milliseconds and no disk; everywhere else it says loudly that it is copying for real.

**Two processes cannot corrupt one.** The database's own lock already stops the second writer
with an IO exception. What is added is the part it does not do: which process is in your way
and for how long, waiting for a turn rather than failing instantly, and giving up a read
handle when a writer wants in — because a read handle parked in an idle process wedges every
writer for no benefit. A dead owner's record is cleared on the way in; a live one's never is.

**Finding things, three ways at once.** Matching characters finds a name typed exactly.
A word index stems, so "compiler" finds "compilers". Vectors find meaning, so "who fixes
machines" finds a robotics technician. All three run and the rankings are fused, without
anyone pretending a cosine similarity and a BM25 score are the same kind of number. Whichever
is unavailable simply does not vote.

**Asking a model about it.** Handing a model the whole graph does not scale, and handing it a
pre-chosen slice makes the choosing the answer. It gets three things it can do instead — find
entries by name, read what is held on them, trace how two connect — and what it touched comes
back with the words, so a caller can show the working rather than a second guess at it. An id
the model invents is refused in one place.

**Looking at it.** One self-contained page: force layout in two dimensions and three, labels
that do not collide and that hold their place as the camera turns, a legend that filters, a
layout that re-settles when a kind is switched off, a map, and evidence for everything drawn.
Everything ships inside the file, which is what makes it mailable and also what limits it —
whoever has the file has the graph, so anything private is served rather than sent.

## Reading a site

Every project that reads a conversation out of a web app writes the same scraper: find the
rows, find who wrote each and what it says, find the identifier that makes a row the same row
tomorrow. Only the selectors differ, so the selectors are data — `website`, `slack` and
`discord` to start with, and a preset can be adjusted without being rewritten.

- **Virtualised lists.** A row scrolled far enough away is removed from the document, not
  hidden, so scrolling to the top and reading once returns the oldest screenful and looks
  exactly like a short conversation. Rows are collected after every step instead.
- **Signing in happens once, by hand.** The browser runs on a profile directory that persists.
  That also sets the etiquette: a profile is a real session, the account looks online while it
  is open, and reading fast enough to notice is how a session stops working. Hours can be held
  and requests paced unevenly, because evenly spaced ones are a signature.
- **Second runs are cheap.** A watermark per source for what is new, and a mark beside it for
  what is old and grew — which a watermark cannot see.

## Entities

- **Resolving names.** Folding duplicates, canonical forms, and telling a handle from a name.
- **Spelling.** Whether two words are one word typed twice. A doubled letter is its own case at
  any length, because it is the commonest way to write a name wrong; a substitution in a short
  word is not, because two four-letter names one letter apart are two people.
- **Paths.** The best-evidenced way from one node to another. Weight is how many sources agree,
  so cost is its reciprocal and two well-attested hops beat one nobody corroborates. Dijkstra,
  not A*: the graph has no geometry to guess distance from, so any admissible heuristic is
  zero, and A* with a zero heuristic is Dijkstra with extra words.
- **Edits.** A request in ordinary words becomes edits against ids that exist, checked before
  anything is applied.

## Known limits

- **Several daemons on one machine share a discovery port**, and only one of them
  answers. One daemon per machine is unaffected; this only shows up when simulating a
  cluster on a single box.
- **The native window is verified on macOS.** Windows and Linux use different webview
  backends, and a window cannot be tested without a display.
- **No dataset browser yet.** Models are catalogued across the network; datasets are
  not. Peer-to-peer transfer of them exists, but nothing indexes what each machine has.
- **Conversations stay on the machine they were held on.** They are not shared across
  the network.
- **Training is one machine per run.** Splitting a single run across machines is not
  built.
