# ml-stack

**Run and train models across every machine in your house.**

Install it on each one and type the same passphrase. They find each other on their own —
no addresses, no keys to copy, no config file. Chat with a model from any machine,
whichever one is actually running it. Work goes to whichever machine is free and fastest:
the box with the card trains, the spare CPUs prepare data, and any of them can be taken
back the moment you want it.

```
$ ml-stack-peers ls
NAME             URL                          FREE    STATE      DEVICE
gpubox           http://192.168.2.27:8770     0/1     busy       NVIDIA GeForce RTX 4090  6.2/24.0 GB free
radeon           http://192.168.2.31:8770     1/1     idle       AMD Radeon RX 7900 XTX  22.1/24.0 GB free
mac-studio       http://192.168.2.44:8770     1/2     idle       Apple M2 Ultra  96.0/128.0 GB free
pi-rack          http://192.168.2.51:8770     5/6     busy +2    16 cpu
```

Everything runs on your own hardware. Nothing leaves the network.

![The cluster view](docs/images/cluster.jpg)

- **Chat from any machine.** The one with the card runs the model; the laptop talks to
  it. A machine that installs nothing extra still gets to use it, and conversations are
  kept.
- **Nothing to configure.** A passphrase is the whole setup. Two households on one
  network stay separate without either of them being told to.
- **Work lands where it fits.** Placement is by what a machine reports and how fast it
  has actually been measured, per kind of work. A machine nobody has measured is tried,
  not skipped.
- **Your machine stays yours.** Block out working hours, or hit pause when you start a
  game — the run stops, requeues, and picks up from its last checkpoint. Pause the whole
  cluster and a machine that was switched off takes the pause when it comes back.
- **Train without writing code.** Pick what you want it to learn, point it at your
  files, and it runs. Or drive it from Python if you would rather.
- **Not only text.** A model with a projector is handed the pictures themselves, and a
  recording is transcribed on whichever machine is free. It checks a model can see
  before it believes what it says about a page.
- **Mixed hardware is the normal case.** NVIDIA, AMD ROCm, Apple silicon and plain CPUs
  in one cluster, each reporting its own temperature, clocks and throttle state.

[Full list of what it does](docs/FEATURES.md).

## Install

One script per platform, and re-running it upgrades in place. The default is the app: a
window, and the daemon behind it.

**macOS and Linux:**

```
curl -fsSL https://raw.githubusercontent.com/adammikulis/ml-stack/main/packaging/install.sh | sh
```

**Windows**, in PowerShell:

```
irm https://raw.githubusercontent.com/adammikulis/ml-stack/main/packaging/install.ps1 | iex
```

Do the same on every machine you want to work with, typing the same passphrase. They find
each other on their own.

**If you write Python**, the library on its own has no dependencies, so the machine you
drive from needs no CUDA, no MLX and no training stack:

```
pip install ml-stack
```

There are three other modes -- a machine with no screen, a machine that starts at boot
before anyone logs in, and a git checkout that follows `main` -- and one model cache per
machine whichever you pick. [Installing](docs/install.md) has all of it, and how a script
answers every prompt without a terminal to type at.

## Starting it

Make this machine a peer, and see who else answers:

```
ml-stack-fleet join --persist
ml-stack-fleet status
```

Put a model up and talk to it:

```
ml-stack-serve up hf:unsloth/gemma-4-E4B-it-qat-GGUF/gemma-4-E4B-it-qat-Q4_K_M.gguf
ml-stack-serve status
```

`ml-stack` on its own starts the daemon and opens the interface in your browser;
`ml-stack --list` prints every command with the first line of its help.

## Work that runs while you do something else

A job here is a command you start and walk away from, not a chat window you sit in front
of. Point it at what you have:

```
ml-stack-ingest ~/texts/*.pdf --out ./sources.ladybug --images --resume --detach
ml-stack-ingest status --out ./sources.ladybug
ml-stack-ingest ask --out ./sources.ladybug "how is heart rate controlled"
```

Chapters, sections and figures come out of each PDF, a model reads each section into one
graph, and every claim keeps the page and the model behind it. A run is hours, so
`--detach` gives the shell straight back and puts the run in its own session with a log;
`status` says how far it has got, how fast the model is reading and how long is left;
`--resume` starts where a killed run stopped. `ml-stack-jobs wait ingest` blocks until it
has ended, so the next step is `wait && next` rather than a loop you wrote by hand.

`ml-stack-do "..."` takes the task in words instead: a model on your own hardware, holding
every command here as a tool, asks what the task leaves open, prints a plan, waits for the
go, runs it and says where the results are. `ml-stack-mcp` hands the same functions to an
agent over MCP, and anything long returns a log and a pid rather than blocking the call.

## Pictures and speech

A model with a projector is given the picture, not a description of one. `--mmproj auto`
finds the projector shipped beside the weights, `ml-stack-ingest --images` hands the model
each section's figures rather than only their captions, `graph.tree.read(client, ORG,
images=[chart])` reads an org chart or a family tree out of a photograph, and `web_look`
brings back a full-page screenshot with the page's largest pictures. A tool of your own
that returns pictures puts them in front of the model the same way. A second model can
transcribe first and the first one structure what it said, which is the slot a document
model -- DeepSeek-OCR, GLM-OCR, surya -- is good in.

**Then it checks the model can see.** `ml_stack.vision.VisionGate` draws a PNG of coloured
bands, asks the model to name them left to right, and raises rather than let a run believe
a model that is answering from the words alone. A server started without a projector says
so instead of quietly reading captions.

Speech is three protocols and one resolver, so the engine is a detail:

```
ml-stack-speech providers
ml-stack-speech transcribe recording.m4a
ml-stack-speech say "the run has finished" --out done.wav
```

faster-whisper, whisper.cpp, Whisper through transformers, piper, kokoro, the operating
system's own voice, Silero and an energy detector that needs nothing installed --
`providers` says which could run here and which is picked when none is named, and
`regions FILE` gives the seconds somebody is speaking. Audio is anything ffmpeg reads,
a video file's track included. A peer transcribes what it is sent, so the hearing happens
on whichever machine is free rather than the one you are typing at.

## The rest of it

| | |
| --- | --- |
| [What it does](docs/FEATURES.md) | every feature, each with a check in `docs/verify_release.py` |
| [Installing](docs/install.md) | the four modes, the one model cache per machine, Windows, and an install a script drives |
| [The commands](docs/commands.md) | every `ml-stack-<command>`, what it takes and what it prints |
| [The fleet](docs/fleet.md) | joining, seating people across machines, following a branch, and running work on peers from Python |
| [Finding and serving a model](docs/serving.md) | one manager per machine, the shape each model measured best in, how many people fit in a card, llama.cpp builds and draft heads |
| [Working with a graph](docs/graph.md) | the six things a model is given instead of the graph, how a question is asked, and a conversation of any length |
| [Documents into a graph](docs/ingest.md) | a book read section by section, with the page and the model behind every claim |
| [Training](docs/training.md) | the loop, the recipes, and a fine-tune that ends in a model calling your own tools |
| [Measuring](docs/bench.md) | timing and scoring a model's answers, what that settled here, and an evening of runs as a file |
| [An invented world](docs/world.md) | a community with people who talk, the days they talk over, and the exports their corpus arrives as; nobody real in any of it |
| [Packages](docs/packages.md) | what each module is, and the extras it carries |
| [Model ranking](docs/model-ranking.md) | one line per model: its best run, and what that run cost |
| [Architectures](docs/architectures/README.md) | the models that behave unlike a dense transformer when served |
| [Working on ml-stack](docs/development.md) | `contracts/`, the git hooks, and how the tests are written |

## Licence

Apache License 2.0. See [LICENSE](LICENSE).
