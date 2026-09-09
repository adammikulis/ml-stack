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
