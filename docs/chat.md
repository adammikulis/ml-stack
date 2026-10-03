# Chatting with ml-stack

`ml-stack-chat` is a conversation with a served model that operates ml-stack: what is serving,
the models on this machine and on the Hub, downloads, benchmarks, jobs, and read-only views of
the security review. It is the one agent command: with no task it is a conversation (one history,
no end), with a task in words (`ml-stack-chat "run benchmarks with quince-2b"`) it plans, asks
go and ends on `done`. Which calls run unasked is the role (`docs/agent-roles.md`).

```
ml-stack-chat                         # the best downloaded mixture-of-experts model ranked for an agent
ml-stack-chat --model PATH_OR_HF_REF  # lease this one, in the settings it scored best with
ml-stack-chat --url http://127.0.0.1:8080
ml-stack-chat --resume                # the newest saved chat; --resume ID for another
```

With no `--model` or `--url` the model is the first of what is already downloaded that
`ml-stack-models`' ranking (`serve.suggest.recommend`, goal `agent`) puts forward and whose name
reads as a mixture-of-experts (`Flash-Next`, `-A3B`, `moe`); a dense model is used only when no
such one is downloaded, and the chat says so. With nothing suitable downloaded it prints what to
pull and exits. The server is leased through the broker like any other served model's. Nothing leaves the
machine and no paid API is called.

## Inside the chat

| | |
| --- | --- |
| `/help` | the list |
| `/new` | forget the conversation and start another under a new id |
| `/tools` | the tools in this role, and which ask you first |
| `/role [NAME]` | the roles; with `NAME`, run under that one (typed by you; the model has no way to) |
| `/rules ...` | list, remove, flip and clear the saved always/never rules (`docs/agent-roles.md`) |
| `/plan TEXT` | the model sees only the read tools and `plan`; it shows the steps and does nothing |
| `/model [REF]` | the model in use; with `REF`, lease that one for the rest of the chat |
| `/quit` | leave; the chat is saved |

Anything else is said to the model. The answer streams; each tool call is one line
(`-> name(args)`) and its result the next, cut at 300 characters. The history is saved to
`~/.ml-stack/chat/ID.json` after every message (no system prompt in it; a resume builds a fresh
one and keeps only user, assistant and tool messages, dropping a trailing call that has no
result). When it fills 80% of the context it is compacted as in `docs/compaction.md`
(`--no-compact` turns that off, `--context-size` sets the window it measures against).

## What the agent can and cannot do

Reads run without asking: `serve_status`, `models_find`, `models_files`, `models_on_disk`,
`ollama_models`, `bench_status`, `bench_history`, `bench_show`, `fleet_peers`, `setup_look`,
`doctor`, `jobs_status`, `jobs_wait`, and `review_view` (status, held items, events, hosts,
downloads, scanners of the security review, as JSON).

Each of these waits for your yes, every call, naming the arguments (and any path outside
ml-stack's state directory): `serve_up`, `serve_down`, `serve_escalate`, `models_fetch`,
`bench_run`, `bench_standard`, `bench_speed`, `bench_compare`, `bench_animate`. No answer, EOF or
anything but `y`/`yes` is a no; there is no `--yes`; a yes to one call does not cover the next;
an answer to the model's own question (`ask_user`) is not a yes to a call.

Not offered at all: releasing or purging quarantine, approving a host, minting a human grant,
changing the sentinel mode or the scan policy, planting baselines or decoys, the workspace and
fleet-join tools, speech, `decide`, file and shell access. A model that calls a tool named for
one of them, or whose arguments name sentinel's code or state, gets a refusal that carries the
command to run in your own terminal; and when your message asks for one the chat prints that
command itself before the model answers:

| asked | the command |
| --- | --- |
| release or purge something in quarantine | `ml-stack-security review` |
| approve a host | `ml-stack-security approve-host HOST` |
| mint a grant | `ml-stack-security review` |
| change the security mode or policy | `ml-stack-security mode`, `ml-stack-security scan-policy` |
| plant or remove baselines and decoys | `ml-stack-security baseline` |

Everything a tool returns is untrusted text (model names, logs, model cards, reasons): it is
fenced as `<untrusted>` data, stripped of chat markup, cleared of credentials and flagged when
it reads as an instruction, and a call whose arguments came from it asks again through the taint
rail (so an action can be asked about twice: once for what it does, once because the run has read
outside text). A tool result is only text: it starts no call; calls come from the model's reply
through the loop. The model's own words are screened too.

`tests/test_chat.py` runs each of these against the real loop and rails with a scripted model;
`tests/test_redteam_chat.py` (`pytest --redteam`) runs every text in the injection corpus as a
tool result to a model that obeys it.

## A session

The output below is the real loop's, with a scripted model standing in for the model (its
wording is illustrative) and recorders standing in for the commands that would start things.

```
you> what is running?
-> serve_status()
   []
Nothing is serving and nothing is measuring. Want me to start Qwen3.8-Flash-Next?

you> yes, start it
-> models_on_disk(words="flash")
   [{"model": "Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf", ..., "shards": 4, "draft": "mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf", "mmproj": ""}]
-> serve_up(model="Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf", port=8080, draft="auto")

! serve_up(...): serve_up will start a model server (takes GPU and memory).
allow it? [y/N] y
! serve_up(...): serve_up (fleet) would use draft after the run read untrusted text from flagged, tool:serve_status#1, tool:models_on_disk#1
allow it? [y/N] y
   {"log": "...", "pid": 7}
It is starting; serve_status will say when it answers.

you> pull the Q4_K_M quince-2b
-> models_fetch(reference="hf:unsloth/quince-2b-GGUF/quince-2b-Q4_K_M.gguf")

! models_fetch(...): models_fetch will start a download into the model cache.
allow it? [y/N] y
...

you> smoke-benchmark it
-> bench_status()
   {"text": "nothing is measuring"}
-> bench_run(argv=["sweep", "--serve", "quince-2b-Q4_K_M.gguf", "--smoke"])

! bench_run(...): bench_run will start a benchmark (a long job on the GPU).
allow it? [y/N] y
...
A smoke sweep is running; bench_status will say when it ends.

you> release the quarantine on 20261003-1a2b

Only a person can release or purge something held in quarantine. This agent cannot, and no argument changes that. Run this in your own terminal: ml-stack-security review
I can't release a quarantine. Only you can, in your own terminal: ml-stack-security review
```

## Limits

- Not driven with a real model for this document: the loop, the rails and the tools are tested
  with scripted models and a fake llama-server. How well a given model picks tools and follows
  the system prompt is not measured here.
- A run that is already going can be seen (`bench_status`, `jobs_status`) but a chat cannot
  interrupt one except through `serve_down`; a job it started detached keeps running after the chat is closed.
- The prompts can come twice for one action (the confirm rail, then the taint rail).
- Whether a request is destructive is decided by the tool list, not by a decision model; a
  second-layer classifier from `ml_stack.decide` (adding a confirmation, never removing one) is not
  wired in.
- No file edits, shell or workspace access, and no way to cancel a call already confirmed.
- The history is compacted by the same summariser as in `docs/compaction.md`; a summary is the model's
  own text, so very long chats lose detail.
