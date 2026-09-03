"""``ml-stack-do`` -- a task in words, done by a served model with the commands as tools.

The tools are `ml_stack.mcp`'s (every command, long ones detached with a log and a pid),
plus the bench subcommands that follow the CLI, the jobs a detached command records, two
lookups (the GGUFs on this machine, the models Ollama holds), and three of the loop's own:
``ask_user`` puts one question to the person and waits, ``plan`` prints the steps and asks
"go?", ``done`` ends the loop. The model is asked with the tools offered, thinking off,
and a high ceiling; each tool's description carries worked examples, which is what
measured as making a small model call tools right.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import re
import sys
import textwrap
import time
import urllib.request
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO

from ml_stack import mcp

__all__ = ["ROUNDS", "SYSTEM", "Outcome", "Person", "bench_cli", "client_for",
           "command_tools", "main", "models_on_disk", "ollama_models", "own_tools", "run",
           "system_for"]

ROUNDS = 40
"""Tool-calling turns one task may spend before the loop ends with the transcript."""
N_PREDICT = 16384
CUT = 6000
"""Characters of one tool result the model is shown."""
OLLAMA_URL = os.environ.get("OLLAMA_HOST") or "http://127.0.0.1:11434"

SYSTEM = (
    "You are operating ml-stack for a person at a terminal: you serve models, measure "
    "them, compare and animate the results, and read what this machine holds, all through "
    "the tools you have been given. You cannot run anything except by calling a tool.\n\n"
    "Ask before assuming. When the task leaves a choice open -- which model or which "
    "file, how many questions, a sample or the full set, with or without a draft head, "
    "where to write -- call ask_user rather than guessing, one question per call, and "
    "wait for the answer before asking the next. Look things up first so the question "
    "names what was found: a task that names a model is answered with models_on_disk "
    "and, when it names Ollama or two backends, ollama_models too, and the person is "
    "asked to confirm the exact files before anything starts. Do not start a measurement "
    "the person has not confirmed.\n\n"
    "Then call plan with the steps in order; it asks the person \"go?\". Only when the "
    "person passed --yes is that skipped, and even then a task naming more than one "
    "backend still confirms the models found. Then act: call the tools in the order "
    "planned. A long command detaches and returns a log and a pid; call jobs_wait to "
    "wait for it rather than calling status again and again.\n\n"
    "Last, call done with what was measured and where it is -- the labels, the numbers if "
    "any came back, the files written. Say plainly when something failed and what the "
    "error said. Never claim a result a tool did not return."
)

YES = ("The person passed --yes: plan prints the steps and does not ask go. Still ask what "
       "the task leaves open, and still confirm the models found when more than one backend "
       "is named.")


def system_for(yes: bool = False) -> str:
    """The system prompt, with what the person passed on the command line."""
    return SYSTEM + ("\n\n" + YES if yes else "")


NUDGE = ("You replied in words and called nothing. If the task is finished, call done with "
         "a summary of what was measured and where it is; if you need something from the "
         "person, call ask_user; otherwise call the next tool.")


# -- worked examples ---------------------------------------------------------------------
def worked(*pairs: tuple[str, str]) -> str:
    """The examples clause of a tool description: ``"task" -> call`` pairs."""
    return " Examples: " + "; ".join(f'"{task}" -> {call}' for task, call in pairs) + "."


ACCEPTANCE = ("benchmark qwen3.8-flash-next with llama.cpp (both with draft head and no "
              "draft head) and with ollama, make some animations")

EXAMPLES: dict[str, tuple[tuple[str, str], ...]] = {
    "serve_status": (("what is serving?", "serve_status()"),
                     ("is anything up on 8099?", "serve_status(port=8099)")),
    "serve_up": (("put quince-2b up", 'serve_up(model="quince-2b.gguf")'),
                 ("serve flash-next with its head on 8099",
                  'serve_up(model="hf:unsloth/Qwen3.8-Flash-Next-GGUF/'
                  'Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf", port=8099, '
                  'draft="auto")')),
    "serve_down": (("stop the server", "serve_down()"),
                   ("take 8099 down", "serve_down(port=8099)")),
    "models_find": (("is there a newer quince?", 'models_find(words="quince")'),
                    ("find flash-next on the hub",
                     'models_find(words="Qwen3.8 Flash Next")')),
    "models_files": (("which quantisations does that repo have?",
                      'models_files(repo="unsloth/Qwen3.8-Flash-Next-GGUF")'),
                     ("show me the files, drafts included",
                      'models_files(repo="unsloth/quince-2b-GGUF")')),
    "models_fetch": (("download the Q4_K_M quince",
                      'models_fetch(reference="hf:unsloth/quince-2b-GGUF/quince-2b-Q4_K_M.gguf")'),
                     ("get the flash-next head too",
                      'models_fetch(reference="hf:unsloth/Qwen3.8-Flash-Next-GGUF/'
                      'mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf")')),
    "bench_run": (("run benchmarks with qwen3.8-flash-next",
                   'ask_user(question="Full hundred questions (~45 min) or a sample of 10 '
                   '(~5 min)?", choices=["the hundred", "a sample of 10"]) first, then '
                   'bench_run(argv=["sweep", "--serve", "Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf", '
                   '"--serve-draft", "auto", "--plain-only", "--sample", "10"]), which keeps '
                   'the run as Qwen3.8-Flash--plain'),
                  (ACCEPTANCE,
                   "after the models are confirmed, three calls, the file being the `model` "
                   "models_on_disk returned and the labels ending -plain, -nodraft-plain and "
                   "-ollama-plain: "
                   'bench_run(argv=["sweep", "--serve", "Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf", '
                   '"--serve-draft", "auto", "--plain-only", "--sample", "10"]); '
                   'bench_run(argv=["sweep", "--serve", "Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf", '
                   '"--serve-draft", "", "--label-suffix", "-nodraft", "--plain-only", '
                   '"--sample", "10"]); bench_run(argv=["run", "Qwen3.8-Flash--ollama-plain", '
                   '"--base-url", "http://127.0.0.1:11434", "--sample", "10"]) with the Ollama '
                   'model already up; jobs_wait(kind="bench") after each'),
                  ("smoke the sweep first",
                   'bench_run(argv=["sweep", "--serve", "quince-2b.gguf", "--smoke"])')),
    "bench_status": (("is it still measuring?", "bench_status()"),
                     ("what is the bench doing?", "bench_status()")),
    "bench_history": (("what ran today?", 'bench_history(since="1d")'),
                      ("the last five runs", "bench_history(limit=5)")),
    "bench_show": (("show me the table", "bench_show()"),
                   ("which model is fastest per GB?", 'bench_show(args=["--rates"])')),
    "fleet_peers": (("who else is on the network?", "fleet_peers()"),
                    ("is the studio serving anything?", "fleet_peers(timeout_s=4)")),
    "fleet_join": (("make this machine a peer", 'fleet_join(passphrase="...")'),
                   ("join at logon too", 'fleet_join(passphrase="...", persist=True)')),
    "world_make": (("invent a small company", 'world_make(kind="company", size="small")'),
                   ("a medium university, seed 3",
                    'world_make(kind="university", size="medium", seed=3)')),
    "setup_look": (("can this machine serve a 100 GB model?", "setup_look()"),
                   ("what is downloaded?", "setup_look()")),
    "doctor": (("is the checkout healthy?", "doctor()"),
               ("check the repos under ~/Documents/repos",
                'doctor(repos=["~/Documents/repos/ml-stack"])')),
    "bench_compare": (("compare the last three runs",
                       'bench_compare(args=["Qwen3.8-Flash--plain", "Qwen3.8-Flash--nodraft-plain", '
                       '"Qwen3.8-Flash--ollama-plain", "--export", "compare.json"])'),
                      ("make an animation of the last comparison",
                       'bench_compare(args=["--last", "--export", "compare.json"]) then '
                       'bench_animate(args=["compare.json", "--out", "compare.mp4"])')),
    "bench_animate": (("make an animation of the last comparison",
                       'bench_compare(args=["--last", "--export", "compare.json"]) then '
                       'bench_animate(args=["compare.json", "--out", "compare.mp4"])'),
                      (ACCEPTANCE,
                       'after the three runs and bench_compare --export: ask_user(question="one '
                       'comparison video of all three, or a clip per panel?", choices=["one '
                       'video", "a clip per panel"]), then bench_animate(args=["compare.json", '
                       '"--out", "compare.mp4"])')),
    "bench_standard": (("run the standard sets on quince",
                        'bench_standard(args=["quince-2b.gguf", "--sample", "10"])'),
                       ("standard sets for flash-next, detached",
                        'bench_standard(args=["Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf"]) then '
                        'jobs_wait(kind="bench")')),
    "bench_speed": (("how fast is it at 1, 2, 4 users?",
                     'bench_speed(args=["quince-2b.gguf", "--users", "1,2,4"])'),
                    ("speed matrix for flash-next with and without the head",
                     'bench_speed(args=["Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf", '
                     '"--serve-draft", "auto"]) then bench_speed(args=['
                     '"Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf", "--serve-draft", ""])')),
    "jobs_status": (("is anything running?", "jobs_status()"),
                    ("did the ingest finish?", "jobs_status()")),
    "jobs_wait": (("wait for the bench", 'jobs_wait(kind="bench")'),
                  ("run the report once the sweep ends",
                   'jobs_wait(kind="bench") then bench_show(args=["--rates"])')),
    "models_on_disk": (("run benchmarks with qwen3.8-flash-next",
                        'models_on_disk(words="qwen3.8-flash-next") first, so the question '
                        'names the file and its draft head'),
                       (ACCEPTANCE,
                        'models_on_disk(words="qwen3.8-flash-next") and '
                        'ollama_models(words="qwen3.8-flash-next"), then one ask_user '
                        'listing both')),
    "ollama_models": (("what does ollama have?", 'ollama_models(words="")'),
                      (ACCEPTANCE,
                       'ollama_models(words="qwen3.8-flash-next") -> '
                       '[{"name": "qwen3.8-flash-next:125b-mlx", "format": "safetensors", '
                       '"quantization": "nvfp4"}], named in the confirming ask_user')),
    "ask_user": (("run benchmarks with qwen3.8-flash-next",
                  'ask_user(question="Full hundred questions (~45 min) or a sample of 10 '
                  '(~5 min)?", choices=["the hundred", "a sample of 10"])'),
                 (ACCEPTANCE,
                  'first, listing what both lookups found: ask_user(question="llama.cpp: '
                  'Qwen3.8-Flash-Next-UD-Q4_K_XL (draft head mtp-Qwen3.8-Flash-Next-shared-Q8_0 '
                  'found); Ollama: qwen3.8-flash-next:125b-mlx (safetensors, nvfp4). Use '
                  'these?", choices=["yes", "no"]); then ask_user(question="graph Q&A on the '
                  'invented community -- sample of 10 (~5 min) or the hundred (~45 min)? plus '
                  'speed matrix and standard sets?"); then ask_user(question="one comparison '
                  'video of all three, or a clip per panel?"); then plan'),
                 ("where should the table go?",
                  'ask_user(question="Write the ranking where? (default: the bench home)")')),
    "plan": (("run benchmarks with qwen3.8-flash-next",
              'plan(steps=["bench_run sweep flash-next with its draft head, a sample of 10, '
              'kept as Qwen3.8-Flash--plain", "jobs_wait bench", "bench_show --rates"])'),
             (ACCEPTANCE,
              'plan(steps=["bench_run sweep with the head -> Qwen3.8-Flash--plain", '
              '"jobs_wait bench", "bench_run sweep without the head, --label-suffix -nodraft '
              '-> Qwen3.8-Flash--nodraft-plain", "jobs_wait bench", "bench_run run '
              'Qwen3.8-Flash--ollama-plain --base-url http://127.0.0.1:11434", "jobs_wait '
              'bench", "bench_compare --export compare.json", "bench_animate compare.json"])')),
    "done": (("run benchmarks with qwen3.8-flash-next",
              'done(summary="Measured 10 questions as Qwen3.8-Flash--plain: 83% F1 at 44 s a '
              'question; the table is under the bench home, `bench_show` prints it.")'),
             ("make an animation of the last comparison",
              'done(summary="compare.mp4 written next to compare.json in the bench home; '
              'three panels, thirty seconds.")')),
}


def _prose(text: str | None) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _describe(name: str, description: str, doc: str, schema: dict[str, Any]) -> str:
    """One description a small model can act on: what it does, then worked calls."""
    head = _prose(doc) or _prose(description)
    pairs = EXAMPLES.get(name)
    if not pairs:
        needed = ", ".join(f"{k}=..." for k in schema.get("required", []))
        pairs = ((f"use {name.replace('_', ' ')}", f"{name}({needed})"),)
    return head + worked(*pairs)


def _schema(name: str, description: str, fn: Callable[..., Any],
            doc: str = "") -> dict[str, Any]:
    params = mcp.schema_of(fn)
    return {"type": "function", "function": {
        "name": name, "description": _describe(name, description, doc, params),
        "parameters": params}}


# -- the bench subcommands that follow the CLI -------------------------------------------
BENCH_SUBS: dict[str, bool] = {"compare": False, "animate": False, "standard": True,
                               "speed": True}
"""Subcommand -> whether it measures (and so detaches) or reads the store (and returns)."""


def bench_cli(sub: str, args: Sequence[str], detach: bool) -> dict[str, Any]:
    """``ml-stack-bench sub args``: detached with a log and pid when it measures, else run
    in-process with what it printed."""
    if detach:
        from ml_stack.graph.bench.run import detach as start

        log = start([sub, *args])
        from ml_stack import jobs
        from ml_stack.graph import bench

        held = jobs.held("bench", home=bench.HOME / "jobs")
        return {"log": str(log), "pid": held.get("pid"), "argv": [sub, *args]}
    from ml_stack.graph.bench.run import _main

    return mcp._captured(lambda: _main([sub, *list(args)]))


def _bench_tool(sub: str, detach: bool) -> mcp.Tool:
    def fn(args: list[str] = []) -> dict[str, Any]:
        return bench_cli(sub, list(args), detach)

    fn.__name__ = f"bench_{sub}"
    what = ("Start it detached; returns the log and pid, and jobs_wait waits for it."
            if detach else "Runs it and returns what it printed.")
    return mcp.Tool(f"bench_{sub}",
                    f"ml-stack-bench {sub} ARGS, exactly as the command line takes them "
                    f"(follows the CLI: `ml-stack-bench {sub} --help` says what). {what}", fn)


# -- jobs -------------------------------------------------------------------------------
def _jobs_home(kind: str) -> Path | None:
    if kind == "bench":
        from ml_stack.graph import bench

        return bench.HOME / "jobs"
    return None


def jobs_status() -> dict[str, Any]:
    """Every long command this machine records -- the bench's and the rest -- running or
    ended, since when, with its log."""
    from ml_stack import jobs
    from ml_stack.graph import bench

    said: list[str] = []
    for home in (jobs.HOME, bench.HOME / "jobs"):
        jobs.status(say=said.append, home=home)
    return {"text": "\n".join(said)}


def jobs_wait(kind: str = "bench", every: float = 60.0) -> dict[str, Any]:
    """Block until the recorded ``kind`` job (``bench``, ``ingest``, ``train``) has ended,
    then return; the way to follow a command that detached."""
    from ml_stack import jobs

    said: list[str] = []
    code = jobs.wait(kind, say=said.append, every=every, home=_jobs_home(kind))
    return {"kind": kind, "ended": code == 0, "said": said}


# -- the lookups ------------------------------------------------------------------------
_SHARD = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$", re.IGNORECASE)
_HEAD = ("mtp-", "eagle3-")


def _key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _matches(words: str, name: str) -> bool:
    haystack = _key(name)
    return all(_key(w) in haystack for w in words.split() if _key(w))


def gguf_files() -> list[Path]:
    """Every GGUF under the roots this machine keeps models in."""
    from ml_stack.fleet.models import default_roots

    found: list[Path] = []
    for root in default_roots(Path("~/.ml-stack")):
        if root.is_dir():
            found.extend(p for p in sorted(root.rglob("*.gguf")) if p.is_file())
    return found


def models_on_disk(words: str = "", files: Sequence[Path] | None = None) -> list[dict[str, Any]]:
    """The GGUF weights this machine holds whose name has every word of ``words``, each
    with the draft head (``mtp-``, ``eagle3-``, ``.draft``) and the ``mmproj`` projector
    kept beside it, the first shard standing for a sharded file."""
    from ml_stack.fleet.models import DRAFT_MARK

    every = [Path(p) for p in (files if files is not None else gguf_files())]
    by_dir: dict[Path, list[Path]] = {}
    for path in every:
        by_dir.setdefault(path.parent, []).append(path)
    out: list[dict[str, Any]] = []
    for path in every:
        name = path.name
        low = name.lower()
        if Path(name).suffix.lower() != ".gguf" or DRAFT_MARK in path.suffixes:
            continue
        if low.startswith(_HEAD) or low.startswith("mmproj") or low.startswith("imatrix"):
            continue
        shard = _SHARD.search(name)
        if shard and int(shard.group(1)) != 1:
            continue
        if not _matches(words, name):
            continue
        beside = by_dir.get(path.parent, [])
        heads = [p.name for p in beside if p.name.lower().startswith(_HEAD)]
        marked = path.with_suffix(DRAFT_MARK + path.suffix)
        if marked in beside:
            heads.append(marked.name)
        projectors = [p.name for p in beside if p.name.lower().startswith("mmproj")]
        try:
            size = path.stat().st_size
        except OSError:
            size = 0
        out.append({"model": name, "path": str(path), "bytes": size, "backend": "llama.cpp",
                    "shards": int(shard.group(2)) if shard else 1,
                    "draft": heads[0] if heads else "",
                    "mmproj": projectors[0] if projectors else ""})
    return out


def _ollama_fetch(path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(f"{OLLAMA_URL}{path}", data=data,
                                     headers={"Content-Type": "application/json"},
                                     method="POST" if data is not None else "GET")
    with urllib.request.urlopen(request, timeout=5) as answer:  # noqa: S310 - loopback
        return json.loads(answer.read().decode("utf-8"))


def ollama_models(words: str = "",
                  fetch: Callable[..., dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """The models Ollama holds whose name has every word of ``words``, each with its
    format, quantisation, family and size from ``/api/show``; one row saying so when
    Ollama is not answering."""
    ask = fetch or _ollama_fetch
    try:
        tags = ask("/api/tags")
    except Exception as exc:  # noqa: BLE001 - not running is an answer, not a failure
        return [{"error": f"ollama is not answering at {OLLAMA_URL}: {exc}"}]
    out: list[dict[str, Any]] = []
    for row in tags.get("models") or []:
        name = str(row.get("name") or row.get("model") or "")
        if not name or not _matches(words, name):
            continue
        details = dict(row.get("details") or {})
        info: dict[str, Any] = {}
        try:
            shown = ask("/api/show", {"model": name})
            details = {**details, **dict(shown.get("details") or {})}
            info = dict(shown.get("model_info") or {})
        except Exception:  # noqa: BLE001 - the tag line already says most of it
            pass
        out.append({"name": name, "backend": "ollama", "bytes": int(row.get("size") or 0),
                    "format": str(details.get("format") or ""),
                    "quantization": str(details.get("quantization_level") or ""),
                    "family": str(details.get("family") or ""),
                    "parameters": str(details.get("parameter_size") or ""),
                    "architecture": str(info.get("general.architecture")
                                        or details.get("family") or "")})
    return out


# -- the registry -----------------------------------------------------------------------
def command_tools(registry: Sequence[mcp.Tool] | None = None, *,
                  files: Sequence[Path] | None = None,
                  fetch: Callable[..., dict[str, Any]] | None = None
                  ) -> list[tuple[dict[str, Any], Callable[..., Any]]]:
    """``[(schema, callable), ...]``: every `ml_stack.mcp` tool, the bench subcommands the
    registry lacks, the jobs, and the two lookups. ``files`` and ``fetch`` are the
    lookups' sources, for a test."""
    listed = list(mcp.TOOLS if registry is None else registry)
    names = {t.name for t in listed}
    for sub, detach in BENCH_SUBS.items():
        if f"bench_{sub}" not in names:
            listed.append(_bench_tool(sub, detach))
    if "jobs_status" not in names:
        listed.append(mcp.Tool("jobs_status", "Every long command recorded: running or ended.",
                               jobs_status))
    if "jobs_wait" not in names:
        listed.append(mcp.Tool("jobs_wait", "Wait for a detached command to end.", jobs_wait))

    def on_disk(words: str = "") -> list[dict[str, Any]]:
        return models_on_disk(words, files=files)

    def in_ollama(words: str = "") -> list[dict[str, Any]]:
        return ollama_models(words, fetch=fetch)

    on_disk.__doc__, in_ollama.__doc__ = models_on_disk.__doc__, ollama_models.__doc__
    listed.append(mcp.Tool("models_on_disk",
                           "The GGUF weights on this machine, with the draft head and "
                           "projector beside each.", on_disk))
    listed.append(mcp.Tool("ollama_models", "The models Ollama holds, with format and "
                                            "quantisation.", in_ollama))
    return [(_schema(t.name, t.description, t.fn, inspect.getdoc(t.fn) or ""), t.fn)
            for t in listed]


# -- the person -------------------------------------------------------------------------
class Person:
    """The three tools that reach the person at the terminal, and what they said."""

    def __init__(self, stdin: TextIO, stdout: TextIO, *, yes: bool = False) -> None:
        self.stdin, self.stdout, self.yes = stdin, stdout, yes
        self.left = False
        self.finished = False
        self.summary = ""

    def say(self, text: str = "") -> None:
        self.stdout.write(text + "\n")
        self.stdout.flush()

    def _read(self, prompt: str) -> str | None:
        self.stdout.write(prompt)
        self.stdout.flush()
        line = self.stdin.readline()
        if line == "":
            self.left = True
            self.say("\n(input ended; the person has left)")
            return None
        return line.strip()

    def ask_user(self, question: str, choices: list[str] = []) -> dict[str, Any]:
        """Put one question to the person and wait for one line; numbered ``choices`` are
        answered by number or in words. Exactly one question per call."""
        self.say(f"\n? {question}")
        for n, choice in enumerate(choices or [], start=1):
            self.say(f"  {n}. {choice}")
        got = self._read("> ")
        if got is None:
            return {"answer": "", "note": "input ended: the person has left; stop here"}
        if choices and got.isdigit() and 1 <= int(got) <= len(choices):
            got = str(choices[int(got) - 1])
        return {"answer": got}

    def plan(self, steps: list[str]) -> dict[str, Any]:
        """Print the steps in order and ask the person \"go?\" once; with --yes the plan is
        printed and taken as agreed."""
        self.say("\nplan:")
        for n, step in enumerate(steps or [], start=1):
            self.say(f"  {n}. {step}")
        if self.yes:
            return {"go": True, "said": "--yes: the plan is agreed, act on it"}
        got = self._read("go? [y/N] ")
        if got is None:
            return {"go": False, "said": "input ended: the person has left; stop here"}
        if got.lower() in ("y", "yes", "go", "ok"):
            return {"go": True, "said": "go"}
        return {"go": False, "said": f"The person said: {got!r}. Change the plan or ask."}

    def done(self, summary: str) -> dict[str, Any]:
        """End the task: ``summary`` is what was measured and where it is."""
        self.finished, self.summary = True, summary
        self.say(f"\n{summary}")
        return {"ended": True}

    def tools(self) -> list[tuple[dict[str, Any], Callable[..., Any]]]:
        return [(_schema(name, "", fn, inspect.getdoc(fn) or ""), fn)
                for name, fn in (("ask_user", self.ask_user), ("plan", self.plan),
                                 ("done", self.done))]


OWN = ("ask_user", "plan", "done")


def own_tools(*, stdin: TextIO, stdout: TextIO,
              yes: bool = False) -> list[tuple[dict[str, Any], Callable[..., Any]]]:
    """The loop's own three tools, bound to a person on ``stdin``/``stdout``."""
    return Person(stdin, stdout, yes=yes).tools()


# -- the loop ---------------------------------------------------------------------------
@dataclass
class Outcome:
    done: bool = False
    summary: str = ""
    rounds: int = 0
    seconds: float = 0.0
    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    messages: list[dict[str, Any]] = field(default_factory=list)


def _compact(args: dict[str, Any], most: int = 160) -> str:
    text = ", ".join(f"{k}={json.dumps(v, ensure_ascii=False)}" for k, v in args.items())
    return text if len(text) <= most else text[: most - 3] + "..."


def transcript(messages: Iterable[dict[str, Any]]) -> str:
    """The conversation as lines a person reads: role, then what was said or called."""
    lines: list[str] = []
    for m in messages:
        role = m.get("role", "")
        if role == "system":
            continue
        calls = m.get("tool_calls") or []
        if calls:
            for call in calls:
                fn = call.get("function") or {}
                lines.append(f"{role}: -> {fn.get('name')}({fn.get('arguments', '')})")
        text = _prose(m.get("content"))
        if text:
            name = f" {m['name']}" if role == "tool" and m.get("name") else ""
            lines.append(f"{role}{name}: {text[:300]}")
    return "\n".join(lines)


def run(task: str, client: Any, *,
        tools: Sequence[tuple[dict[str, Any], Callable[..., Any]]] | None = None,
        stdin: TextIO | None = None, stdout: TextIO | None = None, yes: bool = False,
        rounds: int = ROUNDS, messages: list[dict[str, Any]] | None = None) -> Outcome:
    """One task through the loop: the model is offered every tool, each call is run and
    answered, ``ask_user`` and ``plan`` reach the person, ``done`` ends it. ``messages``
    carries a conversation across tasks; a new one is started when none is given."""
    person = Person(stdin or sys.stdin, stdout or sys.stdout, yes=yes)
    offered = [*(command_tools() if tools is None else tools), *person.tools()]
    schemas = [schema for schema, _ in offered]
    run_by = {schema["function"]["name"]: fn for schema, fn in offered}
    if messages is None:
        messages = [{"role": "system", "content": system_for(yes)}]
    messages.append({"role": "user", "content": task})
    out = Outcome(messages=messages)
    began = time.monotonic()
    nudged = False
    exhausted = True
    for _ in range(rounds):
        reply = client.chat(messages, think=False, tools=schemas)
        calls = list(getattr(reply, "tool_calls", None) or [])
        content = getattr(reply, "content", "") or ""
        if not calls:
            if content.strip():
                person.say(content.strip())
            messages.append({"role": "assistant", "content": content})
            if nudged:
                exhausted = False
                break
            nudged = True
            messages.append({"role": "user", "content": NUDGE})
            continue
        out.rounds += 1
        messages.append({"role": "assistant", "content": content, "tool_calls": calls})
        for call in calls:
            fn = call.get("function") or {}
            name = str(fn.get("name") or "")
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except ValueError:
                args = {}
            if not isinstance(args, dict):
                args = {}
            if name not in OWN:
                person.say(f"-> {name}({_compact(args)})")
            do = run_by.get(name)
            if do is None:
                result: Any = {"error": f"no such tool: {name}"}
            else:
                try:
                    result = do(**args)
                except Exception as exc:  # noqa: BLE001 - the error is the answer
                    result = {"error": f"{type(exc).__name__}: {exc}"}
            out.calls.append((name, args))
            text = json.dumps(mcp._plain(result), ensure_ascii=False, default=str)
            if name not in OWN:
                person.say("   " + (text if len(text) <= 300 else text[:297] + "..."))
            messages.append({"role": "tool", "tool_call_id": call.get("id") or name,
                             "name": name, "content": text[:CUT]})
            if person.finished or person.left:
                break
        if person.finished or person.left:
            exhausted = False
            break
    out.seconds = round(time.monotonic() - began, 2)
    out.done, out.summary = person.finished, person.summary
    if exhausted:
        person.say(f"\nran out of {rounds} rounds without done; the transcript:")
        person.say(transcript(messages))
    return out


# -- the command ------------------------------------------------------------------------
def client_for(args: argparse.Namespace) -> Any:
    """A client on the served model: ``--url`` for one already up, ``--model`` leased in
    its measured shape on one seat."""
    if args.url:
        from ml_stack.client import Client

        return Client(args.url, n_predict=args.n_predict, timeout=args.timeout)
    from ml_stack.graph.bench.serve import find_model
    from ml_stack.serve.profile import profile_for, said
    from ml_stack.serve.shape import Run, Shape, seat

    found = str(find_model(args.model))
    measured = profile_for(found)
    if measured is not None:
        print(f"serving in its measured shape: {said(measured)}")
        run = measured.run(port=args.port, seats=1, model=found, n_predict=args.n_predict,
                           timeout=args.timeout)
    else:
        run = Run(shape=Shape(model=found, port=args.port, seats=1, seat_context=32768,
                              reasoning_budget=0))
    return seat(run, index=0)


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="ml-stack-do", allow_abbrev=False,
        description="A task in words, done by a served model with the ml-stack commands as "
                    "tools: it asks what the task leaves open, one question at a time, "
                    "plans, asks go, acts, and reports what was measured and where it is. "
                    "With no TASK, tasks are read from stdin until EOF.")
    ap.add_argument("task", nargs="?", default="", metavar="TASK")
    which = ap.add_mutually_exclusive_group()
    which.add_argument("--model", default="", help="a model to lease in its measured shape")
    which.add_argument("--url", default="", help="a server already up, e.g. "
                                                 "http://127.0.0.1:8080")
    ap.add_argument("--port", type=int, default=8080, help="where --model is served")
    ap.add_argument("--yes", action="store_true",
                    help="the plan runs without asking go; the questions still come")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the system prompt and the tools as the model sees them")
    ap.add_argument("--rounds", type=int, default=ROUNDS,
                    help="tool-calling turns a task may spend (default: %(default)s)")
    ap.add_argument("--n-predict", type=int, default=N_PREDICT,
                    help="the ceiling on one reply (default: %(default)s)")
    ap.add_argument("--timeout", type=float, default=900.0,
                    help="seconds to wait for one reply (default: %(default)s)")
    return ap


def _print_offer(tools: Sequence[tuple[dict[str, Any], Any]], out: TextIO) -> None:
    for schema, _ in tools:
        fn = schema["function"]
        props = fn["parameters"].get("properties", {})
        required = fn["parameters"].get("required", [])
        shown = ", ".join(f"{k}{'' if k in required else '?'}" for k in props)
        out.write(f"{fn['name']}({shown})\n")
        out.write(textwrap.fill(fn["description"], width=96, initial_indent="    ",
                                subsequent_indent="    ") + "\n")


def main(argv: Sequence[str] | None = None, *, stdin: TextIO | None = None,
         stdout: TextIO | None = None) -> int:
    args = parser().parse_args(list(sys.argv[1:] if argv is None else argv))
    stdin, stdout = stdin or sys.stdin, stdout or sys.stdout
    tools = command_tools()
    if args.dry_run:
        stdout.write(system_for(args.yes) + "\n\n")
        _print_offer([*tools, *own_tools(stdin=stdin, stdout=stdout, yes=args.yes)], stdout)
        stdout.write(f"\n{len(tools) + len(OWN)} tools offered\n")
        return 0
    if not args.model and not args.url:
        parser().error("one of --model or --url is needed to reach a model")
    client = client_for(args)
    messages: list[dict[str, Any]] = [{"role": "system", "content": system_for(args.yes)}]
    if args.task:
        got = run(args.task, client, tools=tools, stdin=stdin, stdout=stdout, yes=args.yes,
                  rounds=args.rounds, messages=messages)
        return 0 if got.done else 1
    code = 0
    stdout.write("a task per line; EOF ends the session\n")
    while True:
        stdout.write("task> ")
        stdout.flush()
        line = stdin.readline()
        if line == "":
            break
        if not line.strip():
            continue
        got = run(line.strip(), client, tools=tools, stdin=stdin, stdout=stdout,
                  yes=args.yes, rounds=args.rounds, messages=messages)
        if not got.done:
            code = 1
    return code


if __name__ == "__main__":  # pragma: no cover - the entry point is `ml-stack-do`
    raise SystemExit(main())
