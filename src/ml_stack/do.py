"""The tools `ml-stack-chat` offers: every command as a tool, the lookups, and the person.

The tools are `ml_stack.mcp`'s (every command, long ones detached with a log and a pid),
plus the bench subcommands that follow the CLI, the jobs a detached command records, two
lookups (the GGUFs on this machine, the models Ollama holds), and three of the loop's own
on `Person`: ``ask_user`` puts one question to the person and waits, ``plan`` prints the
steps and asks "go?", ``done`` ends a task. Each tool's description carries worked examples,
which is what measured as making a small model call tools right.
"""

from __future__ import annotations

import argparse
import contextlib
import inspect
import json
import os
import re
import textwrap
from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path
from typing import Any, TextIO

from ml_stack import hub, mcp
from ml_stack.client import ollama
from ml_stack.interventions import Call, Confirm
from ml_stack.log import say
from ml_stack.rules import Rules, describe

__all__ = ["Person", "bench_cli", "client_for", "command_tools", "models_on_disk",
           "ollama_models", "own_tools"]

N_PREDICT = 16384
NONE: list[str] = []
"""The default of a list argument the tools only read."""
OLLAMA_URL = os.environ.get("OLLAMA_HOST") or "http://127.0.0.1:11434"

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
                   ("what did the last three runs score?", "bench_show(last=3)")),
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
                        'bench_standard(args=["--url", '
                        '"http://127.0.0.1:8080/v1/chat/completions", "--model", '
                        '"quince-2b.gguf", "--limit", "10"])'),
                       ("standard sets for flash-next, detached",
                        'bench_standard(args=["--url", '
                        '"http://127.0.0.1:8080/v1/chat/completions", "--model", '
                        '"Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf"]) then '
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
                   'jobs_wait(kind="bench") then bench_show()')),
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
              'kept as Qwen3.8-Flash--plain", "jobs_wait bench", "bench_show"])'),
             (ACCEPTANCE,
              'plan(steps=[\'bench_run ["sweep", "--serve", "Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf", '
              '"--serve-draft", "auto", "--plain-only", "--sample", "10"]\', "jobs_wait bench", '
              '\'bench_run ["sweep", "--serve", "Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf", '
              '"--serve-draft", "", "--label-suffix", "-nodraft", "--plain-only", "--sample", '
              '"10"]\', "jobs_wait bench", \'bench_run ["run", "Qwen3.8-Flash--ollama-plain", '
              '"--base-url", "http://127.0.0.1:11434", "--sample", "10"]\', "jobs_wait bench", '
              '\'bench_compare ["Qwen3.8-Flash--plain", "Qwen3.8-Flash--nodraft-plain", '
              '"Qwen3.8-Flash--ollama-plain", "--export", "compare.json"]\', '
              '\'bench_animate ["compare.json", "--out", "compare.mp4"]\'])')),
    "done": (("run benchmarks with qwen3.8-flash-next",
              'done(summary="Measured 10 questions as Qwen3.8-Flash--plain: 83% F1 at 44 s a '
              'question; the runs are under the bench home, `bench_show` reads them back.")'),
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
        from ml_stack.bench.underway import detach as start

        log = start([sub, *args])
        from ml_stack import bench, jobs

        record = jobs.recorded("bench", home=bench.home_dir() / "jobs")
        return {"log": str(log), "pid": record.get("pid"), "argv": [sub, *args]}
    from ml_stack.bench.run import _main

    return mcp._captured(lambda: _main([sub, *list(args)]))


def _bench_tool(sub: str, detach: bool) -> mcp.Tool:
    def fn(args: list[str] = NONE) -> dict[str, Any]:
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
        from ml_stack import bench

        return bench.home_dir() / "jobs"
    return None


def jobs_status() -> dict[str, Any]:
    """Every long command this machine records -- the bench's and the rest -- running or
    ended, since when, with its log."""
    from ml_stack import bench, jobs

    said: list[str] = []
    for home in (jobs.home_dir(), bench.home_dir() / "jobs"):
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


def models_on_disk(words: str = "", files: Sequence[Path] | None = None) -> list[dict[str, Any]]:
    """The GGUF weights this machine holds whose name has every word of ``words``, each
    with the draft head (``mtp-``, ``eagle3-``, ``.draft``) and the ``mmproj`` projector
    kept beside it -- in its own directory, or in a sibling directory of the same
    repository, where a Hub snapshot keeps its ``MTP/`` folder -- the first shard standing
    for a sharded file."""
    every = [q for q in (Path(p) for p in (files if files is not None else hub.weight_paths()))
             if q.suffix.lower() == ".gguf"]
    by_dir: dict[Path, list[Path]] = {}
    for path in every:
        by_dir.setdefault(path.parent, []).append(path)
        by_dir.setdefault(path.parent.parent, []).append(path)
    out: list[dict[str, Any]] = []
    for path in every:
        name = path.name
        low = name.lower()
        if hub.DRAFT_MARK in path.suffixes:
            continue
        if low.startswith(_HEAD) or low.startswith("mmproj") or low.startswith("imatrix"):
            continue
        shard = _SHARD.search(name)
        if shard and int(shard.group(1)) != 1:
            continue
        if not _matches(words, name):
            continue
        beside = by_dir.get(path.parent, [])
        # the head in this directory first, then one a sibling directory holds
        heads = [p.name for p in beside if p.name.lower().startswith(_HEAD)]
        heads += [p.name for p in by_dir.get(path.parent.parent, [])
                  if p.name.lower().startswith(_HEAD) and p.name not in heads]
        marked = path.with_suffix(hub.DRAFT_MARK + path.suffix)
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


def on_disk_ids() -> list[str]:
    """The names and paths of the weights on this machine."""
    return [str(row[key]) for row in models_on_disk() for key in ("model", "path")]


def ollama_models(words: str = "",
                  fetch: Callable[..., dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """The models Ollama holds whose name has every word of ``words``, each with its
    format, quantisation, family and size from ``/api/show``; one row saying so when
    Ollama is not answering."""
    ask = fetch or partial(ollama.fetch, base_url=OLLAMA_URL)
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
        with contextlib.suppress(Exception):  # the tag line already says most of it
            shown = ask("/api/show", {"model": name})
            details = {**details, **dict(shown.get("details") or {})}
            info = dict(shown.get("model_info") or {})
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

    def __init__(self, stdin: TextIO, stdout: TextIO, *, rules: Rules | None = None) -> None:
        self.stdin, self.stdout, self.rules = stdin, stdout, rules
        self.asked = 0
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

    def ask_user(self, question: str, choices: list[str] = NONE) -> dict[str, Any]:
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
        """Print the steps in order and ask the person \"go?\" once; each step names a tool and
        the arguments it will get, and the go covers those values."""
        self.say("\nplan:")
        for n, step in enumerate(steps or [], start=1):
            self.say(f"  {n}. {step}")
        got = self._read("go? [y/N] ")
        if got is None:
            return {"go": False, "said": "input ended: the person has left; stop here"}
        if got.lower() in ("y", "yes", "go", "ok"):
            return {"go": True, "said": "go"}
        return {"go": False, "said": f"The person said: {got!r}. Change the plan or ask."}

    def confirm(self, ask: Confirm, call: Call | None = None) -> bool:
        """Put an intervention's question to the person: allow this time, always allow, never
        allow, or (anything else) no. A rule is saved only on the person's own answer."""
        what = f"{call.name}({_compact(call.arguments or {})}): " if call is not None else ""
        self.asked += 1
        self.say(f"\n! {what}{ask.question}")
        can = self.rules is not None and call is not None
        always = can and bool(ask.details.get("always_ok"))
        if can and not always and ask.details.get("always_blocked"):
            self.say(f"  (no 'always allow' here: {ask.details['always_blocked']})")
        menu = "1) allow this time" + ("  2) always allow" if always else "") \
            + ("  3) never allow" if can else "") + "  [Enter = no]"
        got = (self._read(f"allow it? {menu} > ") or "").lower()
        if got in ("y", "yes", "1"):
            return True
        if can and got in ("3", "never"):
            return self._save(call, "never", ask)
        if always and got in ("2", "a", "always"):
            return self._save(call, "always", ask)
        return False

    def _save(self, call: Call, verdict: str, ask: Confirm) -> bool:
        """Say in words what the rule for ``call`` covers and save it (an always rule only
        after a yes); true to go ahead with this call."""
        role = str(ask.details.get("role") or "")
        try:
            rule = self.rules.make(call.name, call.arguments or {}, verdict, role)
            self.say(f"  this rule: {describe(rule)}")
            if verdict == "always" and (self._read("  save it? [y/N] > ") or "").lower() \
                    not in ("y", "yes"):
                self.say("  not saved; allowed this time")
                return True
            self.rules.add(call.name, call.arguments or {}, verdict, role)
        except (ValueError, OSError) as exc:
            self.say(f"  no rule saved: {exc}")
            return verdict == "always"
        self.say("  saved. /rules lists and edits the rules.")
        return verdict == "always"

    def done(self, summary: str) -> dict[str, Any]:
        """End the task: ``summary`` is what was measured and where it is."""
        self.finished, self.summary = True, summary
        self.say(f"\n{summary}")
        return {"ended": True}

    def tools(self) -> list[tuple[dict[str, Any], Callable[..., Any]]]:
        return [(_schema(name, "", fn, inspect.getdoc(fn) or ""), fn)
                for name, fn in (("ask_user", self.ask_user), ("plan", self.plan),
                                 ("done", self.done))]


def own_tools(*, stdin: TextIO, stdout: TextIO) -> list[tuple[dict[str, Any], Callable[..., Any]]]:
    """The loop's own three tools, bound to a person on ``stdin``/``stdout``."""
    return Person(stdin, stdout).tools()


def _compact(args: dict[str, Any], most: int = 160) -> str:
    text = ", ".join(f"{k}={json.dumps(v, ensure_ascii=False)}" for k, v in args.items())
    return text if len(text) <= most else text[: most - 3] + "..."



def client_for(args: argparse.Namespace) -> Any:
    """A client on the served model: ``--url`` for one already up, ``--model`` leased on
    one slot in the settings it scored best with."""
    if args.url:
        from ml_stack.client import Client, Request, Transport

        return Client(args.url, request=Request(n_predict=args.n_predict),
                      transport=Transport(timeout=args.timeout))
    from ml_stack.client import Client, Request, Transport
    from ml_stack.serve.leases import already_up
    from ml_stack.serve.profile import profile_for, said
    from ml_stack.serve.recent import note
    from ml_stack.serve.serving import Config, Serving, drafted, slot

    found = str(hub.located(args.model, loose=True) or args.model)
    note(found, by="chat")
    up = already_up(found, args.port)
    if up is not None:
        # the weights are up already, whatever the settings: use them rather than reload them
        say(f"using the server already up on {args.port}: {Path(found).name}, "
            f"{up.get('slots') or '?'} slot(s)")
        return Client(str(up["base_url"]), request=Request(n_predict=args.n_predict),
                      transport=Transport(timeout=args.timeout))
    measured = profile_for(found)
    if measured is not None:
        config = measured.alone(port=args.port, model=found, n_predict=args.n_predict,
                             timeout=args.timeout)
        say(f"serving alone, one slot of {config.serving.slot_context} tokens "
            f"({config.serving.note}): {said(measured)}")
        config = drafted(config, "none", say=say)
    else:
        config = Config(serving=Serving(model=found, port=args.port, slots=1, slot_context=32768,
                              reasoning_budget=0))
        config = drafted(config, args.draft, say=say)
    return slot(config, index=0)


def _print_offer(tools: Sequence[tuple[dict[str, Any], Any]], out: TextIO) -> None:
    for schema, _ in tools:
        fn = schema["function"]
        props = fn["parameters"].get("properties", {})
        required = fn["parameters"].get("required", [])
        shown = ", ".join(f"{k}{'' if k in required else '?'}" for k in props)
        out.write(f"{fn['name']}({shown})\n")
        out.write(textwrap.fill(fn["description"], width=96, initial_indent="    ",
                                subsequent_indent="    ") + "\n")
