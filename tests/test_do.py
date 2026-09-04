"""``ml-stack-do``: a task in words, a served model asking before assuming, then doing it.

The model is a `ScriptedModel`, the commands are a fake registry that records what it was
called with, the person is a string on stdin. Nothing here serves a model, touches a port or
reads ``~/.ml-stack``.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
from ml_stack import do, mcp
from ml_stack.testing import ScriptedModel
from ml_stack.testing.fakes import reply_from


def call(name, **args):
    return (name, args)


class Scripted(ScriptedModel):
    """`ScriptedModel` whose script may also hold words (an answer with no call) and
    callables asked with ``(messages, tools)``; a call is issued only when its tool is on
    offer."""

    def chat(self, messages, *, tools=None, **extra):
        self.seen.append(list(messages))
        if not self.script:
            return reply_from(self.answer, messages, tools)
        entry = self.script[0]
        offered = {str((t.get("function") or {}).get("name")) for t in (tools or [])}
        if isinstance(entry, tuple) and entry[0] not in offered:
            return reply_from(self.answer, messages, tools)
        return reply_from(self.script.pop(0), messages, tools)


def recording(name: str, seen: list, *, answer=None, **params):
    """A registry entry that records its arguments and answers ``answer``."""
    def fn(**args):
        seen.append((name, dict(args)))
        return answer if answer is not None else {"log": f"/tmp/{name}.log", "pid": 7}

    fn.__annotations__ = {**{k: v for k, v in params.items()}, "return": dict}
    fn.__signature__ = _signature(params)
    return mcp.Tool(name, f"The {name} command.", fn)


def _signature(params):
    import inspect

    return inspect.Signature([inspect.Parameter(k, inspect.Parameter.KEYWORD_ONLY,
                                                annotation=v) for k, v in params.items()])


def registry(seen: list) -> list[mcp.Tool]:
    """A fake ``mcp.TOOLS``: every bench tool the loop offers, none of which measures."""
    return [recording("bench_run", seen, argv=list[str]),
            recording("bench_status", seen, answer={"text": "nothing is measuring"}),
            recording("bench_compare", seen, args=list[str]),
            recording("bench_animate", seen, args=list[str]),
            recording("bench_standard", seen, args=list[str]),
            recording("bench_speed", seen, args=list[str]),
            recording("serve_status", seen, answer=[])]


def tools_over(seen: list, *, files=(), fetch=None):
    return do.command_tools(registry(seen), files=list(files), fetch=fetch)


def drive(script, stdin: str, *, task="run benchmarks with quince-2b", tools=None, seen=None,
          **kw):
    """One task through the loop with a scripted model and a person on stdin."""
    seen = [] if seen is None else seen
    model = Scripted(script, answer="I have nothing more to do.")
    out = io.StringIO()
    got = do.run(task, model, tools=tools if tools is not None else tools_over(seen),
                 stdin=io.StringIO(stdin), stdout=out, **kw)
    return got, model, seen, out.getvalue()


# -- the loop --------------------------------------------------------------------------

def test_a_task_that_leaves_a_choice_open_asks_then_plans_then_acts_then_reports():
    script = [call("ask_user", question="Full hundred questions (~45 min) or a sample of 10 "
                                        "(~5 min)?",
                   choices=["the hundred", "a sample of 10"]),
              call("plan", steps=["serve quince-2b", "bench_run a sample of 10",
                                  "report the table"]),
              call("bench_run", argv=["run", "quince-2b.gguf", "--sample", "10"]),
              call("done", summary="Measured 10 questions; the table is under the bench home.")]
    got, model, seen, printed = drive(script, "2\ny\n")

    assert got.done and "Measured 10 questions" in got.summary
    assert "Full hundred questions" in printed and "2. a sample of 10" in printed
    assert "1. serve quince-2b" in printed and "go?" in printed
    assert seen == [("bench_run", {"argv": ["run", "quince-2b.gguf", "--sample", "10"]})]
    told = model.told()
    assert "a sample of 10" in told, "the person's answer reached the model"
    assert "go" in told.lower()
    assert [c[0] for c in got.calls] == ["ask_user", "plan", "bench_run", "done"]
    assert "Measured 10 questions" in printed


def test_a_number_picks_a_choice_and_free_text_is_taken_as_said():
    script = [call("ask_user", question="Which?", choices=["larch", "quince"]),
              call("ask_user", question="Where to write?"),
              call("done", summary="ok")]
    got, model, _, printed = drive(script, "1\n~/out\n")
    answers = [json.loads(m["content"]) for turn in model.seen for m in turn
               if m.get("role") == "tool" and m.get("name") == "ask_user"]
    assert answers[0]["answer"] == "larch" and answers[-1]["answer"] == "~/out"


def test_yes_skips_the_go_but_not_the_question():
    script = [call("ask_user", question="Sample or full?", choices=["sample", "full"]),
              call("plan", steps=["bench_run a sample"]),
              call("bench_run", argv=["run", "quince-2b.gguf", "--sample", "10"]),
              call("done", summary="done")]
    got, model, seen, printed = drive(script, "1\n", yes=True)
    assert got.done and len(seen) == 1
    assert "Sample or full?" in printed
    assert "go?" not in printed
    assert "sample" in model.told()
    assert "--yes" in model.seen[0][0]["content"], "the model was told go is not asked"


def test_a_plan_the_person_refuses_is_told_to_the_model_and_nothing_runs():
    script = [call("plan", steps=["bench_run the hundred"]),
              call("done", summary="stopped")]
    got, model, seen, printed = drive(script, "no, the sample please\n")
    assert seen == []
    assert "no, the sample please" in model.told()
    assert got.done


def test_rounds_ends_a_loop_that_never_says_done_and_prints_the_transcript():
    forever = [call("bench_status") for _ in range(20)]
    got, model, seen, printed = drive(forever, "", rounds=3)
    assert not got.done and got.rounds == 3
    assert len(seen) == 3
    assert "transcript" in printed.lower()
    assert printed.count("bench_status") >= 3
    assert "run benchmarks with quince-2b" in printed


def test_a_person_who_leaves_ends_the_loop():
    script = [call("ask_user", question="Sample or full?"),
              call("bench_run", argv=["run", "x"])]
    got, model, seen, printed = drive(script, "")
    assert not got.done and seen == []
    assert "input ended" in printed.lower()


def test_prose_with_no_call_is_nudged_once_then_taken_as_the_end():
    script = ["I would run the sample.", "Still just talking."]
    got, model, seen, printed = drive(script, "")
    assert not got.done
    assert len(model.seen) == 2
    nudges = [m for turn in model.seen for m in turn
              if m.get("role") == "user" and "done" in m.get("content", "")]
    assert nudges, "the model was told how to finish"
    assert "Still just talking." in printed


# -- the tools -------------------------------------------------------------------------

def test_command_tools_carry_every_mcp_tool_and_the_bench_subcommands_that_follow_the_cli():
    offered = {s["function"]["name"] for s, _ in do.command_tools()}
    assert {t.name for t in mcp.TOOLS} <= offered
    assert {"bench_compare", "bench_animate", "bench_standard", "bench_speed"} <= offered
    assert {"jobs_status", "jobs_wait", "models_on_disk", "ollama_models"} <= offered
    assert {"ask_user", "plan", "done"}.isdisjoint(offered), "the loop's own are added by it"


def test_every_offered_tool_carries_a_worked_example():
    for schema, _ in [*do.command_tools(), *do.own_tools(stdin=io.StringIO(),
                                                          stdout=io.StringIO(), yes=False)]:
        text = schema["function"]["description"]
        assert "Example" in text and "->" in text, schema["function"]["name"]


def test_every_bench_example_parses_as_the_bench_command_line(capsys):
    import ast
    import re

    from ml_stack.graph.bench.run import _parser

    found = []
    for name, pairs in do.EXAMPLES.items():
        for _asked, said in pairs:
            for sub, args in re.findall(r"bench_(\w+)\(args=(\[[^\]]*\])\)", said):
                found.append((name, sub, ast.literal_eval(args)))
    assert found, "the examples name bench subcommands with their arguments"
    for name, sub, args in found:
        try:
            _parser().parse_args([sub, *args])
        except SystemExit:
            pytest.fail(f"{name}: `ml-stack-bench {sub} {' '.join(args)}` -- "
                        + capsys.readouterr().err.strip().splitlines()[-1])


def test_a_bench_subcommand_not_in_mcp_is_registered_once_and_calls_the_cli(monkeypatch):
    ran = []
    monkeypatch.setattr(do, "bench_cli", lambda sub, args, detach: ran.append((sub, args, detach))
                        or {"sub": sub})
    tools = dict((s["function"]["name"], fn) for s, fn in do.command_tools())
    assert tools["bench_animate"](args=["--out", "x.mp4"]) == {"sub": "animate"}
    assert tools["bench_standard"](args=["quince-2b.gguf"]) == {"sub": "standard"}
    assert ran == [("animate", ["--out", "x.mp4"], False), ("standard", ["quince-2b.gguf"], True)]
    names = [s["function"]["name"] for s, _ in do.command_tools()]
    assert len(names) == len(set(names))


def test_models_on_disk_lists_the_weights_with_the_head_and_projector_beside_them(tmp_path):
    shelf = tmp_path / "models"
    (shelf / "UD-Q4_K_XL").mkdir(parents=True)
    weights = [shelf / "UD-Q4_K_XL" / f"Qwen3.8-Flash-Next-UD-Q4_K_XL-0000{i}-of-00004.gguf"
               for i in (1, 2, 3, 4)]
    head = shelf / "UD-Q4_K_XL" / "mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf"
    proj = shelf / "UD-Q4_K_XL" / "mmproj-F16.gguf"
    other = shelf / "quince-2b.gguf"
    for p in (*weights, head, proj, other):
        p.write_bytes(b"gguf")
    found = do.models_on_disk("qwen3.8-flash-next", files=[*weights, head, proj, other])
    assert [f["model"] for f in found] == ["Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf"]
    assert found[0]["draft"] == "mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf"
    assert found[0]["mmproj"] == "mmproj-F16.gguf"
    assert found[0]["shards"] == 4 and found[0]["path"] == str(weights[0])
    assert found[0]["backend"] == "llama.cpp"
    assert [f["model"] for f in do.models_on_disk("quince", files=[*weights, head, proj, other])] \
        == ["quince-2b.gguf"]
    assert do.models_on_disk("larch", files=[other]) == []


def test_ollama_models_reads_the_tags_and_the_show_of_each_match():
    asked = []

    def fetch(path, payload=None):
        asked.append((path, payload))
        if path == "/api/tags":
            return {"models": [
                {"name": "qwen3.8-flash-next:125b-mlx", "size": 70_000_000_000,
                 "details": {"format": "safetensors", "family": "qwen3next",
                             "parameter_size": "125B", "quantization_level": "nvfp4"}},
                {"name": "quince:2b", "size": 1_500_000_000,
                 "details": {"format": "gguf", "family": "quince", "parameter_size": "2B",
                             "quantization_level": "Q4_K_M"}}]}
        if path == "/api/show":
            return {"details": {"format": "safetensors", "quantization_level": "nvfp4",
                                "family": "qwen3next", "parameter_size": "125B"},
                    "model_info": {"general.architecture": "qwen3next"}}
        raise AssertionError(path)

    found = do.ollama_models("qwen3.8 flash next", fetch=fetch)
    assert found == [{"name": "qwen3.8-flash-next:125b-mlx", "backend": "ollama",
                      "bytes": 70_000_000_000, "format": "safetensors",
                      "quantization": "nvfp4", "family": "qwen3next", "parameters": "125B",
                      "architecture": "qwen3next"}]
    assert asked == [("/api/tags", None), ("/api/show", {"model": "qwen3.8-flash-next:125b-mlx"})]


def test_ollama_that_is_not_running_is_an_empty_list_that_says_so():
    def fetch(path, payload=None):
        raise OSError("connection refused")

    found = do.ollama_models("quince", fetch=fetch)
    assert len(found) == 1 and "not answering" in found[0]["error"]


# -- the acceptance case ---------------------------------------------------------------

PROMPT = ("benchmark qwen3.8-flash-next with llama.cpp (both with draft head and no draft "
          "head) and with ollama, make some animations")


def shelf(tmp_path: Path) -> list[Path]:
    where = tmp_path / "models" / "UD-Q4_K_XL"
    where.mkdir(parents=True)
    files = [where / "Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf",
             where / "Qwen3.8-Flash-Next-UD-Q4_K_XL-00002-of-00004.gguf",
             where / "mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf",
             tmp_path / "models" / "quince-2b.gguf"]
    for p in files:
        p.write_bytes(b"gguf")
    return files


def ollama_fake(path, payload=None):
    if path == "/api/tags":
        return {"models": [{"name": "qwen3.8-flash-next:125b-mlx", "size": 7,
                            "details": {"format": "safetensors", "family": "qwen3next",
                                        "parameter_size": "125B",
                                        "quantization_level": "nvfp4"}}]}
    return {"details": {"format": "safetensors", "quantization_level": "nvfp4",
                        "family": "qwen3next", "parameter_size": "125B"}}


def _found(messages, name):
    """The last result ``name`` returned, as the model saw it."""
    for m in reversed(messages):
        if m.get("role") == "tool" and m.get("name") == name:
            return json.loads(m["content"])
    return None


def confirm_models(messages, tools):
    """The one ask_user that lists what both lookups found."""
    disk, ollama = _found(messages, "models_on_disk"), _found(messages, "ollama_models")
    assert disk and ollama, "both lookups came back before the question was asked"
    return call("ask_user",
                question=f"llama.cpp: {disk[0]['model']} (draft head {disk[0]['draft']} "
                         f"found); Ollama: {ollama[0]['name']} ({ollama[0]['format']}, "
                         f"{ollama[0]['quantization']}). Use these?",
                choices=["yes", "no"])


ACCEPTANCE = [
    call("models_on_disk", words="qwen3.8-flash-next"),
    call("ollama_models", words="qwen3.8-flash-next"),
    confirm_models,
    call("ask_user", question="graph Q&A on the invented community -- sample of 10 (~5 min) "
                              "or the hundred (~45 min)? plus speed matrix and standard sets?",
         choices=["sample of 10", "the hundred", "sample plus speed and standard"]),
    call("ask_user", question="one comparison video of all three, or a clip per panel?",
         choices=["one video", "a clip per panel"]),
    call("plan", steps=["bench_run sweep flash-next with its head -> Qwen3.8-Flash--plain",
                        "bench_run sweep without the head -> Qwen3.8-Flash--nodraft-plain",
                        "bench_run run Qwen3.8-Flash--ollama-plain against ollama",
                        "jobs_wait bench", "bench_compare --export", "bench_animate"]),
    call("bench_run", argv=["sweep", "--serve", "Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf",
                            "--serve-draft", "auto", "--plain-only", "--sample", "10"]),
    call("bench_run", argv=["sweep", "--serve", "Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf",
                            "--serve-draft", "", "--label-suffix", "-nodraft", "--plain-only",
                            "--sample", "10"]),
    call("bench_run", argv=["run", "Qwen3.8-Flash--ollama-plain", "--base-url",
                            "http://127.0.0.1:11434", "--sample", "10"]),
    call("jobs_wait", kind="bench"),
    call("bench_compare", args=["Qwen3.8-Flash--plain", "Qwen3.8-Flash--nodraft-plain",
                                "Qwen3.8-Flash--ollama-plain", "--export", "compare.json"]),
    call("bench_animate", args=["compare.json", "--out", "compare.mp4"]),
    call("done", summary="Three runs measured over the sample; compare.json and compare.mp4 "
                         "are in the bench home."),
]


def test_the_acceptance_prompt_looks_up_both_backends_confirms_asks_twice_more_plans_then_runs(
        tmp_path):
    seen: list = []
    tools = tools_over(seen, files=shelf(tmp_path), fetch=ollama_fake)
    got, model, seen, printed = drive(list(ACCEPTANCE), "1\n1\n1\ny\n", task=PROMPT,
                                      tools=tools, seen=seen)
    names = [c[0] for c in got.calls]
    assert names == ["models_on_disk", "ollama_models", "ask_user", "ask_user", "ask_user",
                     "plan", "bench_run", "bench_run", "bench_run", "jobs_wait",
                     "bench_compare", "bench_animate", "done"]
    first = next(args for name, args in got.calls if name == "ask_user")
    assert "Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf" in first["question"]
    assert "mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf" in first["question"]
    assert "qwen3.8-flash-next:125b-mlx" in first["question"] and "nvfp4" in first["question"]
    assert "Use these?" in printed and "go?" in printed
    runs = [a["argv"] for n, a in seen if n == "bench_run"]
    assert runs[0][:2] == ["sweep", "--serve"] and "auto" in runs[0], "with the head"
    assert "--label-suffix" in runs[1] and "-nodraft" in runs[1], "without the head"
    assert runs[2][:2] == ["run", "Qwen3.8-Flash--ollama-plain"] and "11434" in runs[2][3]
    assert ("bench_compare", {"args": ["Qwen3.8-Flash--plain", "Qwen3.8-Flash--nodraft-plain",
                                       "Qwen3.8-Flash--ollama-plain", "--export",
                                       "compare.json"]}) in seen
    assert seen[-1][0] == "bench_animate"
    assert got.done and "compare.mp4" in printed


def test_with_yes_the_models_are_still_looked_up_and_confirmed_but_nobody_is_asked_go(
        tmp_path):
    seen: list = []
    tools = tools_over(seen, files=shelf(tmp_path), fetch=ollama_fake)
    got, model, seen, printed = drive(list(ACCEPTANCE), "1\n1\n1\n", task=PROMPT,
                                      tools=tools, seen=seen, yes=True)
    names = [c[0] for c in got.calls]
    assert names[:5] == ["models_on_disk", "ollama_models", "ask_user", "ask_user", "ask_user"]
    assert "Use these?" in printed and "go?" not in printed
    assert got.done and [n for n, _ in seen if n == "bench_run"] == ["bench_run"] * 3


# -- the command -----------------------------------------------------------------------

def test_dry_run_prints_the_system_prompt_and_every_tool_with_a_worked_example(capsys):
    assert do.main(["--dry-run", "run benchmarks with quince-2b"]) == 0
    printed = capsys.readouterr().out
    assert do.SYSTEM.splitlines()[0] in printed
    blocks = {}
    for line in printed.splitlines():
        if line and not line.startswith((" ", "\t")) and "(" in line and line.endswith(")"):
            current = line.split("(", 1)[0].strip()
            blocks[current] = ""
        elif line.startswith((" ", "\t")) and blocks:
            blocks[current] += line
    for name in [*(t.name for t in mcp.TOOLS), "ask_user", "plan", "done", "jobs_wait",
                 "models_on_disk", "ollama_models", "bench_animate"]:
        assert name in blocks, name
        assert "Example" in blocks[name] and "->" in blocks[name], name
    assert "ask before" in do.SYSTEM.lower() or "ask first" in do.SYSTEM.lower()
    assert "one question" in do.SYSTEM.lower()
    assert "--yes" in do.SYSTEM


def test_the_system_prompt_says_a_task_naming_two_backends_confirms_the_models():
    text = do.SYSTEM.lower()
    assert "two backends" in text or "more than one backend" in text
    assert "confirm" in text


def test_main_runs_one_task_over_the_client_it_is_given(monkeypatch, capsys):
    seen: list = []
    script = [call("ask_user", question="Sample or full?", choices=["sample", "full"]),
              call("plan", steps=["bench_run a sample"]),
              call("bench_run", argv=["run", "quince-2b.gguf", "--sample", "10"]),
              call("done", summary="Measured the sample.")]
    model = Scripted(script, answer="")
    monkeypatch.setattr(do, "client_for", lambda args: model)
    tools = tools_over(seen)
    monkeypatch.setattr(do, "command_tools", lambda *a, **k: tools)
    code = do.main(["run benchmarks with quince-2b", "--url", "http://127.0.0.1:1"],
                   stdin=io.StringIO("1\ny\n"))
    assert code == 0
    printed = capsys.readouterr().out
    assert "Measured the sample." in printed
    assert seen == [("bench_run", {"argv": ["run", "quince-2b.gguf", "--sample", "10"]})]


def test_main_with_no_task_reads_tasks_from_stdin_until_eof(monkeypatch, capsys):
    seen: list = []
    script = [call("done", summary="first done"), call("done", summary="second done")]
    model = Scripted(script, answer="")
    monkeypatch.setattr(do, "client_for", lambda args: model)
    tools = tools_over(seen)
    monkeypatch.setattr(do, "command_tools", lambda *a, **k: tools)
    code = do.main(["--url", "http://127.0.0.1:1"],
                   stdin=io.StringIO("show me what is serving\n\nwhat ran today?\n"))
    assert code == 0
    printed = capsys.readouterr().out
    assert "first done" in printed and "second done" in printed
    tasks = [m["content"] for m in model.seen[-1] if m.get("role") == "user"]
    assert "show me what is serving" in tasks and "what ran today?" in tasks, \
        "one conversation across the tasks"


def test_main_running_out_of_rounds_exits_nonzero(monkeypatch, capsys):
    seen: list = []
    model = Scripted([call("bench_status") for _ in range(9)], answer="")
    monkeypatch.setattr(do, "client_for", lambda args: model)
    tools = tools_over(seen)
    monkeypatch.setattr(do, "command_tools", lambda *a, **k: tools)
    code = do.main(["look", "--url", "http://127.0.0.1:1", "--rounds", "2"],
                   stdin=io.StringIO(""))
    assert code == 1
    assert "transcript" in capsys.readouterr().out.lower()


def test_model_and_url_are_one_or_the_other(capsys):
    with pytest.raises(SystemExit) as left:
        do.main(["look", "--model", "quince-2b.gguf", "--url", "http://127.0.0.1:1"])
    assert left.value.code == 2


def test_a_task_with_no_model_serves_the_best_measured_one_on_this_disk(monkeypatch, tmp_path, capsys):
    """`ml-stack-do "task"` alone: the profile with the highest F1 whose weights are here,
    said out loud; a better-measured model that is not on disk is passed over."""
    from ml_stack import do
    from ml_stack.serve.profile import Profile

    here = tmp_path / "quince-2b.gguf"
    here.write_bytes(b"gguf")
    records = [Profile(model="glimmer-9b.gguf", questions=100, right=0.9),
               Profile(model="quince-2b.gguf", questions=100, right=0.4),
               Profile(model="ember-1b.gguf", questions=2, right=0.99)]
    monkeypatch.setattr("ml_stack.serve.profile.profiles", lambda **_: records)
    monkeypatch.setattr("ml_stack.graph.bench.serve.find_model",
                        lambda name: str(here) if name == "quince-2b.gguf" else name)
    chosen = do.best_on_disk()
    assert chosen is not None and chosen[0].model == "quince-2b.gguf" and chosen[1] == str(here)

    seen = {}

    def fake_client(args):
        seen["model"] = args.model
        raise SystemExit(0)

    monkeypatch.setattr(do, "client_for", fake_client)
    with pytest.raises(SystemExit):
        do.main(["look"])
    assert seen["model"] == str(here)
    assert "no --model given: quince-2b.gguf, the best measured on this machine (40% F1 over 100 questions)" in capsys.readouterr().out


def test_a_task_with_no_model_and_nothing_measured_on_disk_still_asks_for_one(monkeypatch):
    from ml_stack import do

    monkeypatch.setattr("ml_stack.serve.profile.profiles", lambda **_: [])
    with pytest.raises(SystemExit) as told:
        do.main(["look"])
    assert told.value.code == 2


def test_a_model_already_up_on_the_port_is_used_as_it_stands(monkeypatch, tmp_path, capsys):
    """The weights already up in another shape are used, not reloaded into one seat."""
    from ml_stack import do

    here = tmp_path / "quince-2b.gguf"
    here.write_bytes(b"gguf")
    monkeypatch.setattr("ml_stack.graph.bench.serve.find_model", lambda name: str(here))
    monkeypatch.setattr("ml_stack.serve.manager.already_up",
                        lambda model, port, **_: {"base_url": "http://127.0.0.1:8080", "slots": 2,
                                                  "model": str(here), "pid": 1})
    built = {}

    class FakeClient:
        def __init__(self, url, **kw):
            built["url"] = url

    monkeypatch.setattr("ml_stack.client.Client", FakeClient)
    client = do.client_for(do.parser().parse_args(["look", "--model", "quince-2b.gguf"]))
    assert isinstance(client, FakeClient) and built["url"] == "http://127.0.0.1:8080"
    assert "using the server already up on 8080: quince-2b.gguf, 2 slot(s)" in capsys.readouterr().out
