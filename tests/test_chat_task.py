"""``ml-stack-chat "task"``: the old do behaviour as a role. A task asks what it leaves open,
shows its plan once, asks go, runs the calls the plan names without asking again, asks about
anything else, ends on done and prints what it cost."""

from __future__ import annotations

import io
import json
from pathlib import Path

from ml_stack import chat, do
from ml_stack.guard.untrusted import unfenced
from tests.test_do import Scripted, call, tools_over

PLAN_RUN = 'bench_run ["run", "quince-2b.gguf", "--sample", "10"]'
RUN = call("bench_run", argv=["run", "quince-2b.gguf", "--sample", "10"])


def drive(script, stdin: str, *, task="run benchmarks with quince-2b", tools=None, seen=None,
          **kw):
    """One task through the loop with a scripted model and a person on stdin."""
    seen = [] if seen is None else seen
    model = Scripted(script, answer="I have nothing more to do.")
    out = io.StringIO()
    got = chat.run_task(task, model, tools=tools if tools is not None else tools_over(seen),
                        person=do.Person(io.StringIO(stdin), out), **kw)
    return got, model, seen, out.getvalue()


def test_a_task_that_leaves_a_choice_open_asks_then_plans_then_acts_then_reports():
    script = [call("ask_user", question="Full hundred questions (~45 min) or a sample of 10 "
                                        "(~5 min)?",
                   choices=["the hundred", "a sample of 10"]),
              call("plan", steps=["serve quince-2b", PLAN_RUN, "report the table"]),
              RUN,
              call("done", summary="Measured 10 questions; the table is under the bench home.")]
    got, model, seen, printed = drive(script, "2\ny\n")

    assert got.done and "Measured 10 questions" in got.summary
    assert "Full hundred questions" in printed and "2. a sample of 10" in printed
    assert "1. serve quince-2b" in printed and "go?" in printed
    assert printed.count("allow it?") == 0, "the plan named the call, so it ran unasked"
    assert seen == [("bench_run", {"argv": ["run", "quince-2b.gguf", "--sample", "10"]})]
    assert "a sample of 10" in model.told(), "the person's answer reached the model"
    assert [c[0] for c in got.calls] == ["ask_user", "plan", "bench_run", "done"]


def test_task_mode_ends_on_done_and_prints_the_cost_line():
    got, _, _, printed = drive([call("plan", steps=[PLAN_RUN]), RUN,
                                call("done", summary="all done")], "y\n")
    assert got.done and got.summary == "all done"
    line = [ln for ln in printed.splitlines() if ln.startswith("cost:")]
    assert len(line) == 1 and "3 rounds" in line[0] and "3 calls" in line[0] and "0 asked" in line[0]
    assert printed.rstrip().endswith(line[0]), "the cost line is the last thing printed"


def test_a_number_picks_a_choice_and_free_text_is_taken_as_said():
    script = [call("ask_user", question="Which?", choices=["larch", "quince"]),
              call("ask_user", question="Where to write?"),
              call("done", summary="ok")]
    _, model, _, _ = drive(script, "1\n~/out\n")
    answers = [json.loads(m["content"]) for turn in model.seen for m in turn
               if m.get("role") == "tool" and m.get("name") == "ask_user"]
    assert answers[0]["answer"] == "larch" and answers[-1]["answer"] == "~/out"


def test_a_call_the_plan_does_not_name_asks_again_and_a_no_stops_it():
    script = [call("plan", steps=[PLAN_RUN]),
              call("bench_run", argv=["run", "evil.gguf", "--sample", "10"]),
              call("done", summary="stopped")]
    got, model, seen, printed = drive(script, "y\nn\n")
    assert seen == [] and got.done
    assert "not in the plan you approved" in printed and "declined" in model.told()


def test_a_plan_covers_each_call_once():
    script = [call("plan", steps=[PLAN_RUN]), RUN, RUN, call("done", summary="x")]
    got, _, seen, printed = drive(script, "y\nn\n")
    assert len(seen) == 1 and printed.count("allow it?") == 1


def test_a_plan_the_person_refuses_is_told_to_the_model_and_nothing_runs():
    script = [call("plan", steps=["bench_run the hundred"]), call("done", summary="stopped")]
    got, model, seen, _ = drive(script, "no, the sample please\n")
    assert seen == [] and "no, the sample please" in model.told() and got.done


def test_rounds_ends_a_loop_that_never_says_done_and_prints_the_transcript():
    got, _, seen, printed = drive([call("bench_status") for _ in range(20)], "", rounds=3)
    assert not got.done and got.rounds == 3 and len(seen) == 3
    assert "transcript" in printed.lower() and printed.count("bench_status") >= 3
    assert "run benchmarks with quince-2b" in printed and "cost:" in printed


def test_a_person_who_leaves_ends_the_loop():
    got, _, seen, printed = drive([call("ask_user", question="Sample or full?"),
                                   call("bench_run", argv=["run", "x"])], "")
    assert not got.done and seen == [] and "input ended" in printed.lower()


def test_prose_with_no_call_is_nudged_once_then_taken_as_the_end():
    got, model, _, printed = drive(["I would run the sample.", "Still just talking."], "")
    assert not got.done and len(model.seen) == 2
    assert [m for turn in model.seen for m in turn
            if m.get("role") == "user" and "done" in m.get("content", "")]
    assert "Still just talking." in printed


# -- the acceptance case ---------------------------------------------------------------

PROMPT = ("benchmark qwen3.8-flash-next with llama.cpp (both with draft head and no draft "
          "head) and with ollama, make some animations")


def model_files(tmp_path: Path) -> list[Path]:
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
            return json.loads(unfenced(m["content"]))
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
    call("plan", steps=[
        'bench_run ["sweep", "--serve", "Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf", '
        '"--serve-draft", "auto", "--plain-only", "--sample", "10"] -> Qwen3.8-Flash--plain',
        'bench_run ["sweep", "--serve", "Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf", '
        '"--serve-draft", "", "--label-suffix", "-nodraft", "--plain-only", "--sample", "10"] '
        '-> Qwen3.8-Flash--nodraft-plain',
        'bench_run ["run", "Qwen3.8-Flash--ollama-plain", "--base-url", '
        '"http://127.0.0.1:11434", "--sample", "10"]',
        "jobs_wait bench",
        'bench_compare ["Qwen3.8-Flash--plain", "Qwen3.8-Flash--nodraft-plain", '
        '"Qwen3.8-Flash--ollama-plain", "--export", "compare.json"]',
        'bench_animate ["compare.json", "--out", "compare.mp4"]']),
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
        tmp_path, monkeypatch):
    seen: list = []
    files = model_files(tmp_path)
    monkeypatch.setattr(do.hub, "weight_paths", lambda: files)
    tools = tools_over(seen, files=files, fetch=ollama_fake)
    got, model, seen, printed = drive(list(ACCEPTANCE), "1\n1\n1\ny\ny\ny\ny\ny\ny\n", task=PROMPT,
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
    assert printed.count("allow it?") == 5, "the lookups read outside text, so the plan's calls ask"


# -- the command -----------------------------------------------------------------------

def run_main(argv, stdin: str, monkeypatch, model=None, seen=None):
    out = io.StringIO()
    if model is not None:
        monkeypatch.setattr(do, "client_for", lambda args: model)
        tools = tools_over([] if seen is None else seen)
        monkeypatch.setattr(do, "command_tools", lambda *a, **k: tools)
    args = chat.COMMAND.parser().parse_args(argv)
    return chat.serve(args, io.StringIO(stdin), out), out.getvalue()


def test_dry_run_prints_the_task_prompt_and_every_tool_of_the_role_with_a_worked_example():
    code, printed = run_main(["--dry-run", "run benchmarks with quince-2b"], "", None)
    assert code == 0
    assert chat.TASK.splitlines()[0] in printed and "Your role is runner" in printed
    blocks: dict[str, str] = {}
    current = ""
    for line in printed.splitlines():
        if line and not line.startswith((" ", "\t")) and "(" in line and line.endswith(")"):
            current = line.split("(", 1)[0].strip()
            blocks[current] = ""
        elif line.startswith((" ", "\t")) and blocks:
            blocks[current] += line
    for name in ["serve_up", "bench_run", "ask_user", "plan", "done", "jobs_wait",
                 "models_on_disk", "ollama_models", "bench_animate"]:
        assert name in blocks, name
        assert "Example" in blocks[name] and "->" in blocks[name], name
    assert "one question" in chat.TASK and "two backends" in chat.TASK and "confirm" in chat.TASK


def test_a_task_in_words_runs_over_the_client_it_is_given(monkeypatch):
    seen: list = []
    model = Scripted([call("ask_user", question="Sample or full?", choices=["sample", "full"]),
                      call("plan", steps=[PLAN_RUN]), RUN, call("done", summary="Measured.")],
                     answer="")
    code, printed = run_main(["run benchmarks with quince-2b", "--url", "http://127.0.0.1:1",
                              "--no-compact"],
                             "1\ny\n", monkeypatch, model, seen)
    assert code == 0 and "Measured." in printed and "cost:" in printed
    assert seen == [("bench_run", {"argv": ["run", "quince-2b.gguf", "--sample", "10"]})]


def test_the_task_can_be_given_as_an_option(monkeypatch):
    model = Scripted([call("done", summary="x")], answer="")
    code, printed = run_main(["--task", "look", "--url", "http://127.0.0.1:1", "--no-compact"], "", monkeypatch,
                             model)
    assert code == 0 and "cost:" in printed


def test_a_task_that_runs_out_of_rounds_exits_nonzero(monkeypatch):
    model = Scripted([call("bench_status") for _ in range(9)], answer="")
    code, printed = run_main(["look", "--url", "http://127.0.0.1:1", "--no-compact", "--rounds", "2"], "",
                             monkeypatch, model)
    assert code == 1 and "transcript" in printed.lower()


def test_model_and_url_are_one_or_the_other():
    code, printed = run_main(["look", "--model", "q.gguf", "--url", "http://127.0.0.1:1"], "",
                             None)
    assert code == 2 and "give one" in printed


def test_a_model_already_up_on_the_port_is_used_as_it_stands(monkeypatch, tmp_path, capsys):
    here = tmp_path / "quince-2b.gguf"
    here.write_bytes(b"gguf")
    monkeypatch.setattr("ml_stack.hub.located", lambda *a, **k: here)
    monkeypatch.setattr("ml_stack.serve.leases.already_up",
                        lambda model, port, **_: {"base_url": "http://127.0.0.1:8080", "slots": 2,
                                                  "model": str(here), "pid": 1})
    built = {}

    class FakeClient:
        def __init__(self, url, **kw):
            built["url"] = url

    monkeypatch.setattr("ml_stack.client.Client", FakeClient)
    args = chat.COMMAND.parser().parse_args(["look", "--model", "quince-2b.gguf"])
    client = do.client_for(args)
    assert isinstance(client, FakeClient) and built["url"] == "http://127.0.0.1:8080"
    assert "using the server already up on 8080: quince-2b.gguf, 2 slot(s)" in capsys.readouterr().out
