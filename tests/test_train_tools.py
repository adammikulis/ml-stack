"""ml-stack-train-tools: a project's tool schemas, turned into a model that calls them.

Everything here runs on the CPU over models built in ``tmp_path`` -- a Gemma3 of two layers
and a tokenizer trained on the synthetic questions -- and nothing is downloaded. The chat
template the tiny base carries is written here and is only the *shape* of a tool-calling
template (turn markers, a declarations block, a call block); the recipe must find the
assistant turn in any template, so the test's own is as good as a published one.
"""

from __future__ import annotations

import json
import stat

import pytest
from ml_stack.graph import ask
from ml_stack.train.tools import (
    CHAT,
    Example,
    examples_in,
    main,
    schemas_of,
    split,
    synthesise,
    write_dataset,
)

# -- invented tools, with descriptions in both worked-example shapes ----------------------

TOOLS = [
    {"type": "function", "function": {
        "name": "find_recipe",
        "description": "Search the cookbook for dishes whose ingredients match some words. "
                       "Example: for \"what can I make with lentils and cumin?\" call "
                       "find_recipe with {\"words\": [\"lentils\", \"cumin\"]}.",
        "parameters": {"type": "object", "properties": {
            "words": {"type": "array", "items": {"type": "string"},
                      "description": "ingredients to look for, e.g. [\"lentils\", \"cumin\"]"}},
            "required": ["words"]}}},
    {"type": "function", "function": {
        "name": "list_shelf",
        "description": "Read out everything of one kind in the pantry. Examples: \"Which "
                       "tins do we have?\" → list_shelf(kind=\"tin\"); \"What jars are "
                       "there?\" → list_shelf(kind=\"jar\").",
        "parameters": {"type": "object", "properties": {
            "kind": {"type": "string", "enum": ["tin", "jar", "box"]}},
            "required": ["kind"]}}},
    {"type": "function", "function": {
        "name": "open_page",
        "description": "Read one page of the cookbook site. Example: open_page with "
                       "{\"url\": \"https://marrowfield.example/soups\", \"rendered\": false}.",
        "parameters": {"type": "object", "properties": {
            "url": {"type": "string", "description": "the page's address"},
            "rendered": {"type": "boolean", "description": "open it in a browser"}},
            "required": ["url"]}}},
]

PROMPTS = {
    "find_recipe": ["what goes well with fennel?", "something with chickpeas for dinner",
                    "find me a soup with barley"],
    "list_shelf": ["which tins are on the shelf?", "what boxes are in the pantry?"],
    "open_page": ["open https://marrowfield.example/bread for me"],
    CHAT: ["hi there", "thanks a lot", "what is the capital of Peru?"],
}


class TestExamplesIn:
    def test_the_graph_tools_carry_worked_examples(self):
        """The examples that took a 4B model from 17% to 70% recall are the seed."""
        found = examples_in(ask.TOOLS)
        by_tool = {}
        for e in found:
            by_tool.setdefault(e.tool, []).append(e)

        assert Example("Which companies do people here work for?", "list_kind",
                       {"kind": "org"}) in found
        assert Example("What topics come up here?", "list_kind", {"kind": "topic"}) in found
        assert len([e for e in by_tool["list_kind"] if e.question]) >= 2
        assert [e for e in by_tool["look_up"] if e.question and e.arguments["texts"]]
        # bare calls seed arguments even though they carry no question
        for tool in ("look_at", "path_between", "show"):
            assert any(not e.question and e.arguments for e in by_tool[tool]), tool

    def test_the_web_tools_arrow_examples_are_read(self):
        from ml_stack import web

        found = examples_in(web.SCHEMAS)
        by_tool = {}
        for e in found:
            by_tool.setdefault(e.tool, []).append(e)

        assert Example("What does Quenlow Robotics do?", "web_search",
                       {"query": "Quenlow Robotics"}) in found
        assert len(by_tool["web_search"]) >= 3
        assert len(by_tool["web_look"]) >= 3
        assert Example("What does this page say? https://pellard.example/news", "web_read",
                       {"url": "https://pellard.example/news"}) in found

    def test_an_arrow_call_keeps_its_booleans_as_booleans(self):
        found = examples_in([{"name": "open_page", "description":
                              "\"Open it properly\" → open_page(url=\"https://a.example\", "
                              "rendered=true)", "parameters": {}}])
        assert found == [Example("Open it properly", "open_page",
                                 {"url": "https://a.example", "rendered": True})]

    def test_router_prompts_become_examples_with_arguments_left_open(self):
        found = examples_in(TOOLS, PROMPTS)
        assert Example("what goes well with fennel?", "find_recipe", None) in found
        assert Example("hi there", CHAT, None) in found
        assert CHAT == ask.CHAT, "a router's prompts are handed over as they are"

    def test_a_prompt_for_a_tool_that_does_not_exist_is_refused(self):
        with pytest.raises(ValueError, match="list_shelf"):
            examples_in(TOOLS, {"list_shelves": ["what is on the shelf?"]})

    def test_the_pairs_tools_for_emits_are_accepted(self):
        pairs = ask.tools_for({"nodes": [], "edges": []})
        assert [s["function"]["name"] for s in schemas_of(pairs)] == \
            [s["function"]["name"] for s in ask.TOOLS]
        assert examples_in(pairs) == examples_in(ask.TOOLS)


class TestSynthesise:
    def test_it_is_reproducible_for_a_seed(self):
        assert synthesise(TOOLS, prompts=PROMPTS, seed=3) == \
            synthesise(TOOLS, prompts=PROMPTS, seed=3)

    def test_every_tool_is_covered_and_some_turns_call_nothing(self):
        rows = synthesise(TOOLS, prompts=PROMPTS, per_tool=12)
        tools = {r["tool"] for r in rows}
        assert tools == {"find_recipe", "list_shelf", "open_page", CHAT}
        for name in ("find_recipe", "list_shelf", "open_page"):
            assert sum(1 for r in rows if r["tool"] == name) == 12, name
        chat = [r for r in rows if r["tool"] == CHAT]
        assert 0.1 <= len(chat) / len(rows) <= 0.3
        for r in chat:
            assert r["messages"][-1]["content"] and "tool_calls" not in r["messages"][-1]

    def test_a_row_is_a_conversation_in_the_openai_shape(self):
        row = next(r for r in synthesise(TOOLS, prompts=PROMPTS) if r["tool"] == "list_shelf")
        roles = [m["role"] for m in row["messages"]]
        assert roles == ["system", "user", "assistant"]
        call = row["messages"][-1]["tool_calls"][0]["function"]
        assert call["name"] == "list_shelf"
        assert call["arguments"]["kind"] in ("tin", "jar", "box")
        assert row["tools"] == schemas_of(TOOLS)

    def test_one_seed_in_ten_is_held_out_with_every_paraphrase_of_it(self):
        prompts = {"find_recipe": [f"what can I cook with ingredient number {i}?"
                                   for i in range(60)]}
        rows = synthesise(TOOLS, prompts=prompts, per_tool=400)
        train, holdout = split(rows)

        assert holdout and train
        assert not {r["from"] for r in train} & {r["from"] for r in holdout}
        seeds = {r["from"] for r in rows}
        held = {r["from"] for r in holdout}
        assert 0.03 <= len(held) / len(seeds) <= 0.2

    def test_no_question_appears_twice(self):
        rows = synthesise(TOOLS, prompts=PROMPTS, per_tool=30)
        questions = [r["messages"][1]["content"].lower() for r in rows]
        assert len(questions) == len(set(questions))

    def test_arguments_come_from_the_question_where_they_can(self):
        rows = synthesise(TOOLS, prompts=PROMPTS, per_tool=4)
        by_question = {r["messages"][1]["content"]: r for r in rows}

        def call(question):
            return by_question[question]["messages"][-1]["tool_calls"][0]["function"]["arguments"]

        assert call("which tins are on the shelf?") == {"kind": "tin"}
        assert call("what boxes are in the pantry?") == {"kind": "box"}
        assert call("open https://marrowfield.example/bread for me")["url"] == \
            "https://marrowfield.example/bread"
        assert "chickpeas" in call("something with chickpeas for dinner")["words"]

    def test_a_served_model_adds_questions_and_its_bad_lines_are_dropped(self):
        asked = []

        def fake(prompt):
            asked.append(prompt)
            if "list_shelf" in prompt:
                return ('{"question": "any boxes left?", "arguments": {"kind": "box"}}\n'
                        'not json at all\n'
                        '{"question": "what tins?", "arguments": {"colour": "red"}}\n')
            if "no tool call" in prompt:
                return '{"question": "good evening to you"}\n'
            return ""

        rows = synthesise(TOOLS, prompts=PROMPTS, ask=fake, per_tool=6)
        questions = {r["messages"][1]["content"]: r["tool"] for r in rows}
        assert questions["any boxes left?"] == "list_shelf"
        assert questions["good evening to you"] == CHAT
        assert "what tins?" not in questions
        assert any("what can I make with lentils" in p for p in asked), \
            "the worked examples are the few-shots"

    def test_no_tools_is_refused(self):
        with pytest.raises(ValueError, match="no tools"):
            synthesise([])


# -- a tiny base model, built here -----------------------------------------------------------

TEMPLATE = (
    "{{ bos_token }}"
    "{% if tools or messages[0]['role'] == 'system' %}<start_of_turn>developer\n"
    "{% if messages[0]['role'] == 'system' %}{{ messages[0]['content'] | trim }}"
    "{% set messages = messages[1:] %}{% endif %}"
    "{% for tool in tools %}<start_function_declaration>{{ tool['function']['name'] }}"
    "<end_function_declaration>{% endfor %}<end_of_turn>\n{% endif %}"
    "{% for m in messages %}"
    "<start_of_turn>{{ 'model' if m['role'] == 'assistant' else m['role'] }}\n"
    "{% if m['content'] %}{{ m['content'] | trim }}{% endif %}"
    "{% for c in m.get('tool_calls', []) %}<start_function_call>call:"
    "{{ c['function']['name'] }}{{ c['function']['arguments'] | tojson }}"
    "<end_function_call>{% endfor %}<end_of_turn>\n{% endfor %}"
    "{% if add_generation_prompt %}<start_of_turn>model\n{% endif %}")

SPECIAL = ["<pad>", "<bos>", "<eos>", "<unk>", "<start_of_turn>", "<end_of_turn>",
           "<start_function_call>", "<end_function_call>", "<start_function_declaration>",
           "<end_function_declaration>"]


def make_tiny_base(path, texts):
    """A two-layer Gemma3 with tied embeddings and a BPE tokenizer over ``texts``."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import Gemma3ForCausalLM, Gemma3TextConfig, PreTrainedTokenizerFast

    raw = Tokenizer(models.BPE(unk_token="<unk>"))
    raw.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    raw.decoder = decoders.ByteLevel()
    raw.train_from_iterator(texts, trainers.BpeTrainer(
        vocab_size=400, special_tokens=SPECIAL,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw, bos_token="<bos>", eos_token="<eos>", pad_token="<pad>",
        unk_token="<unk>", additional_special_tokens=SPECIAL[4:], chat_template=TEMPLATE)
    tokenizer.save_pretrained(path)

    torch.manual_seed(0)
    config = Gemma3TextConfig(
        vocab_size=len(tokenizer), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=16,
        max_position_embeddings=512, sliding_window=64, pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id)
    Gemma3ForCausalLM(config).save_pretrained(path)
    return path


@pytest.fixture
def dataset(tmp_path):
    """A synthesised conversation set and the tiny base it was made for, on the CPU."""
    rows = synthesise(TOOLS, prompts=PROMPTS, per_tool=20)
    texts = [m["content"] for r in rows for m in r["messages"] if m.get("content")]
    texts += [json.dumps(m["tool_calls"][0]["function"]["arguments"])
              for r in rows for m in r["messages"] if m.get("tool_calls")]
    texts += ["call:find_recipe", "call:list_shelf", "call:open_page"]
    base = make_tiny_base(tmp_path / "base", texts)
    data = tmp_path / "data"
    write_dataset(data, rows, base=str(base))
    return data, base


@pytest.fixture(autouse=True)
def on_the_cpu(monkeypatch):
    monkeypatch.setenv("ML_STACK_DEVICE", "cpu")


class TestRecipe:
    def test_the_recipe_is_known_and_validates(self):
        from ml_stack.train.recipes import known, validate

        assert "tool-calls" in known()
        got = validate("tool-calls", {"steps": 40})
        assert got["steps"] == 40 and got["context"] > 0 and got["learning_rate"] < 1e-3

    def test_the_loss_is_on_the_assistant_turn_only(self, dataset):
        from transformers import AutoTokenizer

        from ml_stack.train.recipes.tool_calls import IGNORE, render

        data, base = dataset
        tokenizer = AutoTokenizer.from_pretrained(base)
        rows = [json.loads(line) for line in (data / "train.jsonl").read_text().splitlines()]
        row = next(r for r in rows if r["tool"] == "list_shelf")
        ids, labels = render(tokenizer, row["messages"], row["tools"], context=512)

        assert len(ids) == len(labels)
        read = tokenizer.decode([i for i, lab in zip(ids, labels) if lab == IGNORE])
        answer = tokenizer.decode([lab for lab in labels if lab != IGNORE])
        assert row["messages"][1]["content"] in read
        assert "<start_function_declaration>" in read
        assert answer.startswith("<start_function_call>call:list_shelf")
        assert answer.rstrip().endswith("<end_of_turn>")
        assert "<start_of_turn>model" not in answer

    def test_a_row_the_context_cuts_off_entirely_is_dropped_not_taught(self, dataset):
        from transformers import AutoTokenizer

        from ml_stack.train.recipes.tool_calls import render

        data, base = dataset
        tokenizer = AutoTokenizer.from_pretrained(base)
        row = json.loads((data / "train.jsonl").read_text().splitlines()[0])
        assert render(tokenizer, row["messages"], row["tools"], context=8) is None

    def test_the_tiny_base_trains_and_checkpoints_through_the_existing_trainer(self, dataset,
                                                                               tmp_path):
        """Tied embeddings are the case: safetensors refuses a state dict that names one
        storage twice, and a Gemma checkpoint does. The trainer names each storage once
        on the way out and re-ties on the way back, so this model needs no wrapper."""
        from ml_stack.train import read
        from ml_stack.train.run import run

        data, base = dataset
        out = tmp_path / "run"
        got = run("tool-calls", {"steps": 20, "context": 256, "batch_size": 4,
                                 "learning_rate": 0.003}, data, out)

        assert got["steps"] == 20 and got["checkpoint"]
        history = [r for r in read(out / "metrics.jsonl") if r.get("event") == "step"]
        assert got["final_loss"] < history[0]["loss"], "twenty steps taught it nothing"
        start = next(r for r in read(out / "metrics.jsonl") if r.get("event") == "start")
        assert start["config"]["base"] == str(base)
        assert start["config"]["answer_tokens"] > 0
        assert start["config"]["holdout_rows"] >= 0

        again = run("tool-calls", {"steps": 20, "context": 256, "batch_size": 4,
                                   "learning_rate": 0.003}, data, out)
        assert again["steps"] == 20, "a finished run is not trained twice"

    def test_the_checkpoint_goes_back_into_hugging_face_layout_for_export(self, dataset,
                                                                          tmp_path):
        from safetensors.torch import load_file
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from ml_stack.train.recipes.tool_calls import save_pretrained
        from ml_stack.train.run import run

        data, base = dataset
        run("tool-calls", {"steps": 20, "context": 256, "batch_size": 2,
                           "learning_rate": 0.001}, data, tmp_path / "run")
        saved = save_pretrained(tmp_path / "run", str(base), tmp_path / "model")

        tensors = load_file(str(saved / "model.safetensors"))
        assert "model.embed_tokens.weight" in tensors and "lm_head.weight" not in tensors
        reloaded = AutoModelForCausalLM.from_pretrained(saved)
        assert AutoTokenizer.from_pretrained(saved).chat_template
        assert reloaded.config.num_hidden_layers == 2

    def test_an_empty_data_directory_says_what_it_wanted(self, tmp_path):
        from ml_stack.train.recipes import build

        empty = tmp_path / "nothing"
        empty.mkdir()
        with pytest.raises(ValueError, match="messages"):
            build("tool-calls", {}, empty)


class TestCommandLine:
    @pytest.fixture
    def tools_file(self, tmp_path):
        path = tmp_path / "tools.json"
        path.write_text(json.dumps(TOOLS))
        prompts = tmp_path / "prompts.json"
        prompts.write_text(json.dumps(PROMPTS))
        return path, prompts

    def test_a_dry_run_plans_every_stage_and_writes_nothing(self, tools_file, tmp_path, capsys):
        tools, prompts = tools_file
        out = tmp_path / "out"
        code = main(["--tools", str(tools), "--prompts", str(prompts), "--out", str(out),
                     "--base", "invented/never-loaded", "--per-tool", "5", "--dry-run"])

        printed = capsys.readouterr().out
        assert code == 0, printed
        assert "synth: " in printed and "find_recipe 5" in printed
        assert "train: would fine-tune invented/never-loaded" in printed
        assert "export: would" in printed
        assert not out.exists()
        summary = json.loads(printed[printed.index("{"):])
        assert summary["dry_run"] and summary["synth"]["rows"] > 0

    def test_the_synth_stage_writes_the_dataset_and_is_skipped_next_time(self, tools_file,
                                                                         tmp_path, capsys):
        tools, prompts = tools_file
        out = tmp_path / "out"
        assert main(["--tools", str(tools), "--prompts", str(prompts), "--out", str(out),
                     "--only", "synth"]) == 0
        manifest = json.loads((out / "data" / "manifest.json").read_text())
        assert manifest["base"] == "google/functiongemma-270m-it"
        assert manifest["train"] + manifest["holdout"] == manifest["rows"]
        assert (out / "data" / "train.jsonl").read_text().count("\n") == manifest["train"]

        capsys.readouterr()
        assert main(["--tools", str(tools), "--out", str(out), "--only", "synth"]) == 0
        assert "skipping" in capsys.readouterr().out

    def test_the_graph_tools_are_imported_live(self, tmp_path, capsys):
        code = main(["--tools", "python:ml_stack.graph.ask:TOOLS",
                     "--prompts", "python:ml_stack.graph.ask:TOOL_PROMPTS",
                     "--out", str(tmp_path / "out"), "--per-tool", "8", "--dry-run"])
        printed = capsys.readouterr().out
        assert code == 0, printed
        for tool in ("look_up", "look_at", "path_between", "list_kind", "show", "chat"):
            assert f"{tool} " in printed, tool

    def test_a_missing_tools_file_is_a_message_not_a_traceback(self, tmp_path, capsys):
        code = main(["--tools", str(tmp_path / "absent.json"), "--out", str(tmp_path / "o"),
                     "--dry-run"])
        assert code == 2
        assert "absent.json" in capsys.readouterr().err

    def test_training_without_data_says_to_synthesise_first(self, tmp_path, capsys):
        code = main(["--tools", "python:ml_stack.graph.ask:TOOLS", "--out", str(tmp_path / "o"),
                     "--only", "train"])
        assert code == 2
        assert "synth stage first" in capsys.readouterr().err

    def test_train_and_export_run_from_the_command(self, dataset, tmp_path, monkeypatch, capsys):
        """The converter and quantiser are faked -- each copies its input -- because what is
        tested is the path through them: checkpoint -> Hugging Face layout -> GGUF, and
        that a second run finds the GGUF and stops."""
        data, base = dataset
        out = tmp_path / "out"
        out.mkdir()
        (out / "data").symlink_to(data)

        llama = tmp_path / "llama.cpp"
        (llama / "build" / "bin").mkdir(parents=True)
        (llama / "convert_hf_to_gguf.py").write_text(
            "import shutil, sys\n"
            "argv = sys.argv[1:]\n"
            "shutil.copy(argv[0] + '/model.safetensors', argv[argv.index('--outfile') + 1])\n")
        quantize = llama / "build" / "bin" / "llama-quantize"
        quantize.write_text("#!/bin/sh\ncp \"$1\" \"$2\"\n")
        quantize.chmod(quantize.stat().st_mode | stat.S_IEXEC)
        monkeypatch.setenv("LLAMA_CPP_ROOT", str(llama))

        code = main(["--tools", "unused.json", "--out", str(out), "--base", str(base),
                     "--only", "train", "--set", "steps=20", "--set", "batch_size=2",
                     "--set", "context=256"])
        printed = capsys.readouterr().out
        assert code == 0, printed
        assert "train: final loss" in printed

        code = main(["--tools", "unused.json", "--out", str(out), "--base", str(base),
                     "--only", "export"])
        printed = capsys.readouterr().out
        assert code == 0, printed
        gguf = sorted(out.glob("*.gguf"))
        assert gguf and gguf[0].name == "base-tools-Q8_0.gguf"
        assert (out / "model" / "config.json").exists()

        code = main(["--tools", "unused.json", "--out", str(out), "--base", str(base),
                     "--only", "export"])
        assert code == 0 and "already exists, skipping" in capsys.readouterr().out

    def test_export_without_a_converter_names_it(self, dataset, tmp_path, monkeypatch, capsys):
        from ml_stack.gguf import tools as gguf_tools

        data, base = dataset
        out = tmp_path / "out"
        out.mkdir()
        (out / "data").symlink_to(data)
        for key in ("LLAMA_CPP_ROOT", "LLAMA_CPP_DIR"):
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setattr(gguf_tools, "SOURCE_DIRS", ())
        monkeypatch.setenv("PATH", str(tmp_path))
        main(["--tools", "unused.json", "--out", str(out), "--base", str(base),
              "--only", "train", "--set", "steps=20", "--set", "batch_size=2"])
        capsys.readouterr()

        code = main(["--tools", "unused.json", "--out", str(out), "--base", str(base),
                     "--only", "export"])
        assert code == 2
        assert "convert_hf_to_gguf.py" in capsys.readouterr().err
        assert not list(out.glob("*.gguf"))


# -- training data out of what a model actually did ----------------------------------------

def _traced_row(question="who works on compilers?", *, shown=("topic:compiler",),
                expected=("topic:compiler",), calls=3):
    """One kept bench row with a transcript on it, the shape `Counting` writes.

    Invented throughout: an invented community's ids, an invented model file. Nothing here
    reads a real store -- that is operation, not a test.
    """
    schemas = [{"type": "function", "function": {
        "name": "look_up", "description": "find entries by words",
        "parameters": {"type": "object", "properties": {
            "texts": {"type": "array", "items": {"type": "string"}}}, "required": ["texts"]}}},
        {"type": "function", "function": {
            "name": "show", "description": "light entries on the page",
            "parameters": {"type": "object", "properties": {
                "ids": {"type": "array", "items": {"type": "string"}}}, "required": ["ids"]}}}]
    timings = {"prompt_ms": 120.0, "predicted_ms": 300.0, "prompt_n": 900, "cache_n": 0,
               "predicted_n": 20, "draft_n": 0, "draft_n_accepted": 0}

    def said(call, tool_calls, content="", finish="tool_calls"):
        return {"role": "assistant", "call": call, "model": "invented-e4b.gguf",
                "content": content, "chars": len(content), "thinking_chars": 4,
                "finish": finish, "seconds": 0.4, "tool_calls": tool_calls,
                "offered": ["look_up", "show"], "tokens": {"prompt": 900, "completion": 20},
                "timings": dict(timings)}

    return {"question": question, "expected": list(expected), "shown": list(shown),
            "calls": calls, "trace": [
                {"role": "tools", "tools": schemas},
                {"role": "system", "content": "You answer with the tools you are given."},
                {"role": "user", "content": question},
                said(1, [{"name": "look_up", "args": {"texts": ["compiler"]}}]),
                {"role": "tool", "name": "look_up", "chars": 90, "ids": 2,
                 "content": '[{"id": "topic:compiler"}, {"id": "person:ada"}]'},
                said(2, [{"name": "show", "args": {"ids": ["topic:compiler"]}}]),
                {"role": "tool", "name": "show", "chars": 24, "ids": 0,
                 "content": '"selected 1 on the graph"'},
                said(3, [], "Ada Quill works on the compiler.", "stop")]}


def _kept_run(rows, *, label="e4b-shortlist", model="invented-e4b.gguf"):
    return [{"key": f"bench:{label}:20260902T190000", "label": label, "at": "2026-09-02T19:00",
             "server": {"model": model, "context": 32768}, "rows": list(rows)}]


class TestFromBench:
    """`ml-stack-train-tools from-bench`: a bench run's traces into training examples.

    The synthesiser teaches the *shape* of a call from the descriptions; this teaches the
    calls that actually scored, on a real graph, with real ids -- which is the only source
    of an argument that a description never showed.
    """

    def test_one_example_per_model_turn_with_the_conversation_up_to_it(self):
        from ml_stack.train.tools import from_bench

        rows = from_bench(_kept_run([_traced_row()]), min_f1=0.8)
        assert [r["tool"] for r in rows] == ["look_up", "show", CHAT]
        assert [len(r["messages"]) for r in rows] == [3, 5, 7], \
            "each turn sees strictly more than the one before it"

        first, second, last = rows
        assert first["messages"][0]["role"] == "system"
        assert first["messages"][-1]["tool_calls"][0]["function"] == {
            "name": "look_up", "arguments": {"texts": ["compiler"]}}
        assert second["messages"][-2] == {"role": "tool", "name": "look_up",
                                          "content": '[{"id": "topic:compiler"}, '
                                                     '{"id": "person:ada"}]'}
        assert second["messages"][-1]["tool_calls"][0]["function"]["arguments"] == {
            "ids": ["topic:compiler"]}, "the id it showed, which no description could teach"
        assert last["messages"][-1] == {"role": "assistant",
                                        "content": "Ada Quill works on the compiler."}
        assert [t["function"]["name"] for t in first["tools"]] == ["look_up", "show"], \
            "offered what it was offered, and no tool it never had"
        assert all(r["from"] == "who works on compilers?" for r in rows)
        assert all(r["model"] == "invented-e4b.gguf" for r in rows)

    def test_a_wrong_answer_is_not_a_lesson(self):
        """The whole point of scoring the run first. A question that showed the wrong
        entries made its tool calls in some particular wrong way, and that is exactly what
        must not be learned."""
        from ml_stack.train.tools import from_bench, traced_rows

        kept = _kept_run([_traced_row(),
                          _traced_row("who welds?", shown=["topic:compiler"],
                                      expected=["topic:welding"])])
        assert len(traced_rows(kept, min_f1=0.8)) == 1
        assert len(traced_rows(kept, min_f1=0.0)) == 2
        assert {r["from"] for r in from_bench(kept)} == {"who works on compilers?"}

    def test_only_one_model_at_a_time(self):
        """Two models' turns in one dataset teach the average of two callers. A run is
        named for how it was asked and the server says which file was loaded; either
        identifies it."""
        from ml_stack.train.tools import from_bench

        kept = (_kept_run([_traced_row()], label="e4b-shortlist")
                + _kept_run([_traced_row()], label="flash-plain", model="flash-next.gguf"))
        assert len(from_bench(kept, model="e4b")) == 3
        assert len(from_bench(kept, model="flash-next")) == 3, "by the file, not the label"
        assert len(from_bench(kept)) == 6

    def test_a_turn_the_ceiling_cut_off_is_dropped(self):
        """A truncated call is the one thing a tool caller must never learn: a
        `finish_reason` of length means the arguments stop mid-word."""
        from ml_stack.train.tools import from_bench

        row = _traced_row()
        row["trace"][-1]["finish"] = "length"
        rows = from_bench(_kept_run([row]))
        assert [r["tool"] for r in rows] == ["look_up", "show"]

    def test_an_untraced_store_yields_nothing_and_says_what_it_would_have(self, capsys,
                                                                          tmp_path):
        """What every run kept before today is: hundreds of scored questions and not one
        transcript. The number beside the zero is the point -- those turns are recoverable
        only by spending the GPU again."""
        from ml_stack.train.tools import main, would_yield

        untraced = {k: v for k, v in _traced_row().items() if k != "trace"}
        kept = _kept_run([untraced, _traced_row("who welds?", shown=["topic:welding"],
                                                expected=["topic:welding"], calls=5)])
        assert would_yield(kept) == {"questions": 2, "turns": 8, "traced": 1}

        store = tmp_path / "runs.ladybug"
        pytest.importorskip("ladybug", reason="the store needs ml-stack[store]")
        from ml_stack.graph.store import GraphStore

        with GraphStore(store) as writer:
            writer.put_doc(kept[0]["key"], {k: v for k, v in kept[0].items() if k != "key"})
        out = tmp_path / "caller.jsonl"
        assert main(["from-bench", "--kept", str(store), "--out", str(out)]) == 0
        said = capsys.readouterr().out
        assert "2 question(s) at or above F1 0.8, 1 of them traced" in said
        written = [json.loads(line) for line in out.read_text().splitlines()]
        assert [r["tool"] for r in written] == ["look_up", "show", CHAT]

    def test_a_directory_out_is_a_dataset_the_recipe_reads(self, tmp_path):
        """The same rows the synthesiser writes, so the two sources mix in one directory
        and `ml-stack-train-run --recipe tool-calls --data` reads either."""
        from ml_stack.train.recipes.tool_calls import read_conversations
        from ml_stack.train.tools import from_bench, write_dataset

        rows = from_bench(_kept_run([_traced_row(q) for q in
                                     ("who works on compilers?", "who else?", "and who?")]))
        summary = write_dataset(tmp_path / "data", rows, base="invented/tiny-it",
                                source="a bench store")
        assert summary["rows"] == len(rows) and summary["base"] == "invented/tiny-it"
        train, holdout, manifest = read_conversations(tmp_path / "data")
        assert len(train) + len(holdout) == len(rows)
        assert manifest["base"] == "invented/tiny-it"
