"""The bench over more than one serving program: a run records what served it, an
unmeasured figure is None and never 0, and memory is the whole process tree.

Every fixture here is invented. Nothing reads a real store, a real graph, or a real server.
"""

from __future__ import annotations

import json
import sys
import types

import pytest

from ml_stack.graph.bench import Row, runs, save, table

from conftest import a_row

G = 2**30


# -- what --on takes, and what a client is built from -----------------------------------

def test_an_ollama_url_names_the_program_and_the_model():
    from ml_stack.graph.bench.backends import parse_on

    name, url, how = parse_on("flash-ollama=ollama://127.0.0.1:11434/thornfell:125b-mlx")
    assert name == "flash-ollama"
    assert url == "ollama://127.0.0.1:11434/thornfell:125b-mlx"
    assert how == {"api": "ollama", "model": "thornfell:125b-mlx"}


def test_an_openai_url_names_the_program_and_the_model():
    from ml_stack.graph.bench.backends import parse_on

    name, url, how = parse_on("hosted=openai://10.0.0.7:8000/pellard-9b")
    assert name == "hosted"
    assert how == {"api": "openai", "model": "pellard-9b"}


def test_a_plain_http_url_is_llama_cpp_and_names_nothing():
    from ml_stack.graph.bench.backends import parse_on

    name, url, how = parse_on("e4b=http://127.0.0.1:8083")
    assert (name, url, how) == ("e4b", "http://127.0.0.1:8083", {})


def test_a_spec_without_name_or_url_is_refused():
    from ml_stack.graph.bench.backends import parse_on

    with pytest.raises(ValueError):
        parse_on("just-a-name")
    with pytest.raises(ValueError):
        parse_on("=http://127.0.0.1:8083")


def test_the_client_is_built_with_the_program_and_context_when_it_takes_them():
    """The contract with the client: ``Client(url, api=, model=, context=)``. A client that
    takes those keywords is given them; one that does not is given only what it takes,
    and an Ollama URL handed to it is refused by name rather than sent as http."""
    from ml_stack.graph.bench.backends import client_for

    seen = {}

    class Newer:
        def __init__(self, base_url, *, api="llama", model=None, context=None, timeout=180.0,
                     temperature=None):
            seen.update(base_url=base_url, api=api, model=model, context=context,
                        timeout=timeout, temperature=temperature)

    client_for("ollama://127.0.0.1:11434/thornfell:125b-mlx", client=Newer, context=32768,
               timeout=300.0, temperature=0.0)
    assert seen == {"base_url": "http://127.0.0.1:11434",
                    "api": "ollama", "model": "thornfell:125b-mlx", "context": 32768,
                    "timeout": 300.0, "temperature": 0.0}, \
        "the plain address, with the program and the model said outright"
    client_for("openai://10.0.0.7:8000/pellard-9b", client=Newer)
    assert seen["base_url"] == "http://10.0.0.7:8000" and seen["api"] == "openai"
    assert seen["model"] == "pellard-9b"

    class Older:
        def __init__(self, base_url, *, timeout=180.0, temperature=None):
            seen.clear()
            seen.update(base_url=base_url, timeout=timeout)

    client_for("http://127.0.0.1:8083", client=Older, context=32768, timeout=300.0)
    assert seen == {"base_url": "http://127.0.0.1:8083", "timeout": 300.0}
    with pytest.raises(ValueError, match="ollama"):
        client_for("ollama://127.0.0.1:11434/thornfell:125b-mlx", client=Older)


# -- what served the run ----------------------------------------------------------------

def test_served_by_is_read_off_a_client_that_can_say():
    from ml_stack.graph.bench.backends import served_by

    class Ollama:
        base_url = "http://127.0.0.1:11434"
        api = "ollama"

        def served_by(self):
            return {"program": "ollama", "version": "0.33.3", "format": "safetensors",
                    "runtime": "mlx", "quant": "nvfp4", "model": "thornfell:125b-mlx",
                    "weights_bytes": 70 * G}

    got = served_by(Ollama())
    assert got["program"] == "ollama" and got["runtime"] == "mlx" and got["quant"] == "nvfp4"
    assert got["weights_bytes"] == 70 * G


def test_served_by_for_llama_cpp_comes_from_props_and_the_gguf_name(monkeypatch, tmp_path):
    """A client with no ``served_by`` is a llama-server: the program, its build, the
    format and the quantisation are read off ``/props`` and the file it names."""
    from ml_stack.graph.bench import backends

    weights = tmp_path / "Thornfell-Next-UD-Q4_K_XL-00001-of-00002.gguf"
    weights.write_bytes(b"x" * 1000)
    (tmp_path / "Thornfell-Next-UD-Q4_K_XL-00002-of-00002.gguf").write_bytes(b"x" * 500)

    def props(url, **kw):
        assert url.endswith("/props")
        return {"model_path": str(weights), "total_slots": 2, "build_info": "b7912-1a2b3c4",
                "default_generation_settings": {"n_ctx": 32768}}

    monkeypatch.setattr(backends, "request_json", props)

    class Plain:
        base_url = "http://127.0.0.1:8080"

    got = backends.served_by(Plain())
    assert got["program"] == "llama.cpp"
    assert got["version"] == "b7912-1a2b3c4"
    assert got["format"] == "gguf"
    assert got["quant"] == "Q4_K_XL"
    assert got["model"] == "Thornfell-Next-UD-Q4_K_XL-00001-of-00002.gguf"
    assert got["weights_bytes"] == 1500, "every shard, not the first"
    assert got["runtime"] is None, "not read is not a runtime"


def test_a_server_that_will_not_say_what_serves_it_is_none(monkeypatch):
    from ml_stack.graph.bench import backends

    def down(url, **kw):
        raise ConnectionError("nothing on that port")

    monkeypatch.setattr(backends, "request_json", down)

    class Plain:
        base_url = "http://127.0.0.1:8080"

    assert backends.served_by(Plain()) is None


@pytest.mark.parametrize("record, build, expect", [
    ({"program": "ollama", "version": "0.33.3", "format": "safetensors", "runtime": "mlx",
      "quant": "nvfp4"}, "", "ollama 0.33.3 · mlx · nvfp4"),
    ({"program": "llama.cpp", "version": "b7912", "format": "gguf", "quant": "Q4_K_XL"},
     "unsloth", "llama.cpp (unsloth) · gguf · Q4_K_XL"),
    ({"program": "llama.cpp", "version": "b7912", "format": "gguf", "quant": "Q4_K_XL"},
     "", "llama.cpp b7912 · gguf · Q4_K_XL"),
    ({"program": "llama.cpp", "format": "gguf", "quant": None}, "", "llama.cpp · gguf"),
    ({}, "", ""),
    (None, "", ""),
])
def test_the_configuration_line_names_program_runtime_and_quant(record, build, expect):
    from ml_stack.graph.bench.backends import describe

    assert describe(record, build=build) == expect


def test_the_short_form_is_one_word_for_a_table_column():
    from ml_stack.graph.bench.backends import short

    assert short({"program": "ollama", "runtime": "mlx", "format": "safetensors",
                  "quant": "nvfp4"}) == "ollama·mlx·nvfp4"
    assert short({"program": "llama.cpp", "format": "gguf", "quant": "Q4_K_XL"}) \
        == "llama.cpp·gguf·Q4_K_XL"
    assert short(None) == "-"
    assert " " not in short({"program": "llama server", "format": "gguf", "quant": "Q4"})


def test_build_of_and_the_rankings_build_read_served_by_first():
    """An Ollama run has no llama-server binary, and read from the binary alone it was
    the default build. The record says what served it, and that is what is read."""
    from ml_stack.graph.bench.report import build_of
    from ml_stack.graph.bench.score import _build

    ollama = {"served_by": {"program": "ollama", "version": "0.33.3", "format": "safetensors",
                            "runtime": "mlx", "quant": "nvfp4"}}
    assert build_of(ollama) == "ollama 0.33.3 · mlx · nvfp4"
    assert _build(ollama) == "ollama 0.33.3 · mlx · nvfp4"
    llama = {"binary": "/opt/builds/unsloth/bin/llama-server",
             "served_by": {"program": "llama.cpp", "build": "unsloth", "format": "gguf",
                           "quant": "Q4_K_XL"}}
    assert build_of(llama) == "unsloth", "a llama.cpp profile still names the build"
    assert _build(llama) == "llama.cpp (unsloth) · gguf · Q4_K_XL"
    assert _build({"binary": "/opt/builds/thornfell/llama-server"}) == "thornfell/llama-server", \
        "a run kept before served_by reads as it did"
    assert _build("/opt/builds/thornfell/llama-server") == "thornfell/llama-server"


def test_the_table_tells_two_runs_of_one_model_apart_by_what_served_them(tmp_path, capsys):
    store = tmp_path / "runs.ladybug"
    row = a_row("who?", expected=["person:iris"], shown=["person:iris"], label="flash-plain")
    save(store, [row], held={"context": 32768, "slots": 1, "model": "thornfell",
                             "served_by": {"program": "llama.cpp", "format": "gguf",
                                           "quant": "Q4_K_XL"}})
    row = a_row("who?", expected=["person:iris"], shown=["person:iris"], label="flash-ollama")
    save(store, [row], held={"context": 32768, "slots": 1, "model": "thornfell",
                             "served_by": {"program": "ollama", "runtime": "mlx",
                                           "format": "safetensors", "quant": "nvfp4"}})
    table(runs(store))
    said = capsys.readouterr().out
    assert "served" in said.splitlines()[0]
    assert "llama.cpp·gguf·Q4_K_XL" in said
    assert "ollama·mlx·nvfp4" in said


# -- unmeasured is None, never 0 --------------------------------------------------------

class _Reply:
    def __init__(self, raw, content="an answer"):
        self.raw, self.content, self.tool_calls = raw, content, None
        self.thinking = None
        self.finish_reason = "stop"


def test_llama_cpp_timings_are_read_whole_and_a_missing_draft_is_zero():
    from ml_stack.graph.bench.backends import timings_of

    got = timings_of(_Reply({"timings": {"prompt_ms": 120.0, "predicted_ms": 800.0,
                                         "prompt_n": 90, "cache_n": 10, "predicted_n": 40}}))
    assert got["prompt_ms"] == 120.0 and got["predicted_ms"] == 800.0
    assert got["prompt_n"] == 90 and got["cache_n"] == 10 and got["predicted_n"] == 40
    assert got["draft_n"] == 0 and got["draft_n_accepted"] == 0, \
        "llama.cpp reporting no draft is a server with no head"
    assert got["load_ms"] is None


def test_a_reply_with_no_timings_at_all_is_none_everywhere():
    from ml_stack.graph.bench.backends import timings_of

    got = timings_of(_Reply({"usage": {"prompt_tokens": 100, "completion_tokens": 20}}))
    assert all(v is None for v in got.values()), got


def test_a_timings_key_written_null_is_not_measured_and_one_left_out_is_zero():
    """The client turns an Ollama reply into llama.cpp's ``timings`` with ``cache_n`` and
    ``draft_n`` written null: that is a program that cannot say, not a server with no head."""
    from ml_stack.graph.bench.backends import timings_of

    got = timings_of(_Reply({"timings": {"prompt_ms": 120.0, "predicted_ms": 800.0,
                                         "load_ms": 5000.0, "prompt_n": 90, "predicted_n": 40,
                                         "cache_n": None, "draft_n": None,
                                         "draft_n_accepted": None}}))
    assert got["cache_n"] is None and got["draft_n"] is None
    assert got["draft_n_accepted"] is None
    assert got["load_ms"] == 5000.0 and got["prompt_n"] == 90


def test_an_ollama_reply_reports_prefill_and_decode_and_nothing_it_cannot():
    """Ollama says what it read and wrote and how long each took, in nanoseconds; it has no
    prompt cache figure and no draft head, and those are None rather than 0."""
    from ml_stack.graph.bench.backends import timings_of

    got = timings_of(_Reply({"model": "thornfell:125b-mlx", "prompt_eval_count": 90,
                             "prompt_eval_duration": 120_000_000, "eval_count": 40,
                             "eval_duration": 800_000_000, "load_duration": 5_000_000_000,
                             "total_duration": 1_000_000_000}))
    assert got["prompt_n"] == 90 and got["predicted_n"] == 40
    assert got["prompt_ms"] == 120.0 and got["predicted_ms"] == 800.0
    assert got["load_ms"] == 5000.0
    assert got["cache_n"] is None and got["draft_n"] is None \
        and got["draft_n_accepted"] is None


def test_counting_carries_none_through_for_what_a_backend_did_not_report():
    import ml_stack.graph.bench as bench

    class Model:
        def chat(self, messages, **kw):
            return _Reply({"usage": {"prompt_tokens": 100, "completion_tokens": 20}})

    counting = bench.Counting(Model())
    counting.chat([{"role": "user", "content": "who?"}])
    assert counting.cached_tokens is None and counting.draft_tokens is None
    assert counting.draft_taken is None and counting.generating_ms is None
    assert counting.first_token is None
    assert counting.per_call == [(None, None)]
    assert counting.prompt_tokens == 100 and counting.completion_tokens == 20, \
        "what the reply did say is still counted"


def test_counting_adds_up_what_is_reported_and_leaves_none_for_the_rest():
    """Two calls, one with a prompt-cache figure and one without: the total is over the
    calls that said, and a field no call reported stays None."""
    import ml_stack.graph.bench as bench

    replies = iter([
        _Reply({"timings": {"prompt_ms": 100.0, "predicted_ms": 500.0, "prompt_n": 80,
                            "cache_n": 20, "predicted_n": 30}}),
        _Reply({"prompt_eval_count": 50, "prompt_eval_duration": 60_000_000,
                "eval_count": 10, "eval_duration": 200_000_000}),
    ])

    class Model:
        def chat(self, messages, **kw):
            return next(replies)

    counting = bench.Counting(Model())
    counting.chat([{"role": "user", "content": "who?"}])
    counting.chat([{"role": "user", "content": "who else?"}])
    assert counting.cached_tokens == 20
    assert counting.processed_tokens == 130
    assert counting.generating_ms == 860.0
    assert counting.draft_tokens == 0, "a llama.cpp call with no head drafted nothing"
    assert counting.per_call == [(20, 80), (None, 50)]


def test_a_row_from_a_backend_that_reports_nothing_says_so(tmp_path):
    import ml_stack.graph.bench as bench

    class Model:
        def chat(self, messages, **kw):
            return _Reply({"usage": {"prompt_tokens": 100, "completion_tokens": 20}})

    def ask(question, client):
        client.chat([{"role": "user", "content": question}])
        return {"content": "an answer", "show": [], "why": ""}

    rows = bench.measure(ask, [{"q": "who?", "expect": ["n1"]}], label="tried", client=Model())
    row = rows[0]
    assert row.cached_tokens is None and row.draft_tokens is None and row.draft_taken is None
    assert row.queued is None and row.first_token is None
    assert row.prefix_kept is None and row.prefix_turns is None and row.prefix_hits is None
    assert row.cache_calls == [[None, None]]
    assert row.calls == 1 and row.prompt_tokens == 100

    store = tmp_path / "runs.ladybug"
    save(store, rows, held={"context": 32768, "slots": 1})
    back = runs(store)[0]["rows"][0]
    assert back["cached_tokens"] is None and back["queued"] is None, "kept as None"
    assert "prefix_hits" not in runs(store)[0]["server"], "no turn judged is no figure"


def test_prefix_kept_skips_a_call_that_reported_nothing():
    from ml_stack.graph.bench import prefix_kept

    assert prefix_kept([(None, None), (None, None)]) == (0, 0)
    assert prefix_kept([(20, 80), (None, 50), (100, 5), (110, 3)]) == (1, 1), \
        "only the transition between two calls that both reported is judged"


def test_the_table_prints_a_dash_for_each_figure_nobody_measured(tmp_path, capsys):
    store = tmp_path / "runs.ladybug"
    row = a_row("who?", expected=["person:iris"], shown=["person:iris"])
    row.cached_tokens = row.draft_tokens = row.draft_taken = None
    row.queued = row.first_token = None
    row.cache_calls = [[None, None]]
    row.prefix_kept = row.prefix_turns = row.prefix_hits = None
    save(store, [row], held={"context": 32768, "slots": 1})
    table(runs(store))
    said = capsys.readouterr().out
    head, line = said.splitlines()[0], next(ln for ln in said.splitlines()
                                            if ln.startswith("tried"))
    heads, cells = head.split(), line.split()
    for column in ("cached", "peak", "draft"):
        assert cells[heads.index(column)] == "-", f"{column}: {line}"


def test_drafting_tells_no_head_from_not_reported():
    from ml_stack.graph.bench import drafting

    assert drafting([{"draft_tokens": None, "draft_taken": None}]) == "-", "not reported"
    assert drafting([{"draft_tokens": 0, "draft_taken": 0}]) == "none", "a server with no head"
    assert drafting([{"draft_tokens": 10, "draft_taken": 5}]) == "50%"
    assert drafting([]) == "-"


def test_compare_says_not_measured_rather_than_a_percentage(tmp_path):
    from ml_stack.graph.bench import compare

    store = tmp_path / "runs.ladybug"
    a = a_row("who?", expected=["person:iris"], shown=["person:iris"], label="llama")
    a.cached_tokens, a.processed_tokens, a.completion_tokens = 40, 60, 20
    save(store, [a], held={})
    b = a_row("who?", expected=["person:iris"], shown=["person:iris"], label="ollama")
    b.cached_tokens, b.processed_tokens, b.completion_tokens = None, 70, 25
    save(store, [b], held={})
    said = compare(store, "llama", "ollama")
    cached = next(ln for ln in said.splitlines() if "cached" in ln)
    assert "not measured on ollama" in cached and "%" not in cached
    read = next(ln for ln in said.splitlines() if "of those, read" in ln)
    assert "+17%" in read, "a line both sides measured still compares"


# -- memory for real ----------------------------------------------------------------------

class _Proc:
    """One process of a fake tree: its resident set now, and its children."""

    def __init__(self, pid, rss, children=(), cmdline=()):
        self.pid, self.rss, self._children = pid, rss, list(children)
        self.info = {"pid": pid, "cmdline": list(cmdline)}

    def memory_info(self):
        return types.SimpleNamespace(rss=self.rss)

    def children(self, recursive=False):
        return list(self._children)


class _Tree:
    """psutil as far as the sampler reads it: ``Process(pid)`` and the machine."""

    def __init__(self, procs, *, appear_after=0):
        self._by_pid = {p.pid: p for p in procs}
        self._tick = 0
        self._appear_after = appear_after
        self.NoSuchProcess = type("NoSuchProcess", (Exception,), {})

    def Process(self, pid):
        if pid not in self._by_pid:
            raise self.NoSuchProcess(pid)
        return self._by_pid[pid]

    def process_iter(self, _fields=()):
        return list(self._by_pid.values())

    def virtual_memory(self):
        return types.SimpleNamespace(wired=80 * G, free=20 * G, inactive=4 * G,
                                     total=128 * G, available=24 * G)


def test_the_sampler_sums_the_tree_and_sees_a_runner_that_arrives_late(monkeypatch):
    """Ollama's listener holds nothing; the runner it spawns after the first request holds
    the weights. The pids are re-read every tick, so the child is counted from the tick
    it appears, and the peak is the sum over the tree."""
    import ml_stack.graph.bench as measuring

    runner = _Proc(5001, 60 * G)
    listener = _Proc(5000, 1 * G, children=[])
    tree = _Tree([listener, runner])
    monkeypatch.setitem(sys.modules, "psutil", tree)
    monkeypatch.setattr(measuring, "_rusage_footprint", lambda pid: 0)

    class Ollama:
        base_url = "http://127.0.0.1:11434"
        api = "ollama"

        def processes(self):
            return [5000]

    watcher = measuring.Watching("http://127.0.0.1:11434", every=0.01, start=False,
                                 client=Ollama())
    watcher._once()                       # first tick: the listener alone
    listener._children.append(runner)     # the runner spawns on the first request
    watcher._once()
    runner.rss = 71 * G
    watcher._once()
    runner.rss = 70 * G
    watcher._once()
    got = watcher.peaks
    assert got["resident_peak"] == 72 * G, "listener plus runner at its most"
    assert got["resident_peak_at"] == 3, "the sample it was seen on"
    assert got["processes"] == 2
    assert got["samples"] == 4


def test_the_sampler_falls_back_to_the_llama_server_on_the_port(monkeypatch):
    import ml_stack.graph.bench as measuring

    server = _Proc(7000, 50 * G, cmdline=["llama-server", "--port", "8099"])
    monkeypatch.setitem(sys.modules, "psutil", _Tree([server]))
    monkeypatch.setattr(measuring, "_rusage_footprint", lambda pid: 0)

    class Plain:
        base_url = "http://127.0.0.1:8099"

    assert measuring.serving_pids("http://127.0.0.1:8099", client=Plain()) == [7000]
    watcher = measuring.Watching("http://127.0.0.1:8099", every=0.01, start=False,
                                 client=Plain())
    watcher._once()
    server.rss = 52 * G
    watcher._once()
    assert watcher.peaks["resident_peak"] == 52 * G


def test_the_footprint_takes_the_weights_and_the_program_from_what_served_it(monkeypatch):
    import ml_stack.graph.bench as measuring
    from ml_stack.graph.bench import backends

    runner = _Proc(5001, 64 * G)
    listener = _Proc(5000, 1 * G, children=[runner])
    monkeypatch.setitem(sys.modules, "psutil", _Tree([listener, runner]))

    def no_props(url, **kw):
        raise ConnectionError("no /props on ollama")

    monkeypatch.setattr(backends, "request_json", no_props)
    monkeypatch.setattr(measuring, "request_json", no_props, raising=False)

    class Ollama:
        base_url = "http://127.0.0.1:11434"
        api = "ollama"

        def processes(self):
            return [5000]

        def served_by(self):
            return {"program": "ollama", "version": "0.33.3", "format": "safetensors",
                    "runtime": "mlx", "quant": "nvfp4", "model": "thornfell:125b-mlx",
                    "weights_bytes": 60 * G}

    got = measuring.footprint("http://127.0.0.1:11434", client=Ollama())
    assert got["served_by"]["program"] == "ollama"
    assert got["weights_bytes"] == 60 * G
    assert got["resident_bytes"] == 65 * G, "the tree, not one process"
    assert got["model"] == "thornfell:125b-mlx"
    assert got["kv_and_run_bytes"] == 5 * G


def test_a_run_of_speed_or_none_kind_has_no_questions_in_the_answering_table(tmp_path, capsys):
    store = tmp_path / "runs.ladybug"
    save(store, [{"prompt_tokens": 512, "streams": 1, "prefill_tps": 900.0,
                  "decode_tps": 30.0}], held={"context": 32768}, kind="speed",
         label="flash-speed")
    kept = runs(store)
    assert kept[0]["kind"] == "speed" and kept[0]["label"] == "flash-speed"
    assert kept[0]["rows"][0]["prefill_tps"] == 900.0
    table(kept)
    assert "flash-speed" not in capsys.readouterr().out, "a speed run is not an answering run"


# -- the sweep over more than one program -----------------------------------------------

def test_a_sweep_serves_without_the_head_and_labels_the_runs_so(tmp_path, monkeypatch):
    """``--no-draft`` is the profile's shape minus its head, and ``--serve-label`` names
    the stem, so the three configurations are three labels a reader tells apart."""
    from dataclasses import replace

    import ml_stack.graph.bench as bench
    from ml_stack.serve.profile import record

    from test_graph_bench import _serving

    measured = record("tiny.gguf", seat_context=4096, cache_type="q8_0",
                      draft="/models/mtp-tiny.gguf", spec_type="draft-mtp", spec_draft_max=4)
    monkeypatch.setattr("ml_stack.serve.profile.profile_for",
                        lambda m: replace(measured, served=str(m)))
    seen = _serving(monkeypatch, tmp_path)
    assert bench._main(["sweep", "--serve", "tiny.gguf", "--no-draft", "--serve-label", "flash",
                        "--plain-only", "--smoke", *seen["common"]]) == 0
    kw = seen["kwargs"][0]
    assert not kw.get("draft") and not kw.get("spec_type"), "the head is left out"
    assert kw.get("cache_type_k") == "q8_0", "the rest of the measured shape is kept"
    assert [r["label"] for r in runs(seen["kept"])] == ["flash-nodraft-plain-kv-q8_0"], \
        "the stem, -nodraft, the way, and the cache type the profile measured with"
    assert "draft_model" not in runs(seen["kept"])[0]["server"]
    assert bench._main(["sweep", "--serve", "tiny.gguf", "--serve-label", "flash",
                        "--plain-only", "--smoke", *seen["common"]]) == 0
    assert seen["kwargs"][-1].get("draft") == "/models/mtp-tiny.gguf"
    assert [r["label"] for r in runs(seen["kept"])][-1] == "flash-plain-kv-q8_0"


def test_a_sweep_on_an_ollama_url_builds_the_client_for_it_and_records_what_served(tmp_path,
                                                                                    monkeypatch):
    """``--on flash-ollama=ollama://host:port/model``: the client is built with the api
    and the model the URL names at the sweep's context, the busy check goes to the plain
    http port, and the run's record says what served it."""
    import ml_stack.graph.bench as bench

    built = {}

    class Ollama:
        def __init__(self, base_url, *, api="llama", model=None, context=None, timeout=180.0,
                     temperature=None, n_predict=16384, **rest):
            built.update(base_url=base_url, api=api, model=model, context=context,
                         timeout=timeout, temperature=temperature)
            self.base_url, self.api, self.model, self.context = base_url, api, model, context
            self.sampling = {"temperature": temperature if temperature is not None else 0.0}
            self.card = {}

        def served_by(self):
            return {"program": "ollama", "version": "0.33.3", "format": "safetensors",
                    "runtime": "mlx", "quant": "nvfp4", "model": "thornfell:125b-mlx",
                    "weights_bytes": 60 * G}

        def processes(self):
            return []

        def chat(self, messages, **kw):
            return _Reply({"model": "thornfell:125b-mlx", "prompt_eval_count": 40,
                           "prompt_eval_duration": 80_000_000, "eval_count": 12,
                           "eval_duration": 400_000_000}, content="a compiler person")

    asked = []
    monkeypatch.setattr(bench, "HOME", tmp_path / "home")
    monkeypatch.setattr(bench, "busy", lambda url: asked.append(url) or 0)
    monkeypatch.setattr("ml_stack.client.Client", Ollama)
    kept = tmp_path / "runs.ladybug"
    graph = tmp_path / "g.json"
    from test_graph_bench import TINY

    graph.write_text(json.dumps(TINY))
    questions = tmp_path / "q.jsonl"
    questions.write_text(json.dumps({"q": "who works on compilers?",
                                     "expect": ["topic:compiler"]}) + "\n")
    code = bench._main(["sweep", "--on", "flash-ollama=ollama://127.0.0.1:11434/thornfell:125b-mlx",
                        "--plain-only", "--smoke", "--kept", str(kept), "--graph", str(graph),
                        "--questions", str(questions), "--store", "", "--context", "16384",
                        "--no-smoke"])
    assert code == 0
    assert built["base_url"] == "http://127.0.0.1:11434"
    assert built["api"] == "ollama" and built["model"] == "thornfell:125b-mlx"
    assert built["context"] == 16384
    assert asked == ["http://127.0.0.1:11434"], "the busy check goes to the plain port"
    one = runs(kept)[0]
    assert one["label"] == "flash-ollama-plain"
    assert one["server"]["served_by"]["program"] == "ollama"
    assert one["server"]["served_by"]["quant"] == "nvfp4"
    assert one["server"]["model"] == "thornfell:125b-mlx"
    assert one["server"]["weights_bytes"] == 60 * G
    row = one["rows"][0]
    assert row["cached_tokens"] is None and row["draft_tokens"] is None
    assert row["processed_tokens"] == 40 * row["calls"] and row["queued"] is not None
