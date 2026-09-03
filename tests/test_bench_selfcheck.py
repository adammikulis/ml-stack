"""The runner proves itself before it spends the GPU.

Everything here is invented: the models are named tiny.gguf and never exist, the community
is the one that ships with the package, the world is `organisation.make`'s. Nothing reads
~/.ml-stack, a server or a GPU -- the self-check's own scratch is a temporary directory,
and `bench.HOME` is pointed at `tmp_path` so `prepared()` cannot find a real store.
"""

from __future__ import annotations

import json

import pytest

from ml_stack.graph import bench
from ml_stack.graph.bench.selfcheck import (
    ScriptedModel,
    ScriptedReader,
    SelfCheckFailed,
    _scratch_world,
    selfcheck,
)

# -- the fakes are strict -------------------------------------------------------------------------

def test_the_scripted_model_takes_exactly_what_the_client_takes():
    """A fake with **kwargs is what let `tight` reach an 87G load. This one is bound
    against the real `Client.__init__`, so it refuses what the client refuses, by name."""
    ScriptedModel("http://127.0.0.1:1", timeout=3.0, n_predict=16, temperature=0.0, top_k=4)
    with pytest.raises(TypeError, match="tight"):
        ScriptedModel("http://127.0.0.1:1", tight=True)
    with pytest.raises(TypeError, match="rich"):
        ScriptedReader("http://127.0.0.1:1", rich=True)


def test_the_scripted_model_looks_up_once_then_answers():
    model = ScriptedModel()
    tools = [{"function": {"name": "look_up"}}]
    first = model.chat([{"role": "user", "content": "who?"}], tools=tools)
    assert first.tool_calls and json.loads(first.tool_calls[0]["function"]["arguments"]) == {
        "texts": ["compilers"]}
    second = model.chat([{"role": "user", "content": "who?"},
                         {"role": "tool", "content": "found: nobody"}], tools=tools)
    assert second.tool_calls is None and second.content
    assert model.told() == "found: nobody"


# -- the self-check passes for every measuring command ----------------------------------------------

@pytest.mark.slow

def test_a_four_way_sweep_passes_and_every_way_of_both_halves_is_kept():
    said = selfcheck(["sweep", "--serve", "tiny.gguf", "--also", "terse", "--also", "card",
                      "--also", "rich", "--also", "loose"])
    assert said.startswith("sweep: ") and "read back" in said
    labels = said.split(" -- ")[1].split(", ")
    assert sorted(labels) == sorted(f"tiny-{half}{way}" for half in ("plain", "shortlist")
                                    for way in ("", "-terse", "-card", "-rich", "-loose"))


def test_drafts_passes_with_a_head_per_n_max_and_the_baseline():
    said = selfcheck(["drafts", "tiny.gguf", "--draft", "", "--draft", "mtp-tiny.gguf",
                      "--n-max", "2", "--n-max", "8"])
    assert "draft:none" in said and "draft:mtp-tiny@n2" in said and "draft:mtp-tiny@n8" in said
    # the baseline alone, as the contract names it
    assert "draft:none" in selfcheck(["drafts", "tiny.gguf", "--n-max", "2", "--n-max", "8"])


def test_concurrent_passes():
    said = selfcheck(["concurrent", "four-by-three", "--conversations", "4", "--turns", "3"])
    assert said.startswith("concurrent: ") and "four-by-three" in said


def test_extract_passes_over_a_tiny_invented_world():
    said = selfcheck(["extract", "reading", "--world", "/nowhere/at/all", "--serve", "tiny.gguf"])
    assert said.startswith("extract: ") and "message(s)" in said and "reading" in said
    assert "message(s)" in selfcheck(["extract", "reading", "--world", "/nowhere/at/all"])


@pytest.mark.slow


def test_the_flags_about_the_serving_and_the_asking_ride_along():
    """The store, the shortlist, the KV cache, the reasoning budget, the per-question cap:
    each is on the run the self-check keeps, so a flag that breaks one of them breaks
    here and not on the GPU."""
    said = selfcheck(["sweep", "--serve", "tiny.gguf", "--store", "named.ladybug",
                      "--shortlist", "3", "--serve-kv", "q8_0", "--reasoning-budget", "512",
                      "--per-question", "7", "--n-predict", "64", "--embed-url",
                      "http://127.0.0.1:1"])
    assert "tiny-plain-kv-q8_0-rb512" in said and "tiny-shortlist-kv-q8_0-rb512" in said
    assert selfcheck(["run", "plain", "--shortlist", "3", "--store", "named.ladybug"]) \
        .startswith("run: ")


def test_a_command_line_that_does_not_parse_exits_as_it_would(capsys):
    with pytest.raises(SystemExit) as left:
        selfcheck(["sweep", "--serve", "tiny.gguf", "--also", "nonsense"])
    assert left.value.code == 2
    capsys.readouterr()


@pytest.mark.slow


def test_the_self_check_is_fast_enough_to_run_every_time():
    import time

    began = time.monotonic()
    selfcheck(["sweep", "--serve", "tiny.gguf", "--also", "terse", "--also", "card",
               "--also", "rich", "--also", "loose"])
    assert time.monotonic() - began < 10.0


# -- and fails when the asking is broken -------------------------------------------------------------

def _broken_ways(monkeypatch):
    """`_ways` with one more: a way carrying a keyword the client does not take."""
    real = bench._ways

    def with_a_bad_one(args):
        return [*real(args), {"label": "bad", "nonsense": True}]

    monkeypatch.setattr(bench, "_ways", with_a_bad_one)


def test_a_way_the_client_does_not_take_fails_the_self_check_naming_the_keyword(monkeypatch):
    _broken_ways(monkeypatch)
    with pytest.raises(SelfCheckFailed, match="nonsense") as why:
        selfcheck(["sweep", "--serve", "tiny.gguf", "--also", "terse"])
    assert "TypeError" in str(why.value) and "the run printed" in str(why.value)


def test_main_refuses_with_exit_4_before_the_lock_and_before_any_fetch(monkeypatch, capsys):
    import ml_stack.lock

    _broken_ways(monkeypatch)
    locked, fetched = [], []
    monkeypatch.setattr(ml_stack.lock, "only_one", lambda *a, **k: locked.append(a))
    monkeypatch.setattr(bench, "prefetch", lambda refs, **k: fetched.append(refs))
    assert bench.main(["sweep", "--serve", "hf:someone/tiny-GGUF/tiny.gguf",
                       "--also", "terse"]) == 4
    said = capsys.readouterr()
    assert "selfcheck: FAILED" in said.err and "nonsense" in said.err
    assert "--no-selfcheck" in said.err
    assert locked == [] and fetched == [], "nothing was locked or downloaded"
    assert "selfcheck: ok" not in said.out


def test_no_selfcheck_skips_it_and_a_passing_one_is_said_before_the_lock(monkeypatch, capsys,
                                                                          tmp_path):
    """The lock is faked to record its call and refuse, so `main` stops right after the
    check either way: with the flag no check ran, without it the ok line came first."""
    import ml_stack.lock
    from ml_stack.graph.bench import selfcheck as bench_selfcheck

    checked = []
    real = bench_selfcheck.selfcheck

    def counting(argv):
        checked.append(list(argv))
        return real(argv)

    monkeypatch.setattr(bench_selfcheck, "selfcheck", counting)

    def refusing(*a, **k):
        raise ml_stack.lock.Busy("held by the test")

    monkeypatch.setattr(ml_stack.lock, "only_one", refusing)
    argv = ["run", "plain", "--no-prefetch", "--kept", str(tmp_path / "runs.ladybug")]
    assert bench.main([*argv, "--no-selfcheck"]) == 3
    said = capsys.readouterr()
    assert checked == [] and "selfcheck" not in said.out

    assert bench.main(argv) == 3
    said = capsys.readouterr()
    assert checked == [argv]
    assert said.out.splitlines()[0].startswith("selfcheck: ok (")
    assert " s) -- run: " in said.out.splitlines()[0]


# -- the smoke is the first step of a real run ---------------------------------------------------------

def test_an_extract_that_serves_smokes_first_on_the_one_load_and_stops_when_it_fails(
        monkeypatch, tmp_path, capsys):
    pytest.importorskip("ladybug")
    from contextlib import contextmanager

    import ml_stack.client
    import ml_stack.serve
    from ml_stack.graph.bench import extract as bx
    from ml_stack.serve import ServerInfo

    loads, readers = [], []

    class Watched(ScriptedReader):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            readers.append(self)

    @contextmanager
    def fake_serve(model, **kw):
        loads.append(model)
        yield ServerInfo(base_url="http://127.0.0.1:1", port=1, pid=None, backend="fake")

    monkeypatch.setattr(ml_stack.serve, "serve", fake_serve)
    monkeypatch.setattr(ml_stack.client, "Client", Watched)
    monkeypatch.setattr(bx, "footprint", lambda url: {"base_url": url, "model": "tiny.gguf"})
    monkeypatch.setattr(bx, "find_model", lambda named: named)
    world = _scratch_world(tmp_path / "world")
    kept = tmp_path / "runs.ladybug"
    assert bench._main(["extract", "reading", "--world", str(world), "--serve", "tiny.gguf",
                        "--sample", "5", "--kept", str(kept)]) == 0
    said = capsys.readouterr().out
    assert loads == ["tiny.gguf"], "one load for the smoke and the run"
    back = bench.runs(kept)
    assert [len(r["rows"]) for r in back] == [bx.SMOKE_MESSAGES, 5], "the smoke first, kept"
    assert said.index("smoke: 3 message(s)") < said.index("smoke: ok") < said.rindex("kept as")
    assert len(readers) == 1, "one client, asked the smoke and then the sample"

    # a smoke where every message fails ends the run before the sample is read
    class Failing(ScriptedReader):
        def chat(self, messages, **kw):
            raise RuntimeError("the served model answers nothing")

    monkeypatch.setattr(ml_stack.client, "Client", Failing)
    with pytest.raises(bench.SmokeFailed, match="every question failed"):
        bench._main(["extract", "again", "--world", str(world), "--serve", "tiny.gguf",
                     "--sample", "5", "--kept", str(kept)])
    assert [len(r["rows"]) for r in bench.runs(kept, "again")] == [bx.SMOKE_MESSAGES], \
        "the smoke was kept; the sample never ran"

    # --no-smoke: the sample alone
    monkeypatch.setattr(ml_stack.client, "Client", Watched)
    assert bench._main(["extract", "alone", "--world", str(world), "--serve", "tiny.gguf",
                        "--sample", "5", "--kept", str(kept), "--no-smoke"]) == 0
    assert [len(r["rows"]) for r in bench.runs(kept, "alone")] == [5]


# -- the preflight is the real one ----------------------------------------------------------------

HEAD = "hf:someone/tiny-GGUF/MTP/mtp-tiny-Q8_0.gguf"


def test_the_real_preflight_runs_over_the_spec_the_run_builds_and_the_argv_is_built(monkeypatch):
    """Twice on 2026-09-02 the self-check said ok and the run died inside the preflight,
    because the self-check had replaced `Preflight` whole. Now every check runs over the
    spec `served` builds -- a head named by hf: file, a draft length, a quantised cache --
    and `command()` builds the argv for real, twice: in the flags check on the reference's
    stand-in, and in the served fake on the head as `start()` would have fetched it."""
    from ml_stack.serve import preflight
    from ml_stack.serve.backend import LlamaServerBackend

    checked, built = [], []
    real_flags = preflight._flags_check
    real_command = LlamaServerBackend.command

    def watching_flags(spec, binary, **seams):
        checked.append(spec)
        return real_flags(spec, binary, **seams)

    def watching_command(self, spec):
        argv = real_command(self, spec)
        built.append(" ".join(argv))
        return argv

    monkeypatch.setattr(preflight, "_flags_check", watching_flags)
    monkeypatch.setattr(LlamaServerBackend, "command", watching_command)
    said = selfcheck(["drafts", "tiny.gguf", "--draft", HEAD, "--n-max", "2",
                      "--serve-kv", "q8_0"])
    assert "draft:mtp-tiny-Q8_0@n2" in said
    spec = checked[-1]
    assert spec.draft == HEAD and spec.spec_draft_max == 2 and spec.spec_type == "draft-mtp"
    assert spec.cache_type_k == spec.cache_type_v == "q8_0"
    argvs = [a for a in built if "--spec-draft-n-max 2" in a]
    assert any("-md draft-head.gguf" in a for a in argvs), "the flags check's stand-in"
    assert any("mtp-tiny-Q8_0.gguf" in a and "--cache-type-k q8_0" in a
               and "--spec-type draft-mtp" in a and "-md " in a for a in argvs), \
        "the lease's argv, on the head as fetched"
    assert not any(a.split()[0] == "llama-server" for a in argvs), "a binary that is a file"


def test_yesterdays_bug_put_back_fails_the_self_check_naming_the_reference(monkeypatch):
    """A flags check that builds the argv on the raw spec, where `command()` refuses a head
    still named by hf: file. The self-check fails and says which reference, rather than
    saying ok over a preflight that never ran."""
    from ml_stack.serve import preflight
    from ml_stack.serve.backend import LlamaServerBackend
    from ml_stack.serve.preflight import Check

    def raw(spec, binary, **_):
        LlamaServerBackend(binary=binary).command(spec)
        return Check("flags", True, "built on the raw spec")

    monkeypatch.setattr(preflight, "_flags_check", raw)
    with pytest.raises(SelfCheckFailed, match="MTP/mtp-tiny-Q8_0.gguf") as why:
        selfcheck(["drafts", "tiny.gguf", "--draft", HEAD, "--n-max", "2"])
    assert "the preflight raised" in str(why.value) and "ServerFailed" in str(why.value)
    assert "must be fetched" in str(why.value)


def test_a_check_that_refuses_fails_the_self_check_with_the_report(monkeypatch):
    from ml_stack.serve import preflight
    from ml_stack.serve.preflight import Check

    monkeypatch.setattr(preflight, "_fit_check",
                        lambda *a, **k: Check("fit", False, "selfcheck test: does not fit"))
    with pytest.raises(SelfCheckFailed, match="FAIL  fit: selfcheck test: does not fit"):
        selfcheck(["sweep", "--serve", "tiny.gguf"])


@pytest.mark.slow


def test_the_shapes_that_died_in_the_preflight_pass_and_read_nothing(monkeypatch):
    """The two command lines of 2026-09-02, and a model whose file is nowhere: the facts
    say the shards are present, the header is a dense model the build reads, every flag
    is accepted -- and nothing asks the disk, the Hub or a build's --help for any of it."""
    import ml_stack.hub
    import ml_stack.setup
    from ml_stack.serve import backend, preflight

    def never(*a, **k):
        raise AssertionError("the self-check reached a real reader")

    monkeypatch.setattr(ml_stack.hub, "files", never)
    monkeypatch.setattr(preflight, "_local_index", never)
    monkeypatch.setattr(preflight, "read_gguf_header", never)
    monkeypatch.setattr(ml_stack.setup, "_arches", never)
    monkeypatch.setattr(backend, "flags_of", never)

    said = selfcheck(["drafts", "tiny.gguf", "--draft", "hf:someone/tiny-GGUF/MTP/head.gguf",
                      "--n-max", "2", "--n-max", "8"])
    assert "draft:head@n2" in said and "draft:head@n8" in said
    said = selfcheck(["sweep", "--serve", "tiny.gguf", "--serve-draft", "auto",
                      "--serve-kv", "q8_0", "--reasoning-budget", "4096"])
    assert "tiny-plain-kv-q8_0-rb4096" in said and "tiny-shortlist-kv-q8_0-rb4096" in said
    assert selfcheck(["sweep", "--serve", "/nowhere/at/all/absent.gguf"]).startswith("sweep: ")
    assert selfcheck(["sweep", "--serve", "hf:someone/absent-GGUF/absent-Q4_K_M.gguf"]) \
        .startswith("sweep: ")


def test_a_raw_serve_arg_passes_the_self_check_for_the_real_preflight_to_judge(tmp_path, monkeypatch):
    """`--serve-arg=-ub --serve-arg=2048`: the stand-in build cannot know a raw flag, so the
    self-check takes the person's word for it and the real preflight checks it against the
    real binary. Every knob sweep failed here before (2026-09-02)."""
    from ml_stack.graph.bench.selfcheck import selfcheck

    said = selfcheck(["sweep", "--serve", "tiny.gguf", "--plain-only", "--smoke",
                      "--serve-arg=-ub", "--serve-arg=2048", "--serve-mlock",
                      "--label-suffix=-ub2048"])
    assert "ub2048" in said or said
