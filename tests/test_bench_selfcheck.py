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
from ml_stack.graph.bench_selfcheck import (
    ScriptedModel,
    ScriptedReader,
    SelfCheckFailed,
    _scratch_world,
    selfcheck,
)

pytestmark = pytest.mark.usefixtures("_no_real_home")


@pytest.fixture
def _no_real_home(monkeypatch, tmp_path):
    monkeypatch.setattr(bench, "HOME", tmp_path / "home")


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

def test_a_four_way_sweep_passes_and_every_way_of_both_halves_is_kept():
    said = selfcheck(["sweep", "--serve", "tiny.gguf", "--also", "terse", "--also", "card",
                      "--also", "rich", "--also", "tight"])
    assert said.startswith("sweep: ") and "read back" in said
    labels = said.split(" -- ")[1].split(", ")
    assert sorted(labels) == sorted(f"tiny-{half}{way}" for half in ("plain", "shortlist")
                                    for way in ("", "-terse", "-card", "-rich", "-tight"))


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


def test_the_self_check_is_fast_enough_to_run_every_time():
    import time

    began = time.monotonic()
    selfcheck(["sweep", "--serve", "tiny.gguf", "--also", "terse", "--also", "card",
               "--also", "rich", "--also", "tight"])
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
    from ml_stack.graph import bench_selfcheck

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
    from ml_stack.graph import bench_extract as bx
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
