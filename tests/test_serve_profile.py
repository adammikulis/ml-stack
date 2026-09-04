"""One model's measured shape, kept in a file that both ends read.

Everything here is invented: two made-up model files (`thornfield-8B`, `alderpost-2B`)
with made-up heads, written into ``tmp_path``. Both halves of the source of truth are
pointed away from the real ones in an autouse fixture -- ``package_file`` is replaced and
``$MLSTACK_PROFILES_FILE`` moves the local half -- so no test can read the records this
repository ships or write into a real ``~/.ml-stack``. No model is served and no GPU is
touched: what `up --profile` would start is a fake manager that records the spec.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ml_stack.client import Reply
from ml_stack.graph.ask import _under, converse
from ml_stack.serve import cli as serve_cli
from ml_stack.serve import profile as prof
from ml_stack.serve.profile import Profile, add, profile_for, profiles, record, said

MODEL = "thornfield-8B-UD-Q4_K_XL.gguf"
OTHER = "alderpost-2B-Q4_K_M.gguf"
HEAD = "mtp-thornfield-8B-Q8_0.gguf"


@pytest.fixture(autouse=True)
def _profiles_in_tmp(tmp_path, monkeypatch):
    """The shipped file and this machine's own, both inside ``tmp_path`` and both empty."""
    shipped = tmp_path / "ssot" / "profiles.json"
    shipped.parent.mkdir(parents=True, exist_ok=True)
    shipped.write_text("[]\n", encoding="utf-8")
    monkeypatch.setattr(prof, "package_file", lambda: shipped)
    monkeypatch.setenv("MLSTACK_PROFILES_FILE", str(tmp_path / "local" / "profiles.json"))
    monkeypatch.setattr(prof, "writable_file", lambda: shipped)
    return shipped


def written(where: Path, *records: Profile) -> None:
    where.parent.mkdir(parents=True, exist_ok=True)
    where.write_text(json.dumps([one.as_dict() for one in records], indent=2),
                     encoding="utf-8")


def measured(model: str = MODEL, **fields) -> Profile:
    """A record with something in every group, so a round trip has something to lose."""
    return record(model, **{
        "build": "thornfell", "draft": HEAD, "spec_type": "draft-mtp",
        "spec_draft_max": 4, "cache_type": "q8_0", "reasoning_budget": 0, "mmproj": "auto",
        "extra_args": ("-ub", "2048", "--spec-draft-p-min", "0.5"),
        "seat_context": 32768, "parallel": 2,
        "tight": True, "batch": True, "kinds": True, "summary": True,
        "sampling": {"temperature": 0.0},
        "measured_at": "2026-09-02", "label": "thornfield--all-plain-kv-q8_0-rb0",
        "questions": 100, "right": 0.8, "recall": 0.89, "precision": 0.77,
        "seconds_per_question": 26.7, "host": "ladybug", **fields})


# -- the file, and the two halves of it -------------------------------------------------

def test_a_record_survives_being_written_and_read_back(tmp_path):
    add(measured())
    read_back = profiles()

    assert [one.model for one in read_back] == [MODEL]
    assert read_back[0] == measured(), "a record read back is the record that was written"


def test_this_machine_replaces_the_shipped_record_rather_than_sitting_beside_it(
        tmp_path, _profiles_in_tmp):
    written(_profiles_in_tmp, measured())
    written(Path(tmp_path / "local" / "profiles.json"),
            measured(cache_type="f16", label="thornfield--plain", right=0.5))

    every = profiles()
    assert len(every) == 1, "two shapes for one model is a choice nobody can make outside"
    assert every[0].cache_type == "f16" and every[0].right == 0.5


def test_a_file_that_is_not_records_contributes_nothing(_profiles_in_tmp):
    _profiles_in_tmp.write_text("{ not a list", encoding="utf-8")
    assert profiles() == []


def test_a_key_this_version_does_not_know_is_ignored_rather_than_raising():
    read_back = Profile.from_dict({"model": MODEL, "serve": {"cache_type": "q8_0",
                                                             "from_the_future": 7},
                                   "ask": {"tight": False}})
    assert read_back.cache_type == "q8_0" and read_back.tight is False
    assert read_back.spec_draft_max is None, "what the file did not say takes its default"


# -- finding the record for a model -----------------------------------------------------

def test_the_file_name_is_found_through_a_path_or_an_hf_reference():
    add(measured())
    for asked in (MODEL, f"/models/{MODEL}", f"hf:thornfell/thornfield-GGUF/{MODEL}",
                  MODEL.upper()):
        found = profile_for(asked)
        assert found is not None and found.model == MODEL, asked
        assert found.served == asked, "a shape is served by the reference it was asked for"


def test_a_shard_of_the_same_quantisation_finds_the_record_by_family_and_quant():
    add(measured("thornfield-8B-UD-Q4_K_XL-00001-of-00004.gguf"))
    found = profile_for("hf:thornfell/thornfield-GGUF/thornfield-8B-UD-Q4_K_XL.gguf")

    assert found is not None and found.family == "thornfield-8B"
    assert not found.note, "the same model at the same quantisation is the measurement"


def test_another_quantisation_of_the_same_model_says_so_rather_than_passing_as_measured():
    add(measured())
    found = profile_for("thornfield-8B-IQ4_XS.gguf")

    assert found is not None and found.cache_type == "q8_0"
    assert "another quantisation" in found.note
    assert MODEL in found.note, "the note names the file that was actually measured"


def test_a_model_nothing_measured_has_no_profile():
    add(measured())
    assert profile_for("someone-elses-model-Q4_K_M.gguf") is None


# -- what the two ends read -------------------------------------------------------------

def test_the_shape_is_the_whole_serving_and_the_lease_it_becomes():
    shape = measured().shape(port=8099, resolve=False)

    assert (shape.model, shape.port, shape.seats, shape.seat_context) == (MODEL, 8099, 2, 32768)
    assert (shape.build, shape.draft, shape.spec_type) == ("thornfell", HEAD, "draft-mtp")
    assert shape.lease() == {"port": 8099, "context": 65536, "parallel": 2,
                             "cache_type_k": "q8_0", "cache_type_v": "q8_0",
                             "draft": HEAD, "spec_type": "draft-mtp", "spec_draft_max": 4,
                             "mmproj": "auto", "reasoning_budget": 0,
                             "extra_args": ("-ub", "2048", "--spec-draft-p-min", "0.5")}


def test_the_shape_is_served_by_the_reference_asked_for_and_the_seats_asked_for():
    add(measured())
    found = profile_for(f"hf:thornfell/thornfield-GGUF/{MODEL}")
    shape = found.shape(port=8080, seats=8, resolve=False)

    assert shape.model == f"hf:thornfell/thornfield-GGUF/{MODEL}"
    assert shape.seats == 8, "how many conversations a machine wants is that machine's"
    assert shape.seat_context == 32768, "and the rest of the shape is not"


def test_a_head_recorded_by_file_name_is_looked_for_where_this_machine_keeps_models(
        monkeypatch):
    monkeypatch.setattr("ml_stack.hub.located", lambda name: Path(f"/models/{name}"))
    assert measured().shape(resolve=True).draft == f"/models/{HEAD}"

    monkeypatch.setattr("ml_stack.hub.located", lambda name: None)
    assert measured().shape(resolve=True).draft == "", "a head not on this machine is not served"


def test_the_asking_is_what_converse_takes_and_nothing_it_does_not():
    assert measured().asking() == {"tight": True, "batch": True, "kinds": True,
                                   "summary_tool": True}
    plain = record(OTHER, terse=True, sampling={"temperature": 1.0})
    assert plain.asking() == {"tight": True}, \
        "terse chooses the schemas and sampling is the client's; neither is converse's"
    assert plain.terse is True and plain.sampling == {"temperature": 1.0}
    assert record(OTHER, rich=True, reach=8000, tight=False).asking() == {
        "tight": False, "rich": True, "reach": 8000}


# -- ml-stack-serve profile --------------------------------------------------------------

def test_the_command_reads_out_the_serving_the_asking_and_what_measured_it(capsys):
    add(measured())
    assert serve_cli.main(["profile", MODEL]) == 0

    out = capsys.readouterr().out
    assert MODEL in out
    assert "--build thornfell" in out and "--kv q8_0" in out and "--spec-n-max 4" in out
    assert "--reasoning-budget 0" in out
    assert "--context 65536" in out and "--parallel" not in out.split("measured at")[0], \
        "the serve line is one seat holding the whole measured cache"
    assert "measured at --parallel 2, 32768 per seat" in out
    assert "-ub 2048 --spec-draft-p-min 0.5" in out
    assert "tight + batch + kinds + summary + greedy" in out
    assert "80% F1" in out and "26.7 s/question" in out and "100 question(s)" in out
    assert "thornfield--all-plain-kv-q8_0-rb0" in out, "the row that set it is named"


def test_the_command_with_no_model_reads_out_every_record(capsys):
    add(measured())
    add(record(OTHER, cache_type="q8_0"))
    assert serve_cli.main(["profile"]) == 0

    out = capsys.readouterr().out
    assert MODEL in out and OTHER in out


def test_the_command_exits_one_for_a_model_nothing_measured(capsys):
    assert serve_cli.main(["profile", OTHER]) == 1
    assert "nothing measured" in capsys.readouterr().err


def test_the_command_hands_a_script_the_records_as_they_are_kept(capsys):
    add(measured())
    assert serve_cli.main(["profile", MODEL, "--json"]) == 0

    rows = json.loads(capsys.readouterr().out)
    assert rows[0]["serve"]["extra_args"] == ["-ub", "2048", "--spec-draft-p-min", "0.5"]
    assert rows[0]["ask"]["summary"] is True
    assert rows[0]["measured"]["label"] == "thornfield--all-plain-kv-q8_0-rb0"


# -- ml-stack-serve up --profile ---------------------------------------------------------

@pytest.fixture
def leases(monkeypatch, tmp_path):
    """`up` with nothing to start: the manager records the spec it was handed."""
    seen: list = []

    class Manager:
        def __init__(self, backend=None, state_file=None):
            self.backend = backend

        def lease(self, spec, timeout=None):
            seen.append(spec)
            return SimpleNamespace(base_url=f"http://127.0.0.1:{spec.port}", port=spec.port,
                                   pid=None, adopted=True)

    monkeypatch.setattr(serve_cli, "ServerManager", Manager)
    monkeypatch.setattr(serve_cli, "resolve_model", lambda named: named)
    monkeypatch.setattr("ml_stack.hub.located", lambda name: Path(f"/models/{name}"))
    return seen


def upped(*argv: str, root: Path) -> list[str]:
    return ["up", f"/models/{MODEL}", "--root", str(root / "no-fleet"), *argv]


def test_up_with_a_profile_fills_every_flag_that_was_not_given(leases, tmp_path):
    add(measured(mmproj=""))
    assert serve_cli.main(upped("--profile", root=tmp_path)) == 0

    spec = leases[0]
    assert spec.context == 65536 and spec.parallel == 1, \
        "one seat holding the whole cache the record measured across two"
    assert str(spec.draft) == f"/models/{HEAD}" and spec.spec_type == "draft-mtp"
    assert spec.spec_draft_max == 4
    assert spec.cache_type_k == spec.cache_type_v == "q8_0"
    assert spec.reasoning_budget == 0
    assert spec.extra_args == ("-ub", "2048", "--spec-draft-p-min", "0.5")


def test_a_flag_that_was_given_wins_over_the_record(leases, tmp_path):
    add(measured(mmproj=""))
    assert serve_cli.main(upped("--profile", "--kv", "f16", "--parallel", "4",
                                "--spec-n-max", "2", root=tmp_path)) == 0

    spec = leases[0]
    assert spec.cache_type_k == "f16", "a person naming a flag is overruling on purpose"
    assert spec.parallel == 4 and spec.spec_draft_max == 2
    assert spec.context == 32768 * 4, "each seat asked for gets what one measured seat got"
    assert spec.reasoning_budget == 0, "and what nobody named still comes from the record"


def test_a_context_that_was_given_wins_over_the_whole_measured_cache(leases, tmp_path):
    add(measured(mmproj=""))
    assert serve_cli.main(upped("--profile", "--context", "16384", root=tmp_path)) == 0
    assert leases[0].context == 16384 and leases[0].parallel == 1


def test_up_without_the_flag_reads_no_profile_at_all(leases, tmp_path):
    add(measured(mmproj=""))
    assert serve_cli.main(upped(root=tmp_path)) == 0

    spec = leases[0]
    assert spec.cache_type_k == "" and spec.draft is None and spec.extra_args == ()


def test_up_says_so_and_serves_as_asked_when_nothing_measured_this_model(leases, tmp_path,
                                                                        capsys):
    assert serve_cli.main(upped("--profile", root=tmp_path)) == 0
    assert "no measured profile" in capsys.readouterr().err
    assert leases[0].cache_type_k == ""


# -- converse(profile=...) ---------------------------------------------------------------

GRAPH = {"nodes": [{"id": "person:ada", "kind": "person", "label": "Ada Lovelace",
                    "mentions": 2, "attrs": {}, "messages": []}],
         "edges": [], "messages": {}}


class Watcher:
    """A model that answers at once and remembers which tools it was offered."""

    def __init__(self) -> None:
        self.offered: list[list[str]] = []

    def chat(self, messages, *, tools=None, **_):
        self.offered.append([t["function"]["name"] for t in (tools or [])])
        return Reply(content="Ada Lovelace works on compilers.")


def test_a_profile_asks_the_way_that_model_measured_best():
    add(measured())
    watching = Watcher()
    converse("who is here?", GRAPH, watching, profile=MODEL)

    assert "summarise" in watching.offered[0], \
        "the record measured the summary tool, so the model is offered it"


def test_no_profile_leaves_the_asking_exactly_as_it_was():
    add(measured())
    watching = Watcher()
    converse("who is here?", GRAPH, watching)

    assert "summarise" not in watching.offered[0]


def test_a_profile_fills_in_only_what_the_call_left_unsaid():
    given = {"rich": False, "tight": True, "reach": None, "kinds": False, "batch": False,
             "summary_tool": False}
    both = _under(measured(), given)
    assert both["batch"] is True and both["kinds"] is True and both["summary_tool"] is True

    said_outright = _under(measured(), {**given, "tight": False, "rich": True})
    assert said_outright["tight"] is False and said_outright["rich"] is True, \
        "a caller overruling a measurement is not overruled back"
    assert said_outright["batch"] is True, "and the rest still comes from the record"


def test_a_model_nothing_measured_changes_nothing_about_the_asking():
    given = {"rich": False, "tight": True, "reach": None, "kinds": False, "batch": False,
             "summary_tool": False}
    assert _under("nothing-measured-this-Q4_K_M.gguf", given) == given


# -- ml-stack-bench report --profile ------------------------------------------------------

def keep_run(store, label: str, *, right: float = 0.8, n: int = 20,   # SHORT: a record is never set from fewer
             seconds: float = 2.0, **held) -> str:
    """One run through the store the bench actually keeps runs in -- no JSON by hand.

    ``seconds`` is the wall clock of each question, so a row can be made cheap or dear
    without changing what it got right: which of two rows a record is written from is a
    question about both at once.
    """
    from ml_stack.graph import bench
    from ml_stack.graph.bench import Row

    rows = []
    for question in range(n):
        rows.append(Row(label=label, question=f"who is here, question {question}?",
                        expected=["person:ada"],
                        shown=["person:ada"] if question < round(right * n) else [],
                        seconds=seconds, calls=2, processed_tokens=100,
                        completion_tokens=10))
    server = {"model": MODEL, "binary": "/builds/current/llama-server", "slots": 2,
              "context": 65536, "host": "ladybug", **held}
    return bench.save(store, rows, held=server, asking=held.pop("asking", None))


def test_the_record_is_written_from_the_best_row_of_the_store(tmp_path, capsys):
    from ml_stack.graph.bench.report import main as reporting

    store = str(tmp_path / "runs.ladybug")
    keep_run(store, "thornfield--plain-kv-q8_0-rb0", right=0.4, cache_type="q8_0")
    keep_run(store, "thornfield--all-plain-batch-kinds-summary-kv-q8_0-rb0", right=0.8,
             cache_type="q8_0", draft_model=HEAD, spec_draft_max=4, reasoning_budget=0,
             sampling={"temperature": 0.0})

    where = tmp_path / "written.json"
    assert reporting(SimpleNamespace(kept=store, profile=True, profiles=str(where),
                                     since="", last=0, model=[], full_n=0)) == 0
    assert "80% F1" in capsys.readouterr().out

    kept_records = prof.profiles(package=where, local=where)
    assert len(kept_records) == 1
    one = kept_records[0]
    assert one.model == MODEL
    assert (one.draft, one.spec_type, one.spec_draft_max) == (HEAD, "draft-mtp", 4)
    assert one.cache_type == "q8_0" and one.reasoning_budget == 0
    assert one.seat_context == 32768 and one.parallel == 2, "65536 over two slots"
    assert (one.batch, one.kinds, one.summary, one.tight) == (True, True, True, True)
    assert one.sampling == {"temperature": 0.0}
    assert one.questions == 20 and round(one.right, 2) == 0.8
    assert one.label == "thornfield--all-plain-batch-kinds-summary-kv-q8_0-rb0"
    assert one.label in one.note, "the record says which row set it"
    assert one.host == "ladybug"


def test_the_asking_a_run_recorded_is_taken_over_the_words_in_its_label():
    from ml_stack.graph.bench.report import ways_of

    said = {"tight": True, "terse": False, "batch": True, "reach": 8000}
    assert ways_of({"label": "thornfield--plain", "asking": said}) == {
        "tight": True, "batch": True, "kinds": False, "summary": False, "rich": False,
        "terse": False, "single": False, "few": False, "constrain_ids": False,
        "reach": 8000}
    assert ways_of({"label": "thornfield--loose-plain-kinds"}) == {
        "tight": False, "batch": False, "kinds": True, "summary": False, "rich": False,
        "terse": False, "single": False, "few": False, "constrain_ids": False}, \
        "an older run has only its label, read by whole word"
    # the ways of one asking per model: a record has to carry them or a model measured on
    # three tools and twenty turns would be served with eight and ten
    assert ways_of({"label": "x", "asking": {"tight": True, "few": True, "single": True,
                                             "rounds": 20}}) == {
        "tight": True, "batch": False, "kinds": False, "summary": False, "rich": False,
        "terse": False, "single": True, "few": True, "constrain_ids": False, "rounds": 20}
    assert ways_of({"label": "x", "asking": {"tight": True, "constrain_ids": True}})[
        "constrain_ids"] is True


def test_constrain_ids_is_kept_on_the_record_and_read_out(tmp_path):
    made = record(MODEL, constrain_ids=True)
    assert Profile.from_dict(made.as_dict()).constrain_ids is True
    assert made.as_dict()["ask"]["constrain_ids"] is True
    assert made.asking() == {"tight": True, "constrain_ids": True}
    assert made.asked().constrain_ids is True
    line = next(one for one in said(made).splitlines() if "ask with" in one)
    assert "constrain-ids" in line
    assert "constrain-ids" not in said(record(OTHER))

    from ml_stack.graph import bench
    from ml_stack.graph.bench.report import write_profiles

    store = str(tmp_path / "runs.ladybug")
    keep_run(store, "thornfield--plain", asking={"tight": True, "constrain_ids": True})
    where = tmp_path / "written.json"
    write_profiles(bench.runs(store), path=where)
    assert prof.profiles(package=where, local=where)[0].constrain_ids is True, \
        "a run measured under the grammar writes a record that says so"


def test_the_record_takes_the_fastest_row_its_questions_cannot_tell_apart(tmp_path):
    """One asking per model, chosen by measurement. `across` ranks models by F1, which is
    the right question for "which model answers best" and the wrong one for "how should
    this model be asked": two askings whose 95% bands overlap are not two accuracies, and
    between them the record takes the cheaper one, because the seconds are a difference
    the questions can see.

    Three rows over the same twenty questions: a slow one a hair ahead on F1, a fast one
    whose band overlaps it, and a fast one that is genuinely worse -- separated, its band
    clear below. The record has to take the middle one.

    Mutation: rank by F1 alone and the slow row wins; drop the `held_up` guard and the
    cheap wrong one does.
    """
    from ml_stack.graph import bench
    from ml_stack.graph.bench.report import measured_best, write_profiles

    store = str(tmp_path / "runs.ladybug")
    keep_run(store, "thornfield--plain-batch", right=0.75, n=20, seconds=6.0,
             asking={"tight": True, "batch": True})
    keep_run(store, "thornfield--plain-few", right=0.70, n=20, seconds=1.0,
             asking={"tight": True, "few": True, "rounds": 20})
    keep_run(store, "thornfield--plain-single", right=0.10, n=20, seconds=0.5,
             asking={"tight": True, "single": True})

    kept = bench.runs(store)
    from ml_stack.graph.bench.score import band, separated

    slow = next(o for o in kept if o["label"].endswith("batch"))
    quick = next(o for o in kept if o["label"].endswith("few"))
    poor = next(o for o in kept if o["label"].endswith("single"))
    assert band(quick) is not None, "twenty questions carry an interval"
    assert separated(quick, slow) is False, "the questions cannot tell these two apart"
    assert separated(poor, slow) is True, "and they can tell this one from both"

    chosen = measured_best(kept)
    assert chosen["label"] == "thornfield--plain-few"

    where = tmp_path / "written.json"
    write_profiles(kept, path=where)
    one = prof.profiles(package=where, local=where)[0]
    assert (one.few, one.rounds, one.batch, one.single) == (True, 20, False, False), \
        "the winning row's asking, written down whole"
    assert one.label == "thornfield--plain-few", "and the record says which row set it"
    assert one.label in one.note and "fastest" in one.note


def test_the_asking_a_profile_writes_is_the_whole_asking_and_reaches_converse():
    """Every way the bench can measure has to survive the round trip into a record and back
    out as `converse`'s keywords: a way measured and then dropped on the way to the record
    is a measurement paid for and thrown away."""
    made = record(MODEL, tight=True, few=True, single=False, batch=False, rounds=20,
                  reach=8000, kinds=True, summary=True, rich=True,
                  sampling={"temperature": 1.0, "top_p": 0.95, "top_k": 20})
    assert made.asking() == {"tight": True, "kinds": True, "rich": True, "few": True,
                             "summary_tool": True, "reach": 8000, "rounds": 20}
    read_back = Profile.from_dict(made.as_dict())
    assert read_back.asking() == made.asking()

    add(made)
    watching = Watcher()
    converse("who is here?", GRAPH, watching, profile=MODEL)
    assert watching.offered[0] == ["look_up", "look_at", "show"], \
        "the record measured three tools, so three are what the model is offered"


def test_the_shape_a_person_reads_says_the_sampling_it_was_measured_at(capsys):
    """A model card asking for temperature 1.0 and a measurement agreeing with it is the
    thing a person about to serve it needs to see, not infer. Greedy is one word, because
    at temperature 0 nothing else can change an argument."""
    text = said(measured(few=True, rounds=20, sampling={"temperature": 1.0, "top_p": 0.95,
                                                        "top_k": 20}))
    line = next(one for one in text.splitlines() if "ask with" in one)
    assert "few" in line and "rounds 20" in line
    assert "at temperature 1.0 / top-p 0.95 / top-k 20" in line
    assert "greedy" not in line

    greedy = next(one for one in said(measured()).splitlines() if "ask with" in one)
    assert greedy.endswith("greedy") and "temperature" not in greedy

    bare = next(one for one in said(record(OTHER)).splitlines() if "ask with" in one)
    assert bare.strip() == "ask with    tight".strip() or bare.endswith("tight")


def test_rewriting_a_record_keeps_what_a_kept_run_cannot_see(tmp_path):
    from ml_stack.graph.bench.report import write_profiles

    store = str(tmp_path / "runs.ladybug")
    keep_run(store, "thornfield--plain-kv-q8_0-rb0", cache_type="q8_0")
    from ml_stack.graph import bench

    where = tmp_path / "written.json"
    add(measured(), path=where)             # -ub 2048 and the projector, measured by hand
    write_profiles(bench.runs(store), path=where)

    one = prof.profiles(package=where, local=where)[0]
    assert one.extra_args == ("-ub", "2048", "--spec-draft-p-min", "0.5")
    assert one.mmproj == "auto", "a run records neither; a rewrite must not delete them"
    assert one.label == "thornfield--plain-kv-q8_0-rb0", "everything it does see is new"


def test_report_with_no_runs_to_rank_says_so_rather_than_writing_nothing(tmp_path, capsys):
    from ml_stack.graph.bench.report import main as reporting

    assert reporting(SimpleNamespace(kept=str(tmp_path / "nothing.ladybug"), profile=True,
                                     profiles=str(tmp_path / "none.json"),
                                     since="", last=0, model=[], full_n=0)) == 1
    assert "no model has a run" in capsys.readouterr().err


def test_the_named_build_is_read_off_the_binary_a_run_started(tmp_path):
    from ml_stack.graph.bench.report import build_of
    from ml_stack.serve.build import NAMED_DIR

    assert build_of({"binary": str(Path(NAMED_DIR) / "thornfell" / "bin" / "llama-server")}) \
        == "thornfell"
    assert build_of({"binary": "/usr/local/bin/llama-server"}) == "", \
        "the managed current build has no name to be asked for"
    assert build_of({}) == ""


def test_the_shape_a_person_reads_names_the_flags_and_the_ways(capsys):
    text = said(measured())
    assert text.splitlines()[0] == MODEL
    assert "serve with" in text and "ask with" in text and "measured" in text


def test_alone_is_one_seat_holding_the_whole_measured_cache():
    """A record measured at two seats of 32k, asked for as one conversation, is one seat
    of 64k: the least needed, with the largest cache the measurement paid for."""
    from ml_stack.serve.profile import Profile

    record = Profile(model="quince-2b.gguf", seat_context=32768, parallel=2)
    run = record.alone(port=8123, model="quince-2b.gguf", resolve=False)
    assert run.shape.seats == 1 and run.shape.seat_context == 65536 and run.shape.port == 8123
    assert record.run(port=8123, model="quince-2b.gguf", resolve=False).shape.seats == 2
