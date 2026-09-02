"""An extraction scored against the gold the simulation wrote.

Every person, place and organisation here is invented by `ml_stack.world` from a seed or
made up on the spot; the model is a fake that returns scripted extractions. Nothing reads
a real graph, a real store outside `tmp_path`, or a server.
"""

from __future__ import annotations

import dataclasses
import json
import random

import pytest

from ml_stack.client.families import GENERIC
from ml_stack.graph.bench import extract as bx
from ml_stack.graph.bench import MEASURING, RunNotKept, _parser, runs
from ml_stack.world.organisation import make
from ml_stack.world.simulate import simulate


# -- a tiny world with messages -----------------------------------------------------------------

def talked(kind: str = "company", days: int = 5, seed: int = 1) -> tuple[dict, list[dict]]:
    """A small invented world and its template-written messages, as `messages.jsonl` rows."""
    world = make(kind, "small", seed=seed)
    messages = [json.loads(json.dumps(dataclasses.asdict(m)))
                for m in simulate(world, days=days, writer=None, rng=random.Random(seed))]
    return world.graph, messages


def labels(graph: dict) -> dict[str, str]:
    return {n["id"]: n["label"] for n in graph["nodes"]}


# -- the sample -----------------------------------------------------------------------------------

def test_the_sample_keeps_arcs_and_chatter_both_and_is_the_same_for_a_seed():
    _, messages = talked(days=10)
    strata = {bx._stratum(m) for m in messages}
    assert any(s.startswith("arc:") for s in strata) and any(s.startswith("chat:") for s in strata)
    picked = bx.sample_messages(messages, 12, seed=0)
    assert len(picked) == 12
    got = {bx._stratum(m) for m in picked}
    assert any(s.startswith("arc:") for s in got), "an arc's thread is a handful in a fortnight"
    assert any(s.startswith("chat:") for s in got)
    assert [m["id"] for m in picked] == [m["id"] for m in bx.sample_messages(messages, 12, seed=0)]
    assert [m["id"] for m in picked] != [m["id"] for m in bx.sample_messages(messages, 12, seed=1)]
    # asking for everything is everything, in order
    assert [m["id"] for m in bx.sample_messages(messages, 0)] == [m["id"] for m in messages]


# -- the fold ---------------------------------------------------------------------------------------

def test_the_fold_joins_case_and_near_spellings_and_a_first_name_to_its_full_name():
    folded = bx.fold([
        {"people": [{"name": "Tove Varga", "role": "", "org": "Pellard Foundry", "place": ""}],
         "orgs": [{"name": "Pellard Foundry", "kind": "customer"}], "topics": [], "places": [],
         "relations": [{"from": "Tove Varga", "rel": "works_at", "to": "Pellard Foundry"}]},
        {"people": [{"name": "Tove", "role": "", "org": "", "place": "Turin"}],
         "orgs": [{"name": "Pellard foundry", "kind": ""}], "topics": ["Robotics"],
         "places": ["Turin"], "relations": []},
        {"people": [], "orgs": [{"name": "Pelard Foundry", "kind": ""}], "topics": ["robotics"],
         "places": ["turin"], "relations": []},
    ])
    assert [c["name"] for c in folded["nodes"]["orgs"]] == ["Pellard Foundry"]
    assert sorted(folded["nodes"]["orgs"][0]["names"]) == ["Pelard Foundry", "Pellard Foundry",
                                                            "Pellard foundry"]
    assert [c["name"] for c in folded["nodes"]["people"]] == ["Tove Varga"]
    assert folded["nodes"]["people"][0]["attrs"] == {"org": "Pellard Foundry", "place": "Turin"}
    assert [c["name"] for c in folded["nodes"]["topics"]] == ["Robotics"]
    assert [c["name"] for c in folded["nodes"]["places"]] == ["Turin"]
    assert folded["relations"] == [{"from": "Tove Varga", "rel": "works_at", "to": "Pellard Foundry"}]


def test_short_names_are_not_joined_on_one_letter():
    """`spelling.close` affords a four-letter word no edit: Tove and Teve are two people."""
    folded = bx.fold([{"people": [{"name": "Tove Varga", "role": "", "org": "", "place": ""},
                                  {"name": "Teve Varga", "role": "", "org": "", "place": ""}],
                       "orgs": [], "topics": [], "places": [], "relations": []}])
    assert len(folded["nodes"]["people"]) == 2


# -- the gold and the score ---------------------------------------------------------------------

def test_the_gold_is_the_union_of_what_the_messages_assert_and_nothing_a_message_did_not():
    graph, messages = talked()
    picked = bx.sample_messages(messages, 20, seed=0)
    held = bx.gold(graph, picked)
    asserted = {i for m in picked for b in bx.BUCKETS for i in m["attrs"]["asserts"][b]}
    assert {i for b in bx.BUCKETS for i in held["nodes"][b]} == asserted
    # a person the world holds who is in no sampled message is not in the gold
    everyone = {n["id"] for n in graph["nodes"] if n["kind"] == "person"}
    assert everyone - asserted, "a small world has more people than twenty messages name"
    assert not (everyone - asserted) & set(held["nodes"]["people"])
    assert held["exact"] is True
    assert "works_at" in held["vocabulary"]


def test_a_message_without_asserts_is_refused_rather_than_guessed_at():
    graph, messages = talked()
    old = dict(messages[0], attrs={k: v for k, v in messages[0]["attrs"].items()
                                   if k != "asserts"})
    with pytest.raises(ValueError, match="no attrs.asserts"):
        bx.gold(graph, [old])


def test_the_gold_fed_back_scores_everything_and_invents_nothing():
    graph, messages = talked("community", days=10)
    picked = bx.sample_messages(messages, 40, seed=0)
    held = bx.gold(graph, picked)
    folded = bx.fold([bx.as_extraction(bx.gold(graph, [m])) for m in picked])
    scores = bx.score(folded, held, per_message=[m["attrs"]["asserts"] for m in picked])
    assert scores["nodes"]["coverage"] == scores["nodes"]["precision"] == scores["nodes"]["f1"] == 1.0
    for bucket in bx.BUCKETS:
        held_kind = scores["by_kind"][bucket]
        assert held_kind["invented"] == 0
        if held_kind["of"]:
            assert held_kind["coverage"] == held_kind["precision"] == 1.0
    assert scores["invented"] == {"count": 0, "of": scores["by_kind"]["people"]["said"]
                                  + scores["by_kind"]["orgs"]["said"], "rate": 0.0}
    if scores["relations"]["of"]:
        assert scores["relations"]["coverage"] == scores["relations"]["precision"] == 1.0
    assert scores["survival"] == {"mean": 1.0, "messages": len(picked)}
    assert scores["resolution"] == {"splits": 1.0, "merges": 1.0}
    assert scores["conformance"]["entities"]["share"] == 1.0
    assert scores["conformance"]["off_schema"] == 0
    assert scores["topology"]["extracted"] == scores["topology"]["gold"]


def test_an_invented_person_counts_against_precision_and_as_invented():
    graph, messages = talked()
    picked = bx.sample_messages(messages, 10, seed=0)
    held = bx.gold(graph, picked)
    truth = [bx.as_extraction(bx.gold(graph, [m])) for m in picked]
    truth[0]["people"].append({"name": "Orsolya Brandt", "role": "", "org": "", "place": ""})
    scores = bx.score(bx.fold(truth), held)
    people = scores["by_kind"]["people"]
    assert people["coverage"] == 1.0 and people["invented"] == 1
    assert people["precision"] == pytest.approx(people["found"] / (people["found"] + 1), abs=1e-3)
    assert scores["invented"]["count"] == 1 and scores["invented"]["rate"] > 0
    assert scores["nodes"]["precision"] < 1.0


def test_a_project_the_messages_assert_under_others_is_neither_found_nor_invented():
    graph, messages = talked()
    picked = bx.sample_messages(messages, 20, seed=0)
    held = bx.gold(graph, picked)
    assert held["others"], "a company's chatter is about its projects"
    project = next(iter(held["others"].values()))["label"]
    truth = [bx.as_extraction(bx.gold(graph, [m])) for m in picked]
    truth[0]["orgs"].append({"name": project, "kind": "project"})
    scores = bx.score(bx.fold(truth), held)
    assert scores["by_kind"]["orgs"]["invented"] == 0
    assert scores["nodes"]["precision"] == 1.0


def test_relations_match_loosely_on_the_name_and_strictly_on_the_ends():
    nodes = {"person:ada": {"id": "person:ada", "kind": "person", "label": "Ada Marlow"},
             "person:bea": {"id": "person:bea", "kind": "person", "label": "Bea Ostley"},
             "org:pellard": {"id": "org:pellard", "kind": "org", "label": "Pellard Foundry"}}
    held = {"nodes": {"people": {"person:ada": nodes["person:ada"], "person:bea": nodes["person:bea"]},
                      "orgs": {"org:pellard": nodes["org:pellard"]}, "topics": {}, "places": {}},
            "others": {}, "relations": [["person:ada", "works_at", "org:pellard"],
                                         ["person:ada", "works_with", "person:bea"]],
            "exact": True, "vocabulary": ["works_at", "works_with"]}
    said = {"people": [{"name": "Ada Marlow", "role": "", "org": "", "place": ""},
                       {"name": "Bea Ostley", "role": "", "org": "", "place": ""}],
            "orgs": [{"name": "Pellard Foundry", "kind": ""}], "topics": [], "places": [],
            "relations": [{"from": "Ada Marlow", "rel": "Works At", "to": "Pellard foundry"},
                          {"from": "Bea Ostley", "rel": "works with", "to": "Ada Marlow"}]}
    scores = bx.score(bx.fold([said]), held)
    # "Works At" is works_at; "works with" reversed is not the edge the gold states
    assert scores["relations"]["found"] == 1 and scores["relations"]["said"] == 2
    assert scores["relations"]["coverage"] == 0.5 and scores["relations"]["precision"] == 0.5
    assert scores["conformance"]["relations"] == {"in_vocabulary": 2, "of": 2, "share": 1.0}
    # a relation to somebody who is not in the gold is invented; one to an off-schema
    # thing is dropped
    said["relations"] = [{"from": "Ada Marlow", "rel": "mentors", "to": "Orsolya Brandt"}]
    scores = bx.score(bx.fold([said]), held)
    assert scores["relations"]["said"] == 1 and scores["relations"]["invented"] == 1
    assert scores["conformance"]["off_schema"] == 1


def test_a_fold_that_merges_two_people_is_caught_by_survival_and_merges():
    nodes = {"person:tove": {"id": "person:tove", "kind": "person", "label": "Tove Varga"},
             "person:nell": {"id": "person:nell", "kind": "person", "label": "Nell Varga"}}
    held = {"nodes": {"people": dict(nodes), "orgs": {}, "topics": {}, "places": {}},
            "others": {}, "relations": [], "exact": True, "vocabulary": []}
    one_cluster = {"nodes": {"people": [{"name": "Tove Varga", "names": ["Tove Varga", "Nell Varga"],
                                          "attrs": {}}], "orgs": [], "topics": [], "places": []},
                   "relations": []}
    per_message = [{"people": ["person:tove"], "orgs": [], "topics": [], "places": [],
                    "others": [], "relations": []},
                   {"people": ["person:nell"], "orgs": [], "topics": [], "places": [],
                    "others": [], "relations": []}]
    scores = bx.score(one_cluster, held, per_message=per_message)
    assert scores["resolution"]["merges"] == 2.0
    assert scores["survival"]["mean"] == 0.5
    two_clusters = {"nodes": {"people": [{"name": "Tove Varga", "names": ["Tove Varga"], "attrs": {}},
                                          {"name": "Tove  Varga", "names": ["Tove  Varga"], "attrs": {}}],
                              "orgs": [], "topics": [], "places": []}, "relations": []}
    scores = bx.score(two_clusters, held, per_message=per_message)
    assert scores["resolution"]["splits"] == 2.0


def test_consistency_is_the_jaccard_of_two_folds():
    a = bx.fold([{"people": [{"name": "Ada Marlow", "role": "", "org": "", "place": ""}],
                  "orgs": [], "topics": ["robotics"], "places": [],
                  "relations": [{"from": "Ada Marlow", "rel": "interested_in", "to": "robotics"}]}])
    b = bx.fold([{"people": [{"name": "ada marlow", "role": "", "org": "", "place": ""}],
                  "orgs": [], "topics": ["robotics", "compilers"], "places": [], "relations": []}])
    assert bx.consistency(a, a) == {"nodes": 1.0, "relations": 1.0}
    assert bx.consistency(a, b) == {"nodes": pytest.approx(2 / 3, abs=1e-3), "relations": 0.0}


# -- the client, the table and the store ------------------------------------------------------------

@dataclasses.dataclass
class Reply:
    content: str
    raw: dict
    tool_calls: list | None = None
    thinking: str | None = None
    finish_reason: str | None = None


class Reader:
    """A model that reads the sender off the prompt and returns them, and nothing else --
    through the real `Client.extract`, so the prompt, the schema and the JSON are exercised."""

    family = GENERIC
    card: dict = {}

    def __init__(self, base_url: str = "", timeout: float = 0.0, **sampling) -> None:
        self.base_url = base_url
        self.timeout = timeout
        self.sampling = dict(sampling)
        self.seen: list[list[dict]] = []

    def chat(self, messages, **kw):
        self.seen.append(list(messages))
        head = messages[-1]["content"].split(":\n", 1)[0]
        sender = head.removeprefix("From ").split(" in ")[0]
        got = {"people": [{"name": sender, "role": "", "org": "", "place": ""}], "orgs": [],
               "topics": [], "places": [], "relations": []}
        return Reply(content=json.dumps(got),
                     raw={"usage": {"prompt_tokens": 40, "completion_tokens": 12},
                          "timings": {"prompt_n": 30, "cache_n": 10}})


def test_extract_one_goes_through_the_counted_client_and_names_the_sender():
    graph, messages = talked()
    shape = bx.schema()
    reader = Reader()
    m = messages[0]
    sender = labels(graph)[m["sender"]]
    row = bx.extract_one(reader, m, sender, shape)
    assert row.extracted["people"] == [{"name": sender, "role": "", "org": "", "place": ""}]
    assert row.prompt_tokens == 40 and row.completion_tokens == 12 and row.cached_tokens == 10
    assert row.exact is True and not row.timed_out and not row.error
    system, user = reader.seen[0]
    assert system["role"] == "system" and "invent nothing" in system["content"]
    assert user["content"].startswith(f"From {sender} in ") and m["text"] in user["content"]


def test_measure_scores_the_sender_only_reader_at_full_people_precision(tmp_path):
    graph, messages = talked()
    picked = bx.sample_messages(messages, 8, seed=0)
    rows, scores = bx.measure(Reader(), picked, graph)
    assert len(rows) == 8
    people = scores["by_kind"]["people"]
    assert people["precision"] == 1.0 and people["invented"] == 0
    assert 0 < people["coverage"] <= 1.0
    assert scores["by_kind"]["topics"]["found"] == 0
    assert "lower_bound" not in scores


def test_a_run_is_kept_read_back_and_shown_in_its_own_table(tmp_path, capsys):
    pytest.importorskip("ladybug")
    graph, messages = talked()
    picked = bx.sample_messages(messages, 5, seed=0)
    rows, scores = bx.measure(Reader(), picked, graph)
    kept = tmp_path / "runs.ladybug"
    key = bx.save(kept, rows, label="reader", model="fake-reader", world={"kind": "company"},
                  scores=scores, sample={"n": 5}, held={"resident_bytes": 3 * 2**30})
    back = runs(kept)
    assert [r["key"] for r in back] == [key] and back[0]["kind"] == "extract"
    assert back[0]["scores"]["by_kind"] == scores["by_kind"]
    assert len(back[0]["rows"]) == 5 and back[0]["model"] == "fake-reader"
    bx.table(bx.only(back))
    out = capsys.readouterr().out
    head = out.splitlines()[0]
    assert head.split() == ["run", "model", "msgs", "s/msg", "tok/msg", "t/o", "ppl-cov",
                            "ppl-prec", "org-cov", "org-prec", "top-cov", "top-prec", "plc-cov",
                            "plc-prec", "rel-cov", "rel-prec", "n-F1", "r-F1", "invented", "org",
                            "place", "resident"]
    line = next(ln for ln in out.splitlines() if ln.startswith("reader"))
    assert "fake-reader" in line and " 5 " in line and "3.00G" in line and "100%" in line
    assert "topology:" in out and "conformance:" in out and "survival:" in out
    assert "resolution:" in out


def test_the_estimate_comes_from_earlier_runs_of_the_same_model_else_a_guess(tmp_path):
    pytest.importorskip("ladybug")
    graph, messages = talked()
    rows, scores = bx.measure(Reader(), bx.sample_messages(messages, 4, seed=0), graph)
    for r in rows:
        r.seconds = 2.0
    kept = tmp_path / "runs.ladybug"
    assert bx.estimate(kept, "fake-reader", 40) == (bx.GUESS_SECONDS, "a guess, no earlier run of this model")
    bx.save(kept, rows, label="reader", model="fake-reader", world={}, scores=scores, sample={})
    per, source = bx.estimate(kept, "fake-reader", 40)
    assert per == 2.0 and "4 earlier messages of fake-reader" in source


def _world_dir(tmp_path):
    world = make("company", "small", seed=1)
    (tmp_path / "world").mkdir()
    (tmp_path / "world" / "graph.json").write_text(json.dumps(world.graph))
    (tmp_path / "world" / "personas.json").write_text(json.dumps(world.personas))
    (tmp_path / "world" / "world.json").write_text(json.dumps(
        {"kind": "company", "size": "small", "seed": 1, "people": world.people}))
    return tmp_path / "world"


def test_a_world_without_messages_is_simulated_and_said_so(tmp_path):
    graph, messages, note = bx.load_world(_world_dir(tmp_path), days=2)
    assert messages and "no messages.jsonl" in note and "simulated 2 working days" in note
    assert all("asserts" in m["attrs"] for m in messages)
    assert "meta" in graph


def test_the_smoke_run_reads_three_messages_and_reads_the_run_back(tmp_path, monkeypatch, capsys):
    pytest.importorskip("ladybug")
    import ml_stack.client
    from ml_stack.graph import bench

    monkeypatch.setattr(bench, "HOME", tmp_path / "home")
    monkeypatch.setattr(bx, "footprint", lambda url: {"base_url": url, "model": "fake.gguf"})
    monkeypatch.setattr(bench, "busy", lambda url: 0)   # _idle asks bench.busy
    monkeypatch.setattr(ml_stack.client, "Client", Reader)
    kept = tmp_path / "runs.ladybug"
    assert bench._main(["extract", "smoke", "--world", str(_world_dir(tmp_path)), "--smoke",
                        "--kept", str(kept), "--base-url", "http://127.0.0.1:1"]) == 0
    out = capsys.readouterr().out
    assert "no messages.jsonl" in out
    assert "smoke: 3 of " in out and "s/msg" in out and "min" in out
    back = runs(kept)
    assert len(back) == 1 and back[0]["kind"] == "extract" and back[0]["model"] == "fake"
    assert len(back[0]["rows"]) == 3 and back[0]["sample"]["n"] == 3
    assert "kept as bench:smoke:" in out
    assert out.splitlines()[-1].strip().startswith(("topology:", "conformance:", "survival:",
                                                    "resolution:"))


def test_a_smoke_run_whose_run_does_not_come_back_raises(tmp_path, monkeypatch):
    pytest.importorskip("ladybug")
    import ml_stack.client
    from ml_stack.graph import bench

    monkeypatch.setattr(bench, "HOME", tmp_path / "home")
    monkeypatch.setattr(bx, "footprint", lambda url: {"base_url": url})
    monkeypatch.setattr(bench, "busy", lambda url: 0)   # _idle asks bench.busy
    monkeypatch.setattr(ml_stack.client, "Client", Reader)
    real = bx.runs
    calls = {"n": 0}

    def then_nothing(store, label=""):
        calls["n"] += 1
        return real(store, label) if calls["n"] == 1 else []   # save's own check passes

    monkeypatch.setattr(bx, "runs", then_nothing)
    with pytest.raises(RunNotKept):
        bench._main(["extract", "smoke", "--world", str(_world_dir(tmp_path)), "--smoke",
                     "--kept", str(tmp_path / "runs.ladybug"), "--base-url", "http://127.0.0.1:1"])


def test_twice_reads_the_sample_again_and_reports_how_alike_the_two_were(tmp_path, monkeypatch, capsys):
    pytest.importorskip("ladybug")
    import ml_stack.client
    from ml_stack.graph import bench

    monkeypatch.setattr(bench, "HOME", tmp_path / "home")
    monkeypatch.setattr(bx, "footprint", lambda url: {"base_url": url})
    monkeypatch.setattr(bench, "busy", lambda url: 0)   # _idle asks bench.busy
    monkeypatch.setattr(ml_stack.client, "Client", Reader)
    kept = tmp_path / "runs.ladybug"
    assert bench._main(["extract", "again", "--world", str(_world_dir(tmp_path)), "--smoke",
                        "--twice", "--kept", str(kept), "--base-url", "http://127.0.0.1:1"]) == 0
    out = capsys.readouterr().out
    assert "reading the sample again with the same settings again" in out
    assert "consistency: nodes J=1.00" in out
    one = runs(kept)[0]
    assert one["scores"]["consistency"]["nodes"] == 1.0
    assert len(one["server"]["twice"]["rows"]) == 3


def test_show_prints_extraction_runs_under_the_answering_table_or_alone(tmp_path, capsys):
    pytest.importorskip("ladybug")
    from ml_stack.graph import bench

    graph, messages = talked()
    rows, scores = bx.measure(Reader(), bx.sample_messages(messages, 3, seed=0), graph)
    kept = tmp_path / "runs.ladybug"
    bx.save(kept, rows, label="reader", model="fake", world={}, scores=scores, sample={})
    assert bench._main(["show", "--kept", str(kept)]) == 0
    out = capsys.readouterr().out
    assert out.startswith("nothing kept yet")             # no answering run to show
    assert "ppl-cov" in out and "reader" in out
    assert bench._main(["show", "--kept", str(kept), "--extract"]) == 0
    out = capsys.readouterr().out
    assert "nothing kept yet" not in out and out.startswith("run ")


def test_extract_is_a_measuring_subcommand_with_help(capsys):
    assert "extract" in MEASURING
    with pytest.raises(SystemExit) as stopped:
        _parser().parse_args(["extract", "--help"])
    assert stopped.value.code == 0
    out = capsys.readouterr().out
    for flag in ("--world", "--serve", "--sample", "--seed", "--per-message", "--smoke",
                 "--detach", "--twice", "--context", "--parallel"):
        assert flag in out
    args = _parser().parse_args(["extract", "x", "--world", "w"])
    assert args.sample == bx.SAMPLE and args.context == 65536 and args.parallel == 2
    assert args.per_message == 300.0
