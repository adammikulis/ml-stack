"""The judge: a model decides the names a spelling apart -- from what it knows, then from the
passages they were read from -- and the pass applies what it decides and records it."""

from ml_stack import ingest
from ml_stack.graph.store import GraphStore
from ml_stack.graph.tidy import DECISIONS, ModelJudge, excerpts, tidy
from tests.test_graph_tidy import _edge, _ids, _node, _store


class Scripted:
    """A client whose `extract` answers from a script keyed on the names it is shown; a
    second look (the passages attached) answers from `after_reading`."""

    def __init__(self, verdicts, after_reading=None):
        self.verdicts, self.after_reading = verdicts, after_reading or {}
        self.calls = []
        self.model = "kestrel-8B"

    def extract(self, text, schema, **kw):
        self.calls.append(text)
        key = next((k for k in self.verdicts if all(name in text for name in k)), None)
        if "Here are passages" in text:
            return {"verdict": self.after_reading.get(key, "unsure"), "why": "the passages"}
        return {"verdict": self.verdicts.get(key, "unsure"), "why": "from what I know"}


def test_same_merges_different_is_remembered_and_neither_needs_apply(tmp_path):
    path = _store(tmp_path, [
        _node("concept:glimmer-node", "glimmer node", mentions=4),
        _node("concept:glimer-node", "glimer node", mentions=1),
        _node("concept:isobutane", "isobutane", mentions=3),
        _node("concept:isobutene", "isobutene", mentions=2),
    ], [_edge("concept:glimer-node", "part_of", "concept:isobutene")])
    client = Scripted({("glimmer node", "glimer node"): "same",
                       ("isobutane", "isobutene"): "different"})
    report = tidy(path, judge=ModelJudge(client))
    assert not report.dry_run, "with a judge the pass is automated"
    assert report.judged_same == 1 and report.judged_different == 1 and report.possible == []
    nodes, edges = _ids(path)
    assert "concept:glimer-node" not in nodes and "concept:isobutene" in nodes
    assert ("concept:glimmer-node", "part_of", "concept:isobutene") in edges
    with GraphStore(path, read_only=True) as store:
        held = store.get_doc(DECISIONS)["pairs"]
    assert len(held) == 2
    verdict = held["concept:isobutane|concept:isobutene"]
    assert verdict["verdict"] == "different" and verdict["model"] == "kestrel-8B"
    assert verdict["why"] and verdict["when"]

    asked = len(client.calls)
    again = tidy(path, judge=ModelJudge(client))
    assert len(client.calls) == asked, "a pair judged different is never asked again"
    assert again.judged_different == 1 and again.nothing_to_do


def test_unsure_goes_back_to_the_source_and_only_a_second_unsure_reaches_a_person(tmp_path):
    path = _store(tmp_path, [
        _node("concept:flux-ring", "flux ring", mentions=3),
        _node("concept:flux-rings-2", "flux rlng", mentions=1),
        _node("concept:vault", "vault", mentions=3),
        _node("concept:vaulf", "vaulf", mentions=1),
    ], [])
    texts = {"u:concept:flux-ring": "A flux ring holds charge. The flux ring is a ring.",
             "u:concept:flux-rings-2": "The flux rlng (a misprint for flux ring) holds charge.",
             "u:concept:vault": "A vault holds nodes.", "u:concept:vaulf": "The vaulf is unknown."}
    client = Scripted({("flux ring", "flux rlng"): "unsure", ("vault", "vaulf"): "unsure"},
                      after_reading={("flux ring", "flux rlng"): "same"})
    judge = ModelJudge(client, sources=texts.__getitem__)
    report = tidy(path, judge=judge)
    assert judge.read == 2, "both pairs were read for"
    assert report.judged_same == 1
    assert [tuple(sorted(p)) for p in report.possible] == [("vaulf", "vault")]
    reads = [c for c in client.calls if "Here are passages" in c]
    assert any("misprint" in c for c in reads), "the passage the name was read in"
    with GraphStore(path, read_only=True) as store:
        held = store.get_doc(DECISIONS)["pairs"]["concept:flux-ring|concept:flux-rings-2"]
    assert held["verdict"] == "same" and set(held["read"]) == {"u:concept:flux-ring",
                                                               "u:concept:flux-rings-2"}


def test_excerpts_window_around_every_mention_and_fall_back_to_the_start():
    text = "Sodium, once Natrium, is Na. " * 3 + "Nothing else."
    found = excerpts(text, "natrium", chars=20, most=2)
    assert len(found) == 2 and all("Natrium" in piece for piece in found)
    assert excerpts("Only prose here.", "absent", chars=8) == ["Only pro"]


def test_sources_come_from_memory_first_and_the_book_second(tmp_path):
    from tests.test_ingest import a_part_read_book

    store = a_part_read_book(tmp_path)
    text_of = ingest.sources_for(store, texts={"velthorne-open-texts:1:1.1": "in memory"})
    assert text_of("velthorne-open-texts:1:1.1") == "in memory"
    assert text_of("velthorne-open-texts:9:9.9") == "", "no PDF at the recorded path: nothing"


def test_the_run_tidies_each_book_on_the_way_out_with_its_own_model(tmp_path, server, monkeypatch, capsys):
    from tests.test_ingest import a_shelf

    book, instance, _ = a_shelf(tmp_path, server)
    seen = {}

    class Judge:
        model = "scripted"

        def decide(self, a, b):
            seen["asked"] = True
            return {"verdict": "different", "why": "scripted", "read": []}

    monkeypatch.setattr(ingest, "_judge", lambda client, out, **kw: Judge())
    store = tmp_path / "shelf.ladybug"
    assert ingest.main([book, "--out", str(store), "--base-url", instance.base_url]) == 0
    out = capsys.readouterr().out
    assert "tidied:" in out
    assert ingest.main([book, "--out", str(store), "--base-url", instance.base_url,
                        "--resume", "--no-tidy"]) == 0
    assert "tidied:" not in capsys.readouterr().out
