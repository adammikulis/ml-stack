"""The judge: a model decides the names a spelling apart, which of two verbs between the same
ends is the relationship, which of two definitions to keep and what a doubtful label really is
-- from what it knows, then from the passages -- and the pass applies it and records it."""

import json

from ml_stack import ingest
from ml_stack.graph.store import GraphStore
from ml_stack.graph.tidy import (DECISIONS, ModelJudge, absorb, excerpts, judge_gold,
                                 load_gold, tidy)
from tests.test_graph_tidy import _edge, _ids, _node, _store


class Scripted:
    """A client whose `extract` answers from a script keyed on what the question contains.

    `verdicts` answers the two-names question and `after_reading` the second look; `conflicts`,
    `definitions` and `suspects` answer the other three, each keyed the same way -- a tuple of
    words the question must contain.
    """

    def __init__(self, verdicts=None, after_reading=None, conflicts=None, definitions=None,
                 suspects=None):
        self.verdicts, self.after_reading = verdicts or {}, after_reading or {}
        self.conflicts, self.definitions = conflicts or {}, definitions or {}
        self.suspects = suspects or {}
        self.calls = []
        self.model = "kestrel-8B"

    @staticmethod
    def _key(script, text):
        return next((k for k in script if all(word in text for word in k)), None)

    def extract(self, text, schema, **kw):
        self.calls.append(text)
        properties = (schema or {}).get("properties") or {}
        if "keep" in properties:
            return {"keep": self.definitions.get(self._key(self.definitions, text), "both"),
                    "why": "the fuller one"}
        choices = (properties.get("verdict") or {}).get("enum") or []
        if "rename" in choices:
            return {"verdict": "keep", "why": "scripted",
                    **(self.suspects.get(self._key(self.suspects, text)) or {})}
        if any(str(choice).startswith("keep ") for choice in choices):
            return {"verdict": self.conflicts.get(self._key(self.conflicts, text), "unsure"),
                    "why": "from the passages"}
        if "Here are passages" in text:
            return {"verdict": self.after_reading.get(self._key(self.after_reading, text),
                                                      "unsure"), "why": "the passages"}
        return {"verdict": self.verdicts.get(self._key(self.verdicts, text), "unsure"),
                "why": "from what I know"}


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


# -- the gold set: the judge's verdicts scored against pairs whose answers are known

def _a_gold_file(tmp_path):
    pairs = [
        {"class": "same", "verdict": "same",
         "a": {"label": "glimmer node", "mentions": 4, "provenance": ["g:1"]},
         "b": {"label": "glimer node", "mentions": 1, "provenance": ["g:2"]},
         "passages": {"g:1": "A glimmer node holds charge.",
                      "g:2": "The glimer node holds charge."}},
        {"class": "different", "verdict": "different",
         "a": {"label": "sylvane", "mentions": 3, "provenance": ["g:3"]},
         "b": {"label": "sylvene", "mentions": 2, "provenance": ["g:4"]},
         "passages": {"g:3": "Sylvane carries one ring.", "g:4": "Sylvene carries two."}},
        {"class": "unsure-then-same", "verdict": "same",
         "a": {"label": "cinder vault", "mentions": 5, "provenance": ["g:5"]},
         "b": {"label": "cinder vaulf", "mentions": 1, "provenance": ["g:6"]},
         "passages": {"g:5": "A cinder vault holds a stack of plates.",
                      "g:6": "cinder vaulf is a misprint for cinder vault."}},
        {"class": "unsure-then-different", "verdict": "different",
         "a": {"label": "tessel ring", "mentions": 4, "provenance": ["g:7"]},
         "b": {"label": "tessel ridge", "mentions": 2, "provenance": ["g:8"]},
         "passages": {"g:7": "The tessel ring is the circuit cut into a plate.",
                      "g:8": "The tessel ridge is the rim standing around the ring."}},
    ]
    path = tmp_path / "gold.json"
    path.write_text(json.dumps({"pairs": pairs}), encoding="utf-8")
    return path


def test_the_shipped_gold_set_covers_four_classes_with_passages_for_both_names():
    pairs = load_gold()
    assert len(pairs) >= 24
    classes = {}
    for pair in pairs:
        classes[pair["class"]] = classes.get(pair["class"], 0) + 1
        units = set(pair["a"]["provenance"]) | set(pair["b"]["provenance"])
        assert units == set(pair["passages"]), f"{pair['a']['label']}: a passage per unit"
        assert pair["verdict"] in ("same", "different")
        assert pair["verdict"] == pair["class"] or pair["class"].endswith(pair["verdict"])
    assert set(classes) == {"same", "different", "unsure-then-same", "unsure-then-different"}
    assert min(classes.values()) >= 6, "every class carries enough pairs to move the number"


def test_judge_gold_scores_overall_and_per_class_and_counts_the_second_looks(tmp_path):
    client = Scripted({("glimmer node", "glimer node"): "same",
                       ("sylvane", "sylvene"): "different"},
                      after_reading={("cinder vault", "cinder vaulf"): "same",
                                     ("tessel ring", "tessel ridge"): "same"})
    scored = judge_gold(client, _a_gold_file(tmp_path))
    assert scored.total == 4 and scored.right == 3
    assert scored.accuracy == 0.75 and scored.model == "kestrel-8B"
    assert scored.per_class == {"same": [1, 1], "different": [1, 1],
                                "unsure-then-same": [1, 1], "unsure-then-different": [0, 1]}
    assert scored.read == 2, "only the two the names could not settle were read for"
    assert scored.wrong == [("tessel ring", "tessel ridge", "different", "same")]
    assert "4 pair(s), 3 right (75%)" in scored.said() and "needed the passages" in scored.said()
    assert scored.seconds >= 0


def test_the_gold_gate_prints_the_number_and_exits_one_under_the_bar(tmp_path, capsys,
                                                                    monkeypatch):
    from ml_stack.graph import store_cli

    gold = _a_gold_file(tmp_path)
    client = Scripted({("glimmer node", "glimer node"): "same",
                       ("sylvane", "sylvene"): "different"},
                      after_reading={("cinder vault", "cinder vaulf"): "same",
                                     ("tessel ring", "tessel ridge"): "same"})
    monkeypatch.setattr("ml_stack.client.Client", lambda *a, **kw: client)
    assert store_cli.main(["tidy", "--gold", str(gold), "--base-url", "http://nowhere"]) == 0
    out = capsys.readouterr().out
    assert "4 pair(s), 3 right (75%)" in out and "wrong (unsure-then-different)" in out
    assert store_cli.main(["tidy", "--gold", str(gold), "--base-url", "http://nowhere",
                           "--fail-under", "0.7"]) == 0
    assert store_cli.main(["tidy", "--gold", str(gold), "--base-url", "http://nowhere",
                           "--fail-under", "0.9"]) == 1
    assert "below the bar: 75%" in capsys.readouterr().err
    assert store_cli.main(["tidy", "--gold", str(gold)]) == 2, "no model to ask"


# -- verb conflicts

def _two_verbs(tmp_path):
    return _store(tmp_path, [
        _node("concept:vault-current", "vault current", mentions=4,
              definition="the current that runs the length of a vault"),
        _node("concept:thrum-coil", "thrum coil", mentions=3,
              definition="the winding that turns a current into a hum"),
    ], [_edge("concept:vault-current", "causes", "concept:thrum-coil", 2),
        _edge("concept:vault-current", "regulates", "concept:thrum-coil", 5)])


def test_a_verb_conflict_the_judge_settles_drops_the_rejected_edge_into_the_kept_one(tmp_path):
    path = _two_verbs(tmp_path)
    texts = {"u:concept:vault-current":
             "The vault current regulates the thrum coil; it does not cause the coil."}
    client = Scripted(conflicts={("causes", "regulates"): "keep regulates"})
    judge = ModelJudge(client, sources=lambda unit: texts.get(unit, ""))
    report = tidy(path, judge=judge)
    assert report.conflicts_judged == 1 and report.conflict_edges_dropped == 1
    assert report.conflicts == [], "settled, so nothing is left for a person"
    assert judge.read == 1, "both edges' passages were shown"
    asked = client.calls[0]
    assert "does not cause the coil" in asked and "definition:" in asked
    assert "vault current causes thrum coil" in asked
    _nodes, edges = _ids(path)
    assert ("concept:vault-current", "causes", "concept:thrum-coil") not in edges
    kept = edges[("concept:vault-current", "regulates", "concept:thrum-coil")]
    assert kept["weight"] == 7, "the dropped edge's weight folded in"
    assert set(kept["provenance"]) == {"u:concept:vault-current"}
    assert "put 1 conflicting verb pair(s) to the judge (1 edge(s) dropped)" in report.said()


def test_a_conflict_the_judge_keeps_both_is_reported_and_never_asked_twice(tmp_path):
    path = _two_verbs(tmp_path)
    client = Scripted(conflicts={("causes", "regulates"): "keep both"})
    report = tidy(path, judge=ModelJudge(client))
    assert report.conflicts_judged == 1 and report.conflict_edges_dropped == 0
    assert report.conflicts == [("concept:thrum-coil", "concept:vault-current",
                                 "causes", "regulates")]
    assert len(_ids(path)[1]) == 2, "both kept"
    with GraphStore(path, read_only=True) as store:
        held = store.get_doc(DECISIONS)["conflicts"]
    key = "concept:thrum-coil|concept:vault-current::causes|regulates"
    assert held[key]["verdict"] == "keep both" and held[key]["model"] == "kestrel-8B"

    asked = len(client.calls)
    again = tidy(path, judge=ModelJudge(client))
    assert len(client.calls) == asked, "a conflict once judged is never asked again"
    assert again.conflicts_judged == 1 and len(again.conflicts) == 1


# -- definitions

def test_the_judge_picks_the_definition_only_when_the_two_say_different_things(tmp_path):
    path = _store(tmp_path, [
        _node("concept:vault", "vault", mentions=5, definition="a housing for plates"),
        _node("concept:vaults", "vaults", mentions=2,
              definition="the sablon box a stack of lattice plates sits in"),
        _node("concept:ring", "ring", mentions=4, definition="a closed loop"),
        _node("concept:rings", "rings", mentions=1, definition="a closed loop of lattice"),
    ], [])
    client = Scripted(definitions={("a housing for plates",): "b"})
    report = tidy(path, judge=ModelJudge(client))
    assert report.merged_nodes == 2
    assert report.definitions_judged == 1, "one definition is the start of the other: not asked"
    nodes = _ids(path)[0]
    vault = nodes["concept:vault"]["attrs"]
    assert vault["definition"] == "the sablon box a stack of lattice plates sits in"
    assert vault["definitions_also"] == ["a housing for plates"], "nothing is lost"
    ring = nodes["concept:ring"]["attrs"]
    assert ring["definition"] == "a closed loop of lattice", "the fuller of the two"
    assert not ring.get("definitions_also"), "a prefix is not a second definition"
    with GraphStore(path, read_only=True) as store:
        held = store.get_doc(DECISIONS)["definitions"]
    assert held["concept:vault|concept:vaults"]["keep"] == "b"


# -- suspect labels

def test_a_suspect_label_is_renamed_dropped_or_kept_from_its_passages(tmp_path):
    path = _store(tmp_path, [
        _node("concept:clause", "that holds charge between pulses"),
        _node("concept:number", "42"),
        _node("concept:generic", "process"),
        _node("concept:other", "vault current", mentions=3),
    ], [_edge("concept:clause", "part_of", "concept:other")])
    texts = {"u:concept:clause": "Each glimmer node holds charge between pulses.",
             "u:concept:number": "Table 42 lists the plate grades.",
             "u:concept:generic": "The process of grinding a plate is called lapping."}
    client = Scripted(suspects={
        ("that holds charge between pulses",): {"verdict": "rename", "name": "glimmer node"},
        ("'42'",): {"verdict": "drop", "why": "a table number, not a thing"},
        ("'process'",): {"verdict": "keep", "why": "the name the index uses"},
    })
    judge = ModelJudge(client, sources=lambda unit: texts.get(unit, ""))
    report = tidy(path, judge=judge)
    assert report.flagged == 0, "with a judge nothing is left flagged"
    assert report.suspects_resolved == 3 and report.suspects_dropped == 1
    assert any("dropped" in line and "'42'" in line for line in report.lines)
    nodes, edges = _ids(path)
    assert nodes["concept:clause"]["label"] == "glimmer node"
    assert "concept:number" not in nodes, "the pass's one removal beyond duplicates"
    assert nodes["concept:generic"]["attrs"]["suspect"] == "", "kept, the flag cleared"
    assert ("concept:clause", "part_of", "concept:other") in edges, "renamed, not moved"
    assert "resolved 3 suspect label(s) (1 node(s) dropped)" in report.said()

    asked = len(client.calls)
    again = tidy(path, judge=ModelJudge(client, sources=lambda unit: texts.get(unit, "")))
    assert len(client.calls) == asked, "a label once resolved is never asked again"
    assert again.suspects_dropped == 0 and again.nothing_to_do


def test_a_suspect_renamed_to_a_name_the_store_already_holds_merges_into_it(tmp_path):
    path = _store(tmp_path, [
        _node("concept:clause", "that holds charge between pulses", mentions=2),
        _node("concept:glimmer", "glimmer node", mentions=6),
        _node("concept:vault", "vault current", mentions=3),
    ], [_edge("concept:clause", "part_of", "concept:vault", 2)])
    client = Scripted(suspects={("that holds charge between pulses",):
                                {"verdict": "rename", "name": "glimmer node"}})
    report = tidy(path, judge=ModelJudge(client))
    assert report.suspects_resolved == 1 and report.merged_nodes == 1
    assert report.merged_edges == 1
    nodes, edges = _ids(path)
    assert "concept:clause" not in nodes
    assert nodes["concept:glimmer"]["mentions"] == 8
    assert "that holds charge between pulses" in nodes["concept:glimmer"]["attrs"]["aliases"]
    assert ("concept:glimmer", "part_of", "concept:vault") in edges


# -- absorb: an incoming graph reconciled against the store on the way in

def test_absorb_lands_an_incoming_plural_and_a_close_spelling_on_what_the_store_holds(tmp_path):
    path = _store(tmp_path, [
        _node("concept:glimmer-node", "glimmer node", mentions=6,
              definition="a point of the lattice that holds charge"),
        _node("concept:vault-current", "vault current", mentions=4),
        _node("figure:1", "Figure 1.1", kind="figure"),
        {**_node("run:one", "an ingest run"), "kind": "run", "attrs": {"hidden": True}},
    ], [])
    incoming = {"nodes": [
        _node("in:glimmer-nodes", "glimmer nodes", mentions=3),
        _node("in:vault_current", "Vault_Current", mentions=2),
        _node("in:glimer-node", "glimer node", mentions=1,
              passage="The glimer node holds charge, as the glimmer node does."),
        _node("in:thrum-coil", "thrum coil", mentions=5),
        _node("in:figure", "Figure 1.1", kind="figure"),
    ], "edges": [
        _edge("in:glimer-node", "part_of", "in:thrum-coil", 2),
        _edge("in:glimmer-nodes", "part_of", "in:thrum-coil", 1),
    ], "counts": {"read": 1}}
    client = Scripted({("glimer node", "glimmer node"): "unsure"},
                      after_reading={("glimer node", "glimmer node"): "same"})
    report = absorb(path, incoming, judge=ModelJudge(client))
    assert report.mapped_plural == 1 and report.mapped_same_name == 1
    assert report.judged_same == 1 and report.left_possible == 0
    assert any("as the glimmer node does" in call for call in client.calls), \
        "the incoming node's own passage reached the judge"
    out = {node["id"]: node for node in report.graph["nodes"]}
    assert set(out) == {"concept:glimmer-node", "concept:vault-current", "in:thrum-coil",
                        "in:figure"}, "a figure is never folded"
    assert out["concept:glimmer-node"]["mentions"] == 10
    assert set(out["concept:glimmer-node"]["attrs"]["aliases"]) == {"glimmer nodes",
                                                                   "glimer node"}
    assert out["concept:glimmer-node"]["attrs"]["definition"].startswith("a point of the lattice")
    edges = {(e["source"], e["rel"], e["target"]): e for e in report.graph["edges"]}
    assert set(edges) == {("concept:glimmer-node", "part_of", "in:thrum-coil")}
    assert edges[("concept:glimmer-node", "part_of", "in:thrum-coil")]["weight"] == 3
    assert report.graph["counts"] == {"read": 1}, "what rode alongside comes back"
    assert "3 onto" not in report.absorbed() and "already held" in report.lines[0]
    nodes, _ = _ids(path)
    assert set(nodes) == {"concept:glimmer-node", "concept:vault-current", "figure:1", "run:one"}
    assert nodes["concept:glimmer-node"]["mentions"] == 6, "the store is not written"


def test_absorb_records_a_different_verdict_and_the_node_stays_new(tmp_path):
    path = _store(tmp_path, [_node("concept:sylvane", "sylvane", mentions=5)], [])
    incoming = {"nodes": [_node("in:sylvene", "sylvene", mentions=2)], "edges": []}
    texts = {"u:in:sylvene": "Sylvene carries a second ring and conducts nothing."}
    client = Scripted({("sylvane", "sylvene"): "unsure"},
                      after_reading={("sylvane", "sylvene"): "different"})
    report = absorb(path, incoming, judge=ModelJudge(client), sources=texts.get)
    assert report.judged_different == 1 and report.judged_same == 0
    assert [node["id"] for node in report.graph["nodes"]] == ["in:sylvene"]
    assert any("second ring" in call for call in client.calls), "the caller's sources were read"
    with GraphStore(path, read_only=True) as store:
        held = store.get_doc(DECISIONS)["pairs"]
        assert len(store.nodes()) == 1, "nothing but the decisions document is written"
    assert len(held) == 1
    verdict = next(iter(held.values()))
    assert verdict["verdict"] == "different" and verdict["absorbed"] is True

    asked = len(client.calls)
    again = absorb(path, incoming, judge=ModelJudge(client), sources=texts.get)
    assert len(client.calls) == asked and again.judged_different == 1


def test_absorb_without_a_judge_maps_the_plain_names_and_reports_the_spelling(tmp_path):
    path = _store(tmp_path, [_node("concept:flux-ring", "flux ring", mentions=4),
                             _node("concept:vault", "vault", mentions=3)], [])
    incoming = {"nodes": [_node("in:flux-rings", "Flux-Rings", mentions=2),
                          _node("in:vaulf", "vaulf", mentions=1)], "edges": []}
    report = absorb(path, incoming)
    assert report.mapped_plural == 1 and report.left_possible == 1
    assert report.possible == [("vaulf", "vault")]
    out = {node["id"] for node in report.graph["nodes"]}
    assert out == {"concept:flux-ring", "in:vaulf"}
    assert "1 close spelling(s) left new" in report.absorbed()
