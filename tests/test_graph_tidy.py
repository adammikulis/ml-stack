"""The hygiene pass: duplicates merged with everything kept, inverses folded, the doubtful
flagged, the rest reported -- dry by default, idempotent, never a hidden node."""

from ml_stack.graph.store import GraphStore
from ml_stack.graph.tidy import Report, canonical_direction, suspect, tidy


def _node(id_, label, kind="concept", mentions=1, **attrs):
    return {"id": id_, "kind": kind, "label": label, "mentions": mentions,
            "attrs": {"aliases": [], **attrs}, "provenance": [f"u:{id_}"]}


def _edge(a, rel, b, weight=1):
    return {"source": a, "rel": rel, "target": b, "weight": weight, "provenance": [f"u:{a}"]}


def _store(tmp_path, nodes, edges):
    path = tmp_path / "g.ladybug"
    with GraphStore(path) as store:
        store.write({"nodes": nodes, "edges": edges})
    return path


def _ids(path):
    with GraphStore(path, read_only=True) as store:
        return ({n["id"]: n for n in store.nodes()},
                {(e["source"], e["rel"], e["target"]): e for e in store.edges()})


def test_a_plural_and_a_spelling_merge_into_the_heavier_name_with_everything_kept(tmp_path):
    path = _store(tmp_path, [
        _node("concept:acid", "acid", mentions=5, definition="a proton donor"),
        _node("concept:acids", "acids", mentions=2),
        _node("concept:base", "base", mentions=3),
        _node("concept:glimer-node", "glimer node", mentions=1),
        _node("concept:glimmer-node", "glimmer node", mentions=4),
    ], [
        _edge("concept:acid", "contrasts_with", "concept:base", 2),
        _edge("concept:acids", "contrasts_with", "concept:base", 1),
        _edge("concept:acids", "produces", "concept:glimer-node", 1),
    ])
    dry = tidy(path)
    assert dry.dry_run and dry.merged_nodes == 1, "the plural; a spelling apart is a person's call"
    assert [tuple(sorted(p)) for p in dry.possible] == [("glimer node", "glimmer node")]
    assert _ids(path)[0].keys() >= {"concept:acids", "concept:glimer-node"}, "dry: nothing written"

    done = tidy(path, dry_run=False)
    nodes, edges = _ids(path)
    assert done.merged_nodes == 1 and "concept:acids" not in nodes
    assert "concept:glimer-node" in nodes, "reported, not merged"
    acid = nodes["concept:acid"]
    assert acid["mentions"] == 7 and "acids" in acid["attrs"]["aliases"]
    assert set(acid["provenance"]) == {"u:concept:acid", "u:concept:acids"}
    joined = edges[("concept:acid", "contrasts_with", "concept:base")]
    assert joined["weight"] == 3, "the survivor's edge took the sum"
    assert ("concept:acid", "produces", "concept:glimer-node") in edges, "moved with the merge"
    again = tidy(path, dry_run=False)
    assert again.nothing_to_do and [tuple(sorted(p)) for p in again.possible] == [("glimer node", "glimmer node")], \
        "idempotent; the possible pair is reported every time until a person settles it"

    settled = tidy(path, dry_run=False, written={"glimer node": "glimmer node"})
    assert settled.merged_nodes == 1 and settled.possible == []
    nodes, edges = _ids(path)
    assert "concept:glimer-node" not in nodes
    assert ("concept:acid", "produces", "concept:glimmer-node") in edges


def test_case_spacing_and_hyphens_are_one_name_and_a_letter_apart_is_not(tmp_path):
    """A biology sources's dry run would have folded 'Natrium' into 'atrium' and 'Isobutene'
    into 'isobutane' by spelling; 'T-cell', 't cell' and 'T_cell' are one name."""
    path = _store(tmp_path, [_node("concept:t-cell", "T-cell", mentions=5),
                             _node("concept:t-cell-2", "t cell", mentions=2),
                             _node("concept:t_cell", "T_cell", mentions=1),
                             _node("concept:natrium", "Natrium", mentions=1),
                             _node("concept:atrium", "atrium", mentions=9)], [])
    report = tidy(path, dry_run=False)
    assert report.merged_nodes == 2
    nodes, _ = _ids(path)
    assert set(nodes) == {"concept:t-cell", "concept:natrium", "concept:atrium"}
    assert set(nodes["concept:t-cell"]["attrs"]["aliases"]) == {"t cell", "T_cell"}
    assert ("Natrium", "atrium") in report.possible or ("atrium", "Natrium") in report.possible


def test_figures_sources_and_runs_are_never_folded(tmp_path):
    path = _store(tmp_path, [
        _node("figure:u1:1", "Figure 1.1", kind="figure"),
        _node("figure:u2:1", "Figure 10.4", kind="figure"),
        _node("source:a", "Lattice Studies", kind="source"),
        _node("source:b", "Lattice Studies", kind="source"),
    ], [])
    report = tidy(path, dry_run=False)
    assert report.merged_nodes == 0 and report.possible == []
    assert len(_ids(path)[0]) == 4


def test_an_inverse_pair_folds_to_the_canonical_direction(tmp_path):
    path = _store(tmp_path, [_node("concept:cell", "cell"), _node("concept:nucleus", "nucleus"),
                             _node("concept:vault", "vault"), _node("concept:current", "current")],
                  [_edge("concept:nucleus", "part_of", "concept:cell", 2),
                   _edge("concept:cell", "has_part", "concept:nucleus", 1),
                   _edge("concept:vault", "has_part", "concept:current", 1)])
    report = tidy(path, dry_run=False)
    assert report.inverses_folded == 2
    _nodes, edges = _ids(path)
    assert edges[("concept:nucleus", "part_of", "concept:cell")]["weight"] == 3
    assert ("concept:cell", "has_part", "concept:nucleus") not in edges
    assert ("concept:current", "part_of", "concept:vault") in edges, "rewritten, no partner"
    assert canonical_direction("part_of") == ("part_of", False)


def test_the_pass_recounts_the_stats_document_after_its_writes(tmp_path):
    path = tmp_path / "g.ladybug"
    with GraphStore(path) as store:
        store.write({"nodes": [_node("concept:cell", "cell"), _node("concept:nucleus", "nucleus")],
                     "edges": [_edge("concept:nucleus", "part_of", "concept:cell", 2),
                               _edge("concept:cell", "has_part", "concept:nucleus", 1)],
                     "stats": {"nodes": 2, "edges": 2, "built_at": "2026-08-31T10:00:00"}})
    dry = tidy(path)
    with GraphStore(path, read_only=True) as store:
        assert dry.inverses_folded == 1 and store.get_doc("stats")["edges"] == 2
    report = tidy(path, dry_run=False)
    assert report.inverses_folded == 1
    with GraphStore(path, read_only=True) as store:
        counted = store.counts()
        assert counted["edges"] == 1
        assert store.get_doc("stats") == {"nodes": 2, "edges": 1,
                                          "built_at": "2026-08-31T10:00:00"}
        assert store.get_doc("stats")["edges"] == counted["edges"]


def test_suspect_labels_are_flagged_not_removed_and_hidden_nodes_are_left_alone(tmp_path):
    path = _store(tmp_path, [
        _node("concept:clause", "that aims to observe, explore, and investigate"),
        _node("concept:generic", "form of science"),
        _node("concept:number", "42"),
        _node("concept:fine", "glimmer node"),
        {**_node("run:one", "ingest that aims to read"), "kind": "run",
         "attrs": {"hidden": True, "model": "kestrel"}},
    ], [])
    report = tidy(path, dry_run=False)
    assert report.flagged == 3
    nodes, _ = _ids(path)
    assert nodes["concept:clause"]["attrs"]["suspect"].startswith("a clause")
    assert nodes["concept:generic"]["attrs"]["suspect"] == "too generic to be one thing"
    assert "suspect" not in nodes["concept:fine"]["attrs"]
    assert "suspect" not in nodes["run:one"]["attrs"], "hidden: never touched"
    assert nodes["run:one"]["attrs"]["hidden"] is True
    assert tidy(path, dry_run=False).flagged == 0, "flagged once"
    assert suspect("mitochondria") == "" and suspect("x") == "a single letter"


def test_conflicts_orphans_and_self_loops_are_reported_and_left(tmp_path):
    path = _store(tmp_path, [_node("concept:acid", "acid"), _node("concept:ph", "ph"),
                             _node("concept:alone", "alone"), _node("concept:loop", "loop")],
                  [_edge("concept:acid", "causes", "concept:ph"),
                   _edge("concept:acid", "regulates", "concept:ph"),
                   _edge("concept:loop", "requires", "concept:loop")])
    report = tidy(path, dry_run=False)
    assert report.conflicts == [("concept:acid", "concept:ph", "causes", "regulates")]
    assert report.orphans == ["concept:alone"]
    assert report.self_loops == [("concept:loop", "requires")]
    _nodes, edges = _ids(path)
    assert len(edges) == 3, "reported, not removed"
    assert isinstance(report, Report) and "1 verb conflict(s)" in report.said()


def test_the_store_and_ingest_commands_run_the_pass_dry_unless_told_to_apply(tmp_path, capsys):
    from ml_stack import ingest
    from ml_stack.graph import store_cli

    path = _store(tmp_path, [_node("concept:acid", "acid", mentions=3),
                             _node("concept:acids", "acids", mentions=1)], [])
    assert store_cli.main(["tidy", str(path)]) == 0
    assert "would merge 1 node(s)" in capsys.readouterr().out
    assert set(_ids(path)[0]) == {"concept:acid", "concept:acids"}, "dry"
    assert ingest.main(["tidy", "--out", str(path), "--apply"]) == 0
    assert "merged 1 node(s)" in capsys.readouterr().out
    assert set(_ids(path)[0]) == {"concept:acid"}


def test_a_written_file_settles_possible_duplicates_from_the_command_line(tmp_path, capsys):
    import json

    from ml_stack.graph import store_cli

    path = _store(tmp_path, [_node("concept:glimmer-node", "glimmer node", mentions=4),
                             _node("concept:glimer-node", "glimer node", mentions=1)], [])
    assert store_cli.main(["tidy", str(path), "--apply"]) == 0
    assert "1 possible duplicate" in capsys.readouterr().out
    assert set(_ids(path)[0]) == {"concept:glimmer-node", "concept:glimer-node"}
    decided = tmp_path / "written.json"
    decided.write_text(json.dumps({"glimer node": "glimmer node"}))
    assert store_cli.main(["tidy", str(path), "--apply", "--written", str(decided)]) == 0
    assert set(_ids(path)[0]) == {"concept:glimmer-node"}


def test_a_merge_keeps_both_definitions_rather_than_the_one_it_found_first(tmp_path):
    path = _store(tmp_path, [
        _node("concept:vault", "vault", mentions=5, definition="that holds plates"),
        _node("concept:vaults", "vaults", mentions=2,
              definition="a housing that holds a stack of lattice plates"),
        _node("concept:ring", "ring", mentions=4, definition="a closed loop"),
        _node("concept:rings", "rings", mentions=1, definition="a closed loop of lattice"),
    ], [])
    report = tidy(path, dry_run=False)
    assert report.merged_nodes == 2 and report.definitions_judged == 0
    nodes, _edges = _ids(path)
    vault = nodes["concept:vault"]["attrs"]
    assert vault["definition"] == "a housing that holds a stack of lattice plates", \
        "the longer one that does not read as half a sentence"
    assert vault["definitions_also"] == ["that holds plates"], "nothing is lost"
    ring = nodes["concept:ring"]["attrs"]
    assert ring["definition"] == "a closed loop of lattice"
    assert not ring.get("definitions_also"), "one is the start of the other: not a second"


def test_the_pass_checks_the_store_after_its_writes_and_refuses_success_over_an_unsound_one(tmp_path, monkeypatch, capsys):
    """2026-09-03: a store engine blanked other nodes' strings on a delete and the pass
    reported success over a store that no longer read back by id."""
    from ml_stack.graph import store_cli
    from ml_stack.graph.store import GraphStore

    path = _store(tmp_path, [_node("concept:acid", "acid", mentions=3),
                             _node("concept:acids", "acids", mentions=1)], [])
    report = tidy(path, dry_run=False)
    assert report.sound and "NOT SOUND" not in report.said()

    monkeypatch.setattr(GraphStore, "check", lambda self: ["node concept:x: found by scan, not by id"])
    again = _store(tmp_path / "b", [_node("concept:vault", "vault", mentions=3),
                                    _node("concept:vaults", "vaults", mentions=1)], [])
    report = tidy(again, dry_run=False)
    assert not report.sound and report.said().startswith("NOT SOUND")
    assert store_cli.main(["tidy", str(again), "--apply"]) == 1
    assert "NOT SOUND" in capsys.readouterr().out
    assert tidy(again).sound, "a dry run writes nothing and checks nothing"


def test_the_pass_reports_the_soundness_a_fresh_reader_finds(tmp_path):
    from ml_stack.graph.tidy import _recheck

    path = _store(tmp_path, [
        _node("concept:acid", "acid", mentions=5),
        _node("concept:acids", "acids", mentions=2),
        _node("concept:base", "base", mentions=3),
    ], [_edge("concept:acids", "contrasts_with", "concept:base")])
    report = tidy(path, dry_run=False)
    assert report.sound and report.merged_nodes == 1
    with GraphStore(path, read_only=True) as store:
        assert store.check() == []

    report.findings = ["edges: count says 25612, scan returned 25513"]
    report.lines.append(report.said())
    assert "NOT SOUND" in report.lines[-1]

    _recheck(path, report, None)

    assert report.findings == [], "what the store on disk says, not what the writer's handle did"
    assert "NOT SOUND" not in report.lines[-1]
