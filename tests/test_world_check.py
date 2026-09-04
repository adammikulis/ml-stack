"""`ml-stack-world check`: a corpus read back against the truth it was written from, and
every generated name through the name detector.

Every world here is made at the smallest size in ``tmp_path``; nothing reads a home
directory. The detector is the real one where it is installed and a fake where the test
needs a hit it can predict.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from ml_stack.files import write_json
from ml_stack.world import check
from ml_stack.world.cli import main, read_messages
from ml_stack.world.emit import mbox, slack_export, teams
from ml_stack.world.organisation import make
from ml_stack.world.simulate import run
from ml_stack.world.story import ARCS

DOMAIN = "example.test"


def _world(tmp_path: Path, kind: str = "company", seed: int = 0) -> tuple[Path, Path]:
    """A made world under ``world/`` and ten templated days of it under ``talk/``."""
    world = make(kind, "small", seed)
    where = tmp_path / "world"
    write_json(where / "graph.json", world.graph)
    write_json(where / "personas.json", world.personas)
    write_json(where / "calendar.json", world.calendar)
    write_json(where / "world.json", {"kind": world.kind, "size": world.size, "seed": world.seed,
                                      "people": world.people})
    talk = tmp_path / "talk"
    run(where, talk, days=10, mix=0.0, seed=seed)
    return where, talk


def _corpus(tmp_path: Path, talk: Path) -> list[Path]:
    """The talk emitted the way each product exports its own share."""
    graph = json.loads((talk / "graph.json").read_text(encoding="utf-8"))
    people = {n["id"]: {"label": n["label"]} for n in graph["nodes"] if n.get("kind") == "person"}
    messages = read_messages(talk / "messages.jsonl")
    return [slack_export(messages, people, tmp_path / "export", domain=DOMAIN),
            mbox(messages, people, tmp_path / "mail.mbox", domain=DOMAIN),
            teams(messages, people, tmp_path / "teams.json", domain=DOMAIN)]


@pytest.fixture
def empty_lists(tmp_path: Path) -> dict[str, Path]:
    fixtures = tmp_path / "lists" / "fixtures.txt"
    allow = tmp_path / "lists" / "allow.txt"
    fixtures.parent.mkdir()
    fixtures.write_text("# nothing\n", encoding="utf-8")
    allow.write_text("", encoding="utf-8")
    return {"fixtures": fixtures, "allow": allow}


# -- consistency ---------------------------------------------------------------------------------

def test_a_made_world_reads_back_consistent_with_its_truth(tmp_path):
    _, talk = _world(tmp_path)
    report = check.consistency(_corpus(tmp_path, talk), talk, domain=DOMAIN)
    assert report.misses == []
    assert report.counts["messages"] > 0
    assert report.counts["people"] == 50
    assert report.counts["spoken"] > 0
    assert report.counts["spoken"] == report.counts["spoken_found"]


def test_every_outcome_edge_is_named_at_both_ends_in_its_thread(tmp_path):
    from ml_stack.world.story import OUTCOMES

    _, talk = _world(tmp_path)
    graph = json.loads((talk / "graph.json").read_text(encoding="utf-8"))
    nodes = {n["id"]: n for n in graph["nodes"]}
    messages = {m.id: m for m in read_messages(talk / "messages.jsonl")}
    threads: dict[str, list[str]] = {}
    for m in messages.values():
        threads.setdefault(m.thread or m.id, []).append(m.text)

    def named(end: str, text: str) -> bool:
        label = nodes[end]["label"]
        if nodes[end].get("kind") == "person":
            return bool(check._names_first(label).search(text))
        return label in text

    outcomes = [e for e in graph["edges"] if e.get("rel") in OUTCOMES]
    assert outcomes
    for edge in outcomes:
        first = messages[edge["attrs"]["said_in"]]
        text = "\n".join(threads[first.thread or first.id])
        assert named(edge["source"], text) and named(edge["target"], text), edge


@pytest.mark.parametrize("kind", sorted(ARCS))
@pytest.mark.parametrize("seed", range(4))
def test_every_kind_of_world_reads_back_consistent_at_four_seeds(tmp_path, kind, seed):
    _, talk = _world(tmp_path, kind, seed)
    report = check.consistency(_corpus(tmp_path, talk), talk, domain=DOMAIN)
    assert report.misses == []
    assert report.counts["spoken"] == report.counts["spoken_found"] > 0
    assert report.counts["asserted"] == report.counts["asserted_found"] > 0


def test_a_message_doctored_to_name_a_stranger_fails_consistency(tmp_path):
    _, talk = _world(tmp_path)
    corpus = _corpus(tmp_path, talk)
    day = next(p for p in sorted((tmp_path / "export").rglob("*.json"))
               if p.parent != tmp_path / "export")
    rows = json.loads(day.read_text(encoding="utf-8"))
    rows[0]["text"] = "Loop in Ada Lovelace on this, she has the context."
    day.write_text(json.dumps(rows), encoding="utf-8")
    report = check.consistency(corpus, talk, domain=DOMAIN)
    assert [m for m in report.misses if "Ada Lovelace" in m and "not in the truth" in m]


def test_a_sender_the_truth_never_listed_fails_consistency(tmp_path):
    _, talk = _world(tmp_path)
    corpus = _corpus(tmp_path, talk)
    users = json.loads((tmp_path / "export" / "users.json").read_text(encoding="utf-8"))
    users[0]["real_name"] = users[0]["profile"]["real_name"] = "Joan Clarke"
    users[0]["profile"]["email"] = f"joan.clarke@{DOMAIN}"
    users[0]["name"] = users[0]["profile"]["display_name"] = "joan.clarke"
    users[0]["id"] = "U0STRANGER"
    (tmp_path / "export" / "users.json").write_text(json.dumps(users), encoding="utf-8")
    channels = json.loads((tmp_path / "export" / "channels.json").read_text(encoding="utf-8"))
    day = next(p for p in sorted((tmp_path / "export" / channels[0]["name"]).glob("*.json")))
    rows = json.loads(day.read_text(encoding="utf-8"))
    rows[0]["user"] = "U0STRANGER"
    day.write_text(json.dumps(rows), encoding="utf-8")
    report = check.consistency(corpus, talk, domain=DOMAIN)
    assert [m for m in report.misses if "U0STRANGER" in m or "Joan Clarke" in m]


def test_a_person_the_corpus_never_carries_is_a_miss(tmp_path):
    _, talk = _world(tmp_path)
    corpus = _corpus(tmp_path, talk)
    graph = json.loads((talk / "graph.json").read_text(encoding="utf-8"))
    graph["nodes"].append({"id": "person:silent", "kind": "person", "label": "Rosalind Franklin",
                           "mentions": 1, "attrs": {}, "messages": []})
    write_json(talk / "graph.json", graph)
    report = check.consistency(corpus, talk, domain=DOMAIN)
    assert [m for m in report.misses if "Rosalind Franklin" in m and "never" in m]


def test_an_outcome_whose_message_names_neither_end_is_a_miss(tmp_path):
    _, talk = _world(tmp_path)
    corpus = _corpus(tmp_path, talk)
    graph = json.loads((talk / "graph.json").read_text(encoding="utf-8"))
    people = [n["id"] for n in graph["nodes"] if n.get("kind") == "person"]
    graph["edges"].append({"source": people[0], "rel": "now_works_with", "target": people[1],
                           "weight": 1, "messages": ["msg:999999"],
                           "attrs": {"said_in": "msg:999999", "day": 0, "arc": "job_post"}})
    write_json(talk / "graph.json", graph)
    report = check.consistency(corpus, talk, domain=DOMAIN)
    assert [m for m in report.misses if "now_works_with" in m and "msg:999999" in m]


def test_the_truth_may_be_a_graph_file_or_a_world_directory(tmp_path):
    where, talk = _world(tmp_path)
    corpus = _corpus(tmp_path, talk)
    by_file = check.consistency(corpus, talk / "graph.json", domain=DOMAIN)
    assert by_file.misses == []
    # the world before it talked holds no outcomes, so nothing spoken is expected of it
    before = check.consistency(corpus, where, domain=DOMAIN)
    assert before.counts["spoken"] == 0 and before.misses == []


# -- privacy -------------------------------------------------------------------------------------

@dataclass
class _Hit:
    entity_type: str
    score: float
    start: int
    end: int


class _FakeDetector:
    """Reads one name as a person and nothing else."""

    def __init__(self, person: str) -> None:
        self.person = person

    def analyze(self, text: str, language: str) -> list[_Hit]:
        at = text.find(self.person)
        return [_Hit("PERSON", 0.9, at, at + len(self.person))] if at >= 0 else []


def test_a_made_worlds_names_pass_the_detector(tmp_path, empty_lists, monkeypatch):
    monkeypatch.delenv("NAMES_GRAPH", raising=False)
    monkeypatch.delenv("NAMES_SCRAPE", raising=False)
    _, talk = _world(tmp_path)
    report = check.privacy(talk, **empty_lists)
    assert report.hits == []
    assert report.counts["names"] > 50
    assert report.counts["detector"] in ("presidio", "not installed")


def test_a_truth_doctored_with_a_name_the_detector_knows_fails_privacy(tmp_path, empty_lists,
                                                                         monkeypatch):
    monkeypatch.delenv("NAMES_GRAPH", raising=False)
    monkeypatch.delenv("NAMES_SCRAPE", raising=False)
    monkeypatch.setattr(check, "recogniser", lambda: _FakeDetector("Grace Hopper"))
    _, talk = _world(tmp_path)
    graph = json.loads((talk / "graph.json").read_text(encoding="utf-8"))
    next(n for n in graph["nodes"] if n.get("kind") == "person")["label"] = "Grace Hopper"
    write_json(talk / "graph.json", graph)
    report = check.privacy(talk, **empty_lists)
    assert [h for h in report.hits if "Grace Hopper" in h and "person" in h]
    assert report.counts["detector"] == "presidio"


def test_a_name_on_the_fixtures_list_is_allowed(tmp_path, empty_lists, monkeypatch):
    monkeypatch.delenv("NAMES_GRAPH", raising=False)
    monkeypatch.delenv("NAMES_SCRAPE", raising=False)
    monkeypatch.setattr(check, "recogniser", lambda: _FakeDetector("Grace Hopper"))
    _, talk = _world(tmp_path)
    graph = json.loads((talk / "graph.json").read_text(encoding="utf-8"))
    next(n for n in graph["nodes"] if n.get("kind") == "person")["label"] = "Grace Hopper"
    write_json(talk / "graph.json", graph)
    empty_lists["fixtures"].write_text("Grace Hopper\n", encoding="utf-8")
    assert check.privacy(talk, **empty_lists).hits == []


def test_a_name_the_machines_own_list_holds_fails_privacy(tmp_path, empty_lists, monkeypatch):
    _, talk = _world(tmp_path)
    graph = json.loads((talk / "graph.json").read_text(encoding="utf-8"))
    someone = next(n["label"] for n in graph["nodes"] if n.get("kind") == "person")
    write_json(tmp_path / "names.json", {"nodes": [{"kind": "person", "label": someone}]})
    monkeypatch.setenv("NAMES_GRAPH", str(tmp_path / "names.json"))
    monkeypatch.delenv("NAMES_SCRAPE", raising=False)
    monkeypatch.setattr(check, "recogniser", lambda: None)
    report = check.privacy(talk, **empty_lists)
    assert [h for h in report.hits if someone in h and "list" in h]
    assert report.counts["detector"] == "not installed"


# -- the command ---------------------------------------------------------------------------------

def test_help_lists_check(capsys):
    with pytest.raises(SystemExit):
        main(["--help"])
    assert "check" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main(["check", "--help"])
    out = capsys.readouterr().out
    assert "--truth" in out and "--fixtures" in out


def test_the_command_exits_1_on_a_miss_and_0_on_none(tmp_path, empty_lists, monkeypatch, capsys):
    monkeypatch.delenv("NAMES_GRAPH", raising=False)
    monkeypatch.delenv("NAMES_SCRAPE", raising=False)
    monkeypatch.setattr(check, "recogniser", lambda: None)
    _, talk = _world(tmp_path)
    corpus = [str(p) for p in _corpus(tmp_path, talk)]
    lists = ["--fixtures", str(empty_lists["fixtures"]), "--allow", str(empty_lists["allow"])]
    assert main(["check", *corpus, "--truth", str(talk), "--domain", DOMAIN, *lists]) == 0
    out = capsys.readouterr().out
    assert "presidio not installed" in out
    assert "messages" in out and "people" in out
    day = next(p for p in sorted((tmp_path / "export").rglob("*.json"))
               if p.parent != tmp_path / "export")
    rows = json.loads(day.read_text(encoding="utf-8"))
    rows[0]["text"] = "Ask Alan Turing, he wrote it."
    day.write_text(json.dumps(rows), encoding="utf-8")
    assert main(["check", *corpus, "--truth", str(talk), "--domain", DOMAIN, *lists]) == 1
    assert "Alan Turing" in capsys.readouterr().out


def test_a_truth_without_a_graph_is_refused_by_name(tmp_path, capsys):
    assert main(["check", str(tmp_path), "--truth", str(tmp_path / "nowhere")]) == 2
    assert "graph.json" in capsys.readouterr().err
