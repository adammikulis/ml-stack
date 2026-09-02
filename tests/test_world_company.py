"""An invented organised group is a valid graph at every size, for every kind.

Every name here is invented by a syllable table at the moment it is asked for; nothing reads
a real graph, a real store or a real server, and nothing is written outside ``tmp_path``.
"""

from __future__ import annotations

import json
import random
import re
import time

import pytest

from ml_stack.world import World
from ml_stack.world.names import company_name, person_name, product_name, slug
from ml_stack.world.organisation import (KINDS, SIZES, UNIT_KIND, load, make, role_catalogue,
                                         summary)
from ml_stack.world.questions import KINDS as KINDS_OF_QUESTION, questions


def _edges(world: World, rel: str) -> list[tuple[str, str]]:
    return [(e["source"], e["target"]) for e in world.graph["edges"] if e["rel"] == rel]


def _kinds(world: World) -> dict[str, str]:
    return {n["id"]: n["kind"] for n in world.graph["nodes"]}


# --- the same world twice ---------------------------------------------------------------------

def test_the_same_seed_makes_the_same_world_and_another_seed_does_not():
    once, twice = make("company", "small", 7), make("company", "small", 7)
    assert once == twice
    assert json.dumps(once.graph, sort_keys=True) == json.dumps(twice.graph, sort_keys=True)
    assert once.personas == twice.personas
    other = make("company", "small", 8)
    assert other.people != once.people


@pytest.mark.parametrize("size", ["small", "medium"])
def test_each_size_has_the_people_it_promises(size):
    world = make("company", size, 0)
    assert len(world.people) == SIZES[size]
    assert len([n for n in world.graph["nodes"] if n["kind"] == "person"]) == SIZES[size]
    assert world.size == size and world.calendar == []


def test_an_unknown_kind_or_size_is_refused_by_name():
    with pytest.raises(ValueError, match="kind"):
        make("guild", "small", 0)
    with pytest.raises(ValueError, match="size"):
        make("company", "huge", 0)


# --- what every kind must hold ----------------------------------------------------------------

@pytest.mark.parametrize("kind", KINDS)
def test_every_edge_joins_two_nodes_that_exist_and_no_edge_is_written_twice(kind):
    world = make(kind, "small", 0)
    ids = {n["id"] for n in world.graph["nodes"]}
    assert len(ids) == len(world.graph["nodes"]), "an id appears twice"
    for e in world.graph["edges"]:
        assert e["source"] in ids and e["target"] in ids, e
        assert e["source"] != e["target"], e
        assert {"source", "rel", "target", "weight", "messages"} <= set(e)
    keys = [(e["source"], e["rel"], e["target"]) for e in world.graph["edges"]]
    assert len(keys) == len(set(keys))


@pytest.mark.parametrize("kind", KINDS)
def test_every_person_has_a_unit_a_title_a_voice_a_system_prompt_and_something_said(kind):
    world = make(kind, "small", 0)
    kinds = _kinds(world)
    unit_of = {}
    for who, unit in _edges(world, "part_of"):
        if kinds[who] == "person":
            unit_of.setdefault(who, []).append(unit)
    nodes = {n["id"]: n for n in world.graph["nodes"]}
    for who in world.people:
        node = nodes[who]
        assert node["kind"] == "person" and node["attrs"]["member"] is True
        assert node["attrs"].get("title"), who
        assert node["attrs"].get("level"), who
        assert node["attrs"].get("started") and node["attrs"].get("tenure_years") is not None
        assert unit_of.get(who), f"{who} is part_of nothing"
        assert 1 <= len(node["messages"]) <= 3, "one to three things said"
        for mid in node["messages"]:
            assert world.graph["messages"][mid]["text"]
            assert world.graph["messages"][mid]["sender"] == node["label"]
        persona = world.personas[who]
        assert persona["voice"] and persona["system"]
        assert node["label"] in persona["system"]
        assert who in persona["knows"]
        assert node["mentions"] >= 1
    meta = world.graph["meta"]["world"]
    assert meta["kind"] == kind and meta["unit_kind"] == UNIT_KIND[kind]
    assert any(kinds[i] == UNIT_KIND[kind] for i in unit_of[world.people[-1]])


@pytest.mark.parametrize("kind", KINDS)
def test_everyone_knows_the_public_nodes_and_their_own_neighbours(kind):
    world = make(kind, "small", 0)
    kinds = _kinds(world)
    org = world.graph["meta"]["world"]["organisation"]
    public = {i for i, k in kinds.items() if k in (UNIT_KIND[kind], "place")} | {org}
    for who in world.people[:10]:
        knows = set(world.personas[who]["knows"])
        assert public <= knows
        for e in world.graph["edges"]:
            if e["source"] == who:
                assert e["target"] in knows
            if e["target"] == who:
                assert e["source"] in knows


# --- the structure each kind promises ---------------------------------------------------------

@pytest.mark.parametrize("size", ["small", "medium"])
def test_a_company_s_reporting_lines_are_a_forest_rooted_at_the_ceo_with_sane_spans(size):
    world = make("company", size, 1)
    boss = dict(_edges(world, "reports_to"))
    ceo = world.graph["meta"]["world"]["root"]
    assert ceo not in boss, "the CEO reports to nobody"
    assert set(boss) == set(world.people) - {ceo}, "everyone else reports to exactly one person"
    for who in world.people:
        seen, at = set(), who
        while at in boss:
            assert at not in seen, "a cycle in the org chart"
            seen.add(at)
            at = boss[at]
        assert at == ceo
    spans: dict[str, int] = {}
    for child, parent in boss.items():
        spans[parent] = spans.get(parent, 0) + 1
    assert max(spans.values()) <= 9
    managers = [n for n in spans if n != ceo]
    sane = [n for n in managers if 5 <= spans[n] <= 9]
    assert len(sane) >= 0.6 * len(managers), sorted(spans.values())
    levels = {n["id"]: n["attrs"].get("level") for n in world.graph["nodes"]}
    assert levels[ceo] == "c-level"
    assert all(levels[m] in ("manager", "director", "vp", "c-level") for m in spans), \
        "everyone with reports has a manager's level"
    assert any(levels[p] == "IC1" for p in world.people) and \
        any(levels[p] == "IC5" for p in world.people)


def test_a_community_has_no_reporting_lines_and_its_members_work_at_many_organisations():
    world = make("community", "small", 0)
    assert not _edges(world, "reports_to")
    employers = {o for _, o in _edges(world, "works_at")}
    assert len(employers) >= 4
    assert _edges(world, "moderates"), "groups have moderators"
    groups = {u for _, u in _edges(world, "part_of") if u.startswith("group:")}
    assert len(groups) >= 3
    for who in world.people:
        assert any(s == who and t.startswith("group:") for s, t in _edges(world, "part_of"))


def test_a_university_has_labs_led_by_faculty_who_advise_the_rest():
    world = make("university", "small", 0)
    leads = _edges(world, "leads")
    assert leads and all(t.startswith("lab:") for _, t in leads)
    advised = {t for _, t in _edges(world, "advises")}
    levels = {n["id"]: n["attrs"].get("level") for n in world.graph["nodes"]}
    assert advised and all(levels[t] in ("student", "postdoc", "staff") for t in advised)
    assert all(levels[s] == "faculty" for s, _ in leads)
    assert _edges(world, "chairs") and _edges(world, "funds")


def test_an_open_source_project_has_maintainers_contributors_and_releases():
    world = make("open-source", "small", 0)
    kinds = _kinds(world)
    assert _edges(world, "maintains") and _edges(world, "contributes_to")
    assert all(kinds[t] == "repo" for _, t in _edges(world, "maintains"))
    releases = [n for n in world.graph["nodes"] if n["kind"] == "event"]
    assert len(releases) >= 4 and all(n["attrs"].get("version") for n in releases)
    assert _edges(world, "sponsors")
    assert not _edges(world, "reports_to")


def test_a_nonprofit_has_a_board_that_advises_the_director_and_volunteers_who_report_to_nobody():
    world = make("nonprofit", "small", 0)
    director = world.graph["meta"]["world"]["root"]
    board = [s for s, _ in _edges(world, "sits_on")]
    assert 5 <= len(board) <= 8
    assert all((s, director) in _edges(world, "advises") for s in board)
    boss = dict(_edges(world, "reports_to"))
    levels = {n["id"]: n["attrs"].get("level") for n in world.graph["nodes"]}
    volunteers = [p for p in world.people if levels[p] == "volunteer"]
    assert volunteers and not any(v in boss for v in volunteers)
    assert all(levels[s] in ("director", "manager", "staff") for s in boss)
    assert _edges(world, "funds")


# --- through the store and the bench, unchanged ------------------------------------------------

def test_the_graph_goes_through_a_store_and_comes_back_whole(tmp_path):
    pytest.importorskip("ladybug")
    from ml_stack.graph.store import GraphStore, replace

    world = make("company", "small", 2)
    written = replace(tmp_path / "world.ladybug", world.graph)
    assert written == {"nodes": len(world.graph["nodes"]), "edges": len(world.graph["edges"])}
    with GraphStore(tmp_path / "world.ladybug", read_only=True) as store:
        held = {n["id"]: n for n in store.nodes()}
        assert set(held) == {n["id"] for n in world.graph["nodes"]}
        who = world.people[0]
        assert held[who]["attrs"]["title"] == \
            next(n for n in world.graph["nodes"] if n["id"] == who)["attrs"]["title"]
        assert held[who]["messages"], "what was said rides along in data"
        assert len(store.edges()) == len(world.graph["edges"])
        assert store.get_doc("meta")["world"]["kind"] == "company"


@pytest.mark.parametrize("kind", KINDS)
def test_the_graph_and_the_questions_are_what_the_bench_reads(tmp_path, kind):
    from ml_stack.graph.ask import list_kind, look_up, path_between, tools_for
    from ml_stack.graph.bench import SHORT, read_questions, sample, shape

    from ml_stack.world.cli import main

    out = tmp_path / "world"
    assert main(["make", "--kind", kind, "--size", "small", "--seed", "0",
                 "--out", str(out)]) == 0
    assert main(["questions", "--world", str(out), "--n", "40",
                 "--out", str(tmp_path / "q.jsonl")]) == 0
    graph = json.loads((out / "graph.json").read_text())      # exactly as `run --graph` does
    asked = read_questions(tmp_path / "q.jsonl")
    assert len(asked) == 40 and all(set(q) == {"q", "expect", "kind"} for q in asked)
    kinds = {n["id"]: n["kind"] for n in graph["nodes"]}
    assert all(e in kinds for q in asked for e in q["expect"])

    short = sample(asked, SHORT, graph=graph)
    assert len(short) == SHORT
    assert len(tools_for(graph)) == 6
    who = next(q for q in asked if q["q"].startswith("Tell me about "))
    name = who["q"][len("Tell me about "):-1]
    person = next(i for i in who["expect"] if kinds[i] == "person")
    assert person in {r["id"] for r in look_up(graph, name)}, "found by label"
    said = next(graph["messages"][m]["text"] for m in
                next(n for n in graph["nodes"] if n["id"] == person)["messages"])
    assert person in {r["id"] for r in look_up(graph, said.split(". ")[0][4:40])}, "and by a quote"
    listed = list_kind(graph, "person")
    assert listed["total"] == 50
    assert list_kind(graph, UNIT_KIND[kind])["total"] >= 3
    a, b = person, graph["meta"]["world"]["organisation"]
    assert path_between(graph, a, b)["path"] == [a, b] or len(path_between(graph, a, b)["path"]) >= 2


def test_shape_finds_no_expected_id_missing_from_the_graph(capsys):
    from ml_stack.graph.bench import shape

    world = make("community", "small", 3)
    shape(questions(world, 40), world.graph)
    said = capsys.readouterr().out
    assert "DO NOT EXIST" not in said
    assert "want a person" in said and "want no person" in said


# --- the questions --------------------------------------------------------------------------------

@pytest.mark.parametrize("kind", KINDS)
def test_questions_expect_ids_that_exist_and_cover_every_kind_of_answer(kind):
    world = make(kind, "small", 0)
    asked = questions(world, 40)
    kinds = _kinds(world)
    assert len(asked) == 40
    assert len({q["q"] for q in asked}) == 40, "no question twice"
    wanted = set()
    for q in asked:
        assert set(q) == {"q", "expect", "kind"}
        for e in q["expect"]:
            assert e in kinds, (q, e)
        wanted.update(kinds[e] for e in q["expect"])
    assert {"person", "org", "place", "topic", "opportunity", "event", UNIT_KIND[kind]} <= wanted
    nobody = [q for q in asked if not q["expect"]]
    assert 2 <= len(nobody) <= 6, "a few whose right answer is nobody"
    paths = [q for q in asked if q["q"].startswith("How is ")]
    assert paths and all(len(q["expect"]) >= 3 for q in paths)
    peopleless = [q for q in asked if q["expect"] and not any(kinds[e] == "person" for e in q["expect"])]
    assert len(peopleless) >= 0.3 * len(asked), "not nine-tenths person-shaped"


def test_questions_are_the_same_for_a_seed_and_a_given_rng_changes_them():
    world = make("company", "small", 5)
    assert questions(world, 20) == questions(world, 20)
    assert questions(world, 20) != questions(world, 20, random.Random(99))
    assert len(questions(world, 5)) == 5


# --- the four kinds the bench's own set was short of -------------------------------------------
#
# Counting, two hops, a false premise and a quote. Each is held by deriving its answer from
# the world's graph again, so a generated expectation is a fact about the graph and not
# about the generator.

_NEW = ("aggregate", "twohop", "trap", "quote")


def _by(world: World, rel: str) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for s, t in _edges(world, rel):
        out.setdefault(s, set()).add(t)
    return out


def _to(world: World, rel: str) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for s, t in _edges(world, rel):
        out.setdefault(t, set()).add(s)
    return out


@pytest.mark.parametrize("kind", KINDS)
def test_every_new_kind_is_generated_for_every_world_and_every_expected_id_exists(kind):
    world = make(kind, "small", 0)
    kinds = _kinds(world)
    asked = questions(world, 60)
    assert set(KINDS_OF_QUESTION) >= {q["kind"] for q in asked}
    for new in _NEW:
        some = [q for q in asked if q["kind"] == new]
        assert len(some) >= 2, f"{kind} asks no {new} question"
        for q in some:
            assert q["expect"], q
            assert all(e in kinds for e in q["expect"]), q


def test_a_count_is_scored_as_the_people_counted_and_a_tie_is_never_the_most():
    world = make("community", "small", 0)
    kinds = _kinds(world)
    in_unit = _to(world, "part_of")
    label = {n["id"]: n["label"] for n in world.graph["nodes"]}
    for q in questions(world, 200, kinds=["aggregate"]):
        if q["q"].startswith("How many people are in "):
            unit = next(u for u in in_unit if label[u] == q["q"][len("How many people are in "):-1])
            assert set(q["expect"]) == {p for p in in_unit[unit] if kinds[p] == "person"}
        elif q["q"].startswith("Which group has the most"):
            sizes = {u: len(in_unit[u]) for u in in_unit if kinds[u] == "group"}
            top = sorted(sizes.values(), reverse=True)
            assert top[0] > top[1] and sizes[q["expect"][0]] == top[0]
        elif q["q"].startswith("Which company employs"):
            at = _to(world, "works_at")
            sizes = {o: len(at[o]) for o in at}
            top = sorted(sizes.values(), reverse=True)
            assert top[0] > top[1] and sizes[q["expect"][0]] == top[0]
        else:
            assert q["q"].startswith("How many people")
            assert all(kinds[e] == "person" for e in q["expect"])


def test_two_hops_answer_with_the_far_end_and_never_the_person_in_the_middle():
    world = make("company", "small", 1)
    kinds = _kinds(world)
    label = {n["id"]: n["label"] for n in world.graph["nodes"]}
    knows = _to(world, "experienced_in")
    beside: dict[str, set[str]] = {}
    for x, y in _edges(world, "works_with"):
        beside.setdefault(x, set()).add(y)
        beside.setdefault(y, set()).add(x)
    seen = set()
    for q in questions(world, 200, kinds=["twohop"]):
        topic = next(t for t in knows if q["q"].endswith(f" {label[t]}?") or q["q"].endswith(f" {label[t]} in?")
                     or q["q"].endswith(f" {label[t]} live in?"))
        if q["q"].startswith("Who works with someone who knows about "):
            middle = knows[topic]
            assert set(q["expect"]) == {c for p in middle for c in beside.get(p, ())} - middle
            seen.add("with")
        elif q["q"].startswith("Who works with someone in "):
            unit_label = q["q"][len("Who works with someone in "):q["q"].index(" who knows about ")]
            unit = next(i for i, l in label.items() if l == unit_label and kinds[i] == "department")
            middle = knows[topic] & _to(world, "part_of")[unit]
            assert 1 <= len(middle) <= 3
            assert set(q["expect"]) == {c for p in middle for c in beside.get(p, ())} - middle
            seen.add("with-in")
        elif "live in?" in q["q"]:
            lives = _by(world, "based_in")
            assert set(q["expect"]) == {c for p in knows[topic] for c in lives.get(p, ())}
            seen.add("places")
        else:
            unit_of = _by(world, "part_of")
            assert set(q["expect"]) == {u for p in knows[topic] for u in unit_of.get(p, ())
                                        if kinds[u] == "department"}
            seen.add("units")
    assert {"places", "units"} <= seen and seen & {"with", "with-in"}


def test_a_false_premise_leaves_the_place_exactly_as_the_graph_has_it():
    world = make("open-source", "small", 0)
    kinds = _kinds(world)
    label = {n["id"]: n["label"] for n in world.graph["nodes"]}
    lives = _by(world, "based_in")
    settled = _to(world, "based_in")
    asked = questions(world, 200, kinds=["trap"])
    assert asked
    for q in asked:
        m = re.fullmatch(r"Since (.+) moved to (.+), who (?:in (.+) )?is left in (.+)\?", q["q"])
        assert m, q["q"]
        mover = next(i for i, l in label.items() if l == m.group(1) and kinds[i] == "person")
        to = next(i for i, l in label.items() if l == m.group(2) and kinds[i] == "place")
        place = next(i for i, l in label.items() if l == m.group(4) and kinds[i] == "place")
        assert to not in lives[mover] and place not in lives[mover], "the premise is false"
        here = {p for p in settled[place] if kinds[p] == "person"}
        if m.group(3):
            unit = next(i for i, l in label.items() if l == m.group(3))
            here &= _to(world, "part_of")[unit]
        assert set(q["expect"]) == here, "nobody subtracted"


def test_a_quote_question_is_answered_by_the_words_and_by_nothing_else():
    world = make("university", "small", 2)
    g = world.graph
    label = {n["id"]: n["label"] for n in g["nodes"]}
    said = {n["id"]: [g["messages"][m]["text"] for m in n["messages"]]
            for n in g["nodes"] if n["kind"] == "person"}
    asked = questions(world, 200, kinds=["quote"])
    assert asked and {q["kind"] for q in asked} == {"quote"}
    for q in asked:
        if q["q"].startswith("Who said they "):
            phrase = q["q"][len("Who said they "):-1]
            assert set(q["expect"]) == {who for who, lines in said.items()
                                        if any(f"I {phrase}." in l for l in lines)}
        else:
            who = next(i for i, l in label.items() if q["q"] == f"What did {l} say takes most of their time?")
            assert len(q["expect"]) == 1
            assert any(f"Lately most of my time goes to {label[q['expect'][0]]}." in l for l in said[who])


def test_kinds_draws_only_those_buckets_and_refuses_an_unknown_one(tmp_path, capsys):
    from ml_stack.world.cli import main

    world = make("company", "small", 0)
    assert {q["kind"] for q in questions(world, 30, kinds=["trap", "quote"])} == {"trap", "quote"}
    with pytest.raises(ValueError, match="unknown question kind"):
        questions(world, 5, kinds=["riddle"])

    out = tmp_path / "world"
    assert main(["make", "--kind", "company", "--size", "small", "--seed", "0", "--out", str(out)]) == 0
    capsys.readouterr()
    assert main(["questions", "--world", str(out), "--n", "8", "--kinds", "aggregate,twohop"]) == 0
    lines = [json.loads(l) for l in capsys.readouterr().out.strip().splitlines()]
    assert len(lines) == 8 and {l["kind"] for l in lines} == {"aggregate", "twohop"}
    assert main(["questions", "--world", str(out), "--n", "8", "--kinds", "riddle"]) == 2
    assert "riddle" in capsys.readouterr().err


# --- names -------------------------------------------------------------------------------------------

def test_no_two_people_share_a_full_name_in_a_small_world():
    world = make("company", "small", 0)
    labels = [n["label"] for n in world.graph["nodes"] if n["kind"] == "person"]
    assert len(labels) == len(set(labels))
    assert len({n["id"] for n in world.graph["nodes"]}) == len(world.graph["nodes"])


def test_the_name_generator_over_twenty_thousand_draws_repeats_itself_less_than_one_in_a_hundred():
    rng = random.Random(0)
    drawn = [person_name(rng) for _ in range(20_000)]
    assert len(drawn) - len(set(drawn)) < 200
    for given, family in drawn[:500]:
        assert given[0].isupper() and family[0].isupper()
        assert 3 <= len(given) <= 8 and 4 <= len(family) <= 10
        assert given.isalpha() and family.isalpha()


def test_organisations_and_products_are_named_from_stems_and_slugs_are_ids():
    rng = random.Random(0)
    names = {company_name(rng) for _ in range(200)}
    assert len(names) > 150
    assert company_name(random.Random(1), kind="") .count(" ") == 0
    assert company_name(random.Random(1), kind="Trust").endswith(" Trust")
    assert all(product_name(rng) for _ in range(50))
    assert slug("Kraków, Poland!") == "krak-w-poland"
    assert slug("São Paulo") == "s-o-paulo"
    assert slug("Site Reliability") == "site-reliability"


# --- the command ------------------------------------------------------------------------------------

def test_main_make_writes_the_three_files_and_says_what_it_made(tmp_path, capsys):
    from ml_stack.world.cli import main

    out = tmp_path / "w"
    assert main(["make", "--kind", "nonprofit", "--size", "small", "--seed", "4",
                 "--out", str(out), "--json"]) == 0
    said = json.loads(capsys.readouterr().out)
    assert said["kind"] == "nonprofit" and said["people"] == 50
    assert said["edges_by_relation"]["sits_on"] >= 5
    for name in ("graph.json", "personas.json", "calendar.json"):
        assert (out / name).exists(), name
    assert json.loads((out / "calendar.json").read_text()) == []
    back = load(out)
    assert back.people == make("nonprofit", "small", 4).people
    assert back.seed == 4 and back.size == "small"
    assert summary(back)["people"] == 50

    assert main(["questions", "--world", str(out), "--n", "12"]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 12 and all(set(json.loads(l)) == {"q", "expect", "kind"} for l in lines)


def test_main_make_writes_world_json_and_load_reads_kind_size_and_people_from_it(tmp_path):
    from ml_stack.world.cli import main

    out = tmp_path / "w"
    assert main(["make", "--kind", "university", "--seed", "2", "--out", str(out)]) == 0
    about = json.loads((out / "world.json").read_text())
    assert about["kind"] == "university" and about["size"] == "small" and about["seed"] == 2
    assert about["people"] == make("university", "small", 2).people
    assert about["organisation"].startswith("org:")
    back = load(out)
    assert back.kind == "university" and back.people == about["people"]
    (out / "world.json").unlink()
    assert load(out).kind == "university", "the graph's own meta says it too"


def test_main_simulate_then_emit_writes_a_corpus_the_sources_read_back(tmp_path, capsys):
    """The whole path with no model: make, talk for a few days, export as Slack, read it."""
    from ml_stack.sources import read
    from ml_stack.world.cli import main, read_messages

    world_dir, talk, export = tmp_path / "w", tmp_path / "talk", tmp_path / "export"
    assert main(["make", "--kind", "community", "--seed", "0", "--out", str(world_dir)]) == 0
    assert main(["simulate", "--world", str(world_dir), "--out", str(talk), "--days", "3",
                 "--seed", "1"]) == 0
    counts = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert counts["messages"] > 0 and counts["model_threads"] == 0
    messages = read_messages(talk / "messages.jsonl")
    assert len(messages) == counts["messages"]
    assert all(m.sender.startswith("person:") for m in messages)

    assert main(["emit", "--from", str(talk), "--as", "slack-export", "--out", str(export)]) == 0
    assert (export / "users.json").exists() and (export / "channels.json").exists()
    slack = [m for m in messages if m.source == "slack"]
    back = read(export)
    assert len(back) == len(slack)
    assert {m.text for m in back} == {m.text for m in slack}

    assert main(["emit", "--from", str(talk / "messages.jsonl"), "--world", str(world_dir),
                 "--as", "rows", "--out", str(tmp_path / "rows.jsonl")]) == 0
    rows = [json.loads(l) for l in (tmp_path / "rows.jsonl").read_text().splitlines()]
    assert rows and all("channelId" in r and "ts" in r for r in rows)
    assert main(["emit", "--from", str(talk), "--as", "mbox", "--out", str(tmp_path / "mail.mbox")]) == 0
    assert (tmp_path / "mail.mbox").stat().st_size > 0


def test_main_answers_help():
    from ml_stack.world.cli import main

    with pytest.raises(SystemExit) as left:
        main(["--help"])
    assert left.value.code == 0


# --- breadth and speed ------------------------------------------------------------------------------

def test_the_role_catalogue_is_broad_enough_for_an_org_chart_to_vary():
    catalogue = role_catalogue()
    assert len(catalogue) >= 13
    titles = {t for held in catalogue.values() for t in held}
    assert len(titles) >= 200
    assert "Principal Software Engineer" in titles and "Director of Finance" in titles
    world = make("company", "medium", 0)
    used = {n["attrs"]["title"] for n in world.graph["nodes"] if n["kind"] == "person"}
    assert len(used) >= 60


@pytest.mark.parametrize("kind", KINDS)
def test_a_large_world_is_made_in_under_ten_seconds(kind):
    began = time.time()
    world = make(kind, "large", 0)
    took = time.time() - began
    assert took < 10, f"{kind} large took {took:.1f}s"
    assert len(world.people) == SIZES["large"]
    assert len(world.graph["edges"]) > 5 * SIZES["large"]
