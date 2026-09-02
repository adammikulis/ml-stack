"""Known names are swapped for stable placeholders."""

from ml_stack.redact import Redactor, tag


def test_the_same_name_always_gives_the_same_tag():
    assert tag("Ada Lovelace") == tag("  ada lovelace ")
    assert tag("Ada Lovelace") != tag("Bea Marlow")
    assert tag("Ada Lovelace").startswith("person#")


def test_names_are_replaced_wherever_they_appear():
    hide = Redactor({"Ada Lovelace", "Bea Marlow"})
    out = hide("Ada Lovelace wrote to bea marlow about Ada Lovelace")
    assert "Ada Lovelace" not in out and "marlow" not in out.casefold()
    assert out.count(tag("Ada Lovelace")) == 2
    assert tag("Bea Marlow") in out


def test_the_longest_name_wins_when_one_contains_another():
    hide = Redactor({"Pellard", "Pellard Foundry"})
    assert hide("Pellard Foundry ships") == f"{tag('Pellard Foundry')} ships"


def test_a_name_inside_a_word_is_left_alone():
    hide = Redactor({"Pellard"})
    assert hide("the Pellardesque style") == "the Pellardesque style"


def test_no_names_means_no_change():
    assert Redactor([])("Ada Lovelace said nothing") == "Ada Lovelace said nothing"


def test_anything_is_first_made_text():
    hide = Redactor({"Ada Lovelace"})
    assert hide({"who": "Ada Lovelace"}) == "{'who': '" + tag("Ada Lovelace") + "'}"


def test_the_prefix_names_what_kind_of_thing_was_hidden():
    assert tag("Quenlow Robotics", "org").startswith("org#")
    hide = Redactor({"Quenlow Robotics"}, prefix="org")
    assert hide("Quenlow Robotics hires") == f"{tag('Quenlow Robotics', 'org')} hires"


def _graph_and_log(tmp_path):
    import json

    graph = {"nodes": [{"id": "person:ada", "kind": "person", "label": "Ada Lovelace"},
                       {"id": "person:bo", "kind": "person", "label": "Bo"},
                       {"id": "topic:looms", "kind": "topic", "label": "looms"}],
             "messages": {"C1-1": {"sender": "Bea Marlow", "text": "hello"}}}
    (tmp_path / "graph.json").write_text(json.dumps(graph), encoding="utf-8")
    rows = [json.dumps({"ts": "1.000001", "sender": "Joan Clarke", "text": "hi"}),
            "not json", json.dumps(["a", "list"]), json.dumps({"ts": "1.000002", "text": "nobody"})]
    (tmp_path / "messages.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return tmp_path / "graph.json", tmp_path / "messages.jsonl"


def test_names_are_read_from_the_graphs_people_and_from_who_sent_each_message(tmp_path):
    from ml_stack.redact import names_in

    graph, log = _graph_and_log(tmp_path)
    assert names_in(graph, log) == {"Ada Lovelace", "Bea Marlow", "Joan Clarke"}


def test_a_two_letter_name_is_left_out_because_it_is_also_a_word(tmp_path):
    from ml_stack.redact import names_in

    graph, log = _graph_and_log(tmp_path)
    assert "Bo" not in names_in(graph, log)
    assert "Bo" in names_in(graph, log, min_length=2)


def test_a_file_that_is_missing_or_not_json_contributes_nothing(tmp_path):
    from ml_stack.redact import names_in

    graph, log = _graph_and_log(tmp_path)
    (tmp_path / "broken.json").write_text("{", encoding="utf-8")
    assert names_in(tmp_path / "absent.json", log) == {"Joan Clarke"}
    assert names_in(tmp_path / "broken.json", tmp_path / "absent.jsonl") == set()
    assert names_in(graph) == {"Ada Lovelace", "Bea Marlow"}
    assert names_in() == set()


def test_what_the_names_are_read_from_can_be_chosen(tmp_path):
    from ml_stack.redact import names_in

    graph, _ = _graph_and_log(tmp_path)
    assert names_in(graph, kind="topic", field="nobody") == {"looms"}


def test_allow_puts_a_phrase_on_the_list_once(tmp_path):
    """`hook allow PHRASE`: the allow-list grows by the command, not by hand; a phrase
    already there is not added twice, and nothing to allow is a refusal that says how."""
    import io

    from ml_stack.redact import hook

    fixtures = tmp_path / "tests" / "known-fixtures.txt"
    fixtures.parent.mkdir()
    fixtures.write_text("# invented\nAda Lovelace\n")
    out = io.StringIO()
    env = {"NAMES_FIXTURES": "tests/known-fixtures.txt"}
    assert hook.main(["allow", "Windows Defender Firewall", "x1 - x0"], env=env, root=tmp_path,
                     stdout=out) == 0
    text = fixtures.read_text()
    assert "Windows Defender Firewall\n" in text and "x1 - x0\n" in text
    assert "allowed with `hook allow` on" in text and text.startswith("# invented\nAda Lovelace\n")
    assert "allowed in" in out.getvalue()
    out = io.StringIO()
    assert hook.main(["allow", "windows defender firewall"], env=env, root=tmp_path, stdout=out) == 0
    assert "already allowed" in out.getvalue()
    assert fixtures.read_text().count("Windows Defender Firewall") == 1
    out = io.StringIO()
    assert hook.main(["allow"], env=env, root=tmp_path, stdout=out) == 2
    assert "allow what?" in out.getvalue()
