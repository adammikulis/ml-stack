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


def test_a_name_on_the_list_is_allowed_with_the_sentences_punctuation_on_it(tmp_path):
    """'Tinsley Kilnworks.' at the end of a sentence is 'Tinsley Kilnworks' on the list."""
    import io
    import subprocess

    from ml_stack.redact import hook

    repo = tmp_path
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "t@example.invalid"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "t"], check=True)
    fixtures = repo / "tests" / "known-fixtures.txt"
    fixtures.parent.mkdir()
    fixtures.write_text("Tinsley Kilnworks\n")
    (repo / "note.py").write_text('# the kiln was moved to Tinsley Kilnworks. Then it cooled.\n')
    subprocess.run(["git", "-C", str(repo), "add", "note.py", "tests/known-fixtures.txt"], check=True)
    out = io.StringIO()
    assert hook.main([], env={"NAMES_FIXTURES": "tests/known-fixtures.txt", "HOME": str(tmp_path)},
                     root=repo, stdout=out) == 0, out.getvalue()


def _repo(tmp_path, fixtures: str = "", graph: dict | None = None):
    """A git repository with an allow-list and a graph of invented people, nothing staged."""
    import json
    import subprocess

    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    for setting, value in (("user.email", "nobody@example.invalid"), ("user.name", "nobody")):
        subprocess.run(["git", "-C", str(tmp_path), "config", setting, value], check=True)
    (tmp_path / "tests").mkdir(exist_ok=True)
    (tmp_path / "tests" / "known-fixtures.txt").write_text(fixtures, encoding="utf-8")
    (tmp_path / "graph.json").write_text(json.dumps(graph or {"nodes": [], "messages": {}}),
                                         encoding="utf-8")
    return tmp_path


def _check(tmp_path, why: bool = False, **files: str):
    """Stage those files and run the hook over them. Returns (exit code, what it said)."""
    import io
    import subprocess

    from ml_stack.redact import hook

    for name, body in files.items():
        (tmp_path / name).write_text(body, encoding="utf-8")
        subprocess.run(["git", "-C", str(tmp_path), "add", name], check=True)
    said = io.StringIO()
    env = {"NAMES_FIXTURES": "tests/known-fixtures.txt", "HOME": str(tmp_path),
           "NAMES_GRAPH": str(tmp_path / "graph.json")}
    code = hook.main(["--why"] if why else [], env=env, root=tmp_path, stdout=said)
    return code, said.getvalue()


def _needs_recogniser():
    """These fragments are found by Presidio and by nothing else, so without it there is
    nothing to stand down and the test would pass for the wrong reason."""
    import pytest

    from ml_stack.redact import hook

    if hook.recogniser() is None:
        pytest.skip("presidio is not installed")


def test_a_javascript_expression_is_not_a_person(tmp_path):
    """`x1 - x0` out of a chart's JavaScript: an operator with a space beside it is an
    operator, so the pair is an expression. Mutation: drop the `context_expression` clause
    and this file is refused again."""
    _needs_recogniser()
    repo = _repo(tmp_path)
    code, said = _check(repo, why=True, **{"fit.js": (
        "const w = x1 - x0;\nconst h = y1 - y0;\nconst s = w || null || h;\n")})
    assert code == 0, said
    assert "cleared by context_expression" in said


def test_a_number_in_a_markdown_table_cell_is_not_a_person(tmp_path):
    """`| ~3 |`: a markdown row and a cell with no letters in it. Mutation: drop the
    `context_table_cell` clause."""
    _needs_recogniser()
    repo = _repo(tmp_path)
    code, said = _check(repo, why=True, **{"costs.md": "| ~3 |\n"})
    assert code == 0, said
    assert "cleared by context_table_cell" in said


def test_a_keyword_and_an_identifier_is_not_a_person(tmp_path):
    """`assert label.lower` off a test line: a Python keyword and one identifier after it.
    Mutation: drop the `context_keyword` clause."""
    _needs_recogniser()
    repo = _repo(tmp_path)
    code, said = _check(repo, why=True, **{"t_kiln.py": 'assert label.lower() == "kiln"\n'})
    assert code == 0, said
    assert "cleared by context_keyword: assert" in said


def test_a_product_name_is_not_a_person(tmp_path):
    """`Windows Defender Firewall` in a table, `'Practical AI'` in a list of channels: a
    pair with a product, an OS or a vendor token in it. Mutation: drop the `context_product`
    clause."""
    repo = _repo(tmp_path)
    code, said = _check(repo, why=True, **{
        "notes.md": "| what | why |\n| --- | --- |\n| Windows Defender Firewall | blocks it |\n",
        "channels.py": "CHANNELS = ['Practical AI']\n"})
    assert code == 0, said
    assert "cleared by context_product: AI" in said


def test_an_invented_name_in_a_javascript_string_is_still_refused(tmp_path):
    """The whole point of the context rules is that they stand down code, never a person.
    A two-word name in a JS string, inside a call and beside operators, is still refused:
    what surrounds a quoted literal says nothing about what is inside it."""
    repo = _repo(tmp_path)
    code, said = _check(repo, **{"chart.js": (
        'const author = "Marla Quinn";\n'
        'const all = ["Marla Quinn"];\n'
        'label(w - 1, "Marla Quinn");\n')})
    assert code == 1, said
    assert "Marla Quinn" in said


def test_an_invented_name_in_a_table_cell_is_still_refused(tmp_path):
    """A table cell stands down when it has no letters in it. A cell with a name in it is
    a name: nobody is in the graph here, so only the recogniser can catch this one.
    Mutation: drop the `LETTER` check from `_table_cell`."""
    _needs_recogniser()
    repo = _repo(tmp_path)
    code, said = _check(repo, **{"who.md": "| who | kiln |\n| --- | --- |\n| Marla Quinn | 3 |\n"})
    assert code == 1, said
    assert "Marla Quinn" in said


def test_a_name_from_the_data_inside_an_expression_is_still_refused(tmp_path):
    """The exact list read from the graph is never stood down by a context rule: brackets
    and operators around a real name are exactly how one gets committed by accident."""
    repo = _repo(tmp_path, graph={"nodes": [{"kind": "person", "label": "Wren Halloway"}],
                                  "messages": {}})
    code, said = _check(repo, **{"pick.js": "const who = (Wren Halloway) || null;\n"})
    assert code == 1, said
    assert "Wren Halloway" in said and "someone in the data" in said


def test_allow_why_names_the_rule_that_almost_applied(tmp_path):
    """`hook allow --why PHRASE` says which rule already covers the phrase, or which came
    nearest -- so the next person adds a line to a rule, which covers every phrase of that
    shape, rather than a fixture, which covers one."""
    import io

    from ml_stack.redact import hook

    fixtures = tmp_path / "tests" / "known-fixtures.txt"
    fixtures.parent.mkdir()
    fixtures.write_text("", encoding="utf-8")
    env = {"NAMES_FIXTURES": "tests/known-fixtures.txt"}
    out = io.StringIO()
    assert hook.main(["allow", "--why", "Practical AI"], env=env, root=tmp_path,
                     stdout=out) == 0
    assert "already stood down by context_product: AI" in out.getvalue()

    out = io.StringIO()
    assert hook.main(["allow", "--why", "Loomcast Gateway"], env=env, root=tmp_path,
                     stdout=out) == 0
    told = out.getvalue()
    assert "no rule stood it down" in told and "nearest" in told
    assert "name-shapes.json covers every phrase of that shape" in told
    # --why is a flag, not a phrase to allow
    assert "--why" not in fixtures.read_text()
    assert "Loomcast Gateway" in fixtures.read_text()
