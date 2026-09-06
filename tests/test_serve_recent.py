"""The shared picker: what this machine used last, and what it offers to run on."""

import time

from ml_stack.serve import recent


class TestTheRecord:
    """Which models this machine's commands took, and when."""

    def test_a_model_taken_twice_keeps_one_row_the_newest(self, tmp_path):
        where = tmp_path / "recent.json"
        recent.note("/m/quince-2b.gguf", by="claude", path=where, now=100.0)
        recent.note("/m/brack-9b.gguf", by="do", path=where, now=200.0)
        rows = recent.note("/m/quince-2b.gguf", by="agent", path=where, now=300.0)
        assert [r["model"] for r in rows] == ["/m/quince-2b.gguf", "/m/brack-9b.gguf"]
        assert rows[0]["by"] == "agent" and rows[0]["at"] == 300.0

    def test_the_same_file_under_another_path_is_the_same_model(self, tmp_path):
        where = tmp_path / "recent.json"
        recent.note("/one/quince-2b.gguf", by="claude", path=where, now=100.0)
        rows = recent.note("/two/quince-2b.gguf", by="claude", path=where, now=200.0)
        assert len(rows) == 1

    def test_only_the_last_LIMIT_are_kept(self, tmp_path):
        where = tmp_path / "recent.json"
        for n in range(recent.LIMIT + 5):
            recent.note(f"/m/model-{n}.gguf", by="claude", path=where, now=float(n))
        assert len(recent.recent(where)) == recent.LIMIT

    def test_no_record_yet_is_an_empty_list_not_an_error(self, tmp_path):
        assert recent.recent(tmp_path / "nothing.json") == []

    def test_a_file_of_rubbish_reads_as_no_record(self, tmp_path):
        where = tmp_path / "recent.json"
        where.write_text("not json at all")
        assert recent.recent(where) == []

    def test_when_it_was_used_is_said_in_words(self):
        now = time.mktime((2026, 9, 6, 12, 0, 0, 0, 0, -1))
        today = time.mktime((2026, 9, 6, 8, 41, 0, 0, 0, -1))
        assert recent.used_said(today, now=now) == "used 08:41 today"
        assert recent.used_said(now - 86400, now=now) == "used yesterday"
        assert recent.used_said(now - 3 * 86400, now=now) == "used 3 days ago"


class TestOrderedByWhatWasUsedLast:
    """The listing puts the model a person reached for last at the top."""

    def _machine(self, monkeypatch, names):
        from ml_stack.serve import profile as profile_module

        monkeypatch.setattr(recent, "held", lambda: dict.fromkeys(names, 100))
        monkeypatch.setattr(recent, "weight_paths", lambda: [])
        monkeypatch.setattr(recent, "heads_for", lambda *a, **k: [])
        monkeypatch.setattr(recent, "_running", lambda _every: [])
        monkeypatch.setattr(profile_module, "profiles", lambda **_: [])
        monkeypatch.setattr(profile_module, "profile_for", lambda *a, **k: None)

    def test_the_most_recently_used_comes_first(self, monkeypatch, tmp_path):
        self._machine(monkeypatch, ["alpha-2b.gguf", "brack-9b.gguf", "quince-2b.gguf"])
        where = tmp_path / "recent.json"
        recent.note("quince-2b.gguf", by="claude", path=where, now=100.0)
        recent.note("brack-9b.gguf", by="do", path=where, now=200.0)
        listed = [c["name"] for c in recent.choices(path=where)]
        assert listed == ["brack-9b.gguf", "quince-2b.gguf", "alpha-2b.gguf"]

    def test_a_model_never_used_says_nothing_about_when(self, monkeypatch, tmp_path):
        self._machine(monkeypatch, ["alpha-2b.gguf"])
        one = recent.choices(path=tmp_path / "recent.json")[0]
        assert one["used_at"] is None and one["note"] == ""

    def test_the_listing_says_when_a_model_was_last_used(self, monkeypatch, tmp_path):
        self._machine(monkeypatch, ["alpha-2b.gguf"])
        where = tmp_path / "recent.json"
        recent.note("alpha-2b.gguf", by="claude", path=where)
        assert "used " in recent.choices(path=where)[0]["note"]


class TestPicking:
    """`ml-stack-claude` with nothing named."""

    def _options(self):
        return [{"kind": "server", "port": 8080, "url": "http://127.0.0.1:8080",
                 "name": "a served model", "note": "already running"},
                {"kind": "model", "name": "quince-2b.gguf", "record": None, "note": ""},
                {"kind": "model", "name": "brack-9b.gguf", "record": None, "note": ""}]

    def test_a_bare_number_takes_that_one(self):
        from ml_stack.serve.recent import pick

        assert pick(self._options(), say=lambda _: None, ask=lambda _: "2")["name"] == \
            "quince-2b.gguf"

    def test_nothing_typed_takes_the_first(self):
        from ml_stack.serve.recent import pick

        got = pick(self._options(), say=lambda _: None, ask=lambda _: "")
        assert got["kind"] == "server"

    def test_a_name_takes_the_one_it_names(self):
        from ml_stack.serve.recent import pick

        got = pick(self._options(), say=lambda _: None, ask=lambda _: "brack")
        assert got["name"] == "brack-9b.gguf"

    def test_a_name_matching_none_chooses_none(self):
        from ml_stack.serve.recent import pick

        said: list[str] = []
        assert pick(self._options(), say=said.append, ask=lambda _: "velthorne") is None
        assert any("not a choice" in line for line in said)

    def test_nothing_to_run_says_so(self):
        from ml_stack.serve.recent import pick

        said: list[str] = []
        assert pick([], say=said.append, ask=lambda _: "1") is None
        assert any("nothing to run" in line for line in said)

    def test_the_servers_are_offered_before_the_models(self):
        from ml_stack.serve.recent import pick

        said: list[str] = []
        pick(self._options(), say=said.append, ask=lambda _: "1")
        assert said[0] == "already running:"


class TestOfferingADraftHead:
    """A model with no measured record is offered the draft heads on this machine."""

    def _head(self, name="mtp-quince-2b-Q4_K_M.gguf", size=100, build=""):
        from ml_stack.hub import Head

        return Head(path=f"/models/{name}", bytes=size, spec_type="draft-mtp", build=build)

    def test_the_listing_says_what_each_model_would_draft_with(self):
        from ml_stack.serve.recent import head_line

        said = head_line({"kind": "model", "name": "quince-2b.gguf", "record": None,
                          "heads": [self._head(), self._head("mtp-quince-2b-Q8_0.gguf", 200)]})
        assert "mtp-quince-2b-Q4_K_M.gguf" in said and "100B of memory" in said
        assert "or 1 more" in said

    def test_the_listing_says_when_there_is_no_head(self):
        from ml_stack.serve.recent import head_line

        said = head_line({"kind": "model", "name": "brack-9b.gguf", "record": None,
                          "heads": []})
        assert said == "no draft head on this machine: it runs without speculative decoding"

    def test_a_measured_record_keeps_its_own_head(self):
        from ml_stack.serve.profile import record

        kept = record("quince-2b.gguf", draft="mtp-quince-2b-Q8_0.gguf", spec_type="draft-mtp")
        said = recent.head_line({"kind": "model", "name": "quince-2b.gguf", "record": kept,
                          "heads": [self._head()]})
        assert said == "drafts ahead with mtp-quince-2b-Q8_0.gguf, from its measured record"

    def test_a_record_with_no_head_names_the_one_lying_on_disk(self):
        from ml_stack.serve.profile import record

        said = recent.head_line({"kind": "model", "name": "quince-2b.gguf",
                          "record": record("quince-2b.gguf"), "heads": [self._head()]})
        assert "without a draft head" in said and "mtp-quince-2b-Q4_K_M.gguf" in said

    def test_nothing_typed_takes_the_cheapest_head(self):
        from ml_stack.serve.recent import pick_head

        heads = [self._head(), self._head("mtp-quince-2b-Q8_0.gguf", 200)]
        assert pick_head(heads, say=lambda _: None, ask=lambda _: "").bytes == 100

    def test_zero_serves_without_a_head(self):
        from ml_stack.serve.recent import pick_head

        assert pick_head([self._head()], say=lambda _: None, ask=lambda _: "0") is None

    def test_no_head_at_all_says_so_once(self):
        from ml_stack.serve.recent import pick_head

        said: list[str] = []
        assert pick_head([], say=said.append, ask=lambda _: "1") is None
        assert said == ["serving without a draft head: nothing guesses tokens ahead for "
                        "the model to check in one pass, which is slower"]

    def test_the_build_a_borrowing_head_needs_is_said_when_it_is_offered(self):
        from ml_stack.serve.recent import pick_head

        said: list[str] = []
        pick_head([self._head(build="mended")], say=said.append, ask=lambda _: "1")
        assert any("needs --build mended" in line for line in said)


