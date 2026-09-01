"""Reading rows out of a page, including the ones the page threw away.

The page here is a fake that behaves the way the real ones do: it virtualises, so a row far
enough from the viewport is not in the document at all. That is the whole reason the module
exists, and a fake that hands back everything at once would test nothing.
"""

import json
from datetime import datetime

import pytest

from ml_stack.scrape.browser import within_hours
from ml_stack.scrape.presets import DISCORD, SLACK, WEBSITE, Site, preset
from ml_stack.scrape.read import read_all, read_once, scroll
from ml_stack.scrape.seen import Seen


class VirtualList:
    """A list that only keeps a screenful in the document, like the real ones."""

    def __init__(self, rows, screen=3):
        self.rows = rows              # oldest first
        self.screen = screen
        self.top = len(rows) - screen  # showing the newest screenful
        self.reads = 0

    def evaluate(self, js, arg=None):
        if "scrollTop" in js:         # a scroll
            was = self.top
            self.top = max(0, self.top - self.screen)
            return self.top != was
        self.reads += 1
        return [dict(r) for r in self.rows[self.top:self.top + self.screen]]


def rows(n):
    return [{"key": f"17879371{i:02d}.000000", "author": f"person {i}", "text": f"row {i}"}
            for i in range(n)]


def test_reading_once_sees_only_what_is_in_the_document():
    page = VirtualList(rows(9))
    seen = read_once(page, SLACK)
    assert [r["text"] for r in seen] == ["row 6", "row 7", "row 8"]


def test_reading_it_all_walks_back_and_keeps_what_it_passed():
    """Scrolling to the top and reading once would return three rows out of nine."""
    page = VirtualList(rows(9))
    seen = read_all(page, SLACK)
    assert [r["text"] for r in seen] == [f"row {i}" for i in range(9)]
    assert page.reads > 1, "it read once and got lucky, which is not the claim"


def test_the_walk_stops_when_nothing_new_turns_up():
    page = VirtualList(rows(4), screen=4)      # everything is on screen already
    read_all(page, SLACK, quiet_rounds=2)
    assert page.reads <= 3


def test_a_key_is_pulled_out_of_whatever_attribute_carries_it():
    assert SLACK.key_of("message-list_1787937181.651799") == "1787937181.651799"
    assert SLACK.key_of("nothing here") == "nothing here"
    # discord numbers its rows instead, so the whole attribute is the key
    assert DISCORD.key_of("chat-messages-123-456") == "chat-messages-123-456"


def test_a_page_that_matches_nothing_comes_back_whole_and_says_so():
    class Blank:
        def evaluate(self, js, arg=None):
            return [{"key": "", "author": "", "text": "the pane, entire", "degraded": True}]

    seen = read_all(Blank(), WEBSITE, rounds=1)
    assert seen == [{"key": "", "author": "", "text": "the pane, entire", "degraded": True}]


def test_the_presets_are_findable_and_a_typo_is_not_an_empty_site():
    assert preset("slack") is SLACK
    assert preset(" Discord ") is DISCORD
    assert preset("website") is WEBSITE
    with pytest.raises(KeyError, match="discord, slack, website"):
        preset("slacc")


def test_a_preset_can_be_adjusted_without_being_rewritten():
    mine = SLACK.but(rows=".my-own-row", settle_ms=100)
    assert mine.rows == ".my-own-row" and mine.settle_ms == 100
    assert mine.author == SLACK.author          # everything else is still the preset
    assert SLACK.rows != ".my-own-row"          # and the preset itself is untouched


def test_visible_hours_can_wrap_around_midnight():
    at = lambda h: datetime(2026, 1, 1, h)      # noqa: E731
    assert within_hours(9, 17, now=at(12)) and not within_hours(9, 17, now=at(3))
    assert within_hours(22, 4, now=at(3)) and not within_hours(22, 4, now=at(12))


def test_a_second_run_only_reads_what_is_new(tmp_path):
    seen = Seen.load(tmp_path / "seen.json")
    first = rows(5)
    assert len(seen.fresh("#general", first)) == 5      # nothing known yet
    seen.record("#general", first)
    seen.save()

    again = Seen.load(tmp_path / "seen.json")
    assert again.mark("#general") == first[-1]["key"]
    assert again.fresh("#general", first) == []
    later = rows(7)
    assert [r["text"] for r in again.fresh("#general", later)] == ["row 5", "row 6"]


def test_something_older_that_grew_is_noticed(tmp_path):
    """A watermark cannot see a thread that gained a reply; a mark beside it can."""
    seen = Seen.load(tmp_path / "seen.json")
    seen.record("#general", rows(3), counts={"a": 2, "b": 0})
    assert seen.changed("#general", {"a": 2, "b": 0}) == []
    assert seen.changed("#general", {"a": 5, "b": 0}) == ["a"]
    assert seen.changed("#general", {"a": 2, "b": 1, "c": 3}) == ["b", "c"]


def test_an_older_watermark_file_is_still_read(tmp_path):
    path = tmp_path / "seen.json"
    path.write_text(json.dumps({"#general": "1787937181.000000"}), encoding="utf-8")
    assert Seen.load(path).mark("#general") == "1787937181.000000"


def test_an_edited_row_is_caught_by_its_own_digest(tmp_path):
    """A source that keeps a row's id when the wording changes says nothing about the edit.

    The watermark cannot see it — the row is not newer — and a reply count cannot either,
    because nothing grew. Marking the content is what makes an edit visible at all, and
    without it an edit is not late, it is never noticed.
    """
    from ml_stack.scrape import Seen, digest

    said = {"m1": "we should meet on tuesday", "m2": "agreed"}
    marks = {k: digest(v) for k, v in said.items()}
    seen = Seen.load(tmp_path / "seen.json")
    seen.record("#general", [], counts=marks)
    seen.save()

    again = Seen.load(tmp_path / "seen.json")
    assert again.changed("#general", marks) == []          # nothing moved
    said["m1"] = "we should meet on wednesday"
    edited = {k: digest(v) for k, v in said.items()}
    assert again.changed("#general", edited) == ["m1"]

    # a row never marked before is reported alongside the edit, which is right: it has
    # never been read either
    edited["m3"] = digest("and bring the notes")
    assert again.changed("#general", edited) == ["m1", "m3"]


def test_a_digest_is_stable_short_and_not_a_copy_of_what_was_read():
    from ml_stack.scrape import digest

    assert digest("a message") == digest("a message")
    assert digest("a message") != digest("a messagf")
    assert len(digest("a message")) == 16
    assert "message" not in digest("a message")
    assert digest("") == digest(None)  # a row with no text is not a row that changed
