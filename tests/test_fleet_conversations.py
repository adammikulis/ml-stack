"""Chats kept on disk."""

from __future__ import annotations

import pytest
from ml_stack.fleet.conversations import Conversations


@pytest.fixture
def store(tmp_path):
    return Conversations(tmp_path / "chats")


class TestKeeping:
    def test_a_chat_survives_a_restart(self, store, tmp_path):
        made = store.start(model="qwen3-4b.gguf")
        store.append(made.id, "user", "how tall is everest")
        store.append(made.id, "assistant", "8849 metres")

        # A fresh handle, as a restarted daemon would have.
        again = Conversations(tmp_path / "chats").get(made.id)
        assert [m.content for m in again.messages] == ["how tall is everest",
                                                       "8849 metres"]
        assert again.model == "qwen3-4b.gguf"

    def test_the_first_thing_asked_becomes_the_title(self, store):
        made = store.start()
        store.append(made.id, "user", "  what is\n a  gguf file ")
        assert store.get(made.id).title == "what is a gguf file"

    def test_a_long_first_message_is_cut_to_a_title(self, store):
        made = store.start()
        store.append(made.id, "user", "x" * 500)
        assert len(store.get(made.id).title) == 60

    def test_chats_are_listed_newest_first(self, store):
        a = store.start(title="older")
        b = store.start(title="newer")
        store.rename(a.id, "older")
        found = store.all()
        assert {c.id for c in found} == {a.id, b.id}
        assert found == sorted(found, key=lambda c: c.created, reverse=True)

    def test_a_reply_from_nothing_recognisable_is_refused(self, store):
        made = store.start()
        with pytest.raises(ValueError, match="system or user or assistant"):
            store.append(made.id, "wizard", "hello")

    def test_removing_one_leaves_the_others(self, store):
        a, b = store.start(title="a"), store.start(title="b")
        assert store.remove(a.id) is True
        assert store.remove(a.id) is False
        assert [c.id for c in store.all()] == [b.id]


class TestReadingBadFiles:
    def test_a_corrupt_file_is_skipped_rather_than_raising(self, store):
        good = store.start(title="fine")
        store.root.mkdir(parents=True, exist_ok=True)
        (store.root / "broken.json").write_text("{not json")
        (store.root / "empty.json").write_text("{}")
        assert [c.id for c in store.all()] == [good.id]

    def test_a_message_missing_its_role_is_dropped_not_fatal(self, store):
        made = store.start()
        store.append(made.id, "user", "kept")
        path = store.root / f"{made.id}.json"
        raw = path.read_text().replace('"role": "user"', '"rle": "user"')
        path.write_text(raw)
        assert store.get(made.id).messages == []


class TestNaming:
    @pytest.mark.parametrize("bad", ["../secrets", "a/b", "", "..", "x" * 65])
    def test_an_id_that_is_not_an_id_reaches_no_file(self, store, bad):
        assert store.get(bad) is None
        assert store.remove(bad) is False


class TestSearching:
    def test_it_finds_a_chat_by_something_said_in_it(self, store):
        a = store.start()
        store.append(a.id, "user", "what is a safetensors file")
        b = store.start()
        store.append(b.id, "user", "how do i quantise")

        assert [c.id for c in store.search("safetensors")] == [a.id]
        assert [c.id for c in store.search("QUANTISE")] == [b.id]
        assert len(store.search("")) == 2
        assert store.search("nothing here") == []

    def test_it_finds_a_chat_by_its_title(self, store):
        a = store.start(title="Everest")
        store.start(title="Kilimanjaro")
        assert [c.id for c in store.search("everest")] == [a.id]
