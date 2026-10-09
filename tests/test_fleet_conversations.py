"""Chats kept on disk."""

from __future__ import annotations

import pytest

from poolhouse.fleet.conversations import Conversations


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
        store.update(a.id, title="older")
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
        import json
        store.root.mkdir(parents=True)
        (store.root / "legacy.json").write_text(json.dumps({"id": "legacy", "created": 1,
            "messages": [{"rle": "user", "content": "invalid"}]}))
        assert store.get("legacy").messages == []



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


def test_conversation_settings_and_model_survive_reopening(store, tmp_path):
    made = store.start(model="small", settings={"mode": "coding", "project": "/tmp/project", "effort": "low", "max_output_tokens": 23456})
    store.append(made.id, "user", "Review the brakes")
    store.update(made.id, model="large", settings={"temperature": .3, "max_effort": "high"}, title="Brake review")
    again = Conversations(tmp_path / "chats").get(made.id)
    assert again.model == "large"
    assert again.settings["mode"] == "coding"
    assert again.settings["project"] == "/tmp/project"
    assert again.settings["temperature"] == .3
    assert again.settings["effort"] == "low"
    assert again.settings["max_effort"] == "high"
    assert again.settings["harness"] == "pi"
    assert again.settings["context"] == 0
    assert again.settings["max_output_tokens"] == 23456
    assert [message.content for message in again.messages] == ["Review the brakes"]


@pytest.mark.parametrize("settings", [{"temperature": True}, {"temperature": -1}, {"temperature": float("nan")},
                                      {"temperature": 3}, {"mode": "unknown"}, {"unknown": "setting"},
                                      {"max_output_tokens": True}, {"max_output_tokens": 0}, {"max_output_tokens": 1.5},
                                      {"effort": "infinite"}, {"max_effort": "auto"}, {"role": 42}, []])
def test_invalid_settings_leave_the_conversation_unchanged(store, settings):
    made = store.start(title="Kept")
    with pytest.raises(ValueError):
        store.update(made.id, title="Changed", settings=settings)
    assert store.get(made.id).title == "Kept"
    assert store.get(made.id).settings == made.settings


def test_old_saved_messages_acquire_default_settings_and_version_on_update(store):
    import json
    store.root.mkdir(parents=True)
    path = store.root / "legacy.json"
    raw = {"id": "legacy", "title": "Old chat", "model": "old-model", "created": 1,
           "messages": [{"role": "user", "content": "Keep this message", "at": 1}]}
    path.write_text(json.dumps(raw))
    old = store.get("legacy")
    assert old.model == "old-model"
    assert old.settings["mode"] == "chat"
    assert old.settings["max_output_tokens"] is None
    store.update("legacy", settings={"temperature": .7})
    saved = store.get("legacy").public()
    assert saved["version"] == 1
    assert saved["messages"][0]["content"] == "Keep this message"
    assert saved["settings"]["temperature"] == .7
    assert json.loads(path.read_text()) == raw
    store.remove("legacy")
    assert Conversations(store.root).get("legacy") is None


def test_conversations_link_messages_models_and_projects_in_the_graph(store):
    from poolhouse.graph.store import GraphStore

    made = store.start(model="chosen-model", settings={"project": "/tmp/project"})
    store.append(made.id, "user", "Inspect the sensors")
    with GraphStore(store.root / "conversations.db", buffer_pool_size=32 << 20) as graph:
        rows = graph.query("MATCH (c:Node {id:$id})-[e:Edge]->(n:Node) "
                           "RETURN e.rel AS relation, n.label AS label", {"id": "conversation:" + made.id})
    assert {(row["relation"], row["label"]) for row in rows} == {
        ("contains", "Inspect the sensors"), ("uses-model", "chosen-model"),
        ("in-project", "/tmp/project")}


def test_concurrent_handles_append_without_overwriting_messages(store):
    from concurrent.futures import ThreadPoolExecutor

    made = store.start()
    def append(index):
        Conversations(store.root).append(made.id, "user", str(index))
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(append, range(12)))
    assert sorted(int(message.content) for message in store.get(made.id).messages) == list(range(12))


def test_plain_conversations_do_not_install_search_extensions(store, monkeypatch):
    from poolhouse.graph.store import GraphStore

    def refuse(extension):
        raise AssertionError(f"unexpected search extension: {extension}")
    monkeypatch.setattr(GraphStore, "load", lambda self, extension: refuse(extension))
    made = store.start()
    store.append(made.id, "user", "Keep this offline")
    assert store.get(made.id).messages[0].content == "Keep this offline"
