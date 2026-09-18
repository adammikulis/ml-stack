"""`ml_stack.sources.rows`: the rows a Slack scraper writes, read back to messages.

The entry path for scraped data. Every row here is written the way the scraper writes it,
and the round trip goes through `ml_stack.world.emit.rows`, which is the shape it promises.
Every name is invented.
"""

from __future__ import annotations

import json

import pytest

from ml_stack.messages import Message, message_id
from ml_stack.sources.rows import read

PEOPLE = {"person:ada-lovelace": {"label": "Ada Lovelace"},
          "person:bea-marlow": {"label": "Bea Marlow"}}
DOMAIN = "pellard.example"


def a_row(ts, sender, text, **rest):
    """One scraper row in the shape `world.emit.rows` writes."""
    return {"channel": "general", "channelId": "C0GENERAL1", "ts": ts, "sender": sender,
            "text": text, "scrapedAt": "2024-09-01T10:00:00.000Z", **rest}


def a_log(tmp_path, rows):
    path = tmp_path / "messages.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


def test_a_row_becomes_a_message_with_the_id_the_pipeline_names_it_by(tmp_path):
    log = a_log(tmp_path, [a_row("1725100000.000100", "Ada Lovelace", "Opening at nine.",
                                 replies=1, permalink="export/general/2024-08-31.json#p1")])
    (one,) = read(log, PEOPLE, domain=DOMAIN)

    assert one.id == message_id("C0GENERAL1", "1725100000.000100")
    assert one.source == "slack" and one.channel == "general"
    assert one.sender == "person:ada-lovelace"
    assert one.ts == "1725100000.000100" and one.text == "Opening at nine."
    assert one.recipients == () and one.thread is None and one.kind == "message"
    assert one.attrs["slack"] == {"channel": "C0GENERAL1"}
    assert one.attrs["sender_kind"] == "person"
    assert one.attrs["sender_name"] == "Ada Lovelace"
    assert one.attrs["replies"] == 1
    assert one.attrs["permalink"] == "export/general/2024-08-31.json#p1"
    assert one.attrs["scrapedAt"] == "2024-09-01T10:00:00.000Z"


def test_a_reply_points_at_the_root_and_is_named_a_reply(tmp_path):
    log = a_log(tmp_path, [
        a_row("1725100000.000100", "Ada Lovelace", "Opening at nine.", replies=1),
        a_row("1725100060.000200", "Bea Marlow", "Fine by me.",
              threadTs="1725100000.000100")])
    root, reply = read(log, PEOPLE)

    assert root.kind == "message" and root.thread is None
    assert reply.kind == "reply"
    assert reply.thread == root.id == "C0GENERAL1-1725100000.000100"
    assert reply.sender == "person:bea-marlow"


def test_a_sender_nobody_knows_keeps_the_display_name_the_scraper_saw(tmp_path):
    log = a_log(tmp_path, [a_row("1725100000.000100", "Joan Clarke", "hello")])
    (one,) = read(log, PEOPLE)

    assert one.sender == "Joan Clarke"
    assert one.attrs["sender_kind"] == "display_name"
    assert one.attrs["sender_name"] == "Joan Clarke"


def test_a_display_name_is_matched_whatever_its_casing(tmp_path):
    log = a_log(tmp_path, [a_row("1725100000.000100", "ADA LOVELACE", "hi")])
    (one,) = read(log, PEOPLE)

    assert one.sender == "person:ada-lovelace"
    assert one.attrs["sender_name"] == "ADA LOVELACE"


def test_without_the_world_s_people_every_sender_stays_a_display_name(tmp_path):
    log = a_log(tmp_path, [a_row("1725100000.000100", "Ada Lovelace", "hi")])
    (one,) = read(log)

    assert one.sender == "Ada Lovelace"
    assert one.attrs["sender_kind"] == "display_name"


def test_a_row_the_scraper_could_not_read_and_one_with_no_ts_are_dropped(tmp_path):
    log = a_log(tmp_path, [
        {"ts": "1725100000.000100", "channel": "general", "channelId": "C0GENERAL1",
         "sender": "Ada Lovelace", "text": "could not read", "degraded": True},
        a_row("", "Ada Lovelace", "no timestamp"),
        a_row("1725100060.000200", "Bea Marlow", "kept")])

    assert [m.text for m in read(log, PEOPLE)] == ["kept"]


def test_the_last_version_of_a_row_wins_whatever_order_the_polls_appended(tmp_path):
    log = a_log(tmp_path, [
        a_row("1725100000.000100", "Ada Lovelace", "Opening at nien.", replies=0),
        a_row("1725100000.000100", "Ada Lovelace", "Opening at nine.", replies=2)])
    (one,) = read(log, PEOPLE)

    assert one.text == "Opening at nine." and one.attrs["replies"] == 2


def test_blank_lines_and_a_half_written_one_are_stepped_over(tmp_path):
    path = tmp_path / "messages.jsonl"
    path.write_text("\n".join([
        json.dumps(a_row("1725100000.000100", "Ada Lovelace", "first")),
        "",
        '{"channel": "general", "ts": "172510',
        "   ",
        json.dumps(a_row("1725100060.000200", "Bea Marlow", "second")),
    ]) + "\n", encoding="utf-8")

    assert [m.text for m in read(path, PEOPLE)] == ["first", "second"]


def test_a_log_already_saved_as_one_json_array_reads_the_same(tmp_path):
    made = [a_row("1725100000.000100", "Ada Lovelace", "first"),
            a_row("1725100060.000200", "Bea Marlow", "second")]
    array = tmp_path / "messages.json"
    array.write_text(json.dumps([*made, "a string nobody wrote"]), encoding="utf-8")

    assert [m.id for m in read(array, PEOPLE)] == [m.id for m in read(made, PEOPLE)]


def test_rows_handed_over_in_memory_are_not_changed_by_the_reading():
    made = [a_row("1725100000.000100", "Ada Lovelace", "hi")]
    before = json.dumps(made, sort_keys=True)
    read(made, PEOPLE)

    assert json.dumps(made, sort_keys=True) == before


def test_messages_come_back_by_channel_and_then_by_time(tmp_path):
    log = a_log(tmp_path, [
        {**a_row("1725100060.000200", "Ada Lovelace", "later in general")},
        {**a_row("1725100000.000100", "Bea Marlow", "earlier in general")},
        {**a_row("1725100030.000100", "Ada Lovelace", "in the annex"),
         "channel": "annex", "channelId": "C0ANNEX01"}])

    assert [m.text for m in read(log, PEOPLE)] == [
        "in the annex", "earlier in general", "later in general"]


def test_a_row_with_no_channel_id_is_named_by_its_channel(tmp_path):
    log = a_log(tmp_path, [{"channel": "general", "ts": "1725100000.000100",
                            "sender": "Ada Lovelace", "text": "hi"}])
    (one,) = read(log, PEOPLE)

    assert one.id == "general-1725100000.000100"
    assert one.channel == "general" and one.attrs["slack"] == {"channel": "general"}


def test_a_row_with_neither_is_named_x(tmp_path):
    log = a_log(tmp_path, [{"ts": "1725100000.000100", "sender": "Ada Lovelace", "text": "hi"}])
    (one,) = read(log, PEOPLE)

    assert one.id == "x-1725100000.000100" and one.channel == "x"


def test_a_row_with_no_text_reads_as_an_empty_message(tmp_path):
    log = a_log(tmp_path, [{"channel": "general", "channelId": "C0GENERAL1",
                            "ts": "1725100000.000100", "sender": "Ada Lovelace"}])
    (one,) = read(log, PEOPLE)

    assert one.text == "" and one.sender == "person:ada-lovelace"


def test_a_ts_that_is_not_a_number_is_still_read_and_sorts_first(tmp_path):
    log = a_log(tmp_path, [a_row("1725100000.000100", "Ada Lovelace", "numbered"),
                           a_row("not-a-time", "Bea Marlow", "odd")])

    assert [m.text for m in read(log, PEOPLE)] == ["odd", "numbered"]


def test_a_missing_log_is_not_swallowed(tmp_path):
    with pytest.raises(OSError):
        read(tmp_path / "nothing.jsonl")


def test_what_the_emitter_writes_is_what_the_reader_reads_back():
    from ml_stack.world.emit import rows as emit_rows

    said = [Message(id="m1", source="slack", channel="general",
                    sender="person:ada-lovelace", ts="1725100000.000100",
                    text="Opening at nine."),
            Message(id="m2", source="slack", channel="general",
                    sender="person:bea-marlow", ts="1725100060.000200",
                    text="Fine by me.", thread="m1", kind="reply")]
    written = emit_rows(said, PEOPLE, domain=DOMAIN, scraped_at="2024-09-01T10:00:00.000Z")
    back = read(written, PEOPLE, domain=DOMAIN)

    assert [m.sender for m in back] == ["person:ada-lovelace", "person:bea-marlow"]
    assert [m.text for m in back] == ["Opening at nine.", "Fine by me."]
    assert [m.ts for m in back] == ["1725100000.000100", "1725100060.000200"]
    assert back[1].thread == back[0].id and back[1].kind == "reply"
    assert back[0].attrs["replies"] == 1
    assert all(m.source == "slack" for m in back)
