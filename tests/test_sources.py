"""Readers for the formats products export, and the one that sniffs which is which.

Everything is written into tmp_path by hand, in each product's shape, from invented names.
"""

from __future__ import annotations

import json
import mailbox
from email.message import EmailMessage

import pytest

import ml_stack.sources as sources
from ml_stack.sources import mbox, rows, slack_export, teams
from ml_stack.world.emit import slack_user_id, teams_user_id

PEOPLE = {"person:ada-lovelace": {"label": "Ada Lovelace"},
          "person:bea-marlow": {"label": "Bea Marlow"}}
DOMAIN = "pellard.example"
ADA, BEA = slack_user_id("person:ada-lovelace"), slack_user_id("person:bea-marlow")


def a_slack_export(root):
    """A workspace export as Slack lays it out, with a DM directory at the top level."""
    root.mkdir()
    (root / "users.json").write_text(json.dumps([
        {"id": ADA, "name": "ada.lovelace", "real_name": "Ada Lovelace",
         "profile": {"real_name": "Ada Lovelace", "email": "ada.lovelace@pellard.example"}},
        {"id": BEA, "name": "bea.marlow", "real_name": "Bea Marlow",
         "profile": {"real_name": "Bea Marlow", "email": "bea.marlow@pellard.example"}},
        {"id": "U0STRANGER", "name": "joan.clarke", "real_name": "Joan Clarke",
         "profile": {"real_name": "Joan Clarke", "email": "nobody@elsewhere.example"}},
    ]))
    (root / "channels.json").write_text(json.dumps([
        {"id": "C0GENERAL1", "name": "general", "created": 1725100000, "members": [ADA, BEA]}]))
    (root / "dms.json").write_text(json.dumps([
        {"id": "D0ONETOONE", "created": 1725100000, "members": [ADA, BEA]}]))
    (root / "general").mkdir()
    (root / "general" / "2024-08-31.json").write_text(json.dumps([
        {"type": "message", "user": ADA, "text": "Opening at nine.", "ts": "1725100000.000100",
         "thread_ts": "1725100000.000100", "reply_count": 1,
         "replies": [{"user": BEA, "ts": "1725100060.000200"}],
         "reactions": [{"name": "eyes", "users": [BEA], "count": 1}]},
        {"type": "message", "user": BEA, "text": "Fine by me.", "ts": "1725100060.000200",
         "thread_ts": "1725100000.000100", "parent_user_id": ADA},
        {"type": "message", "subtype": "channel_join", "user": "U0STRANGER",
         "text": "<@U0STRANGER> has joined the channel", "ts": "1725100120.000300"},
    ]))
    (root / "D0ONETOONE").mkdir()
    (root / "D0ONETOONE" / "2024-09-01.json").write_text(json.dumps([
        {"type": "message", "user": BEA, "text": "Lunch?", "ts": "1725200000.000100"}]))
    return root


def test_a_slack_export_maps_users_back_to_people_when_given_them(tmp_path):
    back = slack_export.read(a_slack_export(tmp_path / "export"), PEOPLE)
    by_id = {m.id: m for m in back}
    root = by_id["C0GENERAL1-1725100000.000100"]
    assert root.sender == "person:ada-lovelace" and root.thread is None
    assert root.attrs["slack"] == {"user": ADA, "channel": "C0GENERAL1"}
    reply = by_id["C0GENERAL1-1725100060.000200"]
    assert reply.thread == root.id and reply.kind == "reply"
    reaction = next(m for m in back if m.kind == "reaction")
    assert (reaction.sender, reaction.text, reaction.attrs["to"]) == ("person:bea-marlow", "eyes",
                                                                       root.id)
    joined = by_id["C0GENERAL1-1725100120.000300"]
    assert joined.sender == "U0STRANGER" and joined.attrs["sender_kind"] == "slack_user_id"
    assert joined.attrs["subtype"] == "channel_join"


def test_a_slack_dm_directory_at_the_top_level_is_found(tmp_path):
    back = slack_export.read(a_slack_export(tmp_path / "export"), PEOPLE)
    dm = next(m for m in back if m.channel.startswith("dm:"))
    assert dm.channel == "dm:person:ada-lovelace,person:bea-marlow"
    assert dm.sender == "person:bea-marlow" and dm.recipients == ("person:ada-lovelace",)


def test_without_people_a_slack_export_keeps_its_own_ids(tmp_path):
    back = slack_export.read(a_slack_export(tmp_path / "export"))
    root = next(m for m in back if m.text == "Opening at nine.")
    assert root.sender == ADA
    assert root.attrs["sender_kind"] == "slack_user_id"
    assert root.attrs["sender_name"] == "Ada Lovelace"


def a_real_looking_mbox(path):
    """Two mails and a reply, as a mail client would write them: no X-World headers."""
    box = mailbox.mbox(path)
    for i, (frm, to, subject, body, parent) in enumerate([
        ("Ada Lovelace <ada.lovelace@pellard.example>", "Bea Marlow <bea.marlow@pellard.example>",
         "Foundry hours", "Nine to five, starting Monday.", None),
        ("Bea Marlow <bea.marlow@pellard.example>", "ada.lovelace@pellard.example",
         "RE: Foundry hours", "Works for me.", "<one@pellard.example>"),
        ("Grace Hopper <grace.hopper@elsewhere.example>", "ada.lovelace@pellard.example",
         "Invoice", "Attached.", None),
    ]):
        em = EmailMessage()
        em["From"], em["To"], em["Subject"] = frm, to, subject
        em["Date"] = f"Sun, 01 Sep 2024 09:0{i}:00 +0000"
        em["Message-ID"] = f"<{['one', 'two', 'three'][i]}@pellard.example>"
        if parent:
            em["In-Reply-To"] = parent
            em["References"] = parent
        em.set_content(body)
        box.add(em)
    box.flush()
    box.close()
    return path


def test_a_real_looking_mbox_is_read_by_address_and_threaded_by_subject(tmp_path):
    back = mbox.read(a_real_looking_mbox(tmp_path / "in.mbox"), PEOPLE, domain=DOMAIN)
    assert [m.id for m in back] == ["one@pellard.example", "two@pellard.example",
                                    "three@pellard.example"]
    first, reply, other = back
    assert first.sender == "person:ada-lovelace"
    assert first.recipients == ("person:bea-marlow",)
    assert first.channel == reply.channel == "Foundry hours"
    assert reply.thread == first.id and reply.kind == "reply"
    assert first.ts == "1725181200.000000" and reply.ts == "1725181260.000000"
    assert first.text == "Nine to five, starting Monday."
    assert other.sender == "grace.hopper@elsewhere.example"
    assert other.attrs["sender_kind"] == "email" and other.attrs["sender_name"] == "Grace Hopper"


def a_graph_page(path):
    """What `GET /chats/{id}/messages` returns: a bare value list, no channel names."""
    path.write_text(json.dumps({"@odata.context": "x", "value": [
        {"id": "1725100000100", "replyToId": None, "messageType": "message",
         "createdDateTime": "2024-08-31T10:26:40.100Z",
         "from": {"user": {"id": teams_user_id("person:ada-lovelace"),
                           "displayName": "Ada Lovelace"}},
         "body": {"contentType": "html", "content": "<p>Opening at nine.</p>"},
         "chatId": "19:abc@thread.v2", "channelIdentity": None,
         "reactions": [{"reactionType": "like", "createdDateTime": "2024-08-31T10:27:00Z",
                        "user": {"user": {"id": "not-a-known-uuid", "displayName": "Bea Marlow"}}}]},
        {"id": "1725100060200", "replyToId": "1725100000100", "messageType": "message",
         "createdDateTime": "2024-08-31T10:27:40.200Z",
         "from": {"user": {"id": "not-a-known-uuid", "displayName": "Bea Marlow"}},
         "body": {"contentType": "text", "content": "Fine by me."},
         "chatId": "19:abc@thread.v2", "channelIdentity": None},
    ]}))
    return path


def test_a_graph_page_is_read_with_the_chat_id_as_the_channel(tmp_path):
    back = teams.read(a_graph_page(tmp_path / "chat.json"), PEOPLE)
    root, reaction, reply = back
    assert root.channel == "19:abc@thread.v2" and root.sender == "person:ada-lovelace"
    assert root.ts == "1725100000.100000" and root.attrs["content_type"] == "html"
    assert reply.thread == root.id and reply.sender == "person:bea-marlow"  # by display name
    assert reaction.kind == "reaction" and reaction.text == "like"
    assert reaction.attrs["to"] == root.id


def test_scraper_rows_keep_the_last_version_of_a_message(tmp_path):
    log = tmp_path / "messages.jsonl"
    log.write_text("\n".join(json.dumps(r) for r in [
        {"channel": "general", "channelId": "C0GENERAL1", "ts": "1725100000.000100",
         "sender": "Ada Lovelace", "text": "Opening at nien.", "replies": 0,
         "scrapedAt": "2024-09-01T09:00:00.000Z"},
        {"channel": "general", "channelId": "C0GENERAL1", "ts": "1725100060.000200",
         "sender": "Bea Marlow", "text": "Fine by me.", "threadTs": "1725100000.000100",
         "scrapedAt": "2024-09-01T09:00:00.000Z"},
        {"channel": "general", "channelId": "C0GENERAL1", "ts": "1725100000.000100",
         "sender": "Ada Lovelace", "text": "Opening at nine.", "replies": 1,
         "scrapedAt": "2024-09-01T10:00:00.000Z"},
        {"ts": "", "sender": "", "text": "could not read", "degraded": True},
        {"channel": "general", "channelId": "C0GENERAL1", "ts": "1725100120.000300",
         "sender": "Joan Clarke", "text": "hello", "replies": 0,
         "scrapedAt": "2024-09-01T10:00:00.000Z"},
    ]) + "\n")
    back = rows.read(log, PEOPLE)
    assert [m.id for m in back] == ["C0GENERAL1-1725100000.000100", "C0GENERAL1-1725100060.000200",
                                    "C0GENERAL1-1725100120.000300"]
    root, reply, stranger = back
    assert root.text == "Opening at nine." and root.attrs["replies"] == 1
    assert root.sender == "person:ada-lovelace"
    assert reply.thread == root.id
    assert stranger.sender == "Joan Clarke" and stranger.attrs["sender_kind"] == "display_name"


def test_rows_can_be_handed_over_already_loaded():
    back = rows.read([{"channel": "general", "channelId": "C0GENERAL1", "ts": "1725100000.000100",
                       "sender": "Ada Lovelace", "text": "hi", "replies": 0}])
    assert len(back) == 1 and back[0].sender == "Ada Lovelace"


def test_read_sniffs_each_format(tmp_path):
    export = a_slack_export(tmp_path / "export")
    mail = a_real_looking_mbox(tmp_path / "in.mbox")
    graph = a_graph_page(tmp_path / "chat.json")
    log = tmp_path / "messages.jsonl"
    log.write_text(json.dumps({"channel": "general", "channelId": "C0GENERAL1",
                               "ts": "1725100000.000100", "sender": "Ada Lovelace",
                               "text": "hi", "replies": 0}) + "\n")
    assert sources.sniff(export) == "slack_export"
    assert sources.sniff(mail) == "mbox"
    assert sources.sniff(graph) == "teams"
    assert sources.sniff(log) == "rows"
    assert {m.source for m in sources.read(export, PEOPLE)} == {"slack"}
    assert {m.source for m in sources.read(mail, PEOPLE)} == {"email"}
    assert {m.source for m in sources.read(graph, PEOPLE)} == {"teams"}
    assert {m.source for m in sources.read(log, PEOPLE)} == {"slack"}


def test_read_refuses_what_it_cannot_place(tmp_path):
    (tmp_path / "notes.txt").write_text("just some prose\n")
    with pytest.raises(ValueError):
        sources.read(tmp_path / "notes.txt")
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError):
        sources.read(tmp_path / "empty")
