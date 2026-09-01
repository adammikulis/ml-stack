"""What the invented company said, written the way each product exports it.

Every file here is built in tmp_path from invented people (tests/known-fixtures.txt) and read
back with the stdlib or with `ml_stack.sources`; nothing touches a network or a model.
"""

from __future__ import annotations

import dataclasses
import email
import email.policy
import json
import mailbox
from datetime import UTC, datetime

import pytest

from ml_stack.sources import mbox as read_mbox
from ml_stack.sources import rows as read_rows
from ml_stack.sources import slack_export as read_slack
from ml_stack.sources import teams as read_teams
from ml_stack.world import Message
from ml_stack.world.emit import (directory, mbox, msgid, rows, slack_channel_id, slack_export,
                                 slack_user_id, teams, teams_user_id, ts_of, when)

PEOPLE = {
    "person:ada-lovelace": {"label": "Ada Lovelace"},
    "person:bea-marlow": {"label": "Bea Marlow"},
    "person:joan-clarke": {"label": "Joan Clarke", "email": "jc@pellard.example"},
}
DOMAIN = "pellard.example"
MIDNIGHT = 1725148800  # 2024-09-01T00:00:00Z


def msg(i: int, channel: str, sender: str, text: str, *, ts: str | None = None,
        thread: str | None = None, recipients: tuple[str, ...] = (), source: str = "slack",
        kind: str | None = None, attrs: dict | None = None) -> Message:
    return Message(id=f"msg:{i:04d}", source=source, channel=channel, sender=sender,
                   ts=ts or f"{MIDNIGHT - 3600 + i * 60}.{i:06d}", text=text,
                   recipients=recipients, thread=thread,
                   kind=kind or ("reply" if thread else "message"), attrs=attrs or {})


def corpus(source: str = "slack") -> list[Message]:
    """A channel with a thread that crosses midnight, a DM, and a reaction."""
    dm = "dm:person:ada-lovelace,person:bea-marlow"
    return [
        msg(1, "general", "person:ada-lovelace", "The foundry opens at nine.", source=source),
        msg(2, "general", "person:bea-marlow", "Nine works for me.", thread="msg:0001",
            source=source),
        msg(3, "general", "person:joan-clarke", "Nine it is, then.", thread="msg:0001",
            ts=f"{MIDNIGHT}.000100", source=source),
        msg(4, "general", "person:joan-clarke", "Coffee is on the second floor.",
            ts=f"{MIDNIGHT + 300}.000200", source=source),
        msg(5, dm, "person:ada-lovelace", "Did you see the plan?",
            recipients=("person:bea-marlow",), source=source),
        msg(6, dm, "person:bea-marlow", "I did. Looks fine.", thread="msg:0005",
            recipients=("person:ada-lovelace",), source=source),
        Message(id="msg:0007", source=source, channel="general", sender="person:bea-marlow",
                ts=f"{MIDNIGHT + 300}.000200", text="thumbsup", kind="reaction",
                thread="msg:0004", attrs={"to": "msg:0004"}),
    ]


def bare(m: Message) -> Message:
    return dataclasses.replace(m, attrs={})


def said(messages: list[Message]) -> list[Message]:
    return sorted((bare(m) for m in messages if m.kind != "reaction"), key=lambda m: m.id)


# --- people -------------------------------------------------------------------------------

def test_an_address_and_a_handle_are_derived_from_the_label():
    book = directory(PEOPLE, DOMAIN)
    assert book["person:ada-lovelace"] == {"id": "person:ada-lovelace", "label": "Ada Lovelace",
                                           "email": "ada.lovelace@pellard.example",
                                           "handle": "ada.lovelace"}
    assert book["person:joan-clarke"]["email"] == "jc@pellard.example"


def test_two_people_with_one_name_do_not_share_a_mailbox():
    book = directory({"person:a": {"label": "Ada Lovelace"}, "person:b": {"label": "Ada Lovelace"}})
    assert book["person:a"]["email"] != book["person:b"]["email"]
    assert book["person:a"]["handle"] != book["person:b"]["handle"]


def test_product_ids_are_minted_the_same_every_time():
    assert slack_user_id("person:ada-lovelace") == slack_user_id("person:ada-lovelace")
    assert slack_user_id("person:ada-lovelace") != slack_user_id("person:bea-marlow")
    assert slack_user_id("person:ada-lovelace").startswith("U0")
    assert len(slack_user_id("person:ada-lovelace")) == 9
    assert slack_channel_id("general").startswith("C0")
    assert teams_user_id("person:ada-lovelace") == teams_user_id("person:ada-lovelace")


def test_a_slack_ts_survives_the_trip_through_a_datetime():
    assert ts_of(when("1725148800.000100")) == "1725148800.000100"
    assert when("1725148800.000100") == datetime(2024, 9, 1, 0, 0, 0, 100, tzinfo=UTC)


# --- slack --------------------------------------------------------------------------------

def test_a_slack_export_round_trips_through_its_reader(tmp_path):
    original = corpus()
    out = slack_export(original, PEOPLE, tmp_path / "export", domain=DOMAIN)
    back = read_slack.read(out, PEOPLE, domain=DOMAIN)
    assert said(back) == said(original)
    replies = {m.id: m.thread for m in back}
    assert replies["msg:0002"] == "msg:0001" and replies["msg:0006"] == "msg:0005"


def test_a_slack_export_has_the_files_a_workspace_export_has(tmp_path):
    out = slack_export(corpus(), PEOPLE, tmp_path / "export", domain=DOMAIN)
    users = json.loads((out / "users.json").read_text())
    assert {u["name"] for u in users} >= {"ada.lovelace", "bea.marlow", "joan.clarke"}
    ada = next(u for u in users if u["real_name"] == "Ada Lovelace")
    assert ada["id"] == slack_user_id("person:ada-lovelace")
    assert ada["profile"]["email"] == "ada.lovelace@pellard.example"
    channels = json.loads((out / "channels.json").read_text())
    assert [c["name"] for c in channels] == ["general"]
    assert set(channels[0]) >= {"id", "name", "members", "created"}
    assert channels[0]["id"] == slack_channel_id("general")
    dms = json.loads((out / "dms.json").read_text())
    assert len(dms) == 1 and len(dms[0]["members"]) == 2
    assert (out / "dms" / dms[0]["id"]).is_dir()


def test_slack_day_files_split_at_midnight_utc(tmp_path):
    out = slack_export(corpus(), PEOPLE, tmp_path / "export", domain=DOMAIN)
    assert sorted(p.name for p in (out / "general").iterdir()) == ["2024-08-31.json",
                                                                    "2024-09-01.json"]
    before = json.loads((out / "general" / "2024-08-31.json").read_text())
    after = json.loads((out / "general" / "2024-09-01.json").read_text())
    assert all(int(r["ts"].split(".")[0]) < MIDNIGHT for r in before)
    assert all(int(r["ts"].split(".")[0]) >= MIDNIGHT for r in after)
    assert after[0]["ts"] == f"{MIDNIGHT}.000100"


def test_slack_rows_carry_threads_and_reactions_as_slack_writes_them(tmp_path):
    out = slack_export(corpus(), PEOPLE, tmp_path / "export", domain=DOMAIN)
    rows_ = [r for day in sorted((out / "general").iterdir())
             for r in json.loads(day.read_text())]
    root = next(r for r in rows_ if r["client_msg_id"] == "msg:0001")
    assert root["type"] == "message" and root["thread_ts"] == root["ts"]
    assert root["reply_count"] == 2 and len(root["replies"]) == 2
    reply = next(r for r in rows_ if r["client_msg_id"] == "msg:0002")
    assert reply["thread_ts"] == root["ts"]
    assert reply["parent_user_id"] == slack_user_id("person:ada-lovelace")
    reacted = next(r for r in rows_ if r["client_msg_id"] == "msg:0004")
    assert reacted["reactions"] == [{"name": "thumbsup",
                                     "users": [slack_user_id("person:bea-marlow")], "count": 1}]


def test_slack_ids_are_written_back_into_attrs(tmp_path):
    original = corpus()
    slack_export(original, PEOPLE, tmp_path / "export", domain=DOMAIN)
    assert original[0].attrs["slack"] == {"user": slack_user_id("person:ada-lovelace"),
                                          "channel": slack_channel_id("general")}


def test_an_emitter_takes_only_its_own_source_unless_told_otherwise(tmp_path):
    mixed = corpus() + [msg(9, "general", "person:ada-lovelace", "by mail", source="email")]
    out = slack_export(mixed, PEOPLE, tmp_path / "export", domain=DOMAIN)
    assert not any(m.id == "msg:0009" for m in read_slack.read(out, PEOPLE))
    out = slack_export(mixed, PEOPLE, tmp_path / "all", domain=DOMAIN, source=None)
    assert any(m.id == "msg:0009" for m in read_slack.read(out, PEOPLE))


# --- mail ---------------------------------------------------------------------------------

def test_an_mbox_round_trips_through_its_reader(tmp_path):
    original = corpus("email")
    out = mbox(original, PEOPLE, tmp_path / "mail.mbox", domain=DOMAIN)
    back = read_mbox.read(out, PEOPLE, domain=DOMAIN)
    assert said(back) == said(original)


def test_an_mbox_parses_with_the_stdlib_and_threads_by_message_id(tmp_path):
    out = mbox(corpus("email"), PEOPLE, tmp_path / "mail.mbox", domain=DOMAIN)
    assert out.read_bytes().startswith(b"From ada.lovelace@pellard.example ")
    box = mailbox.mbox(out, factory=lambda f: email.message_from_binary_file(
        f, policy=email.policy.default))
    mails = list(box)
    box.close()
    assert len(mails) == 6
    by_id = {m["X-World-Id"]: m for m in mails}
    root, reply = by_id["msg:0001"], by_id["msg:0002"]
    assert root["From"] == "Ada Lovelace <ada.lovelace@pellard.example>"
    assert root["Subject"] == "general" and reply["Subject"] == "Re: general"
    assert reply["In-Reply-To"] == root["Message-ID"] == msgid("msg:0001", DOMAIN)
    assert reply["References"] == root["Message-ID"]
    assert root["Date"].endswith("+0000")
    assert root.get_body(preferencelist=("plain",)).get_content() == "The foundry opens at nine.\n"
    dm = by_id["msg:0005"]
    assert dm["To"] == "Bea Marlow <bea.marlow@pellard.example>"
    assert by_id["msg:0003"]["From"] == "Joan Clarke <jc@pellard.example>"


def test_a_message_id_is_a_legal_dot_atom_even_for_an_id_with_a_colon():
    assert msgid("plain-id", DOMAIN) == "<plain-id@pellard.example>"
    minted = msgid("msg:0001", DOMAIN)
    assert ":" not in minted and minted.endswith("@pellard.example>")
    assert minted != msgid("msg.0001", DOMAIN)


# --- teams --------------------------------------------------------------------------------

def test_a_teams_dump_round_trips_through_its_reader(tmp_path):
    original = corpus("teams")
    out = teams(original, PEOPLE, tmp_path / "teams.json", domain=DOMAIN)
    back = read_teams.read(out, PEOPLE, domain=DOMAIN)
    assert said(back) == said(original)
    reactions = [m for m in back if m.kind == "reaction"]
    assert [(r.sender, r.text, r.attrs["to"]) for r in reactions] == [
        ("person:bea-marlow", "thumbsup", "msg:0004")]


def test_a_teams_dump_is_in_graphs_chat_message_shape(tmp_path):
    out = teams(corpus("teams"), PEOPLE, tmp_path / "teams.json", domain=DOMAIN)
    doc = json.loads(out.read_text())
    assert "value" in doc and doc["@odata.context"].endswith("Collection(chatMessage)")
    by_id = {m["id"]: m for m in doc["value"]}
    root, reply, dm = by_id["msg:0001"], by_id["msg:0002"], by_id["msg:0005"]
    assert root["replyToId"] is None and reply["replyToId"] == "msg:0001"
    assert root["from"]["user"] == {"id": teams_user_id("person:ada-lovelace"),
                                    "displayName": "Ada Lovelace", "userIdentityType": "aadUser"}
    assert root["createdDateTime"].endswith("Z")
    assert datetime.fromisoformat(root["createdDateTime"].replace("Z", "+00:00"))
    assert root["body"] == {"contentType": "text", "content": "The foundry opens at nine."}
    assert root["channelIdentity"]["channelId"].endswith("@thread.tacv2") and root["chatId"] is None
    assert dm["chatId"].endswith("@thread.v2") and dm["channelIdentity"] is None
    assert doc["channels"][0]["displayName"] == "general"
    assert doc["chats"][0]["chatType"] == "oneOnOne"
    assert by_id["msg:0004"]["reactions"][0]["reactionType"] == "thumbsup"


# --- scraper rows -------------------------------------------------------------------------

def test_rows_carry_every_field_the_scraper_row_has():
    out = rows(corpus(), PEOPLE, domain=DOMAIN, scraped_at="2024-09-01T09:00:00.000Z")
    roots = [r for r in out if "threadTs" not in r]
    replies = [r for r in out if "threadTs" in r]
    assert len(roots) == 3 and len(replies) == 3
    root = next(r for r in roots if r["text"] == "The foundry opens at nine.")
    assert set(root) == {"channel", "channelId", "ts", "sender", "text", "replies",
                         "scrapedAt", "permalink"}
    assert root["channel"] == "general" and root["channelId"] == slack_channel_id("general")
    assert root["sender"] == "Ada Lovelace" and root["replies"] == 2
    assert root["scrapedAt"] == "2024-09-01T09:00:00.000Z"
    assert root["permalink"] == f"slack-export/general/2024-08-31.json#p{root['ts'].replace('.', '')}"
    reply = next(r for r in replies if r["text"] == "Nine works for me.")
    assert set(reply) == {"channel", "channelId", "ts", "sender", "text", "threadTs",
                         "scrapedAt", "permalink"}
    assert reply["threadTs"] == root["ts"]
    assert not any(r["text"] == "thumbsup" for r in out)


def test_rows_round_trip_with_the_ids_the_scrapers_pipeline_mints(tmp_path):
    original = corpus()
    out = rows(original, PEOPLE, domain=DOMAIN)
    log = tmp_path / "messages.jsonl"
    log.write_text("".join(json.dumps(r) + "\n" for r in out))
    back = read_rows.read(log, PEOPLE, domain=DOMAIN)
    expected = []
    for m in original:
        if m.kind == "reaction":
            continue
        cid = m.attrs["slack"]["channel"]
        root = next((o for o in original if o.id == m.thread), None)
        expected.append(dataclasses.replace(
            bare(m), id=f"{cid}-{m.ts}", recipients=(),
            thread=f"{cid}-{root.ts}" if root else None))
    assert sorted((bare(m) for m in back), key=lambda m: m.id) == sorted(expected, key=lambda m: m.id)


def test_rows_and_the_export_agree_on_where_a_message_lives(tmp_path):
    original = corpus()
    export = slack_export(original, PEOPLE, tmp_path / "slack-export", domain=DOMAIN)
    for row in rows(original, PEOPLE, domain=DOMAIN):
        where, _, _ = row["permalink"].partition("#")
        day = tmp_path / where
        assert day.exists(), where
        assert any(r["ts"] == row["ts"] for r in json.loads(day.read_text()))
    assert export == tmp_path / "slack-export"


# --- every format -------------------------------------------------------------------------

@pytest.mark.parametrize("fmt", ["slack", "email", "teams", "rows"])
def test_ts_is_monotone_within_a_channel_after_reading(tmp_path, fmt):
    original = corpus("slack" if fmt == "rows" else fmt)
    if fmt == "slack":
        back = read_slack.read(slack_export(original, PEOPLE, tmp_path / "x", domain=DOMAIN), PEOPLE)
    elif fmt == "email":
        back = read_mbox.read(mbox(original, PEOPLE, tmp_path / "x.mbox", domain=DOMAIN), PEOPLE)
    elif fmt == "teams":
        back = read_teams.read(teams(original, PEOPLE, tmp_path / "x.json", domain=DOMAIN), PEOPLE)
    else:
        back = read_rows.read(rows(original, PEOPLE, domain=DOMAIN), PEOPLE)
    per_channel: dict[str, list[float]] = {}
    for m in back:
        if m.kind != "reaction":
            per_channel.setdefault(m.channel, []).append(float(m.ts))
    assert per_channel
    for channel, stamps in per_channel.items():
        assert stamps == sorted(stamps), channel
