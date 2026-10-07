"""Project coordination read visibility and protected authorization boundaries."""

from pathlib import Path

import pytest
from workspace_kit import Kit, clean_env

from ml_stack.workspace import Denied, tokens
from ml_stack.workspace.boardapi import Follow
from ml_stack.workspace.files import Attachment


@pytest.fixture
def project(monkeypatch, tmp_path):
    kit = Kit(clean_env(monkeypatch, tmp_path))
    kit.limits(sends_per_window=1000)
    kit.tokens = {name: kit.agent(name) for name in
                  ("worker-a", "worker-b", "observer", "outsider")}
    owner = kit.ws.auth(kit.owner)
    for name in ("worker-a", "worker-b", "observer"):
        kit.ws.registry.set_project(owner, name, {"key": "project-a", "name": "Widgets"})
    kit.ws.registry.set_project(owner, "outsider", {"key": "project-b", "name": "Widgets"})
    return kit


def coordination(project):
    ws, t = project.ws, project.tokens
    return ws.send(t["worker-a"], "worker-b", "note", "Review documentation", subject="Doc review")


def test_project_conversations_are_discoverable_and_readable(project):
    ws, t = project.ws, project.tokens
    root = coordination(project)
    ws.send(t["worker-b"], "worker-a", "note", "Recommendation", reply_to=root["seq"])
    assert [(row["a"], row["b"]) for row in ws.board.dm_list(t["observer"])] == [
        ("worker-a", "worker-b")]
    assert len(ws.board.dm(t["observer"], "worker-b", between="worker-a")) == 2
    assert len(ws.board.ui_dm(t["observer"], "worker-a", "worker-b")) == 2
    assert len(ws.board.ui_thread(t["observer"], root["seq"])["messages"]) == 2
    assert ws.board.digest(t["observer"], thread=root["seq"])["messages"] == 2
    assert len(ws.board.follow(t["observer"], Follow(thread=root["seq"], after=0))["messages"]) == 2
    assert len(ws.board.follow(t["observer"], Follow(dm="worker-b", between="worker-a", after=0))["messages"]) == 2
    assert len(ws.thread(t["observer"], root["seq"])) == 2


def test_other_projects_do_not_discover_or_read_shared_conversations(project):
    ws, t = project.ws, project.tokens
    root = coordination(project)
    assert ws.board.dm_list(t["outsider"]) == []
    for read in (lambda: ws.board.dm(t["outsider"], "worker-b", between="worker-a"),
                 lambda: ws.board.ui_thread(t["outsider"], root["seq"]),
                 lambda: ws.board.digest(t["outsider"], thread=root["seq"]),
                 lambda: ws.board.follow(t["outsider"], Follow(thread=root["seq"])),
                 lambda: ws.thread(t["outsider"], root["seq"]),
                 lambda: ws.board.subscribe(t["outsider"], "thread", str(root["seq"]))):
        with pytest.raises(Denied):
            read()


def test_shared_read_does_not_grant_participant_reply_authority(project):
    ws, t = project.ws, project.tokens
    root = coordination(project)
    with pytest.raises(Denied, match="participant"):
        ws.send(t["observer"], "worker-b", "note", "Inject reply", reply_to=root["seq"])


def test_project_thread_subscription_delivers_new_replies_and_wakes_followers(project):
    ws, t = project.ws, project.tokens
    root = coordination(project)
    ws.board.subscribe(t["observer"], "thread", str(root["seq"]), "inbox")
    assert ws.inbox(t["observer"]) == []
    reply = ws.send(t["worker-b"], "worker-a", "note", "Ready", reply_to=root["seq"])
    assert [row["seq"] for row in ws.inbox(t["observer"])] == [reply["seq"]]
    names = ws.board.wake_names(ws.bus.get(reply["seq"]))
    assert "observer" in names and "observer.follow" in names
    assert "outsider" not in names and "outsider.follow" not in names


def test_project_digest_subscription_summarizes_new_shared_replies(project):
    ws, t = project.ws, project.tokens
    root = coordination(project)
    ws.board.subscribe(t["observer"], "thread", str(root["seq"]), "digest")
    ws.send(t["worker-b"], "worker-a", "note", "Ready", reply_to=root["seq"])
    assert ws.board.digest(t["observer"])["messages"] == 1


def test_project_attachments_are_discoverable_and_readable(project):
    ws, t = project.ws, project.tokens
    handle = ws.files.attach(t["worker-a"], "worker-b", b"Documentation recommendations\n",
                             Attachment(name="review.txt"))["file"]
    assert [item["id"] for item in ws.files.list(t["observer"])] == [handle]
    assert "Documentation recommendations" in ws.files.read_text(t["observer"], handle)["text"]
    assert ws.files.list(t["outsider"]) == []
    with pytest.raises(Denied):
        ws.files.read_text(t["outsider"], handle)


def test_delegates_inherit_project_reads_but_still_require_read_capability(project):
    ws, t = project.ws, project.tokens
    root = coordination(project)
    child = tokens.read_file(Path(ws.delegate(t["observer"], "reader", can=("read",))["token_file"]))
    assert ws.board.ui_thread(child, root["seq"])["messages"][0]["seq"] == root["seq"]
    blind = tokens.read_file(Path(ws.delegate(t["observer"], "blind", can=("send",))["token_file"]))
    with pytest.raises(Denied, match="read"):
        ws.board.ui_thread(blind, root["seq"])


def test_project_board_read_visibility_preserves_membership_for_posting(project):
    ws, t = project.ws, project.tokens
    board = ws.board.place("worker-a", {"key": "project-a", "name": "Widgets"})[-1]
    ws.send(t["worker-a"], board, "note", "Project progress")
    assert board in {item["name"] for item in ws.board.list(t["observer"])}
    assert ws.board.read(t["observer"], board)[0]["board"] == board
    with pytest.raises(Denied):
        ws.send(t["observer"], board, "note", "Without membership")
    with pytest.raises(Denied):
        ws.board.read(t["outsider"], board)


def test_shared_read_cannot_follow_cross_project_rows_linked_into_thread(project):
    ws, t = project.ws, project.tokens
    root = coordination(project)
    ws.send(t["worker-a"], "outsider", "note", "Separate project detail", reply_to=root["seq"])
    for read in (lambda: ws.board.ui_thread(t["observer"], root["seq"]),
                 lambda: ws.board.digest(t["observer"], thread=root["seq"]),
                 lambda: ws.board.follow(t["observer"], Follow(thread=root["seq"])),
                 lambda: ws.thread(t["observer"], root["seq"])):
        with pytest.raises(Denied):
            read()


def test_participant_thread_subscription_does_not_duplicate_direct_delivery(project):
    ws, t = project.ws, project.tokens
    root = coordination(project)
    ws.inbox(t["worker-b"], ack=True)
    ws.board.subscribe(t["worker-b"], "thread", str(root["seq"]), "inbox")
    reply = ws.send(t["worker-a"], "worker-b", "note", "Update", reply_to=root["seq"])
    assert [row["seq"] for row in ws.inbox(t["worker-b"])] == [reply["seq"]]


def test_digest_does_not_reveal_foreign_root_subject_through_shared_reply(project):
    ws, t = project.ws, project.tokens
    root = ws.send(t["outsider"], "worker-a", "note", "Foreign detail", subject="Foreign subject")
    ws.board.subscribe(t["observer"], "agent", "worker-a", "digest")
    ws.send(t["worker-a"], "worker-b", "note", "Project reply", reply_to=root["seq"])
    digest = ws.board.digest(t["observer"])
    assert digest["messages"] == 1
    assert "Project reply" in digest["text"]
    assert "Foreign subject" not in digest["text"]


def test_follow_filters_foreign_reply_added_after_scope_validation(project, monkeypatch):
    ws, t = project.ws, project.tokens
    root = coordination(project)
    scope = ws.board._scope

    def scope_then_append(*args):
        member = scope(*args)
        ws.send(t["worker-a"], "outsider", "note", "Foreign reply", reply_to=root["seq"])
        return member

    monkeypatch.setattr(ws.board, "_scope", scope_then_append)
    found = ws.board.follow(t["observer"], Follow(thread=root["seq"], after=0))["messages"]
    assert [row["seq"] for row in found] == [root["seq"]]
