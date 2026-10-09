"""The statement hook and the approval answer, run as child processes against transcripts in the shapes Claude Code writes."""

from __future__ import annotations

import functools
import json
import os
import threading
import time

import pytest
from person_support import (
    SESSION,
    VERSION,
    answer_event,
    approve,
    assistant,
    authorized,
    commit,
    git,
    human,
    local_command,
    notification,
    peer,
    person_consume,
    project_dir,
    prompt_event,
    proposal,
    repository,
    rows,
    run_hook,
    say,
    target,
    tip,
    transcript,
)

from ml_stack.workspace import person_auth, person_store, person_transcript
from ml_stack.workspace.person_auth import Identity, NotAuthorized

PROPOSAL = assistant("The tests pass. I'll push main to origin now.")


@pytest.fixture
def world(tmp_path):
    return tmp_path, tmp_path / "state", repository(tmp_path / "repo")


def log_of(state):
    return person_store.open_log(state / "person")


def consume_now(state, repo, sha="", session=SESSION, agent=""):
    return person_auth.consume("release-main", target(repo, sha), Identity(session, agent), log=log_of(state))


def test_the_pinned_versions_name_the_release_the_shapes_were_read_from():
    assert person_transcript.PINNED_VERSIONS == ("2.1.293", "2.1.294")
    assert VERSION in person_transcript.PINNED_VERSIONS


@pytest.mark.parametrize("prompt", ["yes", "ok, go ahead", "sounds good", "approved", "/allow release-main"])
def test_typed_words_record_a_statement_and_never_an_authorization(world, prompt):
    tmp, state, repo = world
    say(tmp, state, repo, prompt, before=(PROPOSAL,))
    assert [r["type"] for r in rows(state)][:1] == ["statement"] and authorized(state) == []


def test_a_typed_yes_to_a_release_proposal_is_told_to_use_the_approval_question(world):
    tmp, state, repo = world
    done = say(tmp, state, repo, "yes", before=(PROPOSAL,))
    context = json.loads(done.stdout)["hookSpecificOutput"]["additionalContext"]
    assert "AskUserQuestion" in context and tip(repo) in context and "Approve release of" in context
    assert authorized(state) == []


def test_the_statement_keeps_a_hash_a_short_redacted_excerpt_and_where_the_transcript_was(world):
    tmp, state, repo = world
    prompt = "please read api_key=abcd1234efgh5678 and then " + "think about this carefully " * 30
    say(tmp, state, repo, prompt)
    statement = rows(state)[0]
    text = (state / "person" / "statements.log").read_text()
    assert prompt not in text and "abcd1234efgh5678" not in text
    assert len(statement["prompt_sha256"]) == 64 and len(statement["excerpt"]) <= 80
    assert "<redacted>" in statement["excerpt"] and statement["version"] == 1
    assert statement["transcript_dir"] == str(project_dir(state))
    assert statement["claude_version"] == VERSION and statement["source"] == "harness-hook:UserPromptSubmit"


@pytest.mark.parametrize("entry", [peer, notification, local_command,
                                   functools.partial(human, version="9.9.9"),
                                   functools.partial(human, promptSource="system"),
                                   functools.partial(human, turnOrigin="peer"),
                                   functools.partial(human, origin={"kind": "peer"}),
                                   functools.partial(human, sessionId="other")])
def test_a_prompt_the_person_did_not_type_leaves_no_record(world, entry):
    tmp, state, repo = world
    done = say(tmp, state, repo, "yes", before=(PROPOSAL,), entry=entry)
    assert done.returncode == 0 and done.stdout == "" and rows(state) == []


def test_a_prompt_typed_while_a_turn_runs_counts_as_typed(world):
    tmp, state, repo = world
    say(tmp, state, repo, "hello", entry=functools.partial(human, promptSource="queued"))
    assert rows(state)[0]["type"] == "statement"


def test_a_missing_transcript_entry_fails_closed_after_a_bounded_wait(world):
    _tmp, state, repo = world
    path = transcript(project_dir(state) / f"{SESSION}.jsonl", PROPOSAL, human("something else", "other-id"))
    began = time.monotonic()
    done = run_hook(prompt_event("yes", "p-1", path, repo), state)
    assert done.returncode == 0 and rows(state) == []
    assert 1.5 < time.monotonic() - began < 15


def test_an_unreadable_transcript_and_a_mismatched_text_fail_closed(world):
    _tmp, state, repo = world
    assert run_hook(prompt_event("yes", "p-1", project_dir(state) / f"{SESSION}.jsonl", repo), state).returncode == 0
    path = transcript(project_dir(state) / f"{SESSION}.jsonl", human("no, something else entirely", "p-1"))
    assert run_hook(prompt_event("yes", "p-1", path, repo), state).returncode == 0
    assert rows(state) == []


def test_a_late_transcript_write_is_waited_for(world):
    _tmp, state, repo = world
    path = project_dir(state) / f"{SESSION}.jsonl"
    path.write_text("")
    threading.Timer(0.6, lambda: transcript(path, human("hello", "p-1"))).start()
    done = run_hook(prompt_event("hello", "p-1", path, repo), state)
    assert done.returncode == 0 and [r["type"] for r in rows(state)][:1] == ["statement"]


@pytest.mark.parametrize("change", [{"agent_id": "sub-1"}, {"hook_event_name": "PostToolUse"}, {"session_id": ""},
                                    {"session_id": "bad id\n"}])
def test_events_that_are_not_the_main_sessions_prompt_are_ignored(world, change):
    _tmp, state, repo = world
    path = transcript(project_dir(state) / f"{SESSION}.jsonl", human("hello", "p-1"))
    assert run_hook({**prompt_event("hello", "p-1", path, repo), **change}, state).returncode == 0
    assert rows(state) == []


@pytest.mark.parametrize("env", [{"ML_STACK_NONINTERACTIVE": "1"}, {"CLAUDE_CODE_SESSION_ATTENDED": "false"}])
def test_a_session_nobody_is_attending_records_nothing(world, env):
    _tmp, state, repo = world
    path = transcript(project_dir(state) / f"{SESSION}.jsonl", human("hello", "p-1"))
    assert run_hook(prompt_event("hello", "p-1", path, repo), state, **env).returncode == 0
    assert rows(state) == []


def test_garbage_on_stdin_never_blocks_the_session(world):
    import subprocess
    import sys

    from person_support import HOOKS, environment
    _tmp, state, _ = world
    done = subprocess.run([sys.executable, str(HOOKS / "claude-user-prompt")], input="{not json", text=True,
                          capture_output=True, timeout=30, check=False, env=environment(state))
    assert done.returncode == 0


def test_a_transcript_the_calling_agent_wrote_elsewhere_records_nothing(world, tmp_path_factory):
    """An agent can pipe an event at the hook; a transcript outside the Claude projects folder is not believed."""
    _tmp, state, repo = world
    outside = tmp_path_factory.mktemp("elsewhere") / f"{SESSION}.jsonl"
    transcript(outside, PROPOSAL, human("hello", "p-1"))
    assert run_hook(prompt_event("hello", "p-1", outside, repo), state).returncode == 0
    inside = project_dir(state) / f"{SESSION}.jsonl"
    transcript(inside, human("hello", "p-1"))
    link = project_dir(state) / "link.jsonl"
    link.symlink_to(outside)
    renamed = transcript(project_dir(state) / "not-the-session.jsonl", human("hello", "p-1"))
    for path in (link, renamed):
        assert run_hook(prompt_event("hello", "p-1", path, repo), state).returncode == 0
    assert run_hook(prompt_event("hello", "p-1", inside, repo), state, ML_STACK_SESSION_ID="other").returncode == 0
    assert rows(state) == []
    assert run_hook(prompt_event("hello", "p-1", inside, repo), state, ML_STACK_SESSION_ID=SESSION).returncode == 0
    assert [r["type"] for r in rows(state)][:1] == ["statement"]


def test_a_hundred_megabyte_transcript_is_read_from_its_tail_within_the_hook_budget(world):
    _tmp, state, repo = world
    path = project_dir(state) / f"{SESSION}.jsonl"
    filler = (json.dumps({"type": "attachment", "pad": "x" * 900}) + "\n").encode() * 1000
    with path.open("wb") as handle:
        for _ in range(120):
            handle.write(filler)
        handle.write((json.dumps(human("hello", "p-1")) + "\n").encode())
    assert path.stat().st_size > 100 * 1024 * 1024
    began = time.monotonic()
    done = run_hook(prompt_event("hello", "p-1", path, repo), state)
    assert done.returncode == 0 and time.monotonic() - began < 10
    assert [r["type"] for r in rows(state)][:1] == ["statement"]


def test_the_approval_answer_creates_one_release_main_authorization_bound_to_the_commit(world):
    tmp, state, repo = world
    done = approve(tmp, state, repo)
    assert "recorded for release of main to " + tip(repo) in done.stdout
    (auth,) = authorized(state)
    assert (auth["kind"], auth["how"], auth["uses"], auth["version"]) == ("release-main", "asked", 1, 1)
    assert auth["target"] == target(repo) and 0 < auth["expires"] - auth["ts"] <= 15 * 60 + 1
    assert [r["type"] for r in rows(state)].count("answer") == 1
    assert consume_now(state, repo) == auth["id"]


def test_the_question_is_made_of_what_git_reports(world):
    _tmp, state, repo = world
    question, yes, no = proposal(repo, state)
    sha = tip(repo)
    assert question.splitlines()[0] == f"Release main to {sha}?"
    assert "Subject: add release.txt" in question and "Commits ahead of origin/main: 1" in question
    assert "1 file changed" in question and "Remote: origin" in question
    assert question.splitlines()[-1].startswith(f"authorize release-main {target(repo)} 15m x1 #")
    assert (yes, no) == (f"Approve release of {sha[:7]}", "Do not release")


@pytest.mark.parametrize("variant", ["declined", "other-sha", "forged-tag", "extra-proposal", "echo-not-last",
                                     "claims-other-sha", "other-label", "subagent", "other-tool"])
def test_an_answer_that_is_not_the_exact_approval_of_what_git_shows_creates_nothing(world, variant):
    tmp, state, repo = world
    say(tmp, state, repo, "hello", prompt_id="bind")
    question, yes, no = proposal(repo, state)
    label, more = yes, {}
    first = tip(repo)
    if variant == "declined":
        label = no
    if variant == "other-sha":
        question = question.replace(first, tip(repo, "HEAD~1"))
    if variant == "forged-tag":
        question = question[:-3] + "000"
    if variant == "extra-proposal":
        question = question.replace("Remote: origin", "Remote: origin\nI'll also delete the cache.")
    if variant == "echo-not-last":
        question += "\nSincerely, the assistant"
    if variant == "claims-other-sha":
        question = question.replace("Release main to " + first, "Release main to " + "a" * 40)
    if variant == "other-label":
        label = "Approve release of 0000000"
    if variant == "subagent":
        more["agent_id"] = "sub"
    if variant == "other-tool":
        more["tool_name"] = "Bash"
    assert run_hook(answer_event(repo, question, label, **more), state).returncode == 0
    assert authorized(state) == []


def test_the_answer_to_a_question_whose_remote_moved_is_void(world):
    tmp, state, repo = world
    say(tmp, state, repo, "hello", prompt_id="bind")
    question, yes, _ = proposal(repo, state)
    other = repo.with_name("other-clone")
    git(repo, "clone", "-q", str(repo.with_name(repo.name + "-remote.git")), str(other))
    commit(other, "elsewhere.txt")
    git(other, "push", "-q", "origin", "main")
    git(repo, "fetch", "-q", "origin")
    assert run_hook(answer_event(repo, question, yes), state).returncode == 0
    assert authorized(state) == []


def test_a_release_named_in_prose_is_answered_with_the_hook_rendered_question(world):
    tmp, state, repo = world
    done = say(tmp, state, repo, "release main")
    context = json.loads(done.stdout)["hookSpecificOutput"]["additionalContext"]
    assert tip(repo) in context and "Approve release of " + tip(repo)[:7] in context
    assert authorized(state) == []


def test_the_single_use_replay_and_other_sessions_are_refused(world):
    tmp, state, repo = world
    approve(tmp, state, repo)
    with pytest.raises(NotAuthorized):
        consume_now(state, repo, session="another-session")
    consume_now(state, repo)
    with pytest.raises(NotAuthorized):
        consume_now(state, repo)


def test_a_named_subagent_acts_as_its_parent_and_the_use_names_it(world):
    tmp, state, repo = world
    approve(tmp, state, repo)
    assert consume_now(state, repo, agent="claude-ab12cd") == authorized(state)[0]["id"]
    assert rows(state)[-1]["by"] == "claude-ab12cd"


def test_expiry_is_fifteen_minutes(world):
    tmp, state, repo = world
    approve(tmp, state, repo)
    ident = Identity(SESSION)
    with pytest.raises(NotAuthorized):
        person_auth.consume("release-main", target(repo), ident, log=log_of(state), now=time.time() + 16 * 60)
    assert person_auth.consume("release-main", target(repo), ident, log=log_of(state), now=time.time() + 14 * 60)


def test_another_commit_another_repository_or_another_remote_does_not_match(world):
    tmp, state, repo = world
    approve(tmp, state, repo)
    other = repository(tmp / "second")
    wrong = [target(repo, tip(repo, "HEAD~1")), target(other, tip(other)), target(repo, remote="fork"),
             target(repo) + "0", ""]
    for text in wrong:
        with pytest.raises(NotAuthorized):
            person_auth.consume("release-main", text, Identity(SESSION), log=log_of(state))
    with pytest.raises(NotAuthorized):
        person_auth.consume("release-main", target(repo), Identity(""), log=log_of(state))


@pytest.mark.parametrize("word", ["stop", "cancel that", "/revoke", "don't push main yet"])
def test_stopping_words_revoke_a_live_authorization(world, word):
    tmp, state, repo = world
    approve(tmp, state, repo)
    say(tmp, state, repo, word, prompt_id="p-2")
    with pytest.raises(NotAuthorized):
        consume_now(state, repo)
    assert [r["state"] for r in rows(state) if r["type"] == "transition"] == ["revoked"]


@pytest.mark.parametrize("name", ["SessionEnd", "SessionStart"])
def test_a_session_boundary_closes_that_sessions_authorizations(world, name):
    tmp, state, repo = world
    approve(tmp, state, repo)
    run_hook({"hook_event_name": name, "session_id": "another-session"}, state)
    run_hook({"hook_event_name": name, "session_id": SESSION, "agent_id": "sub"}, state)
    assert person_store.authorizations(person_store.records(log_of(state)))[0].state == "live"
    run_hook({"hook_event_name": name, "session_id": SESSION, "cwd": str(repo)}, state)
    with pytest.raises(NotAuthorized):
        consume_now(state, repo)


def test_an_edited_record_makes_every_consumer_refuse(world):
    tmp, state, repo = world
    approve(tmp, state, repo)
    log = state / "person" / "statements.log"
    log.write_text(log.read_text().replace('"uses": 1', '"uses": 9'))
    with pytest.raises(NotAuthorized, match="unavailable"):
        consume_now(state, repo)


def test_a_cut_off_record_and_a_forged_row_are_refused(world):
    tmp, state, repo = world
    approve(tmp, state, repo)
    log = state / "person" / "statements.log"
    lines = log.read_text().splitlines()
    forged = json.loads(lines[-1])
    forged.update(id="forged", seq=99, hash="0" * 64)
    log.write_text("\n".join([*lines, json.dumps(forged)]) + "\n")
    with pytest.raises(NotAuthorized):
        consume_now(state, repo)
    log.write_text(lines[0] + "\n")
    with pytest.raises(NotAuthorized):
        consume_now(state, repo)


def test_the_session_comes_from_the_recorded_harness_process_and_not_from_the_environment(world, tmp_path_factory):
    tmp, state, repo = world
    approve(tmp, state, repo)
    sha = tip(repo)
    line = f"refs/heads/main {sha} refs/heads/main {tip(repo, 'origin/main')}\n"
    ok = person_consume(repo, state, "consume", "origin", stdin=line, ML_STACK_SESSION_ID="forged")
    assert ok.returncode == 0, ok.stderr
    forged_store = tmp_path_factory.mktemp("forged") / "state"
    assert person_consume(repo, forged_store, "consume", "origin", stdin=line,
                          ML_STACK_SESSION_ID=SESSION).returncode == 1


def test_revoking_and_consuming_take_the_same_lock(world):
    tmp, state, repo = world
    approve(tmp, state, repo)
    log = log_of(state)
    finished = []
    with person_store.consume_lock(log):
        thread = threading.Thread(target=lambda: finished.append(
            person_auth_revoke(log)))
        thread.start()
        time.sleep(0.4)
        assert finished == []
    thread.join(10)
    assert len(finished[0]) == 1
    with pytest.raises(NotAuthorized):
        consume_now(state, repo)


def person_auth_revoke(log):
    from ml_stack.workspace import person_record
    return person_record.revoke_session(log, SESSION, "test")


def test_a_use_never_follows_a_revocation_under_contention(world):
    tmp, state, repo = world
    from ml_stack.workspace import person_record
    approve(tmp, state, repo)
    log = log_of(state)
    statement = next(r for r in rows(state) if r["type"] == "statement")
    grant = person_record.Grant("release-main", target(repo), "asked")
    for _ in range(8):
        person_record.record_authorization(log, statement, grant)
    results = []

    def use():
        try:
            results.append(person_auth.consume("release-main", target(repo), Identity(SESSION), log=log))
        except NotAuthorized:
            results.append(None)

    threads = [threading.Thread(target=use) for _ in range(6)]
    revoker = threading.Thread(target=lambda: person_record.revoke_session(log, SESSION, "race"))
    for t in threads[:3]:
        t.start()
    revoker.start()
    for t in threads[3:]:
        t.start()
    for t in (*threads, revoker):
        t.join(30)
    seen_revoked = set()
    for row in rows(state):
        if row.get("state") == "revoked":
            seen_revoked.add(row["auth_id"])
        assert not (row.get("state") == "used" and row["auth_id"] in seen_revoked)


def test_the_hooks_view_shows_an_answer_as_quoted_model_text(world, monkeypatch):
    tmp, state, repo = world
    approve(tmp, state, repo)
    monkeypatch.setenv("ML_STACK_HOME", str(state))
    from ml_stack.workspace import person_view
    answer = next(r for r in person_view.listing() if r["record"] == "answer")
    assert "quoted_model_text" in answer["detail"] and "excerpt" not in answer["detail"]
    assert os.environ["ML_STACK_HOME"] == str(state)
