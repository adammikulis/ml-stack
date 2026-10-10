mod kit;

use kit::*;
use poolhouse_node::node::Node;
use serde_json::{json, Value};
use tempfile::tempdir;

fn sha(n: u8) -> String {
    format!("{n:02x}{}", "abcdef0123456789abcdef0123456789abcdef"[..38].to_string())
}

fn ask(n: &mut Node, token: &str, branch: &str, at: &str) -> Value {
    req(n, "land_request", "demo", token, json!({"branch": branch, "sha": at, "target": "dev", "selectors": ["tests/test_x.py"], "replaces": "nothing"}))
}

fn queue(n: &mut Node, token: &str) -> Value {
    ok(req(n, "land_queue", "demo", token, json!({})))
}

fn status_of(n: &mut Node, token: &str, id: &str) -> String {
    let q = queue(n, token);
    q["requests"].as_array().unwrap().iter().find(|r| r["id"] == id).unwrap()["status"].as_str().unwrap().into()
}

fn child_of(n: &mut Node, parent: &str, id: &str) -> (String, String) {
    let r = ok(req(n, "register", "demo", parent, json!({"model": "claude-sonnet-5-5", "harness": "claude-code", "session": id})));
    (r["name"].as_str().unwrap().into(), r["token"].as_str().unwrap().into())
}

fn id_of(made: Value) -> String {
    ok(made)["id"].as_str().unwrap().to_string()
}

#[test]
fn a_request_is_stamped_with_its_sender_and_waits_for_an_independent_review_of_its_exact_commit() {
    let dir = tempdir().unwrap();
    let mut n = node(dir.path());
    let ((alice, at), (_, bt)) = (session(&mut n, "demo", "a"), session(&mut n, "demo", "b"));
    let tip = sha(1);
    let id = id_of(ask(&mut n, &at, "feat", &tip));
    let listed = queue(&mut n, &bt);
    let r = &listed["requests"][0];
    assert_eq!((r["by"].clone(), r["status"].clone(), r["branch"].clone(), r["target"].clone()), (json!(alice), json!("needs-review"), json!("feat"), json!("dev")));
    assert_eq!(code(&req(&mut n, "land_review", "demo", &at, json!({"req": id, "sha": tip, "verdict": "accept"}))), "denied", "the author does not review its own request");
    assert_eq!(code(&req(&mut n, "land_review", "demo", &bt, json!({"req": id, "sha": sha(2), "verdict": "accept"}))), "invalid", "a review names the exact commit");
    assert_eq!(code(&req(&mut n, "land_review", "demo", &bt, json!({"req": id, "sha": tip, "verdict": "maybe"}))), "invalid");
    ok(req(&mut n, "land_review", "demo", &bt, json!({"req": id, "sha": tip, "verdict": "accept"})));
    assert_eq!(status_of(&mut n, &at, &id), "queued");
    assert_eq!(queue(&mut n, &at)["requests"][0]["reviews"].as_object().unwrap().len(), 1);
}

#[test]
fn a_parent_and_its_child_are_not_independent_and_one_rejection_blocks() {
    let dir = tempdir().unwrap();
    let mut n = node(dir.path());
    let ((_, lead), (_, other), (_, third)) = (session(&mut n, "demo", "lead"), session(&mut n, "demo", "o"), session(&mut n, "demo", "t"));
    let (_, kid) = child_of(&mut n, &lead, "kid");
    let tip = sha(3);
    let id = id_of(ask(&mut n, &kid, "kidwork", &tip));
    assert_eq!(code(&req(&mut n, "land_review", "demo", &lead, json!({"req": id, "sha": tip, "verdict": "accept"}))), "denied", "its parent is not independent");
    ok(req(&mut n, "land_review", "demo", &other, json!({"req": id, "sha": tip, "verdict": "accept"})));
    ok(req(&mut n, "land_review", "demo", &third, json!({"req": id, "sha": tip, "verdict": "reject"})));
    assert_eq!(status_of(&mut n, &kid, &id), "needs-review");
    ok(req(&mut n, "land_review", "demo", &third, json!({"req": id, "sha": tip, "verdict": "accept"})));
    assert_eq!(status_of(&mut n, &kid, &id), "queued", "a later review of one reviewer replaces its earlier one");
}

#[test]
fn a_new_request_for_a_branch_supersedes_the_old_one_and_main_is_never_a_branch_or_a_target() {
    let dir = tempdir().unwrap();
    let mut n = node(dir.path());
    let (_, at) = session(&mut n, "demo", "a");
    let old = id_of(ask(&mut n, &at, "s", &sha(1)));
    let new = id_of(ask(&mut n, &at, "s", &sha(2)));
    assert_eq!((status_of(&mut n, &at, &old), status_of(&mut n, &at, &new)), ("superseded".into(), "needs-review".into()));
    for (branch, target, at_sha) in [("main", "dev", sha(1)), ("master", "dev", sha(1)), ("s", "main", sha(1)), ("s", "dev", "abc".to_string()), ("../x", "dev", sha(1))] {
        let r = req(&mut n, "land_request", "demo", &at, json!({"branch": branch, "sha": at_sha, "target": target, "selectors": ["t"]}));
        assert_eq!(code(&r), "invalid", "{branch} {target} {at_sha}");
    }
    for selectors in [json!([]), json!("t"), json!(["a\nb"])] {
        assert_eq!(code(&req(&mut n, "land_request", "demo", &at, json!({"branch": "s", "sha": sha(1), "target": "dev", "selectors": selectors}))), "invalid");
    }
}

#[test]
fn only_the_holder_of_the_branch_claim_records_states_and_a_landed_request_does_not_move_again() {
    let dir = tempdir().unwrap();
    let mut n = node(dir.path());
    let ((_, at), (_, rt), (_, ot)) = (session(&mut n, "demo", "a"), session(&mut n, "demo", "runner"), session(&mut n, "demo", "o"));
    let tip = sha(4);
    let id = id_of(ask(&mut n, &at, "f", &tip));
    ok(req(&mut n, "land_review", "demo", &ot, json!({"req": id, "sha": tip, "verdict": "accept"})));
    let state = |n: &mut Node, token: &str, status: &str| req(n, "land_state", "demo", token, json!({"req": id, "status": status, "detail": "d", "evidence": {"commit": "abc", "seen": true}}));
    assert_eq!(code(&state(&mut n, &rt, "running")), "denied", "no claim, no state");
    assert_eq!(code(&req(&mut n, "land_beat", "demo", &rt, json!({"what": "x"}))), "denied");
    ok(req(&mut n, "claim", "demo", &rt, json!({"kind": "branch", "key": "dev"})));
    assert_eq!(code(&state(&mut n, &at, "running")), "denied", "the requester is not the runner");
    ok(state(&mut n, &rt, "running"));
    assert_eq!(status_of(&mut n, &at, &id), "running");
    ok(req(&mut n, "land_beat", "demo", &rt, json!({"what": "gate step 3"})));
    assert_eq!(queue(&mut n, &at)["beat"]["what"], "gate step 3");
    assert_eq!(code(&state(&mut n, &rt, "bogus")), "invalid");
    ok(state(&mut n, &rt, "landed-unpushed"));
    ok(state(&mut n, &rt, "landed"));
    ok(state(&mut n, &rt, "queued"));
    let after = queue(&mut n, &at)["requests"][0].clone();
    assert_eq!((after["status"].clone(), after["evidence"]["commit"].clone()), (json!("landed"), json!("abc")));
}

#[test]
fn cancel_is_for_the_requester_the_runner_or_a_parentless_session_and_a_helper_cannot_pause() {
    let dir = tempdir().unwrap();
    let mut n = node(dir.path());
    let ((_, at), (_, bt), (_, lead)) = (session(&mut n, "demo", "a"), session(&mut n, "demo", "b"), session(&mut n, "demo", "lead"));
    let (_, kid) = child_of(&mut n, &lead, "kid");
    let id = id_of(ask(&mut n, &at, "c", &sha(5)));
    assert_eq!(code(&req(&mut n, "land_cancel", "demo", &kid, json!({"req": id}))), "denied", "a subagent cancels only its own");
    assert_eq!(code(&req(&mut n, "land_brake", "demo", &kid, json!({"pause": true}))), "denied");
    ok(req(&mut n, "land_brake", "demo", &bt, json!({"pause": true, "reason": "checking"})));
    assert_eq!(queue(&mut n, &at)["paused"], true);
    ok(req(&mut n, "land_brake", "demo", &lead, json!({"pause": false})));
    assert_eq!(queue(&mut n, &at)["paused"], false);
    ok(req(&mut n, "land_cancel", "demo", &at, json!({"req": id})));
    assert_eq!(status_of(&mut n, &at, &id), "cancelled");
    assert_eq!(code(&req(&mut n, "land_cancel", "demo", &at, json!({"req": id}))), "invalid", "a closed request cannot be cancelled again");
}

#[test]
fn a_retired_session_and_a_session_of_another_board_cannot_use_the_queue() {
    let dir = tempdir().unwrap();
    let mut n = node(dir.path());
    let (_, lead) = session(&mut n, "demo", "lead");
    let (name, kid) = child_of(&mut n, &lead, "kid");
    let (_, elsewhere) = session(&mut n, "other", "x");
    assert_eq!(code(&req(&mut n, "land_queue", "demo", &elsewhere, json!({}))), "denied");
    ok(req(&mut n, "retire", "demo", &lead, json!({"target": name})));
    assert_eq!(code(&ask(&mut n, &kid, "z", &sha(6))), "denied");
}
