mod kit;

use kit::*;
use poolside_node::sync::exchange_nodes;
use serde_json::{json, Value};
use tempfile::tempdir;

fn dm(n: &mut poolside_node::node::Node, token: &str, to: &str, text: &str) -> Value {
    req(n, "post", "demo", token, json!({"kind": "message", "fields": {"type": "note", "to": to, "body": text}}))
}

fn bodies(reply: &Value) -> Vec<String> {
    reply["entries"].as_array().unwrap().iter().filter(|e| e["kind"] == "message").map(|e| e["fields"]["body"].as_str().unwrap().to_string()).collect()
}

fn sub(n: &mut poolside_node::node::Node, parent_token: &str, id: &str) -> (String, String) {
    let r = ok(req(n, "register", "demo", parent_token, json!({"model": "claude-haiku-5-5", "harness": "claude-code", "session": id})));
    (r["name"].as_str().unwrap().into(), r["token"].as_str().unwrap().into())
}

#[test]
fn a_direct_message_is_read_only_by_its_sender_and_its_recipient() {
    let dir = tempdir().unwrap();
    let mut n = node(dir.path());
    let ((alice, at), (bob, bt), (_, ct)) = (session(&mut n, "demo", "a"), session(&mut n, "demo", "b"), session(&mut n, "demo", "c"));
    ok(dm(&mut n, &at, &bob, "for bob only"));
    ok(post(&mut n, "demo", &at, "for everyone"));
    for (who, token, sees) in [("alice", &at, 2), ("bob", &bt, 2), ("carol", &ct, 1)] {
        assert_eq!(bodies(&ok(req(&mut n, "read", "demo", token, json!({})))).len(), sees, "{who} reads the board");
    }
    assert!(bodies(&ok(req(&mut n, "read", "demo", &ct, json!({"channel": "#dm"})))).is_empty(), "a third party cannot ask for the channel");
    assert!(bodies(&ok(req(&mut n, "read", "demo", &ct, json!({"with": alice, "inbox": true})))).is_empty());
    assert!(bodies(&ok(req(&mut n, "read", "demo", &ct, json!({"by": alice})))).iter().all(|b| b != "for bob only"));
    assert_eq!(bodies(&ok(req(&mut n, "read", "demo", &bt, json!({"inbox": true})))), vec!["for bob only"]);
    assert_eq!(bodies(&ok(req(&mut n, "read", "demo", &bt, json!({"with": alice})))), vec!["for bob only"]);
    assert!(bodies(&ok(req(&mut n, "read", "demo", &at, json!({"inbox": true})))).is_empty(), "the sender's inbox is what others wrote to it");
}

#[test]
fn a_link_that_shares_the_whole_board_does_not_share_direct_messages() {
    let dir = tempdir().unwrap();
    let mut n = node(dir.path());
    let ((_, at), (bob, _)) = (session(&mut n, "demo", "a"), session(&mut n, "demo", "b"));
    let (_, ot) = session(&mut n, "other", "o");
    ok(dm(&mut n, &at, &bob, "private"));
    ok(post(&mut n, "demo", &at, "public"));
    ok(req(&mut n, "link", "demo", &at, json!({"to": "other", "channels": [], "mode": "ro"})));
    assert_eq!(bodies(&ok(req(&mut n, "read", "demo", &ot, json!({})))), vec!["public"]);
}

#[test]
fn a_message_to_a_name_nobody_holds_or_a_forged_recipient_is_refused() {
    let dir = tempdir().unwrap();
    let mut n = node(dir.path());
    let ((_, at), (bob, _)) = (session(&mut n, "demo", "a"), session(&mut n, "demo", "b"));
    for bad in ["claude-000000", "#private", "*", "all", "human", "../x", "Bob", ""] {
        let reply = dm(&mut n, &at, bad, "x");
        assert!(!reply["ok"].as_bool().unwrap(), "{bad}: {reply}");
    }
    let (_, other) = session(&mut n, "other", "x");
    assert_eq!(code(&dm(&mut n, &other, &bob, "from the wrong board")), "denied", "a session of another board has no way to a name here");
    let forged = req(&mut n, "post", "demo", &at, json!({"kind": "message", "fields": {"type": "note", "to": bob, "body": "x", "from": bob}}));
    assert_eq!(code(&forged), "denied");
    assert_eq!(code(&req(&mut n, "post", "demo", &at, json!({"kind": "message", "fields": {"type": "milestone", "to": bob, "body": "x"}}))), "invalid", "announcement types stay on #announcements");
}

#[test]
fn a_direct_message_crosses_devices_and_is_read_by_the_recipient_only() {
    let (da, db) = (tempdir().unwrap(), tempdir().unwrap());
    let (mut a, mut b) = (node(da.path()), node(db.path()));
    let ((_, at), (bob, bt)) = (session(&mut a, "demo", "alice"), session(&mut b, "demo", "bob"));
    let (_, ct) = session(&mut b, "demo", "carol");
    assert_eq!(code(&dm(&mut a, &at, &bob, "before bob is known here")), "denied");
    exchange_nodes(&mut a, &mut b).unwrap();
    ok(dm(&mut a, &at, &bob, "across"));
    exchange_nodes(&mut a, &mut b).unwrap();
    let got = ok(req(&mut b, "read", "demo", &bt, json!({"inbox": true})));
    assert_eq!(bodies(&got), vec!["across"]);
    assert_eq!(got["entries"][0]["foreign"], true);
    assert!(bodies(&ok(req(&mut b, "read", "demo", &ct, json!({})))).is_empty());
}

#[test]
fn a_forged_foreign_direct_message_cannot_name_a_local_sender() {
    let (da, db) = (tempdir().unwrap(), tempdir().unwrap());
    let (mut a, mut b) = (node(da.path()), node(db.path()));
    let (victim, _) = session(&mut b, "demo", "victim");
    let (rname, vt) = session(&mut b, "demo", "recipient");
    let evil = a.host("demo").unwrap();
    let body = json!({"type": "note", "from": victim, "to": rname, "subject": "", "body": "I am the victim"});
    evil.board.append(poolside_node::row::Kind::Message, &victim, body, "").unwrap();
    exchange_nodes(&mut a, &mut b).unwrap();
    assert!(bodies(&ok(req(&mut b, "read", "demo", &vt, json!({"inbox": true})))).is_empty(), "a sender named like a local session is refused");
}

#[test]
fn the_cursor_a_read_returns_gives_each_unread_message_exactly_once() {
    let dir = tempdir().unwrap();
    let mut n = node(dir.path());
    let ((_, at), (bob, bt)) = (session(&mut n, "demo", "a"), session(&mut n, "demo", "b"));
    ok(dm(&mut n, &at, &bob, "one"));
    let first = ok(req(&mut n, "read", "demo", &bt, json!({"inbox": true})));
    assert_eq!(bodies(&first), vec!["one"]);
    assert!(bodies(&ok(req(&mut n, "read", "demo", &bt, json!({"inbox": true, "since": first["cursor"]})))).is_empty());
    ok(dm(&mut n, &at, &bob, "two"));
    ok(post(&mut n, "demo", &at, "unrelated"));
    ok(dm(&mut n, &at, &bob, "three"));
    let next = ok(req(&mut n, "read", "demo", &bt, json!({"inbox": true, "since": first["cursor"], "limit": 1})));
    assert_eq!(bodies(&next), vec!["two"]);
    let rest = ok(req(&mut n, "read", "demo", &bt, json!({"inbox": true, "since": next["cursor"]})));
    assert_eq!(bodies(&rest), vec!["three"]);
    // a cursor kept for one question is not a cursor for another: the board read from the start still finds everything
    assert_eq!(bodies(&ok(req(&mut n, "read", "demo", &bt, json!({})))).len(), 4);
}

#[test]
fn a_retired_subagent_loses_its_token_its_leases_and_its_name() {
    let dir = tempdir().unwrap();
    let mut n = node(dir.path());
    let (lead, lt) = session(&mut n, "demo", "lead");
    let (child, ct) = sub(&mut n, &lt, "child");
    ok(req(&mut n, "claim", "demo", &ct, json!({"kind": "branch", "key": "work"})));
    ok(req(&mut n, "lease_acquire", "demo", &ct, json!({"resources": [{"type": "gpu"}]})));
    let done = ok(req(&mut n, "retire", "demo", &ct, json!({})));
    assert_eq!((done["name"].as_str().unwrap(), done["leases_released"].clone()), (child.as_str(), json!(2)));
    assert_eq!(code(&req(&mut n, "whoami", "demo", &ct, json!({}))), "denied", "its token stopped working");
    assert_eq!(code(&post(&mut n, "demo", &ct, "late")), "denied");
    assert_eq!(ok(req(&mut n, "claims", "demo", &lt, json!({})))["claims"].as_array().unwrap().len(), 0);
    assert_eq!(ok(req(&mut n, "claim", "demo", &lt, json!({"kind": "branch", "key": "work"})))["claim"]["owner"], lead, "the claim is free");
    let again = req(&mut n, "register", "demo", &lt, json!({"model": "claude-haiku-5-5", "harness": "claude-code", "session": "child"}));
    assert_eq!(code(&again), "denied", "the retired session cannot register itself back");
    let listed = |n: &mut poolside_node::node::Node, all: bool| ok(req(n, "agents", "demo", &lt, json!({"retired": all})))["agents"].as_array().unwrap().len();
    assert_eq!((listed(&mut n, false), listed(&mut n, true)), (1, 2));
    let audit = ok(req(&mut n, "read", "demo", &lt, json!({"kind": "audit"})));
    let event = audit["entries"].as_array().unwrap().iter().find(|e| e["fields"]["event"] == "retire").expect("retire is audited");
    assert_eq!(event["fields"]["subject"], child.as_str());
    let identity = ok(req(&mut n, "read", "demo", &lt, json!({"kind": "identity"})));
    assert_eq!(identity["entries"].as_array().unwrap().last().unwrap()["fields"]["retired"], true);
}

#[test]
fn only_a_parent_retires_a_child_and_a_main_session_does_not_retire_itself() {
    let dir = tempdir().unwrap();
    let mut n = node(dir.path());
    let ((lead, lt), (_, st)) = (session(&mut n, "demo", "lead"), session(&mut n, "demo", "stranger"));
    let (mid, mt) = sub(&mut n, &lt, "mid");
    let (leaf, _) = sub(&mut n, &mt, "leaf");
    assert_eq!(code(&req(&mut n, "retire", "demo", &lt, json!({}))), "denied", "a main session ends with its harness");
    assert_eq!(code(&req(&mut n, "retire", "demo", &st, json!({"target": leaf}))), "denied", "a stranger");
    assert_eq!(code(&req(&mut n, "retire", "demo", &mt, json!({"target": lead}))), "denied", "a child cannot retire its parent");
    assert_eq!(code(&req(&mut n, "retire", "demo", &mt, json!({"target": "claude-nobody"}))), "denied");
    let (_, other) = session(&mut n, "other", "x");
    assert_eq!(code(&req(&mut n, "retire", "demo", &other, json!({"target": mid}))), "denied", "another board");
    let done = ok(req(&mut n, "retire", "demo", &lt, json!({"target": leaf})));
    assert_eq!((done["by"].as_str().unwrap(), done["retired"].clone()), (lead.as_str(), json!(true)));
    assert_eq!(code(&req(&mut n, "retire", "demo", &lt, json!({"target": leaf}))), "denied", "already retired");
    assert_eq!(ok(req(&mut n, "retire", "demo", &lt, json!({"target": mid})))["name"], mid.as_str());
}

#[test]
fn a_session_registered_under_one_parent_is_not_taken_by_another() {
    let dir = tempdir().unwrap();
    let mut n = node(dir.path());
    let ((_, at), (_, bt)) = (session(&mut n, "demo", "a"), session(&mut n, "demo", "b"));
    let (child, _) = sub(&mut n, &at, "kid");
    let take = req(&mut n, "register", "demo", &bt, json!({"model": "claude-haiku-5-5", "harness": "claude-code", "session": "kid"}));
    assert_eq!(code(&take), "denied");
    let orphan = req(&mut n, "register", "demo", "", json!({"model": "claude-haiku-5-5", "harness": "claude-code", "session": "kid"}));
    assert_eq!(code(&orphan), "denied", "{child} has a parent");
    assert_eq!(sub(&mut n, &at, "kid").0, child, "its own parent gets the same name back");
}

#[test]
fn models_are_claimed_listed_and_a_verified_one_is_not_overwritten() {
    let dir = tempdir().unwrap();
    let mut n = node(dir.path());
    let (name, token) = session(&mut n, "demo", "a");
    let me = |n: &mut poolside_node::node::Node| ok(req(n, "agents", "demo", &token, json!({})))["agents"][0].clone();
    assert_eq!((me(&mut n)["model"].clone(), me(&mut n)["model_state"].clone(), me(&mut n)["harness"].clone()), (json!("claude-sonnet-5-5"), json!("claimed"), json!("claude-code")));
    let who = ok(req(&mut n, "whoami", "demo", &token, json!({"model": "claude-opus-5-5", "harness": "claude-code"})));
    assert_eq!((who["name"].as_str().unwrap(), who["identity"]["model"].clone()), (name.as_str(), json!("claude-opus-5-5")));
    assert_eq!(me(&mut n)["model"], "claude-opus-5-5");
    n.host("demo").unwrap().names.verify_model(&name, "claude-opus-5-5").unwrap();
    assert_eq!(me(&mut n)["model_state"], "verified");
    assert_eq!(code(&req(&mut n, "whoami", "demo", &token, json!({"model": "claude-haiku-5-5"}))), "denied", "a claim cannot lower a verified model");
    assert_eq!(ok(req(&mut n, "whoami", "demo", &token, json!({"model": "claude-opus-5-5"})))["identity"]["model_state"], "verified");
    assert_eq!(code(&req(&mut n, "whoami", "demo", &token, json!({"model": "x\ny"}))), "invalid");
    let entries = ok(req(&mut n, "read", "demo", &token, json!({"kind": "identity"})));
    assert!(entries["entries"].as_array().unwrap().iter().any(|e| e["fields"]["model"] == "claude-opus-5-5"), "the claim is on the board");
}

#[test]
fn a_subagent_that_names_no_model_inherits_its_parents_and_a_claim_replaces_it() {
    let dir = tempdir().unwrap();
    let mut n = node(dir.path());
    let (_, lt) = session(&mut n, "demo", "lead");
    let made = ok(req(&mut n, "register", "demo", &lt, json!({"model": "", "harness": "claude-code", "session": "kid"})));
    assert!(made["name"].as_str().unwrap().starts_with("claude-"), "the family comes from the parent's model: {made}");
    let kid = made["token"].as_str().unwrap().to_string();
    let listed = |n: &mut poolside_node::node::Node| ok(req(n, "agents", "demo", &lt, json!({})))["agents"].as_array().unwrap().iter().find(|a| a["parent"] != "").cloned().unwrap();
    assert_eq!((listed(&mut n)["model"].clone(), listed(&mut n)["model_state"].clone()), (json!("claude-sonnet-5-5"), json!("inherited")));
    ok(req(&mut n, "whoami", "demo", &kid, json!({"model": "claude-haiku-5-5"})));
    assert_eq!((listed(&mut n)["model"].clone(), listed(&mut n)["model_state"].clone()), (json!("claude-haiku-5-5"), json!("claimed")));
    let lone = ok(req(&mut n, "register", "demo", "", json!({"model": "", "harness": "claude-code", "session": "nomodel"})));
    assert!(lone["name"].as_str().unwrap().starts_with("agent-"), "no parent, no model: {lone}");
}

#[test]
fn agents_show_parent_last_seen_and_a_session_lookup_finds_a_name_without_a_token() {
    let dir = tempdir().unwrap();
    let mut n = node(dir.path());
    let (lead, lt) = session(&mut n, "demo", "lead");
    let (child, _) = sub(&mut n, &lt, "kid");
    let list = ok(req(&mut n, "agents", "demo", &lt, json!({})));
    let kid = list["agents"].as_array().unwrap().iter().find(|a| a["name"] == child.as_str()).unwrap();
    assert_eq!((kid["parent"].as_str().unwrap(), kid["retired"].clone()), (lead.as_str(), json!(false)));
    assert!(kid["seen_ms"].as_u64().unwrap() > 0);
    let found = ok(req(&mut n, "session_lookup", "demo", "", json!({"harness": "claude-code", "session": "kid"})));
    assert_eq!(found["name"], child.as_str());
    assert_eq!(code(&req(&mut n, "session_lookup", "demo", "", json!({"harness": "claude-code", "session": "nobody"}))), "denied");
    assert_eq!(code(&req(&mut n, "session_lookup", "nope", "", json!({"harness": "claude-code", "session": "kid"}))), "denied");
    assert_eq!(code(&req(&mut n, "agents", "demo", "not-a-token", json!({}))), "denied");
}
