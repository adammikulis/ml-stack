mod kit;

use kit::*;
use poolhouse_node::grants::Grants;
use poolhouse_node::identity::Holder;
use poolhouse_node::node::Node;
use serde_json::{json, Value};
use tempfile::tempdir;

const SHA: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

fn note(n: &mut Node, token: &str, extra: Value) -> Value {
    let mut fields = json!({"nkind": "fact", "title": "the port", "body": "the broker listens on a unix socket"});
    for (k, v) in extra.as_object().unwrap() {
        fields[k] = v.clone();
    }
    req(n, "post", "demo", token, json!({"kind": "note", "fields": fields}))
}

fn notes(n: &mut Node, token: &str, params: Value) -> Vec<Value> {
    ok(req(n, "notes", "demo", token, params))["notes"].as_array().unwrap().clone()
}

#[test]
fn a_note_carries_its_kind_title_source_tags_and_command_and_reads_back_with_a_short_ref() {
    let dir = tempdir().unwrap();
    let mut n = node(dir.path());
    let (author, token) = session(&mut n, "demo", "a");
    for kind in ["decision", "rule", "fact", "question"] {
        ok(note(&mut n, &token, json!({"nkind": kind, "title": format!("a {kind}")})));
    }
    assert_eq!(code(&note(&mut n, &token, json!({"nkind": "opinion"}))), "invalid");
    ok(note(&mut n, &token, json!({"title": "full", "source": "docs/node.md", "tags": ["socket", "broker"], "verify_cmd": "true", "ttl_days": 30})));
    let all = notes(&mut n, &token, json!({"limit": 50}));
    assert_eq!(all.len(), 5);
    let full = all.iter().find(|x| x["title"] == "full").unwrap();
    assert_eq!((full["author"].as_str().unwrap(), full["trust"].clone(), full["binding"].clone()), (author.as_str(), json!("agent-claimed"), json!(false)));
    assert_eq!((full["source"].clone(), full["tags"].clone(), full["verify_cmd"].clone(), full["ttl_days"].clone()), (json!("docs/node.md"), json!(["socket", "broker"]), json!("true"), json!(30)));
    let short = full["ref"].as_str().unwrap().to_string();
    assert!(short.bytes().all(|b| b.is_ascii_digit()), "a note of this device is named by its number");
    assert_eq!(notes(&mut n, &token, json!({"ref": short}))[0]["id"], full["id"]);
    assert_eq!(notes(&mut n, &token, json!({"ref": full["id"]}))[0]["title"], "full");
    assert_eq!(notes(&mut n, &token, json!({"kind": "rule"})).len(), 1);
    for bad in [json!({"ttl_days": 99999}), json!({"ttl_days": -1}), json!({"tags": ["a\nb"]}), json!({"verify_cmd": "a\nb"}), json!({"supersedes": [1]})] {
        assert!(!note(&mut n, &token, bad.clone())["ok"].as_bool().unwrap(), "{bad}");
    }
}

#[test]
fn search_finds_notes_holding_every_word_first_and_skips_superseded_ones() {
    let dir = tempdir().unwrap();
    let mut n = node(dir.path());
    let (_, token) = session(&mut n, "demo", "a");
    ok(note(&mut n, &token, json!({"title": "socket", "body": "unix"})));
    ok(note(&mut n, &token, json!({"title": "broker socket", "body": "unix socket for the broker"})));
    ok(note(&mut n, &token, json!({"title": "other", "body": "nothing"})));
    let found = notes(&mut n, &token, json!({"query": "broker socket"}));
    assert_eq!(found[0]["title"], "broker socket", "both words beat one");
    assert_eq!(found.len(), 2);
    assert!(notes(&mut n, &token, json!({"query": "absent"})).is_empty());
    let old = found[1]["ref"].clone();
    ok(note(&mut n, &token, json!({"title": "socket v2", "body": "unix socket", "supersedes": [old]})));
    assert!(notes(&mut n, &token, json!({"query": "socket"})).iter().all(|x| x["title"] != "socket"));
    let with_old = notes(&mut n, &token, json!({"query": "socket", "all": true}));
    let replaced = with_old.iter().find(|x| x["title"] == "socket").unwrap();
    assert!(!replaced["superseded_by"].as_str().unwrap().is_empty());
    assert_eq!(code(&note(&mut n, &token, json!({"supersedes": ["99999"]}))), "invalid", "a note that is not there cannot be superseded");
}

#[test]
fn a_verification_stores_the_notes_own_command_and_makes_it_test_verified() {
    let dir = tempdir().unwrap();
    let mut n = node(dir.path());
    let (checker, token) = session(&mut n, "demo", "a");
    let (_, liar) = session(&mut n, "demo", "b");
    let id = ok(note(&mut n, &token, json!({"verify_cmd": "ls /tmp"})))["seq"].to_string();
    assert_eq!(code(&req(&mut n, "note_verify", "demo", &token, json!({"note": "999", "exit": 0, "out_sha": SHA}))), "invalid");
    assert_eq!(code(&req(&mut n, "note_verify", "demo", &token, json!({"note": id, "exit": 0, "out_sha": "short"}))), "invalid");
    assert_eq!(code(&req(&mut n, "note_verify", "demo", &token, json!({"note": id, "exit": 0, "out_sha": SHA, "cmd": "rm -rf /"}))), "invalid", "the caller does not name the command");
    let failed = ok(req(&mut n, "note_verify", "demo", &token, json!({"note": id, "exit": 1, "out_sha": SHA})))["notes"][0].clone();
    assert_eq!((failed["trust"].clone(), failed["last_verified"]["exit"].clone()), (json!("agent-claimed"), json!(1)));
    let passed = ok(req(&mut n, "note_verify", "demo", &liar, json!({"note": id, "exit": 0, "out_sha": SHA})))["notes"][0].clone();
    assert_eq!((passed["trust"].clone(), passed["last_verified"]["cmd"].clone()), (json!("test-verified"), Value::Null));
    assert_eq!(passed["verify_cmd"], "ls /tmp");
    assert_eq!(passed["last_verified"]["output_sha256"], SHA);
    assert_ne!(passed["last_verified"]["by"], json!(checker), "the entry names whoever recorded it");
    let bare = ok(note(&mut n, &token, json!({"title": "no command"})))["seq"].to_string();
    assert_eq!(code(&req(&mut n, "note_verify", "demo", &token, json!({"note": bare, "exit": 0, "out_sha": SHA}))), "invalid");
    let entries = ok(req(&mut n, "read", "demo", &token, json!({"kind": "verify"})));
    assert_eq!(entries["entries"].as_array().unwrap().len(), 2, "each result is its own board entry");
    assert_eq!(entries["entries"][0]["fields"]["cmd"], "ls /tmp");
}

#[test]
fn a_verified_note_is_not_replaced_by_a_claimed_one() {
    let dir = tempdir().unwrap();
    let mut n = node(dir.path());
    let (_, token) = session(&mut n, "demo", "a");
    let id = ok(note(&mut n, &token, json!({"verify_cmd": "true"})))["seq"].to_string();
    ok(req(&mut n, "note_verify", "demo", &token, json!({"note": id, "exit": 0, "out_sha": SHA})));
    assert_eq!(code(&note(&mut n, &token, json!({"title": "better", "supersedes": [id]}))), "denied");
    ok(req(&mut n, "note_verify", "demo", &token, json!({"note": id, "exit": 2, "out_sha": SHA})));
    ok(note(&mut n, &token, json!({"title": "better", "supersedes": [id]})));
}

struct Nobody;
impl Grants for Nobody {
    fn allows(&self, _: &Holder, action: &str) -> bool {
        action != "note_verify"
    }
}

#[test]
fn recording_a_verification_is_a_grant() {
    let dir = tempdir().unwrap();
    let mut n = node(dir.path());
    n.grants = Box::new(Nobody);
    let (_, token) = session(&mut n, "demo", "a");
    let id = ok(note(&mut n, &token, json!({"verify_cmd": "true"})))["seq"].to_string();
    assert_eq!(code(&req(&mut n, "note_verify", "demo", &token, json!({"note": id, "exit": 0, "out_sha": SHA}))), "denied");
    assert_eq!(notes(&mut n, &token, json!({}))[0]["trust"], "agent-claimed");
}

#[test]
fn a_link_shares_notes_only_when_it_names_them() {
    let dir = tempdir().unwrap();
    let mut n = node(dir.path());
    let ((_, at), (_, bt)) = (session(&mut n, "demo", "a"), session(&mut n, "other", "b"));
    ok(note(&mut n, &at, json!({})));
    ok(req(&mut n, "link", "demo", &at, json!({"to": "other", "channels": ["#general"], "mode": "ro"})));
    assert_eq!(code(&req(&mut n, "notes", "demo", &bt, json!({}))), "denied");
    assert_eq!(code(&req(&mut n, "note_verify", "demo", &bt, json!({"note": "1", "exit": 0, "out_sha": SHA}))), "denied");
}
