mod kit;

use kit::*;
use poolside_node::identity::digest;
use poolside_node::row::Kind;
use poolside_node::sync::exchange_nodes;
use serde_json::json;
use tempfile::tempdir;

#[test]
fn a_name_is_family_and_six_hex_of_the_session_digest_and_stable() {
    let dir = tempdir().unwrap();
    let mut n = node(dir.path());
    let (name, _) = session(&mut n, "demo", "sess-1");
    assert_eq!(name, format!("claude-{}", &digest("claude-code", "sess-1")[..6]));
    let again = ok(req(&mut n, "register", "demo", "", json!({"model": "claude-sonnet-5-5", "harness": "claude-code", "session": "sess-1"})));
    assert_eq!((again["name"].as_str().unwrap(), again["created"].clone()), (name.as_str(), json!(false)));
    let other = ok(req(&mut n, "register", "demo", "", json!({"model": "gpt-5", "harness": "codex", "session": "sess-1"})));
    assert!(other["name"].as_str().unwrap().starts_with("chatgpt-"));
}

#[test]
fn a_colliding_short_name_extends_by_two_characters() {
    let dir = tempdir().unwrap();
    let full = digest("claude-code", "sess-1");
    let board_dir = dir.path().join("boards").join("demo");
    std::fs::create_dir_all(&board_dir).unwrap();
    let taken = format!("claude-{}", &full[..6]);
    std::fs::write(board_dir.join("names.json"), json!({taken: {"digest": "other", "family": "claude", "parent": "", "model": "", "model_state": "unknown", "harness": "", "retired": false, "seen_ms": 0}}).to_string()).unwrap();
    let mut n = node(dir.path());
    let (name, _) = session(&mut n, "demo", "sess-1");
    assert_eq!(name, format!("claude-{}", &full[..8]));
}

#[test]
fn a_subagent_has_its_own_name_and_records_its_parent() {
    let dir = tempdir().unwrap();
    let mut n = node(dir.path());
    let (lead, lead_token) = session(&mut n, "demo", "lead");
    let sub = ok(req(&mut n, "register", "demo", &lead_token, json!({"model": "claude-haiku-5-5", "harness": "claude-code", "session": "sub"})));
    assert_eq!(sub["parent"], lead);
    assert_ne!(sub["name"], json!(lead));
    let (_, foreign) = session(&mut n, "other", "x");
    let denied = req(&mut n, "register", "demo", &foreign, json!({"model": "m", "harness": "h", "session": "y"}));
    assert_eq!(code(&denied), "denied", "a parent must be a session of the same board");
}

#[test]
fn every_write_is_stamped_from_the_token_and_a_forged_sender_is_refused() {
    let dir = tempdir().unwrap();
    let mut n = node(dir.path());
    let ((alice, at), (bob, bt)) = (session(&mut n, "demo", "a"), session(&mut n, "demo", "b"));
    let wrote = ok(post(&mut n, "demo", &at, "hello"));
    assert_eq!(wrote["sender"], alice);
    let read = ok(req(&mut n, "read", "demo", &bt, json!({})));
    let message = read["entries"].as_array().unwrap().iter().find(|e| e["kind"] == "message").unwrap();
    assert_eq!(message["sender"], alice);
    for forged in ["from", "sender", "name", "parent", "label", "author", "actor"] {
        let reply = req(&mut n, "post", "demo", &bt, json!({"kind": "message", "fields": {"type": "status", "body": "x", forged: alice}}));
        assert_eq!(code(&reply), "denied", "{forged}");
        let reply = req(&mut n, "post", "demo", &bt, json!({"kind": "message", forged: alice, "fields": {"type": "status", "body": "x"}}));
        assert_eq!(code(&reply), "denied", "param {forged}");
        let mut envelope = json!({"v": 1, "id": 1, "method": "post", "board": "demo", "token": bt, "params": {"kind": "message", "fields": {"type": "status", "body": "x"}}});
        envelope[forged] = json!(alice);
        assert_eq!(code(&poolside_node::api::handle(&mut n, &envelope)), "denied", "envelope {forged}");
    }
    assert_eq!(texts(&ok(req(&mut n, "read", "demo", &bt, json!({})))["entries"]), vec!["hello"]);
    assert_eq!(code(&post(&mut n, "demo", "not-a-token", "x")), "denied");
    assert_eq!(ok(req(&mut n, "whoami", "demo", &bt, json!({})))["name"], bob);
}

#[test]
fn the_token_is_stored_hashed_in_an_owner_only_file() {
    let dir = tempdir().unwrap();
    let mut n = node(dir.path());
    let (_, token) = session(&mut n, "demo", "a");
    let file = dir.path().join("tokens.json");
    assert!(!std::fs::read_to_string(&file).unwrap().contains(&token));
    assert!(poolside_node::sys::owner_only(&file));
    assert!(poolside_node::sys::owner_only(dir.path()));
    drop(n);
    let mut n = node(dir.path());
    assert!(ok(post(&mut n, "demo", &token, "after restart"))["seq"].as_u64().is_some(), "a token survives a restart");
}

#[test]
fn a_read_cursor_returns_each_entry_once_even_after_a_sync_brings_older_ones() {
    let (da, db) = (tempdir().unwrap(), tempdir().unwrap());
    let (mut a, mut b) = (node(da.path()), node(db.path()));
    let ((_, at), (_, bt)) = (session(&mut a, "demo", "a"), session(&mut b, "demo", "b"));
    ok(post(&mut a, "demo", &at, "a1"));
    let first = ok(req(&mut a, "read", "demo", &at, json!({"kind": "message"})));
    assert_eq!(texts(&first["entries"]), vec!["a1"]);
    ok(post(&mut b, "demo", &bt, "b1"));
    exchange_nodes(&mut a, &mut b).unwrap();
    let next = ok(req(&mut a, "read", "demo", &at, json!({"kind": "message", "since": first["cursor"]})));
    assert_eq!(texts(&next["entries"]), vec!["b1"]);
    let none = ok(req(&mut a, "read", "demo", &at, json!({"kind": "message", "since": next["cursor"]})));
    assert!(none["entries"].as_array().unwrap().is_empty());
    let limited = ok(req(&mut a, "read", "demo", &at, json!({"kind": "message", "limit": 1})));
    let rest = ok(req(&mut a, "read", "demo", &at, json!({"kind": "message", "since": limited["cursor"]})));
    assert_eq!(texts(&limited["entries"]).len() + texts(&rest["entries"]).len(), 2);
}

#[test]
fn claims_conflict_within_a_board_and_release_only_by_the_holder() {
    let dir = tempdir().unwrap();
    let mut n = node(dir.path());
    let ((alice, at), (_, bt)) = (session(&mut n, "demo", "a"), session(&mut n, "demo", "b"));
    let first = ok(req(&mut n, "claim", "demo", &at, json!({"kind": "branch", "key": "feature"})));
    assert_eq!(first["claim"]["owner"], alice);
    assert_eq!(ok(req(&mut n, "claim", "demo", &at, json!({"kind": "branch", "key": "feature"})))["changed"], false);
    let refused = req(&mut n, "claim", "demo", &bt, json!({"kind": "branch", "key": "feature"}));
    assert_eq!(code(&refused), "denied");
    assert!(refused["error"]["message"].as_str().unwrap().contains(&alice), "{refused}");
    assert_eq!(code(&req(&mut n, "release", "demo", &bt, json!({"kind": "branch", "key": "feature"}))), "denied");
    assert_eq!(ok(req(&mut n, "release", "demo", &at, json!({"kind": "branch", "key": "feature"})))["released"], true);
    assert_eq!(ok(req(&mut n, "claim", "demo", &bt, json!({"kind": "branch", "key": "feature"})))["changed"], true);
}

#[test]
fn two_boards_on_one_node_are_isolated() {
    let dir = tempdir().unwrap();
    let mut n = node(dir.path());
    let ((name1, t1), (name2, t2)) = (session(&mut n, "alpha", "same-session"), session(&mut n, "beta", "same-session"));
    assert_eq!(name1, name2, "names are per board: the same session is registered on each");
    assert_ne!(t1, t2);
    ok(post(&mut n, "alpha", &t1, "alpha secret"));
    ok(req(&mut n, "claim", "alpha", &t1, json!({"kind": "branch", "key": "t"})));
    // the other board sees nothing and its claim of the same name does not conflict
    assert!(texts(&ok(req(&mut n, "read", "beta", &t2, json!({})))["entries"]).is_empty());
    assert_eq!(ok(req(&mut n, "claim", "beta", &t2, json!({"kind": "branch", "key": "t"})))["claim"]["key"], "t");
    assert_eq!(ok(req(&mut n, "claim", "beta", &t2, json!({"kind": "branch", "key": "t"})))["changed"], false);
    assert_eq!(ok(req(&mut n, "claims", "beta", &t2, json!({})))["claims"].as_array().unwrap().len(), 1);
    // a token cannot reach the other board without a link, for any method
    assert_eq!(code(&req(&mut n, "read", "alpha", &t2, json!({}))), "denied");
    assert_eq!(code(&post(&mut n, "alpha", &t2, "x")), "denied");
    assert_eq!(code(&req(&mut n, "claim", "alpha", &t2, json!({"kind": "server", "key": "u"}))), "denied");
    assert_eq!(code(&req(&mut n, "read", "no-such-board", &t2, json!({}))), "denied");
    // registering a third session on beta leaves alpha's names alone
    session(&mut n, "beta", "someone-else");
    assert_eq!(ok(req(&mut n, "status", "alpha", &t1, json!({})))["board"]["sessions"], 1);
    // separate directories, separate origins
    assert!(dir.path().join("boards/alpha/log").is_dir() && dir.path().join("boards/beta/log").is_dir());
    assert_ne!(n.boards["alpha"].board.origin(), n.boards["beta"].board.origin());
}

#[test]
fn a_link_shares_only_what_it_names_and_revoking_it_stops_it() {
    let dir = tempdir().unwrap();
    let mut n = node(dir.path());
    let ((_, at), (_, bt)) = (session(&mut n, "alpha", "a"), session(&mut n, "beta", "b"));
    ok(post(&mut n, "alpha", &at, "public words"));
    ok(req(&mut n, "post", "alpha", &at, json!({"kind": "note", "fields": {"nkind": "fact", "title": "private", "body": "n"}})));
    // only a session of the sharing board may link
    assert_eq!(code(&req(&mut n, "link", "alpha", &bt, json!({"to": "beta"}))), "denied");
    let link = ok(req(&mut n, "link", "alpha", &at, json!({"to": "beta", "channels": ["#general"], "mode": "ro"})));
    let seen = ok(req(&mut n, "read", "alpha", &bt, json!({})));
    let kinds: Vec<_> = seen["entries"].as_array().unwrap().iter().map(|e| e["channel"].as_str().unwrap().to_string()).collect();
    assert!(kinds.iter().all(|c| c == "#general"), "{kinds:?}");
    assert_eq!(texts(&seen["entries"]), vec!["public words"]);
    // read-only: no writes, no claims
    assert_eq!(code(&post(&mut n, "alpha", &bt, "x")), "denied");
    assert_eq!(code(&req(&mut n, "claim", "alpha", &bt, json!({"kind": "server", "key": "t"}))), "denied");
    // the link does not run backwards
    assert_eq!(code(&req(&mut n, "read", "beta", &at, json!({}))), "denied");
    // it is audited on both boards
    for (b, t) in [("alpha", &at), ("beta", &bt)] {
        let audit = ok(req(&mut n, "read", b, t, json!({"kind": "audit"})));
        assert_eq!(audit["entries"][0]["fields"]["event"], "link", "{b}");
    }
    ok(req(&mut n, "unlink", "alpha", &at, json!({"id": link["id"]})));
    assert_eq!(code(&req(&mut n, "read", "alpha", &bt, json!({}))), "denied");
}

#[test]
fn a_read_write_link_lets_a_foreign_session_post_only_into_its_channels() {
    let dir = tempdir().unwrap();
    let mut n = node(dir.path());
    let ((_, at), (bname, bt)) = (session(&mut n, "alpha", "a"), session(&mut n, "beta", "b"));
    ok(req(&mut n, "link", "alpha", &at, json!({"to": "beta", "channels": ["#general"], "mode": "rw"})));
    let wrote = ok(post(&mut n, "alpha", &bt, "hello from beta"));
    assert_eq!(wrote["sender"], format!("{bname}@beta"));
    assert_eq!(texts(&ok(req(&mut n, "read", "alpha", &at, json!({})))["entries"]), vec!["hello from beta"]);
    let note = req(&mut n, "post", "alpha", &bt, json!({"kind": "note", "fields": {"nkind": "fact", "title": "t", "body": "b"}}));
    assert_eq!(code(&note), "denied", "#notes is not shared");
}

#[test]
fn a_device_only_syncs_boards_it_holds() {
    let (da, db) = (tempdir().unwrap(), tempdir().unwrap());
    let (mut a, mut b) = (node(da.path()), node(db.path()));
    let ((_, at1), (_, at2)) = (session(&mut a, "shared", "x"), session(&mut a, "mine-only", "x"));
    session(&mut b, "shared", "y");
    ok(post(&mut a, "shared", &at1, "to share"));
    ok(post(&mut a, "mine-only", &at2, "stay home"));
    let reports = exchange_nodes(&mut a, &mut b).unwrap();
    assert_eq!(reports.keys().collect::<Vec<_>>(), vec!["shared"]);
    assert!(!b.boards.contains_key("mine-only"));
}

#[test]
fn foreign_entries_are_built_from_allow_lists_and_checked() {
    let (da, db) = (tempdir().unwrap(), tempdir().unwrap());
    let (mut a, mut b) = (node(da.path()), node(db.path()));
    let (_, bt) = session(&mut b, "demo", "b");
    let (local, _) = session(&mut b, "demo", "victim");
    let general = |extra: serde_json::Value, from: &str| {
        let mut body = json!({"type": "status", "from": from, "to": "#general", "subject": "", "body": "hi"});
        for (k, v) in extra.as_object().unwrap() {
            body[k] = v.clone();
        }
        body
    };
    let bad = [
        general(json!({"flags": ["reviewed"]}), "mallory"),
        general(json!({"to": "#private"}), "mallory"),
        general(json!({"subject": "two\nlines"}), "mallory"),
        general(json!({}), "someone-else"),
        general(json!({}), "@local"),
        general(json!({"type": "milestone"}), "mallory"),
        general(json!({"body": "x".repeat(70_000)}), "mallory"),
    ];
    let mallory = a.host("demo").unwrap();
    for body in bad {
        let actor = if body["from"] == "@local" { local.as_str() } else { "mallory" };
        let body = if body["from"] == "@local" { json!({"type": "status", "from": actor, "to": "#general", "subject": "", "body": "hi"}) } else { body };
        mallory.board.append(Kind::Message, actor, body, "").unwrap();
    }
    mallory.board.append(Kind::Message, "mallory", general(json!({}), "mallory"), "").unwrap();
    mallory.board.append(Kind::Note, "mallory", json!({"nkind": "fact", "title": "t", "body": "b", "author": "mallory", "trust": "human"}), "").unwrap();
    exchange_nodes(&mut a, &mut b).unwrap();
    let read = ok(req(&mut b, "read", "demo", &bt, json!({"kind": "message"})));
    let entries = read["entries"].as_array().unwrap();
    assert_eq!(entries.len(), 1, "{read}");
    assert_eq!(entries[0]["sender"], "mallory@d1");
    assert_eq!(entries[0]["foreign"], true);
    assert!(ok(req(&mut b, "read", "demo", &bt, json!({"kind": "note"})))["entries"].as_array().unwrap().is_empty());
    assert_eq!(ok(req(&mut b, "status", "demo", &bt, json!({})))["board"]["rejected"], 8);
}
