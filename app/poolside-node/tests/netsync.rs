mod kit;

use kit::netkit::*;
use poolside_node::error::Error;
use poolside_node::netsync::sync_all;
use poolside_node::peer::PeerClient;
use poolside_node::sync::{MAX_BATCH_BYTES, MAX_ROWS};
use serde_json::{json, Value};

fn reach(from: &Dev, to: &Dev) -> PeerClient {
    PeerClient::connect(&from.net.me, to.addr(), Some(&to.fp())).unwrap()
}

fn sync(from: &Dev, to: &Dev) {
    sync_all(&from.node, &mut reach(from, to)).unwrap();
}

fn origin_of(d: &Dev) -> String {
    d.node.lock().unwrap().boards["demo"].board.origin().to_string()
}

/// Three paired devices that all hold `demo`, each with its own session.
fn trio() -> [(Dev, String); 3] {
    let (a, b, c) = (device(), device(), device());
    a.pair_secure(&b);
    a.pair_secure(&c);
    sync(&b, &a);
    sync(&c, &a);
    // Distinct session ids: a foreign row from a name that is also a local session's is refused.
    let mut n = 0;
    [a, b, c].map(|d| {
        n += 1;
        let t = d.session("demo", &format!("writer-{n}")).1;
        (d, t)
    })
}

#[test]
fn a_third_device_gets_rows_through_a_relay_and_they_still_verify() {
    let [(a, at), (b, bt), (c, ct)] = trio();
    a.say("demo", &at, "written on a");
    sync(&b, &a);
    sync(&c, &b);
    assert!(c.texts("demo", &ct).contains(&"written on a".to_string()), "a reached c though c never spoke to a");
    let read = c.ok("read", "demo", &ct, json!({"limit": 1000}));
    let entry = read["entries"].as_array().unwrap().iter().find(|e| e["fields"]["body"] == "written on a").unwrap();
    assert_eq!(entry["foreign"], true);
    assert!(entry["sender"].as_str().unwrap().contains("@d"), "a foreign sender is shown as name@dN");
    let _ = (bt, b);
}

#[test]
fn a_forged_copy_from_a_relay_is_refused_and_does_not_mark_the_origin_damaged() {
    let [(a, at), (b, _), (c, ct)] = trio();
    a.say("demo", &at, "genuine");
    sync(&b, &a);
    let origin = origin_of(&a);
    let mut rows: Vec<Value> = {
        let n = b.node.lock().unwrap();
        n.boards["demo"].board.trusted(&origin).iter().map(|r| serde_json::to_value(r).unwrap()).collect()
    };
    let last = rows.len() - 1;
    let mut forged = rows[last].clone();
    forged["body"] = json!({"type": "status", "from": "someone", "to": "#general", "subject": "", "body": "forged"});
    rows[last] = forged;
    let c_origin = origin_of(&c);
    let mut relay = reach(&b, &c);
    let reply = relay.call(&json!({"op": "push", "board": "demo", "origin": origin_of(&b), "logs": {origin.clone(): rows}, "trusted_vector": {}})).unwrap();
    assert_eq!(reply["refused"][&origin], "damaged", "{reply}");
    let damaged = c.node.lock().unwrap().boards["demo"].board.damaged().clone();
    assert!(damaged.is_empty(), "a relay's bad copy never marks the origin: {damaged:?}");
    sync(&c, &a);
    assert!(c.texts("demo", &ct).contains(&"genuine".to_string()), "the genuine rows still arrive from their owner");
    let _ = c_origin;
}

#[test]
fn the_owner_of_a_log_that_sends_a_forged_copy_of_it_is_marked_damaged() {
    let [(a, at), (_, _), (c, _)] = trio();
    a.say("demo", &at, "genuine");
    a.say("demo", &at, "second");
    sync(&c, &a);
    let origin = origin_of(&a);
    let mut rows: Vec<Value> = {
        let n = a.node.lock().unwrap();
        n.boards["demo"].board.trusted(&origin).iter().map(|r| serde_json::to_value(r).unwrap()).collect()
    };
    let last = rows.len() - 1;
    rows[last]["hash"] = json!("0".repeat(64));
    let mut owner = reach(&a, &c);
    let reply = owner.call(&json!({"op": "push", "board": "demo", "origin": origin, "logs": {origin.clone(): rows}, "trusted_vector": {}})).unwrap();
    assert_eq!(reply["refused"][&origin], "damaged");
    assert!(c.node.lock().unwrap().boards["demo"].board.damaged().contains_key(&origin));
}

#[test]
fn a_pull_carries_at_most_the_rows_and_bytes_of_a_batch() {
    let [(a, at), (b, bt), _] = trio();
    for i in 0..MAX_ROWS + 20 {
        a.say("demo", &at, &format!("row {i}"));
    }
    let mut c = reach(&b, &a);
    let reply = c.call(&json!({"op": "pull", "board": "demo", "origin": origin_of(&b), "vector": {}})).unwrap();
    let rows: usize = reply["logs"].as_object().unwrap().values().map(|v| v.as_array().unwrap().len()).sum();
    assert!(rows <= MAX_ROWS && rows > 0, "{rows} rows in one reply");
    sync(&b, &a);
    assert_eq!(b.texts("demo", &bt).iter().filter(|t| t.starts_with("row ")).count(), MAX_ROWS + 20, "many rounds finish what one batch could not");

    let big = "x".repeat(60_000);
    let before = a.texts("demo", &at).len();
    for _ in 0..40 {
        a.say("demo", &at, &big);
    }
    let mut c = reach(&b, &a);
    let vector = b.node.lock().unwrap().boards["demo"].board.vector();
    let reply = c.call(&json!({"op": "pull", "board": "demo", "origin": origin_of(&b), "vector": vector})).unwrap();
    let bytes = serde_json::to_vec(&reply["logs"]).unwrap().len();
    assert!(bytes < MAX_BATCH_BYTES + 130 * 1024, "{bytes} bytes in one reply");
    sync(&b, &a);
    assert_eq!(b.texts("demo", &bt).len(), before + 40);
}

#[test]
fn a_push_past_the_request_bound_is_refused_whole() {
    let [(a, _), (b, _), _] = trio();
    let many: serde_json::Map<String, Value> = (0..20).map(|i| (format!("{i:032x}"), json!([]))).collect();
    let mut c = reach(&b, &a);
    let r = c.call(&json!({"op": "push", "board": "demo", "origin": origin_of(&b), "logs": many, "trusted_vector": {}}));
    assert!(matches!(r, Err(Error::Invalid(_))), "{r:?}");
}

#[test]
fn a_device_that_writes_a_different_log_than_it_presented_is_refused() {
    let [(a, _), (b, _), _] = trio();
    sync(&b, &a);
    let mut c = reach(&b, &a);
    let r = c.call(&json!({"op": "vector", "board": "demo", "origin": "f".repeat(32)}));
    assert!(matches!(r, Err(Error::Denied(_))), "{r:?}");
}

#[test]
fn a_long_unsigned_run_is_signed_as_it_grows_so_a_batch_always_holds_a_head() {
    let dir = tempfile::tempdir().unwrap();
    let mut board = kit::board(&dir, 1);
    for i in 0..200 {
        kit::say(&mut board, "claude-aaaaaa", &format!("m{i}"));
    }
    let origin = board.origin().to_string();
    let rows = board.rows(&origin);
    let heads: Vec<usize> = rows.iter().enumerate().filter(|(_, r)| r.kind == poolside_node::row::Kind::Head).map(|(i, _)| i).collect();
    assert!(heads.len() >= 3, "{heads:?}");
    assert!(heads.windows(2).all(|w| w[1] - w[0] <= poolside_node::board::HEAD_EVERY_ROWS + 1), "{heads:?}");
}

#[test]
fn two_partitioned_devices_converge_when_they_meet() {
    let [(a, at), (b, bt), _] = trio();
    a.say("demo", &at, "a1");
    b.say("demo", &bt, "b1");
    a.say("demo", &at, "a2");
    b.say("demo", &bt, "b2");
    sync(&a, &b);
    let (mut x, mut y) = (a.texts("demo", &at), b.texts("demo", &bt));
    assert_eq!(x, y, "same rows in the same merged order");
    x.sort();
    y.sort();
    assert_eq!(x, vec!["a1", "a2", "b1", "b2"]);
    let r = a.ok("sync_now", "", &a.token, json!({}));
    assert!(r["reached"].as_u64().unwrap() >= 1);
}
