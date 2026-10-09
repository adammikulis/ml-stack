mod kit;

use kit::*;
use poolhouse_node::node::Node;
use serde_json::{json, Value};
use std::process::{Command, Stdio};
use tempfile::tempdir;

fn claim(n: &mut Node, token: &str, kind: &str, key: &str, extra: Value) -> Value {
    let mut params = json!({"kind": kind, "key": key});
    for (k, v) in extra.as_object().unwrap() {
        params[k] = v.clone();
    }
    req(n, "claim", "demo", token, params)
}

fn listed(n: &mut Node, token: &str) -> Vec<Value> {
    ok(req(n, "claims", "demo", token, json!({})))["claims"].as_array().unwrap().clone()
}

#[test]
fn every_kind_is_a_claim_with_its_owner_shown_and_a_wrong_kind_is_refused() {
    let dir = tempdir().unwrap();
    let mut n = node(dir.path());
    let (name, token) = session(&mut n, "demo", "a");
    for (kind, key) in [("worktree", "/work/a"), ("branch", "feat"), ("area", "/repo/src"), ("port", "8080"), ("server", "broker"), ("install", "/venv")] {
        let got = ok(claim(&mut n, &token, kind, key, json!({"ttl_s": 600})));
        assert_eq!((got["claim"]["kind"].clone(), got["claim"]["key"].clone(), got["claim"]["owner"].clone()), (json!(kind), json!(key), json!(name)), "{kind}");
        assert!(got["claim"]["expires_in_s"].as_u64().unwrap() <= 600);
    }
    assert_eq!(listed(&mut n, &token).len(), 6);
    assert_eq!(ok(req(&mut n, "claims", "demo", &token, json!({"kind": "port"})))["claims"].as_array().unwrap().len(), 1);
    for (kind, key) in [("file", "/x"), ("port", "99999"), ("worktree", "relative/path"), ("branch", "../x")] {
        assert_eq!(code(&claim(&mut n, &token, kind, key, json!({}))), "invalid", "{kind} {key}");
    }
}

#[test]
fn a_second_holder_is_refused_naming_the_first_and_a_nested_path_conflicts() {
    let dir = tempdir().unwrap();
    let mut n = node(dir.path());
    let ((first, at), (_, bt)) = (session(&mut n, "demo", "a"), session(&mut n, "demo", "b"));
    ok(claim(&mut n, &at, "worktree", "/work/repo", json!({})));
    for path in ["/work/repo", "/work/repo/sub", "/work"] {
        let refused = claim(&mut n, &bt, "worktree", path, json!({}));
        assert_eq!(code(&refused), "denied", "{path}");
        assert!(refused["error"]["message"].as_str().unwrap().contains(&first), "{refused}");
    }
    ok(claim(&mut n, &bt, "worktree", "/work/other", json!({})));
    assert_eq!(listed(&mut n, &bt).len(), 2, "claims show every holder");
}

#[test]
fn a_claim_whose_process_died_or_whose_time_ran_out_is_gone_and_a_heartbeat_extends_it() {
    let dir = tempdir().unwrap();
    let mut n = node(dir.path());
    let ((_, at), (_, bt)) = (session(&mut n, "demo", "a"), session(&mut n, "demo", "b"));
    let mut child = Command::new("sleep").arg("120").stdout(Stdio::null()).stderr(Stdio::null()).spawn().unwrap();
    ok(claim(&mut n, &at, "port", "9001", json!({"pid": child.id()})));
    assert_eq!(code(&claim(&mut n, &bt, "port", "9001", json!({}))), "denied");
    child.kill().unwrap();
    child.wait().unwrap();
    assert_eq!(ok(claim(&mut n, &bt, "port", "9001", json!({})))["changed"], true, "the dead holder's claim was dropped");
    ok(claim(&mut n, &bt, "server", "short", json!({"ttl_s": 1})));
    let lease = listed(&mut n, &bt).into_iter().find(|c| c["key"] == "short").unwrap()["lease"].clone();
    let renewed = ok(req(&mut n, "lease_renew", "demo", &bt, json!({"id": lease, "ttl_s": 600})));
    assert!(renewed["expires_ms"].as_u64().unwrap() > renewed["granted_ms"].as_u64().unwrap() + 100_000);
    ok(claim(&mut n, &bt, "server", "gone", json!({"ttl_s": 1})));
    std::thread::sleep(std::time::Duration::from_millis(1200));
    let keys: Vec<Value> = listed(&mut n, &bt).iter().map(|c| c["key"].clone()).collect();
    assert!(!keys.contains(&json!("gone")) && keys.contains(&json!("short")), "{keys:?}");
}

#[test]
fn only_the_holder_releases_and_a_claim_of_one_board_is_not_listed_on_another() {
    let dir = tempdir().unwrap();
    let mut n = node(dir.path());
    let ((_, at), (_, bt)) = (session(&mut n, "demo", "a"), session(&mut n, "demo", "b"));
    let (_, ot) = session(&mut n, "other", "o");
    ok(claim(&mut n, &at, "area", "/repo/a", json!({})));
    assert_eq!(code(&req(&mut n, "release", "demo", &bt, json!({"kind": "area", "key": "/repo/a"}))), "denied");
    assert_eq!(ok(req(&mut n, "release", "demo", &bt, json!({"kind": "area", "key": "/repo/none"})))["released"], false);
    assert!(ok(req(&mut n, "claims", "other", &ot, json!({})))["claims"].as_array().unwrap().is_empty());
    assert_eq!(ok(req(&mut n, "release", "demo", &at, json!({"kind": "area", "key": "/repo/a"})))["released"], true);
    assert!(listed(&mut n, &at).is_empty());
    // the lease table shows the same fact a lease client sees
    ok(claim(&mut n, &at, "server", "s", json!({})));
    let leases = ok(req(&mut n, "lease_list", "demo", &at, json!({})));
    assert_eq!(leases["leases"].as_array().unwrap().len(), 1);
}
