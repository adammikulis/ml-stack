//! Landing authority is this device's own: two real nodes, paired over loopback TLS, and a request, a review and a
//! runner state written on the other one count here only while that device is on this device's list.

mod kit;

use kit::netkit::*;
use poolhouse_node::netsync::sync_all;
use poolhouse_node::peer::PeerClient;
use serde_json::{json, Value};

const TIP: &str = "abcdef0123456789abcdef0123456789abcdef01";

fn sync(from: &Dev, to: &Dev) {
    let mut peer = PeerClient::connect(&from.net.me, to.addr(), Some(&to.fp())).unwrap();
    sync_all(&from.node, &mut peer).unwrap();
}

fn both_ways(a: &Dev, b: &Dev) {
    sync(a, b);
    sync(b, a);
}

/// This device (a) and one that joined its pool (b), which holds three sessions: asker, judge and runner.
fn pair() -> (Dev, Dev, [String; 3]) {
    let (a, b) = (device(), device());
    a.pair_secure(&b);
    both_ways(&a, &b);
    let tokens = ["asker", "judge", "runner"].map(|id| b.session("demo", &format!("b-{id}")).1);
    (a, b, tokens)
}

/// The sequence a hostile or merely careless member writes: a request, an independent review and the runner's `landed`.
fn forge(b: &Dev, [asker, judge, runner]: &[String; 3]) {
    let made = b.ok("land_request", "demo", asker, json!({"branch": "feat", "sha": TIP, "target": "dev", "selectors": ["tests/test_x.py"]}));
    let id = made["id"].as_str().unwrap().to_string();
    b.ok("land_review", "demo", judge, json!({"req": id, "sha": TIP, "verdict": "accept"}));
    b.ok("claim", "demo", runner, json!({"kind": "branch", "key": "dev"}));
    b.ok("land_state", "demo", runner, json!({"req": id, "status": "landed", "detail": "forged"}));
}

fn queue(d: &Dev) -> Value {
    d.ok("land_queue", "demo", &d.token, json!({}))
}

fn statuses(q: &Value) -> Vec<String> {
    q["requests"].as_array().unwrap().iter().map(|r| r["status"].as_str().unwrap().to_string()).collect()
}

fn trust(a: &Dev, b: &Dev, on: bool) -> Value {
    a.call("land_trust", "demo", &a.token, json!({"device": b.fp(), "trusted": on}))
}

#[test]
fn a_device_that_joined_the_pool_lands_nothing_here_until_this_device_names_it() {
    let (a, b, tokens) = pair();
    forge(&b, &tokens);
    both_ways(&a, &b);

    let q = queue(&a);
    assert!(statuses(&q).is_empty(), "foreign entries are not folded into the queue: {q}");
    assert_eq!(q["foreign"]["total"], 3, "a request, a review and a state were seen and kept");
    assert_eq!(q["foreign"]["status"], "foreign, ignored");
    assert_eq!(q["trusted_devices"], json!([]), "default: only this device");
    let listed = q["foreign"]["latest"].as_array().unwrap();
    assert!(listed.iter().all(|e| e["device"].as_str().unwrap().contains("(d")), "each names the device: {listed:?}");
    assert_eq!(listed.iter().map(|e| e["ev"].as_str().unwrap()).collect::<Vec<_>>(), ["request", "review", "state"]);

    assert_eq!(trust(&a, &b, true)["ok"], true);
    let q = queue(&a);
    assert_eq!(statuses(&q), ["landed"], "the same sequence counts once the device is named: {q}");
    assert_eq!(q["foreign"]["total"], 0);
    assert_eq!(q["trusted_devices"], json!([b.fp()]));

    assert_eq!(trust(&a, &b, false)["result"]["changed"], true);
    let q = queue(&a);
    assert!(statuses(&q).is_empty() && q["foreign"]["total"] == 3, "removing the device removes it again: {q}");
}

#[test]
fn an_entry_of_this_device_still_counts_beside_ignored_foreign_ones() {
    let (a, b, tokens) = pair();
    forge(&b, &tokens);
    both_ways(&a, &b);
    a.ok("land_request", "demo", &a.token, json!({"branch": "mine", "sha": TIP, "target": "dev", "selectors": ["t"]}));
    let q = queue(&a);
    assert_eq!(statuses(&q), ["needs-review"]);
    assert_eq!(q["requests"][0]["branch"], "mine");
    assert_eq!(q["foreign"]["total"], 3);
}

#[test]
fn putting_a_trusted_device_out_of_the_pool_ends_its_standing_too() {
    let (a, b, tokens) = pair();
    forge(&b, &tokens);
    both_ways(&a, &b);
    trust(&a, &b, true);
    assert_eq!(statuses(&queue(&a)), ["landed"]);
    a.ok("member_revoke", "", &a.token, json!({"fingerprint": b.fp()}));
    assert!(statuses(&queue(&a)).is_empty(), "a revoked device no longer counts, listed or not");
}

#[test]
fn naming_a_device_is_for_a_session_with_no_parent_and_only_for_an_active_member_and_is_audited() {
    let (a, b, _) = pair();
    let kid = a.ok("register", "demo", &a.token, json!({"model": "claude-sonnet-5-5", "harness": "claude-code", "session": "kid"}))["token"].as_str().unwrap().to_string();
    let code = |r: Value| r["error"]["code"].as_str().unwrap_or("ok").to_string();
    assert_eq!(code(a.call("land_trust", "demo", &kid, json!({"device": b.fp(), "trusted": true}))), "denied", "a subagent never names a device");
    assert_eq!(code(a.call("land_trust", "demo", "", json!({"device": b.fp(), "trusted": true}))), "denied");
    assert_eq!(code(a.call("land_trust", "demo", &a.token, json!({"device": "ab".repeat(32), "trusted": true}))), "invalid", "not a member");
    assert_eq!(code(a.call("land_trust", "demo", &a.token, json!({"device": a.fp(), "trusted": true}))), "invalid", "this device always counts");
    assert_eq!(code(a.call("land_trust", "demo", &a.token, json!({"device": b.fp()}))), "invalid", "trusted is a boolean");
    assert_eq!(code(trust(&a, &b, true)), "ok");
    assert_eq!(code(trust(&a, &b, false)), "ok");
    let audit = a.ok("read", "demo", &a.token, json!({"kind": "audit", "limit": 50}));
    let events: Vec<&str> = audit["entries"].as_array().unwrap().iter().map(|e| e["fields"]["event"].as_str().unwrap()).collect();
    assert!(events.contains(&"land.trust") && events.contains(&"land.untrust"), "recorded on the board: {events:?}");
}

#[test]
fn the_list_is_this_devices_own_and_never_replicated() {
    let (a, b, tokens) = pair();
    forge(&b, &tokens);
    trust(&a, &b, true);
    both_ways(&a, &b);
    let theirs = b.ok("land_queue", "demo", &b.token, json!({}));
    assert_eq!(theirs["trusted_devices"], json!([]), "naming b on a says nothing on b");
    assert_eq!(theirs["requests"].as_array().unwrap().len(), 1, "b counts its own entries");
}
