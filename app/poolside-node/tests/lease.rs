mod kit;

use kit::{code, ok, req, session};
use poolside_node::lease::live;
use poolside_node::lease::merge;
use poolside_node::node::Node;
use serde_json::{json, Value};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tempfile::TempDir;

const T0: u64 = 1_000_000_000;

fn clocked(node: &mut Node) -> Arc<AtomicU64> {
    let t = Arc::new(AtomicU64::new(T0));
    let c = t.clone();
    node.leases.clock = Box::new(move || c.load(Ordering::SeqCst));
    t
}

fn advance(t: &Arc<AtomicU64>, secs: u64) {
    t.fetch_add(secs * 1000, Ordering::SeqCst);
}

fn gpu() -> Value {
    json!({"type": "gpu"})
}

fn acquire(n: &mut Node, token: &str, params: Value) -> Value {
    ok(req(n, "lease_acquire", "demo", token, params))
}

fn state(n: &mut Node, token: &str, id: &Value) -> String {
    ok(req(n, "lease_wait", "demo", token, json!({"id": id, "timeout_ms": 0})))["state"].as_str().unwrap().into()
}

fn release(n: &mut Node, token: &str, id: &Value) {
    ok(req(n, "lease_release", "demo", token, json!({"id": id})));
}

fn sleeper() -> Child {
    Command::new("sleep").arg("120").stdout(Stdio::null()).stderr(Stdio::null()).spawn().unwrap()
}

fn two() -> (TempDir, Node, (String, String), (String, String)) {
    let dir = tempfile::tempdir().unwrap();
    let mut n = Node::open(dir.path()).unwrap();
    let (a, b) = (session(&mut n, "demo", "a"), session(&mut n, "demo", "b"));
    (dir, n, a, b)
}

#[test]
fn the_gpu_is_held_by_one_requester_and_the_second_gets_it_on_release() {
    let (_d, mut n, (_, ta), (_, tb)) = two();
    let first = acquire(&mut n, &ta, json!({"resources": [gpu()]}));
    assert_eq!(first["state"], "held");
    let second = acquire(&mut n, &tb, json!({"resources": [gpu()]}));
    assert_eq!(second["state"], "queued");
    assert_eq!(second["position"], 1);
    assert_eq!(state(&mut n, &tb, &second["id"]), "queued");
    release(&mut n, &ta, &first["id"]);
    assert_eq!(state(&mut n, &tb, &second["id"]), "held");
}

#[test]
fn a_requester_that_will_not_wait_is_told_who_is_in_the_way() {
    let (_d, mut n, (name_a, ta), (_, tb)) = two();
    acquire(&mut n, &ta, json!({"resources": [gpu()]}));
    let busy = acquire(&mut n, &tb, json!({"resources": [gpu()], "wait": false}));
    assert_eq!(busy["state"], "busy");
    assert_eq!(busy["blockers"][0]["holder"], format!("demo/{name_a}"));
    assert_eq!(ok(req(&mut n, "lease_list", "demo", &tb, json!({})))["leases"].as_array().unwrap().len(), 1);
}

#[test]
fn shortest_estimate_goes_first_and_equals_keep_arrival_order() {
    let dir = tempfile::tempdir().unwrap();
    let mut n = Node::open(dir.path()).unwrap();
    let holder = session(&mut n, "demo", "holder").1;
    let hold = acquire(&mut n, &holder, json!({"resources": [gpu()]}));
    let mut waiting = Vec::new();
    for (i, estimate) in [50u64, 10, 30, 10].iter().enumerate() {
        let token = session(&mut n, "demo", &format!("w{i}")).1;
        let lease = acquire(&mut n, &token, json!({"resources": [gpu()], "estimate_s": estimate, "ttl_s": 1000}));
        waiting.push((token, lease["id"].clone(), *estimate));
    }
    let mut order = Vec::new();
    let mut release_id = hold["id"].clone();
    let mut release_token = holder;
    for _ in 0..4 {
        release(&mut n, &release_token, &release_id);
        let at = waiting.iter().position(|(t, id, _)| state(&mut n, t, id) == "held").unwrap();
        order.push(waiting[at].2);
        let taken = waiting.remove(at);
        (release_token, release_id) = (taken.0, taken.1);
    }
    assert_eq!(order, vec![10, 10, 30, 50]);
}

#[test]
fn equal_estimates_are_first_come_first_served() {
    let dir = tempfile::tempdir().unwrap();
    let mut n = Node::open(dir.path()).unwrap();
    let holder = session(&mut n, "demo", "holder").1;
    let hold = acquire(&mut n, &holder, json!({"resources": [gpu()]}));
    let tokens: Vec<String> = (0..3).map(|i| session(&mut n, "demo", &format!("q{i}")).1).collect();
    let ids: Vec<Value> = tokens.iter().map(|t| acquire(&mut n, t, json!({"resources": [gpu()], "estimate_s": 5}))["id"].clone()).collect();
    release(&mut n, &holder, &hold["id"]);
    assert_eq!(state(&mut n, &tokens[0], &ids[0]), "held");
    assert_eq!(state(&mut n, &tokens[1], &ids[1]), "queued");
    assert_eq!(state(&mut n, &tokens[2], &ids[2]), "queued");
}

#[test]
fn a_long_run_that_waited_past_the_bound_goes_ahead_of_a_fresh_short_one() {
    let dir = tempfile::tempdir().unwrap();
    let mut n = Node::open(dir.path()).unwrap();
    let t = clocked(&mut n);
    let holder = session(&mut n, "demo", "holder").1;
    let (long_t, short_t) = (session(&mut n, "demo", "long").1, session(&mut n, "demo", "short").1);
    let hold = acquire(&mut n, &holder, json!({"resources": [gpu()], "ttl_s": 5000}));
    let long = acquire(&mut n, &long_t, json!({"resources": [gpu()], "class": "background", "ttl_s": 5000}));
    advance(&t, 10);
    let short = acquire(&mut n, &short_t, json!({"resources": [gpu()], "estimate_s": 1, "ttl_s": 5000}));
    ok(req(&mut n, "lease_renew", "demo", &holder, json!({"id": hold["id"]})));
    // not yet aged: the short one is ahead of the long one
    assert_eq!(ok(req(&mut n, "lease_wait", "demo", &short_t, json!({"id": short["id"]})))["position"], 1);
    for _ in 0..18 {
        advance(&t, 40); // waiters keep asking; one that stops for a minute is dropped
        for token_id in [(&holder, &hold["id"]), (&long_t, &long["id"]), (&short_t, &short["id"])] {
            ok(req(&mut n, "lease_renew", "demo", token_id.0, json!({"id": token_id.1})));
        }
    }
    // the long one has waited past 600 seconds and the short one has not
    assert_eq!(ok(req(&mut n, "lease_wait", "demo", &long_t, json!({"id": long["id"]})))["position"], 1);
    release(&mut n, &holder, &hold["id"]);
    assert_eq!(state(&mut n, &long_t, &long["id"]), "held");
    assert_eq!(state(&mut n, &short_t, &short["id"]), "queued");
}

#[test]
fn long_runs_are_held_to_half_the_slots_while_a_short_run_is_around() {
    let dir = tempfile::tempdir().unwrap();
    let mut n = Node::open(dir.path()).unwrap();
    n.leases.cfg.cpu_slots = 4;
    let (bg1, bg2, fg) = (session(&mut n, "demo", "bg1").1, session(&mut n, "demo", "bg2").1, session(&mut n, "demo", "fg").1);
    let cpu = |count: u32| json!([{"type": "cpu_slots", "count": count}]);
    assert_eq!(acquire(&mut n, &fg, json!({"resources": cpu(1)}))["state"], "held");
    assert_eq!(acquire(&mut n, &bg1, json!({"resources": cpu(2), "class": "background"}))["state"], "held");
    // two slots of four are long runs'; a third would take more than half while the short run holds one
    assert_eq!(acquire(&mut n, &bg2, json!({"resources": cpu(1), "class": "background"}))["state"], "queued");
    // a short run still gets one
    let fg2 = session(&mut n, "demo", "fg2").1;
    assert_eq!(acquire(&mut n, &fg2, json!({"resources": cpu(1)}))["state"], "held");
}

#[test]
fn a_dead_holder_is_detected_and_its_lease_taken_over_with_an_audit_entry() {
    let (_d, mut n, (_, ta), (name_b, tb)) = two();
    let mut child = sleeper();
    let held = acquire(&mut n, &ta, json!({"resources": [gpu()], "pid": child.id(), "ttl_s": 3600}));
    assert_eq!(held["state"], "held");
    let queued = acquire(&mut n, &tb, json!({"resources": [gpu()]}));
    assert_eq!(queued["state"], "queued");
    // still alive: nothing changes
    assert_eq!(state(&mut n, &tb, &queued["id"]), "queued");
    child.kill().unwrap(); // kill -9, deliberately left unreaped: a zombie is dead too
    assert_eq!(state(&mut n, &tb, &queued["id"]), "held");
    let audit = ok(req(&mut n, "read", "demo", &tb, json!({"kind": "audit"})));
    let text = audit["entries"].as_array().unwrap().iter().map(|e| e["fields"]["detail"].as_str().unwrap().to_string()).collect::<Vec<_>>().join("\n");
    assert!(text.contains("took the resources of") && text.contains("dead"), "{text}");
    assert_eq!(audit["entries"][0]["sender"], name_b);
    child.wait().unwrap();
}

#[test]
fn a_reused_pid_does_not_keep_a_lease_alive() {
    let child_pid = {
        let mut c = sleeper();
        let pid = c.id();
        c.kill().unwrap();
        c.wait().unwrap();
        pid
    };
    assert!(!live::is_alive(child_pid, None));
    let mut c = sleeper();
    let start = live::start_of(c.id()).unwrap();
    assert!(live::is_alive(c.id(), Some(start)));
    assert!(!live::is_alive(c.id(), Some(start + 5000)), "a process that started at another moment is not the holder");
    c.kill().unwrap();
    c.wait().unwrap();
}

#[test]
fn renewing_extends_a_lease_and_silence_lets_it_expire() {
    let (_d, mut n, (_, ta), (_, tb)) = two();
    let t = clocked(&mut n);
    let held = acquire(&mut n, &ta, json!({"resources": [gpu()], "ttl_s": 10}));
    let waiting = acquire(&mut n, &tb, json!({"resources": [gpu()], "ttl_s": 10}));
    advance(&t, 8);
    ok(req(&mut n, "lease_renew", "demo", &ta, json!({"id": held["id"]})));
    ok(req(&mut n, "lease_renew", "demo", &tb, json!({"id": waiting["id"]})));
    advance(&t, 8);
    assert_eq!(state(&mut n, &ta, &held["id"]), "held", "renewed at 8s, so alive at 16s");
    assert_eq!(state(&mut n, &tb, &waiting["id"]), "queued");
    advance(&t, 11);
    assert_eq!(state(&mut n, &tb, &waiting["id"]), "held", "the silent holder expired at 26s and b took over");
    assert_eq!(code(&req(&mut n, "lease_renew", "demo", &ta, json!({"id": held["id"]}))), "invalid", "an expired lease cannot be renewed");
}

#[test]
fn claims_conflict_on_the_same_worktree_and_on_a_nested_path() {
    let (_d, mut n, (_, ta), (_, tb)) = two();
    let wt = |p: &str| json!([{"type": "claim", "kind": "worktree", "name": p}]);
    let mine = acquire(&mut n, &ta, json!({"resources": wt("/repos/x/wt1")}));
    assert_eq!(mine["state"], "held");
    assert_eq!(acquire(&mut n, &ta, json!({"resources": wt("/repos/x/wt1")}))["id"], mine["id"], "the holder asking again gets its own lease");
    assert_eq!(acquire(&mut n, &tb, json!({"resources": wt("/repos/x/wt1"), "wait": false}))["state"], "busy");
    assert_eq!(acquire(&mut n, &tb, json!({"resources": wt("/repos/x/wt1/src"), "wait": false}))["state"], "busy");
    assert_eq!(acquire(&mut n, &tb, json!({"resources": wt("/repos/x/wt10"), "wait": false}))["state"], "held", "a sibling with a longer name is not nested");
    let branch = json!([{"type": "claim", "kind": "branch", "name": "feature"}]);
    assert_eq!(acquire(&mut n, &ta, json!({"resources": branch}))["state"], "held");
    assert_eq!(acquire(&mut n, &tb, json!({"resources": branch, "wait": false}))["state"], "busy");
    assert_eq!(code(&req(&mut n, "lease_acquire", "demo", &tb, json!({"resources": [{"type": "claim", "kind": "port", "name": "99999"}]}))), "invalid");
}

#[test]
fn memory_admission_refuses_over_the_budget_and_queues_over_what_is_left() {
    let (_d, mut n, (_, ta), (_, tb)) = two();
    n.leases.cfg.memory_mb = 1000;
    let mem = |mb: u64| json!([{"type": "memory_mb", "mb": mb}]);
    assert_eq!(code(&req(&mut n, "lease_acquire", "demo", &ta, json!({"resources": mem(1500)}))), "quota");
    let first = acquire(&mut n, &ta, json!({"resources": mem(600)}));
    assert_eq!(first["state"], "held");
    let second = acquire(&mut n, &tb, json!({"resources": mem(600)}));
    assert_eq!(second["state"], "queued");
    release(&mut n, &ta, &first["id"]);
    assert_eq!(state(&mut n, &tb, &second["id"]), "held");
}

#[test]
fn a_model_slot_is_one_holder_per_model_and_keeps_its_shape() {
    let (_d, mut n, (_, ta), (_, tb)) = two();
    let slot = |ctx: u64| json!([{"type": "model_slot", "model": "qwen-27b", "context": ctx, "draft": "head", "parallel": 1}]);
    let held = acquire(&mut n, &ta, json!({"resources": slot(32768)}));
    assert_eq!(held["resources"][0]["context"], 32768);
    assert_eq!(acquire(&mut n, &tb, json!({"resources": slot(8192), "wait": false}))["state"], "busy");
}

#[test]
fn a_lease_survives_a_restart_and_a_stale_holder_is_dropped() {
    let dir = tempfile::tempdir().unwrap();
    let (tok, live_id, dead_id, lapsed_id);
    let (mut alive, mut doomed) = (sleeper(), sleeper());
    {
        let mut n = Node::open(dir.path()).unwrap();
        tok = session(&mut n, "demo", "a").1;
        let other = session(&mut n, "demo", "b").1;
        let third = session(&mut n, "demo", "c").1;
        let lease = |n: &mut Node, t: &str, name: &str, pid: u32, ttl: u64| acquire(n, t, json!({"resources": [{"type": "claim", "kind": "server", "name": name}], "pid": pid, "ttl_s": ttl}))["id"].clone();
        live_id = lease(&mut n, &tok, "alive", alive.id(), 3600);
        dead_id = lease(&mut n, &other, "doomed", doomed.id(), 3600);
        lapsed_id = lease(&mut n, &third, "lapsed", 0, 1);
    }
    doomed.kill().unwrap();
    doomed.wait().unwrap();
    std::thread::sleep(std::time::Duration::from_millis(1100));
    let mut n = Node::open(dir.path()).unwrap();
    let listed = ok(req(&mut n, "lease_list", "demo", &tok, json!({})));
    let ids: Vec<&Value> = listed["leases"].as_array().unwrap().iter().map(|l| &l["id"]).collect();
    assert!(ids.contains(&&live_id), "the live holder's lease survived the restart");
    assert!(!ids.contains(&&dead_id), "the dead holder's lease was dropped");
    assert!(!ids.contains(&&lapsed_id), "the expired lease was dropped");
    alive.kill().unwrap();
    alive.wait().unwrap();
}

#[test]
fn another_identity_cannot_renew_release_or_pose_as_the_holder() {
    let (_d, mut n, (name_a, ta), (_, tb)) = two();
    let held = acquire(&mut n, &ta, json!({"resources": [gpu()]}));
    assert_eq!(code(&req(&mut n, "lease_release", "demo", &tb, json!({"id": held["id"]}))), "denied");
    assert_eq!(code(&req(&mut n, "lease_renew", "demo", &tb, json!({"id": held["id"]}))), "denied");
    assert_eq!(code(&req(&mut n, "lease_acquire", "demo", &tb, json!({"resources": [gpu()], "holder": name_a}))), "invalid");
    assert_eq!(code(&req(&mut n, "lease_acquire", "demo", &tb, json!({"resources": [gpu()], "name": name_a}))), "denied");
    assert_eq!(code(&req(&mut n, "lease_acquire", "demo", "not-a-token", json!({"resources": [gpu()]}))), "denied");
    let other = session(&mut n, "elsewhere", "x").1;
    assert_eq!(code(&req(&mut n, "lease_acquire", "demo", &other, json!({"resources": [gpu()]}))), "denied", "a token of another board holds nothing here");
    assert_eq!(state(&mut n, &ta, &held["id"]), "held");
}

#[test]
fn leases_of_another_device_are_decided_by_that_device() {
    let (_d, mut n, (_, ta), _) = two();
    let r = req(&mut n, "lease_acquire", "demo", &ta, json!({"resources": [{"type": "gpu", "device": "laptop"}]}));
    assert_eq!(code(&r), "denied");
}

#[test]
fn grants_and_ends_are_board_entries_and_peers_merge_them_earliest_acquire_first() {
    let (da, db) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
    let (mut a, mut b) = (Node::open(da.path()).unwrap(), Node::open(db.path()).unwrap());
    let (ta, tb) = (session(&mut a, "demo", "a").1, session(&mut b, "demo", "b").1);
    let branch = json!([{"type": "claim", "kind": "branch", "name": "feature"}]);
    let gpu_a = acquire(&mut a, &ta, json!({"resources": [gpu(), branch[0].clone()]}));
    assert_eq!(gpu_a["state"], "held");
    let mine = ok(req(&mut a, "read", "demo", &ta, json!({"kind": "lease"})));
    assert_eq!(mine["entries"][0]["fields"]["action"], "acquire");
    // b never saw a's entries, so it grants the same branch to itself; the board must say a was first
    assert_eq!(acquire(&mut b, &tb, json!({"resources": branch}))["state"], "held");
    poolside_node::sync::exchange_nodes(&mut a, &mut b).unwrap();
    let entries = a.view("demo").unwrap();
    let merged = merge::view(&entries);
    let pool = merged.exclusive.get(&("pool".to_string(), "claim:branch:feature".to_string())).unwrap();
    assert!(!pool.foreign, "a's acquire is first in the total order, so a holds the branch");
    // the gpu is scoped to a's device: b only reads it, and b's own gpu would be a different holder
    assert_eq!(merged.exclusive.iter().filter(|((_, k), _)| k == "gpu:local").count(), 1);
    // a request on b for a branch only a holds waits behind a's claim on the board
    let other = json!([{"type": "claim", "kind": "branch", "name": "other"}]);
    assert_eq!(acquire(&mut a, &ta, json!({"resources": other}))["state"], "held");
    poolside_node::sync::exchange_nodes(&mut a, &mut b).unwrap();
    let tb2 = session(&mut b, "demo", "b2").1;
    assert_eq!(acquire(&mut b, &tb2, json!({"resources": other}))["state"], "queued");
    assert_eq!(acquire(&mut b, &tb2, json!({"resources": [{"type": "claim", "kind": "worktree", "name": "/only/b"}]}))["state"], "held", "device-scoped claims are not shared");
}

#[test]
fn lease_wait_over_the_socket_blocks_until_the_holder_releases() {
    use poolside_node::client::Client;
    use poolside_node::server::Server;
    let dir = tempfile::Builder::new().prefix("pn").tempdir_in("/tmp").unwrap();
    let server = Server::bind(dir.path()).unwrap();
    let handle = std::thread::spawn(move || server.serve());
    let reg = |c: &mut Client, s: &str| {
        let r = c.call("register", "demo", "", json!({"model": "claude-sonnet-5-5", "harness": "claude-code", "session": s})).unwrap();
        r["token"].as_str().unwrap().to_string()
    };
    let mut c = Client::connect(dir.path()).unwrap();
    let (ta, tb) = (reg(&mut c, "a"), reg(&mut c, "b"));
    let first = c.call("lease_acquire", "demo", &ta, json!({"resources": [gpu()]})).unwrap();
    let second = c.call("lease_acquire", "demo", &tb, json!({"resources": [gpu()]})).unwrap();
    assert_eq!(second["state"], "queued");
    let (path, id) = (dir.path().to_path_buf(), second["id"].clone());
    let waiter = std::thread::spawn(move || {
        let mut w = Client::connect(&path).unwrap();
        let started = std::time::Instant::now();
        let r = w.call("lease_wait", "demo", &tb, json!({"id": id, "timeout_ms": 10_000})).unwrap();
        (r["state"].as_str().unwrap().to_string(), started.elapsed())
    });
    std::thread::sleep(std::time::Duration::from_millis(300));
    c.call("lease_release", "demo", &ta, json!({"id": first["id"]})).unwrap();
    let (state, took) = waiter.join().unwrap();
    assert_eq!(state, "held");
    assert!(took >= std::time::Duration::from_millis(250) && took < std::time::Duration::from_secs(5), "{took:?}");
    c.call("shutdown", "demo", &ta, json!({})).unwrap();
    handle.join().unwrap().unwrap();
}
