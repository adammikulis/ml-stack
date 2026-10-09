//! Two real `poolhouse-node` processes on one machine, each with its own state directory, find each
//! other by the signed beacon, enrol under `open` and share a board. Nothing here is in-process.

mod kit;

use std::net::UdpSocket;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::Duration;

use kit::netkit::eventually;
use kit::short_dir;
use poolhouse_node::client::Client;
use serde_json::{json, Value};
use tempfile::TempDir;

const BIN: &str = env!("CARGO_BIN_EXE_poolhouse-node");

struct Proc {
    child: Child,
    dir: TempDir,
    token: String,
}

impl Proc {
    fn start(args: &[String], session: &str) -> Proc {
        let dir = short_dir();
        let log = std::fs::File::create(dir.path().join("out.log")).unwrap();
        let child = Command::new(BIN).arg("run").arg("--state").arg(dir.path().join("s")).args(args)
            .stdin(Stdio::null()).stdout(log.try_clone().unwrap()).stderr(log).spawn().unwrap();
        let mut p = Proc { child, dir, token: String::new() };
        eventually("the node to answer", 15, || Client::connect(&p.state()).is_ok());
        p.token = p.call("register", "demo", "", json!({"model": "claude-sonnet-5-5", "harness": "claude-code", "session": session}))["token"].as_str().unwrap().into();
        p
    }

    fn state(&self) -> PathBuf {
        self.dir.path().join("s")
    }

    fn call(&self, method: &str, board: &str, token: &str, params: Value) -> Value {
        Client::connect(&self.state()).unwrap().call(method, board, token, params).unwrap_or_else(|e| panic!("{method}: {e}"))
    }

    fn status(&self) -> Value {
        self.call("pool_status", "", "", json!({}))
    }

    fn member(&self, fp: &str) -> Option<Value> {
        self.status()["members"].as_array().unwrap().iter().find(|m| m["fingerprint"] == fp && m["status"] == "active").cloned()
    }

    fn say(&self, text: &str) {
        self.call("post", "demo", &self.token, json!({"kind": "message", "fields": {"type": "status", "body": text}}));
    }

    fn texts(&self) -> Vec<String> {
        let r = self.call("read", "demo", &self.token, json!({"limit": 1000}));
        let mut t: Vec<String> = r["entries"].as_array().unwrap().iter().filter(|e| e["kind"] == "message").map(|e| e["fields"]["body"].as_str().unwrap().to_string()).collect();
        t.sort();
        t
    }
}

impl Drop for Proc {
    fn drop(&mut self) {
        let _ = Client::connect(&self.state()).and_then(|mut c| c.call("shutdown", "demo", &self.token, json!({})));
        std::thread::sleep(Duration::from_millis(200));
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn free_udp() -> u16 {
    UdpSocket::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port()
}

fn loopback_args(bind: u16, to: u16) -> Vec<String> {
    ["--listen", "127.0.0.1:0", "--beacon-bind", &format!("127.0.0.1:{bind}"), "--beacon-send", &format!("127.0.0.1:{to}"),
     "--advertise", "127.0.0.1", "--allow-loopback", "--sync-ms", "300"].iter().map(|s| s.to_string()).collect()
}

fn converge(a: &Proc, b: &Proc) {
    let (fa, fb) = (a.status()["fingerprint"].as_str().unwrap().to_string(), b.status()["fingerprint"].as_str().unwrap().to_string());
    eventually("both processes to list each other as active members", 30, || a.member(&fb).is_some() && b.member(&fa).is_some());
    assert_eq!(a.status()["pool"], b.status()["pool"], "one pool");
    let by = [a.member(&fb).unwrap()["by"].clone(), b.member(&fa).unwrap()["by"].clone()];
    assert!(by.contains(&json!("open")), "the enrolment is recorded as automatic: {by:?}");
    eventually("the boards to converge", 30, || a.texts() == vec!["from a", "from b"] && b.texts() == vec!["from a", "from b"]);
}

fn open(p: &Proc) {
    p.call("set_join_policy", "", &p.token, json!({"policy": "open"}));
}

#[test]
fn two_processes_find_each_other_by_unicast_beacon_enrol_under_open_and_converge() {
    let (pa, pb) = (free_udp(), free_udp());
    let a = Proc::start(&loopback_args(pa, pb), "x");
    let b = Proc::start(&loopback_args(pb, pa), "y");
    a.say("from a");
    b.say("from b");
    open(&a);
    open(&b);
    converge(&a, &b);
}

#[test]
fn under_secure_two_processes_hear_each_other_and_enrol_nobody() {
    let (pa, pb) = (free_udp(), free_udp());
    let a = Proc::start(&loopback_args(pa, pb), "x");
    let b = Proc::start(&loopback_args(pb, pa), "y");
    std::thread::sleep(Duration::from_millis(2500));
    let fb = b.status()["fingerprint"].as_str().unwrap().to_string();
    assert!(a.member(&fb).is_none() && b.member(&a.status()["fingerprint"].as_str().unwrap().to_string()).is_none());
    assert_ne!(a.status()["pool"], b.status()["pool"]);
}

// ---- OPT-IN BELOW: every test after this line is #[ignore]d. They use the machine's network (every interface,
// multicast, broadcast) and so make macOS ask the person for permission; `loopback_only.rs` enforces the split.

/// The multicast code path (the group, the join, the sender's interface).
#[test]
#[ignore = "binds 0.0.0.0 and joins a multicast group: macOS asks the person for network permission; run only with --ignored, at the machine"]
fn two_processes_join_by_multicast_beacon_on_the_loopback_interface() {
    let beacon_port = free_udp();
    let args = || {
        ["--listen", "127.0.0.1:0", "--beacon-bind", &format!("0.0.0.0:{beacon_port}"), "--beacon-send", &format!("239.255.116.1:{beacon_port}"),
         "--multicast-if", "127.0.0.1", "--advertise", "127.0.0.1", "--allow-loopback", "--sync-ms", "300"].iter().map(|s| s.to_string()).collect::<Vec<_>>()
    };
    let a = Proc::start(&args(), "x");
    let b = Proc::start(&args(), "y");
    a.say("from a");
    b.say("from b");
    open(&a);
    open(&b);
    converge(&a, &b);
    assert!(a.status()["beacon_own"].as_u64().unwrap() > 0, "a node hears its own beacon on the group: {}", a.status());
}

