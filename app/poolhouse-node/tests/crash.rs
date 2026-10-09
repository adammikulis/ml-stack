//! Two real node processes; one is killed with SIGKILL (TerminateProcess on Windows) while it is taking rows from the other.

mod kit;

use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use kit::netkit::eventually;
use poolhouse_node::client::Client;
use serde_json::{json, Value};
use tempfile::TempDir;

struct Proc {
    state: TempDir,
    child: Child,
}

fn start(state: &Path) -> Child {
    let child = Command::new(env!("CARGO_BIN_EXE_poolhouse-node"))
        .args(["run", "--state"]).arg(state).args(["--listen", "127.0.0.1:0"])
        .stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::null()).spawn().unwrap();
    eventually("the node to answer", 20, || Client::connect(state).and_then(|mut c| c.call("hello", "", "", json!({}))).is_ok());
    child
}

impl Proc {
    fn new() -> Proc {
        let state = tempfile::tempdir().unwrap();
        let child = start(state.path());
        Proc { state, child }
    }

    fn call(&self, method: &str, board: &str, token: &str, params: Value) -> Value {
        Client::connect(self.state.path()).unwrap().call(method, board, token, params).unwrap_or_else(|e| panic!("{method}: {e}"))
    }

    fn port(&self) -> u64 {
        let listen = self.call("pool_status", "", "", json!({}))["listen"].as_str().unwrap().to_string();
        listen.rsplit(':').next().unwrap().parse().unwrap()
    }

    fn kill9(&mut self) {
        self.child.kill().unwrap();
        self.child.wait().unwrap();
    }

    fn restart(&mut self) {
        self.child = start(self.state.path());
    }

    fn log_of(&self, origin: &str) -> PathBuf {
        self.state.path().join("boards").join("demo").join("log").join(format!("{origin}.jsonl"))
    }
}

impl Drop for Proc {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn register(p: &Proc, session: &str) -> (String, String, String) {
    let r = p.call("register", "demo", "", json!({"model": "claude-sonnet-5-5", "harness": "claude-code", "session": session}));
    (r["name"].as_str().unwrap().into(), r["token"].as_str().unwrap().into(), r["origin"].as_str().unwrap().into())
}

fn messages(p: &Proc, token: &str) -> usize {
    let mut since = json!({});
    let mut count = 0;
    loop {
        let r = p.call("read", "demo", token, json!({"since": since, "limit": 1000}));
        let entries = r["entries"].as_array().unwrap();
        count += entries.iter().filter(|e| e["kind"] == "message").count();
        if entries.is_empty() {
            return count;
        }
        since = r["cursor"].clone();
    }
}

#[test]
fn a_node_killed_with_sigkill_while_taking_rows_recovers_and_converges() {
    const ROWS: usize = 900;
    let (a, mut b) = (Proc::new(), Proc::new());
    let (_, at, a_origin) = register(&a, "writer-a");
    let (_, bt, _) = register(&b, "writer-b");
    for i in 0..ROWS {
        a.call("post", "demo", &at, json!({"kind": "message", "fields": {"type": "status", "body": format!("a row {i}")}}));
    }
    let code = a.call("pair_accept", "", &at, json!({}))["code"].as_str().unwrap().to_string();
    b.call("pair_start", "", &bt, json!({"host": "127.0.0.1", "port": a.port(), "passphrase": code}));
    let copy = b.log_of(&a_origin);

    // B starts taking A's rows; kill it the moment its copy has begun to grow.
    let state = b.state.path().to_path_buf();
    let token = bt.clone();
    let syncing = std::thread::spawn(move || Client::connect(&state).and_then(|mut c| c.call("sync_now", "", &token, json!({}))));
    let end = Instant::now() + Duration::from_secs(30);
    let size = |p: &Path| std::fs::metadata(p).map_or(0, |m| m.len());
    while size(&copy) == 0 && Instant::now() < end {
        std::thread::sleep(Duration::from_micros(300));
    }
    b.kill9();
    let _ = syncing.join();
    let held = size(&copy);
    assert!(held > 0, "the sync never began");
    assert!(held < size(&a.log_of(&a_origin)), "the sync had already finished; the kill did not land in the middle of it");

    // A torn tail, as a SIGKILL in the middle of a write leaves: the last line has no newline.
    let bytes = std::fs::read(&copy).unwrap();
    std::fs::write(&copy, &bytes[..bytes.len().saturating_sub(40)]).unwrap();

    b.restart();
    assert_eq!(b.call("hello", "", "", json!({}))["node"], "poolhouse-node", "the node starts again over the torn copy");
    b.call("sync_now", "", &bt, json!({}));
    assert_eq!(messages(&b, &bt), ROWS, "every row arrived, none twice");
    b.call("post", "demo", &bt, json!({"kind": "message", "fields": {"type": "status", "body": "b after the crash"}}));
    b.call("sync_now", "", &bt, json!({}));
    assert_eq!(messages(&a, &at), ROWS + 1, "and what b wrote after the crash reached a");
    assert_eq!(b.call("status", "demo", &bt, json!({}))["board"]["damaged"], json!({}), "a recovered copy is not a damaged one");
}
