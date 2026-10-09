//! Test shards between two real devices of one pool (TLS on loopback, no multicast): who may send,
//! what is refused, the lease, the kill of the whole process group. The interpreter is a shell stub
//! that answers the version probe and plays the executor, so no Python is needed.

#![cfg(unix)]

mod kit;

use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::sync::Once;

use kit::netkit::*;
use poolhouse_node::cert::Identity;
use poolhouse_node::error::Error;
use poolhouse_node::fsutil::{hex, sha256_hex};
use poolhouse_node::peer::PeerClient;
use serde_json::{json, Value};
use tempfile::TempDir;

const STUB: &str = r#"#!/bin/sh
if [ "$1" = "-c" ]; then echo "${STUB_VERSION:-3.13}"; exit 0; fi
dir="$5"
tier=$(sed -n 's/.*"tier":"\([a-z]*\)".*/\1/p' "$dir/spec.json")
case "$tier" in
  all) printf '{"state":"done","exit":0,"seen":"%s%s","files":{}}' "${CLAUDECODE-unset}" "${PYTHONPATH-unset}" > "$dir/result.json" ;;
  slow) sleep 300 & echo $! > "$dir/grandchild.pid"; echo $$ > "$dir/executor.pid"; wait ;;
  gate) printf '{"state":"done","exit":0,"files":{}}' > "$dir/result.json.part"; echo $$ > "$dir/executor.pid"; sleep 300 ;;
  *) echo "the executor stub stops here" ;;
esac
"#;

static MARKER: Once = Once::new();

struct Rig {
    a: Dev,
    b: Dev,
    tools: TempDir,
}

impl Rig {
    fn python(&self) -> String {
        self.tools.path().join("python").to_string_lossy().into_owned()
    }

    fn repo(&self) -> String {
        self.tools.path().join("repo").to_string_lossy().into_owned()
    }

    /// `b` with shards switched on, the way its person would.
    fn enable(&self) -> Value {
        self.b.ok("shard_consent", "", &self.b.token, json!({"enabled": true, "python": self.python(), "repo": self.repo(), "allow": [self.a.fp()]}))
    }

    /// Ask `b` something as `a`'s session.
    fn call(&self, op: &str, args: Value) -> Value {
        self.a.call("shard_call", "", &self.a.token, json!({"device": self.b.fp(), "op": op, "args": args}))
    }

    fn ok(&self, op: &str, args: Value) -> Value {
        let r = self.call(op, args);
        assert_eq!(r["ok"], true, "{op}: {r}");
        r["result"].clone()
    }

    fn code(&self, op: &str, args: Value) -> String {
        let r = self.call(op, args);
        assert_eq!(r["ok"], false, "{op} should be refused: {r}");
        r["error"]["code"].as_str().unwrap().to_string()
    }

    /// Upload ``tree`` in chunks and start it.
    fn run(&self, id: &str, tree: &[u8], tier: &str, files: &[&str], seconds: u64) -> Value {
        for (i, chunk) in tree.chunks(100).enumerate() {
            self.ok("shard_put", json!({"id": id, "offset": i * 100, "data": hex(chunk)}));
        }
        self.call("shard_start", json!({"id": id, "tree_sha256": sha256_hex(tree), "size": tree.len(), "tier": tier, "files": files, "timeout_s": seconds}))
    }

    fn state(&self, id: &str) -> Value {
        self.ok("shard_status", json!({"id": id}))
    }

    fn wait_for(&self, id: &str, want: &str) -> Value {
        let mut last = Value::Null;
        eventually(&format!("shard {id} reaches {want}"), 30, || {
            last = self.state(id);
            last["state"] == want
        });
        last
    }
}

fn id(n: u8) -> String {
    format!("{n:02x}").repeat(16)
}

fn rig() -> Rig {
    MARKER.call_once(|| {
        // SAFETY: set once before any test thread reads it; every test in this file starts here.
        unsafe {
            std::env::set_var("CLAUDECODE", "1");
            std::env::set_var("PYTHONPATH", "/definitely/not/the/trees");
        }
    });
    let tools = tempfile::tempdir().unwrap();
    std::fs::write(tools.path().join("python"), STUB).unwrap();
    std::fs::set_permissions(tools.path().join("python"), std::fs::Permissions::from_mode(0o755)).unwrap();
    let executor = tools.path().join("repo/src/poolhouse/fleet");
    std::fs::create_dir_all(&executor).unwrap();
    std::fs::write(executor.join("shard_exec.py"), "# the stub never reads this\n").unwrap();
    let (a, b) = (device(), device());
    a.pair_secure(&b);
    Rig { a, b, tools }
}

fn dead(pid: i32) -> bool {
    // SAFETY: signal 0 only asks whether the process exists.
    unsafe { libc::kill(pid, 0) != 0 }
}

fn pid_in(dir: &Path, name: &str) -> i32 {
    let path: PathBuf = dir.join(name);
    eventually(&format!("{name} is written"), 20, || std::fs::read_to_string(&path).is_ok_and(|t| t.trim().parse::<i32>().is_ok()));
    std::fs::read_to_string(path).unwrap().trim().parse().unwrap()
}

fn job_dir(rig: &Rig, id: &str) -> PathBuf {
    rig.b.dir.path().join("shards").join(id)
}

fn stranger_client(rig: &Rig, who: &Identity) -> PeerClient {
    PeerClient::connect(who, rig.b.addr(), Some(&rig.b.fp())).unwrap()
}

// -- consent ------------------------------------------------------------------

#[test]
fn a_device_nobody_enabled_takes_nothing_and_says_how_to_turn_it_on() {
    let rig = rig();
    let caps = rig.ok("shard_caps", json!({}));
    assert_eq!(caps["accepts"], false);
    assert!(caps["reason"].as_str().unwrap().contains("testfarm.consent"), "{caps}");
    assert_eq!(rig.code("shard_put", json!({"id": id(1), "offset": 0, "data": "00"})), "denied");
    assert!(!job_dir(&rig, &id(1)).exists(), "a refused upload leaves nothing on the disk");
    assert_eq!(rig.b.ok("shard_consent", "", "", json!({}))["enabled"], false);
}

#[test]
fn enabling_is_an_audited_action_by_a_registered_session_and_checks_the_python() {
    let rig = rig();
    let no_token = rig.b.call("shard_consent", "", "", json!({"enabled": true, "python": rig.python(), "repo": rig.repo()}));
    assert_eq!(no_token["error"]["code"], "denied", "{no_token}");
    let wrong = rig.b.call("shard_consent", "", &rig.b.token, json!({"enabled": true, "python": "relative/python", "repo": rig.repo()}));
    assert_eq!(wrong["error"]["code"], "invalid", "{wrong}");
    let not_a_checkout = rig.b.call("shard_consent", "", &rig.b.token, json!({"enabled": true, "python": rig.python(), "repo": rig.tools.path().to_string_lossy()}));
    assert!(not_a_checkout["error"]["message"].as_str().unwrap().contains("shard_exec.py"), "{not_a_checkout}");
    let saved = rig.enable();
    assert_eq!((saved["enabled"].clone(), saved["python"].clone()), (json!(true), json!(rig.python())));
    assert_eq!(rig.ok("shard_caps", json!({}))["accepts"], true);
    let trail = rig.b.node.lock().unwrap().view("pool").unwrap();
    let said = trail.iter().find(|e| e.fields.get("event").and_then(Value::as_str) == Some("shard_consent")).expect("the pool board records it");
    assert!(said.fields["detail"].as_str().unwrap().contains("demo/"), "{:?}", said.fields);
    rig.b.ok("shard_consent", "", &rig.b.token, json!({"enabled": false}));
    assert_eq!(rig.ok("shard_caps", json!({}))["accepts"], false, "off stops the next request at once");
}

#[test]
fn a_python_that_is_not_3_13_is_refused_with_what_to_do() {
    let rig = rig();
    let old = rig.tools.path().join("python312");
    std::fs::write(&old, "#!/bin/sh\necho 3.12\n").unwrap();
    std::fs::set_permissions(&old, std::fs::Permissions::from_mode(0o755)).unwrap();
    let r = rig.b.call("shard_consent", "", &rig.b.token, json!({"enabled": true, "python": old.to_string_lossy(), "repo": rig.repo()}));
    let message = r["error"]["message"].as_str().unwrap();
    assert!(message.contains("Python 3.13 is needed") && message.contains("is 3.12") && message.contains("testfarm.consent on"), "{r}");
    assert_eq!(rig.b.ok("shard_consent", "", "", json!({}))["enabled"], false, "nothing was saved");
    let gone = poolhouse_node::shard::runtime::python_version("/definitely/not/a/python").unwrap_err().to_string();
    assert!(gone.contains("Python 3.13 is missing") && gone.contains("testfarm.consent on"), "{gone}");
    let none = poolhouse_node::shard::runtime::python_version("").unwrap_err().to_string();
    assert!(none.contains("no Python is set"), "{none}");
}

// -- who may ask --------------------------------------------------------------

#[test]
fn only_pool_members_reach_the_shard_ops_and_a_revoked_one_is_put_out_at_its_next_request() {
    let rig = rig();
    rig.enable();
    let stranger = kit_identity(40);
    let mut client = stranger_client(&rig, &stranger);
    for op in poolhouse_node::shard::OPS {
        assert!(matches!(client.call(&json!({"op": op, "id": id(2)})), Err(Error::Denied(_))), "{op} from a stranger");
    }
    assert!(!job_dir(&rig, &id(2)).exists());
    let c = device();
    rig.a.pair_secure(&c);
    let mut member = PeerClient::connect(&c.net.me, rig.a.addr(), Some(&rig.a.fp())).unwrap();
    assert_eq!(member.call(&json!({"op": "shard_caps"})).unwrap()["accepts"], false, "a member of the pool is answered");
    rig.a.ok("member_revoke", "", &rig.a.token, json!({"fingerprint": c.fp()}));
    assert!(matches!(member.call(&json!({"op": "shard_caps"})), Err(Error::Denied(_))), "revoked on the connection it already held");
}

#[test]
fn the_requester_is_the_certificate_never_a_field_of_the_request() {
    let rig = rig();
    rig.enable();
    let i = id(3);
    rig.ok("shard_put", json!({"id": i, "offset": 0, "data": "6162"}));
    let local = rig.a.call("shard_call", "", &rig.a.token, json!({"device": rig.b.fp(), "op": "shard_status", "args": {"id": i, "by": "claude-lead", "op": "shard_put"}}));
    assert_eq!(local["error"]["code"], "denied", "a caller cannot name who asks: {local}");
    let c = device();
    rig.a.pair_secure(&c);
    rig.b.ok("sync_now", "", &rig.b.token, json!({}));
    rig.b.ok("shard_consent", "", &rig.b.token, json!({"allow": [c.fp()]}));
    let mut other = PeerClient::connect(&c.net.me, rig.b.addr(), Some(&rig.b.fp())).unwrap();
    {
        assert!(other.call(&json!({"op": "shard_caps"})).is_ok(), "c is a member of b's record by now");
        let forged = other.call(&json!({"op": "shard_status", "id": i, "by": "demo/claude-lead", "requester": rig.a.fp(), "owner": rig.a.fp()}));
        assert!(matches!(forged, Err(Error::Invalid(_))), "an unknown field is refused: {forged:?}");
        let plain = other.call(&json!({"op": "shard_status", "id": i, "by": "demo/claude-lead"}));
        assert!(matches!(plain, Err(Error::Invalid(ref m)) if m.contains("no such shard")), "another member's shard does not exist for it: {plain:?}");
        let cancel = other.call(&json!({"op": "shard_cancel", "id": i}));
        assert!(matches!(cancel, Err(Error::Invalid(_))), "and cannot be cancelled by it");
    }
    assert_eq!(rig.state(&i)["state"], "uploading", "the owner still has it");
}

// -- hostile requests ---------------------------------------------------------

#[test]
fn start_refuses_every_malformed_request_and_nothing_runs() {
    let rig = rig();
    rig.enable();
    let tree = b"y".repeat(200);
    let upload = |i: u8| {
        rig.ok("shard_put", json!({"id": id(i), "offset": 0, "data": hex(&tree)}));
    };
    let base = |i: u8| json!({"id": id(i), "tree_sha256": sha256_hex(&tree), "size": 200, "tier": "all", "files": ["tests/test_a.py"], "timeout_s": 60});
    let hostile: Vec<(&str, &str, Value)> = vec![
        ("an extra field", "extra", json!("x")),
        ("an argv", "argv", json!(["rm", "-rf", "/"])),
        ("an env", "env", json!({"A": "1"})),
        ("a cwd", "cwd", json!("/")),
        ("a tier that is not listed", "tier", json!("quick")),
        ("a tier that is an option", "tier", json!("--help")),
        ("a path outside tests", "files", json!(["scripts/test"])),
        ("a parent path", "files", json!(["tests/../scripts/test.py"])),
        ("a nested path", "files", json!(["tests/sub/test_a.py"])),
        ("an absolute path", "files", json!(["/etc/passwd"])),
        ("a backslash path", "files", json!(["tests\\test_a.py"])),
        ("an option as a file", "files", json!(["--rootdir=/", "tests/test_a.py"])),
        ("a node id", "files", json!(["tests/test_a.py::test_b"])),
        ("a hidden file", "files", json!(["tests/.test_a.py"])),
        ("a repeated file", "files", json!(["tests/test_a.py", "tests/test_a.py"])),
        ("a file that is not text", "files", json!([7])),
        ("the gate with files", "tier", json!("gate")),
        ("no time limit", "timeout_s", json!(0)),
        ("a long time limit", "timeout_s", json!(99999)),
        ("a time limit that is text", "timeout_s", json!("60")),
        ("a size that is not the upload", "size", json!(201)),
        ("a huge size", "size", json!(1u64 << 40)),
        ("a digest in capitals", "tree_sha256", json!(sha256_hex(&tree).to_uppercase())),
        ("a digest that does not match", "tree_sha256", json!("0".repeat(64))),
        ("an id that is not hex", "id", json!("../../etc/passwd")),
    ];
    for (n, (what, key, value)) in hostile.into_iter().enumerate() {
        let i = 50 + n as u8;
        if key != "id" {
            upload(i);
        }
        let mut req = base(i);
        req[key] = value;
        let code = rig.code("shard_start", req);
        assert!(["invalid", "quota"].contains(&code.as_str()), "{what}: {code}");
        let after = rig.call("shard_status", json!({"id": id(i)}));
        assert!(after["ok"] == false || after["result"]["state"] == "uploading", "{what} ran something: {after}");
        let _ = rig.call("shard_cancel", json!({"id": id(i)}));
    }
    let many: Vec<String> = (0..201).map(|n| format!("tests/test_{n}.py")).collect();
    upload(9);
    assert_eq!(rig.code("shard_start", json!({"id": id(9), "tree_sha256": sha256_hex(&tree), "size": 200, "tier": "all", "files": many, "timeout_s": 60})), "invalid");
    assert!(std::fs::read_dir(rig.b.dir.path().join("shards")).unwrap().all(|e| !e.unwrap().path().join("spec.json").exists()), "no refused shard reached the executor");
}

#[test]
fn an_upload_arrives_in_order_once_and_within_its_size() {
    let rig = rig();
    rig.enable();
    let i = id(5);
    assert_eq!(rig.code("shard_put", json!({"id": i, "offset": 7, "data": "00"})), "invalid", "a tree starts at offset 0");
    rig.ok("shard_put", json!({"id": i, "offset": 0, "data": "0102"}));
    assert_eq!(rig.code("shard_put", json!({"id": i, "offset": 0, "data": "0102"})), "invalid", "a replayed chunk");
    assert_eq!(rig.code("shard_put", json!({"id": i, "offset": 5, "data": "0102"})), "invalid", "a gap");
    assert_eq!(rig.code("shard_put", json!({"id": i, "offset": 2, "data": "xyz"})), "invalid", "not hex");
    assert_eq!(rig.code("shard_put", json!({"id": i, "offset": 2, "data": ""})), "invalid", "empty");
    assert_eq!(rig.code("shard_put", json!({"id": i, "offset": 2, "data": hex(&vec![0u8; 256 * 1024 + 1])})), "invalid", "a chunk past the cap");
    assert_eq!(rig.code("shard_put", json!({"id": "../up", "offset": 0, "data": "00"})), "invalid", "a path as an id");
    assert_eq!(rig.code("shard_put", json!({"id": id(6), "offset": 0, "data": "00", "mode": "x"})), "invalid", "an unknown field");
    assert_eq!(rig.code("shard_start", json!({"id": i, "tree_sha256": sha256_hex(b"\x01\x02"), "size": 3, "tier": "all", "files": [], "timeout_s": 60})), "invalid", "start before the whole tree arrived");
    assert_eq!(rig.state(&i)["got"], 2);
    assert_eq!(rig.code("shard_cancel", json!({"id": id(77)})), "invalid");
    assert_eq!(rig.ok("shard_cancel", json!({"id": i}))["state"], "cancelled");
    assert!(!job_dir(&rig, &i).exists(), "a cancelled upload is removed from the disk");
    let big = 25usize << 20;
    let mut offset = 0usize;
    let j = id(8);
    let chunk = hex(&vec![0u8; 256 * 1024]);
    while offset + 256 * 1024 <= big {
        let r = rig.call("shard_put", json!({"id": j, "offset": offset, "data": chunk}));
        if r["ok"] == false {
            assert_eq!(r["error"]["code"], "invalid");
            assert!(offset + 256 * 1024 > 24 << 20, "refused only past 24 MiB: {offset}");
            return;
        }
        offset += 256 * 1024;
    }
    panic!("a tree past 24 MiB was taken");
}

// -- running ------------------------------------------------------------------

#[test]
fn a_shard_runs_under_a_lease_with_a_clean_environment_and_its_result_comes_back() {
    let rig = rig();
    rig.enable();
    let tree = b"tree bytes".repeat(30);
    let i = id(10);
    let started = rig.run(&i, &tree, "all", &["tests/test_a.py"], 60);
    assert_eq!(started["ok"], true, "{started}");
    let done = rig.wait_for(&i, "done");
    assert_eq!(done["result"]["exit"], 0);
    assert_eq!(done["result"]["seen"], "unsetunset", "no agent marker and no PYTHONPATH reached the run");
    let leases = rig.b.ok("lease_list", "demo", &rig.b.token, json!({}));
    assert!(leases["leases"].as_array().unwrap().is_empty(), "the lease is given back: {leases}");
    let trail = rig.b.node.lock().unwrap().view("pool").unwrap();
    let events: Vec<&str> = trail.iter().filter_map(|e| e.fields.get("event").and_then(Value::as_str)).collect();
    assert!(events.contains(&"shard_start") && events.contains(&"shard_done"), "{events:?}");
    assert!(!job_dir(&rig, &i).join("tree.tgz").exists(), "the tree is deleted once the run ends");
}

#[test]
fn a_run_that_leaves_no_result_is_failed_with_the_executors_own_words() {
    let rig = rig();
    rig.enable();
    let tree = b"z".repeat(64);
    assert_eq!(rig.run(&id(11), &tree, "full", &[], 60)["ok"], true);
    let failed = rig.wait_for(&id(11), "failed");
    assert!(failed["error"].as_str().unwrap().contains("the executor stub stops here"), "{failed}");
}

#[test]
fn at_most_two_shards_run_at_once() {
    let rig = rig();
    rig.enable();
    let tree = b"q".repeat(64);
    for n in [20, 21] {
        assert_eq!(rig.run(&id(n), &tree, "slow", &[], 600)["ok"], true);
    }
    let third = rig.run(&id(22), &tree, "slow", &[], 600);
    assert_eq!(third["error"]["code"], "quota", "{third}");
    assert_eq!(rig.ok("shard_caps", json!({}))["free"], 0);
    for n in [20, 21] {
        rig.ok("shard_cancel", json!({"id": id(n)}));
        rig.wait_for(&id(n), "cancelled");
    }
}

#[test]
fn cancel_ends_the_executor_and_everything_it_started() {
    let rig = rig();
    rig.enable();
    let tree = b"w".repeat(64);
    let i = id(30);
    assert_eq!(rig.run(&i, &tree, "slow", &[], 600)["ok"], true);
    let dir = job_dir(&rig, &i);
    let (executor, grandchild) = (pid_in(&dir, "executor.pid"), pid_in(&dir, "grandchild.pid"));
    assert!(!dead(executor) && !dead(grandchild));
    assert_eq!(rig.ok("shard_cancel", json!({"id": i}))["state"], "cancelling");
    rig.wait_for(&i, "cancelled");
    eventually("the executor and its child are gone", 30, || dead(executor) && dead(grandchild));
    let leases = rig.b.ok("lease_list", "demo", &rig.b.token, json!({}));
    assert!(leases["leases"].as_array().unwrap().is_empty(), "the lease went with it: {leases}");
}

fn kit_identity(seed: u8) -> Identity {
    let dir = tempfile::tempdir().unwrap();
    poolhouse_node::cert::load_or_create(dir.path(), &kit::key(seed)).unwrap()
}

// -- the allowed list ---------------------------------------------------------

/// A third device paired into b's pool, and a client of b as it.
fn member_of_b(rig: &Rig) -> (Dev, PeerClient) {
    let c = device();
    rig.b.pair_secure(&c);
    let client = PeerClient::connect(&c.net.me, rig.b.addr(), Some(&rig.b.fp())).unwrap();
    (c, client)
}

#[test]
fn a_member_that_is_not_allowed_is_refused_every_op_but_the_question_and_the_refusal_is_on_the_board() {
    let rig = rig();
    rig.enable();
    let (_c, mut client) = member_of_b(&rig);
    let caps = client.call(&json!({"op": "shard_caps"})).unwrap();
    assert_eq!((caps["accepts"].clone(), caps["allowed"].clone()), (json!(false), json!(false)));
    assert!(caps["reason"].as_str().unwrap().contains("consent allow"), "{caps}");
    for req in [json!({"op": "shard_put", "id": id(60), "offset": 0, "data": "00"}), json!({"op": "shard_status", "id": id(60)}),
                json!({"op": "shard_cancel", "id": id(60)}), json!({"op": "shard_start", "id": id(60)}), json!({"op": "shard_put"})] {
        assert!(matches!(client.call(&req), Err(Error::Denied(_))), "{req}");
    }
    assert!(!job_dir(&rig, &id(60)).exists());
    let trail = rig.b.node.lock().unwrap().view("pool").unwrap();
    let refused = trail.iter().filter(|e| e.fields.get("event").and_then(Value::as_str) == Some("shard_refused")).count();
    assert_eq!(refused, 1, "one entry for a burst");
    assert_eq!(rig.ok("shard_caps", json!({}))["allowed"], true, "the allowed device is answered");
}

#[test]
fn allowing_names_active_members_only_is_audited_and_denying_ends_the_runs() {
    let rig = rig();
    rig.enable();
    let stranger = device();
    let r = rig.b.call("shard_consent", "", &rig.b.token, json!({"allow": [stranger.fp()]}));
    assert_eq!(r["error"]["code"], "invalid", "a device outside the pool cannot be allowed: {r}");
    for bad in [json!(["xyz"]), json!("a"), json!([rig.b.fp()])] {
        assert_eq!(rig.b.call("shard_consent", "", &rig.b.token, json!({"allow": bad}))["ok"], false);
    }
    assert_eq!(rig.b.call("shard_consent", "", "", json!({"deny": [rig.a.fp()]}))["error"]["code"], "denied", "a token is needed");
    let (c, mut client) = member_of_b(&rig);
    rig.b.ok("shard_consent", "", &rig.b.token, json!({"allow": [c.fp()]}));
    assert_eq!(client.call(&json!({"op": "shard_status", "id": id(61)})).map_err(|e| e.code()), Err("invalid"), "allowed now");
    let trail = rig.b.node.lock().unwrap().view("pool").unwrap();
    assert!(trail.iter().any(|e| e.fields.get("event").and_then(Value::as_str) == Some("shard_allow") && e.fields["detail"].as_str().unwrap().contains("demo/")), "who allowed whom is recorded");
    let tree = b"w".repeat(64);
    assert_eq!(rig.run(&id(62), &tree, "slow", &[], 600)["ok"], true);
    let dir = job_dir(&rig, &id(62));
    let (executor, child) = (pid_in(&dir, "executor.pid"), pid_in(&dir, "grandchild.pid"));
    rig.b.ok("shard_consent", "", &rig.b.token, json!({"deny": [rig.a.fp()]}));
    eventually("the denied device's run is ended", 30, || dead(executor) && dead(child));
    assert_eq!(rig.code("shard_status", json!({"id": id(62)})), "denied");
}

#[test]
fn putting_a_member_out_of_the_pool_takes_it_off_the_allowed_list() {
    let rig = rig();
    rig.enable();
    let (c, mut client) = member_of_b(&rig);
    rig.b.ok("shard_consent", "", &rig.b.token, json!({"allow": [c.fp()]}));
    assert_eq!(rig.b.ok("shard_consent", "", "", json!({}))["allowed"].as_array().unwrap().len(), 2);
    rig.b.ok("member_revoke", "", &rig.b.token, json!({"fingerprint": c.fp()}));
    let left = rig.b.ok("shard_consent", "", "", json!({}));
    assert_eq!(left["allowed"], json!([rig.a.fp()]), "only the device still in the pool is left");
    assert!(client.call(&json!({"op": "shard_caps"})).is_err(), "and it is refused at its next request");
    let trail = rig.b.node.lock().unwrap().view("pool").unwrap();
    assert!(trail.iter().any(|e| e.fields.get("event").and_then(Value::as_str) == Some("shard_allow") && e.fields["detail"].as_str().unwrap().contains("put out of the pool")));
}
