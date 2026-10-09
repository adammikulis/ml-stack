#![allow(dead_code)]

pub mod netkit;

use ed25519_dalek::SigningKey;
use poolhouse_node::board::Board;
use poolhouse_node::node::Node;
use poolhouse_node::row::{Kind, Row};
use serde_json::{json, Value};
use std::path::Path;
use tempfile::TempDir;

/// A scratch directory whose path is short enough for a Unix socket inside it.
pub fn short_dir() -> TempDir {
    #[cfg(unix)]
    return tempfile::Builder::new().prefix("pn").tempdir_in("/tmp").unwrap();
    #[cfg(windows)]
    return tempfile::Builder::new().prefix("pn").tempdir().unwrap();
}

/// End the process ``pid`` without letting it clean up (kill -9; `taskkill /F` on Windows).
pub fn kill_hard(pid: u32) {
    #[cfg(unix)]
    {
        // SAFETY: kill -9 of a node process the calling test started.
        assert_eq!(unsafe { libc::kill(pid as i32, libc::SIGKILL) }, 0);
    }
    #[cfg(windows)]
    assert!(std::process::Command::new("taskkill").args(["/F", "/PID"]).arg(pid.to_string()).output().unwrap().status.success());
}

pub fn key(seed: u8) -> SigningKey {
    SigningKey::from_bytes(&[seed; 32])
}

pub fn board(dir: &TempDir, seed: u8) -> Board {
    Board::open(&dir.path().join("log"), "demo", "", key(seed)).unwrap()
}

pub fn say(board: &mut Board, actor: &str, text: &str) -> Row {
    let body = json!({"type": "status", "from": actor, "to": "#general", "subject": "", "body": text});
    board.append(Kind::Message, actor, body, "").unwrap()
}

/// Recompute a row's hash so a tampered row is still a well-formed link in its chain.
pub fn rehash(mut row: Row) -> Row {
    row.hash = row.digest();
    row
}

pub fn texts(entries: &Value) -> Vec<String> {
    entries.as_array().unwrap().iter().filter(|e| e["kind"] == "message")
        .map(|e| e["fields"]["body"].as_str().unwrap().to_string()).collect()
}

pub fn node(dir: &Path) -> Node {
    Node::open(dir).unwrap()
}

pub fn req(node: &mut Node, method: &str, board: &str, token: &str, params: Value) -> Value {
    let mut r = json!({"v": 1, "id": 1, "method": method, "params": params});
    if !board.is_empty() {
        r["board"] = json!(board);
    }
    if !token.is_empty() {
        r["token"] = json!(token);
    }
    poolhouse_node::api::handle(node, &r)
}

pub fn ok(reply: Value) -> Value {
    assert_eq!(reply["ok"], true, "{reply}");
    reply["result"].clone()
}

pub fn code(reply: &Value) -> String {
    assert_eq!(reply["ok"], false, "{reply}");
    reply["error"]["code"].as_str().unwrap().to_string()
}

/// Register a session; returns (name, token).
pub fn session(node: &mut Node, board: &str, id: &str) -> (String, String) {
    let r = ok(req(node, "register", board, "", json!({"model": "claude-sonnet-5-5", "harness": "claude-code", "session": id})));
    (r["name"].as_str().unwrap().into(), r["token"].as_str().unwrap().into())
}

pub fn post(node: &mut Node, board: &str, token: &str, text: &str) -> Value {
    req(node, "post", board, token, json!({"kind": "message", "fields": {"type": "status", "body": text}}))
}
