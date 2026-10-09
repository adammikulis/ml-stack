mod kit;

use poolside_node::client::{ensure_running, Client, Started};
use poolside_node::error::Error;
use poolside_node::server::{socket_path, Server};
use poolside_node::wire::my_uid;
use serde_json::json;
use std::io::{Read, Write};
use std::os::unix::fs::PermissionsExt;
use std::os::unix::net::UnixStream;
use std::path::Path;
use tempfile::TempDir;

const BIN: &str = env!("CARGO_BIN_EXE_poolside-node");

fn short_dir() -> TempDir {
    tempfile::Builder::new().prefix("pn").tempdir_in("/tmp").unwrap()
}

fn register(c: &mut Client, board: &str, session: &str) -> (String, String) {
    let r = c.call("register", board, "", json!({"model": "claude-sonnet-5-5", "harness": "claude-code", "session": session})).unwrap();
    (r["name"].as_str().unwrap().into(), r["token"].as_str().unwrap().into())
}

fn stop(state: &Path, token: &str, board: &str) {
    let _ = Client::connect(state).unwrap().call("shutdown", board, token, json!({}));
    for _ in 0..200 {
        if Client::connect(state).is_err() || UnixStream::connect(socket_path(state)).is_err() {
            return;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    panic!("the node did not stop");
}

#[test]
fn a_round_trip_over_the_socket_with_owner_only_modes() {
    let dir = short_dir();
    let server = Server::bind(dir.path()).unwrap();
    let handle = std::thread::spawn(move || server.serve());
    let mut c = Client::connect(dir.path()).unwrap();
    assert_eq!(c.call("hello", "", "", json!({})).unwrap()["node"], "poolside-node");
    let (name, token) = register(&mut c, "demo", "s1");
    c.call("post", "demo", &token, json!({"kind": "message", "fields": {"type": "status", "body": "over the wire"}})).unwrap();
    let read = c.call("read", "demo", &token, json!({"kind": "message"})).unwrap();
    assert_eq!(read["entries"][0]["sender"], name);
    let mode = |p: &Path| std::fs::metadata(p).unwrap().permissions().mode() & 0o777;
    assert_eq!(mode(&socket_path(dir.path())), 0o600);
    assert_eq!(mode(dir.path()), 0o700);
    c.call("shutdown", "demo", &token, json!({})).unwrap();
    handle.join().unwrap().unwrap();
}

#[test]
fn a_peer_with_another_uid_is_refused() {
    let dir = short_dir();
    let server = Server::bind(dir.path()).unwrap().allow_only(my_uid() + 1);
    let node = server.node();
    std::thread::spawn(move || server.serve());
    let mut c = Client::connect(dir.path()).unwrap();
    assert!(matches!(c.call("hello", "", "", json!({})), Err(Error::Denied(m)) if m.contains("another user")));
    node.lock().unwrap().stop.store(true, std::sync::atomic::Ordering::SeqCst);
}

#[test]
fn only_one_node_runs_per_state_directory() {
    let dir = short_dir();
    let first = Server::bind(dir.path()).unwrap();
    assert!(matches!(Server::bind(dir.path()), Err(Error::Denied(_))));
    drop(first);
    assert!(Server::bind(dir.path()).is_ok(), "the lock is released with the node");
}

#[test]
fn a_bad_frame_closes_that_connection_and_not_the_node() {
    let dir = short_dir();
    let server = Server::bind(dir.path()).unwrap();
    let node = server.node();
    std::thread::spawn(move || server.serve());
    let mut raw = UnixStream::connect(socket_path(dir.path())).unwrap();
    raw.write_all(&(u32::MAX).to_be_bytes()).unwrap();
    let mut rest = Vec::new();
    let _ = raw.read_to_end(&mut rest);
    assert!(rest.is_empty());
    let mut garbled = UnixStream::connect(socket_path(dir.path())).unwrap();
    garbled.write_all(&5u32.to_be_bytes()).unwrap();
    garbled.write_all(b"[not ").unwrap();
    let _ = garbled.read_to_end(&mut rest);
    assert!(Client::connect(dir.path()).unwrap().call("hello", "", "", json!({})).is_ok());
    node.lock().unwrap().stop.store(true, std::sync::atomic::Ordering::SeqCst);
}

#[test]
fn many_clients_starting_the_node_at_once_start_exactly_one() {
    let dir = short_dir();
    let state = dir.path().to_path_buf();
    let threads: Vec<_> = (0..8).map(|_| {
        let state = state.clone();
        std::thread::spawn(move || ensure_running(&state, Path::new(BIN)).unwrap())
    }).collect();
    let started: Vec<Started> = threads.into_iter().map(|t| t.join().unwrap()).collect();
    assert_eq!(started.iter().filter(|s| **s == Started::Spawned).count(), 1, "{started:?}");
    let pid = Client::connect(&state).unwrap().call("hello", "", "", json!({})).unwrap()["pid"].as_u64().unwrap();
    assert_ne!(pid, std::process::id() as u64);
    let (_, token) = register(&mut Client::connect(&state).unwrap(), "demo", "s");
    stop(&state, &token, "demo");
}

#[test]
fn a_killed_node_is_restarted_on_demand_and_loses_nothing() {
    let dir = short_dir();
    let state = dir.path().to_path_buf();
    assert_eq!(ensure_running(&state, Path::new(BIN)).unwrap(), Started::Spawned);
    let mut c = Client::connect(&state).unwrap();
    let (name, token) = register(&mut c, "demo", "s");
    for i in 0..5 {
        c.call("post", "demo", &token, json!({"kind": "message", "fields": {"type": "status", "body": format!("m{i}")}})).unwrap();
    }
    let pid = c.call("hello", "", "", json!({})).unwrap()["pid"].as_i64().unwrap();
    // SAFETY: kill -9 of the node process this test started.
    assert_eq!(unsafe { libc::kill(pid as i32, libc::SIGKILL) }, 0);
    for _ in 0..200 {
        if Client::connect(&state).is_err() {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    assert!(socket_path(&state).exists(), "the dead node left its socket behind");
    assert_eq!(ensure_running(&state, Path::new(BIN)).unwrap(), Started::Spawned);
    let mut c = Client::connect(&state).unwrap();
    let read = c.call("read", "demo", &token, json!({"kind": "message"})).unwrap();
    assert_eq!(read["entries"].as_array().unwrap().len(), 5, "committed entries and the token survive kill -9");
    assert_eq!(read["entries"][0]["sender"], name);
    stop(&state, &token, "demo");
}
