//! Board sync over the peer connection: the client side of `peer`'s `pull`, `vector` and
//! `push`, built from the same `sync` logic the in-process exchange uses, so a row crossing the
//! wire meets the same checks (chain, hashes, heads, pinned key, size, quotas).
//!
//! A relay's copy never marks an origin damaged: the server and the client both store rows
//! through `sync::take`, which marks an origin damaged only when the sender is the device that
//! owns it.

use std::collections::BTreeMap;
use std::sync::{Mutex, MutexGuard};

use serde_json::{json, Value};

use crate::board::Tip;
use crate::error::{Error, Result};
use crate::node::{Hosted, Node};
use crate::peer::{policy_json, take_policy, PeerClient};
use crate::row::Row;
use crate::sync::{rows_since, seqs, take, Report, MAX_ROUNDS};

fn locked(node: &Mutex<Node>) -> Result<MutexGuard<'_, Node>> {
    node.lock().map_err(|_| Error::Damaged("node poisoned".into()))
}

fn with_board<T>(node: &Mutex<Node>, id: &str, f: impl FnOnce(&mut Hosted) -> Result<T>) -> Result<T> {
    let mut n = locked(node)?;
    let h = n.boards.get_mut(id).ok_or_else(|| Error::Denied("this device does not hold that board".into()))?;
    f(h)
}

fn tips(v: &Value, key: &str) -> Result<BTreeMap<String, Tip>> {
    serde_json::from_value(v.get(key).cloned().unwrap_or(json!({}))).map_err(|_| Error::Invalid(format!("{key} maps origins to tips")))
}

/// Swap pool records with the peer; how many of ours changed.
pub fn sync_members(node: &Mutex<Node>, c: &mut PeerClient) -> Result<usize> {
    let ask = {
        let n = locked(node)?;
        json!({"op": "members", "pool": n.members.id, "rows": n.members.export(), "policy": policy_json(&n)})
    };
    let reply = c.call(&ask)?;
    let rows = reply.get("rows").and_then(Value::as_array).ok_or_else(|| Error::Invalid("rows is a list".into()))?;
    let mut n = locked(node)?;
    let changed = n.members.merge(rows, &c.fingerprint)?;
    if let Some(p) = reply.get("policy") {
        take_policy(&mut n, p)?;
    }
    Ok(changed)
}

/// The boards the peer holds.
pub fn peer_boards(c: &mut PeerClient) -> Result<Vec<String>> {
    let reply = c.call(&json!({"op": "boards"}))?;
    serde_json::from_value(reply["boards"].clone()).map_err(|_| Error::Invalid("boards is a list".into()))
}

/// Make this device's copy of board ``id`` and the peer's hold the same rows.
pub fn sync_board(node: &Mutex<Node>, c: &mut PeerClient, id: &str) -> Result<Report> {
    let mut report = Report::default();
    for _ in 0..MAX_ROUNDS {
        let (origin, vector) = with_board(node, id, |h| {
            h.board.seal()?;
            Ok((h.board.origin().to_string(), h.board.vector()))
        })?;
        let reply = c.call(&json!({"op": "pull", "board": id, "origin": origin, "vector": vector}))?;
        let pulled = with_board(node, id, |h| {
            let remote = reply["origin"].as_str().unwrap_or("");
            if !crate::row::valid_origin(remote) || !h.board.bind_peer(&c.board_fingerprint, remote)? {
                return Err(Error::Denied("this device writes a different log than the one it presented before".into()));
            }
            h.board.acknowledge(&c.board_fingerprint, &tips(&reply, "trusted_vector")?)?;
            let logs: BTreeMap<String, Vec<Row>> = serde_json::from_value(reply["logs"].clone()).map_err(|_| Error::Invalid("logs map origins to rows".into()))?;
            take(&mut h.board, &logs, &c.board_fingerprint, &mut report)
        })?;
        report.pulled += pulled;
        let pushed = push_once(node, c, id, &origin, &mut report)?;
        if pulled == 0 && pushed == 0 {
            break;
        }
    }
    Ok(report)
}

fn push_once(node: &Mutex<Node>, c: &mut PeerClient, id: &str, origin: &str, report: &mut Report) -> Result<usize> {
    let reply = c.call(&json!({"op": "vector", "board": id, "origin": origin}))?;
    let remote = tips(&reply, "vector")?;
    let (logs, trusted) = with_board(node, id, |h| {
        h.board.acknowledge(&c.board_fingerprint, &remote)?;
        Ok((rows_since(&h.board, &seqs(&remote)), h.board.trusted_vector()))
    })?;
    if logs.is_empty() {
        return Ok(0);
    }
    let reply = c.call(&json!({"op": "push", "board": id, "origin": origin, "logs": logs, "trusted_vector": trusted}))?;
    let stored = reply["stored"].as_u64().unwrap_or(0) as usize;
    if let Some(refused) = reply["refused"].as_object() {
        report.refused.extend(refused.iter().map(|(k, v)| (k.clone(), v.as_str().unwrap_or("").to_string())));
    }
    report.pushed += stored;
    Ok(stored)
}

/// Sync the pool record, then every board both devices hold; origin to refusal code per board.
pub fn sync_all(node: &Mutex<Node>, c: &mut PeerClient) -> Result<BTreeMap<String, Report>> {
    sync_members(node, c)?;
    let theirs = peer_boards(c)?;
    let mine: Vec<String> = locked(node)?.boards.keys().cloned().collect();
    let mut out = BTreeMap::new();
    for id in mine.into_iter().filter(|b| theirs.contains(b)) {
        let report = sync_board(node, c, &id)?;
        out.insert(id, report);
    }
    Ok(out)
}
