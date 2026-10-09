//! Reading leases back from a board: what every device says it holds.
//!
//! Merge rules. Entries come in the board's total order (hybrid clock, then origin). For an
//! exclusive resource the earliest acquire in that order holds it until its own release, expiry
//! or death entry; a later acquire of the same resource loses. A device-scoped resource
//! (gpu, model slot, worktree, port, server, install, and the counted cpu and memory) is
//! scoped to the origin that wrote the entry: a device's table decides its own resources and
//! other devices only read them. Branch and area claims are about the shared repository, so
//! they are scoped to the whole pool and the earliest acquire from any device wins.

use std::collections::BTreeMap;

use serde_json::{Map, Value};

use super::table::Foreign;
use crate::fold::Entry;
use crate::row::Kind;

/// Who holds one exclusive resource, as the board says.
#[derive(Clone, Debug, PartialEq)]
pub struct Held {
    pub lease: String,
    pub holder: String,
    pub origin: String,
    pub hlc: (u64, u64),
    pub foreign: bool,
}

#[derive(Debug, Default, PartialEq)]
pub struct View {
    /// (scope, resource key) to holder; scope is an origin id or `pool`.
    pub exclusive: BTreeMap<(String, String), Held>,
    /// (origin, key) to the amount of a counted resource held (cpu slots, memory MB).
    pub counted: BTreeMap<(String, String), u64>,
}

fn pool_wide(key: &str) -> bool {
    key.starts_with("claim:branch:") || key.starts_with("claim:area:")
}

/// The (key, amount) a resource line names; None for a line that is not one.
fn parse(line: &str) -> Option<(String, Option<u64>)> {
    let counted = line.starts_with("cpu_slots:") || line.starts_with("memory_mb:");
    if counted {
        let (key, amount) = line.rsplit_once(':')?;
        return Some((key.to_string(), Some(amount.parse().ok()?)));
    }
    Some((line.split('|').next()?.to_string(), None))
}

fn lines(fields: &Map<String, Value>) -> Vec<String> {
    fields.get("resources").and_then(Value::as_array).map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect()).unwrap_or_default()
}

/// Fold `lease` entries (in board order) into what is held.
pub fn view(entries: &[Entry]) -> View {
    let mut out = View::default();
    // (scope, key) -> amounts per lease id, for counted resources
    let mut amounts: BTreeMap<(String, String), BTreeMap<String, (String, u64)>> = BTreeMap::new();
    for e in entries.iter().filter(|e| e.kind == Kind::Lease) {
        let id = e.fields.get("lease").and_then(Value::as_str).unwrap_or("").to_string();
        let action = e.fields.get("action").and_then(Value::as_str).unwrap_or("");
        let resources: Vec<(String, Option<u64>)> = lines(&e.fields).iter().filter_map(|l| parse(l)).collect();
        if action == "acquire" {
            for (key, amount) in resources {
                let scope = if pool_wide(&key) { "pool".to_string() } else { e.origin.clone() };
                match amount {
                    Some(n) => { amounts.entry((scope, key)).or_default().entry(id.clone()).or_insert((e.sender.clone(), n)); }
                    None => {
                        out.exclusive.entry((scope, key)).or_insert(Held {
                            lease: id.clone(), holder: e.sender.clone(), origin: e.origin.clone(), hlc: e.hlc, foreign: e.foreign,
                        });
                    }
                }
            }
        } else {
            out.exclusive.retain(|_, h| h.lease != id || h.holder != e.sender);
            for per in amounts.values_mut() {
                if per.get(&id).is_some_and(|(who, _)| *who == e.sender) {
                    per.remove(&id);
                }
            }
        }
    }
    out.counted = amounts.into_iter().map(|(k, per)| (k, per.values().map(|(_, n)| n).sum())).filter(|(_, n)| *n > 0).collect();
    out
}

/// The pool-wide claims another device holds on this board: what a local request for the
/// same thing must wait behind.
pub fn foreign_holds(board: &str, entries: &[Entry]) -> Foreign {
    view(entries).exclusive.into_iter().filter(|((scope, _), h)| scope == "pool" && h.foreign).map(|((_, key), h)| ((board.to_string(), key), h.holder)).collect()
}
