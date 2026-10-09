//! Test shards on the node: run a tree's tests on another device of the pool.
//!
//! A shard rides the node because the node already knows who is in the pool: peer ops reach a
//! device only over TLS 1.3 pinned to the certificate the pool record holds for it, and a revoked
//! device is refused at its next request. Nothing runs unless the person at the receiving device
//! turned shards on (`shard_consent`, written to the pool board), and the receiver builds the
//! one command it runs. See docs/test-farm.md.
//!
//! | piece | does |
//! |---|---|
//! | `consent` | `<state>/shards.json`: on or off, the Python 3.13 and the checkout that run shards |
//! | `spec` | every check on a request before anything is stored or run |
//! | `host` | the five peer ops, the lease for CPU slots, the run and its kill |
//! | `proc` | the executor's process group: start, ask to stop, kill |
//! | `runtime` | this device's platform, the Python check, the result file |

pub mod consent;
pub mod host;
pub mod proc;
pub mod runtime;
pub mod spec;

use serde_json::{json, Map, Value};

use crate::error::{Error, Result};
use crate::node::Node;
use crate::poolapi::authorize;
use crate::poolops::wall_ms;

pub use host::Shards;

/// The peer ops a pool member may send.
pub const OPS: [&str; 5] = ["shard_caps", "shard_put", "shard_start", "shard_status", "shard_cancel"];

/// The list of fingerprints a `shard_consent` param holds (empty when absent).
fn fingerprints(p: &Map<String, Value>, key: &str) -> Result<Vec<String>> {
    let Some(v) = p.get(key) else { return Ok(Vec::new()) };
    let list: Vec<String> = serde_json::from_value(v.clone()).map_err(|_| Error::Invalid(format!("{key} is a list of device fingerprints")))?;
    if list.len() > 64 || list.iter().any(|f| !crate::cert::valid_fingerprint(f)) {
        return Err(Error::Invalid(format!("{key} is up to 64 device fingerprints, 64 hex digits each")));
    }
    Ok(list)
}

/// `shard_consent`: show the state; with `enabled`, turn shards on or off and write it to the pool board.
pub fn consent_call(node: &mut Node, token: &str, p: &Map<String, Value>) -> Result<Value> {
    let mut saved = consent::load(&node.dir)?;
    if let Some(on) = p.get("enabled") {
        let on = on.as_bool().ok_or_else(|| Error::Invalid("enabled is true or false".into()))?;
        let who = authorize(node, token, "shard_consent")?;
        let text = |k: &str, old: &str| p.get(k).and_then(Value::as_str).map_or_else(|| old.to_string(), String::from);
        if on {
            let (python, repo) = (text("python", &saved.python), text("repo", &saved.repo));
            consent::check_places(&python, &repo)?;
            runtime::python_version(&python)?;
            (saved.python, saved.repo) = (python, repo);
        }
        (saved.enabled, saved.by, saved.at_ms) = (on, format!("{}/{}", who.board, who.name), wall_ms());
        consent::save(&node.dir, &saved)?;
        node.record_event("shard_consent", if on { "on" } else { "off" }, &format!("by {}; python {}; repo {}", saved.by, saved.python, saved.repo))?;
    }
    let (allow, deny) = (fingerprints(p, "allow")?, fingerprints(p, "deny")?);
    if !allow.is_empty() || !deny.is_empty() {
        let who = authorize(node, token, "shard_consent")?;
        let by = format!("{}/{}", who.board, who.name);
        for fp in &allow {
            if *fp == node.cert.fingerprint() || !node.members.is_active(fp) {
                return Err(Error::Invalid("only another active member of this pool is allowed".into()));
            }
        }
        saved.allowed.retain(|a| !deny.contains(a));
        for fp in &allow {
            if !saved.allowed.contains(fp) {
                saved.allowed.push(fp.clone());
            }
        }
        (saved.by, saved.at_ms) = (by.clone(), wall_ms());
        consent::save(&node.dir, &saved)?;
        for (word, list) in [("allowed", &allow), ("denied", &deny)] {
            for fp in list {
                let name = node.members.get(fp).map(|d| d.name.clone()).unwrap_or_default();
                node.record_event("shard_allow", fp, &format!("{by} {word} {name}"))?;
            }
        }
    }
    Ok(json!({"enabled": saved.enabled, "python": saved.python, "repo": saved.repo, "allowed": saved.allowed, "by": saved.by, "at_ms": saved.at_ms}))
}
