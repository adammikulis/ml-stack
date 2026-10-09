//! The local API for the pool: `pool_status`, `member_revoke`, `set_join_policy`, and the
//! network ones (`pair_accept`, `pair_start`, `sync_now`) which `net::Net` answers.
//!
//! Reading the pool needs no token (the socket is the owner's); changing it needs a registered
//! session that holds the grant (`grants::Grants`).

use serde_json::{json, Map, Value};

use crate::cert::valid_fingerprint;
use crate::error::{Error, Result};
use crate::identity::Holder;
use crate::membership::Policy;
use crate::node::Node;

/// The methods that act on the network and so run without the node lock held.
pub const NETWORK: [&str; 4] = ["pair_accept", "pair_start", "sync_now", "shard_call"];

/// The params each pool method accepts; None for any other method.
pub fn params(method: &str) -> Option<&'static [&'static str]> {
    Some(match method {
        "pool_status" | "sync_now" => &[],
        "member_revoke" => &["fingerprint"],
        "set_join_policy" => &["policy"],
        "pair_accept" => &["passphrase", "ttl_s"],
        "pair_start" => &["host", "port", "passphrase", "fingerprint"],
        "shard_call" => &["device", "op", "args"],
        "shard_consent" => &["enabled", "python", "repo", "allow", "deny"],
        _ => return None,
    })
}

pub fn handles(method: &str) -> bool {
    params(method).is_some()
}

/// The session behind ``token``, if it holds the grant for ``action``.
pub fn authorize(node: &Node, token: &str, action: &str) -> Result<Holder> {
    let holder = node.tokens.resolve(token).cloned().ok_or_else(|| Error::Denied("a registered session is needed".into()))?;
    if !node.grants.allows(&holder, action) {
        return Err(Error::Denied(format!("this session does not hold the grant for {action}")));
    }
    Ok(holder)
}

fn text<'a>(p: &'a Map<String, Value>, key: &str) -> Result<&'a str> {
    match p.get(key) {
        Some(Value::String(s)) => Ok(s),
        None => Ok(""),
        _ => Err(Error::Invalid(format!("{key} is text"))),
    }
}

pub fn dispatch(node: &mut Node, method: &str, token: &str, p: &Map<String, Value>) -> Result<Value> {
    match method {
        "pool_status" => Ok(node.pool_status()),
        "member_revoke" => {
            let who = authorize(node, token, method)?;
            let fingerprint = text(p, "fingerprint")?;
            if !valid_fingerprint(fingerprint) {
                return Err(Error::Invalid("a device is named by its fingerprint, 64 hex digits".into()));
            }
            if fingerprint == node.cert.fingerprint() {
                return Err(Error::Denied("a device does not revoke itself; leave by removing its state".into()));
            }
            let device = node.revoke_device(fingerprint, &format!("{}/{}", who.board, who.name))?;
            if crate::shard::consent::forget(&node.dir, fingerprint)? {
                node.record_event("shard_allow", fingerprint, &format!("removed: put out of the pool by {}/{}", who.board, who.name))?;
            }
            Ok(json!({"fingerprint": device.fingerprint, "status": device.status}))
        }
        "set_join_policy" => {
            let who = authorize(node, token, method)?;
            let policy = Policy::parse(text(p, "policy")?)?;
            let changed = node.change_policy(policy, &format!("{}/{}", who.board, who.name))?;
            Ok(json!({"policy": policy.name(), "changed": changed}))
        }
        "shard_consent" => crate::shard::consent_call(node, token, p),
        _ => Err(Error::Invalid("this node's network is not running".into())),
    }
}
