//! What the node does with its pool record: enrol, revoke, change the policy, and write each
//! change to the board named `pool` so it is part of the signed history.

use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::{json, Value};

use crate::error::Result;
use crate::membership::{Device, Policy};
use crate::row::Kind;

/// The board that carries the pool's own history (members added and put out, policy changes).
pub const POOL_BOARD: &str = "pool";
const ACTOR: &str = "node";

pub fn wall_ms() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_millis() as u64)
}

fn host_name() -> String {
    let mut buf = [0u8; 256];
    // SAFETY: the buffer is valid for its length for the call.
    let rc = unsafe { libc::gethostname(buf.as_mut_ptr().cast(), buf.len() - 1) };
    let name = if rc == 0 { String::from_utf8_lossy(&buf).trim_end_matches('\0').to_string() } else { String::new() };
    let name: String = name.chars().filter(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '.' | '_')).take(40).collect();
    if name.is_empty() { "device".into() } else { name }
}

impl crate::node::Node {
    /// Milliseconds since the epoch.
    pub fn now_ms(&self) -> u64 {
        wall_ms()
    }

    /// List this device's own certificate in the pool, once.
    pub fn enrol_self(&mut self) -> Result<()> {
        if self.members.get(&self.cert.fingerprint()).is_none() {
            let (der, now) = (self.cert.der.clone(), wall_ms());
            self.members.enrol(&der, &host_name(), "self", now)?;
        }
        Ok(())
    }

    /// Write one event to the pool board; the pool record stays the authority.
    pub fn record_event(&mut self, event: &str, subject: &str, detail: &str) -> Result<()> {
        let body = json!({"event": event, "subject": subject.chars().take(100).collect::<String>(), "detail": detail.chars().take(250).collect::<String>()});
        self.host(POOL_BOARD)?.board.append(Kind::Audit, ACTOR, body, "")?;
        Ok(())
    }

    /// Let the device with this certificate in, and note it on the pool board.
    pub fn enrol_device(&mut self, der: &[u8], name: &str, by: &str) -> Result<Device> {
        let existed = self.members.get(&crate::cert::cert_fingerprint(der)).is_some();
        let device = self.members.enrol(der, name, by, wall_ms())?;
        if !existed {
            self.record_event("member_added", &device.fingerprint, &format!("by {by}"))?;
        }
        Ok(device)
    }

    /// Put a device out for good, and note it on the pool board.
    pub fn revoke_device(&mut self, fingerprint: &str, by: &str) -> Result<Device> {
        let was_out = self.members.get(fingerprint).is_some_and(|d| d.status == crate::membership::Status::Revoked);
        let device = self.members.revoke(fingerprint, by, wall_ms())?;
        if !was_out {
            self.record_event("member_revoked", fingerprint, &format!("by {by}"))?;
        }
        Ok(device)
    }

    /// Change the join policy; true when it changed.
    pub fn change_policy(&mut self, policy: Policy, by: &str) -> Result<bool> {
        let changed = self.members.set_policy(policy, wall_ms().max(self.members.policy_at + 1), by)?;
        if changed {
            self.record_event("join_policy", policy.name(), &format!("by {by}"))?;
        }
        Ok(changed)
    }

    /// The pool as a local client sees it: members with what is known of each, the policy,
    /// whether pairing is open, and this node's listening address.
    pub fn pool_status(&self) -> Value {
        let me = self.cert.fingerprint();
        let now = wall_ms();
        let members: Vec<Value> = self.members.devices().iter().map(|d| {
            let f = self.facts.peers.get(&d.fingerprint);
            json!({
                "fingerprint": d.fingerprint, "name": d.name, "status": d.status, "since": d.at, "by": d.by, "self": d.fingerprint == me,
                "connected": f.is_some_and(|f| f.sessions > 0), "addr": f.and_then(|f| f.addr.clone()),
                "last_seen_ms": f.map_or(0, |f| f.last_seen_ms), "last_sync_ms": f.map_or(0, |f| f.last_sync_ms),
                "last_error": f.map(|f| f.last_error.clone()).unwrap_or_default(),
            })
        }).collect();
        json!({
            "pool": self.members.id, "policy": self.members.policy.name(), "fingerprint": me, "listen": self.facts.listen,
            "beacon": self.facts.beacon, "pairing_open": self.facts.pairing_until_ms > now,
            "pairing_expires_in_s": self.facts.pairing_until_ms.saturating_sub(now) / 1000, "members": members,
        })
    }
}
