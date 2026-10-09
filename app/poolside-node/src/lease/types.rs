//! The things a lease table is made of: typed resources, requests, leases, configuration.

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::row::{is_line, valid_name};

/// The device this node is authoritative for.
pub const LOCAL: &str = "local";
/// Claim kinds, as `src/ml_stack/workspace/claims.py` names them (`install` is new).
pub const CLAIM_KINDS: [&str; 6] = ["worktree", "branch", "area", "port", "server", "install"];
pub const MAX_RESOURCES: usize = 8;

fn local() -> String {
    LOCAL.into()
}

/// One thing a lease holds. Exclusive ones have one holder at a time; `CpuSlots` and
/// `MemoryMb` are counted against a capacity.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum Resource {
    /// A GPU, one holder at a time.
    Gpu { #[serde(default = "local")] device: String },
    /// ``count`` CPU test or job slots of a device.
    CpuSlots { #[serde(default = "local")] device: String, count: u32 },
    /// ``mb`` megabytes of the device's memory budget.
    MemoryMb { #[serde(default = "local")] device: String, mb: u64 },
    /// A named claim: worktree, branch, area, port, server or install.
    Claim { kind: String, name: String },
    /// A served model in one shape; one holder per device and model.
    ModelSlot {
        #[serde(default = "local")] device: String,
        model: String,
        #[serde(default)] context: u64,
        #[serde(default)] draft: String,
        #[serde(default)] parallel: u32,
    },
}

impl Resource {
    pub fn device(&self) -> &str {
        match self {
            Resource::Gpu { device } | Resource::CpuSlots { device, .. } | Resource::MemoryMb { device, .. } | Resource::ModelSlot { device, .. } => device,
            Resource::Claim { .. } => LOCAL,
        }
    }

    /// The type word.
    pub fn word(&self) -> &'static str {
        match self {
            Resource::Gpu { .. } => "gpu",
            Resource::CpuSlots { .. } => "cpu_slots",
            Resource::MemoryMb { .. } => "memory_mb",
            Resource::Claim { .. } => "claim",
            Resource::ModelSlot { .. } => "model_slot",
        }
    }

    /// Whether one holder at a time (everything but the counted kinds).
    pub fn exclusive(&self) -> bool {
        !matches!(self, Resource::CpuSlots { .. } | Resource::MemoryMb { .. })
    }

    /// Claims of these kinds are about the pool's shared repository, not one device's disk.
    pub fn pool_wide(&self) -> bool {
        matches!(self, Resource::Claim { kind, .. } if kind == "branch" || kind == "area")
    }

    /// The name that identifies the contended thing: two resources with equal keys conflict
    /// (an exclusive one) or share a capacity (a counted one).
    pub fn key(&self) -> String {
        match self {
            Resource::Gpu { device } => format!("gpu:{device}"),
            Resource::CpuSlots { device, .. } => format!("cpu_slots:{device}"),
            Resource::MemoryMb { device, .. } => format!("memory_mb:{device}"),
            Resource::Claim { kind, name } => format!("claim:{kind}:{name}"),
            Resource::ModelSlot { device, model, .. } => format!("model_slot:{device}:{model}"),
        }
    }

    /// The amount a counted resource asks for; 1 for an exclusive one.
    pub fn amount(&self) -> u64 {
        match self {
            Resource::CpuSlots { count, .. } => u64::from(*count),
            Resource::MemoryMb { mb, .. } => *mb,
            _ => 1,
        }
    }

    /// Whether this resource and ``other`` cannot both be held by different holders at once.
    /// Equal keys conflict; worktree and area claims also conflict with a nested path.
    pub fn conflicts(&self, other: &Resource) -> bool {
        match (self, other) {
            (Resource::Claim { kind: a, name: x }, Resource::Claim { kind: b, name: y }) if a == b && (a == "worktree" || a == "area") => nested(x, y),
            _ => self.exclusive() && self.key() == other.key(),
        }
    }

    pub fn check(&self) -> Result<()> {
        let bad = |why: &str| Err(Error::Invalid(why.into()));
        if !valid_name(self.device()) {
            return bad("a device is a short lower-case word");
        }
        match self {
            Resource::CpuSlots { count: 0, .. } => bad("cpu_slots asks for at least one"),
            Resource::MemoryMb { mb: 0, .. } => bad("memory_mb asks for at least one"),
            Resource::Claim { kind, name } => {
                if !CLAIM_KINDS.contains(&kind.as_str()) {
                    return bad("a claim kind is worktree, branch, area, port, server or install");
                }
                if name.is_empty() || name.len() > 300 || !is_line(name) || name.contains("..") {
                    return bad("a claim names one clean line without ..");
                }
                match kind.as_str() {
                    "port" if !name.parse::<u32>().is_ok_and(|p| (1..65536).contains(&p)) => bad("a port is a number from 1 to 65535"),
                    "worktree" | "area" if !is_absolute(name) => bad("a worktree or area claim names an absolute path"),
                    _ => Ok(()),
                }
            }
            Resource::ModelSlot { model, draft, .. } => {
                if model.is_empty() || model.len() > 300 || !is_line(model) || draft.len() > 300 || !is_line(draft) {
                    return bad("a model slot names a model on one clean line");
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }

    /// The one-line text a board entry shows for this resource.
    pub fn line(&self) -> String {
        match self {
            Resource::CpuSlots { device, count } => format!("cpu_slots:{device}:{count}"),
            Resource::MemoryMb { device, mb } => format!("memory_mb:{device}:{mb}"),
            Resource::ModelSlot { device, model, context, draft, parallel } => format!("model_slot:{device}:{model}:{context}:{draft}:{parallel}"),
            other => other.key(),
        }
    }
}

/// A Unix path, a drive path (`C:\x`, `C:/x`) or a UNC path (`\\host\x`).
fn is_absolute(name: &str) -> bool {
    let b = name.as_bytes();
    name.starts_with('/') || name.starts_with("\\\\") || (b.len() > 2 && b[0].is_ascii_alphabetic() && b[1] == b':' && matches!(b[2], b'/' | b'\\'))
}

/// A path with one separator, and no case on Windows drives, so `C:\X` and `c:/x` are one place.
fn place(path: &str) -> String {
    let path = path.replace('\\', "/");
    if path.as_bytes().get(1) == Some(&b':') { path.to_lowercase() } else { path }
}

fn nested(a: &str, b: &str) -> bool {
    let (a, b) = (place(a), place(b));
    let (a, b) = (a.trim_end_matches('/'), b.trim_end_matches('/'));
    a == b || a.starts_with(&format!("{b}/")) || b.starts_with(&format!("{a}/"))
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Class {
    Interactive,
    Background,
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum State {
    Queued,
    Held,
}

/// One request or grant. A lease is queued until every resource it names is free for it.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Lease {
    pub id: String,
    /// `board/name`, from the token that asked.
    pub holder: String,
    pub board: String,
    pub name: String,
    pub resources: Vec<Resource>,
    pub class: Class,
    pub state: State,
    pub estimate_s: u64,
    /// Arrival order, the tie-break of every ordering.
    pub seq: u64,
    pub since_ms: u64,
    pub granted_ms: u64,
    pub expires_ms: u64,
    /// The local process whose life keeps the lease; 0 for a remote holder or none.
    pub pid: u32,
    pub pid_start: Option<u64>,
    pub remote: bool,
    /// How long one renewal extends the lease.
    pub ttl_s: u64,
}

/// What admission decides, in one place so it is configuration, not code.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    /// CPU slots of the local device.
    pub cpu_slots: u32,
    /// Memory budget of the local device in MB; 0 means no budget is enforced.
    pub memory_mb: u64,
    /// Seconds after which a waiting request goes ahead of every ordering and cap.
    pub max_wait_s: u64,
    /// While a short run is queued or running, long runs may hold at most this percent of the CPU slots.
    pub background_share_percent: u32,
    /// Estimate assumed for a background request that gave none.
    pub unknown_background_s: u64,
    /// A queued request that is not polled for this long is dropped.
    pub queue_patience_s: u64,
    pub default_ttl_s: u64,
    pub max_ttl_s: u64,
}

impl Default for Config {
    fn default() -> Config {
        let cores = std::thread::available_parallelism().map_or(1, |n| n.get() as u32);
        Config {
            cpu_slots: cores, memory_mb: 0, max_wait_s: 600, background_share_percent: 50, unknown_background_s: 180,
            queue_patience_s: 60, default_ttl_s: 300, max_ttl_s: 86_400,
        }
    }
}

/// A request to hold resources, from one holder.
#[derive(Clone, Debug)]
pub struct Request {
    pub board: String,
    pub name: String,
    pub resources: Vec<Resource>,
    pub class: Class,
    pub estimate_s: Option<u64>,
    pub ttl_s: Option<u64>,
    pub pid: u32,
    pub remote: bool,
    /// Queue when not free now; otherwise answer `busy`.
    pub wait: bool,
}
