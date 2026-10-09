//! Whether this device takes test shards, and which interpreter and checkout run them.
//!
//! `<state>/shards.json` is read at every request, so switching a device off stops the next
//! upload at once. The file is absent on a device nobody enabled: that is off. Enabling is
//! `shard_consent` on the local API, which writes an audit entry to the pool board.

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::fsutil::write_atomic;
use crate::row::is_line;

pub const FILE: &str = "shards.json";
/// The file inside a checkout the executor is started from; a `repo` that lacks it is refused.
pub const EXECUTOR: &str = "src/ml_stack/fleet/shard_exec.py";

#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq, Eq)]
#[serde(default, deny_unknown_fields)]
pub struct Consent {
    pub enabled: bool,
    /// The interpreter that runs the executor and the tests: absolute, and Python 3.13.
    pub python: String,
    /// This device's own checkout, where the executor code comes from (never from a sender).
    pub repo: String,
    /// Fingerprints of the pool members whose tests this device takes. Empty: nobody, whoever is in the pool.
    pub allowed: Vec<String>,
    /// Who last changed it (`board/name`) and when.
    pub by: String,
    pub at_ms: u64,
}

/// The saved consent; off when there is none.
pub fn load(dir: &Path) -> Result<Consent> {
    match std::fs::read(dir.join(FILE)) {
        Ok(bytes) => serde_json::from_slice(&bytes).map_err(|e| Error::Damaged(format!("{FILE} is unreadable: {e}"))),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Consent::default()),
        Err(e) => Err(e.into()),
    }
}

fn absolute(text: &str) -> bool {
    let b = text.as_bytes();
    text.starts_with('/') || (b.len() > 2 && b[0].is_ascii_alphabetic() && b[1] == b':' && matches!(b[2], b'/' | b'\\'))
}

/// Check the interpreter and checkout a person named; the reason when one cannot be used.
pub fn check_places(python: &str, repo: &str) -> Result<()> {
    for (what, text) in [("python", python), ("repo", repo)] {
        if text.is_empty() || text.len() > 500 || !is_line(text) || !absolute(text) || text.contains("..") {
            return Err(Error::Invalid(format!("{what} is one absolute path with no .. in it")));
        }
    }
    if !Path::new(python).is_file() {
        return Err(Error::Invalid(format!("python {python} is not a file on this device")));
    }
    if !Path::new(repo).join(EXECUTOR).is_file() {
        return Err(Error::Invalid(format!("repo {repo} has no {EXECUTOR}: name this device's own checkout")));
    }
    Ok(())
}

/// Take ``fp`` out of the allowed list (a member put out of the pool, or denied); whether it was there.
pub fn forget(dir: &Path, fp: &str) -> Result<bool> {
    let mut saved = load(dir)?;
    let before = saved.allowed.len();
    saved.allowed.retain(|a| a != fp);
    let changed = saved.allowed.len() != before;
    if changed {
        save(dir, &saved)?;
    }
    Ok(changed)
}

pub fn save(dir: &Path, consent: &Consent) -> Result<()> {
    write_atomic(&dir.join(FILE), &serde_json::to_vec(consent)?)
}
