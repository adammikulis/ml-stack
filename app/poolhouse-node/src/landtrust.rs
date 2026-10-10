//! Whose landing entries this device counts. Landing authority is this device's own: an entry of
//! the `landing` kind counts in the queue only when this device wrote it, or when a device that
//! joined the pool is on this list and still an active member. The list is a decision of this
//! device alone (never replicated, empty by default) and is set through `land_trust`.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::cert::{board_fingerprint, valid_fingerprint};
use crate::error::Result;
use crate::fsutil::write_atomic;
use crate::membership::Status;
use crate::node::Node;

#[derive(Serialize, Deserialize, Default)]
struct File {
    pool: String,
    devices: BTreeSet<String>,
}

/// The devices of one pool whose landing entries are counted, kept in one atomically replaced file.
pub struct LandTrust {
    path: PathBuf,
    file: File,
}

impl LandTrust {
    pub fn open(path: &Path) -> Result<LandTrust> {
        let file = match std::fs::read(path) {
            Ok(bytes) => serde_json::from_slice(&bytes).unwrap_or_default(),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => File::default(),
            Err(e) => return Err(e.into()),
        };
        Ok(LandTrust { path: path.into(), file })
    }

    /// The certificate fingerprints trusted in ``pool``; a list made for another pool counts for nothing.
    pub fn devices(&self, pool: &str) -> BTreeSet<String> {
        if self.file.pool == pool { self.file.devices.clone() } else { BTreeSet::new() }
    }

    /// Add or remove ``fingerprint`` in ``pool``; whether the list changed.
    pub fn set(&mut self, pool: &str, fingerprint: &str, trusted: bool) -> Result<bool> {
        if !valid_fingerprint(fingerprint) {
            return Err(crate::error::Error::Invalid("a device is named by its fingerprint, 64 hex digits".into()));
        }
        if self.file.pool != pool {
            self.file = File { pool: pool.into(), devices: BTreeSet::new() };
        }
        let changed = if trusted { self.file.devices.insert(fingerprint.into()) } else { self.file.devices.remove(fingerprint) };
        if changed {
            write_atomic(&self.path, &serde_json::to_vec(&self.file)?)?;
        }
        Ok(changed)
    }
}

/// The foreign logs of one board, as far as landing is concerned.
pub struct Writers {
    /// Origins written by a device that is on the list and still active in the pool.
    pub trusted: BTreeSet<String>,
    /// How to name the writer of each foreign origin: its device name, or its label.
    pub shown: BTreeMap<String, String>,
}

/// Which foreign origins of ``board`` hold trusted landing devices, decided now from the pool record and the
/// key that verified each log, never from anything a log says about itself.
pub fn writers(node: &mut Node, board: &str) -> Result<Writers> {
    let allowed = node.land_trust.devices(&node.members.id);
    let by_key: BTreeMap<String, (String, String, bool)> = node.members.devices().into_iter()
        .filter_map(|d| {
            let key = board_fingerprint(&d.der()?).ok()?;
            let name = if d.name.is_empty() { d.fingerprint[..12].to_string() } else { d.name.clone() };
            let live = d.status == Status::Active && allowed.contains(&d.fingerprint);
            Some((key, (name, d.fingerprint, live)))
        })
        .collect();
    let board = &mut node.host(board)?.board;
    let own = board.origin().to_string();
    let mut out = Writers { trusted: BTreeSet::new(), shown: BTreeMap::new() };
    for origin in board.origins().into_iter().filter(|o| *o != own) {
        let label = board.label(&origin)?;
        let found = board.writer_fingerprint(&origin).and_then(|k| by_key.get(&k));
        out.shown.insert(origin.clone(), found.map_or(label.clone(), |(name, _, _)| format!("{name} ({label})")));
        if found.is_some_and(|(_, _, live)| *live) {
            out.trusted.insert(origin);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_list_is_kept_per_pool_and_a_list_of_another_pool_counts_for_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("land_trust.json");
        let fp = "ab".repeat(32);
        let mut t = LandTrust::open(&path).unwrap();
        assert!(t.devices("p1").is_empty(), "empty by default: only this device");
        assert!(t.set("p1", &fp, true).unwrap());
        assert!(!t.set("p1", &fp, true).unwrap());
        assert_eq!(LandTrust::open(&path).unwrap().devices("p1").len(), 1, "kept on disk");
        assert!(t.devices("p2").is_empty());
        assert!(t.set("p1", &fp, false).unwrap());
        assert!(t.devices("p1").is_empty());
        assert!(t.set("p1", "short", true).is_err());
    }
}
