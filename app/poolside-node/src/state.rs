//! The small tables a board keeps beside its logs, in one atomically replaced file.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::fsutil::write_atomic;

/// The key an origin's heads verified under, and whether its owner presented it directly.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Pin {
    pub public: String,
    pub bound: bool,
}

#[derive(Serialize, Deserialize, Default, Debug)]
#[serde(default)]
pub struct Tables {
    /// Pinned key of each foreign origin.
    pub keys: BTreeMap<String, Pin>,
    /// Origins whose own copy was refused, with the reason.
    pub damaged: BTreeMap<String, String>,
    /// For each paired device fingerprint, how much of this device's log it holds.
    pub acks: BTreeMap<String, u64>,
    /// Device fingerprint to the origin it writes.
    pub peers: BTreeMap<String, String>,
    /// The highest clock value seen from foreign rows.
    pub seen: (u64, u64),
    /// Local names of foreign origins, `d1`, `d2`, in the order first seen.
    pub labels: BTreeMap<String, String>,
    /// The latest foreign rows that were not folded, with the reason.
    pub rejected: Vec<(String, String)>,
}

pub struct State {
    path: PathBuf,
    pub t: Tables,
}

impl State {
    pub fn open(path: &Path) -> Result<State> {
        let t = match std::fs::read(path) {
            Ok(bytes) => serde_json::from_slice(&bytes).map_err(|e| Error::Damaged(format!("board state unreadable: {e}")))?,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Tables::default(),
            Err(e) => return Err(e.into()),
        };
        Ok(State { path: path.to_path_buf(), t })
    }

    pub fn save(&self) -> Result<()> {
        write_atomic(&self.path, &serde_json::to_vec(&self.t)?)
    }
}
