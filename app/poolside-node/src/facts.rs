//! What the node knows of its peers right now: not part of the pool record, never replicated.
//! Only the last address of each peer is kept on disk, so a restarted node can dial again
//! before it hears a beacon.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::error::{Error, Result};
use crate::fsutil::write_atomic;

#[derive(Default, Clone, Debug)]
pub struct PeerFacts {
    pub addr: Option<String>,
    pub last_seen_ms: u64,
    /// Live sessions this peer holds open to us.
    pub sessions: usize,
    pub last_sync_ms: u64,
    pub last_error: String,
}

pub struct Facts {
    path: PathBuf,
    pub listen: Option<String>,
    pub pairing_until_ms: u64,
    pub beacon: bool,
    pub peers: BTreeMap<String, PeerFacts>,
}

impl Facts {
    pub fn open(path: &Path) -> Result<Facts> {
        let addrs: BTreeMap<String, String> = match std::fs::read(path) {
            Ok(bytes) => serde_json::from_slice(&bytes).map_err(|e| Error::Damaged(format!("peer addresses unreadable: {e}")))?,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => BTreeMap::new(),
            Err(e) => return Err(e.into()),
        };
        let peers = addrs.into_iter().map(|(fp, a)| (fp, PeerFacts { addr: Some(a), ..PeerFacts::default() })).collect();
        Ok(Facts { path: path.into(), listen: None, pairing_until_ms: 0, beacon: false, peers })
    }

    pub fn peer(&mut self, fingerprint: &str) -> &mut PeerFacts {
        self.peers.entry(fingerprint.into()).or_default()
    }

    /// Remember where ``fingerprint`` was last reached.
    pub fn set_addr(&mut self, fingerprint: &str, addr: &str) -> Result<()> {
        let peer = self.peer(fingerprint);
        if peer.addr.as_deref() == Some(addr) {
            return Ok(());
        }
        peer.addr = Some(addr.into());
        let addrs: BTreeMap<&String, &String> = self.peers.iter().filter_map(|(f, p)| p.addr.as_ref().map(|a| (f, a))).collect();
        write_atomic(&self.path, &serde_json::to_vec(&addrs)?)
    }
}
