//! The lease service: one table for GPU one-at-a-time, CPU slots, memory admission, named
//! claims and served-model slots, with a queue, dead-holder detection and takeover.
//!
//! The device's own table is authoritative for the device's resources and lives in
//! `<state>/leases.json`. Every grant and end is also written to the holder's board as a
//! `lease` entry, so peers see it; `merge` reads those entries back.

pub mod live;
pub mod merge;
pub mod policy;
pub mod rpc;
pub mod table;
pub mod types;

use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::error::Result;
use crate::fsutil::write_atomic;
use table::{Event, Foreign, Table};
use types::Config;

pub const TABLE_FILE: &str = "leases.json";
pub const CONFIG_FILE: &str = "lease-config.json";

fn wall_ms() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_millis() as u64)
}

pub struct Leases {
    pub cfg: Config,
    pub table: Table,
    path: PathBuf,
    pub clock: Box<dyn Fn() -> u64 + Send>,
}

impl Leases {
    /// Open the table under ``dir``; holders that died or ran out while the node was down are
    /// dropped at the next `tick`.
    pub fn open(dir: &Path) -> Result<Leases> {
        let read = |name: &str| match std::fs::read(dir.join(name)) {
            Ok(bytes) => Ok(Some(bytes)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e),
        };
        let cfg = match read(CONFIG_FILE)? {
            Some(bytes) => serde_json::from_slice(&bytes)?,
            None => Config::default(),
        };
        let table = match read(TABLE_FILE)? {
            Some(bytes) => serde_json::from_slice(&bytes)?,
            None => Table::default(),
        };
        Ok(Leases { cfg, table, path: dir.join(TABLE_FILE), clock: Box::new(wall_ms) })
    }

    pub fn now(&self) -> u64 {
        (self.clock)()
    }

    pub fn save(&self) -> Result<()> {
        write_atomic(&self.path, &serde_json::to_vec(&self.table)?)
    }

    /// Drop the dead and the expired, grant what the freed resources allow, and save when
    /// anything changed. Returns what happened, for the board.
    pub fn tick(&mut self, foreign: &Foreign) -> Result<Vec<Event>> {
        let now = self.now();
        let ended = self.table.sweep(now);
        let mut events = self.table.schedule(&self.cfg, now, foreign, &ended);
        events.extend(ended.into_iter().map(|(l, why)| Event::Ended(l, why)));
        if !events.is_empty() {
            self.save()?;
        }
        Ok(events)
    }
}
