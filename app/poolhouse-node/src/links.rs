//! Board links: the only way anything crosses from one board to another.
//!
//! A link says the sessions of board `to` may use named channels (or the whole) of board
//! `from`, read-only or read-write. Nothing else crosses. Revoking stops it at once.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::fsutil::write_atomic;
use crate::row::valid_name;

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Link {
    pub id: String,
    /// The board whose entries are shared.
    pub from: String,
    /// The board whose sessions may use them.
    pub to: String,
    /// The channels shared; empty means the whole board.
    pub channels: Vec<String>,
    pub write: bool,
    pub active: bool,
    /// The session that made the link, as `board/name`.
    pub by: String,
}

pub struct Links {
    path: PathBuf,
    all: Vec<Link>,
}

impl Links {
    pub fn open(path: &Path) -> Result<Links> {
        let all = match std::fs::read(path) {
            Ok(bytes) => serde_json::from_slice(&bytes).map_err(|e| Error::Damaged(format!("links unreadable: {e}")))?,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Vec::new(),
            Err(e) => return Err(e.into()),
        };
        Ok(Links { path: path.into(), all })
    }

    pub fn all(&self) -> &[Link] {
        &self.all
    }

    fn save(&self) -> Result<()> {
        write_atomic(&self.path, &serde_json::to_vec(&self.all)?)
    }

    /// Share ``channels`` of ``from`` with the sessions of ``to``.
    pub fn add(&mut self, from: &str, to: &str, channels: Vec<String>, write: bool, by: &str) -> Result<Link> {
        if from == to || !valid_name(to) || channels.iter().any(|c| !c.starts_with('#') || c.len() > 64) || channels.len() > 16 {
            return Err(Error::Invalid("a link joins two different boards through named #channels".into()));
        }
        let link = Link { id: format!("link-{}", self.all.len() + 1), from: from.into(), to: to.into(), channels, write, active: true, by: by.into() };
        self.all.push(link.clone());
        self.save()?;
        Ok(link)
    }

    /// Revoke link ``id`` of board ``from``; the link is kept, inactive, for the record.
    pub fn revoke(&mut self, from: &str, id: &str) -> Result<Link> {
        let link = self.all.iter_mut().find(|l| l.from == from && l.id == id && l.active)
            .ok_or_else(|| Error::Invalid("no such active link on this board".into()))?;
        link.active = false;
        let done = link.clone();
        self.save()?;
        Ok(done)
    }

    /// The channels of ``target`` that sessions of ``home`` may use, `None` when nothing is
    /// shared; an empty list means the whole board.
    pub fn grants(&self, home: &str, target: &str, write: bool) -> Option<Vec<String>> {
        let mut found: Option<Vec<String>> = None;
        for link in self.all.iter().filter(|l| l.active && l.from == target && l.to == home && (!write || l.write)) {
            if link.channels.is_empty() {
                return Some(Vec::new());
            }
            found.get_or_insert_with(Vec::new).extend(link.channels.iter().cloned());
        }
        found
    }

    /// Whether a session of ``home`` may use ``channel`` of ``target`` (writing or reading).
    pub fn allows(&self, home: &str, target: &str, channel: &str, write: bool) -> bool {
        self.grants(home, target, write).is_some_and(|c| c.is_empty() || c.iter().any(|x| x == channel))
    }
}
