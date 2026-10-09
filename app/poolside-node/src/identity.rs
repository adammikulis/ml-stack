//! Names and tokens. A session gets one unique readable name per board, made from its model
//! family and a suffix of the hash of its native identity; the node stamps every write with
//! the name its token resolves to, so no request ever says who it is.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::error::{Error, Result};
use crate::fsutil::{hex, random_hex, sha256_hex, write_atomic};

pub const SHORT: usize = 6;
pub const MAX_IDENTITIES: usize = 10_000;
const MAX_TOKENS_PER_NAME: usize = 8;

/// The lower-case model family of an exact model id (`agent` when it names none).
pub fn family_word(model: &str) -> &'static str {
    let model = model.to_lowercase();
    let model = model.rsplit('/').next().unwrap_or("");
    if model.starts_with("claude-") {
        "claude"
    } else if ["gpt-", "chatgpt-", "o1", "o3", "o4"].iter().any(|p| model.starts_with(p)) {
        "chatgpt"
    } else if model.starts_with("qwen") || model.starts_with("thinkingcap-qwen") {
        "qwen"
    } else {
        "agent"
    }
}

/// The full identity of a native session: SHA-256 of harness, a NUL, and the session id.
pub fn digest(harness: &str, session: &str) -> String {
    hex(&Sha256::digest(format!("{harness}\0{session}").as_bytes()))
}

/// How far a session's model id is believed: what it said, or what the node checked.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ModelState {
    Unknown,
    Claimed,
    /// A subagent that named no model runs the one its parent runs.
    Inherited,
    Verified,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Ident {
    pub digest: String,
    pub family: String,
    pub parent: String,
    pub model: String,
    pub model_state: ModelState,
    pub harness: String,
    pub retired: bool,
    /// Wall milliseconds of the last request this session made (written at most every 30 s).
    pub seen_ms: u64,
}

/// A model id or harness name as one short clean line.
pub fn clean(text: &str, most: usize) -> Result<String> {
    let text = text.trim();
    if text.len() > most || text.chars().any(char::is_control) {
        return Err(Error::Invalid(format!("a model or harness is one clean line of at most {most} bytes")));
    }
    Ok(text.into())
}

/// The names of one board.
pub struct Registry {
    path: PathBuf,
    names: BTreeMap<String, Ident>,
}

impl Registry {
    pub fn open(path: &Path) -> Result<Registry> {
        let names = match std::fs::read(path) {
            Ok(bytes) => serde_json::from_slice(&bytes).map_err(|e| Error::Damaged(format!("names unreadable: {e}")))?,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => BTreeMap::new(),
            Err(e) => return Err(e.into()),
        };
        Ok(Registry { path: path.into(), names })
    }

    pub fn contains(&self, name: &str) -> bool {
        self.names.contains_key(name)
    }

    pub fn get(&self, name: &str) -> Option<&Ident> {
        self.names.get(name)
    }

    pub fn all(&self) -> &BTreeMap<String, Ident> {
        &self.names
    }

    fn save(&self) -> Result<()> {
        write_atomic(&self.path, &serde_json::to_vec(&self.names)?)
    }

    /// The name already given to this native session, if any.
    pub fn find(&self, harness: &str, session: &str) -> Option<&str> {
        let full = digest(harness, session);
        self.names.iter().find(|(_, i)| i.digest == full).map(|(n, _)| n.as_str())
    }

    /// Record what a session says it runs on; a verified model is never lowered to a claim.
    pub fn claim_model(&mut self, name: &str, model: &str, harness: &str) -> Result<bool> {
        let (model, harness) = (clean(model, 256)?, clean(harness, 64)?);
        let ident = self.names.get_mut(name).ok_or_else(|| Error::Denied("no such session".into()))?;
        let new_harness = if harness.is_empty() { ident.harness.clone() } else { harness };
        let (new, state) = match (model.is_empty(), ident.model_state) {
            (true, state) => (ident.model.clone(), state),
            (false, ModelState::Verified) if model == ident.model => (model, ModelState::Verified),
            (false, ModelState::Verified) => return Err(Error::Denied("the model of this session was verified; a claim cannot change it".into())),
            (false, _) => (model, ModelState::Claimed),
        };
        let changed = (new.as_str(), new_harness.as_str(), state) != (ident.model.as_str(), ident.harness.as_str(), ident.model_state);
        (ident.model, ident.harness, ident.model_state) = (new, new_harness, state);
        if changed {
            self.save()?;
        }
        Ok(changed)
    }

    /// Record that the node checked ``name`` runs ``model``; only the node's own checks call this.
    pub fn verify_model(&mut self, name: &str, model: &str) -> Result<()> {
        let model = clean(model, 256)?;
        let ident = self.names.get_mut(name).ok_or_else(|| Error::Denied("no such session".into()))?;
        (ident.model, ident.model_state) = (model, ModelState::Verified);
        self.save()
    }

    /// Mark ``name`` retired; true when it was not already.
    pub fn retire(&mut self, name: &str) -> Result<bool> {
        let ident = self.names.get_mut(name).ok_or_else(|| Error::Denied("no such session".into()))?;
        let fresh = !ident.retired;
        ident.retired = true;
        self.save()?;
        Ok(fresh)
    }

    /// Note that ``name`` was just active, writing at most every 30 seconds.
    pub fn touch(&mut self, name: &str, now_ms: u64) -> Result<()> {
        match self.names.get_mut(name) {
            Some(i) if now_ms >= i.seen_ms + 30_000 => {
                i.seen_ms = now_ms;
                self.save()
            }
            _ => Ok(()),
        }
    }

    /// ``name`` and everything below it by parent links.
    pub fn descendants(&self, name: &str) -> Vec<String> {
        let mut out = Vec::new();
        let mut frontier = vec![name.to_string()];
        while let Some(top) = frontier.pop() {
            for (n, i) in &self.names {
                if i.parent == top && !out.contains(n) {
                    out.push(n.clone());
                    frontier.push(n.clone());
                }
            }
        }
        out
    }

    pub fn len(&self) -> usize {
        self.names.len()
    }

    pub fn is_empty(&self) -> bool {
        self.names.is_empty()
    }

    /// The unique name of this session, recorded so no later session takes the same one. The
    /// same session always gets the same name back (``true`` when it is new); a session whose
    /// short suffix collides with another's gets a longer one.
    pub fn assign(&mut self, model: &str, state: ModelState, harness: &str, session: &str, parent: &str, now_ms: u64) -> Result<(String, bool)> {
        if session.is_empty() || harness.is_empty() {
            return Err(Error::Denied("a session name needs the native harness and session id".into()));
        }
        let (full, word) = (digest(harness, session), family_word(model));
        let model = clean(model, 256)?;
        if let Some((name, known)) = self.names.iter().find(|(_, i)| i.digest == full) {
            if known.retired {
                return Err(Error::Denied(format!("{name} was retired; its session cannot register again")));
            }
            if known.parent != parent {
                return Err(Error::Denied(format!("{name} is already registered under another parent")));
            }
            return Ok((name.clone(), false));
        }
        if self.names.len() >= MAX_IDENTITIES {
            return Err(Error::Quota("this board holds as many sessions as it will".into()));
        }
        let name = (SHORT..=full.len()).step_by(2).map(|w| format!("{word}-{}", &full[..w])).find(|n| !self.names.contains_key(n))
            .ok_or_else(|| Error::Quota("no free name".into()))?;
        let state = if model.is_empty() { ModelState::Unknown } else { state };
        self.names.insert(name.clone(), Ident {
            digest: full, family: word.into(), parent: parent.into(), model, model_state: state, harness: clean(harness, 64)?,
            retired: false, seen_ms: now_ms,
        });
        self.save()?;
        Ok((name, true))
    }
}

/// Where a token leads: a board and a name on it.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Holder {
    pub board: String,
    pub name: String,
}

/// The tokens of this node, kept as hashes in a file only its owner reads.
pub struct Tokens {
    path: PathBuf,
    held: BTreeMap<String, Holder>,
}

impl Tokens {
    pub fn open(path: &Path) -> Result<Tokens> {
        let held = match std::fs::read(path) {
            Ok(bytes) => serde_json::from_slice(&bytes).map_err(|e| Error::Damaged(format!("tokens unreadable: {e}")))?,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => BTreeMap::new(),
            Err(e) => return Err(e.into()),
        };
        Ok(Tokens { path: path.into(), held })
    }

    /// A fresh random token for ``board``/``name``; the oldest of a name's tokens is dropped
    /// past a few. The token itself is returned once and never stored.
    pub fn issue(&mut self, board: &str, name: &str) -> Result<String> {
        let token = random_hex(32)?;
        let holder = Holder { board: board.into(), name: name.into() };
        let same: Vec<String> = self.held.iter().filter(|(_, h)| **h == holder).map(|(k, _)| k.clone()).collect();
        for old in same.iter().take((same.len() + 1).saturating_sub(MAX_TOKENS_PER_NAME)) {
            self.held.remove(old);
        }
        self.held.insert(sha256_hex(token.as_bytes()), holder);
        write_atomic(&self.path, &serde_json::to_vec(&self.held)?)?;
        Ok(token)
    }

    /// Drop every token of ``board``/``name``; how many there were.
    pub fn revoke_name(&mut self, board: &str, name: &str) -> Result<usize> {
        let before = self.held.len();
        self.held.retain(|_, h| !(h.board == board && h.name == name));
        let gone = before - self.held.len();
        if gone > 0 {
            write_atomic(&self.path, &serde_json::to_vec(&self.held)?)?;
        }
        Ok(gone)
    }

    /// Who ``token`` belongs to.
    pub fn resolve(&self, token: &str) -> Option<&Holder> {
        self.held.get(&sha256_hex(token.as_bytes()))
    }
}
