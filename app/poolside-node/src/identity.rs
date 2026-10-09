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

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Ident {
    pub digest: String,
    pub family: String,
    pub parent: String,
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

    pub fn len(&self) -> usize {
        self.names.len()
    }

    pub fn is_empty(&self) -> bool {
        self.names.is_empty()
    }

    /// The unique name of this session, recorded so no later session takes the same one. The
    /// same session always gets the same name back (``true`` when it is new); a session whose
    /// short suffix collides with another's gets a longer one.
    pub fn assign(&mut self, model: &str, harness: &str, session: &str, parent: &str) -> Result<(String, bool)> {
        if session.is_empty() || harness.is_empty() {
            return Err(Error::Denied("a session name needs the native harness and session id".into()));
        }
        let (full, word) = (digest(harness, session), family_word(model));
        if let Some((name, _)) = self.names.iter().find(|(_, i)| i.digest == full) {
            return Ok((name.clone(), false));
        }
        if self.names.len() >= MAX_IDENTITIES {
            return Err(Error::Quota("this board holds as many sessions as it will".into()));
        }
        let name = (SHORT..=full.len()).step_by(2).map(|w| format!("{word}-{}", &full[..w])).find(|n| !self.names.contains_key(n))
            .ok_or_else(|| Error::Quota("no free name".into()))?;
        self.names.insert(name.clone(), Ident { digest: full, family: word.into(), parent: parent.into() });
        write_atomic(&self.path, &serde_json::to_vec(&self.names)?)?;
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

    /// Who ``token`` belongs to.
    pub fn resolve(&self, token: &str) -> Option<&Holder> {
        self.held.get(&sha256_hex(token.as_bytes()))
    }
}
