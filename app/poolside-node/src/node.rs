//! The node: every board this device hosts, the tokens, the links, and what a session may do.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::Instant;

use ed25519_dalek::SigningKey;
use serde_json::{json, Map, Value};

use crate::board::Board;
use crate::device::{fingerprint, load_or_create};
use crate::error::{Error, Result};
use crate::fold::{self, check_local, holders, Context, Entry};
use crate::fsutil::private_dir;
use crate::identity::{Holder, Registry, Tokens};
use crate::links::Links;
use crate::registry::Projects;
use crate::row::{valid_name, Kind};

pub const READ_DEFAULT: usize = 200;
pub const READ_MAX: usize = 1000;

/// One hosted board and the names registered on it.
pub struct Hosted {
    pub board: Board,
    pub names: Registry,
}

pub struct Node {
    pub dir: PathBuf,
    pub pool: String,
    pub key: SigningKey,
    pub boards: BTreeMap<String, Hosted>,
    pub tokens: Tokens,
    pub links: Links,
    pub projects: Projects,
    pub stop: Arc<AtomicBool>,
    pub started: Instant,
}

/// What a request is allowed to touch, decided from its token and the board it names.
pub struct Access {
    pub holder: Holder,
    /// The actor name rows are written under on the target board.
    pub actor: String,
    /// Channels of the target usable by this session; `None` means all of the board.
    pub channels: Option<Vec<String>>,
}

impl Node {
    /// Open the node whose state lives in ``dir`` (a directory only its owner can enter).
    pub fn open(dir: &Path) -> Result<Node> {
        private_dir(dir)?;
        let key = load_or_create(dir)?;
        let mut node = Node {
            dir: dir.into(), pool: String::new(), key, boards: BTreeMap::new(),
            tokens: Tokens::open(&dir.join("tokens.json"))?, links: Links::open(&dir.join("links.json"))?,
            projects: Projects::open(&dir.join("projects.json"))?,
            stop: Arc::new(AtomicBool::new(false)), started: Instant::now(),
        };
        let root = dir.join("boards");
        if root.is_dir() {
            for entry in std::fs::read_dir(&root)? {
                let id = entry?.file_name().to_string_lossy().to_string();
                if valid_name(&id) {
                    node.host(&id)?;
                }
            }
        }
        Ok(node)
    }

    pub fn fingerprint(&self) -> String {
        fingerprint(&self.key)
    }

    /// The board ``id``, opened (and created) under this node's directory.
    pub fn host(&mut self, id: &str) -> Result<&mut Hosted> {
        if !self.boards.contains_key(id) {
            if !valid_name(id) {
                return Err(Error::Invalid("a board id is a short lower-case word".into()));
            }
            let dir = self.dir.join("boards").join(id);
            private_dir(&dir)?;
            let board = Board::open(&dir.join("log"), id, &self.pool, self.key.clone())?;
            let names = Registry::open(&dir.join("names.json"))?;
            self.boards.insert(id.into(), Hosted { board, names });
        }
        Ok(self.boards.get_mut(id).expect("hosted above"))
    }

    fn hosted(&mut self, id: &str) -> Result<&mut Hosted> {
        self.boards.get_mut(id).ok_or_else(|| Error::Denied("this node does not host that board".into()))
    }

    /// Decide what ``token`` may do on ``board``: its own board fully, another only through a link.
    pub fn access(&self, token: &str, board: &str, write: bool) -> Result<Access> {
        let holder = self.tokens.resolve(token).cloned().ok_or_else(|| Error::Denied("the token is not known".into()))?;
        if !self.boards.contains_key(board) {
            return Err(Error::Denied("this node does not host that board".into()));
        }
        if holder.board == board {
            return Ok(Access { actor: holder.name.clone(), holder, channels: None });
        }
        match self.links.grants(&holder.board, board, write) {
            Some(channels) => Ok(Access {
                actor: format!("{}@{}", holder.name, holder.board), holder,
                channels: if channels.is_empty() { None } else { Some(channels) },
            }),
            None => Err(Error::Denied("the session is not registered on that board and no link shares it".into())),
        }
    }

    /// Register a session on ``board`` (creating the board on first use); a token given makes
    /// the new session a subagent of the token's owner. Returns name, token, whether new.
    pub fn register(&mut self, board: &str, parent_token: Option<&str>, model: &str, harness: &str, session: &str) -> Result<Value> {
        let parent = match parent_token {
            None => String::new(),
            Some(t) => match self.tokens.resolve(t) {
                Some(h) if h.board == board => h.name.clone(),
                _ => return Err(Error::Denied("a parent must be a session of this board".into())),
            },
        };
        let hosted = self.host(board)?;
        let (name, created) = hosted.names.assign(model, harness, session, &parent)?;
        if created {
            let family = hosted.names.get(&name).map(|i| i.family.clone()).unwrap_or_default();
            hosted.board.append(Kind::Identity, &name, json!({"name": name, "parent": parent, "family": family}), "identity")?;
        }
        let origin = hosted.board.origin().to_string();
        let token = self.tokens.issue(board, &name)?;
        Ok(json!({"name": name, "token": token, "created": created, "parent": parent, "board": board, "origin": origin}))
    }

    /// The folded entries of ``board``, in merged order, with the foreign rows that were refused recorded.
    pub fn view(&mut self, board: &str) -> Result<Vec<Entry>> {
        let Hosted { board, names } = self.hosted(board)?;
        let rows = board.merged();
        let own = board.origin().to_string();
        let mut labels: BTreeMap<String, String> = BTreeMap::new();
        for r in rows.iter().filter(|r| r.origin != own) {
            if !labels.contains_key(&r.origin) {
                let l = board.label(&r.origin)?;
                labels.insert(r.origin.clone(), l);
            }
        }
        let mut label = |o: &str| labels.get(o).cloned().unwrap_or_default();
        let is_local = |n: &str| names.contains(n);
        let (entries, rejected) = fold::fold(&rows, Context { own_origin: &own, label: &mut label, is_local: &is_local });
        for (id, why) in rejected {
            board.reject(&id, &why)?;
        }
        Ok(entries)
    }

    /// Write a message or note as the session behind ``token``.
    pub fn post(&mut self, token: &str, board: &str, kind: Kind, fields: &Map<String, Value>, idem: &str) -> Result<Value> {
        let channel = fold::channel_of(kind, fields);
        let access = self.access(token, board, true)?;
        if access.channels.as_ref().is_some_and(|c| !c.contains(&channel)) {
            return Err(Error::Denied("that channel is not shared with this session".into()));
        }
        let mut body = fields.clone();
        body.insert(if kind == Kind::Note { "author" } else { "from" }.into(), Value::String(access.actor.clone()));
        let body = Value::Object(body);
        check_local(kind, &access.actor, &body, "")?;
        let hosted = self.hosted(board)?;
        let row = hosted.board.append(kind, &access.actor, body, idem)?;
        Ok(json!({"id": row.id(), "seq": row.seq, "sender": access.actor, "status": hosted.board.status(row.seq, &[])}))
    }

    /// Claim (or release) ``target`` for the session behind ``token``.
    pub fn claim(&mut self, token: &str, board: &str, target: &str, note: &str, release: bool) -> Result<Value> {
        let access = self.access(token, board, true)?;
        if access.channels.as_ref().is_some_and(|c| !c.iter().any(|x| x == "#claims")) {
            return Err(Error::Denied("claims are not shared with this session".into()));
        }
        let held = holders(&self.view(board)?).get(target).cloned();
        let mine = held.as_deref() == Some(access.actor.as_str());
        match (release, held) {
            (false, None) | (true, Some(_)) if release == mine || !release => {}
            (false, Some(_)) if mine => return Ok(json!({"target": target, "holder": access.actor, "changed": false})),
            (false, Some(who)) => return Err(Error::Denied(format!("held by {who}"))),
            _ => return Err(Error::Denied("only the holder can release".into())),
        }
        let action = if release { "release" } else { "claim" };
        let body = json!({"target": target, "action": action, "note": note});
        check_local(Kind::Claim, &access.actor, &body, "")?;
        self.hosted(board)?.board.append(Kind::Claim, &access.actor, body, "")?;
        Ok(json!({"target": target, "holder": if release { Value::Null } else { json!(access.actor) }, "changed": true}))
    }
}
