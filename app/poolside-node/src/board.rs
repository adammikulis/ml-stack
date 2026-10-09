//! One board on one device: its own signed log and verified copies of the other origins' logs.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use ed25519_dalek::{Signer, SigningKey};
use serde_json::{json, Value};

use crate::device::{fingerprint, public_hex};
use crate::error::{Error, Result};
use crate::fsutil::{hex, private_dir, random_hex, unhex, write_atomic};
use crate::log::Log;
use crate::row::{check_body, valid_actor, valid_name, valid_origin, is_line, Kind, Row, GENESIS, MAX_ROW_BYTES, VERSION};
use crate::rules::{self, head_message, held_back, tick};
use crate::state::{Pin, State};

pub const MAX_ORIGINS: usize = 64;
pub const MAX_LOG_ROWS: usize = 200_000;
const MAX_REJECTED: usize = 200;

/// The sequence number and hash of the last row of a log.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Tip {
    pub seq: u64,
    pub hash: String,
}

pub struct Board {
    pub id: String,
    dir: PathBuf,
    pool: String,
    key: SigningKey,
    origin: String,
    logs: BTreeMap<String, Log>,
    state: State,
    clock: Box<dyn Fn() -> u64 + Send>,
}

fn wall_ms() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_millis() as u64)
}

impl Board {
    /// Open the board ``id`` under ``dir``, creating it on first use.
    pub fn open(dir: &Path, id: &str, pool: &str, key: SigningKey) -> Result<Board> {
        Board::open_with_clock(dir, id, pool, key, Box::new(wall_ms))
    }

    pub fn open_with_clock(dir: &Path, id: &str, pool: &str, key: SigningKey, clock: Box<dyn Fn() -> u64 + Send>) -> Result<Board> {
        if !valid_name(id) {
            return Err(Error::Invalid("a board id is a short lower-case word".into()));
        }
        private_dir(dir)?;
        let origin = Board::origin_of(dir)?;
        let state = State::open(&dir.join("state.json"))?;
        let mut logs = BTreeMap::new();
        for entry in std::fs::read_dir(dir)? {
            let path = entry?.path();
            let origin = path.file_stem().and_then(|s| s.to_str()).unwrap_or("").to_string();
            if path.extension().is_some_and(|e| e == "jsonl") && valid_origin(&origin) {
                logs.insert(origin.clone(), Log::open(&path, id, &origin)?);
            }
        }
        let mut board = Board { id: id.into(), dir: dir.into(), pool: pool.into(), key, origin: origin.clone(), logs, state, clock };
        board.log_mut(&origin)?;
        Ok(board)
    }

    fn origin_of(dir: &Path) -> Result<String> {
        let path = dir.join("origin");
        match std::fs::read_to_string(&path) {
            Ok(text) if valid_origin(text.trim()) => Ok(text.trim().to_string()),
            Ok(_) => Err(Error::Damaged("origin file unreadable".into())),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                let origin = random_hex(16)?;
                write_atomic(&path, origin.as_bytes())?;
                Ok(origin)
            }
            Err(e) => Err(e.into()),
        }
    }

    pub fn origin(&self) -> &str {
        &self.origin
    }

    pub fn pool(&self) -> &str {
        &self.pool
    }

    pub fn fingerprint(&self) -> String {
        fingerprint(&self.key)
    }

    pub fn now_ms(&self) -> u64 {
        (self.clock)()
    }

    fn log_mut(&mut self, origin: &str) -> Result<&mut Log> {
        if !self.logs.contains_key(origin) {
            let log = Log::open(&self.dir.join(format!("{origin}.jsonl")), &self.id, origin)?;
            self.logs.insert(origin.to_string(), log);
        }
        Ok(self.logs.get_mut(origin).expect("inserted above"))
    }

    pub fn rows(&self, origin: &str) -> &[Row] {
        self.logs.get(origin).map_or(&[], |l| l.rows())
    }

    pub fn origins(&self) -> Vec<String> {
        self.logs.keys().cloned().collect()
    }

    /// For each held origin, the tip of its log.
    pub fn vector(&self) -> BTreeMap<String, Tip> {
        let tip = |l: &Log| l.rows().last().map(|r| Tip { seq: r.seq, hash: r.hash.clone() });
        self.logs.iter().filter_map(|(o, l)| tip(l).map(|t| (o.clone(), t))).collect()
    }

    /// The rows of ``origin`` through its last head: the part a peer may be given.
    pub fn trusted(&self, origin: &str) -> &[Row] {
        let rows = self.rows(origin);
        let last = rows.iter().filter(|r| r.kind == Kind::Head).map(|r| r.seq).max().unwrap_or(0);
        &rows[..last as usize]
    }

    /// Like `vector`, over the trusted rows only.
    pub fn trusted_vector(&self) -> BTreeMap<String, Tip> {
        let tip = |r: &[Row]| r.last().map(|r| Tip { seq: r.seq, hash: r.hash.clone() });
        self.logs.keys().filter_map(|o| tip(self.trusted(o)).map(|t| (o.clone(), t))).collect()
    }

    pub fn damaged(&self) -> &BTreeMap<String, String> {
        &self.state.t.damaged
    }

    /// Add a row to this device's log; a repeated ``idem`` of the same actor returns the row
    /// already written.
    pub fn append(&mut self, kind: Kind, actor: &str, body: Value, idem: &str) -> Result<Row> {
        if kind == Kind::Head || !(actor.is_empty() || valid_actor(actor)) || idem.len() > 128 || !is_line(idem) {
            return Err(Error::Invalid("a row needs a real kind, a usable actor and a clean key".into()));
        }
        check_body(&body)?;
        let origin = self.origin.clone();
        if let Some(found) = self.rows(&origin).iter().find(|r| !idem.is_empty() && r.actor == actor && r.idem == idem) {
            return Ok(found.clone());
        }
        self.add(kind, actor, idem, body)
    }

    fn add(&mut self, kind: Kind, actor: &str, idem: &str, body: Value) -> Result<Row> {
        let origin = self.origin.clone();
        let (last, seen, now) = (self.rows(&origin).last().cloned(), self.state.t.seen, self.now_ms());
        if self.rows(&origin).len() >= MAX_LOG_ROWS {
            return Err(Error::Quota("this log is past the rows a device keeps".into()));
        }
        let (wall, counter) = tick(last.as_ref().map_or((0, 0), |r| (r.hlc.0, r.hlc.1)), now, seen);
        let mut row = Row {
            v: VERSION, board: self.id.clone(), origin: origin.clone(),
            seq: last.as_ref().map_or(1, |r| r.seq + 1),
            prev: last.as_ref().map_or(GENESIS.to_string(), |r| r.hash.clone()),
            hlc: (wall, counter, origin.clone()), kind, actor: actor.into(), idem: idem.into(), body, hash: String::new(),
        };
        row.hash = row.digest();
        if row.size() > MAX_ROW_BYTES {
            return Err(Error::Quota("the row is larger than a row may be".into()));
        }
        self.log_mut(&origin)?.append(row.clone())?;
        Ok(row)
    }

    /// Sign the last row of this device's log with a head row; None when already covered.
    pub fn seal(&mut self) -> Result<Option<Row>> {
        let Some(tip) = self.rows(&self.origin.clone()).last().cloned() else { return Ok(None) };
        if tip.kind == Kind::Head {
            return Ok(None);
        }
        let signature = self.key.sign(&head_message(&self.pool, &self.id, &self.origin, tip.seq, &tip.hash));
        let body = json!({"head": tip.seq, "hash": tip.hash, "public": public_hex(&self.key), "sig": hex(&signature.to_bytes())});
        self.add(Kind::Head, "", "", body).map(Some)
    }

    /// Store the rows of another device's log that verify; returns how many. A forged or
    /// forked copy is `Damaged`; it marks the origin damaged only when it came from the device
    /// that owns the origin (``authoritative``), never when a relay passed it on.
    pub fn ingest(&mut self, origin: &str, incoming: &[Row], authoritative: bool) -> Result<usize> {
        if !valid_origin(origin) {
            return Err(Error::Invalid("origin is not a log id".into()));
        }
        if origin == self.origin {
            return Ok(0);
        }
        if let Some(why) = self.state.t.damaged.get(origin) {
            return Err(Error::Damaged(format!("log {origin} was refused: {why}")));
        }
        if !self.logs.contains_key(origin) && self.logs.len() >= MAX_ORIGINS {
            return Err(Error::Quota("this device holds as many logs as it will".into()));
        }
        match self.store(origin, incoming, authoritative) {
            Err(Error::Damaged(why)) if authoritative => {
                self.state.t.damaged.insert(origin.into(), why.chars().take(200).collect());
                self.state.save()?;
                Err(Error::Damaged(why))
            }
            other => other,
        }
    }

    fn store(&mut self, origin: &str, incoming: &[Row], authoritative: bool) -> Result<usize> {
        let pin = self.state.t.keys.get(origin).cloned();
        let key_of = |p: &Option<Pin>| p.as_ref().and_then(|p| unhex(&p.public)).and_then(|b| <[u8; 32]>::try_from(b).ok());
        let held: Vec<Row> = self.rows(origin).to_vec();
        let (board, pool) = (self.id.clone(), self.pool.clone());
        let (took, pin) = match rules::accept(&board, origin, &held, incoming, key_of(&pin), &pool) {
            Ok(t) => (t, pin),
            Err(Error::Damaged(_)) if authoritative && pin.as_ref().is_some_and(|p| !p.bound) => {
                self.reset(origin)?;
                (rules::accept(&board, origin, &[], incoming, None, &pool)?, None)
            }
            Err(e) => return Err(e),
        };
        if self.rows(origin).len() + took.rows.len() > MAX_LOG_ROWS {
            return Err(Error::Quota("this log is past the rows a device keeps".into()));
        }
        if took.rows.is_empty() {
            return Ok(0);
        }
        let count = took.rows.len();
        self.observe(&took.rows);
        self.log_mut(origin)?.extend(took.rows)?;
        if pin.as_ref().is_none_or(|p| authoritative && !p.bound) {
            let public = took.public.map(|p| hex(&p)).unwrap_or_default();
            self.state.t.keys.insert(origin.into(), Pin { public, bound: authoritative });
        }
        self.state.save()?;
        Ok(count)
    }

    fn reset(&mut self, origin: &str) -> Result<()> {
        if let Some(log) = self.logs.remove(origin) {
            log.remove()?;
        }
        Ok(())
    }

    /// Drop the refusal of ``origin`` and the copy held, so it is fetched again.
    pub fn forgive(&mut self, origin: &str) -> Result<()> {
        self.reset(origin)?;
        self.state.t.damaged.remove(origin);
        self.state.t.keys.remove(origin);
        self.state.save()
    }

    fn observe(&mut self, rows: &[Row]) {
        let now = self.now_ms();
        let top = rows.iter().filter(|r| !held_back(r, now)).map(|r| (r.hlc.0, r.hlc.1)).max();
        self.state.t.seen = self.state.t.seen.max(top.unwrap_or((0, 0)));
    }

    /// Tie a device to the origin it writes the first time it is seen; false when it already
    /// writes a different one.
    pub fn bind_peer(&mut self, fingerprint: &str, origin: &str) -> Result<bool> {
        let bound = self.state.t.peers.entry(fingerprint.into()).or_insert_with(|| origin.into()).clone();
        self.state.save()?;
        Ok(bound == origin)
    }

    /// Whether the device ``fingerprint`` is the writer of ``origin``.
    pub fn owns(&self, fingerprint: &str, origin: &str) -> bool {
        !fingerprint.is_empty() && self.state.t.peers.get(fingerprint).is_some_and(|o| o == origin)
    }

    /// Record how much of this device's log the device ``fingerprint`` holds, when its hash
    /// for that row is the one held here; true when it holds less than it claimed before.
    pub fn acknowledge(&mut self, fingerprint: &str, vector: &BTreeMap<String, Tip>) -> Result<bool> {
        let rows = self.rows(&self.origin);
        let seq = vector.get(&self.origin).filter(|t| t.seq >= 1 && t.seq as usize <= rows.len() && rows[t.seq as usize - 1].hash == t.hash).map_or(0, |t| t.seq);
        let before = self.state.t.acks.get(fingerprint).copied().unwrap_or(0);
        if seq > before {
            self.state.t.acks.insert(fingerprint.into(), seq);
            self.state.save()?;
        }
        Ok(seq < before)
    }

    /// `synced` once every device of ``roster`` holds row ``seq`` of this log, else `provisional`.
    pub fn status(&self, seq: u64, roster: &[String]) -> &'static str {
        let behind = roster.iter().any(|f| self.state.t.acks.get(f).copied().unwrap_or(0) < seq);
        if behind { "provisional" } else { "synced" }
    }

    /// The local name of a foreign origin, `d1`, `d2`, in the order first seen.
    pub fn label(&mut self, origin: &str) -> Result<String> {
        if let Some(label) = self.state.t.labels.get(origin) {
            return Ok(label.clone());
        }
        let label = format!("d{}", self.state.t.labels.len() + 1);
        self.state.t.labels.insert(origin.into(), label.clone());
        self.state.save()?;
        Ok(label)
    }

    /// Record a foreign row that was not folded, keeping the latest few.
    pub fn reject(&mut self, id: &str, reason: &str) -> Result<()> {
        let reason: String = reason.chars().filter(|c| !c.is_control()).take(200).collect();
        let rejected = &mut self.state.t.rejected;
        if !rejected.iter().any(|(i, _)| i == id) {
            rejected.push((id.into(), reason));
            let excess = rejected.len().saturating_sub(MAX_REJECTED);
            rejected.drain(..excess);
            self.state.save()?;
        }
        Ok(())
    }

    pub fn rejected(&self) -> &[(String, String)] {
        &self.state.t.rejected
    }

    /// Every log's rows in one total order; a foreign row dated too far ahead is held back.
    pub fn merged(&self) -> Vec<Row> {
        let now = self.now_ms();
        let own = &self.origin;
        let trimmed: Vec<Vec<Row>> = self.logs.iter().map(|(o, l)| {
            let rows = l.rows();
            let cut = if o == own { rows.len() } else { rows.iter().position(|r| held_back(r, now)).unwrap_or(rows.len()) };
            rows[..cut].to_vec()
        }).collect();
        rules::merge(&trimmed.iter().map(|v| v.as_slice()).collect::<Vec<_>>())
    }
}
