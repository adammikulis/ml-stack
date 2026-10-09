//! Pure rules for per-origin logs: the clock, chain and head checks, and the merge.

use std::collections::BTreeSet;

use ed25519_dalek::{Signature, VerifyingKey};

use crate::error::{Error, Result};
use crate::fsutil::unhex;
use crate::row::{check_body, valid_name, valid_origin, Kind, Row, GENESIS, MAX_ROW_BYTES, VERSION};

/// How far ahead of the local clock a foreign row may be before it is held back.
pub const SKEW_MAX_MS: u64 = 300_000;

/// The rows to store for an origin and the key its heads verified under.
pub struct Accepted {
    pub rows: Vec<Row>,
    pub public: Option<[u8; 32]>,
}

/// The next clock value after ``previous``, the wall clock and ``seen``.
pub fn tick(previous: (u64, u64), now_ms: u64, seen: (u64, u64)) -> (u64, u64) {
    let wall = now_ms.max(previous.0).max(seen.0);
    let from = |pair: (u64, u64)| if pair.0 == wall { Some(pair.1) } else { None };
    let counter = from(previous).into_iter().chain(from(seen)).max().map_or(0, |c| c + 1);
    (wall, counter)
}

/// Whether the row's wall clock is further ahead of ``now_ms`` than the allowed skew.
pub fn held_back(row: &Row, now_ms: u64) -> bool {
    row.hlc.0 > now_ms.saturating_add(SKEW_MAX_MS)
}

/// The bytes a head signs.
pub fn head_message(pool: &str, board: &str, origin: &str, seq: u64, digest: &str) -> Vec<u8> {
    format!("poolside-board-head-v1|{pool}|{board}|{origin}|{seq}|{digest}").into_bytes()
}

/// Every log's rows in one total order: heads dropped, a row delivered twice kept once, and a
/// repeated idempotency key of one actor on one origin kept only at its first row.
pub fn merge(logs: &[&[Row]]) -> Vec<Row> {
    let mut rows: Vec<&Row> = logs.iter().flat_map(|l| l.iter()).filter(|r| r.kind != Kind::Head).collect();
    rows.sort_by(|a, b| a.order().cmp(&b.order()));
    let mut ids = BTreeSet::new();
    let mut keys = BTreeSet::new();
    let mut kept = Vec::new();
    for row in rows {
        if !ids.insert((row.origin.clone(), row.seq)) {
            continue;
        }
        if !row.idem.is_empty() && !keys.insert((row.origin.clone(), row.actor.clone(), row.idem.clone())) {
            continue;
        }
        kept.push(row.clone());
    }
    kept
}

fn shape(row: &Row, board: &str, origin: &str) -> Result<()> {
    if row.v != VERSION || row.board != board || row.origin != origin {
        return Err(Error::Damaged("row does not belong to this log".into()));
    }
    if row.hlc.2 != origin {
        return Err(Error::Damaged("row carries no clock value".into()));
    }
    if check_body(&row.body).is_err() {
        return Err(Error::Damaged("row body is not allowed".into()));
    }
    Ok(())
}

/// Check ``rows`` follow one another (and ``tip``) by sequence, hash and clock.
pub fn check_chain(rows: &[Row], board: &str, origin: &str, tip: Option<&Row>) -> Result<()> {
    let (mut prev, mut seq, mut wall) = tip.map_or((GENESIS.to_string(), 0, 0), |t| (t.hash.clone(), t.seq, t.hlc.0));
    for row in rows {
        shape(row, board, origin)?;
        if row.seq != seq + 1 || row.prev != prev {
            return Err(Error::Damaged(format!("row {} does not follow row {seq}", row.seq)));
        }
        if row.hash != row.digest() {
            return Err(Error::Damaged(format!("row {} hash does not match its contents", seq + 1)));
        }
        if row.hlc.0 < wall {
            return Err(Error::Damaged(format!("row {} moves the wall clock backwards", seq + 1)));
        }
        (prev, seq, wall) = (row.hash.clone(), row.seq, row.hlc.0);
    }
    Ok(())
}

fn head_fields(row: &Row) -> Result<([u8; 32], Vec<u8>, u64, String)> {
    let bad = || Error::Damaged(format!("head at row {} is malformed", row.seq));
    let text = |k: &str| row.body.get(k).and_then(|v| v.as_str()).ok_or_else(bad);
    let public: [u8; 32] = unhex(text("public")?).and_then(|b| b.try_into().ok()).ok_or_else(bad)?;
    let sig = unhex(text("sig")?).ok_or_else(bad)?;
    let target = row.body.get("head").and_then(|v| v.as_u64()).ok_or_else(bad)?;
    Ok((public, sig, target, text("hash")?.to_string()))
}

fn check_heads(new: &[Row], tops: &std::collections::BTreeMap<u64, String>, pinned: Option<[u8; 32]>, pool: &str)
    -> Result<(Option<[u8; 32]>, u64)> {
    let (mut public, mut last) = (pinned, 0);
    for row in new.iter().filter(|r| r.kind == Kind::Head) {
        let (key, sig, target, digest) = head_fields(row)?;
        if public.is_some_and(|p| p != key) {
            return Err(Error::Damaged(format!("head at row {} is signed by a different key", row.seq)));
        }
        if target + 1 != row.seq || tops.get(&target) != Some(&digest) {
            return Err(Error::Damaged(format!("head at row {} does not sign the row before it", row.seq)));
        }
        let message = head_message(pool, &row.board, &row.origin, target, &digest);
        let verified = VerifyingKey::from_bytes(&key).ok().zip(Signature::from_slice(&sig).ok())
            .is_some_and(|(k, s)| k.verify_strict(&message, &s).is_ok());
        if !verified {
            return Err(Error::Damaged(format!("head at row {} has a bad signature", row.seq)));
        }
        (public, last) = (Some(key), row.seq);
    }
    Ok((public, last))
}

/// The rows of ``incoming`` to append to the ``held`` copy of ``origin``'s log: those past
/// what is held, through the last row a verified head covers. `Damaged` for a fork, a broken
/// chain or a forged head; `Gap` when the rows start after the held end; `Invalid` for a row
/// that is oversized or of a bad origin (never evidence against the origin).
pub fn accept(board: &str, origin: &str, held: &[Row], incoming: &[Row], pinned: Option<[u8; 32]>, pool: &str)
    -> Result<Accepted> {
    if !valid_origin(origin) || !valid_name(board) {
        return Err(Error::Invalid("origin or board is not an id".into()));
    }
    if incoming.iter().any(|r| r.size() > MAX_ROW_BYTES) {
        return Err(Error::Invalid("a row exceeds its size bound".into()));
    }
    let Some(first) = incoming.first().map(|r| r.seq) else {
        return Ok(Accepted { rows: vec![], public: pinned });
    };
    if first < 1 {
        return Err(Error::Damaged("rows carry no sequence number".into()));
    }
    if first as usize > held.len() + 1 {
        return Err(Error::Gap(format!("rows start at {first} but {} are held", held.len())));
    }
    let start = first as usize - 1;
    let overlap = &held[start..held.len().min(start + incoming.len())];
    if let Some((mine, _)) = overlap.iter().zip(incoming).find(|(m, t)| m.hash != t.hash) {
        return Err(Error::Damaged(format!("row {} differs from the held copy", mine.seq)));
    }
    let new = &incoming[overlap.len()..];
    if new.is_empty() {
        return Ok(Accepted { rows: vec![], public: pinned });
    }
    check_chain(new, board, origin, held.last())?;
    let tops = held.iter().chain(new).map(|r| (r.seq, r.hash.clone())).collect();
    let (public, last) = check_heads(new, &tops, pinned, pool)?;
    Ok(Accepted { rows: new.iter().filter(|r| r.seq <= last).cloned().collect(), public })
}
