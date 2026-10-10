//! The view of a board: merged rows turned into entries through explicit field allow-lists.
//!
//! A row from another device is never trusted for what it says about itself: only the fields
//! named here are copied, each checked for type and size, and the sender is the log's actor,
//! shown as `actor@dN`.

use std::collections::BTreeMap;

use serde_json::{Map, Value};

use crate::error::{Error, Result};
use crate::row::{is_line, valid_actor, valid_name, Kind, Row};

pub const GENERAL: &str = "#general";
pub const ANNOUNCE: &str = "#announcements";
/// The channel of a message to one session; never shared by a link.
pub const DIRECT: &str = "#dm";
const MESSAGE_TYPES: [&str; 9] = ["task", "status", "handoff", "question", "answer", "claim", "release", "note", "file"];
const ANNOUNCE_TYPES: [&str; 4] = ["joined", "milestone", "done", "blocked"];
const NOTE_KINDS: [&str; 4] = ["decision", "rule", "fact", "question"];
const MESSAGE_KEYS: [&str; 7] = ["type", "from", "to", "subject", "body", "reply_id", "thread_id"];
const NOTE_KEYS: [&str; 9] = ["nkind", "title", "body", "source", "tags", "author", "supersedes", "verify_cmd", "ttl_days"];
const IDENTITY_KEYS: [&str; 7] = ["name", "parent", "family", "model", "model_state", "harness", "retired"];
const VERIFY_KEYS: [&str; 4] = ["note", "exit", "cmd", "out_sha"];
const LEASE_KEYS: [&str; 3] = ["lease", "action", "resources"];
const LEASE_ACTIONS: [&str; 5] = ["acquire", "release", "expire", "dead", "abandon"];
const AUDIT_KEYS: [&str; 3] = ["event", "subject", "detail"];
const LANDING_KEYS: [&str; 13] = ["ev", "branch", "sha", "target", "selectors", "replaces", "req", "verdict", "status", "detail", "evidence", "reason", "what"];
pub const BODY_BYTES: usize = 64 * 1024;
const NOTES_PER_SENDER: usize = 200;

/// One entry as clients see it.
#[derive(Clone, Debug, PartialEq)]
pub struct Entry {
    pub id: String,
    pub origin: String,
    pub seq: u64,
    pub hlc: (u64, u64),
    pub kind: Kind,
    pub sender: String,
    pub foreign: bool,
    pub channel: String,
    pub fields: Map<String, Value>,
}

impl Entry {
    /// Whether ``viewer`` may read this entry: a message to one session is read only by its
    /// sender and its recipient, whatever channels a link shares.
    pub fn readable_by(&self, viewer: &str) -> bool {
        self.channel != DIRECT || self.sender == viewer || self.fields.get("to").and_then(Value::as_str) == Some(viewer)
    }
}

/// What a fold needs to know about the device it runs on.
pub struct Context<'a> {
    pub own_origin: &'a str,
    /// The `dN` label of a foreign origin.
    pub label: &'a mut dyn FnMut(&str) -> String,
    /// Whether a name is registered on this board here.
    pub is_local: &'a dyn Fn(&str) -> bool,
}

/// The channel a kind of entry belongs to, the unit a board link names.
pub fn channel_of(kind: Kind, fields: &Map<String, Value>) -> String {
    match kind {
        Kind::Message => match fields.get("to").and_then(Value::as_str).unwrap_or(GENERAL) {
            to @ (GENERAL | ANNOUNCE) => to.to_string(),
            _ => DIRECT.into(),
        },
        Kind::Note | Kind::Verify => "#notes".into(),
        Kind::Lease => "#leases".into(),
        Kind::Landing => "#landing".into(),
        Kind::Identity => "#identity".into(),
        Kind::Audit => "#audit".into(),
        _ => "#other".into(),
    }
}

fn reject<T>(why: &str) -> Result<T> {
    Err(Error::Invalid(why.into()))
}

fn only(body: &Value, allowed: &[&str]) -> Result<Map<String, Value>> {
    let map = body.as_object().ok_or_else(|| Error::Invalid("body is not an object".into()))?;
    match map.keys().find(|k| !allowed.contains(&k.as_str())) {
        Some(k) => reject(&format!("unknown field {}", k.chars().filter(|c| !c.is_control()).take(40).collect::<String>())),
        None => Ok(map.clone()),
    }
}

fn text<'a>(map: &'a Map<String, Value>, key: &str, most: usize, required: bool) -> Result<&'a str> {
    match map.get(key) {
        None if !required => Ok(""),
        Some(Value::String(s)) if s.len() <= most => Ok(s),
        _ => reject(&format!("{key} is not text within {most} bytes")),
    }
}

fn line<'a>(map: &'a Map<String, Value>, key: &str, most: usize) -> Result<&'a str> {
    let value = text(map, key, most, false)?;
    if is_line(value) { Ok(value) } else { reject(&format!("{key} is not one clean line")) }
}

fn sender(row: &Row, declared: Option<&Value>, ctx: &mut Context, foreign: bool) -> Result<String> {
    if declared.is_some_and(|d| d.as_str() != Some(row.actor.as_str())) || !valid_actor(&row.actor) {
        return reject("the sender is not the actor of the log");
    }
    if !foreign {
        return Ok(row.actor.clone());
    }
    if (ctx.is_local)(&row.actor) {
        return reject("the sender is named like a local session");
    }
    Ok(format!("{}@{}", row.actor, (ctx.label)(&row.origin)))
}

fn message(row: &Row, ctx: &mut Context, foreign: bool) -> Result<(String, Map<String, Value>)> {
    let mut map = only(&row.body, &MESSAGE_KEYS)?;
    let who = sender(row, map.get("from"), ctx, foreign)?;
    let (to, kind) = (text(&map, "to", 64, true)?.to_string(), text(&map, "type", 32, true)?.to_string());
    let announce = to == ANNOUNCE;
    let direct = to != GENERAL && !announce && valid_name(&to);
    if !((to == GENERAL || direct) && MESSAGE_TYPES.contains(&kind.as_str()) || announce && ANNOUNCE_TYPES.contains(&kind.as_str())) {
        return reject("the post is not an announcement, a #general message or a message to one session");
    }
    line(&map, "subject", 200)?;
    text(&map, "body", BODY_BYTES, true)?;
    for link in ["reply_id", "thread_id"] {
        line(&map, link, 128)?;
    }
    map.remove("from");
    Ok((who, map))
}

fn note(row: &Row, ctx: &mut Context, foreign: bool) -> Result<(String, Map<String, Value>)> {
    let mut map = only(&row.body, &NOTE_KEYS)?;
    let who = sender(row, map.get("author"), ctx, foreign)?;
    if !NOTE_KINDS.contains(&text(&map, "nkind", 16, true)?) {
        return reject("the note kind is unknown");
    }
    line(&map, "title", 200)?;
    text(&map, "body", BODY_BYTES, true)?;
    line(&map, "source", 300)?;
    match map.get("tags") {
        None => {}
        Some(Value::Array(tags)) if tags.len() <= 10 && tags.iter().all(|t| t.as_str().is_some_and(|s| s.len() <= 40 && is_line(s))) => {}
        _ => return reject("tags are at most ten short words"),
    }
    match map.get("supersedes") {
        None => {}
        Some(Value::Array(ids)) if ids.len() <= 10 && ids.iter().all(|t| t.as_str().is_some_and(|s| !s.is_empty() && s.len() <= 128 && is_line(s))) => {}
        _ => return reject("supersedes lists at most ten note ids"),
    }
    line(&map, "verify_cmd", 300)?;
    match map.get("ttl_days") {
        None => {}
        Some(v) if v.as_u64().is_some_and(|d| d <= 3650) => {}
        _ => return reject("ttl_days is a whole number of days up to 3650"),
    }
    map.remove("author");
    Ok((who, map))
}

fn identity(row: &Row, ctx: &mut Context, foreign: bool) -> Result<(String, Map<String, Value>)> {
    let map = only(&row.body, &IDENTITY_KEYS)?;
    let who = sender(row, map.get("name"), ctx, foreign)?;
    if !valid_name(text(&map, "name", 48, true)?) || line(&map, "family", 16)?.is_empty() {
        return reject("an identity names a session and its family");
    }
    if !line(&map, "parent", 48)?.is_empty() && !valid_name(text(&map, "parent", 48, true)?) {
        return reject("the parent is not a session name");
    }
    line(&map, "model", 256)?;
    line(&map, "harness", 64)?;
    if !["unknown", "claimed", "inherited", "verified"].contains(&text(&map, "model_state", 16, true)?) {
        return reject("the model state is unknown, claimed, inherited or verified");
    }
    if map.get("retired").is_some_and(|r| !r.is_boolean()) {
        return reject("retired is true or false");
    }
    Ok((who, map))
}

fn verify(row: &Row, ctx: &mut Context, foreign: bool) -> Result<(String, Map<String, Value>)> {
    let map = only(&row.body, &VERIFY_KEYS)?;
    let who = sender(row, None, ctx, foreign)?;
    if line(&map, "note", 128)?.is_empty() || !map.get("exit").is_some_and(|e| e.as_i64().is_some()) {
        return reject("a verification names a note and the exit code of its command");
    }
    line(&map, "cmd", 300)?;
    if text(&map, "out_sha", 64, true)?.len() != 64 {
        return reject("a verification carries the SHA-256 of the command output");
    }
    Ok((who, map))
}

fn lease(row: &Row, ctx: &mut Context, foreign: bool) -> Result<(String, Map<String, Value>)> {
    let map = only(&row.body, &LEASE_KEYS)?;
    let who = sender(row, None, ctx, foreign)?;
    if line(&map, "lease", 64)?.is_empty() || !LEASE_ACTIONS.contains(&text(&map, "action", 8, true)?) {
        return reject("a lease entry names a lease and what happened to it");
    }
    match map.get("resources") {
        Some(Value::Array(r)) if !r.is_empty() && r.len() <= 8 && r.iter().all(|v| v.as_str().is_some_and(|s| !s.is_empty() && s.len() <= 400 && is_line(s))) => {}
        _ => return reject("a lease entry lists one to eight resources"),
    }
    Ok((who, map))
}

fn audit(row: &Row, ctx: &mut Context, foreign: bool) -> Result<(String, Map<String, Value>)> {
    let map = only(&row.body, &AUDIT_KEYS)?;
    let who = sender(row, None, ctx, foreign)?;
    for key in AUDIT_KEYS {
        line(&map, key, 300)?;
    }
    Ok((who, map))
}

fn landing(row: &Row, ctx: &mut Context, foreign: bool) -> Result<(String, Map<String, Value>)> {
    let map = only(&row.body, &LANDING_KEYS)?;
    let who = sender(row, None, ctx, foreign)?;
    crate::landing::check(&map)?;
    Ok((who, map))
}

fn entry(row: &Row, ctx: &mut Context) -> Result<Entry> {
    let foreign = row.origin != ctx.own_origin;
    let (who, fields) = match row.kind {
        Kind::Message => message(row, ctx, foreign)?,
        Kind::Note => note(row, ctx, foreign)?,
        Kind::Identity => identity(row, ctx, foreign)?,
        Kind::Verify => verify(row, ctx, foreign)?,
        Kind::Audit => audit(row, ctx, foreign)?,
        Kind::Lease => lease(row, ctx, foreign)?,
        Kind::Landing => landing(row, ctx, foreign)?,
        _ => return reject("this kind of entry is not accepted yet"),
    };
    Ok(Entry {
        id: row.id(), origin: row.origin.clone(), seq: row.seq, hlc: (row.hlc.0, row.hlc.1), kind: row.kind,
        sender: who, foreign, channel: channel_of(row.kind, &fields), fields,
    })
}

/// Fold merged rows into entries; rows that cannot be folded come back with the reason.
pub fn fold(rows: &[Row], mut ctx: Context) -> (Vec<Entry>, Vec<(String, String)>) {
    let (mut entries, mut rejected) = (Vec::new(), Vec::new());
    let mut notes: BTreeMap<String, usize> = BTreeMap::new();
    for row in rows {
        match entry(row, &mut ctx) {
            Ok(e) if e.kind == Kind::Note && e.foreign && {
                let n = notes.entry(e.sender.clone()).or_default();
                *n += 1;
                *n > NOTES_PER_SENDER
            } => rejected.push((row.id(), "the sender has written as many notes as it may".into())),
            Ok(e) => entries.push(e),
            Err(why) => rejected.push((row.id(), why.to_string())),
        }
    }
    (entries, rejected)
}

/// Check the body of a row about to be written locally, with the same rules a peer applies.
pub fn check_local(kind: Kind, actor: &str, body: &Value, own_origin: &str) -> Result<()> {
    let row = Row {
        v: 1, board: String::new(), origin: own_origin.into(), seq: 0, prev: String::new(), hlc: (0, 0, own_origin.into()),
        kind, actor: actor.into(), idem: String::new(), body: body.clone(), hash: String::new(),
    };
    let mut label = |_: &str| "d0".to_string();
    let none = |_: &str| false;
    entry(&row, &mut Context { own_origin, label: &mut label, is_local: &none }).map(|_| ())
}
