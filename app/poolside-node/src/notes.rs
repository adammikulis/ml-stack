//! Notes read back: each note with who superseded it, its latest verification, how far it is
//! trusted and whether it has gone stale. The node never runs a note's command; a client does
//! and records the result with `note_verify`, which stores the command the note carries (not
//! one the caller names) beside the exit code and the hash of the output.

use std::collections::BTreeMap;

use serde_json::{json, Map, Value};

use crate::error::{Error, Result};
use crate::fold::Entry;
use crate::node::Node;
use crate::row::Kind;

pub const METHODS: [&str; 2] = ["notes", "note_verify"];
const DAY_MS: u64 = 86_400_000;
const SEARCH_DEFAULT: usize = 10;

pub fn params_for(method: &str) -> Option<&'static [&'static str]> {
    Some(match method {
        "notes" => &["ref", "query", "kind", "all", "limit"],
        "note_verify" => &["note", "exit", "out_sha"],
        _ => return None,
    })
}

struct Proof {
    at: u64,
    exit: i64,
    by: String,
    out_sha: String,
}

struct Note {
    entry: Entry,
    superseded_by: String,
    proof: Option<Proof>,
}

fn str_field<'a>(e: &'a Entry, key: &str) -> &'a str {
    e.fields.get(key).and_then(Value::as_str).unwrap_or("")
}

fn fold_notes(entries: &[Entry]) -> BTreeMap<String, Note> {
    let mut notes: BTreeMap<String, Note> = BTreeMap::new();
    for e in entries {
        match e.kind {
            Kind::Note => {
                for old in e.fields.get("supersedes").and_then(Value::as_array).into_iter().flatten().filter_map(Value::as_str) {
                    if let Some(n) = notes.get_mut(old).filter(|n| n.superseded_by.is_empty()) {
                        n.superseded_by = e.id.clone();
                    }
                }
                notes.insert(e.id.clone(), Note { entry: e.clone(), superseded_by: String::new(), proof: None });
            }
            Kind::Verify => {
                let target = str_field(e, "note");
                if let Some(n) = notes.get_mut(target).filter(|n| str_field(&n.entry, "verify_cmd") == str_field(e, "cmd")) {
                    let exit = e.fields.get("exit").and_then(Value::as_i64).unwrap_or(-1);
                    n.proof = Some(Proof { at: e.hlc.0, exit, by: e.sender.clone(), out_sha: str_field(e, "out_sha").into() });
                }
            }
            _ => {}
        }
    }
    notes
}

fn rotted(ttl_days: u64, since: u64, now: u64) -> bool {
    ttl_days > 0 && now.saturating_sub(since) > ttl_days * DAY_MS
}

/// `test-verified` while the latest passing verification is fresh, else `agent-claimed`.
fn trust(n: &Note, now: u64) -> &'static str {
    let ttl = n.entry.fields.get("ttl_days").and_then(Value::as_u64).unwrap_or(0);
    match &n.proof {
        Some(p) if p.exit == 0 && !rotted(ttl, p.at, now) => "test-verified",
        _ => "agent-claimed",
    }
}

fn show(n: &Note, own_origin: &str, now: u64) -> Value {
    let e = &n.entry;
    let ttl = e.fields.get("ttl_days").and_then(Value::as_u64).unwrap_or(0);
    let since = n.proof.as_ref().map_or(e.hlc.0, |p| p.at);
    let short = if e.origin == own_origin { e.seq.to_string() } else { e.id.clone() };
    json!({"id": e.id, "ref": short, "kind": str_field(e, "nkind"), "title": str_field(e, "title"), "body": str_field(e, "body"),
           "author": e.sender, "ts_ms": e.hlc.0, "source": str_field(e, "source"), "tags": e.fields.get("tags").cloned().unwrap_or(json!([])),
           "trust": trust(n, now), "stale": rotted(ttl, since, now), "ttl_days": ttl,
           "supersedes": e.fields.get("supersedes").cloned().unwrap_or(json!([])), "superseded_by": n.superseded_by,
           "verify_cmd": str_field(e, "verify_cmd"), "binding": false,
           "last_verified": n.proof.as_ref().map(|p| json!({"at_ms": p.at, "exit": p.exit, "by": p.by, "output_sha256": p.out_sha}))})
}

/// The id a reference names: a bare number is a note of this device, anything else an id.
fn resolve(board: &str, own_origin: &str, reference: &str) -> String {
    if !reference.is_empty() && reference.bytes().all(|b| b.is_ascii_digit()) {
        format!("{board}:{own_origin}:{reference}")
    } else {
        reference.to_string()
    }
}

/// How well a note answers the words: the notes holding every word first, a hit in the title counting thrice.
fn score(n: &Note, words: &[String]) -> (bool, usize) {
    let e = &n.entry;
    let tags = e.fields.get("tags").and_then(Value::as_array).map(|t| t.iter().filter_map(Value::as_str).collect::<Vec<_>>().join(" ")).unwrap_or_default();
    let (title, body, rest) = (str_field(e, "title").to_lowercase(), str_field(e, "body").to_lowercase(), format!("{tags} {}", str_field(e, "source")).to_lowercase());
    let hits: Vec<usize> = words.iter().map(|w| 3 * title.matches(w.as_str()).count() + body.matches(w.as_str()).count() + rest.matches(w.as_str()).count()).collect();
    (hits.iter().all(|h| *h > 0), hits.iter().sum())
}

fn notes_of(node: &mut Node, board: &str) -> Result<(BTreeMap<String, Note>, String, u64)> {
    let entries = node.view(board)?;
    let host = node.host(board)?;
    Ok((fold_notes(&entries), host.board.origin().to_string(), host.board.now_ms()))
}

fn list(node: &mut Node, board: &str, p: &Map<String, Value>) -> Result<Value> {
    let (notes, origin, now) = notes_of(node, board)?;
    let text = |k: &str| p.get(k).and_then(Value::as_str).unwrap_or("");
    let (kind, all) = (text("kind"), p.get("all").and_then(Value::as_bool).unwrap_or(false));
    let limit = p.get("limit").and_then(Value::as_u64).map_or(SEARCH_DEFAULT, |n| (n as usize).clamp(1, 200));
    if !text("ref").is_empty() {
        let found = notes.get(&resolve(board, &origin, text("ref"))).ok_or_else(|| Error::Invalid("no such note".into()))?;
        return Ok(json!({"notes": [show(found, &origin, now)]}));
    }
    let words: Vec<String> = text("query").split_whitespace().map(str::to_lowercase).collect();
    let mut ranked: Vec<(&Note, (bool, usize))> = notes.values()
        .filter(|n| (all || n.superseded_by.is_empty()) && (kind.is_empty() || str_field(&n.entry, "nkind") == kind))
        .map(|n| (n, score(n, &words))).filter(|(_, s)| words.is_empty() || s.1 > 0).collect();
    if words.is_empty() {
        ranked.reverse();
    } else {
        ranked.sort_by(|a, b| b.1.cmp(&a.1));
    }
    Ok(json!({"notes": ranked.iter().take(limit).map(|(n, _)| show(n, &origin, now)).collect::<Vec<_>>()}))
}

/// Turn the references in a note's `supersedes` into ids, refusing a note that does not exist or
/// has been verified: only a better-checked note replaces a checked one.
pub fn resolve_supersedes(node: &mut Node, board: &str, fields: &mut Map<String, Value>) -> Result<()> {
    let Some(Value::Array(refs)) = fields.get("supersedes").cloned() else { return Ok(()) };
    let (notes, origin, now) = notes_of(node, board)?;
    let mut ids = Vec::new();
    for r in refs {
        let id = resolve(board, &origin, r.as_str().unwrap_or(""));
        let target = notes.get(&id).ok_or_else(|| Error::Invalid(format!("no note {} to supersede", r.as_str().unwrap_or(""))))?;
        if trust(target, now) != "agent-claimed" {
            return Err(Error::Denied(format!("note {} is {}; a claimed note cannot replace it, add a question instead", r.as_str().unwrap_or(""), trust(target, now))));
        }
        ids.push(json!(id));
    }
    fields.insert("supersedes".into(), Value::Array(ids));
    Ok(())
}

fn verify(node: &mut Node, board: &str, token: &str, p: &Map<String, Value>) -> Result<Value> {
    let access = node.access(token, board, true)?;
    if !node.grants.allows(&access.holder, "note_verify") {
        return Err(Error::Denied("this session may not record a note verification".into()));
    }
    let (notes, origin, now) = notes_of(node, board)?;
    let id = resolve(board, &origin, p.get("note").and_then(Value::as_str).unwrap_or(""));
    let note = notes.get(&id).ok_or_else(|| Error::Invalid("no such note".into()))?;
    let cmd = str_field(&note.entry, "verify_cmd").to_string();
    if cmd.is_empty() {
        return Err(Error::Invalid("this note has no re-derive command".into()));
    }
    let exit = p.get("exit").and_then(Value::as_i64).ok_or_else(|| Error::Invalid("exit is the command's exit code".into()))?;
    let body = json!({"note": id, "exit": exit, "cmd": cmd, "out_sha": p.get("out_sha").and_then(Value::as_str).unwrap_or("")});
    crate::fold::check_local(Kind::Verify, &access.actor, &body, "")?;
    node.host(board)?.board.append(Kind::Verify, &access.actor, body, "")?;
    let (notes, origin, _) = notes_of(node, board)?;
    Ok(json!({"notes": [show(&notes[&id], &origin, now)]}))
}

pub fn call(node: &mut Node, method: &str, board: &str, token: &str, p: &Map<String, Value>) -> Result<Value> {
    if method == "note_verify" {
        return verify(node, board, token, p);
    }
    let access = node.access(token, board, false)?;
    if access.channels.as_ref().is_some_and(|c| !c.iter().any(|x| x == "#notes")) {
        return Err(Error::Denied("notes are not shared with this session".into()));
    }
    list(node, board, p)
}
