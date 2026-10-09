//! The local API: one request value in, one response value out. No socket here.
//!
//! A request is `{v, id, method, board, token, params}`. Who is asking comes from the token
//! alone; a request that carries a sender, name, parent or label of its own is refused.

use std::collections::BTreeMap;
use std::sync::atomic::Ordering;

use serde_json::{json, Map, Value};

use crate::error::{Error, Result};
use crate::fold::Entry;
use crate::node::{Node, READ_DEFAULT, READ_MAX};
use crate::row::Kind;

pub const API_VERSION: u64 = 1;
const ENVELOPE: [&str; 6] = ["v", "id", "method", "board", "token", "params"];
const FORBIDDEN: [&str; 9] = ["sender", "name", "parent", "label", "from", "author", "actor", "identity", "as"];

fn params_for(method: &str) -> Option<&'static [&'static str]> {
    Some(match method {
        "hello" | "whoami" | "status" | "shutdown" | "links" => &[],
        "register" => &["model", "harness", "session"],
        "post" => &["kind", "idem", "fields"],
        "read" => &["since", "kind", "channel", "sender", "limit"],
        "claim" | "release" => &["target", "note"],
        "link" => &["to", "channels", "mode"],
        "unlink" => &["id"],
        "project_add" | "source_add" => &["id", "kind", "path"],
        "project_list" => &[],
        "project_resolve" => &["path"],
        m => return crate::lease::rpc::params_for(m),
    })
}

fn refuse_identity(map: &Map<String, Value>) -> Result<()> {
    match map.keys().find(|k| FORBIDDEN.contains(&k.to_lowercase().as_str())) {
        Some(k) => Err(Error::Denied(format!("`{k}` is not a field: the node knows who you are from your token"))),
        None => Ok(()),
    }
}

fn str_of<'a>(p: &'a Map<String, Value>, key: &str) -> Result<&'a str> {
    match p.get(key) {
        Some(Value::String(s)) => Ok(s),
        None => Ok(""),
        _ => Err(Error::Invalid(format!("{key} is text"))),
    }
}

fn entry_json(e: &Entry) -> Value {
    json!({"id": e.id, "origin": e.origin, "seq": e.seq, "hlc": [e.hlc.0, e.hlc.1], "kind": e.kind, "sender": e.sender,
           "foreign": e.foreign, "channel": e.channel, "fields": e.fields})
}

fn read(node: &mut Node, token: &str, board: &str, p: &Map<String, Value>) -> Result<Value> {
    let access = node.access(token, board, false)?;
    let mut cursor: BTreeMap<String, u64> = match p.get("since") {
        None => BTreeMap::new(),
        Some(v) => serde_json::from_value(v.clone()).map_err(|_| Error::Invalid("since maps origins to sequence numbers".into()))?,
    };
    let limit = p.get("limit").and_then(Value::as_u64).map_or(READ_DEFAULT, |n| (n as usize).clamp(1, READ_MAX));
    let (kind, channel, sender) = (str_of(p, "kind")?, str_of(p, "channel")?, str_of(p, "sender")?);
    let mut out = Vec::new();
    for e in node.view(board)? {
        let kind_name = serde_json::to_value(e.kind)?;
        let wanted = e.seq > cursor.get(&e.origin).copied().unwrap_or(0)
            && access.channels.as_ref().is_none_or(|c| c.contains(&e.channel))
            && (kind.is_empty() || kind_name == kind) && (channel.is_empty() || e.channel == channel)
            && (sender.is_empty() || e.sender == sender);
        if wanted && out.len() < limit {
            cursor.insert(e.origin.clone(), e.seq);
            out.push(entry_json(&e));
        }
    }
    Ok(json!({"entries": out, "cursor": cursor}))
}

fn post(node: &mut Node, token: &str, board: &str, p: &Map<String, Value>) -> Result<Value> {
    let kind = match str_of(p, "kind")? {
        "message" => Kind::Message,
        "note" => Kind::Note,
        _ => return Err(Error::Invalid("post writes a message or a note".into())),
    };
    let fields = p.get("fields").and_then(Value::as_object).ok_or_else(|| Error::Invalid("fields is an object".into()))?;
    refuse_identity(fields)?;
    let fields = if kind == Kind::Message && !fields.contains_key("to") {
        let mut f = fields.clone();
        f.insert("to".into(), json!(crate::fold::GENERAL));
        f
    } else {
        fields.clone()
    };
    node.post(token, board, kind, &fields, str_of(p, "idem")?)
}

fn link(node: &mut Node, token: &str, board: &str, p: &Map<String, Value>, make: bool) -> Result<Value> {
    let access = node.access(token, board, true)?;
    if access.holder.board != board {
        return Err(Error::Denied("only a session of the sharing board makes or revokes its links".into()));
    }
    let by = format!("{}/{}", board, access.actor);
    let done = if make {
        let channels: Vec<String> = serde_json::from_value(p.get("channels").cloned().unwrap_or(json!([])))
            .map_err(|_| Error::Invalid("channels is a list of #names".into()))?;
        let write = match str_of(p, "mode")? { "ro" | "" => false, "rw" => true, _ => return Err(Error::Invalid("mode is ro or rw".into())) };
        let to = str_of(p, "to")?;
        node.host(to)?;
        node.links.add(board, to, channels, write, &by)?
    } else {
        node.links.revoke(board, str_of(p, "id")?)?
    };
    let detail = format!("{}>{} {} {}", done.from, done.to, if done.write { "rw" } else { "ro" }, done.channels.join(","));
    let event = if make { "link" } else { "unlink" };
    for side in [&done.from, &done.to] {
        let actor = if side == board { access.actor.clone() } else { format!("{}@{board}", access.actor) };
        node.host(side)?.board.append(Kind::Audit, &actor, json!({"event": event, "subject": done.id, "detail": detail}), "")?;
    }
    Ok(json!(done))
}

fn project(node: &mut Node, method: &str, token: &str, p: &Map<String, Value>) -> Result<Value> {
    let (id, kind, path) = (str_of(p, "id")?, str_of(p, "kind")?, str_of(p, "path")?);
    let done = match method {
        "project_add" => node.projects.add(id, kind, path)?,
        _ => {
            // adding a place to an existing project takes a session of that project
            let holder = node.tokens.resolve(token).cloned();
            if holder.is_none_or(|h| h.board != id) {
                return Err(Error::Denied("only a session of the project adds a place to it".into()));
            }
            node.projects.add_source(id, kind, path)?
        }
    };
    node.host(id)?;
    Ok(json!(done))
}

fn status(node: &mut Node, token: &str, board: &str) -> Result<Value> {
    let mut out = json!({"v": API_VERSION, "pid": std::process::id(), "version": env!("CARGO_PKG_VERSION"),
                         "boards": node.boards.len(), "uptime_s": node.started.elapsed().as_secs()});
    if !token.is_empty() {
        let access = node.access(token, board, false)?;
        let h = node.host(board)?;
        out["board"] = json!({"id": board, "origin": h.board.origin(), "sessions": h.names.len(), "damaged": h.board.damaged(),
                              "rejected": h.board.rejected().len(), "name": access.actor});
    }
    Ok(out)
}

fn dispatch(node: &mut Node, method: &str, board: &str, token: &str, p: &Map<String, Value>) -> Result<Value> {
    match method {
        "hello" => Ok(json!({"v": API_VERSION, "node": "poolside-node", "version": env!("CARGO_PKG_VERSION"),
                             "pid": std::process::id(), "fingerprint": node.fingerprint()})),
        "register" => node.register(board, (!token.is_empty()).then_some(token), str_of(p, "model")?, str_of(p, "harness")?, str_of(p, "session")?),
        "status" => status(node, token, board),
        "whoami" => node.access(token, board, false).map(|a| json!({"name": a.actor, "board": a.holder.board, "target": board})),
        "post" => post(node, token, board, p),
        "read" => read(node, token, board, p),
        "claim" | "release" => node.claim(token, board, str_of(p, "target")?, str_of(p, "note")?, method == "release"),
        "link" | "unlink" => link(node, token, board, p, method == "link"),
        "project_add" | "source_add" => project(node, method, token, p),
        "project_list" => Ok(json!(node.projects.all())),
        "project_resolve" => node.projects.resolve(str_of(p, "path")?).map(|p| json!({"board": p.id})),
        m if crate::lease::rpc::METHODS.contains(&m) => crate::lease::rpc::call(node, m, board, token, p),
        "links" => node.access(token, board, false).map(|_| json!(node.links.all().iter().filter(|l| l.from == board || l.to == board).collect::<Vec<_>>())),
        "shutdown" => node.access(token, board, false).map(|_| {
            node.stop.store(true, Ordering::SeqCst);
            json!({"stopping": true})
        }),
        _ => Err(Error::Invalid("unknown method".into())),
    }
}

/// Answer one request.
pub fn handle(node: &mut Node, request: &Value) -> Value {
    let id = request.get("id").cloned().unwrap_or(Value::Null);
    match run(node, request) {
        Ok(result) => json!({"v": API_VERSION, "id": id, "ok": true, "result": result}),
        Err(e) => json!({"v": API_VERSION, "id": id, "ok": false, "error": {"code": e.code(), "message": e.to_string()}}),
    }
}

fn run(node: &mut Node, request: &Value) -> Result<Value> {
    let map = request.as_object().ok_or_else(|| Error::Invalid("a request is an object".into()))?;
    refuse_identity(&map.keys().map(|k| (k.clone(), Value::Null)).collect())?;
    if let Some(k) = map.keys().find(|k| !ENVELOPE.contains(&k.as_str())) {
        return Err(Error::Invalid(format!("unknown request field {}", k.chars().take(40).collect::<String>())));
    }
    if map.get("v").and_then(Value::as_u64) != Some(API_VERSION) {
        return Err(Error::Invalid(format!("this node speaks version {API_VERSION}")));
    }
    let method = map.get("method").and_then(Value::as_str).unwrap_or("");
    let allowed = params_for(method).ok_or_else(|| Error::Invalid("unknown method".into()))?;
    let params = map.get("params").map_or(Ok(Map::new()), |p| p.as_object().cloned().ok_or_else(|| Error::Invalid("params is an object".into())))?;
    refuse_identity(&params)?;
    if let Some(k) = params.keys().find(|k| !allowed.contains(&k.as_str())) {
        return Err(Error::Invalid(format!("unknown param {}", k.chars().take(40).collect::<String>())));
    }
    let (board, token) = (str_of(map, "board")?, str_of(map, "token")?);
    if board.is_empty() && !matches!(method, "hello" | "status" | "project_add" | "source_add" | "project_list" | "project_resolve") {
        return Err(Error::Invalid("this method names a board".into()));
    }
    dispatch(node, method, board, token, &params)
}
