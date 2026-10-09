//! Claims, in terms of leases: `claim` takes one `claim` resource without waiting, `release`
//! gives it back, `claims` lists who holds what. A claim is a lease of one resource, so it is
//! dropped when its holder's process dies or its time runs out, like every other lease.

use serde_json::{json, Map, Value};

use crate::error::{Error, Result};
use crate::lease::rpc;
use crate::lease::table::Event;
use crate::lease::table::Why;
use crate::lease::types::{Lease, Resource, State, CLAIM_KINDS};
use crate::node::Node;

pub const METHODS: [&str; 3] = ["claim", "release", "claims"];

pub fn params_for(method: &str) -> Option<&'static [&'static str]> {
    Some(match method {
        "claim" => &["kind", "key", "ttl_s", "pid"],
        "release" => &["kind", "key"],
        "claims" => &["kind"],
        _ => return None,
    })
}

fn text<'a>(p: &'a Map<String, Value>, key: &str) -> Result<&'a str> {
    match p.get(key) {
        None => Ok(""),
        Some(Value::String(s)) => Ok(s),
        _ => Err(Error::Invalid(format!("{key} is text"))),
    }
}

fn kind_of<'a>(p: &'a Map<String, Value>, required: bool) -> Result<&'a str> {
    let kind = text(p, "kind")?;
    if kind.is_empty() && !required || CLAIM_KINDS.contains(&kind) {
        Ok(kind)
    } else {
        Err(Error::Invalid("a claim kind is worktree, branch, area, port, server or install".into()))
    }
}

/// The name a claim is held under: a branch belongs to its board's repository, so two boards
/// may each claim a branch of the same name; ports, servers and paths belong to the device.
fn held_as(kind: &str, board: &str, key: &str) -> String {
    if kind == "branch" { format!("{board}/{key}") } else { key.to_string() }
}

fn shown_as<'a>(kind: &str, board: &str, name: &'a str) -> &'a str {
    if kind == "branch" { name.strip_prefix(board).and_then(|n| n.strip_prefix('/')).unwrap_or(name) } else { name }
}

fn mine(l: &Lease, kind: &str, name: &str) -> bool {
    l.resources.iter().any(|r| matches!(r, Resource::Claim { kind: k, name: n } if k == kind && n == name))
}

/// One claim of a lease as clients see it.
fn show(node: &Node, l: &Lease, kind: &str, name: &str, board: &str) -> Value {
    let key = shown_as(kind, board, name);
    let now = node.leases.now();
    let left = l.expires_ms.saturating_sub(now) / 1000;
    let own = l.board == board;
    json!({"kind": kind, "key": key, "owner": if own { l.name.as_str() } else { "other-board" }, "lease": if own { l.id.as_str() } else { "" },
           "expires_in_s": left, "expiring_soon": left < 300, "pid": if own { l.pid } else { 0 }, "since_ms": l.granted_ms})
}

fn claim(node: &mut Node, board: &str, token: &str, p: &Map<String, Value>) -> Result<Value> {
    let (kind, key) = (kind_of(p, true)?, text(p, "key")?);
    let name = held_as(kind, board, key);
    let access = node.access(token, board, true)?;
    let mut ask = Map::new();
    ask.insert("resources".into(), json!([{"type": "claim", "kind": kind, "name": name}]));
    ask.insert("wait".into(), json!(false));
    for field in ["ttl_s", "pid"] {
        if let Some(v) = p.get(field) {
            ask.insert(field.into(), v.clone());
        }
    }
    let holder = format!("{board}/{}", access.holder.name);
    rpc::tick(node, &[board])?;
    if let Some(l) = node.leases.table.leases.iter().find(|l| l.holder == holder && l.state == State::Held && mine(l, kind, &name)).cloned() {
        let ttl = p.get("ttl_s").and_then(Value::as_u64).unwrap_or(l.ttl_s);
        let mut renew = Map::new();
        renew.insert("id".into(), json!(l.id));
        renew.insert("ttl_s".into(), json!(ttl));
        rpc::call(node, "lease_renew", board, token, &renew)?;
        let l = node.leases.table.find(&l.id).cloned().unwrap_or(l);
        return Ok(json!({"changed": false, "claim": show(node, &l, kind, &name, board)}));
    }
    let got = rpc::call(node, "lease_acquire", board, token, &ask)?;
    if got["state"] == "busy" {
        let who = got["blockers"].as_array().and_then(|b| b.first()).and_then(|b| b["holder"].as_str()).unwrap_or("another session of yours");
        return Err(Error::Denied(format!("{kind} {key} is held by {who}")));
    }
    let l = node.leases.table.find(got["id"].as_str().unwrap_or("")).cloned().ok_or_else(|| Error::Invalid("the claim ended at once".into()))?;
    Ok(json!({"changed": true, "claim": show(node, &l, kind, &name, board)}))
}

fn release(node: &mut Node, board: &str, token: &str, p: &Map<String, Value>) -> Result<Value> {
    let (kind, key) = (kind_of(p, true)?, text(p, "key")?);
    let name = held_as(kind, board, key);
    let access = node.access(token, board, true)?;
    let holder = format!("{board}/{}", access.holder.name);
    rpc::tick(node, &[board])?;
    let ids: Vec<String> = node.leases.table.leases.iter().filter(|l| l.holder == holder && l.state == State::Held && mine(l, kind, &name)).map(|l| l.id.clone()).collect();
    if ids.is_empty() {
        let other = node.leases.table.leases.iter().find(|l| l.state == State::Held && l.board == board && mine(l, kind, &name)).map(|l| l.name.clone());
        return match other {
            Some(who) => Err(Error::Denied(format!("{kind} {key} is held by {who}; only the holder releases it"))),
            None => Ok(json!({"kind": kind, "key": key, "released": false})),
        };
    }
    let mut events = Vec::new();
    for id in &ids {
        events.push(Event::Ended(node.leases.table.release(id, &holder)?, Why::Released));
    }
    node.leases.save()?;
    rpc::mirror(node, &events)?;
    rpc::tick(node, &[board])?;
    Ok(json!({"kind": kind, "key": key, "released": true}))
}

fn claims(node: &mut Node, board: &str, token: &str, p: &Map<String, Value>) -> Result<Value> {
    let kind = kind_of(p, false)?;
    let access = node.access(token, board, false)?;
    if access.channels.as_ref().is_some_and(|c| !c.iter().any(|x| x == "#leases")) {
        return Err(Error::Denied("claims are not shared with this session".into()));
    }
    rpc::tick(node, &[board])?;
    let mut out = Vec::new();
    for l in node.leases.table.leases.iter().filter(|l| l.state == State::Held && l.board == board) {
        for r in &l.resources {
            if let Resource::Claim { kind: k, name } = r {
                if kind.is_empty() || kind == k {
                    out.push(show(node, l, k, name, board));
                }
            }
        }
    }
    Ok(json!({"claims": out}))
}

pub fn call(node: &mut Node, method: &str, board: &str, token: &str, p: &Map<String, Value>) -> Result<Value> {
    match method {
        "claim" => claim(node, board, token, p),
        "release" => release(node, board, token, p),
        _ => claims(node, board, token, p),
    }
}
