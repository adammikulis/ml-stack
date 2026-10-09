//! The socket methods of the lease service: `lease_acquire`, `lease_renew`, `lease_release`,
//! `lease_list` and `lease_wait`. Who holds a lease is the name behind the token, never a
//! request field; a lease can be renewed or released only by its holder.

use std::time::{Duration, Instant};

use serde_json::{json, Map, Value};

use super::merge;
use super::table::{Event, Foreign, Table};
use super::types::{Class, Lease, Request, Resource, State, LOCAL};
use super::Leases;
use crate::error::{Error, Result};
use crate::fsutil::random_hex;
use crate::node::Node;
use crate::row::Kind;

pub const METHODS: [&str; 5] = ["lease_acquire", "lease_renew", "lease_release", "lease_list", "lease_wait"];
pub const MAX_WAIT_MS: u64 = 600_000;
const POLL: Duration = Duration::from_millis(25);

/// The params each lease method takes.
pub fn params_for(method: &str) -> Option<&'static [&'static str]> {
    Some(match method {
        "lease_acquire" => &["resources", "class", "estimate_s", "ttl_s", "pid", "remote", "wait"],
        "lease_renew" => &["id", "ttl_s"],
        "lease_release" => &["id"],
        "lease_list" => &[],
        "lease_wait" => &["id", "timeout_ms"],
        _ => return None,
    })
}

fn num(p: &Map<String, Value>, key: &str) -> Result<Option<u64>> {
    match p.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(v) => v.as_u64().map(Some).ok_or_else(|| Error::Invalid(format!("{key} is a whole number"))),
    }
}

fn text<'a>(p: &'a Map<String, Value>, key: &str) -> Result<&'a str> {
    p.get(key).and_then(Value::as_str).ok_or_else(|| Error::Invalid(format!("{key} is text")))
}

/// The pool-wide claims other devices hold on the boards this table has dealings with.
fn foreign(node: &mut Node, also: &[&str]) -> Result<Foreign> {
    let mut boards: Vec<String> = also.iter().map(|b| b.to_string()).collect();
    boards.extend(node.leases.table.leases.iter().filter(|l| l.resources.iter().any(Resource::pool_wide)).map(|l| l.board.clone()));
    boards.sort();
    boards.dedup();
    let mut out = Foreign::new();
    for b in boards {
        if node.boards.contains_key(&b) {
            let entries = node.view(&b)?;
            out.extend(merge::foreign_holds(&b, &entries));
        }
    }
    Ok(out)
}

fn lines(l: &Lease) -> Vec<String> {
    l.resources.iter().map(Resource::line).collect()
}

/// Write what happened to the boards of the holders it concerns.
fn mirror(node: &mut Node, events: &[Event]) -> Result<()> {
    for event in events {
        let (l, kind, body) = match event {
            Event::Granted(l) => (l, Kind::Lease, json!({"lease": l.id, "action": "acquire", "resources": lines(l)})),
            Event::Ended(l, why) if l.state == State::Held => (l, Kind::Lease, json!({"lease": l.id, "action": why.word(), "resources": lines(l)})),
            Event::Ended(..) => continue,
            Event::Takeover { from, why, to } => {
                let detail = format!("took the resources of {} ({} {}) for {}", from.holder, from.id, why.word(), to.id);
                (to, Kind::Audit, json!({"event": "lease_takeover", "subject": to.id, "detail": detail}))
            }
        };
        node.host(&l.board)?.board.append(kind, &l.name, body, "")?;
    }
    Ok(())
}

/// Sweep, grant what is now free, tell the boards.
fn tick(node: &mut Node, also: &[&str]) -> Result<()> {
    let foreign = foreign(node, also)?;
    let events = node.leases.tick(&foreign)?;
    mirror(node, &events)
}

fn show(t: &Table, node: &Node, l: &Lease, viewer: &str) -> Value {
    let (position, wait_estimate_s) = t.standing(&node.leases.cfg, node.leases.now(), &l.id);
    let own = l.board == viewer;
    json!({"id": l.id, "state": l.state, "holder": if own { l.holder.as_str() } else { "other-board" },
           "resources": if own || l.state == State::Held { json!(l.resources) } else { json!([]) },
           "class": l.class, "estimate_s": l.estimate_s, "since_ms": l.since_ms, "granted_ms": l.granted_ms,
           "expires_ms": l.expires_ms, "ttl_s": l.ttl_s, "pid": if own { l.pid } else { 0 }, "remote": l.remote,
           "position": position, "wait_estimate_s": wait_estimate_s})
}

fn authority(r: &Resource) -> Result<()> {
    if r.device() != LOCAL {
        return Err(Error::Denied(format!("device {} decides its own leases; this node is authoritative for `{LOCAL}`", r.device())));
    }
    Ok(())
}

fn acquire(node: &mut Node, board: &str, who: &str, p: &Map<String, Value>) -> Result<Value> {
    let resources: Vec<Resource> = serde_json::from_value(p.get("resources").cloned().unwrap_or(Value::Null))
        .map_err(|e| Error::Invalid(format!("resources is a list of typed resources: {e}")))?;
    resources.iter().try_for_each(authority)?;
    let class = match p.get("class").and_then(Value::as_str) {
        None | Some("interactive") => Class::Interactive,
        Some("background") => Class::Background,
        _ => return Err(Error::Invalid("class is interactive or background".into())),
    };
    let req = Request {
        board: board.into(), name: who.into(), resources, class, estimate_s: num(p, "estimate_s")?, ttl_s: num(p, "ttl_s")?,
        pid: num(p, "pid")?.map_or(Ok(0), |n| u32::try_from(n).map_err(|_| Error::Invalid("pid is a process id".into())))?,
        remote: p.get("remote").and_then(Value::as_bool).unwrap_or(false), wait: p.get("wait").and_then(Value::as_bool).unwrap_or(true),
    };
    if req.pid != 0 && req.remote {
        return Err(Error::Invalid("a remote holder has no local pid; its lease lives by expiry".into()));
    }
    tick(node, &[board])?;
    let (now, id) = (node.leases.now(), format!("l{}", random_hex(8)?));
    let id = node.leases.table.enqueue(&node.leases.cfg, now, &req, id)?;
    tick(node, &[board])?;
    node.leases.save()?;
    let table = &node.leases.table;
    let lease = table.find(&id).cloned().ok_or_else(|| Error::Invalid("the lease ended at once".into()))?;
    if lease.state == State::Queued && !req.wait {
        let blockers: Vec<Value> = table.leases.iter().filter(|o| o.state == State::Held && o.holder != lease.holder
            && o.resources.iter().any(|x| lease.resources.iter().any(|r| r.conflicts(x) || (!r.exclusive() && r.key() == x.key()))))
            .map(|o| json!({"holder": if o.board == board { o.holder.as_str() } else { "other-board" }, "expires_ms": o.expires_ms})).collect();
        node.leases.table.release(&id, &lease.holder)?;
        node.leases.save()?;
        return Ok(json!({"state": "busy", "blockers": blockers}));
    }
    Ok(show(&node.leases.table, node, &lease, board))
}

/// Handle one lease request. ``wait_ms`` of `lease_wait` is honoured by `serve`, not here:
/// this answers with the lease's state now.
pub fn call(node: &mut Node, method: &str, board: &str, token: &str, p: &Map<String, Value>) -> Result<Value> {
    let access = node.access(token, board, true)?;
    if access.holder.board != board {
        return Err(Error::Denied("leases belong to the sessions of their own board".into()));
    }
    let who = access.holder.name.clone();
    let holder = format!("{board}/{who}");
    match method {
        "lease_acquire" => acquire(node, board, &who, p),
        "lease_renew" | "lease_wait" => {
            tick(node, &[board])?;
            let (now, id) = (node.leases.now(), text(p, "id")?.to_string());
            let ttl = if method == "lease_renew" { num(p, "ttl_s")?.unwrap_or(0) } else { 0 };
            let Leases { cfg, table, .. } = &mut node.leases;
            table.renew(cfg, now, &id, &holder, ttl)?;
            if method == "lease_renew" {
                node.leases.save()?;
            }
            let lease = node.leases.table.find(&id).cloned().ok_or_else(|| Error::Invalid("no such lease".into()))?;
            Ok(show(&node.leases.table, node, &lease, board))
        }
        "lease_release" => {
            tick(node, &[board])?;
            let ended = node.leases.table.release(text(p, "id")?, &holder)?;
            node.leases.save()?;
            mirror(node, &[Event::Ended(ended.clone(), super::table::Why::Released)])?;
            tick(node, &[board])?;
            Ok(json!({"id": ended.id, "released": true}))
        }
        "lease_list" => {
            tick(node, &[board])?;
            let t = &node.leases.table;
            let all: Vec<Value> = t.leases.iter().map(|l| show(t, node, l, board)).collect();
            Ok(json!({"leases": all, "config": node.leases.cfg}))
        }
        _ => Err(Error::Invalid("unknown method".into())),
    }
}

/// Answer a request, letting `lease_wait` block up to its `timeout_ms` for a grant without
/// holding the node's lock while it sleeps.
pub fn serve(run: &dyn Fn(&Value) -> Result<Value>, request: &Value) -> Result<Value> {
    let waits = request.get("method").and_then(Value::as_str) == Some("lease_wait");
    let timeout = request.pointer("/params/timeout_ms").and_then(Value::as_u64).unwrap_or(0).min(MAX_WAIT_MS);
    let (start, mut reply) = (Instant::now(), run(request)?);
    while waits && reply["ok"] == true && reply["result"]["state"] == "queued" && start.elapsed() < Duration::from_millis(timeout) {
        std::thread::sleep(POLL);
        reply = run(request)?;
    }
    Ok(reply)
}
