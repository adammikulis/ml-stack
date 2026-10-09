//! What a session can ask about sessions: `whoami` (and say what model it runs), find the name
//! of a native session, list the sessions of its board, and retire one.

use serde_json::{json, Map, Value};

use crate::error::{Error, Result};
use crate::identity::Ident;
use crate::lease::table::{Event, Why};
use crate::node::Node;
use crate::row::Kind;

pub const METHODS: [&str; 4] = ["whoami", "session_lookup", "agents", "retire"];

pub fn params_for(method: &str) -> Option<&'static [&'static str]> {
    Some(match method {
        "whoami" => &["model", "harness"],
        "session_lookup" => &["harness", "session"],
        "agents" => &["retired"],
        "retire" => &["target"],
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

fn show(name: &str, i: &Ident) -> Value {
    json!({"name": name, "parent": i.parent, "family": i.family, "model": i.model, "model_state": i.model_state,
           "harness": i.harness, "retired": i.retired, "seen_ms": i.seen_ms})
}

pub fn call(node: &mut Node, method: &str, board: &str, token: &str, p: &Map<String, Value>) -> Result<Value> {
    match method {
        "session_lookup" => lookup(node, board, p),
        "whoami" => whoami(node, board, token, p),
        "agents" => agents(node, board, token, p),
        _ => retire(node, board, token, p),
    }
}

fn lookup(node: &mut Node, board: &str, p: &Map<String, Value>) -> Result<Value> {
    let hosted = node.boards.get(board).ok_or_else(|| Error::Denied("this node does not host that board".into()))?;
    match hosted.names.find(text(p, "harness")?, text(p, "session")?) {
        Some(name) => Ok(json!({"name": name})),
        None => Err(Error::Denied("no session of that harness and id is registered on this board".into())),
    }
}

fn whoami(node: &mut Node, board: &str, token: &str, p: &Map<String, Value>) -> Result<Value> {
    let access = node.access(token, board, false)?;
    let (model, harness) = (text(p, "model")?, text(p, "harness")?);
    if !(model.is_empty() && harness.is_empty()) {
        if access.holder.board != board {
            return Err(Error::Denied("a model is claimed on the session's own board".into()));
        }
        if node.host(board)?.names.claim_model(&access.actor, model, harness)? {
            node.write_identity(board, &access.actor)?;
        }
    }
    let mut out = json!({"name": access.actor, "board": access.holder.board, "target": board});
    if let Some(i) = node.host(board)?.names.get(&access.actor) {
        out["identity"] = show(&access.actor, i);
    }
    Ok(out)
}

fn agents(node: &mut Node, board: &str, token: &str, p: &Map<String, Value>) -> Result<Value> {
    let access = node.access(token, board, false)?;
    if access.channels.as_ref().is_some_and(|c| !c.iter().any(|x| x == "#identity")) {
        return Err(Error::Denied("identities are not shared with this session".into()));
    }
    let all = p.get("retired").and_then(Value::as_bool).unwrap_or(false);
    let list: Vec<Value> = node.host(board)?.names.all().iter().filter(|(_, i)| all || !i.retired).map(|(n, i)| show(n, i)).collect();
    Ok(json!({"agents": list}))
}

fn retire(node: &mut Node, board: &str, token: &str, p: &Map<String, Value>) -> Result<Value> {
    let access = node.access(token, board, true)?;
    if access.holder.board != board {
        return Err(Error::Denied("only a session of the board retires its sessions".into()));
    }
    let (caller, asked) = (access.holder.name.clone(), text(p, "target")?);
    let target = if asked.is_empty() { caller.clone() } else { asked.to_string() };
    let names = &node.host(board)?.names;
    let ident = names.get(&target).ok_or_else(|| Error::Denied("no such session".into()))?;
    if ident.retired {
        return Err(Error::Denied(format!("{target} is already retired")));
    }
    if target == caller {
        if ident.parent.is_empty() {
            return Err(Error::Denied("only a spawned subagent retires itself; a main session ends with its harness".into()));
        }
    } else if !names.descendants(&caller).contains(&target) {
        return Err(Error::Denied(format!("{target} is not below {caller}; only a parent retires its subagent")));
    }
    let freed = release_leases(node, board, &target)?;
    let revoked = node.tokens.revoke_name(board, &target)?;
    node.host(board)?.names.retire(&target)?;
    node.write_identity(board, &target)?;
    let detail = format!("retired by {caller}: {revoked} tokens revoked, {freed} leases released");
    node.host(board)?.board.append(Kind::Audit, &caller, json!({"event": "retire", "subject": target, "detail": detail}), "")?;
    Ok(json!({"name": target, "retired": true, "by": caller, "tokens_revoked": revoked, "leases_released": freed}))
}

/// End every lease of ``name``, held or queued, and tell the board; how many there were.
fn release_leases(node: &mut Node, board: &str, name: &str) -> Result<usize> {
    let holder = format!("{board}/{name}");
    let ids: Vec<String> = node.leases.table.leases.iter().filter(|l| l.holder == holder).map(|l| l.id.clone()).collect();
    let mut events = Vec::new();
    for id in &ids {
        events.push(Event::Ended(node.leases.table.release(id, &holder)?, Why::Released));
    }
    if !ids.is_empty() {
        node.leases.save()?;
        crate::lease::rpc::mirror(node, &events)?;
        crate::lease::rpc::tick(node, &[board])?;
    }
    Ok(ids.len())
}
