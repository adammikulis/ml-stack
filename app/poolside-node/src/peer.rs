//! The peer protocol over a TLS 1.3 connection: length-prefixed JSON requests, one reply each,
//! any number on one connection.
//!
//! Who is asking is the certificate shown in the handshake and nothing in the request. Every
//! request is checked against the pool record when it arrives, not when the connection was made,
//! so a device put out is refused at its next request on a connection it already holds.
//!
//! | op | who | what |
//! |---|---|---|
//! | `hello` | anyone not revoked | pool id, join policy, fingerprints; the boards held, to members |
//! | `join_open` | anyone, under policy `open` | enrol the caller; reply with the pool record |
//! | `pair_exchange`, `pair_confirm` | anyone, while pairing is open | SPAKE2 (`pairing`) |
//! | `members` | members | swap pool records |
//! | `boards`, `vector`, `pull`, `push` | members | board sync (`netsync`) |

use std::collections::BTreeMap;
use std::net::{IpAddr, SocketAddr, TcpStream};
use std::time::Duration;

use serde_json::{json, Value};

use crate::board::Tip;
use crate::cert::{board_fingerprint, cert_fingerprint, Identity};
use crate::error::{Error, Result};
use crate::membership::{Policy, Standing};
use crate::net::Net;
use crate::node::Node;
use crate::row::{valid_origin, valid_name, Row};
use crate::sync::{rows_since, seqs, take, Report};
use crate::tls::{self, Secured};
use crate::wire::{read_frame_max, write_frame_max, MAX_PEER_FRAME};

pub const IO_TIMEOUT: Duration = Duration::from_secs(30);

fn text<'a>(req: &'a Value, key: &str) -> Result<&'a str> {
    req.get(key).and_then(Value::as_str).ok_or_else(|| Error::Invalid(format!("{key} is text")))
}

fn vector_of(req: &Value, key: &str) -> Result<BTreeMap<String, Tip>> {
    serde_json::from_value(req.get(key).cloned().unwrap_or(json!({}))).map_err(|_| Error::Invalid(format!("{key} maps origins to tips")))
}

fn reply_of(result: Result<Value>) -> Value {
    match result {
        Ok(v) => json!({"ok": true, "result": v}),
        Err(e) => json!({"ok": false, "error": {"code": e.code(), "message": e.to_string()}}),
    }
}

/// The policy attribute as it travels.
pub fn policy_json(node: &Node) -> Value {
    json!({"policy": node.members.policy.name(), "at": node.members.policy_at, "by": node.members.policy_by})
}

/// Take a peer's policy if it is newer (the caller is an active member).
pub fn take_policy(node: &mut Node, v: &Value) -> Result<()> {
    let (Some(p), Some(at), Some(by)) = (v.get("policy").and_then(Value::as_str), v.get("at").and_then(Value::as_u64), v.get("by").and_then(Value::as_str)) else {
        return Ok(());
    };
    if node.members.set_policy(Policy::parse(p)?, at, by)? {
        node.record_event("join_policy", p, "by a peer")?;
    }
    Ok(())
}

fn hello(node: &Node, standing: Standing) -> Value {
    let boards: Vec<&String> = if standing == Standing::Active { node.boards.keys().collect() } else { Vec::new() };
    json!({"pool": node.members.id, "policy": node.members.policy.name(), "fingerprint": node.cert.fingerprint(), "project": node.members.project,
           "board_fingerprint": crate::device::fingerprint(&node.key), "member": standing == Standing::Active, "boards": boards})
}

fn join_open(node: &mut Node, der: &[u8], req: &Value) -> Result<Value> {
    if node.members.policy != Policy::Open {
        return Err(Error::Denied("this pool takes a device only through a pairing code".into()));
    }
    let name = req.get("name").and_then(Value::as_str).unwrap_or("");
    node.enrol_device(der, name, "open")?;
    Ok(json!({"pool": node.members.id, "rows": node.members.export(), "policy": policy_json(node)}))
}

fn members(node: &mut Node, fp: &str, req: &Value) -> Result<Value> {
    let rows = req.get("rows").and_then(Value::as_array).ok_or_else(|| Error::Invalid("rows is a list".into()))?;
    if req.get("pool").and_then(Value::as_str) != Some(node.members.id.as_str()) {
        return Err(Error::Denied("this is not the same pool".into()));
    }
    let changed = node.members.merge(rows, fp)?;
    if let Some(p) = req.get("policy") {
        take_policy(node, p)?;
    }
    Ok(json!({"pool": node.members.id, "changed": changed, "rows": node.members.export(), "policy": policy_json(node)}))
}

fn hosted<'a>(node: &'a mut Node, req: &Value, bfp: &str) -> Result<&'a mut crate::node::Hosted> {
    let (id, origin) = (text(req, "board")?, text(req, "origin")?);
    if !valid_name(id) || !valid_origin(origin) {
        return Err(Error::Invalid("a board and an origin are named".into()));
    }
    let h = node.boards.get_mut(id).ok_or_else(|| Error::Denied("this device does not hold that board".into()))?;
    if !h.board.bind_peer(bfp, origin)? {
        return Err(Error::Denied("this device writes a different log than the one it presented before".into()));
    }
    Ok(h)
}

fn pull(node: &mut Node, bfp: &str, req: &Value) -> Result<Value> {
    let vector = vector_of(req, "vector")?;
    let h = hosted(node, req, bfp)?;
    h.board.acknowledge(bfp, &vector)?;
    h.board.seal()?;
    let logs = rows_since(&h.board, &seqs(&vector));
    Ok(json!({"origin": h.board.origin(), "trusted_vector": h.board.trusted_vector(), "logs": logs}))
}

fn vector(node: &mut Node, bfp: &str, req: &Value) -> Result<Value> {
    let h = hosted(node, req, bfp)?;
    Ok(json!({"origin": h.board.origin(), "vector": h.board.vector()}))
}

fn push(node: &mut Node, bfp: &str, req: &Value) -> Result<Value> {
    let logs: BTreeMap<String, Vec<Row>> = serde_json::from_value(req.get("logs").cloned().unwrap_or(json!({}))).map_err(|_| Error::Invalid("logs map origins to rows".into()))?;
    let trusted = vector_of(req, "trusted_vector")?;
    let h = hosted(node, req, bfp)?;
    let mut report = Report::default();
    let stored = take(&mut h.board, &logs, bfp, &mut report)?;
    h.board.acknowledge(bfp, &trusted)?;
    Ok(json!({"stored": stored, "refused": report.refused}))
}

/// Answer one request from the device with certificate ``der``. The pool record is read here,
/// under the node's lock, so the check and the work see the same record.
fn dispatch(net: &Net, node: &mut Node, der: &[u8], ip: IpAddr, req: &Value) -> Result<Value> {
    let fp = cert_fingerprint(der);
    let result = answer(net, node, der, req)?;
    // A device that has just joined says which port it listens on; its address is where it called from.
    if matches!(req["op"].as_str(), Some("join_open" | "pair_confirm")) {
        if let Some(port) = req["port"].as_u64().and_then(|p| u16::try_from(p).ok()).filter(|p| *p > 0) {
            node.facts.set_addr(&fp, &SocketAddr::new(ip, port).to_string())?;
        }
    }
    Ok(result)
}

fn answer(net: &Net, node: &mut Node, der: &[u8], req: &Value) -> Result<Value> {
    let fp = cert_fingerprint(der);
    let standing = node.members.standing(&fp);
    if standing == Standing::Revoked {
        return Err(Error::Denied("this device was put out of the pool".into()));
    }
    let bfp = board_fingerprint(der)?;
    let op = text(req, "op")?;
    match (op, standing) {
        ("hello", _) => Ok(hello(node, standing)),
        ("join_open", Standing::Unknown) => join_open(node, der, req),
        ("pair_exchange" | "pair_confirm", Standing::Unknown) => net.pairing_step(node, der, op, req),
        ("join_open" | "pair_exchange" | "pair_confirm", _) => Err(Error::Denied("this device is already a member".into())),
        (_, Standing::Unknown) => Err(Error::Denied("this device is not a member of the pool".into())),
        ("members", _) => members(node, &fp, req),
        ("boards", _) => Ok(json!({"boards": node.boards.keys().collect::<Vec<_>>()})),
        ("vector", _) => vector(node, &bfp, req),
        ("pull", _) => pull(node, &bfp, req),
        ("push", _) => push(node, &bfp, req),
        _ => Err(Error::Invalid("unknown op".into())),
    }
}

/// Serve one accepted connection until the peer closes it or is refused.
pub fn serve(net: &Net, tcp: TcpStream) -> Result<()> {
    tcp.set_read_timeout(Some(IO_TIMEOUT))?;
    tcp.set_write_timeout(Some(IO_TIMEOUT))?;
    let ip = tcp.peer_addr()?.ip();
    let mut s = tls::accept(net.server_config.clone(), tcp)?;
    let fp = cert_fingerprint(&s.peer);
    net.session_opened(&fp);
    let result = serve_requests(net, &mut s, ip);
    net.session_closed(&fp);
    result
}

fn serve_requests(net: &Net, s: &mut Secured<rustls::ServerConnection>, ip: IpAddr) -> Result<()> {
    let der = s.peer.clone();
    while let Some(req) = read_frame_max(s, MAX_PEER_FRAME)? {
        let (result, revoked) = {
            let mut node = net.node.lock().map_err(|_| Error::Damaged("node poisoned".into()))?;
            let result = dispatch(net, &mut node, &der, ip, &req);
            (result, node.members.standing(&cert_fingerprint(&der)) == Standing::Revoked)
        };
        write_frame_max(s, &reply_of(result), MAX_PEER_FRAME)?;
        if revoked {
            return Ok(());
        }
    }
    Ok(())
}

/// A connection to a peer, authenticated both ways.
pub struct PeerClient {
    secured: Secured<rustls::ClientConnection>,
    pub fingerprint: String,
    pub board_fingerprint: String,
    pub der: Vec<u8>,
}

impl PeerClient {
    /// Connect to ``addr`` showing ``me``; with ``expect`` the peer must show exactly that
    /// certificate (its fingerprint), without it any certificate is taken and the caller must
    /// authenticate it (pairing).
    pub fn connect(me: &Identity, addr: SocketAddr, expect: Option<&str>) -> Result<PeerClient> {
        let tcp = TcpStream::connect_timeout(&addr, Duration::from_secs(5))?;
        tcp.set_read_timeout(Some(IO_TIMEOUT))?;
        tcp.set_write_timeout(Some(IO_TIMEOUT))?;
        tcp.set_nodelay(true)?;
        let secured = tls::connect(tls::client_config(me, expect)?, tcp)?;
        let der = secured.peer.clone();
        Ok(PeerClient { fingerprint: cert_fingerprint(&der), board_fingerprint: board_fingerprint(&der)?, der, secured })
    }

    /// Send one request; a refusal comes back as the matching error.
    pub fn call(&mut self, request: &Value) -> Result<Value> {
        write_frame_max(&mut self.secured, request, MAX_PEER_FRAME)?;
        let reply = read_frame_max(&mut self.secured, MAX_PEER_FRAME)?.ok_or_else(|| Error::Denied("the peer closed the connection".into()))?;
        if reply["ok"] == true {
            return Ok(reply["result"].clone());
        }
        let message = reply["error"]["message"].as_str().unwrap_or("refused").to_string();
        Err(match reply["error"]["code"].as_str() {
            Some("denied") => Error::Denied(message),
            Some("quota") => Error::Quota(message),
            Some("damaged") => Error::Damaged(message),
            Some("gap") => Error::Gap(message),
            _ => Error::Invalid(message),
        })
    }
}
