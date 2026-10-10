//! The landing queue as board entries: a request names one branch at one exact commit, an
//! independent session reviews that commit, and the runner (the session holding the target's
//! branch claim) records how each request moved. The queue is a fold of the `landing` entries of
//! the board, so it needs no other state; the node stamps every entry with the caller's name.

use serde_json::{json, Map, Value};

use crate::error::{Error, Result};
use crate::fold::Entry;
use crate::node::Node;
use crate::row::{is_line, Kind};

pub const METHODS: [&str; 8] = ["land_request", "land_review", "land_cancel", "land_brake", "land_state", "land_beat", "land_queue", "land_trust"];
const EVENTS: [&str; 7] = ["request", "review", "cancel", "pause", "resume", "state", "beat"];
const OPEN: [&str; 3] = ["queued", "needs-review", "running"];
const STATES: [&str; 9] = ["queued", "needs-review", "running", "landed", "landed-unpushed", "failed", "needs-human", "refused", "cancelled"];
const TERMINAL: [&str; 7] = ["landed", "landed-unpushed", "failed", "needs-human", "refused", "cancelled", "superseded"];
const MAX_SELECTORS: usize = 64;
const MAX_TEXT: usize = 400;
const MAX_LINEAGE: usize = 32;
/// The grant a session needs, besides having no parent, to pause, resume or cancel for others.
pub const CONTROL: &str = "land_control";
/// The grant a session needs, besides having no parent, to name another device whose landing entries count.
pub const TRUST: &str = "land_trust";
const MOST_IGNORED: usize = 50;

pub fn params_for(method: &str) -> Option<&'static [&'static str]> {
    Some(match method {
        "land_request" => &["branch", "sha", "target", "selectors", "replaces"],
        "land_review" => &["req", "sha", "verdict"],
        "land_cancel" => &["req"],
        "land_brake" => &["pause", "reason"],
        "land_state" => &["req", "status", "detail", "evidence"],
        "land_beat" => &["what"],
        "land_queue" => &[],
        "land_trust" => &["device", "trusted"],
        _ => return None,
    })
}

fn bad<T>(why: &str) -> Result<T> {
    Err(Error::Invalid(why.into()))
}

fn word(map: &Map<String, Value>, key: &str, most: usize) -> Result<String> {
    match map.get(key) {
        None => Ok(String::new()),
        Some(Value::String(s)) if s.len() <= most && is_line(s) => Ok(s.trim().to_string()),
        _ => bad(&format!("{key} is one line of at most {most} characters")),
    }
}

fn plain_branch(name: &str) -> bool {
    let first = name.bytes().next().is_some_and(|b| b.is_ascii_alphanumeric());
    first && name.len() <= 128 && name.bytes().all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'/' | b'-')) && !matches!(name, "main" | "master")
}

fn full_sha(sha: &str) -> bool {
    sha.len() == 40 && sha.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

/// Check the fields of one landing entry, with the rules a peer applies to a row it receives.
pub fn check(map: &Map<String, Value>) -> Result<()> {
    let ev = word(map, "ev", 16)?;
    if !EVENTS.contains(&ev.as_str()) {
        return bad("a landing entry is a request, review, cancel, pause, resume, state or beat");
    }
    let (req, sha, status) = (word(map, "req", 128)?, word(map, "sha", 64)?, word(map, "status", 24)?);
    word(map, "detail", MAX_TEXT)?;
    word(map, "reason", MAX_TEXT)?;
    word(map, "what", 300)?;
    word(map, "replaces", MAX_TEXT)?;
    let needs_req = matches!(ev.as_str(), "review" | "cancel" | "state");
    if needs_req == req.is_empty() {
        return bad("review, cancel and state name a request, and nothing else does");
    }
    match ev.as_str() {
        "request" => {
            let (branch, target) = (word(map, "branch", 128)?, word(map, "target", 128)?);
            if !plain_branch(&branch) || !plain_branch(&target) {
                return bad("branch and target must be plain branch names, never main");
            }
            if !full_sha(&sha) {
                return bad("land-request needs the full 40-character commit SHA of the branch tip");
            }
            match map.get("selectors") {
                Some(Value::Array(s)) if !s.is_empty() && s.len() <= MAX_SELECTORS && s.iter().all(|v| v.as_str().is_some_and(|t| t.len() <= 300 && is_line(t))) => {}
                _ => return bad("name the affected test selectors (at least one, at most 64)"),
            }
        }
        "review" => {
            if !full_sha(&sha) || !matches!(word(map, "verdict", 8)?.as_str(), "accept" | "reject") {
                return bad("a review names the full commit and a verdict, accept or reject");
            }
        }
        "state" if !STATES.contains(&status.as_str()) => return bad("a request moves to a known state"),
        _ => {}
    }
    match map.get("standing") {
        None => {}
        Some(Value::Object(st)) if matches!(ev.as_str(), "request" | "review") && st.len() <= 6 && standing_fits(st) => {}
        _ => return bad("a stamped standing is the sender's name, parent, ancestors, model and model state"),
    }
    match map.get("evidence") {
        None => Ok(()),
        Some(Value::Object(e)) if e.len() <= 8 && e.values().all(|v| v.is_boolean() || v.is_i64() || v.as_str().is_some_and(|s| s.len() <= 300 && is_line(s))) => Ok(()),
        _ => bad("evidence is at most eight short facts"),
    }
}

/// Whether a stamped standing holds only short lines and a short list of ancestors.
fn standing_fits(st: &Map<String, Value>) -> bool {
    let short = |v: &Value| v.as_str().is_some_and(|t| t.len() <= 256 && is_line(t));
    st.iter().all(|(k, v)| match k.as_str() {
        "name" | "parent" | "model" | "model_state" => short(v),
        "ancestors" => v.as_array().is_some_and(|a| a.len() <= MAX_LINEAGE && a.iter().all(short)),
        _ => false,
    })
}

/// One review as recorded: who, what verdict, when, and the standing the node stamped on it.
struct Review {
    by: String,
    verdict: String,
    at: u64,
    standing: Value,
}

struct Req {
    id: String,
    shown: String,
    by: String,
    at: u64,
    branch: String,
    sha: String,
    target: String,
    selectors: Value,
    replaces: String,
    status: String,
    detail: String,
    evidence: Value,
    standing: Value,
    reviews: Vec<Review>,
}

#[derive(Default)]
struct Queue {
    reqs: Vec<Req>,
    paused: bool,
    paused_by: String,
    beat: Option<(String, String, u64)>,
    /// The latest entries of devices this device does not trust, kept in the view and counted for nothing.
    ignored: Vec<Value>,
    ignored_total: usize,
}

fn field(e: &Entry, key: &str) -> String {
    e.fields.get(key).and_then(Value::as_str).unwrap_or("").to_string()
}

impl Queue {
    fn ignore(&mut self, e: &Entry, device: &str) {
        self.ignored_total += 1;
        if self.ignored.len() == MOST_IGNORED {
            self.ignored.remove(0);
        }
        self.ignored.push(json!({"entry": e.id, "ev": field(e, "ev"), "by": e.sender, "device": device, "ts_ms": e.hlc.0}));
    }

    fn apply(&mut self, e: &Entry, own: &str) {
        let ev = field(e, "ev");
        if ev == "request" {
            for old in self.reqs.iter_mut().filter(|r| r.branch == field(e, "branch") && OPEN.contains(&r.status.as_str())) {
                old.status = "superseded".into();
                old.detail = format!("replaced by {}", shown_id(e, own));
            }
            self.reqs.push(Req {
                id: e.id.clone(), shown: shown_id(e, own), by: e.sender.clone(), at: e.hlc.0, branch: field(e, "branch"), sha: field(e, "sha"),
                target: field(e, "target"), selectors: e.fields.get("selectors").cloned().unwrap_or(json!([])), replaces: field(e, "replaces"),
                status: "needs-review".into(), detail: "no independent review yet".into(), evidence: json!({}), standing: e.fields.get("standing").cloned().unwrap_or(Value::Null),
                reviews: Vec::new(),
            });
            return;
        }
        match ev.as_str() {
            "pause" => (self.paused, self.paused_by) = (true, e.sender.clone()),
            "resume" => (self.paused, self.paused_by) = (false, String::new()),
            "beat" => self.beat = Some((e.sender.clone(), field(e, "what"), e.hlc.0)),
            _ => {}
        }
        let Some(req) = self.reqs.iter_mut().find(|r| r.id == field(e, "req")) else { return };
        let late = req.status == "landed-unpushed" && ev == "state" && field(e, "status") == "landed";
        if TERMINAL.contains(&req.status.as_str()) && ev != "review" && !late {
            return;
        }
        match ev.as_str() {
            "review" => {
                req.reviews.retain(|r| r.by != e.sender);
                req.reviews.push(Review {
                    by: e.sender.clone(), verdict: field(e, "verdict"), at: e.hlc.0, standing: e.fields.get("standing").cloned().unwrap_or(Value::Null),
                });
                refresh(req);
            }
            "cancel" => (req.status, req.detail) = ("cancelled".into(), format!("cancelled by {}", e.sender)),
            "state" => {
                (req.status, req.detail) = (field(e, "status"), field(e, "detail"));
                req.evidence = e.fields.get("evidence").cloned().unwrap_or(json!({}));
            }
            _ => {}
        }
    }
}

fn shown_id(e: &Entry, own: &str) -> String {
    if e.origin == own { format!("land-{}", e.seq) } else { e.id.clone() }
}

fn refresh(req: &mut Req) {
    if req.status == "queued" || req.status == "needs-review" {
        let accepted = req.reviews.iter().any(|r| r.verdict == "accept") && !req.reviews.iter().any(|r| r.verdict == "reject");
        (req.status, req.detail) = if accepted { ("queued".into(), String::new()) } else { ("needs-review".into(), "no independent review yet".into()) };
    }
}

fn fold(node: &mut Node, board: &str) -> Result<(Queue, String, u64)> {
    let entries = node.view(board)?;
    let host = node.host(board)?;
    let (own, now) = (host.board.origin().to_string(), host.board.now_ms());
    let writers = crate::landtrust::writers(node, board)?;
    let mut queue = Queue::default();
    for e in entries.iter().filter(|e| e.kind == Kind::Landing) {
        if e.origin == own || writers.trusted.contains(&e.origin) {
            queue.apply(e, &own);
        } else {
            queue.ignore(e, writers.shown.get(&e.origin).map_or("an unknown device", String::as_str));
        }
    }
    Ok((queue, own, now))
}

fn show(r: &Req) -> Value {
    let reviews: Map<String, Value> = r.reviews.iter().map(|v| (v.by.clone(), json!({"verdict": v.verdict, "ts_ms": v.at, "standing": v.standing}))).collect();
    json!({"id": r.shown, "entry": r.id, "branch": r.branch, "sha": r.sha, "target": r.target, "selectors": r.selectors, "replaces": r.replaces,
           "by": r.by, "ts_ms": r.at, "standing": r.standing, "status": r.status, "detail": r.detail, "evidence": r.evidence, "reviews": reviews})
}

fn text<'a>(p: &'a Map<String, Value>, key: &str) -> &'a str {
    p.get(key).and_then(Value::as_str).unwrap_or("")
}

/// Whether ``a`` and ``b`` are one session or one is an ancestor of the other.
fn related(node: &mut Node, board: &str, a: &str, b: &str) -> Result<bool> {
    let names = &node.host(board)?.names;
    Ok(a == b || names.descendants(a).iter().any(|n| n == b) || names.descendants(b).iter().any(|n| n == a))
}

/// What the node knows of ``name`` now, stamped on the entry it writes: the name, the parent, every ancestor, and
/// the model with how far it is believed. Later, the runner asks about this and not about who is still registered.
fn stamp(node: &mut Node, board: &str, name: &str) -> Result<Value> {
    let names = &node.host(board)?.names;
    let ident = names.get(name).ok_or_else(|| Error::Denied("no such session".into()))?;
    let mut ancestors: Vec<String> = Vec::new();
    let mut up = ident.parent.clone();
    while !up.is_empty() && !ancestors.contains(&up) && ancestors.len() < MAX_LINEAGE {
        ancestors.push(up.clone());
        up = names.get(&up).map(|i| i.parent.clone()).unwrap_or_default();
    }
    Ok(json!({"name": name, "parent": ident.parent, "ancestors": ancestors, "model": ident.model, "model_state": ident.model_state}))
}

/// Whether ``actor`` may steer landing on ``target``: it holds the target's runner claim, or it is a
/// session with no parent that the grants allow.
fn controls(node: &mut Node, board: &str, token: &str, actor: &str, target: &str) -> Result<bool> {
    if crate::claims::owner(node, board, "branch", target)?.as_deref() == Some(actor) {
        return Ok(true);
    }
    let holder = node.tokens.resolve(token).cloned().ok_or_else(|| Error::Denied("the token is not known".into()))?;
    let top = node.host(board)?.names.get(actor).is_some_and(|i| i.parent.is_empty());
    Ok(top && node.grants.allows(&holder, CONTROL))
}

fn write(node: &mut Node, board: &str, actor: &str, body: Value) -> Result<Value> {
    crate::fold::check_local(Kind::Landing, actor, &body, "")?;
    let row = node.host(board)?.board.append(Kind::Landing, actor, body, "")?;
    Ok(json!({"seq": row.seq, "id": row.id()}))
}

fn audit(node: &mut Node, board: &str, actor: &str, event: &str, subject: &str, detail: &str) -> Result<()> {
    let clip = |s: &str| s.chars().filter(|c| !c.is_control()).take(300).collect::<String>();
    node.host(board)?.board.append(Kind::Audit, actor, json!({"event": event, "subject": clip(subject), "detail": clip(detail)}), "")?;
    Ok(())
}

/// The session of this board behind ``token``, live; a retired or linked-in session does not land work.
fn member(node: &mut Node, token: &str, board: &str) -> Result<String> {
    let access = node.access(token, board, true)?;
    let retired = node.host(board)?.names.get(&access.actor).is_none_or(|i| i.retired);
    if access.holder.board != board || retired {
        return Err(Error::Denied("only a live session of this board uses its landing queue".into()));
    }
    Ok(access.actor)
}

fn open_request<'q>(queue: &'q Queue, own: &str, board: &str, asked: &str, closed: bool) -> Result<&'q Req> {
    let id = match asked.strip_prefix("land-").filter(|n| !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit())) {
        Some(n) => format!("{board}:{own}:{n}"),
        None => asked.to_string(),
    };
    queue.reqs.iter().find(|r| r.id == id && (closed || !TERMINAL.contains(&r.status.as_str()))).ok_or_else(|| Error::Invalid(format!("{asked} is not an open request")))
}

fn request(node: &mut Node, board: &str, token: &str, p: &Map<String, Value>) -> Result<Value> {
    let actor = member(node, token, board)?;
    let target = match text(p, "target") {
        "" => return Err(Error::Invalid("a request names the target branch".into())),
        t => t,
    };
    let mut body = Map::new();
    body.insert("ev".into(), json!("request"));
    for key in ["branch", "sha", "replaces"] {
        body.insert(key.into(), json!(text(p, key)));
    }
    body.insert("target".into(), json!(target));
    body.insert("selectors".into(), p.get("selectors").cloned().unwrap_or(Value::Null));
    body.insert("standing".into(), stamp(node, board, &actor)?);
    let done = write(node, board, &actor, Value::Object(body))?;
    audit(node, board, &actor, "land.request", &format!("land-{}", done["seq"]), &format!("{}@{}", text(p, "branch"), text(p, "sha")))?;
    Ok(json!({"id": format!("land-{}", done["seq"]), "branch": text(p, "branch"), "sha": text(p, "sha"), "status": "needs-review"}))
}

fn review(node: &mut Node, board: &str, token: &str, p: &Map<String, Value>) -> Result<Value> {
    let actor = member(node, token, board)?;
    let (queue, own, _) = fold(node, board)?;
    let req = open_request(&queue, &own, board, text(p, "req"), false)?;
    if related(node, board, &actor, &req.by)? {
        return Err(Error::Denied("an independent reviewer is not the requester or one of its delegates".into()));
    }
    if text(p, "sha") != req.sha {
        return Err(Error::Invalid(format!("{} is for {}, not {}; review the exact commit", req.shown, req.sha, text(p, "sha"))));
    }
    let (shown, verdict, id, tip) = (req.shown.clone(), text(p, "verdict").to_string(), req.id.clone(), req.sha.clone());
    let body = json!({"ev": "review", "req": id, "sha": tip, "verdict": verdict, "standing": stamp(node, board, &actor)?});
    write(node, board, &actor, body)?;
    audit(node, board, &actor, "land.review", &shown, &verdict)?;
    Ok(json!({"id": shown, "verdict": verdict}))
}

fn cancel(node: &mut Node, board: &str, token: &str, p: &Map<String, Value>) -> Result<Value> {
    let actor = member(node, token, board)?;
    let (queue, own, _) = fold(node, board)?;
    let req = open_request(&queue, &own, board, text(p, "req"), false)?;
    if req.by != actor && !controls(node, board, token, &actor, &req.target)? {
        return Err(Error::Denied("only the requester, the runner or a controlling session cancels a request".into()));
    }
    let (shown, id) = (req.shown.clone(), req.id.clone());
    write(node, board, &actor, json!({"ev": "cancel", "req": id}))?;
    audit(node, board, &actor, "land.cancel", &shown, "")?;
    Ok(json!({"id": shown, "status": "cancelled"}))
}

fn brake(node: &mut Node, board: &str, token: &str, p: &Map<String, Value>) -> Result<Value> {
    let actor = member(node, token, board)?;
    let pause = p.get("pause").and_then(Value::as_bool).ok_or_else(|| Error::Invalid("pause is true or false".into()))?;
    let target = crate::claims::owned_branch(node, board, &actor)?;
    if target.is_none() && !controls(node, board, token, &actor, "")? {
        return Err(Error::Denied("only the runner or a controlling session pauses or resumes landing".into()));
    }
    let event = if pause { "pause" } else { "resume" };
    write(node, board, &actor, json!({"ev": event, "reason": text(p, "reason")}))?;
    audit(node, board, &actor, &format!("land.{event}"), "", text(p, "reason"))?;
    Ok(json!({"paused": pause}))
}

fn state(node: &mut Node, board: &str, token: &str, p: &Map<String, Value>) -> Result<Value> {
    let actor = member(node, token, board)?;
    let (queue, own, _) = fold(node, board)?;
    let req = open_request(&queue, &own, board, text(p, "req"), true)?;
    if crate::claims::owner(node, board, "branch", &req.target)?.as_deref() != Some(actor.as_str()) {
        return Err(Error::Denied("only the landing runner, the holder of the branch claim, records request states".into()));
    }
    let (shown, id, status) = (req.shown.clone(), req.id.clone(), text(p, "status").to_string());
    let mut body = json!({"ev": "state", "req": id, "status": status, "detail": text(p, "detail")});
    if let Some(e) = p.get("evidence") {
        body["evidence"] = e.clone();
    }
    write(node, board, &actor, body)?;
    audit(node, board, &actor, "land.state", &shown, &status)?;
    Ok(json!({"id": shown, "status": status}))
}

fn beat(node: &mut Node, board: &str, token: &str, p: &Map<String, Value>) -> Result<Value> {
    let actor = member(node, token, board)?;
    if crate::claims::owned_branch(node, board, &actor)?.is_none() {
        return Err(Error::Denied("only the landing runner, the holder of a branch claim, marks progress".into()));
    }
    write(node, board, &actor, json!({"ev": "beat", "what": text(p, "what")}))?;
    Ok(json!({"beat": true}))
}

/// The branch claims another device holds on the branches this device lands onto (a request's target, or a branch its
/// runner holds). The runner's own claim is held among this device's sessions only, so these are listed and ignored.
fn foreign_claims(node: &mut Node, board: &str, targets: &[String]) -> Result<Vec<Value>> {
    let entries = node.view(board)?;
    let writers = crate::landtrust::writers(node, board)?;
    let mut keys: Vec<String> = targets.to_vec();
    let prefix = format!("{board}/");
    for l in node.leases.table.leases.iter().filter(|l| l.board == board && l.local) {
        keys.extend(l.resources.iter().filter_map(|r| match r {
            crate::lease::types::Resource::Claim { kind, name } if kind == "branch" => name.strip_prefix(&prefix).map(String::from),
            _ => None,
        }));
    }
    let held = crate::lease::merge::view(&entries);
    Ok(held.exclusive.iter().filter(|(_, h)| h.foreign).filter_map(|((_, key), h)| {
        let branch = key.strip_prefix("claim:branch:")?.strip_prefix(&prefix)?;
        let shown = writers.shown.get(&h.origin).map_or("an unknown device", String::as_str);
        keys.iter().any(|k| k == branch).then(|| json!({"branch": branch, "holder": h.holder, "device": shown, "status": "foreign, ignored"}))
    }).collect())
}

fn queue(node: &mut Node, board: &str, token: &str) -> Result<Value> {
    let access = node.access(token, board, false)?;
    if access.channels.as_ref().is_some_and(|c| !c.iter().any(|x| x == "#landing")) {
        return Err(Error::Denied("the landing queue is not shared with this session".into()));
    }
    let (queue, _, now) = fold(node, board)?;
    let beat = queue.beat.as_ref().map(|(by, what, at)| json!({"by": by, "what": what, "age_s": now.saturating_sub(*at) / 1000}));
    let targets: Vec<String> = queue.reqs.iter().map(|r| r.target.clone()).collect();
    let claims = foreign_claims(node, board, &targets)?;
    let trusted: Vec<String> = node.land_trust.devices(&node.members.id).into_iter().collect();
    Ok(json!({"requests": queue.reqs.iter().map(show).collect::<Vec<_>>(), "paused": queue.paused, "paused_by": queue.paused_by, "beat": beat,
              "foreign": {"total": queue.ignored_total, "latest": queue.ignored, "status": "foreign, ignored"}, "foreign_claims": claims, "trusted_devices": trusted}))
}

/// Name a device whose landing entries this device counts, or stop counting them: a session with no parent
/// that holds the grant ``land_trust``, recorded on the board.
fn trust(node: &mut Node, board: &str, token: &str, p: &Map<String, Value>) -> Result<Value> {
    let actor = member(node, token, board)?;
    let holder = node.tokens.resolve(token).cloned().ok_or_else(|| Error::Denied("the token is not known".into()))?;
    if !node.host(board)?.names.get(&actor).is_some_and(|i| i.parent.is_empty()) || !node.grants.allows(&holder, TRUST) {
        return Err(Error::Denied("only a session with no parent that holds the grant land_trust names the devices whose landing entries count".into()));
    }
    let on = p.get("trusted").and_then(Value::as_bool).ok_or_else(|| Error::Invalid("trusted is true or false".into()))?;
    let fingerprint = text(p, "device");
    let device = node.members.get(fingerprint).cloned();
    if on && !device.as_ref().is_some_and(|d| d.status == crate::membership::Status::Active) {
        return Err(Error::Invalid("only an active member of this pool is trusted to land".into()));
    }
    if fingerprint == node.cert.fingerprint() {
        return Err(Error::Invalid("this device's own entries always count".into()));
    }
    let pool = node.members.id.clone();
    let changed = node.land_trust.set(&pool, fingerprint, on)?;
    let name = device.map(|d| d.name).unwrap_or_default();
    audit(node, board, &actor, if on { "land.trust" } else { "land.untrust" }, fingerprint, &name)?;
    Ok(json!({"device": fingerprint, "trusted": on, "changed": changed}))
}

pub fn call(node: &mut Node, method: &str, board: &str, token: &str, p: &Map<String, Value>) -> Result<Value> {
    match method {
        "land_request" => request(node, board, token, p),
        "land_review" => review(node, board, token, p),
        "land_cancel" => cancel(node, board, token, p),
        "land_brake" => brake(node, board, token, p),
        "land_state" => state(node, board, token, p),
        "land_beat" => beat(node, board, token, p),
        "land_trust" => trust(node, board, token, p),
        _ => queue(node, board, token),
    }
}
