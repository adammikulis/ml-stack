//! The receiving side of test shards: who may send, the tree arriving, the run under a lease.
//!
//! Every op is a peer op, so the sender is the certificate it showed and a pool member by the
//! record at that moment. A shard belongs to the device that created it: only that device sees
//! its result or cancels it. The sender names a tier, test files and a time limit; the command
//! is built here from the saved consent, never from the request.

use std::collections::BTreeMap;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};

use super::consent::{self, Consent};
use super::spec::{self, Spec};
use super::{proc, runtime};
use crate::error::{Error, Result};
use crate::fsutil::{hex, random_hex, unhex};
use crate::lease::table::{Event, Why};
use crate::lease::types::{Class, Request, Resource, State, LOCAL};
use crate::lease::rpc;
use crate::node::Node;
use crate::poolops::{wall_ms, POOL_BOARD};

pub const MOST_RUNNING: usize = 2;
pub const MOST_OPEN: usize = 6;
pub const MOST_KEPT: usize = 40;
pub const CHUNK: usize = 256 << 10;
pub const MOST_RESULT: u64 = 900 << 10;
const STALE_UPLOAD: Duration = Duration::from_secs(600);
/// Seconds past its time limit a run is left before the node ends it itself.
const GRACE_S: u64 = 120;
const KILL_AFTER: Duration = Duration::from_secs(45);
const SHARDS: &str = "shards";
const BOOT: &str = "import sys; sys.path.insert(0, sys.argv[1]); from poolhouse.fleet.shard_exec import run; raise SystemExit(run(sys.argv[2]))";
/// Variables that mark a shell as an agent's, or point a tree somewhere else: none reaches a run.
const DROPPED: [&str; 18] = ["CLAUDECODE", "AI_AGENT", "CLAUDE_CODE_ENTRYPOINT", "CLAUDE_CODE_SESSION_ATTENDED", "CODEX_THREAD_ID",
    "CODEX_SESSION_ID", "POOLHOUSE_AGENT", "POOLHOUSE_NONINTERACTIVE", "POOLHOUSE_WORKSPACE_TOKEN", "POOLHOUSE_WORKSPACE_AGENT",
    "POOLHOUSE_WORKSPACE_DENYLIST", "POOLHOUSE_SESSION_HARNESS", "POOLHOUSE_SESSION_ID", "POOLHOUSE_BOARD", "DEV_TEST_JOB",
    "DEV_TEST_AGENT", "PYTHONPATH", "PYTHONHOME"];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    Uploading,
    Running,
    Done,
    Failed,
    Cancelled,
}

impl Phase {
    fn word(self) -> &'static str {
        match self {
            Phase::Uploading => "uploading",
            Phase::Running => "running",
            Phase::Done => "done",
            Phase::Failed => "failed",
            Phase::Cancelled => "cancelled",
        }
    }

    fn open(self) -> bool {
        matches!(self, Phase::Uploading | Phase::Running)
    }
}

struct Job {
    owner: String,
    phase: Phase,
    dir: PathBuf,
    got: u64,
    hasher: Sha256,
    cancel: bool,
    touched: Instant,
    result: Option<Value>,
    error: String,
    created_ms: u64,
}

pub struct Shards {
    node: Arc<Mutex<Node>>,
    root: PathBuf,
    stop: Arc<AtomicBool>,
    jobs: Mutex<BTreeMap<String, Job>>,
    refused: Mutex<BTreeMap<String, Instant>>,
}

fn lock<T>(m: &Mutex<T>) -> Result<MutexGuard<'_, T>> {
    m.lock().map_err(|_| Error::Damaged("shards poisoned".into()))
}

fn field<'a>(req: &'a Value, key: &str) -> Result<&'a str> {
    req.get(key).and_then(Value::as_str).ok_or_else(|| Error::Invalid(format!("{key} is text")))
}

/// ``req`` holds no field outside ``allowed`` (and the envelope's `op` and `by`).
fn only(req: &Value, allowed: &[&str]) -> Result<()> {
    let m = req.as_object().ok_or_else(|| Error::Invalid("a shard request is an object".into()))?;
    match m.keys().find(|k| !matches!(k.as_str(), "op" | "by") && !allowed.contains(&k.as_str())) {
        Some(k) => Err(Error::Invalid(format!("a shard takes no field called {}", k.chars().take(40).collect::<String>()))),
        None => Ok(()),
    }
}

/// Who the sender says it is, as a label for the audit trail only: the certificate decides.
fn label(req: &Value) -> String {
    req.get("by").and_then(Value::as_str).unwrap_or("").chars().filter(|c| c.is_ascii_alphanumeric() || matches!(c, '@' | '/' | '.' | '_' | '-')).take(60).collect()
}

fn unknown() -> Error {
    Error::Invalid("no such shard".into())
}

impl Shards {
    /// The host for ``node``; shards a node left behind when it stopped are removed.
    pub fn new(node: Arc<Mutex<Node>>, dir: &Path, stop: Arc<AtomicBool>) -> Result<Arc<Shards>> {
        let root = dir.join(SHARDS);
        let _ = std::fs::remove_dir_all(&root);
        crate::fsutil::private_dir(&root)?;
        Ok(Arc::new(Shards { node, root, stop, jobs: Mutex::new(BTreeMap::new()), refused: Mutex::new(BTreeMap::new()) }))
    }

    pub fn handle(self: &Arc<Shards>, node: &mut Node, fp: &str, op: &str, req: &Value) -> Result<Value> {
        self.forget_stale()?;
        if op != "shard_caps" {
            self.allowed(node, fp, op)?;
        }
        match op {
            "shard_caps" => only(req, &[]).and_then(|()| self.caps(node, fp)),
            "shard_put" => only(req, &["id", "offset", "data"]).and_then(|()| self.put(node, fp, req)),
            "shard_start" => self.start(node, fp, req),
            "shard_status" => only(req, &["id"]).and_then(|()| self.status(fp, req)),
            "shard_cancel" => only(req, &["id"]).and_then(|()| self.cancel(fp, req)),
            _ => Err(Error::Invalid("unknown op".into())),
        }
    }

    /// A member that is not on the allowed list asks nothing but `shard_caps`; the refusal is on the pool board,
    /// at most once a minute for each device.
    fn allowed(&self, node: &mut Node, fp: &str, op: &str) -> Result<()> {
        let consent = consent::load(&node.dir).unwrap_or_default();
        if consent.allowed.iter().any(|a| a == fp) {
            return Ok(());
        }
        let due = lock(&self.refused)?.get(fp).is_none_or(|at| at.elapsed() > Duration::from_secs(60));
        if due {
            lock(&self.refused)?.insert(fp.into(), Instant::now());
            let name = node.members.get(fp).map(|d| d.name.clone()).unwrap_or_default();
            node.record_event("shard_refused", fp, &format!("{op} from {name} {}: not on the allowed list", &fp[..8]))?;
        }
        if !consent.enabled {
            return Err(Error::Denied("test shards are off on this device; its person turns them on".into()));
        }
        Err(Error::Denied("this device does not take tests from you; its person allows a device with `python -m poolhouse.testfarm.consent allow DEVICE`".into()))
    }

    fn caps(&self, node: &Node, fp: &str) -> Result<Value> {
        let consent = consent::load(&node.dir).unwrap_or_default();
        let mut reason = String::new();
        let mut python = Value::Null;
        let allowed = consent.allowed.iter().any(|a| a == fp);
        if !consent.enabled {
            reason = "test shards are off on this device; its person turns them on (python -m poolhouse.testfarm.consent on)".into();
        } else if !allowed {
            reason = "this device does not take tests from you; its person allows you with `python -m poolhouse.testfarm.consent allow DEVICE`".into();
        } else {
            match runtime::python_version(&consent.python) {
                Ok(v) => python = json!(v),
                Err(e) => reason = e.to_string(),
            }
        }
        let running = lock(&self.jobs)?.values().filter(|j| j.phase == Phase::Running).count();
        let name = node.members.get(&node.cert.fingerprint()).map(|d| d.name.clone()).unwrap_or_default();
        Ok(json!({"accepts": reason.is_empty(), "reason": reason, "platform": runtime::platform(), "python": python, "name": name, "allowed": allowed,
                  "fingerprint": node.cert.fingerprint(), "active": running, "most_active": MOST_RUNNING, "free": MOST_RUNNING.saturating_sub(running)}))
    }

    fn enabled(&self, node: &Node) -> Result<Consent> {
        let consent = consent::load(&node.dir)?;
        if !consent.enabled {
            return Err(Error::Denied("test shards are off on this device; its person turns them on".into()));
        }
        Ok(consent)
    }

    fn put(&self, node: &Node, fp: &str, req: &Value) -> Result<Value> {
        self.enabled(node)?;
        let id = field(req, "id")?;
        let offset = req.get("offset").and_then(Value::as_u64).ok_or_else(|| Error::Invalid("offset is a whole number".into()))?;
        let data = unhex(field(req, "data")?).filter(|d| !d.is_empty() && d.len() <= CHUNK).ok_or_else(|| Error::Invalid(format!("data is 1 to {CHUNK} bytes of hex")))?;
        if !spec::valid_id(id) {
            return Err(Error::Invalid("the shard id is 32 lowercase hex digits".into()));
        }
        let mut jobs = lock(&self.jobs)?;
        if offset == 0 && !jobs.contains_key(id) {
            if jobs.values().filter(|j| j.phase.open()).count() >= MOST_OPEN || jobs.values().filter(|j| j.owner == fp && j.phase == Phase::Uploading).count() >= 2 {
                return Err(Error::Quota("this device is taking as many shards as it can; try again shortly".into()));
            }
            let dir = self.root.join(id);
            std::fs::create_dir(&dir)?;
            jobs.insert(id.into(), Job { owner: fp.into(), phase: Phase::Uploading, dir, got: 0, hasher: Sha256::new(), cancel: false,
                                          touched: Instant::now(), result: None, error: String::new(), created_ms: wall_ms() });
        }
        let job = jobs.get_mut(id).filter(|j| j.owner == fp).ok_or_else(unknown)?;
        if job.phase != Phase::Uploading || job.got != offset || job.got + data.len() as u64 > spec::MOST_PACKED {
            return Err(Error::Invalid("a tree arrives once, in order, and is at most 24 MiB".into()));
        }
        let mut file = crate::sys::private_file(OpenOptions::new().create(true).append(true)).open(job.dir.join("tree.tgz"))?;
        file.write_all(&data)?;
        job.hasher.update(&data);
        job.got += data.len() as u64;
        job.touched = Instant::now();
        Ok(json!({"id": id, "got": job.got}))
    }

    fn forget_stale(&self) -> Result<()> {
        let mut jobs = lock(&self.jobs)?;
        let old: Vec<String> = jobs.iter().filter(|(_, j)| j.phase == Phase::Uploading && j.touched.elapsed() > STALE_UPLOAD).map(|(i, _)| i.clone()).collect();
        for id in old {
            if let Some(j) = jobs.remove(&id) {
                let _ = std::fs::remove_dir_all(j.dir);
            }
        }
        let mut done: Vec<(u64, String)> = jobs.iter().filter(|(_, j)| !j.phase.open()).map(|(i, j)| (j.created_ms, i.clone())).collect();
        done.sort();
        while done.len() > MOST_KEPT {
            let (_, id) = done.remove(0);
            if let Some(j) = jobs.remove(&id) {
                let _ = std::fs::remove_dir_all(j.dir);
            }
        }
        Ok(())
    }

    fn start(self: &Arc<Shards>, node: &mut Node, fp: &str, req: &Value) -> Result<Value> {
        let consent = self.enabled(node)?;
        let spec = spec::parse(req)?;
        let version = runtime::python_version(&consent.python)?;
        let by = label(req);
        let dir = {
            let mut jobs = lock(&self.jobs)?;
            if jobs.values().filter(|j| j.phase == Phase::Running).count() >= MOST_RUNNING {
                return Err(Error::Quota(format!("{MOST_RUNNING} shards are already running here")));
            }
            let job = jobs.get_mut(&spec.id).filter(|j| j.owner == fp).ok_or_else(unknown)?;
            if job.phase != Phase::Uploading || job.got != spec.size {
                return Err(Error::Invalid("the whole tree must arrive before the shard starts".into()));
            }
            if hex(&job.hasher.clone().finalize()) != spec.tree_sha256 {
                let gone = jobs.remove(&spec.id);
                gone.into_iter().for_each(|j| drop(std::fs::remove_dir_all(j.dir)));
                return Err(Error::Invalid("the tree does not match its digest".into()));
            }
            job.dir.clone()
        };
        let lease = self.take_slots(node, &spec)?;
        let child = match launch(&consent, &spec, &dir) {
            Ok(c) => c,
            Err(e) => {
                self.give_back(node, &lease);
                return Err(e);
            }
        };
        let device = node.members.get(fp).map(|d| d.name.clone()).unwrap_or_default();
        node.record_event("shard_start", &spec.id, &format!("from {device} {} by {by}: {} {} files, python {version}", &fp[..8], spec.tier, spec.files.len()))?;
        if let Some(j) = lock(&self.jobs)?.get_mut(&spec.id) {
            j.phase = Phase::Running;
        }
        let (host, id) = (self.clone(), spec.id.clone());
        let owner = fp.to_string();
        std::thread::spawn(move || host.watch(&id, (child, owner), &lease, spec.timeout_s + GRACE_S));
        Ok(json!({"id": req["id"], "state": "running", "python": version}))
    }

    /// The device's CPU slots for this shard from the lease service; `Quota` when none are free.
    fn take_slots(&self, node: &mut Node, spec: &Spec) -> Result<String> {
        let name = format!("shard-{}", &spec.id[..8]);
        let count = (node.leases.cfg.cpu_slots / 2).max(1);
        let request = Request { board: POOL_BOARD.into(), name: name.clone(), resources: vec![Resource::CpuSlots { device: LOCAL.into(), count }],
            class: Class::Background, estimate_s: Some(spec.timeout_s), ttl_s: Some(spec.timeout_s + GRACE_S + 300), pid: 0, remote: true, local: false, wait: false };
        rpc::tick(node, &[POOL_BOARD])?;
        let (now, id) = (node.leases.now(), format!("l{}", random_hex(8)?));
        let id = node.leases.table.enqueue(&node.leases.cfg, now, &request, id)?;
        rpc::tick(node, &[POOL_BOARD])?;
        let held = node.leases.table.find(&id).is_some_and(|l| l.state == State::Held);
        if !held {
            let _ = node.leases.table.release(&id, &format!("{POOL_BOARD}/{name}"));
            node.leases.save()?;
            return Err(Error::Quota("this device's CPU slots are in use; try again shortly".into()));
        }
        node.leases.save()?;
        Ok(id)
    }

    fn give_back(&self, node: &mut Node, lease: &str) {
        let Some(l) = node.leases.table.find(lease).cloned() else { return };
        if node.leases.table.release(lease, &l.holder).is_ok() {
            let _ = node.leases.save();
            let _ = rpc::mirror(node, &[Event::Ended(l, Why::Released)]);
            let _ = rpc::tick(node, &[POOL_BOARD]);
        }
    }

    fn watch(&self, id: &str, (mut child, owner): (std::process::Child, String), lease: &str, allowed_s: u64) {
        let mut ticks = 0u32;
        let pid = child.id();
        let deadline = Instant::now() + Duration::from_secs(allowed_s);
        let mut asked: Option<Instant> = None;
        let mut cancelled = false;
        let code = loop {
            match child.try_wait() {
                Ok(Some(status)) => break status.code(),
                Ok(None) => {}
                Err(_) => break None,
            }
            ticks += 1;
            cancelled = cancelled || (ticks % 10 == 0 && !self.still_allowed(&owner)) || self.stop.load(Ordering::SeqCst) || lock(&self.jobs).map_or(true, |j| j.get(id).is_none_or(|x| x.cancel));
            if cancelled || Instant::now() > deadline {
                match asked {
                    None => {
                        proc::terminate(pid);
                        asked = Some(Instant::now());
                    }
                    Some(at) if at.elapsed() > KILL_AFTER => proc::kill_group(pid),
                    Some(_) => {}
                }
            }
            std::thread::sleep(Duration::from_millis(100));
        };
        proc::kill_group(pid);
        let _ = child.wait();
        self.finish(id, code, cancelled, lease);
    }

    fn finish(&self, id: &str, code: Option<i32>, cancelled: bool, lease: &str) {
        let dir = self.root.join(id);
        let (phase, result, error) = outcome(&dir, code, cancelled);
        let _ = std::fs::remove_file(dir.join("tree.tgz"));
        let owner = lock(&self.jobs).ok().and_then(|mut jobs| jobs.get_mut(id).map(|j| {
            (j.phase, j.result, j.error) = (phase, result, error.clone());
            j.owner.clone()
        })).unwrap_or_default();
        if let Ok(mut node) = self.node.lock() {
            self.give_back(&mut node, lease);
            let who = node.members.get(&owner).map(|d| d.name.clone()).unwrap_or_default();
            let _ = node.record_event("shard_done", id, &format!("{} for {who}: exit {}", phase.word(), code.map_or(-1, i64::from)));
        }
    }

    /// Whether the sender of a running shard is still on the allowed list (taking it off ends its runs).
    fn still_allowed(&self, owner: &str) -> bool {
        let dir = self.root.parent().map(Path::to_path_buf).unwrap_or_default();
        consent::load(&dir).is_ok_and(|c| c.allowed.iter().any(|a| a == owner))
    }

    fn status(&self, fp: &str, req: &Value) -> Result<Value> {
        let id = field(req, "id")?;
        let jobs = lock(&self.jobs)?;
        let job = jobs.get(id).filter(|j| j.owner == fp).ok_or_else(unknown)?;
        let mut out = json!({"id": id, "state": job.phase.word(), "got": job.got});
        if let Some(r) = &job.result {
            out["result"] = r.clone();
        }
        if !job.error.is_empty() {
            out["error"] = json!(job.error);
        }
        Ok(out)
    }

    fn cancel(&self, fp: &str, req: &Value) -> Result<Value> {
        let id = field(req, "id")?;
        let mut jobs = lock(&self.jobs)?;
        let job = jobs.get_mut(id).filter(|j| j.owner == fp).ok_or_else(unknown)?;
        match job.phase {
            Phase::Uploading => {
                let gone = jobs.remove(id);
                gone.into_iter().for_each(|j| drop(std::fs::remove_dir_all(j.dir)));
                Ok(json!({"id": id, "state": "cancelled"}))
            }
            Phase::Running => {
                job.cancel = true;
                Ok(json!({"id": id, "state": "cancelling"}))
            }
            done => Ok(json!({"id": id, "state": done.word()})),
        }
    }
}

/// How a finished executor ended: its result when it wrote one that fits, else why it did not.
fn outcome(dir: &Path, code: Option<i32>, cancelled: bool) -> (Phase, Option<Value>, String) {
    if cancelled {
        return (Phase::Cancelled, None, "cancelled by the sender or because the node stopped".into());
    }
    match runtime::read_result(dir) {
        Ok(Some(result)) => (Phase::Done, Some(result), String::new()),
        Ok(None) => (Phase::Failed, None, format!("the executor ended (exit {}) without a result: {}", code.map_or(-1, i64::from), runtime::tail(&dir.join("executor.log")))),
        Err(e) => (Phase::Failed, None, e.to_string()),
    }
}

/// Start the executor in a group of its own: this device's python and checkout, the job's folder.
fn launch(consent: &Consent, spec: &Spec, dir: &Path) -> Result<std::process::Child> {
    let job = json!({"id": spec.id, "tree_sha256": spec.tree_sha256, "tier": spec.tier, "files": spec.files, "timeout_s": spec.timeout_s});
    std::fs::write(dir.join("spec.json"), serde_json::to_vec(&job)?)?;
    let log = crate::sys::private_file(OpenOptions::new().create(true).write(true).truncate(true)).open(dir.join("executor.log"))?;
    let mut cmd = Command::new(&consent.python);
    cmd.args(["-E", "-c", BOOT]).arg(Path::new(&consent.repo).join("src")).arg(dir)
        .current_dir(dir).stdin(Stdio::null()).stdout(log.try_clone()?).stderr(log);
    for name in DROPPED {
        cmd.env_remove(name);
    }
    proc::group_command(&mut cmd);
    cmd.spawn().map_err(|e| Error::Invalid(format!("python {} did not start: {e}", consent.python)))
}

/// A `Map` of the fields of ``req`` that a local caller may pass on (everything but `op` and `by`).
pub fn passable(args: &Map<String, Value>) -> Result<()> {
    match args.keys().find(|k| matches!(k.as_str(), "op" | "by")) {
        Some(k) => Err(Error::Denied(format!("`{k}` is not an argument: the node says which device and session ask"))),
        None => Ok(()),
    }
}
