//! The network side of the node: the one TLS listener, the beacon, the sync loop, and the local
//! API calls that reach other machines (`pair_accept`, `pair_start`, `sync_now`).
//!
//! The node's lock is never held across network I/O: a request takes it, decides, and lets go.

use std::collections::BTreeSet;
use std::net::{SocketAddr, TcpListener, ToSocketAddrs};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};
use std::time::Duration;

use rustls::ServerConfig;
use serde_json::{json, Value};

use crate::api::{parse, respond, Call};
use crate::beacon::{self, Beacon, BeaconConfig, BeaconStats, Hooks};
use crate::cert::{board_fingerprint, Identity};
use crate::error::{Error, Result};
use crate::fsutil::{sha256_hex, unhex};
use crate::membership::{Device, Policy, Standing, Status};
use crate::netsync::sync_all;
use crate::node::Node;
use crate::pairing::{context_for, new_code, Session, Window};
use crate::peer::{self, PeerClient};
use crate::poolapi::{authorize, NETWORK};
use crate::poolops::wall_ms;
use crate::tls::{server_config, StandingFn};

/// How this node reaches other machines.
#[derive(Clone, Debug)]
pub struct NetConfig {
    pub listen: SocketAddr,
    pub beacon: Option<BeaconConfig>,
    /// How often to swap pool records and boards with every member; None for only on request.
    pub sync_every: Option<Duration>,
}

impl NetConfig {
    /// Listen on every interface on ``port`` with the default beacon on ``beacon_port``.
    pub fn standard(port: u16, beacon_port: u16) -> NetConfig {
        NetConfig { listen: SocketAddr::from(([0, 0, 0, 0], port)), beacon: Some(BeaconConfig::multicast(beacon_port)), sync_every: Some(Duration::from_secs(20)) }
    }
}

pub struct Net {
    pub node: Arc<Mutex<Node>>,
    pub me: Identity,
    pub server_config: Arc<ServerConfig>,
    pub port: u16,
    pub stop: Arc<AtomicBool>,
    window: Mutex<Option<Window>>,
    joining: Mutex<BTreeSet<String>>,
    beacon_stats: OnceLock<Arc<BeaconStats>>,
    /// The last attempt to join a pool automatically, and how it ended.
    last_join: Mutex<Value>,
}

fn locked<T>(m: &Mutex<T>) -> Result<MutexGuard<'_, T>> {
    m.lock().map_err(|_| Error::Damaged("node poisoned".into()))
}

fn addr_of(host: &str, port: u64) -> Result<SocketAddr> {
    let port = u16::try_from(port).ok().filter(|p| *p > 0).ok_or_else(|| Error::Invalid("port is 1 to 65535".into()))?;
    (host, port).to_socket_addrs()?.next().ok_or_else(|| Error::Invalid("host does not resolve".into()))
}

/// ``base`` with the keys of ``more`` added (both are objects).
fn merged(mut base: Value, more: Value) -> Value {
    if let (Some(b), Value::Object(m)) = (base.as_object_mut(), more) {
        b.extend(m);
    }
    base
}

/// A join failure as a person should read it: what happened and what to do about it.
fn join_hint(e: &Error) -> String {
    let text = e.to_string();
    let blocked = matches!(e, Error::Io(io) if matches!(io.kind(), std::io::ErrorKind::TimedOut | std::io::ErrorKind::WouldBlock | std::io::ErrorKind::PermissionDenied | std::io::ErrorKind::ConnectionRefused));
    if blocked {
        return format!("{text}: the other device did not accept the connection; allow this program through its firewall on the private network (macOS: Privacy & Security > Local Network)");
    }
    text
}

impl Net {
    /// Start listening (and beaconing and syncing, as configured) for ``node``.
    pub fn start(node: Arc<Mutex<Node>>, cfg: NetConfig) -> Result<Arc<Net>> {
        let (me, stop) = {
            let n = locked(&node)?;
            (n.cert.clone(), n.stop.clone())
        };
        let standing_of = node.clone();
        let standing: StandingFn = Arc::new(move |fp| standing_of.lock().map_or(Standing::Revoked, |n| n.members.standing(fp)));
        let listener = TcpListener::bind(cfg.listen)?;
        listener.set_nonblocking(true)?;
        let port = listener.local_addr()?.port();
        let net = Arc::new(Net {
            node, server_config: server_config(&me, standing)?, me, port, stop,
            window: Mutex::new(None), joining: Mutex::new(BTreeSet::new()), beacon_stats: OnceLock::new(), last_join: Mutex::new(Value::Null),
        });
        locked(&net.node)?.facts.listen = Some(listener.local_addr()?.to_string());
        let n = net.clone();
        std::thread::spawn(move || n.accept_loop(listener));
        if let Some(b) = cfg.beacon {
            net.start_beacon(b)?;
        }
        if let Some(every) = cfg.sync_every {
            let n = net.clone();
            std::thread::spawn(move || n.sync_loop(every));
        }
        Ok(net)
    }

    pub fn shutdown(&self) {
        self.stop.store(true, Ordering::SeqCst);
    }

    fn accept_loop(self: Arc<Net>, listener: TcpListener) {
        while !self.stop.load(Ordering::SeqCst) {
            match listener.accept() {
                Ok((tcp, _)) => {
                    let _ = tcp.set_nonblocking(false);
                    let n = self.clone();
                    std::thread::spawn(move || {
                        let _ = peer::serve(&n, tcp);
                    });
                }
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => std::thread::sleep(Duration::from_millis(10)),
                Err(_) => std::thread::sleep(Duration::from_millis(100)),
            }
        }
    }

    pub fn session_opened(&self, fp: &str) {
        if let Ok(mut n) = self.node.lock() {
            if n.members.is_active(fp) {
                let f = n.facts.peer(fp);
                f.sessions += 1;
                f.last_seen_ms = wall_ms();
            }
        }
    }

    pub fn session_closed(&self, fp: &str) {
        if let Ok(mut n) = self.node.lock() {
            if let Some(f) = n.facts.peers.get_mut(fp) {
                f.sessions = f.sessions.saturating_sub(1);
            }
        }
    }

    // -- beacon -----------------------------------------------------------
    fn start_beacon(self: &Arc<Net>, cfg: BeaconConfig) -> Result<()> {
        let key = locked(&self.node)?.key.clone();
        let (a, b) = (self.clone(), self.clone());
        let hooks = Hooks {
            identity: Box::new(move || a.node.lock().map(|n| (n.members.id.clone(), n.cert.fingerprint(), a.port)).unwrap_or_default()),
            on_beacon: Box::new(move |beacon| b.heard(beacon)),
        };
        locked(&self.node)?.facts.beacon = true;
        let stats = beacon::start(cfg, key, hooks, self.stop.clone(), wall_ms)?;
        let _ = self.beacon_stats.set(stats);
        Ok(())
    }

    /// What the beacon has sent and heard, and how the last automatic join ended.
    fn network_facts(&self) -> Value {
        let count = |f: fn(&BeaconStats) -> &std::sync::atomic::AtomicU64| self.beacon_stats.get().map_or(0, |s| f(s).load(Ordering::Relaxed));
        let text = |f: fn(&BeaconStats) -> &Mutex<String>| self.beacon_stats.get().and_then(|s| f(s).lock().ok().map(|t| t.clone())).unwrap_or_default();
        json!({
            "beacon_sent": count(|s| &s.sent), "beacon_send_errors": count(|s| &s.send_errors), "beacon_heard": count(|s| &s.heard),
            "beacon_own": count(|s| &s.own), "beacon_rejected": count(|s| &s.rejected),
            "beacon_last_send_error": text(|s| &s.last_send_error), "beacon_last_reject": text(|s| &s.last_reject),
            "last_join": self.last_join.lock().map(|j| j.clone()).unwrap_or(Value::Null),
        })
    }

    fn record_join(&self, fingerprint: &str, addr: SocketAddr, result: &Result<()>) {
        let (ok, error) = match result {
            Ok(()) => (true, String::new()),
            Err(e) => (false, join_hint(e)),
        };
        if let Ok(mut j) = self.last_join.lock() {
            *j = json!({"fingerprint": fingerprint, "addr": addr.to_string(), "ok": ok, "error": error, "at": wall_ms()});
        }
    }

    /// A valid beacon was heard. A member's address is remembered; a non-member is answered
    /// only under policy `open`, and only by a device in the same pool or alone in a pool whose id
    /// sorts later (so two lone devices never join each other at once). Every other pool is ignored.
    fn heard(self: &Arc<Net>, b: Beacon) {
        let Ok(mut node) = self.node.lock() else { return };
        if b.fingerprint == node.cert.fingerprint() {
            return;
        }
        let addr = SocketAddr::new(b.addr, b.port);
        match node.members.standing(&b.fingerprint) {
            Standing::Revoked => {}
            Standing::Active => {
                let proven = node.members.get(&b.fingerprint).and_then(Device::der).and_then(|d| board_fingerprint(&d).ok()) == Some(sha256_hex(&b.public));
                if proven && b.pool == node.members.id {
                    let _ = node.facts.set_addr(&b.fingerprint, &addr.to_string());
                    node.facts.peer(&b.fingerprint).last_seen_ms = wall_ms();
                }
            }
            Standing::Unknown => {
                let eligible = node.members.policy == Policy::Open
                    && (b.pool == node.members.id || (node.members.alone() && b.pool < node.members.id))
                    && crate::netif::on_segment(b.addr);
                drop(node);
                if eligible && self.joining.lock().is_ok_and(|mut j| j.insert(b.fingerprint.clone())) {
                    let (net, fp) = (self.clone(), b.fingerprint.clone());
                    std::thread::spawn(move || {
                        let result = net.join_open(addr, &fp);
                        net.record_join(&fp, addr, &result);
                        if let Ok(mut j) = net.joining.lock() {
                            j.remove(&fp);
                        }
                    });
                }
            }
        }
    }

    // -- joining ----------------------------------------------------------
    /// Join the pool of the device at ``addr`` whose certificate has fingerprint ``expect``,
    /// under its policy `open`; the device is enrolled here and this one there.
    pub fn join_open(&self, addr: SocketAddr, expect: &str) -> Result<()> {
        if !crate::netif::on_segment(addr.ip()) {
            return Err(Error::Denied(format!("{} is not on a network this device is on; a device joins without a code only from the same network, so pair with a code", addr.ip())));
        }
        let mut c = PeerClient::connect(&self.me, addr, Some(expect))?;
        let name = locked(&self.node)?.members.get(&self.me.fingerprint()).map(|d| d.name.clone()).unwrap_or_default();
        let hello = c.call(&json!({"op": "hello"}))?;
        if hello["policy"] != "open" {
            return Err(Error::Denied("that pool does not take a device without a code".into()));
        }
        let reply = if hello["member"] == true {
            // That device already lists this one (another member let it in): swap records instead.
            let ask = {
                let n = locked(&self.node)?;
                json!({"op": "members", "pool": n.members.id, "rows": n.members.export(), "policy": peer::policy_json(&n)})
            };
            c.call(&ask)?
        } else {
            c.call(&json!({"op": "join_open", "name": name, "port": self.port}))?
        };
        self.adopt(&c, &reply, addr, "open")
    }

    /// Take the pool a peer's reply describes: its id, its record (which must list the
    /// certificate the handshake showed), its policy.
    fn adopt(&self, c: &PeerClient, reply: &Value, addr: SocketAddr, by: &str) -> Result<()> {
        let rows = reply["rows"].as_array().ok_or_else(|| Error::Invalid("rows is a list".into()))?;
        let host = rows.iter().filter_map(Device::read).find(|d| d.fingerprint == c.fingerprint && d.status == Status::Active);
        let host = host.ok_or_else(|| Error::Denied("the pool record does not list the device that sent it".into()))?;
        let mut node = locked(&self.node)?;
        node.members.adopt(reply["pool"].as_str().unwrap_or(""))?;
        node.enrol_device(&host.der().unwrap_or_default(), &host.name, by)?;
        node.members.merge(rows, &c.fingerprint)?;
        if let Some(p) = reply.get("policy") {
            peer::take_policy(&mut node, p)?;
        }
        node.facts.set_addr(&c.fingerprint, &addr.to_string())?;
        node.facts.peer(&c.fingerprint).last_seen_ms = wall_ms();
        Ok(())
    }

    // -- pairing ----------------------------------------------------------
    /// The server half of pairing, run under the node lock by `peer::dispatch`.
    pub fn pairing_step(&self, node: &mut Node, der: &[u8], op: &str, req: &Value) -> Result<Value> {
        let fp = crate::cert::cert_fingerprint(der);
        let mut window = locked(&self.window)?;
        let w = window.as_mut().filter(|w| w.open(wall_ms())).ok_or_else(|| Error::Denied("this device is not taking a pairing".into()))?;
        if op == "pair_exchange" {
            let nonce = req.get("nonce").and_then(Value::as_str).filter(|n| n.len() <= 64).ok_or_else(|| Error::Invalid("a nonce is sent".into()))?;
            let name = req.get("name").and_then(Value::as_str).unwrap_or("").to_string();
            let mut s = Session::start(false, &w.code, &context_for(nonce), &node.cert.fingerprint(), &fp);
            if let Err(e) = s.receive(req.get("message").and_then(Value::as_str).unwrap_or("")) {
                w.failed();
                return Err(e);
            }
            let message = s.message();
            w.session = Some((fp, name, s));
            return Ok(json!({"message": message}));
        }
        let Some((_, name, session)) = w.session.take().filter(|(who, _, _)| *who == fp) else {
            return Err(Error::Denied("no exchange is open for this device".into()));
        };
        if !session.check(req.get("confirmation").and_then(Value::as_str).unwrap_or("")) {
            let left = w.failed();
            if !left {
                *window = None;
                node.facts.pairing_until_ms = 0;
            }
            return Err(Error::Denied(if left { "wrong code".into() } else { "wrong code; the pairing is closed".into() }));
        }
        node.enrol_device(der, &name, "pairing")?;
        let grant = json!({"v": 1, "pool": node.members.id, "rows": node.members.export(), "policy": peer::policy_json(node)}).to_string().into_bytes();
        *window = None;
        node.facts.pairing_until_ms = 0;
        Ok(json!({"confirmation": session.confirmation()?, "grant": crate::fsutil::hex(&grant), "tag": session.seal(&grant)?}))
    }

    /// Open a pairing window: returns the code (``passphrase`` when given) and seconds it lasts.
    pub fn pair_accept(&self, passphrase: &str, ttl_s: u64) -> Result<Value> {
        let code = if passphrase.is_empty() { new_code()? } else { passphrase.to_string() };
        if code.len() < 6 || code.len() > 200 {
            return Err(Error::Invalid("a passphrase is 6 to 200 characters".into()));
        }
        let until = wall_ms() + ttl_s.clamp(1, 600) * 1000;
        let mut node = locked(&self.node)?;
        *locked(&self.window)? = Some(Window::new(&code, until));
        node.facts.pairing_until_ms = until;
        Ok(json!({"code": code, "expires_in_s": ttl_s.clamp(1, 600), "port": self.port, "fingerprint": self.me.fingerprint()}))
    }

    /// Pair with the device at ``addr`` using its code (without one, join its pool under policy
    /// `open`, which needs its certificate fingerprint); this device must be alone in its pool.
    pub fn pair_start(&self, addr: SocketAddr, passphrase: &str, fingerprint: &str) -> Result<Value> {
        let (alone, name) = {
            let n = locked(&self.node)?;
            (n.members.alone(), n.members.get(&self.me.fingerprint()).map(|d| d.name.clone()).unwrap_or_default())
        };
        if !alone {
            return Err(Error::Denied("a device that already has members does not change pool".into()));
        }
        if passphrase.is_empty() {
            if !crate::cert::valid_fingerprint(fingerprint) {
                return Err(Error::Invalid("without a passphrase the device is named by its fingerprint".into()));
            }
            self.join_open(addr, fingerprint)?;
            return Ok(json!({"pool": locked(&self.node)?.members.id, "peer": fingerprint}));
        }
        let mut c = PeerClient::connect(&self.me, addr, None)?;
        let nonce = crate::fsutil::random_hex(16)?;
        let mut s = Session::start(true, passphrase, &context_for(&nonce), &self.me.fingerprint(), &c.fingerprint);
        let reply = c.call(&json!({"op": "pair_exchange", "message": s.message(), "nonce": nonce, "name": name}))?;
        s.receive(reply["message"].as_str().unwrap_or(""))?;
        let reply = c.call(&json!({"op": "pair_confirm", "confirmation": s.confirmation()?, "port": self.port}))?;
        if !s.check(reply["confirmation"].as_str().unwrap_or("")) {
            return Err(Error::Denied("the device did not prove it knew the code".into()));
        }
        let grant = reply["grant"].as_str().and_then(unhex).ok_or_else(|| Error::Invalid("the grant is hex".into()))?;
        if !s.open(&grant, reply["tag"].as_str().unwrap_or("")) {
            return Err(Error::Denied("the grant does not carry the exchange's tag".into()));
        }
        let grant: Value = serde_json::from_slice(&grant)?;
        self.adopt(&c, &grant, addr, "pairing")?;
        Ok(json!({"pool": grant["pool"], "peer": c.fingerprint}))
    }

    // -- sync -------------------------------------------------------------
    /// Swap pool records and boards with every active member we can reach; the number reached.
    pub fn sync_once(&self) -> Result<usize> {
        let targets: Vec<(String, String)> = {
            let n = locked(&self.node)?;
            let me = self.me.fingerprint();
            n.members.active().into_iter().filter(|d| d.fingerprint != me)
                .filter_map(|d| n.facts.peers.get(&d.fingerprint).and_then(|f| f.addr.clone()).map(|a| (d.fingerprint, a))).collect()
        };
        let mut reached = 0;
        for (fp, addr) in targets {
            let result = addr.parse::<SocketAddr>().map_err(|e| Error::Invalid(e.to_string()))
                .and_then(|a| PeerClient::connect(&self.me, a, Some(&fp)))
                .and_then(|mut c| sync_all(&self.node, &mut c));
            let mut n = locked(&self.node)?;
            let f = n.facts.peer(&fp);
            match result {
                Ok(_) => {
                    (f.last_sync_ms, f.last_seen_ms, f.last_error) = (wall_ms(), wall_ms(), String::new());
                    reached += 1;
                }
                Err(e) => f.last_error = e.to_string(),
            }
        }
        Ok(reached)
    }

    fn sync_loop(self: Arc<Net>, every: Duration) {
        while !self.stop.load(Ordering::SeqCst) {
            let _ = self.sync_once();
            let mut waited = Duration::ZERO;
            while waited < every && !self.stop.load(Ordering::SeqCst) {
                std::thread::sleep(Duration::from_millis(50));
                waited += Duration::from_millis(50);
            }
        }
    }

    // -- the local API ----------------------------------------------------
    /// Answer a local API request: the network methods here, the rest by the node under its lock.
    pub fn call(&self, request: &Value) -> Value {
        let parsed = parse(request);
        match parsed {
            Ok(c) if NETWORK.contains(&c.method.as_str()) => respond(request, self.network(&c)),
            Ok(c) if c.method == "pool_status" => match self.node.lock() {
                Ok(n) => respond(request, Ok(merged(n.pool_status(), self.network_facts()))),
                Err(_) => respond(request, Err(Error::Damaged("node poisoned".into()))),
            },
            _ => match self.node.lock() {
                Ok(mut n) => crate::api::handle(&mut n, request),
                Err(_) => respond(request, Err(Error::Damaged("node poisoned".into()))),
            },
        }
    }

    fn network(&self, c: &Call) -> Result<Value> {
        let text = |k: &str| c.params.get(k).and_then(Value::as_str).unwrap_or("").to_string();
        let number = |k: &str, d: u64| c.params.get(k).and_then(Value::as_u64).unwrap_or(d);
        {
            let node = locked(&self.node)?;
            authorize(&node, &c.token, &c.method)?;
        }
        match c.method.as_str() {
            "pair_accept" => self.pair_accept(&text("passphrase"), number("ttl_s", 120)),
            "pair_start" => self.pair_start(addr_of(&text("host"), number("port", 0))?, &text("passphrase"), &text("fingerprint")),
            _ => Ok(json!({"reached": self.sync_once()?})),
        }
    }
}
