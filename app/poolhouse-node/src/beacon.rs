//! Discovery: one signed UDP beacon carrying the pool id, the device fingerprint, an address and
//! a port. Never matched by name.
//!
//! The beacon is signed with the device key (its public key rides along), so a datagram cannot
//! be altered or replayed after a minute; that proves it was not tampered with, not that its
//! sender is a member. A member's beacon is further checked against the certificate in the
//! pool record (the key must be that certificate's), and the connection it leads to is pinned
//! to the fingerprint it names. Loopback, unspecified and multicast addresses are never advertised.
//!
//! It goes out on every real interface (`netif`), to the multicast group and to the broadcast
//! address of each network, because either may be dropped by a given network, and it advertises
//! the address of the interface it leaves by.

use std::net::{IpAddr, Ipv4Addr, SocketAddr, UdpSocket};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use serde_json::{json, Value};
use socket2::{Domain, Protocol, Socket, Type};

use crate::error::{Error, Result};
use crate::fsutil::{hex, unhex};
use crate::netif::{self, Iface};

/// The multicast group beacons go to unless told otherwise (administratively scoped).
pub const GROUP: Ipv4Addr = Ipv4Addr::new(239, 255, 116, 1);
pub const MAX_AGE_MS: u64 = 60_000;
const MAX_DATAGRAM: usize = 1024;
const SIGN_PREFIX: &[u8] = b"poolhouse-beacon/v1\0";

/// What a beacon says.
#[derive(Clone, Debug, PartialEq)]
pub struct Beacon {
    pub pool: String,
    pub fingerprint: String,
    pub public: [u8; 32],
    pub addr: IpAddr,
    pub port: u16,
    pub ts: u64,
    /// The project key of the sender's pool (`projectid::project_key`), or empty for none.
    pub project: String,
    /// Whether the sender is alone in its pool.
    pub lone: bool,
}

/// Whether an address may be advertised to other machines.
pub fn advertisable(ip: IpAddr) -> bool {
    !(ip.is_loopback() || ip.is_unspecified() || ip.is_multicast())
}

fn signed_bytes(b: &Beacon) -> Vec<u8> {
    let mut out = SIGN_PREFIX.to_vec();
    out.extend_from_slice(format!("{}\0{}\0{}\0{}\0{}\0{}\0{}\0", b.pool, b.fingerprint, b.addr, b.port, b.ts, b.project, b.lone).as_bytes());
    out.extend_from_slice(&b.public);
    out
}

/// The datagram for ``b``, signed with ``key`` (whose public key must be ``b.public``).
pub fn encode(b: &Beacon, key: &SigningKey) -> Vec<u8> {
    let sig = key.sign(&signed_bytes(b));
    json!({"v": 1, "pool": b.pool, "fp": b.fingerprint, "pub": hex(&b.public), "addr": b.addr.to_string(), "port": b.port, "ts": b.ts, "project": b.project, "lone": b.lone, "sig": hex(&sig.to_bytes())})
        .to_string().into_bytes()
}

fn field<'a>(v: &'a Value, key: &str) -> Result<&'a str> {
    v.get(key).and_then(Value::as_str).ok_or_else(|| Error::Invalid(format!("beacon lacks {key}")))
}

/// Read and verify a datagram: its signature under the key it carries, its age against ``now_ms``,
/// and an advertisable address (``allow_loopback`` is for tests on one machine).
pub fn decode(bytes: &[u8], now_ms: u64, allow_loopback: bool) -> Result<Beacon> {
    if bytes.len() > MAX_DATAGRAM {
        return Err(Error::Invalid("beacon too large".into()));
    }
    let v: Value = serde_json::from_slice(bytes)?;
    if v.get("v").and_then(Value::as_u64) != Some(1) {
        return Err(Error::Invalid("beacon version".into()));
    }
    let public: [u8; 32] = unhex(field(&v, "pub")?).and_then(|b| b.try_into().ok()).ok_or_else(|| Error::Invalid("beacon key".into()))?;
    let addr: IpAddr = field(&v, "addr")?.parse().map_err(|_| Error::Invalid("beacon address".into()))?;
    let port = v.get("port").and_then(Value::as_u64).filter(|p| (1..65536).contains(p)).ok_or_else(|| Error::Invalid("beacon port".into()))? as u16;
    let ts = v.get("ts").and_then(Value::as_u64).ok_or_else(|| Error::Invalid("beacon time".into()))?;
    let project = v.get("project").and_then(Value::as_str).unwrap_or("").to_string();
    let lone = v.get("lone").and_then(Value::as_bool).unwrap_or(false);
    let b = Beacon { pool: field(&v, "pool")?.into(), fingerprint: field(&v, "fp")?.into(), public, addr, port, ts, project, lone };
    if !crate::cert::valid_fingerprint(&b.fingerprint) || b.pool.len() != 16 || !(b.project.is_empty() || crate::projectid::valid_key(&b.project)) {
        return Err(Error::Invalid("beacon names".into()));
    }
    if !(advertisable(addr) || allow_loopback) {
        return Err(Error::Denied("a beacon does not advertise this address".into()));
    }
    if now_ms.abs_diff(ts) > MAX_AGE_MS {
        return Err(Error::Denied("beacon is stale".into()));
    }
    let sig: [u8; 64] = unhex(field(&v, "sig")?).and_then(|b| b.try_into().ok()).ok_or_else(|| Error::Invalid("beacon signature".into()))?;
    let key = VerifyingKey::from_bytes(&public).map_err(|_| Error::Invalid("beacon key".into()))?;
    key.verify(&signed_bytes(&b), &Signature::from_bytes(&sig)).map_err(|_| Error::Denied("beacon signature does not verify".into()))?;
    Ok(b)
}

/// Where beacons are sent and heard.
#[derive(Clone, Debug)]
pub struct BeaconConfig {
    /// The local address and port to listen on.
    pub bind: SocketAddr,
    /// Where each beacon is sent: the multicast group by default; tests give unicast peers.
    pub send_to: Vec<SocketAddr>,
    /// The address advertised; the address of the interface a beacon leaves by when None.
    pub advertise: Option<IpAddr>,
    pub interval: Duration,
    /// Tests on one machine advertise loopback; a real node never does.
    pub allow_loopback: bool,
    /// Send and hear on this one interface only (its address), not on every real one.
    pub interface: Option<Ipv4Addr>,
    /// Also send to the broadcast address of each network, for the ones that drop multicast.
    pub broadcast: bool,
}

impl BeaconConfig {
    /// Multicast and broadcast on ``port``, on every real interface.
    pub fn multicast(port: u16) -> BeaconConfig {
        BeaconConfig {
            bind: SocketAddr::from((Ipv4Addr::UNSPECIFIED, port)), send_to: vec![SocketAddr::from((GROUP, port))],
            advertise: None, interval: Duration::from_secs(5), allow_loopback: false, interface: None, broadcast: true,
        }
    }

    fn multicasts(&self) -> bool {
        self.bind.is_ipv4() && self.send_to.iter().any(|a| a.ip().is_multicast())
    }

    /// The interfaces this node sends and listens on; none when it should leave that to the system.
    fn interfaces(&self) -> Vec<Iface> {
        match self.interface {
            Some(addr) => vec![Iface { addr, mask: Ipv4Addr::BROADCAST }],
            None => netif::interfaces(),
        }
    }
}

/// What the beacon has done so far; a node that sends but never hears its own beacon is not
/// receiving (a firewall, a privacy setting, a network that drops multicast).
#[derive(Debug, Default)]
pub struct BeaconStats {
    pub sent: AtomicU64,
    pub send_errors: AtomicU64,
    pub heard: AtomicU64,
    pub own: AtomicU64,
    pub rejected: AtomicU64,
    pub last_send_error: Mutex<String>,
    pub last_reject: Mutex<String>,
}

/// The address of this machine on the network its default route uses, or None.
pub fn local_address() -> Option<IpAddr> {
    let s = UdpSocket::bind("0.0.0.0:0").ok()?;
    s.connect("192.0.2.1:9").ok()?;
    s.local_addr().ok().map(|a| a.ip()).filter(|ip| advertisable(*ip))
}

/// A socket listening for beacons on ``bind``, joined to the group ``join`` on each of ``on``
/// (the system's choice of interface when ``on`` is empty).
pub fn listener(bind: SocketAddr, join: Option<Ipv4Addr>, on: &[Iface]) -> Result<UdpSocket> {
    let socket = Socket::new(Domain::for_address(bind), Type::DGRAM, Some(Protocol::UDP))?;
    socket.set_reuse_address(true)?;
    #[cfg(unix)]
    socket.set_reuse_port(true)?;
    socket.bind(&bind.into())?;
    if let Some(group) = join {
        join_all(&socket, group, on)?;
    }
    let socket: UdpSocket = socket.into();
    socket.set_read_timeout(Some(Duration::from_millis(200)))?;
    Ok(socket)
}

/// Join ``group`` on every interface that will take it; an error only when none does.
fn join_all(socket: &Socket, group: Ipv4Addr, on: &[Iface]) -> Result<()> {
    if on.is_empty() {
        return Ok(socket.join_multicast_v4(&group, &Ipv4Addr::UNSPECIFIED)?);
    }
    let mut last = None;
    let mut joined = false;
    for i in on {
        match socket.join_multicast_v4(&group, &i.addr) {
            Ok(()) => joined = true,
            Err(e) => last = Some(e),
        }
    }
    match (joined, last) {
        (false, Some(e)) => Err(Error::Io(e)),
        _ => Ok(()),
    }
}

/// What a node says of itself in a beacon, asked afresh each time. An empty ``pool`` says nothing
/// (a node still listening for a pool to join stays quiet).
#[derive(Clone, Debug, Default)]
pub struct Advert {
    pub pool: String,
    pub fingerprint: String,
    pub port: u16,
    pub project: String,
    pub lone: bool,
}

/// What the beacon loops need from the node.
pub struct Hooks {
    pub identity: Box<dyn Fn() -> Advert + Send + Sync>,
    pub on_beacon: Box<dyn Fn(Beacon) + Send + Sync>,
}

fn note(slot: &Mutex<String>, text: String) {
    if let Ok(mut s) = slot.lock() {
        *s = text;
    }
}

fn hear(inbound: &UdpSocket, cfg: &BeaconConfig, hooks: &Hooks, stats: &BeaconStats, now: fn() -> u64, buf: &mut [u8]) {
    let Ok(n) = inbound.recv(buf) else { return };
    match decode(&buf[..n], now(), cfg.allow_loopback) {
        Ok(b) if b.fingerprint == (hooks.identity)().fingerprint => {
            stats.own.fetch_add(1, Ordering::Relaxed);
        }
        Ok(b) => {
            stats.heard.fetch_add(1, Ordering::Relaxed);
            (hooks.on_beacon)(b);
        }
        Err(e) => {
            stats.rejected.fetch_add(1, Ordering::Relaxed);
            note(&stats.last_reject, e.to_string());
        }
    }
}

/// One beacon for each way out: each interface (advertising its own address) to the group, to
/// the broadcast address of its network, and to the explicit targets.
fn send_round(out: &UdpSocket, cfg: &BeaconConfig, ifaces: &[Iface], key: &SigningKey, hooks: &Hooks, stats: &BeaconStats, now: fn() -> u64) {
    let Advert { pool, fingerprint, port, project, lone } = (hooks.identity)();
    if pool.is_empty() {
        return;
    }
    let public = key.verifying_key().to_bytes();
    let route = local_address();
    let ways: Vec<Option<Iface>> = if ifaces.is_empty() { vec![None] } else { ifaces.iter().copied().map(Some).collect() };
    for way in ways {
        let from = cfg.advertise.or(way.map(|i| IpAddr::V4(i.addr))).or(route);
        let Some(addr) = from.filter(|a| advertisable(*a) || cfg.allow_loopback) else { continue };
        let datagram = encode(&Beacon { pool: pool.clone(), fingerprint: fingerprint.clone(), public, addr, port, ts: now(), project: project.clone(), lone }, key);
        if let Some(i) = way {
            let _ = socket2::SockRef::from(out).set_multicast_if_v4(&i.addr);
        }
        let mut to: Vec<SocketAddr> = cfg.send_to.clone();
        if cfg.broadcast {
            to.extend(way.map(|i| SocketAddr::from((i.broadcast(), cfg.bind.port()))));
            to.push(SocketAddr::from((Ipv4Addr::BROADCAST, cfg.bind.port())));
        }
        for target in to {
            match out.send_to(&datagram, target) {
                Ok(_) => stats.sent.fetch_add(1, Ordering::Relaxed),
                Err(e) => {
                    note(&stats.last_send_error, format!("{target}: {e}"));
                    stats.send_errors.fetch_add(1, Ordering::Relaxed)
                }
            };
        }
    }
}

/// Send a beacon every interval and hand every valid one heard to ``hooks.on_beacon``, until ``stop``.
pub fn start(cfg: BeaconConfig, key: SigningKey, hooks: Hooks, stop: Arc<AtomicBool>, now: fn() -> u64) -> Result<Arc<BeaconStats>> {
    let ifaces = cfg.interfaces();
    let inbound = listener(cfg.bind, cfg.multicasts().then_some(GROUP), &ifaces)?;
    let hooks = Arc::new(hooks);
    let stats = Arc::new(BeaconStats::default());
    let (h, s, c, st) = (hooks.clone(), stop.clone(), cfg.clone(), stats.clone());
    std::thread::spawn(move || {
        let mut buf = [0u8; MAX_DATAGRAM + 1];
        while !s.load(Ordering::SeqCst) {
            hear(&inbound, &c, &h, &st, now, &mut buf);
        }
    });
    let st = stats.clone();
    std::thread::spawn(move || {
        let Ok(out) = UdpSocket::bind("0.0.0.0:0") else { return };
        let _ = out.set_multicast_loop_v4(true);
        let _ = out.set_broadcast(true);
        while !stop.load(Ordering::SeqCst) {
            send_round(&out, &cfg, &ifaces, &key, &hooks, &st, now);
            let mut waited = Duration::ZERO;
            while waited < cfg.interval && !stop.load(Ordering::SeqCst) {
                std::thread::sleep(Duration::from_millis(50));
                waited += Duration::from_millis(50);
            }
        }
    });
    Ok(stats)
}
