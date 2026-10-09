//! Discovery: one signed UDP beacon carrying the pool id, the device fingerprint, an address and
//! a port. Never matched by name.
//!
//! The beacon is signed with the device key (its public key rides along), so a datagram cannot
//! be altered or replayed after a minute; that proves it was not tampered with, not that its
//! sender is a member. A member's beacon is further checked against the certificate in the
//! pool record (the key must be that certificate's), and the connection it leads to is pinned
//! to the fingerprint it names. Loopback, unspecified and multicast addresses are never advertised.

use std::net::{IpAddr, Ipv4Addr, SocketAddr, UdpSocket};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use serde_json::{json, Value};
use socket2::{Domain, Protocol, Socket, Type};

use crate::error::{Error, Result};
use crate::fsutil::{hex, unhex};

/// The multicast group beacons go to unless told otherwise (administratively scoped).
pub const GROUP: Ipv4Addr = Ipv4Addr::new(239, 255, 116, 1);
pub const MAX_AGE_MS: u64 = 60_000;
const MAX_DATAGRAM: usize = 1024;
const SIGN_PREFIX: &[u8] = b"poolside-beacon/v1\0";

/// What a beacon says.
#[derive(Clone, Debug, PartialEq)]
pub struct Beacon {
    pub pool: String,
    pub fingerprint: String,
    pub public: [u8; 32],
    pub addr: IpAddr,
    pub port: u16,
    pub ts: u64,
}

/// Whether an address may be advertised to other machines.
pub fn advertisable(ip: IpAddr) -> bool {
    !(ip.is_loopback() || ip.is_unspecified() || ip.is_multicast())
}

fn signed_bytes(b: &Beacon) -> Vec<u8> {
    let mut out = SIGN_PREFIX.to_vec();
    out.extend_from_slice(format!("{}\0{}\0{}\0{}\0{}\0", b.pool, b.fingerprint, b.addr, b.port, b.ts).as_bytes());
    out.extend_from_slice(&b.public);
    out
}

/// The datagram for ``b``, signed with ``key`` (whose public key must be ``b.public``).
pub fn encode(b: &Beacon, key: &SigningKey) -> Vec<u8> {
    let sig = key.sign(&signed_bytes(b));
    json!({"v": 1, "pool": b.pool, "fp": b.fingerprint, "pub": hex(&b.public), "addr": b.addr.to_string(), "port": b.port, "ts": b.ts, "sig": hex(&sig.to_bytes())})
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
    let b = Beacon { pool: field(&v, "pool")?.into(), fingerprint: field(&v, "fp")?.into(), public, addr, port, ts };
    if !crate::cert::valid_fingerprint(&b.fingerprint) || b.pool.len() != 16 {
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
    /// The address advertised; found from the route to the network when None.
    pub advertise: Option<IpAddr>,
    pub interval: Duration,
    /// Tests on one machine advertise loopback; a real node never does.
    pub allow_loopback: bool,
}

impl BeaconConfig {
    /// Multicast on ``port``.
    pub fn multicast(port: u16) -> BeaconConfig {
        BeaconConfig {
            bind: SocketAddr::from((Ipv4Addr::UNSPECIFIED, port)), send_to: vec![SocketAddr::from((GROUP, port))],
            advertise: None, interval: Duration::from_secs(5), allow_loopback: false,
        }
    }
}

/// The address of this machine on the network its default route uses, or None.
pub fn local_address() -> Option<IpAddr> {
    let s = UdpSocket::bind("0.0.0.0:0").ok()?;
    s.connect("192.0.2.1:9").ok()?;
    s.local_addr().ok().map(|a| a.ip()).filter(|ip| advertisable(*ip))
}

/// A socket listening for beacons on ``bind``, joined to the group when ``bind`` is for multicast.
pub fn listener(bind: SocketAddr, join: Option<Ipv4Addr>) -> Result<UdpSocket> {
    let socket = Socket::new(Domain::for_address(bind), Type::DGRAM, Some(Protocol::UDP))?;
    socket.set_reuse_address(true)?;
    #[cfg(unix)]
    socket.set_reuse_port(true)?;
    socket.bind(&bind.into())?;
    if let Some(group) = join {
        socket.join_multicast_v4(&group, &Ipv4Addr::UNSPECIFIED)?;
    }
    let socket: UdpSocket = socket.into();
    socket.set_read_timeout(Some(Duration::from_millis(200)))?;
    Ok(socket)
}

/// What the beacon loops need from the node, asked afresh each time.
pub struct Hooks {
    /// `(pool id, fingerprint, listening port)` now.
    pub identity: Box<dyn Fn() -> (String, String, u16) + Send + Sync>,
    pub on_beacon: Box<dyn Fn(Beacon) + Send + Sync>,
}

/// Send a beacon every interval and hand every valid one heard to ``hooks.on_beacon``, until ``stop``.
pub fn start(cfg: BeaconConfig, key: SigningKey, hooks: Hooks, stop: Arc<AtomicBool>, now: fn() -> u64) -> Result<()> {
    let join = match cfg.bind.ip() {
        IpAddr::V4(_) if cfg.send_to.iter().any(|a| a.ip().is_multicast()) => Some(GROUP),
        _ => None,
    };
    let inbound = listener(cfg.bind, join)?;
    let hooks = Arc::new(hooks);
    let (h, s, c) = (hooks.clone(), stop.clone(), cfg.clone());
    std::thread::spawn(move || {
        let mut buf = [0u8; MAX_DATAGRAM + 1];
        while !s.load(Ordering::SeqCst) {
            if let Ok(n) = inbound.recv(&mut buf) {
                if let Ok(b) = decode(&buf[..n], now(), c.allow_loopback) {
                    (h.on_beacon)(b);
                }
            }
        }
    });
    let public = key.verifying_key().to_bytes();
    std::thread::spawn(move || {
        let Ok(out) = UdpSocket::bind("0.0.0.0:0") else { return };
        let _ = out.set_multicast_loop_v4(true);
        while !stop.load(Ordering::SeqCst) {
            let (pool, fingerprint, port) = (hooks.identity)();
            let addr = cfg.advertise.or_else(local_address);
            if let Some(addr) = addr.filter(|a| advertisable(*a) || cfg.allow_loopback) {
                let datagram = encode(&Beacon { pool, fingerprint, public, addr, port, ts: now() }, &key);
                for to in &cfg.send_to {
                    let _ = out.send_to(&datagram, to);
                }
            }
            let mut waited = Duration::ZERO;
            while waited < cfg.interval && !stop.load(Ordering::SeqCst) {
                std::thread::sleep(Duration::from_millis(50));
                waited += Duration::from_millis(50);
            }
        }
    });
    Ok(())
}
