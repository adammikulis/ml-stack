//! `poolside-node probe --seconds N`: one short check that this process may use the local network.
//!
//! macOS asks the person once for Local Network permission, per app identity, the first time a process joins a multicast
//! group or sends to one. The probe does exactly that with a datagram of its own on a port the node never uses: it
//! joins the beacon group, sends the datagram out of each real interface, and listens for it coming back. It stops a few seconds after
//! it first hears itself, or after N seconds, and prints one JSON line: what was sent, what failed, whether it heard itself.

use std::net::{Ipv4Addr, SocketAddr, UdpSocket};
use std::time::{Duration, Instant};

use serde_json::{json, Value};

use crate::beacon::{self, GROUP};
use crate::error::Result;
use crate::netif;

/// The port the probe uses (the node's beacon is on 7448).
pub const PORT: u16 = 7449;
const PREFIX: &[u8] = b"poolhouse-probe/v1\0";
/// How long the probe keeps sending after hearing itself, so a permission prompt macOS is still raising is not cut off.
const LINGER: Duration = Duration::from_secs(4);
const EVERY: Duration = Duration::from_millis(500);

/// The datagram that carries ``nonce``.
pub fn payload(nonce: &str) -> Vec<u8> {
    let mut out = PREFIX.to_vec();
    out.extend_from_slice(nonce.as_bytes());
    out
}

/// Whether ``datagram`` is the one ``payload(nonce)`` made.
pub fn is_mine(datagram: &[u8], nonce: &str) -> bool {
    datagram == payload(nonce).as_slice()
}

/// The JSON line the probe prints.
pub fn report(sent: u64, send_errors: u64, heard_self: bool, last_error: &str) -> Value {
    json!({"sent": sent, "send_errors": send_errors, "heard_self": heard_self, "last_error": last_error})
}

/// Run the probe for at most ``seconds``, also sending to each of ``peers`` (a router, say: a unicast to the local network is what
/// macOS refuses outright when permission is denied, where a multicast it may only drop); the report, or the error that stopped it joining the group.
pub fn run(seconds: u64, peers: &[Ipv4Addr]) -> Result<Value> {
    let mut raw = [0u8; 8];
    getrandom::fill(&mut raw).map_err(|e| std::io::Error::other(e.to_string()))?;
    let nonce = crate::fsutil::hex(&raw);
    let ifaces = netif::interfaces();
    let inbound = beacon::listener(SocketAddr::from((Ipv4Addr::UNSPECIFIED, PORT)), Some(GROUP), &ifaces)?;
    let out = UdpSocket::bind("0.0.0.0:0")?;
    out.set_multicast_loop_v4(true)?;
    out.set_broadcast(true)?;
    let datagram = payload(&nonce);
    let (mut sent, mut errors, mut last) = (0u64, 0u64, String::new());
    let end = Instant::now() + Duration::from_secs(seconds);
    let mut buf = [0u8; 256];
    let mut next = Instant::now();
    let (mut heard, linger) = (false, Instant::now() + LINGER);
    while Instant::now() < end {
        if Instant::now() >= next {
            let ways: Vec<Option<netif::Iface>> = if ifaces.is_empty() { vec![None] } else { ifaces.iter().copied().map(Some).collect() };
            for way in ways {
                if let Some(i) = way {
                    let _ = socket2::SockRef::from(&out).set_multicast_if_v4(&i.addr);
                }
                let mut to = vec![SocketAddr::from((GROUP, PORT))];
                to.extend(way.map(|i| SocketAddr::from((i.broadcast(), PORT))));
                to.extend(peers.iter().map(|p| SocketAddr::from((*p, PORT))));
                for target in to {
                    match out.send_to(&datagram, target) {
                        Ok(_) => sent += 1,
                        Err(e) => {
                            errors += 1;
                            last = format!("{target}: {e}");
                        }
                    }
                }
            }
            next = Instant::now() + EVERY;
        }
        if let Ok(n) = inbound.recv(&mut buf) {
            heard |= is_mine(&buf[..n], &nonce);
        }
        if heard && Instant::now() >= linger {
            break;
        }
    }
    Ok(report(sent, errors, heard, &last))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_its_own_nonce_is_mine() {
        assert!(is_mine(&payload("ab12"), "ab12"));
        assert!(!is_mine(&payload("ab12"), "ab13"));
        assert!(!is_mine(b"ab12", "ab12"));
    }

    #[test]
    fn the_report_names_what_happened() {
        let r = report(4, 1, false, "No route to host");
        assert_eq!((r["sent"].as_u64(), r["heard_self"].as_bool(), r["last_error"].as_str()), (Some(4), Some(false), Some("No route to host")));
    }
}
