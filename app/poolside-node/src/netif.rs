//! This machine's IPv4 networks, and whether an address is on one of them.
//!
//! A beacon is sent and heard on every real interface (not only the one the default route uses,
//! which on a machine with a VPN, a container bridge or a virtual adapter is often the wrong one),
//! and policy `open` is offered only to a device on the same segment: an address inside the
//! subnet of one of these interfaces. Point-to-point links (a VPN) and loopback are not segments.
//! On unix the interfaces are read with `getifaddrs`; where that is not done the machine's
//! route address stands for the one interface, and "the same segment" falls back to the private
//! and link-local ranges.

use std::net::{IpAddr, Ipv4Addr};

/// One IPv4 interface that can carry a beacon.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Iface {
    pub addr: Ipv4Addr,
    pub mask: Ipv4Addr,
}

impl Iface {
    pub fn contains(&self, ip: Ipv4Addr) -> bool {
        u32::from(ip) & u32::from(self.mask) == u32::from(self.addr) & u32::from(self.mask)
    }

    /// The directed broadcast address of the subnet.
    pub fn broadcast(&self) -> Ipv4Addr {
        Ipv4Addr::from(u32::from(self.addr) | !u32::from(self.mask))
    }
}

#[cfg(unix)]
fn read_interfaces() -> Vec<Iface> {
    let mut head: *mut libc::ifaddrs = std::ptr::null_mut();
    // SAFETY: getifaddrs fills `head` with a list it owns until freeifaddrs; each node is read once, before the free.
    if unsafe { libc::getifaddrs(&mut head) } != 0 {
        return Vec::new();
    }
    let mut out = Vec::new();
    let mut at = head;
    while !at.is_null() {
        // SAFETY: `at` is a node of the list above.
        let (flags, addr, mask, next) = unsafe { ((*at).ifa_flags, (*at).ifa_addr, (*at).ifa_netmask, (*at).ifa_next) };
        let wanted = (libc::IFF_UP | libc::IFF_RUNNING) as u32;
        let barred = (libc::IFF_LOOPBACK | libc::IFF_POINTOPOINT) as u32;
        if !addr.is_null() && !mask.is_null() && flags as u32 & wanted == wanted && flags as u32 & barred == 0 {
            // SAFETY: both pointers are sockaddrs; they are read as sockaddr_in only after the family says so.
            unsafe {
                if i32::from((*addr).sa_family) == libc::AF_INET && i32::from((*mask).sa_family) == libc::AF_INET {
                    let (a, m) = (*(addr as *const libc::sockaddr_in), *(mask as *const libc::sockaddr_in));
                    out.push(Iface { addr: Ipv4Addr::from(u32::from_be(a.sin_addr.s_addr)), mask: Ipv4Addr::from(u32::from_be(m.sin_addr.s_addr)) });
                }
            }
        }
        at = next;
    }
    // SAFETY: `head` came from the successful getifaddrs above and nothing points into it any more.
    unsafe { libc::freeifaddrs(head) };
    out
}

#[cfg(not(unix))]
fn read_interfaces() -> Vec<Iface> {
    Vec::new()
}

/// The interfaces a beacon goes out on: every up, non-loopback, non-point-to-point IPv4 one.
pub fn interfaces() -> Vec<Iface> {
    read_interfaces()
}

fn private_range(ip: Ipv4Addr) -> bool {
    ip.is_private() || ip.is_link_local()
}

/// Whether ``ip`` is on the same segment as this machine: loopback (the same machine), or inside
/// the subnet of one of its interfaces.
pub fn on_segment(ip: IpAddr) -> bool {
    on_segment_of(ip, &interfaces())
}

/// `on_segment` against a given list of interfaces; the private and link-local ranges when there are none.
pub fn on_segment_of(ip: IpAddr, known: &[Iface]) -> bool {
    match ip {
        IpAddr::V4(v4) if v4.is_loopback() => true,
        IpAddr::V4(v4) if known.is_empty() => private_range(v4),
        IpAddr::V4(v4) => known.iter().any(|i| i.contains(v4)),
        IpAddr::V6(v6) => v6.is_loopback(),
    }
}
