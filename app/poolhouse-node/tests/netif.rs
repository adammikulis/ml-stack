use std::net::{IpAddr, Ipv4Addr};

use poolhouse_node::netif::{interfaces, on_segment_of, Iface};

fn lan() -> Iface {
    Iface { addr: Ipv4Addr::new(192, 168, 2, 27), mask: Ipv4Addr::new(255, 255, 255, 0) }
}

#[test]
fn an_interface_knows_its_subnet_and_its_broadcast_address() {
    let i = lan();
    assert!(i.contains(Ipv4Addr::new(192, 168, 2, 200)) && !i.contains(Ipv4Addr::new(192, 168, 3, 1)));
    assert_eq!(i.broadcast(), Ipv4Addr::new(192, 168, 2, 255));
}

#[test]
fn the_same_segment_is_loopback_or_a_subnet_of_this_machine() {
    let known = [lan(), Iface { addr: Ipv4Addr::new(10, 0, 0, 5), mask: Ipv4Addr::new(255, 255, 0, 0) }];
    let on = |s: &str| on_segment_of(s.parse::<IpAddr>().unwrap(), &known);
    assert!(on("127.0.0.1") && on("192.168.2.9") && on("10.0.77.1"));
    assert!(!on("192.168.9.9"), "a private address of another network is not this segment");
    assert!(!on("8.8.8.8") && !on("100.64.0.1"), "a public address, or one across a VPN, is not this segment");
    assert!(!on("fe80::1") && !on("2001:db8::1"));
}

#[test]
fn without_a_list_of_interfaces_only_private_and_link_local_ranges_count() {
    let on = |s: &str| on_segment_of(s.parse::<IpAddr>().unwrap(), &[]);
    assert!(on("192.168.2.9") && on("10.1.2.3") && on("172.16.0.4") && on("169.254.1.1") && on("127.0.0.1"));
    assert!(!on("8.8.8.8") && !on("100.64.0.1") && !on("172.32.0.1"));
}

#[test]
fn the_machines_interfaces_are_real_networks_only() {
    for i in interfaces() {
        assert!(!i.addr.is_loopback() && !i.addr.is_unspecified() && !i.addr.is_multicast(), "{i:?}");
        assert!(i.contains(i.addr), "{i:?} is inside its own subnet");
    }
}
