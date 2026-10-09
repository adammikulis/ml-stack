mod kit;

use std::net::{IpAddr, UdpSocket};
use std::time::Duration;

use kit::netkit::*;
use poolside_node::beacon::{advertisable, decode, encode, Beacon};
use poolside_node::error::Error;
use poolside_node::net::NetConfig;
use poolside_node::poolops::wall_ms;
use serde_json::json;

fn sample(addr: &str) -> (Beacon, ed25519_dalek::SigningKey) {
    let key = kit::key(7);
    let b = Beacon {
        pool: "0123456789abcdef".into(), fingerprint: "a".repeat(64), public: key.verifying_key().to_bytes(),
        addr: addr.parse().unwrap(), port: 4000, ts: wall_ms(),
    };
    (b, key)
}

fn cfg_with(beacon: poolside_node::beacon::BeaconConfig) -> NetConfig {
    NetConfig { listen: "127.0.0.1:0".parse().unwrap(), beacon: Some(beacon), sync_every: None }
}

#[test]
fn a_beacon_round_trips_and_any_change_to_it_is_caught() {
    let (b, key) = sample("192.0.2.7");
    let bytes = encode(&b, &key);
    assert_eq!(decode(&bytes, wall_ms(), false).unwrap(), b);
    let text = String::from_utf8(bytes.clone()).unwrap();
    for (from, to) in [("\"port\":4000", "\"port\":4001"), ("192.0.2.7", "192.0.2.8"), ("0123456789abcdef", "0123456789abcdee")] {
        let forged = text.replace(from, to);
        assert_ne!(forged, text);
        assert!(decode(forged.as_bytes(), wall_ms(), false).is_err(), "changed {from}");
    }
    assert!(matches!(decode(&bytes, wall_ms() + 120_000, false), Err(Error::Denied(_))), "a stale beacon is not heard");
    let other = kit::key(8);
    let foreign = text.replace(&poolside_node::fsutil::hex(&b.public), &poolside_node::fsutil::hex(&other.verifying_key().to_bytes()));
    assert!(decode(foreign.as_bytes(), wall_ms(), false).is_err(), "a signature only verifies under its own key");
}

#[test]
fn loopback_unspecified_and_multicast_are_never_advertisable() {
    for ip in ["127.0.0.1", "::1", "0.0.0.0", "224.0.0.1"] {
        assert!(!advertisable(ip.parse::<IpAddr>().unwrap()), "{ip}");
    }
    assert!(advertisable("192.0.2.7".parse().unwrap()));
    let (b, key) = sample("127.0.0.1");
    assert!(decode(&encode(&b, &key), wall_ms(), false).is_err(), "a beacon that advertises loopback is not heard");
    assert!(decode(&encode(&b, &key), wall_ms(), true).is_ok(), "except in a test on one machine");
}

#[test]
fn a_node_does_not_send_a_loopback_address_unless_a_test_allows_it() {
    let port = udp_port();
    let listener = UdpSocket::bind(("127.0.0.1", port)).unwrap();
    listener.set_read_timeout(Some(Duration::from_millis(1500))).unwrap();
    let mut quiet = beacon(udp_port(), &[port]);
    quiet.allow_loopback = false;
    let _a = device_with(cfg_with(quiet));
    let mut buf = [0u8; 1024];
    assert!(listener.recv(&mut buf).is_err(), "a loopback address was advertised");
    let _b = device_with(cfg_with(beacon(udp_port(), &[port])));
    let n = listener.recv(&mut buf).expect("the control node beacons");
    let heard = decode(&buf[..n], wall_ms(), true).unwrap();
    assert_eq!(heard.addr.to_string(), "127.0.0.1");
}

#[test]
fn under_open_two_devices_on_one_segment_enrol_each_other_and_sync() {
    let (pa, pb) = (udp_port(), udp_port());
    let a = device_with(cfg_with(beacon(pa, &[pb])));
    let b = device_with(cfg_with(beacon(pb, &[pa])));
    let (_, at) = a.session("demo", "x");
    let (_, bt) = b.session("demo", "y");
    a.say("demo", &at, "from a");
    b.say("demo", &bt, "from b");
    a.set_policy("open");
    b.set_policy("open");
    eventually("the two devices to share a pool", 15, || a.pool() == b.pool() && a.is_member(&b.fp()) && b.is_member(&a.fp()));
    let by = [a.member(&b.fp()).unwrap()["by"].clone(), b.member(&a.fp()).unwrap()["by"].clone()];
    assert!(by.contains(&json!("open")), "enrolment is recorded as automatic: {by:?}");
    a.ok("sync_now", "", &a.token, json!({}));
    b.ok("sync_now", "", &b.token, json!({}));
    for (d, t) in [(&a, &at), (&b, &bt)] {
        let mut texts = d.texts("demo", t);
        texts.sort();
        assert_eq!(texts, vec!["from a", "from b"]);
    }
}

#[test]
fn under_secure_a_beacon_enrols_nobody() {
    let (pa, pb) = (udp_port(), udp_port());
    let a = device_with(cfg_with(beacon(pa, &[pb])));
    let b = device_with(cfg_with(beacon(pb, &[pa])));
    a.set_policy("open");
    std::thread::sleep(Duration::from_millis(1500));
    assert!(!a.is_member(&b.fp()) && !b.is_member(&a.fp()), "b is secure, so it enrols no one; a's open policy alone enrols only whom b names");
    assert_ne!(a.pool(), b.pool());
}

/// Send `a` a beacon that `z` signed, claiming pool `claimed`, from the real address of `z`.
fn forge_beacon(to_port: u16, z: &Dev, claimed: &str) {
    let key = z.node.lock().unwrap().key.clone();
    let b = Beacon {
        pool: claimed.into(), fingerprint: z.fp(), public: key.verifying_key().to_bytes(),
        addr: "127.0.0.1".parse().unwrap(), port: z.net.port, ts: wall_ms(),
    };
    let out = UdpSocket::bind("127.0.0.1:0").unwrap();
    for _ in 0..3 {
        out.send_to(&encode(&b, &key), ("127.0.0.1", to_port)).unwrap();
        std::thread::sleep(Duration::from_millis(50));
    }
}

#[test]
fn a_beacon_from_another_pool_id_is_ignored_and_only_the_devices_own_pool_gets_through() {
    let pa = udp_port();
    let a = device_with(cfg_with(beacon(pa, &[])));
    let z = device();
    a.set_policy("open");
    z.set_policy("open");
    forge_beacon(pa, &z, "ffffffffffffffff");
    std::thread::sleep(Duration::from_millis(1200));
    assert!(!a.is_member(&z.fp()) && !z.is_member(&a.fp()), "a beacon naming a pool this device would not join is ignored, even from an open device");
    assert_ne!(a.pool(), z.pool());
    forge_beacon(pa, &z, "0000000000000001");
    eventually("the control beacon (a smaller pool id, which a lone open device joins) to enrol", 10, || a.is_member(&z.fp()) && z.is_member(&a.fp()));
    assert_eq!(a.pool(), z.pool());
}

#[test]
fn a_device_with_members_answers_only_its_own_pool_and_a_secure_one_answers_nobody() {
    let pa = udp_port();
    let a = device_with(cfg_with(beacon(pa, &[])));
    let (member, z, y) = (device(), device(), device());
    a.pair_secure(&member);
    a.set_policy("open");
    z.set_policy("open");
    y.set_policy("open");
    forge_beacon(pa, &z, "0000000000000001");
    std::thread::sleep(Duration::from_millis(500));
    assert!(!a.is_member(&z.fp()), "a device that has members does not follow a smaller pool id");
    member.pair_secure(&z);
    assert_eq!(z.pool(), a.pool(), "z is in a's pool through another member, which a has not heard of yet");
    forge_beacon(pa, &z, &a.pool());
    eventually("a beacon of the pool's own id from an open device to be answered", 10, || a.is_member(&z.fp()) && z.is_member(&a.fp()));

    let pl = udp_port();
    let lone = device_with(cfg_with(beacon(pl, &[])));
    forge_beacon(pl, &y, &lone.pool());
    std::thread::sleep(Duration::from_millis(1200));
    assert!(!lone.is_member(&y.fp()), "this device's policy is secure, so a beacon enrols nobody");
}
