//! A device with a project finds its pool on its own: no key file, no code. It listens for an
//! `open` pool of its project, joins the one it hears, and makes one when it hears none.

mod kit;

use std::net::UdpSocket;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use kit::netkit::*;
use poolhouse_node::beacon::{decode, encode, Beacon};
use poolhouse_node::net::{Net, NetConfig};
use poolhouse_node::node::Node;
use poolhouse_node::poolops::wall_ms;
use poolhouse_node::projectid::project_key;

fn auto(beacon_cfg: poolhouse_node::beacon::BeaconConfig, project: &str, settle_ms: u64) -> NetConfig {
    NetConfig {
        listen: "127.0.0.1:0".parse().unwrap(), beacon: Some(beacon_cfg), sync_every: None,
        project: Some(project_key(project)), settle: Duration::from_millis(settle_ms),
    }
}

/// A device whose pool id sorts above `floor`, so the id order cannot be what joins it.
fn device_above(floor: &str, cfg: NetConfig) -> Dev {
    loop {
        let dir = tempfile::tempdir().unwrap();
        let node = Node::open(dir.path()).unwrap();
        if node.members.id.as_str() <= floor {
            continue;
        }
        let node = Arc::new(Mutex::new(node));
        let net = Net::start(node.clone(), cfg).unwrap();
        let mut d = Dev { dir, node, net, token: String::new() };
        d.token = d.session("demo", "main").1;
        return d;
    }
}

fn policy(d: &Dev) -> String {
    d.status()["policy"].as_str().unwrap().to_string()
}

fn project(d: &Dev) -> String {
    d.status()["project"].as_str().unwrap().to_string()
}

#[test]
fn alone_it_listens_quietly_then_makes_an_open_pool_for_the_project_and_beacons_it() {
    let heard = UdpSocket::bind("127.0.0.1:0").unwrap();
    let port = heard.local_addr().unwrap().port();
    heard.set_read_timeout(Some(Duration::from_millis(700))).unwrap();
    let d = device_with(auto(beacon(udp_port(), &[port]), "poolhouse", 2500));
    assert_eq!(project(&d), project_key("poolhouse"), "the project is the device's from the start");
    assert_eq!(policy(&d), "secure", "it has made nothing open while it listens");
    let mut buf = [0u8; 1024];
    assert!(heard.recv(&mut buf).is_err(), "a device that is still listening says nothing");
    eventually("the device to make its pool open", 10, || policy(&d) == "open");
    heard.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
    let n = heard.recv(&mut buf).expect("then it beacons");
    let b = decode(&buf[..n], wall_ms(), true).unwrap();
    assert_eq!((b.project, b.pool, b.lone), (project_key("poolhouse"), d.pool(), true));
}

#[test]
fn it_joins_an_open_pool_of_its_project_that_it_hears_instead_of_making_one() {
    let (pa, pb) = (udp_port(), udp_port());
    let a = device_with(auto(beacon(pa, &[pb]), "poolhouse", 0));
    eventually("a to open its pool", 10, || policy(&a) == "open");
    // a has a member, so it is not alone; b's id sorts above a's, so only "join a pool that is not alone" can bring b in
    let member = device();
    a.pair_secure(&member);
    let b = device_above(&a.pool(), auto(beacon(pb, &[pa]), "poolhouse", 60_000));
    eventually("b to join a's pool well inside its listening window", 10, || a.is_member(&b.fp()) && b.is_member(&a.fp()));
    assert_eq!(b.pool(), a.pool());
    assert_eq!(project(&b), project_key("poolhouse"));
    assert_eq!(a.member(&b.fp()).unwrap()["by"], "open", "it came in with no code");
}

#[test]
fn a_pool_of_another_project_is_ignored_and_the_device_makes_its_own() {
    let (pa, pb) = (udp_port(), udp_port());
    let a = device_with(auto(beacon(pa, &[pb]), "one", 0));
    let b = device_with(auto(beacon(pb, &[pa]), "two", 600));
    eventually("both to make an open pool", 10, || policy(&a) == "open" && policy(&b) == "open");
    std::thread::sleep(Duration::from_millis(2500));
    assert_ne!(a.pool(), b.pool());
    assert!(!a.is_member(&b.fp()) && !b.is_member(&a.fp()), "two projects share no pool");
    assert_eq!((project(&a), project(&b)), (project_key("one"), project_key("two")));
}

#[test]
fn a_secure_pool_of_its_project_is_ignored_and_the_device_makes_its_own() {
    let (pa, pb) = (udp_port(), udp_port());
    let a = device_with(auto(beacon(pa, &[pb]), "poolhouse", 0));
    eventually("a to open its pool", 10, || policy(&a) == "open");
    let member = device();
    a.pair_secure(&member);
    a.set_policy("secure");
    let b = device_above(&a.pool(), auto(beacon(pb, &[pa]), "poolhouse", 600));
    eventually("b to make its own open pool", 10, || policy(&b) == "open");
    std::thread::sleep(Duration::from_millis(2500));
    assert_ne!(a.pool(), b.pool());
    assert!(!a.is_member(&b.fp()) && !b.is_member(&a.fp()), "a secure pool takes a device only through a code");
}

#[test]
fn two_devices_that_start_together_end_in_one_pool() {
    let (pa, pb) = (udp_port(), udp_port());
    let a = device_with(auto(beacon(pa, &[pb]), "poolhouse", 1500));
    let b = device_with(auto(beacon(pb, &[pa]), "poolhouse", 1500));
    let lower = a.pool().min(b.pool());
    eventually("the two to share one pool", 20, || a.pool() == b.pool() && a.is_member(&b.fp()) && b.is_member(&a.fp()));
    assert_eq!(a.pool(), lower, "the pool with the smaller id is the one that stays");
    assert_eq!(policy(&a), "open");
}

#[test]
fn a_beacon_carries_its_project_and_whether_the_sender_is_alone_and_signs_both() {
    let key = kit::key(7);
    let b = Beacon {
        pool: "0123456789abcdef".into(), fingerprint: "a".repeat(64), public: key.verifying_key().to_bytes(),
        addr: "192.0.2.7".parse().unwrap(), port: 4000, ts: wall_ms(), project: project_key("poolhouse"), lone: false,
    };
    let bytes = encode(&b, &key);
    assert_eq!(decode(&bytes, wall_ms(), false).unwrap(), b);
    let text = String::from_utf8(bytes).unwrap();
    for (from, to) in [(b.project.as_str(), project_key("other")), ("\"lone\":false", "\"lone\":true".to_string())] {
        let forged = text.replace(from, &to);
        assert_ne!(forged, text);
        assert!(decode(forged.as_bytes(), wall_ms(), false).is_err(), "changed {from}");
    }
    let bad = Beacon { project: "not a key".into(), ..b };
    assert!(decode(&encode(&bad, &key), wall_ms(), false).is_err(), "a project is 16 hex digits");
}
