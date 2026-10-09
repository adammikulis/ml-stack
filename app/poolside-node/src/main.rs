//! `poolside-node run --state DIR` serves the node; `status` talks to a running one.
//!
//! The network is off unless `--network` (every interface, the multicast beacon, ports `--port`
//! and `--beacon-port`) or `--listen ADDR` names where to listen. `--beacon-bind ADDR` turns
//! the beacon on, `--beacon-send ADDR` (repeatable) says where to send it (the multicast
//! group in production, a peer in tests), `--advertise IP` the address it carries,
//! `--allow-loopback` lets a test advertise loopback, `--sync-ms N` sets the sync interval.

use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Duration;

use poolside_node::beacon::BeaconConfig;
use poolside_node::client::Client;
use poolside_node::net::{Net, NetConfig};
use poolside_node::server::Server;
use serde_json::json;

/// The TCP port of the node's one listener and the UDP port of its beacon, unless told otherwise.
const DEFAULT_PORT: u16 = 47321;
const DEFAULT_BEACON_PORT: u16 = 47322;

fn state_dir(args: &[String]) -> PathBuf {
    match flag(args, "--state") {
        Some(dir) => PathBuf::from(dir),
        None => poolside_node::sys::home_dir().join(".poolside").join("node"),
    }
}

fn flag<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).map(String::as_str)
}

fn addr(text: &str) -> Result<SocketAddr, String> {
    text.parse::<SocketAddr>().map_err(|e| format!("{text}: {e}"))
}

fn beacon_config(args: &[String], bind: &str) -> Result<BeaconConfig, String> {
    let send_to = args.windows(2).filter(|w| w[0] == "--beacon-send").map(|w| addr(&w[1])).collect::<Result<Vec<_>, _>>()?;
    let advertise = flag(args, "--advertise").map(|a| a.parse::<IpAddr>().map_err(|e| e.to_string())).transpose()?;
    let interface = flag(args, "--multicast-if").map(|a| a.parse::<Ipv4Addr>().map_err(|e| format!("--multicast-if: {e}"))).transpose()?;
    Ok(BeaconConfig {
        bind: addr(bind)?, send_to, advertise, interval: Duration::from_millis(500), allow_loopback: args.iter().any(|a| a == "--allow-loopback"),
        interface, broadcast: false,
    })
}

fn number(args: &[String], name: &str, default: u64) -> Result<u64, String> {
    flag(args, name).map_or(Ok(default), |n| n.parse::<u64>().map_err(|e| format!("{name}: {e}")))
}

/// `--network`: listen on every interface and beacon on the multicast group, on the default ports.
fn standard_config(args: &[String]) -> Result<NetConfig, String> {
    let port = |name, default| u16::try_from(number(args, name, default)?).map_err(|_| format!("{name} is a port, 1 to 65535"));
    let mut config = NetConfig::standard(port("--port", DEFAULT_PORT.into())?, port("--beacon-port", DEFAULT_BEACON_PORT.into())?);
    if let Some(b) = config.beacon.as_mut() {
        b.interval = Duration::from_millis(number(args, "--beacon-ms", 5000)?);
        b.interface = flag(args, "--multicast-if").map(|a| a.parse::<Ipv4Addr>().map_err(|e| format!("--multicast-if: {e}"))).transpose()?;
    }
    config.sync_every = Some(Duration::from_millis(number(args, "--sync-ms", 20_000)?));
    Ok(config)
}

fn net_config(args: &[String]) -> Result<Option<NetConfig>, String> {
    if args.iter().any(|a| a == "--network") {
        return standard_config(args).map(Some);
    }
    let Some(listen) = flag(args, "--listen") else { return Ok(None) };
    let beacon = flag(args, "--beacon-bind").map(|b| beacon_config(args, b)).transpose()?;
    let sync_every = flag(args, "--sync-ms").map(|n| n.parse::<u64>().map(Duration::from_millis).map_err(|e| e.to_string())).transpose()?;
    Ok(Some(NetConfig { listen: addr(listen)?, beacon, sync_every }))
}

fn run(state: &Path, args: &[String]) -> Result<(), String> {
    let config = net_config(args)?;
    let mut server = Server::bind(state).map_err(|e| e.to_string())?;
    if let Some(config) = config {
        let net = Net::start(server.node(), config).map_err(|e| e.to_string())?;
        server = server.attach(net);
    }
    server.serve().map_err(|e| e.to_string())
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let state = state_dir(&args);
    let result = match args.first().map(String::as_str) {
        Some("run") => run(&state, &args).map(|_| String::new()),
        Some("status") => Client::connect(&state).and_then(|mut c| c.call("status", "", "", json!({}))).map(|v| v.to_string()).map_err(|e| e.to_string()),
        _ => {
            eprintln!("usage: poolside-node run|status [--state DIR] [--listen ADDR ...]");
            return ExitCode::from(2);
        }
    };
    match result {
        Ok(text) => {
            if !text.is_empty() {
                println!("{text}");
            }
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("poolside-node: {e}");
            ExitCode::FAILURE
        }
    }
}
