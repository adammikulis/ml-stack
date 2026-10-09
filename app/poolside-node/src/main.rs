//! `poolside-node run --state DIR` serves the node; `status` talks to a running one.
//!
//! The network is off unless `--listen ADDR` names where to listen. `--beacon-bind ADDR` turns
//! the beacon on, `--beacon-send ADDR` (repeatable) says where to send it (the multicast
//! group in production, a peer in tests), `--advertise IP` the address it carries,
//! `--allow-loopback` lets a test advertise loopback, `--sync-ms N` sets the sync interval.

use std::net::{IpAddr, SocketAddr};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Duration;

use poolside_node::beacon::BeaconConfig;
use poolside_node::client::Client;
use poolside_node::net::{Net, NetConfig};
use poolside_node::server::Server;
use serde_json::json;

fn state_dir(args: &[String]) -> PathBuf {
    match flag(args, "--state") {
        Some(dir) => PathBuf::from(dir),
        None => PathBuf::from(std::env::var_os("HOME").unwrap_or_default()).join(".poolside").join("node"),
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
    Ok(BeaconConfig { bind: addr(bind)?, send_to, advertise, interval: Duration::from_millis(500), allow_loopback: args.iter().any(|a| a == "--allow-loopback") })
}

fn net_config(args: &[String]) -> Result<Option<NetConfig>, String> {
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
