//! `poolside-node run --state DIR` serves the node; `status` talks to a running one.
//!
//! The network is off unless `--listen ADDR` names where to listen, or `--lan` says to listen on
//! every interface and beacon on the multicast group (ports 7447 and 7448). `--beacon-bind ADDR` turns
//! the beacon on, `--beacon-send ADDR` (repeatable) says where to send it (the multicast
//! group in production, a peer in tests), `--advertise IP` the address it carries,
//! `--allow-loopback` lets a test advertise loopback, `--sync-ms N` sets the sync interval.
//!
//! With the beacon on, the device finds a pool without a key or a code: it takes the project of
//! the repository it runs in (`--project-dir DIR`, else the working directory; `--project NAME`
//! names one instead; `--no-project` turns this off), listens `--settle-ms N` for an `open` pool
//! of that project to join, and makes one when it hears none.

use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Duration;

use poolside_node::beacon::{BeaconConfig, GROUP};
use poolside_node::client::Client;
use poolside_node::net::{Net, NetConfig, SETTLE};
use poolside_node::projectid;
use poolside_node::server::Server;
use serde_json::json;

/// What `--lan` stands for: listen on every interface and beacon to the multicast group.
const LAN_LISTEN: &str = "0.0.0.0:7447";
const LAN_BEACON_PORT: u16 = 7448;

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
    let mut send_to = args.windows(2).filter(|w| w[0] == "--beacon-send").map(|w| addr(&w[1])).collect::<Result<Vec<_>, _>>()?;
    if send_to.is_empty() && lan(args) {
        send_to.push(SocketAddr::from((GROUP, LAN_BEACON_PORT)));
    }
    let advertise = flag(args, "--advertise").map(|a| a.parse::<IpAddr>().map_err(|e| e.to_string())).transpose()?;
    let interface = flag(args, "--multicast-if").map(|a| a.parse::<Ipv4Addr>().map_err(|e| format!("--multicast-if: {e}"))).transpose()?;
    Ok(BeaconConfig {
        bind: addr(bind)?, send_to, advertise, interval: Duration::from_millis(500), allow_loopback: args.iter().any(|a| a == "--allow-loopback"),
        interface, broadcast: lan(args),
    })
}

/// The project key: the named project, else that of the repository around the directory; None
/// outside a repository when none is named.
fn project_key(args: &[String]) -> Result<Option<String>, String> {
    let over = flag(args, "--project");
    let dir = flag(args, "--project-dir").map(PathBuf::from).or_else(|| std::env::current_dir().ok()).unwrap_or_default();
    match projectid::project_of(&dir, over) {
        Ok(key) => Ok(Some(key)),
        Err(_) if over.is_none() => Ok(None),
        Err(e) => Err(e.to_string()),
    }
}

fn lan(args: &[String]) -> bool {
    args.iter().any(|a| a == "--lan")
}

fn net_config(args: &[String]) -> Result<Option<NetConfig>, String> {
    let Some(listen) = flag(args, "--listen").or(lan(args).then_some(LAN_LISTEN)) else { return Ok(None) };
    let beacon_bind = format!("0.0.0.0:{LAN_BEACON_PORT}");
    let beacon = flag(args, "--beacon-bind").or(lan(args).then_some(beacon_bind.as_str())).map(|b| beacon_config(args, b)).transpose()?;
    let sync_every = flag(args, "--sync-ms").map(|n| n.parse::<u64>().map(Duration::from_millis).map_err(|e| e.to_string())).transpose()?;
    let settle = flag(args, "--settle-ms").map(|n| n.parse::<u64>().map(Duration::from_millis).map_err(|e| e.to_string())).transpose()?;
    let project = if beacon.is_some() && !args.iter().any(|a| a == "--no-project") { project_key(args)? } else { None };
    Ok(Some(NetConfig { listen: addr(listen)?, beacon, sync_every, project, settle: settle.unwrap_or(SETTLE) }))
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

/// `probe`: the local-network check (`poolside_node::probe`); its JSON line goes to `--out FILE` when given (a bundled app
/// has no terminal), else to stdout.
fn probe(args: &[String]) -> Result<String, String> {
    let seconds = flag(args, "--seconds").map(|n| n.parse::<u64>().map_err(|e| format!("--seconds: {e}"))).transpose()?.unwrap_or(10);
    let peers = args.windows(2).filter(|w| w[0] == "--peer").map(|w| w[1].parse::<Ipv4Addr>().map_err(|e| format!("--peer {}: {e}", w[1]))).collect::<Result<Vec<_>, _>>()?;
    let found = poolside_node::probe::run(seconds, &peers).map_err(|e| e.to_string());
    match (flag(args, "--out"), found) {
        (Some(file), found) => {
            let text = found.unwrap_or_else(|e| json!({"error": e})).to_string();
            std::fs::write(file, text).map(|_| String::new()).map_err(|e| e.to_string())
        }
        (None, found) => found.map(|v| v.to_string()),
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let state = state_dir(&args);
    let result = match args.first().map(String::as_str) {
        Some("run") => run(&state, &args).map(|_| String::new()),
        Some("status") => Client::connect(&state).and_then(|mut c| c.call("status", "", "", json!({}))).map(|v| v.to_string()).map_err(|e| e.to_string()),
        Some("probe") => probe(&args),
        _ => {
            eprintln!("usage: poolside-node run|status|probe [--state DIR] [--listen ADDR ...] [--seconds N] [--out FILE]");
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
