//! `poolside-node run --state DIR` serves the node; `status` and `shutdown` talk to a running one.

use std::path::PathBuf;
use std::process::ExitCode;

use poolside_node::client::Client;
use poolside_node::server::Server;
use serde_json::json;

fn state_dir(args: &[String]) -> PathBuf {
    match args.iter().position(|a| a == "--state").and_then(|i| args.get(i + 1)) {
        Some(dir) => PathBuf::from(dir),
        None => PathBuf::from(std::env::var_os("HOME").unwrap_or_default()).join(".poolside").join("node"),
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let state = state_dir(&args);
    let result = match args.first().map(String::as_str) {
        Some("run") => Server::bind(&state).and_then(Server::serve).map(|_| String::new()),
        Some("status") => Client::connect(&state).and_then(|mut c| c.call("status", "", "", json!({}))).map(|v| v.to_string()),
        _ => {
            eprintln!("usage: poolside-node run|status [--state DIR]");
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
