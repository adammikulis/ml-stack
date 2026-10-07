//! The ml-stack daemon: the one already answering on the port, or one started here.

use std::io::{Read, Write};
use std::net::{Shutdown, TcpStream};
use std::time::{Duration, Instant};

use tauri::AppHandle;
use tauri_plugin_shell::process::CommandChild;
use tauri_plugin_shell::ShellExt;

const SIDECAR: &str = "ml-stack-headless";
const CONNECT_TIMEOUT: Duration = Duration::from_millis(500);
const HEALTH_SECONDS: u64 = 60;
const HEALTH_BYTES: u64 = 1024 * 1024;

#[cfg(test)]
#[path = "daemon_tests.rs"]
mod tests;

/// Whether a daemon answers `/health` on this port.
pub fn healthy(port: u16) -> bool {
    let addr = format!("127.0.0.1:{port}").parse();
    let Ok(addr) = addr else { return false };
    let Ok(mut sock) = TcpStream::connect_timeout(&addr, CONNECT_TIMEOUT) else {
        return false;
    };
    let _ = sock.set_read_timeout(Some(CONNECT_TIMEOUT));
    let request = format!("GET /health HTTP/1.0\r\nHost: 127.0.0.1:{port}\r\n\r\n");
    if sock.write_all(request.as_bytes()).is_err() {
        return false;
    }
    let mut response = Vec::new();
    let read = Read::by_ref(&mut sock).take(HEALTH_BYTES + 1).read_to_end(&mut response);
    let _ = sock.shutdown(Shutdown::Both);
    read.is_ok() && response.len() <= HEALTH_BYTES as usize && valid_health(&response)
}

fn valid_health(response: &[u8]) -> bool {
    let Some(split) = response.windows(4).position(|part| part == b"\r\n\r\n") else {
        return false;
    };
    let headers = String::from_utf8_lossy(&response[..split]);
    let status: Vec<_> = headers.lines().next().unwrap_or("").split_whitespace().collect();
    if status.len() < 2 || !matches!(status[0], "HTTP/1.0" | "HTTP/1.1") || status[1] != "200" {
        return false;
    }
    let Ok(body) = serde_json::from_slice::<serde_json::Value>(&response[split + 4..]) else {
        return false;
    };
    body["ok"].as_bool() == Some(true)
        && body["name"].as_str().is_some()
        && body["slots"].as_u64().is_some()
        && body["free"].as_u64().is_some()
        && body["busy"].as_bool().is_some()
}

/// Block until the daemon answers, or give up.
pub fn wait_for_health(port: u16) -> bool {
    let deadline = Instant::now() + Duration::from_secs(HEALTH_SECONDS);
    while Instant::now() < deadline {
        if healthy(port) {
            return true;
        }
        std::thread::sleep(Duration::from_millis(150));
    }
    false
}

/// Start the sidecar daemon on this port and wait for it to answer.
pub fn start(app: &AppHandle, port: u16, root: &str) -> Result<CommandChild, String> {
    let command = app
        .shell()
        .sidecar(SIDECAR)
        .map_err(|e| format!("no {SIDECAR} beside the app: {e}"))?
        .args(["--port", &port.to_string(), "--root", root, "--no-browser"]);
    let (_events, child) = command
        .spawn()
        .map_err(|e| format!("could not start {SIDECAR}: {e}"))?;
    if !wait_for_health(port) {
        stop(child);
        return Err(format!("the daemon did not start on port {port}"));
    }
    Ok(child)
}

/// Stop a daemon this app started, and the worker the frozen launcher runs under it.
pub fn stop(child: CommandChild) {
    let pid = child.pid();
    #[cfg(unix)]
    let _ = std::process::Command::new("/bin/kill")
        .args(["-TERM", &pid.to_string()])
        .status();
    #[cfg(unix)]
    std::thread::sleep(Duration::from_millis(500));
    let _ = child.kill();
}
