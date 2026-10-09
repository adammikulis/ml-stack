//! The ml-stack daemon: the one already answering on the port, or one started here.

use std::io::{Read, Write};
use std::net::{Shutdown, TcpStream};
use std::path::Path;
use std::time::{Duration, Instant};

use crate::identity;
use tauri::AppHandle;
use tauri_plugin_shell::process::CommandChild;
use tauri_plugin_shell::ShellExt;

const SIDECAR: &str = "ml-stack-headless";
const CONNECT_TIMEOUT: Duration = Duration::from_millis(500);
const HEALTH_SECONDS: u64 = 60;
const HEALTH_BYTES: u64 = 1024 * 1024;
const STOP_GRACE: Duration = Duration::from_secs(30);

#[cfg(test)]
#[path = "daemon_tests.rs"]
mod tests;

/// Whether our daemon answers `/health` on this port: one that proves, from the secret kept
/// in `root`, that it is the daemon of this installation and not another process on the port.
pub fn healthy(port: u16, root: &Path) -> bool {
    let addr = format!("127.0.0.1:{port}").parse();
    let Ok(addr) = addr else { return false };
    let Some(challenge) = identity::challenge() else {
        return false;
    };
    let Ok(mut sock) = TcpStream::connect_timeout(&addr, CONNECT_TIMEOUT) else {
        return false;
    };
    let _ = sock.set_read_timeout(Some(CONNECT_TIMEOUT));
    let request = format!(
        "GET /health HTTP/1.0\r\nHost: 127.0.0.1:{port}\r\n{}: {challenge}\r\n\r\n",
        identity::HEADER
    );
    if sock.write_all(request.as_bytes()).is_err() {
        return false;
    }
    let mut response = Vec::new();
    let read = Read::by_ref(&mut sock).take(HEALTH_BYTES + 1).read_to_end(&mut response);
    let _ = sock.shutdown(Shutdown::Both);
    // the secret is read after the answer: the daemon makes it while handling the challenge
    let Some(secret) = identity::secret(root) else {
        return false;
    };
    read.is_ok()
        && response.len() <= HEALTH_BYTES as usize
        && valid_health(&response, &identity::expected(&secret, &challenge))
}

fn valid_health(response: &[u8], proof: &str) -> bool {
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
        && body["proof"].as_str().is_some_and(|said| identity::same(said, proof))
}

/// Block until the daemon answers, or give up.
pub fn wait_for_health(port: u16, root: &Path) -> bool {
    let deadline = Instant::now() + Duration::from_secs(HEALTH_SECONDS);
    while Instant::now() < deadline {
        if healthy(port, root) {
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
    if !wait_for_health(port, Path::new(root)) {
        stop(child);
        return Err(format!("the daemon did not start on port {port}"));
    }
    Ok(child)
}

/// Whether a process with this id is still there.
#[cfg(unix)]
fn alive(pid: u32) -> bool {
    std::process::Command::new("/bin/kill")
        .args(["-0", &pid.to_string()])
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

/// Poll `alive` until it says no or `grace` has passed; true when the process is gone.
fn gone_within(alive: impl Fn() -> bool, grace: Duration, poll: Duration) -> bool {
    let deadline = Instant::now() + grace;
    while alive() {
        if Instant::now() >= deadline {
            return false;
        }
        std::thread::sleep(poll);
    }
    true
}

/// Stop a daemon this app started, and the worker the frozen launcher runs under it.
///
/// The daemon closes its stores on SIGTERM; killing it first would leave the graph store's
/// write-ahead log behind, so it is given `STOP_GRACE` to exit before anything stronger.
pub fn stop(child: CommandChild) {
    #[cfg(unix)]
    {
        let pid = child.pid();
        let _ = std::process::Command::new("/bin/kill")
            .args(["-TERM", &pid.to_string()])
            .status();
        if gone_within(|| alive(pid), STOP_GRACE, Duration::from_millis(100)) {
            return;
        }
    }
    let _ = child.kill();
}
