//! The ml-stack daemon: the one already answering on the port, or one started here.

use std::io::{Read, Write};
use std::net::{Shutdown, TcpStream};
use std::time::{Duration, Instant};

use tauri::AppHandle;
use tauri_plugin_shell::process::CommandChild;
use tauri_plugin_shell::ShellExt;

const SIDECAR: &str = "ml-stack-headless";
const CONNECT_TIMEOUT: Duration = Duration::from_millis(500);
const HEALTH_SECONDS: u64 = 30;

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
    let mut head = [0u8; 16];
    let read = sock.read(&mut head).unwrap_or(0);
    let _ = sock.shutdown(Shutdown::Both);
    String::from_utf8_lossy(&head[..read]).contains(" 200")
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
