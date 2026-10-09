//! The one-shot sign-in ticket the app's own window opens with.

use std::io::{Read, Write};
use std::net::{Shutdown, TcpStream};
use std::path::Path;
use std::time::Duration;

const TIMEOUT: Duration = Duration::from_secs(5);
const RESPONSE_BYTES: u64 = 64 * 1024;
const RECORD_VERSION: u64 = 1;

#[cfg(test)]
#[path = "launch_tests.rs"]
mod tests;

/// The window's address: carrying a ticket the daemon on this port issued against the secret
/// recorded under `root`, or the bare page when none is issued.
pub fn page_url(root: &Path, port: u16) -> String {
    let bare = crate::origin::address(port);
    match ticket(root, port) {
        Some(ticket) => format!("{bare}?launch_ticket={ticket}"),
        None => bare,
    }
}

fn ticket(root: &Path, port: u16) -> Option<String> {
    let path = root.join("launch").join("secret.json");
    if !private(&path) {
        return None;
    }
    let record: serde_json::Value = serde_json::from_slice(&std::fs::read(&path).ok()?).ok()?;
    if record["version"].as_u64()? != RECORD_VERSION || record["port"].as_u64()? != u64::from(port) {
        return None;
    }
    let secret = record["secret"].as_str().filter(|s| token(s))?;
    ticket_from(&exchange(port, secret)?)
}

/// Whether `path` is a plain file only its owner can read.
#[cfg(unix)]
fn private(path: &Path) -> bool {
    use std::os::unix::fs::MetadataExt;
    match std::fs::symlink_metadata(path) {
        Ok(info) => info.file_type().is_file() && info.mode() & 0o077 == 0,
        Err(_) => false,
    }
}

#[cfg(not(unix))]
fn private(path: &Path) -> bool {
    std::fs::symlink_metadata(path).map(|i| i.file_type().is_file()).unwrap_or(false)
}

/// Whether `text` is made of the characters a URL-safe random token holds.
fn token(text: &str) -> bool {
    !text.is_empty()
        && text.len() <= 256
        && text.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
}

fn exchange(port: u16, secret: &str) -> Option<Vec<u8>> {
    let addr = format!("127.0.0.1:{port}").parse().ok()?;
    let mut sock = TcpStream::connect_timeout(&addr, TIMEOUT).ok()?;
    sock.set_read_timeout(Some(TIMEOUT)).ok()?;
    let request = format!(
        "POST /ui/launch/ticket HTTP/1.0\r\nHost: 127.0.0.1:{port}\r\nX-Poolhouse-UI: 1\r\n\
         X-Poolhouse-Launch: {secret}\r\nContent-Type: application/json\r\nContent-Length: 2\r\n\r\n{{}}"
    );
    sock.write_all(request.as_bytes()).ok()?;
    let mut response = Vec::new();
    Read::by_ref(&mut sock).take(RESPONSE_BYTES).read_to_end(&mut response).ok()?;
    let _ = sock.shutdown(Shutdown::Both);
    Some(response)
}

/// The ticket in a complete `200` response from the daemon.
fn ticket_from(response: &[u8]) -> Option<String> {
    let split = response.windows(4).position(|part| part == b"\r\n\r\n")?;
    let headers = String::from_utf8_lossy(&response[..split]);
    let status: Vec<_> = headers.lines().next()?.split_whitespace().collect();
    if status.len() < 2 || !matches!(status[0], "HTTP/1.0" | "HTTP/1.1") || status[1] != "200" {
        return None;
    }
    let body: serde_json::Value = serde_json::from_slice(&response[split + 4..]).ok()?;
    body["ticket"].as_str().filter(|t| token(t)).map(str::to_string)
}
