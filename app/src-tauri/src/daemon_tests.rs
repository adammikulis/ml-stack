use super::{healthy, valid_health};
use crate::identity::{self, SECRET_FILE};
use std::fs;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::path::{Path, PathBuf};

const SECRET: [u8; 32] = [0x5a; 32];
const BODY: &str = r#""ok":true,"name":"demo","slots":1,"free":1,"busy":false"#;

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("poolside-health-{name}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn write_secret(root: &Path, secret: &[u8]) {
    let path = root.join(SECRET_FILE);
    fs::write(&path, secret.iter().map(|byte| format!("{byte:02x}")).collect::<String>()).unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&path, fs::Permissions::from_mode(0o600)).unwrap();
    }
}

/// A listener that answers every `/health` the way `answer` says, given the challenge it was sent.
fn listening(answer: impl Fn(&str) -> String + Send + 'static) -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { return };
            let mut request = [0u8; 2048];
            let read = stream.read(&mut request).unwrap_or(0);
            let text = String::from_utf8_lossy(&request[..read]).to_string();
            let challenge = text
                .lines()
                .find_map(|line| line.strip_prefix(&format!("{}: ", identity::HEADER)))
                .unwrap_or("")
                .trim()
                .to_string();
            let body = answer(&challenge);
            let _ = write!(stream, "HTTP/1.0 200 OK\r\nContent-Type: application/json\r\n\r\n{body}");
        }
    });
    port
}

fn proving(secret: [u8; 32]) -> impl Fn(&str) -> String + Send + 'static {
    move |challenge| format!(r#"{{{BODY},"proof":"{}"}}"#, identity::expected(&secret, challenge))
}

#[test]
fn health_requires_complete_daemon_json_and_the_proof() {
    let proof = "ab".repeat(32);
    let response = format!("HTTP/1.0 200 OK\r\nContent-Type: application/json\r\n\r\n{{{BODY},\"proof\":\"{proof}\"}}");
    assert!(valid_health(response.as_bytes(), &proof));
    for bad in [
        "HTTP/1.0 200 OK\r\n\r\nhello".to_string(),
        "HTTP/1.0 200 OK\r\n\r\n{\"ok\":true}".to_string(),
        "HTTP/1.0 200 OK\r\n\r\n{".to_string(),
        "HTTP/1.0 500 200\r\n\r\n{}".to_string(),
        "HTTP/1.0 200 OK".to_string(),
        format!("HTTP/1.0 200 OK\r\n\r\n{{{BODY}}}"),
        format!("HTTP/1.0 200 OK\r\n\r\n{{{BODY},\"proof\":7}}"),
        format!("HTTP/1.0 200 OK\r\n\r\n{{{BODY},\"proof\":\"{}\"}}", "cd".repeat(32)),
    ] {
        assert!(!valid_health(bad.as_bytes(), &proof), "{bad}");
    }
    assert!(!valid_health(response.replace("200 OK", "503 Unavailable").as_bytes(), &proof));
    assert!(!valid_health(response.replace("true", "false").as_bytes(), &proof));
}

#[test]
fn our_daemon_is_recognised_by_its_proof() {
    let root = scratch("ours");
    write_secret(&root, &SECRET);
    assert!(healthy(listening(proving(SECRET)), &root));
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn a_process_with_the_right_json_but_not_the_secret_is_not_our_daemon() {
    let root = scratch("impostor");
    write_secret(&root, &SECRET);
    // the right shape and no proof
    assert!(!healthy(listening(|_| format!("{{{BODY}}}")), &root));
    // the right shape and a proof under a secret of its own
    assert!(!healthy(listening(proving([0x11; 32])), &root));
    // a proof replayed from an earlier challenge
    let replayed = format!(r#"{{{BODY},"proof":"{}"}}"#, identity::expected(&SECRET, &"ab".repeat(16)));
    assert!(!healthy(listening(move |_| replayed.clone()), &root));
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn nothing_is_recognised_without_the_secret_or_a_listener() {
    let root = scratch("absent");
    // a daemon that proves under a secret this installation cannot read
    assert!(!healthy(listening(proving(SECRET)), &root));
    write_secret(&root, &SECRET);
    let silent = TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port();
    assert!(!healthy(silent, &root));
    let _ = fs::remove_dir_all(&root);
}
