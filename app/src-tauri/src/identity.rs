//! Knowing that what answers on the port is our daemon, and not whatever got there first.
//!
//! The daemon keeps a secret beside its settings, readable by its user alone. It answers a
//! fresh challenge with an HMAC-SHA256 of it. This app reads the same file, so it can check
//! the answer; a process that cannot read the file cannot give one.

use std::fs;
use std::io::Read;
use std::path::Path;

use sha2::{Digest, Sha256};

pub const SECRET_FILE: &str = "app-identity";
pub const HEADER: &str = "X-ML-Stack-Challenge";
const BLOCK: usize = 64;
const SECRET_BYTES: usize = 32;
const CHALLENGE_BYTES: usize = 16;

/// A fresh unpredictable challenge, as lowercase hex.
pub fn challenge() -> Option<String> {
    let mut bytes = [0u8; CHALLENGE_BYTES];
    getrandom::fill(&mut bytes).ok()?;
    Some(hex(&bytes))
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn unhex(text: &str) -> Option<Vec<u8>> {
    if text.len() % 2 != 0 || !text.is_ascii() {
        return None;
    }
    (0..text.len())
        .step_by(2)
        .map(|at| u8::from_str_radix(&text[at..at + 2], 16).ok())
        .collect()
}

/// The root's secret, if it is a plain file only its owner can read.
pub fn secret(root: &Path) -> Option<Vec<u8>> {
    let path = root.join(SECRET_FILE);
    let meta = fs::symlink_metadata(&path).ok()?;
    if !meta.is_file() || meta.len() > 256 {
        return None;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if meta.permissions().mode() & 0o077 != 0 {
            return None;
        }
    }
    let mut text = String::new();
    fs::File::open(&path).ok()?.take(256).read_to_string(&mut text).ok()?;
    let bytes = unhex(text.trim())?;
    (bytes.len() == SECRET_BYTES).then_some(bytes)
}

/// HMAC-SHA256 (RFC 2104).
pub fn hmac_sha256(key: &[u8], message: &[u8]) -> [u8; 32] {
    let mut block = [0u8; BLOCK];
    if key.len() > BLOCK {
        block[..32].copy_from_slice(&Sha256::digest(key));
    } else {
        block[..key.len()].copy_from_slice(key);
    }
    let inner: Vec<u8> = block.iter().map(|byte| byte ^ 0x36).collect();
    let outer: Vec<u8> = block.iter().map(|byte| byte ^ 0x5c).collect();
    let mut first = Sha256::new();
    first.update(&inner);
    first.update(message);
    let mut second = Sha256::new();
    second.update(&outer);
    second.update(first.finalize());
    second.finalize().into()
}

/// The proof the daemon owes for this challenge under this secret, as lowercase hex.
pub fn expected(secret: &[u8], challenge: &str) -> String {
    hex(&hmac_sha256(secret, challenge.as_bytes()))
}

/// Whether two proofs are equal, without stopping at the first byte that differs.
pub fn same(left: &str, right: &str) -> bool {
    left.len() == right.len()
        && left.bytes().zip(right.bytes()).fold(0u8, |seen, (a, b)| seen | (a ^ b)) == 0
}

#[cfg(test)]
#[path = "identity_tests.rs"]
mod tests;
