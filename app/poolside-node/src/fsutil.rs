//! Files that only their owner can read, written so a crash leaves the old or the new content.

use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;

use sha2::{Digest, Sha256};

use crate::error::Result;

pub use crate::sys::{private_dir, sync_dir};

/// Replace ``path`` with ``bytes`` through a synced temporary file only the owner can read.
pub fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    let tmp = path.with_extension("tmp");
    let mut file = crate::sys::private_file(OpenOptions::new().write(true).create(true).truncate(true)).open(&tmp)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    drop(file);
    crate::sys::replace_file(&tmp, path)?;
    if let Some(parent) = path.parent() {
        sync_dir(parent)?;
    }
    Ok(())
}

/// Lower-case hex of ``bytes``.
pub fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// The bytes of a hex string, or None.
pub fn unhex(text: &str) -> Option<Vec<u8>> {
    if text.len() % 2 != 0 || !text.is_ascii() {
        return None;
    }
    (0..text.len()).step_by(2).map(|i| u8::from_str_radix(&text[i..i + 2], 16).ok()).collect()
}

/// SHA-256 of ``data`` as hex.
pub fn sha256_hex(data: &[u8]) -> String {
    hex(&Sha256::digest(data))
}

/// ``n`` random bytes as hex.
pub fn random_hex(n: usize) -> Result<String> {
    let mut bytes = vec![0u8; n];
    getrandom::fill(&mut bytes).map_err(|e| std::io::Error::other(e.to_string()))?;
    Ok(hex(&bytes))
}
