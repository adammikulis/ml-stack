//! The device key: made once, kept 0600, signs the heads of every board's own logs.

use std::path::Path;

use ed25519_dalek::SigningKey;

use crate::error::{Error, Result};
use crate::fsutil::{hex, random_hex, sha256_hex, unhex, write_atomic};

const FILE: &str = "device.key";

/// Load the device key from ``dir``, creating it on first use.
pub fn load_or_create(dir: &Path) -> Result<SigningKey> {
    let path = dir.join(FILE);
    let seed = match std::fs::read_to_string(&path) {
        Ok(text) => text.trim().to_string(),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            let seed = random_hex(32)?;
            write_atomic(&path, seed.as_bytes())?;
            seed
        }
        Err(e) => return Err(e.into()),
    };
    let bytes: [u8; 32] = unhex(&seed).and_then(|b| b.try_into().ok()).ok_or_else(|| Error::Damaged("device key unreadable".into()))?;
    Ok(SigningKey::from_bytes(&bytes))
}

/// The fingerprint of a device: SHA-256 of its public key as hex.
pub fn fingerprint(key: &SigningKey) -> String {
    sha256_hex(key.verifying_key().as_bytes())
}

/// The public key as hex.
pub fn public_hex(key: &SigningKey) -> String {
    hex(key.verifying_key().as_bytes())
}
