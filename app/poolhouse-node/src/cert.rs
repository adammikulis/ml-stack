//! The device certificate: self-signed, made from the device key, valid ten years, so the SHA-256
//! of its DER is the device's identity for as long as the device exists.
//!
//! The certificate is the device key in an X.509 wrapper (Ed25519), so the board fingerprint
//! (hash of the public key) can always be read back out of a certificate a peer showed.

use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use ed25519_dalek::pkcs8::EncodePrivateKey;
use ed25519_dalek::SigningKey;
use rcgen::{date_time_ymd, CertificateParams, DistinguishedName, DnType, KeyPair, PKCS_ED25519};
use rustls::pki_types::{CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer};
use x509_parser::prelude::{FromDer, X509Certificate};

use crate::error::{Error, Result};
use crate::fsutil::{sha256_hex, write_atomic};

const FILE: &str = "device.cert";
const TEN_YEARS_S: i64 = 10 * 365 * 24 * 3600 + 3 * 24 * 3600;

/// The fingerprint of a certificate: SHA-256 of its DER, as lower-case hex.
pub fn cert_fingerprint(der: &[u8]) -> String {
    sha256_hex(der)
}

/// Whether ``text`` is a fingerprint: 64 lower-case hex digits.
pub fn valid_fingerprint(text: &str) -> bool {
    text.len() == 64 && text.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

/// Whether ``der`` is a well-formed certificate.
pub fn parses(der: &[u8]) -> bool {
    X509Certificate::from_der(der).is_ok_and(|(rest, _)| rest.is_empty())
}

/// The board fingerprint (hash of the Ed25519 public key) of the device whose certificate this is.
pub fn board_fingerprint(der: &[u8]) -> Result<String> {
    let (_, cert) = X509Certificate::from_der(der).map_err(|_| Error::Invalid("not a certificate".into()))?;
    let key = cert.tbs_certificate.subject_pki.subject_public_key.data.as_ref();
    if key.len() != 32 {
        return Err(Error::Invalid("a device certificate holds an Ed25519 key".into()));
    }
    Ok(sha256_hex(key))
}

/// A device's certificate and the key it signs with.
#[derive(Clone)]
pub struct Identity {
    pub der: Vec<u8>,
    pkcs8: Vec<u8>,
}

impl Identity {
    pub fn fingerprint(&self) -> String {
        cert_fingerprint(&self.der)
    }

    pub fn chain(&self) -> Vec<CertificateDer<'static>> {
        vec![CertificateDer::from(self.der.clone())]
    }

    pub fn private_key(&self) -> PrivateKeyDer<'static> {
        PrivateKeyDer::from(PrivatePkcs8KeyDer::from(self.pkcs8.clone()))
    }
}

fn now_s() -> i64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs() as i64)
}

/// The calendar date of a day count since 1970-01-01 (proleptic Gregorian).
fn civil(days: i64) -> (i32, u8, u8) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u8;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u8;
    ((yoe + era * 400 + i64::from(m <= 2)) as i32, m, d)
}

fn pkcs8_of(key: &SigningKey) -> Result<Vec<u8>> {
    key.to_pkcs8_der().map(|d| d.as_bytes().to_vec()).map_err(|e| Error::Invalid(format!("device key: {e}")))
}

fn make(key: &SigningKey, pkcs8: &[u8]) -> Result<Vec<u8>> {
    let pair = KeyPair::from_pkcs8_der_and_sign_algo(&PrivatePkcs8KeyDer::from(pkcs8.to_vec()), &PKCS_ED25519)
        .map_err(|e| Error::Invalid(format!("device key: {e}")))?;
    let mut params = CertificateParams::new(Vec::<String>::new()).map_err(|e| Error::Invalid(e.to_string()))?;
    let mut name = DistinguishedName::new();
    name.push(DnType::CommonName, "poolhouse-node");
    params.distinguished_name = name;
    let now = now_s();
    let ymd = |s: i64| {
        let (y, m, d) = civil(s.div_euclid(86_400));
        date_time_ymd(y, m, d)
    };
    params.not_before = ymd(now - 24 * 3600);
    params.not_after = ymd(now + TEN_YEARS_S);
    let cert = params.self_signed(&pair).map_err(|e| Error::Invalid(format!("certificate: {e}")))?;
    debug_assert!(board_fingerprint(cert.der()).is_ok_and(|f| f == crate::device::fingerprint(key)));
    Ok(cert.der().to_vec())
}

/// Load the certificate of ``key`` from ``dir``, making it on first use. A stored certificate
/// that is not this key's is replaced (it is a new identity).
pub fn load_or_create(dir: &Path, key: &SigningKey) -> Result<Identity> {
    let pkcs8 = pkcs8_of(key)?;
    let path = dir.join(FILE);
    if let Ok(der) = std::fs::read(&path) {
        if board_fingerprint(&der).is_ok_and(|f| f == crate::device::fingerprint(key)) {
            return Ok(Identity { der, pkcs8 });
        }
    }
    let der = make(key, &pkcs8)?;
    write_atomic(&path, &der)?;
    Ok(Identity { der, pkcs8 })
}
