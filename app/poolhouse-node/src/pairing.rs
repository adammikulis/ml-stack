//! Pairing by a code or passphrase: SPAKE2 (the `spake2` crate) with HMAC-SHA256 confirmations,
//! the construction of the Python pairing (`fleet/onboard/pake.py`).
//!
//! A short code is a few bits of entropy, so nothing sent may let a listener test a guess:
//! in SPAKE2 nothing does, and a side's confirmation is sent only after the other's verified,
//! so each guess costs one conversation, of which a window allows three. Both certificate
//! fingerprints are the two identities: the exchange succeeds only if the certificate each
//! side saw in the TLS handshake is the one the other presented, so a machine in the middle
//! that terminates TLS cannot also complete the exchange.

use hmac::{Hmac, Mac};
use sha2::{Digest, Sha256};
use spake2::{Ed25519Group, Identity, Password, Spake2};

use crate::error::{Error, Result};
use crate::fsutil::{hex, unhex};

type HmacSha256 = Hmac<Sha256>;

pub const MOST_TRIES: u8 = 3;
pub const MOST_MESSAGE: usize = 64;

fn length_prefixed(parts: &[&[u8]]) -> Vec<u8> {
    parts.iter().flat_map(|p| (p.len() as u32).to_be_bytes().into_iter().chain(p.iter().copied())).collect()
}

fn mac(key: &[u8], data: &[u8]) -> HmacSha256 {
    let mut m = HmacSha256::new_from_slice(key).expect("HMAC takes any key length");
    m.update(data);
    m
}

/// One end of one exchange.
pub struct Session {
    initiator: bool,
    state: Option<Spake2<Ed25519Group>>,
    out: Vec<u8>,
    context: Vec<u8>,
    ids: (String, String),
    tt: Vec<u8>,
    root: Vec<u8>,
}

impl Session {
    /// Begin as the side that asks (``initiator``) or the side that accepts. ``mine`` and
    /// ``theirs`` are certificate fingerprints, the initiator's first in the exchange.
    pub fn start(initiator: bool, code: &str, context: &[u8], mine: &str, theirs: &str) -> Session {
        let ids = if initiator { (mine.to_string(), theirs.to_string()) } else { (theirs.to_string(), mine.to_string()) };
        let mut password = context.to_vec();
        password.push(0);
        password.extend_from_slice(code.as_bytes());
        let (a, b) = (Identity::new(ids.0.as_bytes()), Identity::new(ids.1.as_bytes()));
        let (state, out) = if initiator {
            Spake2::<Ed25519Group>::start_a(&Password::new(&password), &a, &b)
        } else {
            Spake2::<Ed25519Group>::start_b(&Password::new(&password), &a, &b)
        };
        Session { initiator, state: Some(state), out, context: context.to_vec(), ids, tt: Vec::new(), root: Vec::new() }
    }

    /// What this end sends first, as hex.
    pub fn message(&self) -> String {
        hex(&self.out)
    }

    /// Take the other end's message and work out the shared key.
    pub fn receive(&mut self, other: &str) -> Result<()> {
        if other.is_empty() || other.len() > 2 * MOST_MESSAGE {
            return Err(Error::Invalid("not a pairing message".into()));
        }
        let raw = unhex(other).ok_or_else(|| Error::Invalid("not a pairing message".into()))?;
        let state = self.state.take().ok_or_else(|| Error::Invalid("the exchange already has its key".into()))?;
        let key = state.finish(&raw).map_err(|_| Error::Invalid("bad pairing message".into()))?;
        let (first, second): (&[u8], &[u8]) = if self.initiator { (&self.out, &raw) } else { (&raw, &self.out) };
        self.tt = length_prefixed(&[&self.context, self.ids.0.as_bytes(), self.ids.1.as_bytes(), first, second]);
        self.root = Sha256::digest(&key).to_vec();
        Ok(())
    }

    fn named(&self, name: &str) -> Result<HmacSha256> {
        if self.tt.is_empty() {
            return Err(Error::Invalid("no exchange yet".into()));
        }
        let sub = mac(&self.root, name.as_bytes()).finalize().into_bytes();
        Ok(mac(&sub, &self.tt))
    }

    /// What this end sends to say it holds the code.
    pub fn confirmation(&self) -> Result<String> {
        let role = if self.initiator { "A" } else { "B" };
        Ok(hex(&self.named(&format!("confirm-{role}"))?.finalize().into_bytes()))
    }

    /// Whether the other end's confirmation is right (compared in constant time).
    pub fn check(&self, theirs: &str) -> bool {
        let role = if self.initiator { "B" } else { "A" };
        match (unhex(theirs), self.named(&format!("confirm-{role}"))) {
            (Some(tag), Ok(m)) => m.verify_slice(&tag).is_ok(),
            _ => false,
        }
    }

    /// A tag over ``payload`` under the exchange's payload key.
    pub fn seal(&self, payload: &[u8]) -> Result<String> {
        let sub = mac(&self.root, b"payload").finalize().into_bytes();
        Ok(hex(&mac(&sub, payload).finalize().into_bytes()))
    }

    /// Whether ``tag`` is the tag of ``payload``.
    pub fn open(&self, payload: &[u8], tag: &str) -> bool {
        if self.root.is_empty() {
            return false;
        }
        let sub = mac(&self.root, b"payload").finalize().into_bytes();
        unhex(tag).is_some_and(|t| mac(&sub, payload).verify_slice(&t).is_ok())
    }
}

/// The context one exchange is bound to: a nonce the asking side chose.
pub fn context_for(nonce: &str) -> Vec<u8> {
    format!("poolhouse-pair/v1/{nonce}").into_bytes()
}

/// A code the owner reads out: six digits.
pub fn new_code() -> Result<String> {
    let mut bytes = [0u8; 4];
    getrandom::fill(&mut bytes).map_err(|e| std::io::Error::other(e.to_string()))?;
    Ok(format!("{:06}", u32::from_be_bytes(bytes) % 1_000_000))
}

/// The window in which this device takes a pairing: one code, three tries, a time limit.
pub struct Window {
    pub code: String,
    pub until_ms: u64,
    pub tries: u8,
    /// The exchange in progress: the asking device's fingerprint, its chosen name, the session.
    pub session: Option<(String, String, Session)>,
}

impl Window {
    pub fn new(code: &str, until_ms: u64) -> Window {
        Window { code: code.into(), until_ms, tries: MOST_TRIES, session: None }
    }

    pub fn open(&self, now_ms: u64) -> bool {
        now_ms < self.until_ms && self.tries > 0
    }

    /// Spend a try; true while any remain.
    pub fn failed(&mut self) -> bool {
        self.tries = self.tries.saturating_sub(1);
        self.session = None;
        self.tries > 0
    }
}
