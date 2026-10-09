//! A shard request, checked before anything is stored or run. The sender chooses a tier, test
//! files and a time limit; the argv, environment and working directory are the host's.

use serde_json::{Map, Value};

use crate::error::{Error, Result};

pub const TIERS: [&str; 5] = ["all", "fast", "full", "gate", "slow"];
pub const MOST_FILES: usize = 200;
pub const MOST_SECONDS: u64 = 3600;
pub const MOST_PACKED: u64 = 24 << 20;
const FIELDS: [&str; 6] = ["id", "tree_sha256", "size", "tier", "files", "timeout_s"];

#[derive(Clone, Debug)]
pub struct Spec {
    pub id: String,
    pub tree_sha256: String,
    pub size: u64,
    pub tier: String,
    pub files: Vec<String>,
    pub timeout_s: u64,
}

fn lower_hex(text: &str, len: usize) -> bool {
    text.len() == len && text.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

/// `tests/NAME.py`: a plain file directly under tests, no dot first, no separators beyond the one.
pub fn test_file(name: &str) -> bool {
    let Some(base) = name.strip_prefix("tests/").and_then(|b| b.strip_suffix(".py")) else { return false };
    !base.is_empty() && name.len() <= 120
        && base.bytes().next().is_some_and(|b| b.is_ascii_alphanumeric() || b == b'_')
        && base.bytes().all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'.' | b'-'))
}

pub fn valid_id(id: &str) -> bool {
    lower_hex(id, 32)
}

fn text<'a>(m: &'a Map<String, Value>, key: &str) -> Result<&'a str> {
    m.get(key).and_then(Value::as_str).ok_or_else(|| Error::Invalid(format!("{key} is text")))
}

fn whole(m: &Map<String, Value>, key: &str) -> Result<u64> {
    m.get(key).and_then(Value::as_u64).ok_or_else(|| Error::Invalid(format!("{key} is a whole number")))
}

/// The `shard_start` request as a `Spec`; every field is known, typed and in range.
pub fn parse(req: &Value) -> Result<Spec> {
    let m = req.as_object().ok_or_else(|| Error::Invalid("a shard request is an object".into()))?;
    if let Some(k) = m.keys().find(|k| k.as_str() != "op" && k.as_str() != "by" && !FIELDS.contains(&k.as_str())) {
        return Err(Error::Invalid(format!("a shard takes no field called {}", k.chars().take(40).collect::<String>())));
    }
    let (id, digest) = (text(m, "id")?, text(m, "tree_sha256")?);
    if !valid_id(id) || !lower_hex(digest, 64) {
        return Err(Error::Invalid("the shard id is 32 and the tree digest 64 lowercase hex digits".into()));
    }
    let tier = text(m, "tier")?;
    if !TIERS.contains(&tier) {
        return Err(Error::Invalid(format!("tier is one of {}", TIERS.join(", "))));
    }
    let size = whole(m, "size")?;
    let seconds = whole(m, "timeout_s")?;
    if size == 0 || size > MOST_PACKED || !(1..=MOST_SECONDS).contains(&seconds) {
        return Err(Error::Invalid(format!("size is 1 to {MOST_PACKED} bytes and timeout_s 1 to {MOST_SECONDS}")));
    }
    let listed = m.get("files").and_then(Value::as_array).ok_or_else(|| Error::Invalid("files is a list".into()))?;
    let files: Vec<String> = listed.iter().map(|f| f.as_str().map(String::from)).collect::<Option<_>>()
        .ok_or_else(|| Error::Invalid("each file is text".into()))?;
    if files.len() > MOST_FILES || files.iter().enumerate().any(|(i, f)| !test_file(f) || files[..i].contains(f)) {
        return Err(Error::Invalid(format!("a shard names at most {MOST_FILES} distinct tests/NAME.py files")));
    }
    if tier == "gate" && !files.is_empty() {
        return Err(Error::Invalid("the gate takes no files".into()));
    }
    Ok(Spec { id: id.into(), tree_sha256: digest.into(), size, tier: tier.into(), files, timeout_s: seconds })
}
