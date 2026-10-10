//! A board entry: one signed-log row, its kinds, its canonical hash and the name rules.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::{Error, Result};
use crate::fsutil::sha256_hex;

/// The schema version every row carries; a row of another version is refused.
pub const VERSION: u32 = 1;
/// The `prev` of the first row of a log.
pub const GENESIS: &str = "0000000000000000000000000000000000000000000000000000000000000000";
/// The most bytes one serialized row may take.
pub const MAX_ROW_BYTES: usize = 128 * 1024;
const MAX_DEPTH: usize = 16;

/// What an entry is. Only message, note, verify, identity, lease, audit and landing are folded so far; the others
/// are part of the schema so a log written later still parses.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    Message,
    Note,
    /// The result of running a note's command: its exit code and the hash of its output.
    Verify,
    Task,
    /// A step of the landing queue: a request, a review, a cancel, a brake, a runner state or beat.
    Landing,
    Identity,
    Audit,
    ReputationEvent,
    /// A lease granted or ended on a device: what its table says it holds.
    Lease,
    /// A signature over the row before it; part of the log, never of the view.
    Head,
}

/// One entry of one origin's log on one board.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Row {
    pub v: u32,
    pub board: String,
    pub origin: String,
    pub seq: u64,
    pub prev: String,
    /// Hybrid logical clock: wall milliseconds, counter, origin.
    pub hlc: (u64, u64, String),
    pub kind: Kind,
    pub actor: String,
    pub idem: String,
    pub body: Value,
    pub hash: String,
}

/// ``value`` as JSON with every object's keys sorted, so a hash never depends on map order.
pub fn canonical(value: &Value) -> String {
    match value {
        Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            let parts: Vec<String> = keys
                .iter()
                .map(|k| format!("{}:{}", Value::String((*k).clone()), canonical(&map[*k])))
                .collect();
            format!("{{{}}}", parts.join(","))
        }
        Value::Array(items) => format!("[{}]", items.iter().map(canonical).collect::<Vec<_>>().join(",")),
        other => other.to_string(),
    }
}

impl Row {
    /// The hash of this row after ``prev``: SHA-256 of ``prev`` and the canonical row without its hash.
    pub fn digest(&self) -> String {
        let mut doc = serde_json::to_value(self).unwrap_or(Value::Null);
        if let Value::Object(map) = &mut doc {
            map.remove("hash");
        }
        sha256_hex(format!("{}{}", self.prev, canonical(&doc)).as_bytes())
    }

    /// The id of the row: board, origin and sequence number.
    pub fn id(&self) -> String {
        format!("{}:{}:{}", self.board, self.origin, self.seq)
    }

    /// The key rows are merged by: wall clock, counter, origin, sequence number.
    pub fn order(&self) -> (u64, u64, &str, u64) {
        (self.hlc.0, self.hlc.1, self.origin.as_str(), self.seq)
    }

    /// The serialized size of the row.
    pub fn size(&self) -> usize {
        serde_json::to_vec(self).map(|v| v.len()).unwrap_or(usize::MAX)
    }
}

/// Whether ``origin`` is an origin id: 32 lower-case hexadecimal characters.
pub fn valid_origin(origin: &str) -> bool {
    origin.len() == 32 && origin.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

/// Whether ``name`` can be a session name or a board id: a short lower-case word.
pub fn valid_name(name: &str) -> bool {
    let bytes = name.as_bytes();
    let first = bytes.first().is_some_and(|b| b.is_ascii_lowercase() || b.is_ascii_digit());
    first
        && bytes.len() <= 48
        && bytes.iter().all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || matches!(b, b'.' | b'_' | b'-'))
        && !name.contains("..")
        && !RESERVED.contains(&name)
}

const RESERVED: [&str; 7] = ["*", "all", "everyone", "workspace", "system", "human", "owner-token"];

/// Whether ``actor`` can write a row: a name, or a name from a linked board as `name@board`.
pub fn valid_actor(actor: &str) -> bool {
    match actor.split_once('@') {
        None => valid_name(actor),
        Some((name, board)) => valid_name(name) && valid_name(board),
    }
}

/// Whether ``text`` is one clean line: no control characters.
pub fn is_line(text: &str) -> bool {
    !text.chars().any(char::is_control)
}

/// Refuse a body that is not an object of integers, text, booleans, arrays and objects.
pub fn check_body(body: &Value) -> Result<()> {
    if !body.is_object() {
        return Err(Error::Invalid("a body is an object".into()));
    }
    walk(body, 0)
}

fn walk(value: &Value, depth: usize) -> Result<()> {
    if depth > MAX_DEPTH {
        return Err(Error::Invalid("a body nests too deep".into()));
    }
    match value {
        Value::Number(n) if !(n.is_i64() || n.is_u64()) => Err(Error::Invalid("a body holds no fractions".into())),
        Value::Array(items) => items.iter().try_for_each(|v| walk(v, depth + 1)),
        Value::Object(map) => map.values().try_for_each(|v| walk(v, depth + 1)),
        _ => Ok(()),
    }
}
