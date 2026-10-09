//! Framing of the local API and of the peer protocol.

use std::io::{Read, Write};

use serde_json::Value;

use crate::error::{Error, Result};

/// The largest frame either side will read.
pub const MAX_FRAME: usize = 1024 * 1024;

/// The largest frame a peer will read: a sync batch is up to 2 MiB of rows.
pub const MAX_PEER_FRAME: usize = 4 * 1024 * 1024;

/// Write ``value`` as a 4-byte big-endian length and its JSON.
pub fn write_frame(out: &mut impl Write, value: &Value) -> Result<()> {
    write_frame_max(out, value, MAX_FRAME)
}

/// Like `write_frame`, with a frame limit of ``max`` bytes.
pub fn write_frame_max(out: &mut impl Write, value: &Value, max: usize) -> Result<()> {
    let bytes = serde_json::to_vec(value)?;
    if bytes.len() > max {
        return Err(Error::Quota("the frame is larger than a frame may be".into()));
    }
    out.write_all(&(bytes.len() as u32).to_be_bytes())?;
    out.write_all(&bytes)?;
    out.flush()?;
    Ok(())
}

/// Read one frame; None when the peer closed before a new one.
pub fn read_frame(input: &mut impl Read) -> Result<Option<Value>> {
    read_frame_max(input, MAX_FRAME)
}

/// Like `read_frame`, with a frame limit of ``max`` bytes.
pub fn read_frame_max(input: &mut impl Read, max: usize) -> Result<Option<Value>> {
    let mut len = [0u8; 4];
    match input.read_exact(&mut len) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => return Err(e.into()),
    }
    let len = u32::from_be_bytes(len) as usize;
    if len > max {
        return Err(Error::Quota("the frame is larger than a frame may be".into()));
    }
    let mut bytes = vec![0u8; len];
    input.read_exact(&mut bytes)?;
    Ok(Some(serde_json::from_slice(&bytes)?))
}
