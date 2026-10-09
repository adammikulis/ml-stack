//! Framing, peer identity and the single-instance lock of the local socket.

use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::os::fd::AsRawFd;
use std::os::unix::fs::OpenOptionsExt;
use std::os::unix::net::UnixStream;
use std::path::Path;

use serde_json::Value;

use crate::error::{Error, Result};

/// The largest frame either side will read.
pub const MAX_FRAME: usize = 1024 * 1024;

/// Write ``value`` as a 4-byte big-endian length and its JSON.
pub fn write_frame(out: &mut impl Write, value: &Value) -> Result<()> {
    let bytes = serde_json::to_vec(value)?;
    if bytes.len() > MAX_FRAME {
        return Err(Error::Quota("the frame is larger than a frame may be".into()));
    }
    out.write_all(&(bytes.len() as u32).to_be_bytes())?;
    out.write_all(&bytes)?;
    out.flush()?;
    Ok(())
}

/// Read one frame; None when the peer closed before a new one.
pub fn read_frame(input: &mut impl Read) -> Result<Option<Value>> {
    let mut len = [0u8; 4];
    match input.read_exact(&mut len) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => return Err(e.into()),
    }
    let len = u32::from_be_bytes(len) as usize;
    if len > MAX_FRAME {
        return Err(Error::Quota("the frame is larger than a frame may be".into()));
    }
    let mut bytes = vec![0u8; len];
    input.read_exact(&mut bytes)?;
    Ok(Some(serde_json::from_slice(&bytes)?))
}

/// The user id of the process on the other end of ``stream``.
pub fn peer_uid(stream: &UnixStream) -> Result<u32> {
    #[cfg(target_os = "linux")]
    {
        let mut cred = libc::ucred { pid: 0, uid: 0, gid: 0 };
        let mut len = std::mem::size_of::<libc::ucred>() as libc::socklen_t;
        // SAFETY: `cred` and `len` are valid for the call and sized as getsockopt expects.
        let rc = unsafe { libc::getsockopt(stream.as_raw_fd(), libc::SOL_SOCKET, libc::SO_PEERCRED, (&mut cred as *mut libc::ucred).cast(), &mut len) };
        if rc != 0 {
            return Err(std::io::Error::last_os_error().into());
        }
        Ok(cred.uid)
    }
    #[cfg(not(target_os = "linux"))]
    {
        let (mut uid, mut gid) = (0, 0);
        // SAFETY: the out-pointers are valid for the call.
        if unsafe { libc::getpeereid(stream.as_raw_fd(), &mut uid, &mut gid) } != 0 {
            return Err(std::io::Error::last_os_error().into());
        }
        Ok(uid)
    }
}

/// The user id this process runs as.
pub fn my_uid() -> u32 {
    // SAFETY: getuid has no preconditions.
    unsafe { libc::getuid() }
}

/// A held advisory lock on a file; released when dropped.
pub struct Lock {
    _file: File,
}

fn flock(path: &Path, blocking: bool) -> Result<Option<Lock>> {
    let file = OpenOptions::new().read(true).write(true).create(true).truncate(false).mode(0o600).open(path)?;
    let op = if blocking { libc::LOCK_EX } else { libc::LOCK_EX | libc::LOCK_NB };
    // SAFETY: the descriptor is open for the whole call.
    if unsafe { libc::flock(file.as_raw_fd(), op) } == 0 {
        return Ok(Some(Lock { _file: file }));
    }
    let err = std::io::Error::last_os_error();
    if err.kind() == std::io::ErrorKind::WouldBlock { Ok(None) } else { Err(err.into()) }
}

/// Take the lock at ``path`` without waiting; None when someone holds it.
pub fn try_lock(path: &Path) -> Result<Option<Lock>> {
    flock(path, false)
}

/// Take the lock at ``path``, waiting for its holder.
pub fn lock(path: &Path) -> Result<Lock> {
    flock(path, true).map(|l| l.expect("a blocking lock is held when it returns"))
}
