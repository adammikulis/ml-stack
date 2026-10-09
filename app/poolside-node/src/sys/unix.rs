//! The Unix side: modes, `flock`, `SO_PEERCRED` and the socket.

use std::fs::{self, File, OpenOptions};
use std::os::fd::AsRawFd;
use std::os::unix::fs::{DirBuilderExt, OpenOptionsExt, PermissionsExt};
use std::os::unix::net::{UnixListener, UnixStream};
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use super::Principal;
use crate::error::Result;

pub type Stream = UnixStream;

const SOCKET: &str = "node.sock";

pub(super) fn endpoint_of(state: &Path) -> PathBuf {
    state.join(SOCKET)
}

/// Create ``path`` and its parents as directories only the owner can enter.
pub fn private_dir(path: &Path) -> Result<()> {
    fs::DirBuilder::new().recursive(true).mode(0o700).create(path)?;
    fs::set_permissions(path, fs::Permissions::from_mode(0o700))?;
    Ok(())
}

/// Whether only the owner can reach ``path``.
pub fn owner_only(path: &Path) -> bool {
    fs::metadata(path).is_ok_and(|m| m.permissions().mode() & 0o077 == 0)
}

/// Flush a directory so a new or renamed entry survives a crash.
pub fn sync_dir(path: &Path) -> Result<()> {
    File::open(path)?.sync_all()?;
    Ok(())
}

/// Move ``from`` over ``to`` in one step.
pub fn replace_file(from: &Path, to: &Path) -> Result<()> {
    fs::rename(from, to)?;
    Ok(())
}

/// Options for a file only its owner can read.
pub fn private_file(options: &mut OpenOptions) -> &mut OpenOptions {
    options.mode(0o600)
}

/// Open the append-only log at ``path``, readable and writable by the owner alone.
pub fn open_log(path: &Path) -> Result<File> {
    Ok(private_file(OpenOptions::new().read(true).append(true).create(true)).open(path)?)
}

/// This machine's name, or empty.
pub fn hostname() -> String {
    let mut buf = [0u8; 256];
    // SAFETY: the buffer is valid for its length for the call.
    let rc = unsafe { libc::gethostname(buf.as_mut_ptr().cast(), buf.len() - 1) };
    if rc == 0 { String::from_utf8_lossy(&buf).trim_end_matches('\0').to_string() } else { String::new() }
}

/// The directory a user's own state lives under.
pub fn home_dir() -> PathBuf {
    PathBuf::from(std::env::var_os("HOME").unwrap_or_default())
}

pub(super) fn current_principal() -> Result<Principal> {
    // SAFETY: getuid has no preconditions.
    Ok(Principal::new(unsafe { libc::getuid() }.to_string()))
}

/// The user of the process on the other end of ``stream``.
pub fn peer(stream: &Stream) -> Result<Principal> {
    #[cfg(target_os = "linux")]
    {
        let mut cred = libc::ucred { pid: 0, uid: 0, gid: 0 };
        let mut len = std::mem::size_of::<libc::ucred>() as libc::socklen_t;
        // SAFETY: `cred` and `len` are valid for the call and sized as getsockopt expects.
        let rc = unsafe { libc::getsockopt(stream.as_raw_fd(), libc::SOL_SOCKET, libc::SO_PEERCRED, (&mut cred as *mut libc::ucred).cast(), &mut len) };
        if rc != 0 {
            return Err(std::io::Error::last_os_error().into());
        }
        Ok(Principal::new(cred.uid.to_string()))
    }
    #[cfg(not(target_os = "linux"))]
    {
        let (mut uid, mut gid) = (0, 0);
        // SAFETY: the out-pointers are valid for the call.
        if unsafe { libc::getpeereid(stream.as_raw_fd(), &mut uid, &mut gid) } != 0 {
            return Err(std::io::Error::last_os_error().into());
        }
        Ok(Principal::new(uid.to_string()))
    }
}

/// A held advisory lock on a file; released when dropped.
pub struct Lock {
    _file: File,
}

fn flock(path: &Path, blocking: bool) -> Result<Option<Lock>> {
    let file = private_file(OpenOptions::new().read(true).write(true).create(true).truncate(false)).open(path)?;
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

/// The socket the node of a state directory listens on.
pub struct Listener {
    inner: UnixListener,
}

impl Listener {
    /// Bind the socket of ``state``, replacing a dead node's file; owner-only.
    pub fn bind(state: &Path) -> Result<Listener> {
        let path = endpoint_of(state);
        let _ = fs::remove_file(&path);
        let inner = UnixListener::bind(&path)?;
        fs::set_permissions(&path, fs::Permissions::from_mode(0o600))?;
        inner.set_nonblocking(true)?;
        Ok(Listener { inner })
    }

    /// The next client; `WouldBlock` when none is waiting.
    pub fn accept(&self) -> std::io::Result<Stream> {
        let (stream, _) = self.inner.accept()?;
        stream.set_nonblocking(false)?;
        Ok(stream)
    }
}

/// Connect to the node of ``state``.
pub fn connect(state: &Path) -> std::io::Result<Stream> {
    UnixStream::connect(endpoint_of(state))
}

/// Start ``command`` so that it outlives this process and its terminal.
pub fn spawn_detached(command: &mut Command) -> std::io::Result<std::process::Child> {
    command.process_group(0).spawn()
}

/// Nothing on Unix: a stop is the `shutdown` request or a signal.
pub fn watch_stop(_state: &Path, _stop: Arc<AtomicBool>) {}
