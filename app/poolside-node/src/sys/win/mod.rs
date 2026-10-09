//! The Windows side: DACLs for owner-only access, `LockFileEx`, `MoveFileEx` with write-through,
//! and the named pipe. Every function here is `unsafe` FFI wrapped once, with its failure
//! turned into an `io::Error`, so nothing above this module sees a handle.

use std::ffi::OsStr;
use std::fs::{File, OpenOptions};
use std::os::windows::ffi::OsStrExt;
use std::os::windows::io::AsRawHandle;
use std::os::windows::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::Command;

use windows_sys::Win32::Foundation::{CloseHandle, GetLastError, HANDLE, INVALID_HANDLE_VALUE};
use windows_sys::Win32::Storage::FileSystem::{LockFileEx, MoveFileExW, LOCKFILE_EXCLUSIVE_LOCK, LOCKFILE_FAIL_IMMEDIATELY, MOVEFILE_REPLACE_EXISTING, MOVEFILE_WRITE_THROUGH};
use windows_sys::Win32::System::IO::OVERLAPPED;

use crate::error::Result;

mod acl;
mod pipe;
mod proc;
mod stop;

pub use acl::{owner_only, private_dir};
pub use pipe::{connect, peer, Listener, Stream};
pub use proc::process_info;
pub use stop::{signal_stop, stop_event_name, watch_stop};

pub(in crate::sys) use acl::current_principal;
pub(in crate::sys) use pipe::endpoint_of;

const ERROR_LOCK_VIOLATION: u32 = 33;
const CREATE_NEW_PROCESS_GROUP: u32 = 0x0000_0200;
const DETACHED_PROCESS: u32 = 0x0000_0008;
/// The byte the Python side's `ml_stack.lock` locks too, far past any text a person reads.
const LOCKED_BYTE: u32 = 1 << 30;

/// A kernel handle, closed when dropped.
pub(crate) struct Handle(pub(crate) HANDLE);

// SAFETY: a kernel handle may be used and closed from any thread.
unsafe impl Send for Handle {}
// SAFETY: the calls made on a shared handle are the thread-safe ones of the Win32 API.
unsafe impl Sync for Handle {}

impl Handle {
    /// The handle, or the last error when the call returned a null or invalid one.
    pub(crate) fn checked(raw: HANDLE) -> std::io::Result<Handle> {
        if raw.is_null() || raw == INVALID_HANDLE_VALUE {
            return Err(std::io::Error::last_os_error());
        }
        Ok(Handle(raw))
    }
}

impl Drop for Handle {
    fn drop(&mut self) {
        if !self.0.is_null() {
            // SAFETY: the handle is owned by this value and closed once.
            unsafe { CloseHandle(self.0) };
        }
    }
}

/// ``path`` as the NUL-ended UTF-16 the wide calls take.
pub(crate) fn wide(text: impl AsRef<OsStr>) -> Vec<u16> {
    text.as_ref().encode_wide().chain(std::iter::once(0)).collect()
}

/// A directory entry is on disk once the rename that made it was written through; nothing to flush.
pub fn sync_dir(_path: &Path) -> Result<()> {
    Ok(())
}

/// Move ``from`` over ``to`` in one step and write it through to disk.
pub fn replace_file(from: &Path, to: &Path) -> Result<()> {
    let (from, to) = (wide(from), wide(to));
    // SAFETY: both are NUL-ended and live for the call.
    if unsafe { MoveFileExW(from.as_ptr(), to.as_ptr(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) } == 0 {
        return Err(std::io::Error::last_os_error().into());
    }
    Ok(())
}

/// Files made under a private directory inherit its owner-only ACL, so nothing is set per file.
pub fn private_file(options: &mut OpenOptions) -> &mut OpenOptions {
    options
}

/// Open the append-only log at ``path``. It is opened for writing at a position, not for
/// appending, because an append-only handle cannot be cut back; the log seeks to the end.
pub fn open_log(path: &Path) -> Result<File> {
    Ok(OpenOptions::new().read(true).write(true).create(true).truncate(false).open(path)?)
}

/// This machine's name, or empty.
pub fn hostname() -> String {
    std::env::var("COMPUTERNAME").unwrap_or_default()
}

/// The directory a user's own state lives under.
pub fn home_dir() -> PathBuf {
    PathBuf::from(std::env::var_os("USERPROFILE").unwrap_or_default())
}

/// A held lock on a file; released when dropped or when the holder dies.
pub struct Lock {
    _file: File,
}

fn lock_file(path: &Path, blocking: bool) -> Result<Option<Lock>> {
    let file = OpenOptions::new().read(true).write(true).create(true).truncate(false).open(path)?;
    // SAFETY: an all-zero OVERLAPPED is the documented starting state.
    let mut at: OVERLAPPED = unsafe { std::mem::zeroed() };
    at.Anonymous.Anonymous.Offset = LOCKED_BYTE;
    let flags = LOCKFILE_EXCLUSIVE_LOCK | if blocking { 0 } else { LOCKFILE_FAIL_IMMEDIATELY };
    // SAFETY: the handle is open for the call and `at` is valid for it.
    if unsafe { LockFileEx(file.as_raw_handle(), flags, 0, 1, 0, &mut at) } != 0 {
        return Ok(Some(Lock { _file: file }));
    }
    // SAFETY: reads the calling thread's last error.
    if unsafe { GetLastError() } == ERROR_LOCK_VIOLATION {
        return Ok(None);
    }
    Err(std::io::Error::last_os_error().into())
}

/// Take the lock at ``path`` without waiting; None when someone holds it.
pub fn try_lock(path: &Path) -> Result<Option<Lock>> {
    lock_file(path, false)
}

/// Take the lock at ``path``, waiting for its holder.
pub fn lock(path: &Path) -> Result<Lock> {
    lock_file(path, true).map(|l| l.expect("a blocking lock is held when it returns"))
}

/// Start ``command`` with no console and in a group of its own, so it outlives this process and a Ctrl+C here.
pub fn spawn_detached(command: &mut Command) -> std::io::Result<std::process::Child> {
    command.creation_flags(CREATE_NEW_PROCESS_GROUP | DETACHED_PROCESS).spawn()
}

/// The key of the absolute path of ``state`` (see `key_of`).
pub(crate) fn state_key(state: &Path) -> String {
    super::key_of(&std::path::absolute(state).unwrap_or_else(|_| state.to_path_buf()).to_string_lossy())
}
