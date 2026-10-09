//! The local API over a named pipe: byte mode, overlapped so a read can time out and the
//! listener can poll like a non-blocking socket, a DACL that admits the current user alone,
//! remote clients refused, and the first instance created exclusively.

use std::cell::Cell;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use windows_sys::Win32::Foundation::{GetLastError, GENERIC_READ, GENERIC_WRITE, HANDLE, WAIT_OBJECT_0};
use windows_sys::Win32::Security::{RevertToSelf, TOKEN_QUERY};
use windows_sys::Win32::Storage::FileSystem::{
    CreateFileW, ReadFile, WriteFile, FILE_FLAG_FIRST_PIPE_INSTANCE, FILE_FLAG_OVERLAPPED, OPEN_EXISTING, PIPE_ACCESS_DUPLEX, SECURITY_IDENTIFICATION, SECURITY_SQOS_PRESENT,
};
use windows_sys::Win32::System::IO::{CancelIoEx, GetOverlappedResult, OVERLAPPED};
use windows_sys::Win32::System::Pipes::{
    ConnectNamedPipe, CreateNamedPipeW, GetNamedPipeServerProcessId, ImpersonateNamedPipeClient, WaitNamedPipeW, PIPE_READMODE_BYTE, PIPE_REJECT_REMOTE_CLIENTS, PIPE_TYPE_BYTE,
    PIPE_UNLIMITED_INSTANCES, PIPE_WAIT,
};
use windows_sys::Win32::System::Threading::{CreateEventW, GetCurrentThread, OpenThreadToken, ResetEvent, SetEvent, WaitForSingleObject};

use super::acl::{only_me, process_principal, token_sid, Descriptor};
use super::{state_key, wide, Handle};
use crate::error::{Error, Result};
use crate::sys::Principal;

const ERROR_BROKEN_PIPE: u32 = 109;
const ERROR_PIPE_BUSY: u32 = 231;
const ERROR_IO_PENDING: u32 = 997;
const ERROR_PIPE_CONNECTED: u32 = 535;
const ERROR_ACCESS_DENIED: u32 = 5;
const BUFFER: u32 = 64 * 1024;
const ACCEPT_POLL_MS: u32 = 10;
const CONNECT_WAIT: Duration = Duration::from_secs(2);

pub(in crate::sys) fn endpoint_of(state: &Path) -> PathBuf {
    PathBuf::from(format!(r"\\.\pipe\poolside-node-{}", state_key(state)))
}

fn last() -> u32 {
    // SAFETY: reads the calling thread's last error.
    unsafe { GetLastError() }
}

/// A manual-reset event, closed when dropped.
struct Event(Handle);

impl Event {
    fn new() -> std::io::Result<Event> {
        // SAFETY: an unnamed manual-reset, unsignalled event.
        Handle::checked(unsafe { CreateEventW(std::ptr::null(), 1, 0, std::ptr::null()) }).map(Event)
    }
}

/// One end of a connected pipe.
pub struct Stream {
    handle: Handle,
    read_timeout: Cell<Option<Duration>>,
}

impl Stream {
    fn new(handle: Handle) -> Stream {
        Stream { handle, read_timeout: Cell::new(None) }
    }

    /// Reads wait at most ``timeout`` (None: for ever) and then fail with `TimedOut`.
    pub fn set_read_timeout(&self, timeout: Option<Duration>) -> std::io::Result<()> {
        self.read_timeout.set(timeout);
        Ok(())
    }

    /// One overlapped transfer: start it with ``start``, wait up to ``timeout``, cancel it if that passes.
    fn transfer(&self, timeout: Option<Duration>, start: impl FnOnce(*mut OVERLAPPED) -> i32) -> std::io::Result<usize> {
        let event = Event::new()?;
        // SAFETY: an all-zero OVERLAPPED is the documented starting state.
        let mut at: OVERLAPPED = unsafe { std::mem::zeroed() };
        at.hEvent = (event.0).0;
        let mut done = 0u32;
        if start(&mut at) == 0 {
            match last() {
                ERROR_BROKEN_PIPE => return Ok(0),
                ERROR_IO_PENDING => {}
                _ => return Err(std::io::Error::last_os_error()),
            }
            let ms = timeout.map_or(u32::MAX, |t| t.as_millis().min(u32::MAX as u128 - 1) as u32);
            // SAFETY: the event belongs to this call.
            if unsafe { WaitForSingleObject((event.0).0, ms) } != WAIT_OBJECT_0 {
                // SAFETY: cancels this call's own transfer, then waits until the system is done with `at`.
                unsafe {
                    CancelIoEx(self.handle.0, &at);
                    GetOverlappedResult(self.handle.0, &at, &mut done, 1);
                }
                return Err(std::io::ErrorKind::TimedOut.into());
            }
        }
        // SAFETY: the transfer has completed, so this only collects its size.
        if unsafe { GetOverlappedResult(self.handle.0, &at, &mut done, 1) } == 0 {
            return if last() == ERROR_BROKEN_PIPE { Ok(0) } else { Err(std::io::Error::last_os_error()) };
        }
        Ok(done as usize)
    }
}

impl Read for Stream {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let (handle, len) = (self.handle.0, buf.len().min(u32::MAX as usize) as u32);
        // SAFETY: `buf` stays valid until `transfer` returns, which waits out or cancels the read.
        self.transfer(self.read_timeout.get(), |at| unsafe { ReadFile(handle, buf.as_mut_ptr().cast(), len, std::ptr::null_mut(), at) })
    }
}

impl Write for Stream {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let (handle, len) = (self.handle.0, buf.len().min(u32::MAX as usize) as u32);
        // SAFETY: `buf` stays valid until `transfer` returns, which waits out or cancels the write.
        self.transfer(None, |at| unsafe { WriteFile(handle, buf.as_ptr().cast(), len, std::ptr::null_mut(), at) })
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// The user on the other end of a pipe the node serves.
pub fn peer(stream: &Stream) -> Result<Principal> {
    // SAFETY: the handle is a connected pipe.
    if unsafe { ImpersonateNamedPipeClient(stream.handle.0) } == 0 {
        return Err(std::io::Error::last_os_error().into());
    }
    let mut token: HANDLE = std::ptr::null_mut();
    // SAFETY: opens the impersonation token of this thread, as the process, for a handle we own.
    let opened = unsafe { OpenThreadToken(GetCurrentThread(), TOKEN_QUERY, 1, &mut token) };
    let seen = if opened == 0 { Err(std::io::Error::last_os_error()) } else { Handle::checked(token).and_then(|t| token_sid(t.0)) };
    // SAFETY: ends the impersonation begun above, whatever came of it.
    unsafe { RevertToSelf() };
    Ok(Principal::new(seen?))
}

struct Pending {
    handle: Handle,
    at: Box<OVERLAPPED>,
    event: Event,
}

// SAFETY: the OVERLAPPED is touched by one thread at a time, under the listener's mutex.
unsafe impl Send for Pending {}

impl Pending {
    /// A new pipe instance, waiting for a client.
    fn new(name: &[u16], sd: &Descriptor, first: bool) -> Result<Pending> {
        let attributes = sd.attributes();
        let open = PIPE_ACCESS_DUPLEX | FILE_FLAG_OVERLAPPED | if first { FILE_FLAG_FIRST_PIPE_INSTANCE } else { 0 };
        let mode = PIPE_TYPE_BYTE | PIPE_READMODE_BYTE | PIPE_WAIT | PIPE_REJECT_REMOTE_CLIENTS;
        // SAFETY: `name` is NUL-ended and `attributes` points into `sd`; both outlive the call.
        let raw = unsafe { CreateNamedPipeW(name.as_ptr(), open, mode, PIPE_UNLIMITED_INSTANCES, BUFFER, BUFFER, 0, &attributes) };
        let handle = Handle::checked(raw).map_err(|e| match e.raw_os_error().map(|c| c as u32) {
            Some(ERROR_ACCESS_DENIED | ERROR_PIPE_BUSY) if first => Error::Denied("a node already runs on this state directory".into()),
            _ => Error::Io(e),
        })?;
        let event = Event::new()?;
        // SAFETY: an all-zero OVERLAPPED is the documented starting state.
        let mut at: Box<OVERLAPPED> = Box::new(unsafe { std::mem::zeroed() });
        at.hEvent = (event.0).0;
        let pending = Pending { handle, at, event };
        pending.listen()?;
        Ok(pending)
    }

    fn listen(&self) -> std::io::Result<()> {
        let at: *const OVERLAPPED = &*self.at;
        // SAFETY: the event and `at` live as long as the instance, and `Drop` waits out the call.
        unsafe { ResetEvent((self.event.0).0) };
        // SAFETY: as above.
        if unsafe { ConnectNamedPipe(self.handle.0, at.cast_mut()) } != 0 {
            return Ok(());
        }
        match last() {
            ERROR_IO_PENDING => Ok(()),
            // SAFETY: a client got in before the call; the event says so the way a finished wait would.
            ERROR_PIPE_CONNECTED => {
                unsafe { SetEvent((self.event.0).0) };
                Ok(())
            }
            _ => Err(std::io::Error::last_os_error()),
        }
    }
}

impl Drop for Pending {
    fn drop(&mut self) {
        let mut done = 0u32;
        // SAFETY: cancels the connect still waiting on `at`, then waits until the system is done with it.
        unsafe {
            CancelIoEx(self.handle.0, &*self.at);
            GetOverlappedResult(self.handle.0, &*self.at, &mut done, 1);
        }
    }
}

/// The pipe the node of a state directory listens on.
pub struct Listener {
    name: Vec<u16>,
    sd: Descriptor,
    next: Mutex<Option<Pending>>,
}

impl Listener {
    /// Create the pipe of ``state``, exclusively: `Denied` when another node holds its name.
    pub fn bind(state: &Path) -> Result<Listener> {
        let name = wide(endpoint_of(state));
        let sd = only_me("GA", "")?;
        let first = Pending::new(&name, &sd, true)?;
        Ok(Listener { name, sd, next: Mutex::new(Some(first)) })
    }

    /// The next client; `WouldBlock` when none is waiting.
    pub fn accept(&self) -> std::io::Result<Stream> {
        let mut slot = self.next.lock().map_err(|_| std::io::Error::other("the pipe lock is poisoned"))?;
        let waiting = slot.as_ref().ok_or_else(|| std::io::Error::other("the pipe has no instance"))?;
        // SAFETY: the event belongs to the pending instance.
        if unsafe { WaitForSingleObject((waiting.event.0).0, ACCEPT_POLL_MS) } != WAIT_OBJECT_0 {
            return Err(std::io::ErrorKind::WouldBlock.into());
        }
        // The event is set: a client connected, either through the overlapped call or before it (then the
        // OVERLAPPED was never filled in, so its result is not asked for). A connect that failed shows as a
        // failed first read, which closes that one connection.
        let replacement = Pending::new(&self.name, &self.sd, false).map_err(|e| std::io::Error::other(e.to_string()))?;
        let mut connected = slot.replace(replacement).ok_or_else(|| std::io::Error::other("the pipe has no instance"))?;
        Ok(Stream::new(std::mem::replace(&mut connected.handle, Handle(std::ptr::null_mut()))))
    }
}

/// Connect to the node of ``state``, waiting briefly for a busy pipe, and only to a server that runs as this user.
pub fn connect(state: &Path) -> std::io::Result<Stream> {
    let name = wide(endpoint_of(state));
    let until = Instant::now() + CONNECT_WAIT;
    let handle = loop {
        // SAFETY: `name` is NUL-ended; the pipe is opened at identification level so its server can not act as us.
        let raw = unsafe {
            CreateFileW(name.as_ptr(), GENERIC_READ | GENERIC_WRITE, 0, std::ptr::null(), OPEN_EXISTING, FILE_FLAG_OVERLAPPED | SECURITY_SQOS_PRESENT | SECURITY_IDENTIFICATION, std::ptr::null_mut())
        };
        match Handle::checked(raw) {
            Ok(handle) => break handle,
            Err(e) if e.raw_os_error() == Some(ERROR_PIPE_BUSY as i32) && Instant::now() < until => {
                // SAFETY: waits for a free instance of the named pipe.
                unsafe { WaitNamedPipeW(name.as_ptr(), 200) };
            }
            Err(e) => return Err(e),
        }
    };
    let mut server = 0u32;
    // SAFETY: the handle is a connected pipe.
    if unsafe { GetNamedPipeServerProcessId(handle.0, &mut server) } == 0 {
        return Err(std::io::Error::last_os_error());
    }
    let mine = Principal::me().map_err(|e| std::io::Error::other(e.to_string()))?;
    if process_principal(server)? != mine.0 {
        return Err(std::io::Error::new(std::io::ErrorKind::PermissionDenied, "the pipe is served by another user"));
    }
    Ok(Stream::new(handle))
}
