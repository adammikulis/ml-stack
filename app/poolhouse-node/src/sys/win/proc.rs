//! Whether a process exists and when it started: a process handle and its creation time, which
//! is what a reused process id can not share with the process that held it before.

use windows_sys::Win32::Foundation::{GetLastError, FILETIME, STILL_ACTIVE};
use windows_sys::Win32::System::Threading::{GetExitCodeProcess, GetProcessTimes, OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION};

use super::Handle;

const ERROR_ACCESS_DENIED: u32 = 5;
/// 100 ns ticks in a millisecond.
const TICKS_PER_MS: u64 = 10_000;

/// `(creation time in milliseconds since 1601, zombie)` of ``pid``, or None when no such process runs.
/// A process this user may not inspect exists: its start is reported as 0, which means unknown.
pub fn process_info(pid: u32) -> Option<(u64, bool)> {
    // SAFETY: opens a handle we own; a null result is an error.
    let raw = unsafe { OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid) };
    let Ok(process) = Handle::checked(raw) else {
        // SAFETY: reads the calling thread's last error.
        return (unsafe { GetLastError() } == ERROR_ACCESS_DENIED).then_some((0, false));
    };
    let mut code = 0u32;
    // SAFETY: the handle is open for the call.
    if unsafe { GetExitCodeProcess(process.0, &mut code) } == 0 || code != STILL_ACTIVE as u32 {
        return None;
    }
    let empty = FILETIME { dwLowDateTime: 0, dwHighDateTime: 0 };
    let (mut created, mut exited, mut kernel, mut user) = (empty, empty, empty, empty);
    // SAFETY: the handle is open and the out-pointers are valid for the call.
    if unsafe { GetProcessTimes(process.0, &mut created, &mut exited, &mut kernel, &mut user) } == 0 {
        return None;
    }
    Some((((u64::from(created.dwHighDateTime) << 32) | u64::from(created.dwLowDateTime)) / TICKS_PER_MS, false))
}
