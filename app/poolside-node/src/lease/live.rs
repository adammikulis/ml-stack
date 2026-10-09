//! Whether the process behind a lease is still the process that took it: the pid exists, is
//! not a zombie, and started when it started at acquire time (a reused pid did not).

/// What the system says about a process.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Proc {
    /// Start time in milliseconds on the system's own clock; only compared with itself.
    pub start: u64,
    /// Exited but not yet reaped by its parent.
    pub zombie: bool,
}

#[cfg(target_os = "macos")]
pub fn inspect(pid: u32) -> Option<Proc> {
    const ZOMBIE: u32 = 5;
    let size = std::mem::size_of::<libc::proc_bsdinfo>();
    // SAFETY: `info` is a zeroed plain-data struct of exactly `size` bytes, which proc_pidinfo fills.
    unsafe {
        let mut info: libc::proc_bsdinfo = std::mem::zeroed();
        let got = libc::proc_pidinfo(pid as i32, libc::PROC_PIDTBSDINFO, 0, (&mut info as *mut libc::proc_bsdinfo).cast(), size as i32);
        if got as usize != size {
            return None;
        }
        Some(Proc { start: info.pbi_start_tvsec * 1000 + info.pbi_start_tvusec / 1000, zombie: info.pbi_status == ZOMBIE })
    }
}

#[cfg(target_os = "linux")]
pub fn inspect(pid: u32) -> Option<Proc> {
    let stat = std::fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
    // the command name may hold spaces and parentheses; everything after the last `)` is fields
    let rest: Vec<&str> = stat.rsplit_once(')')?.1.split_whitespace().collect();
    let ticks: u64 = rest.get(19)?.parse().ok()?;
    // SAFETY: sysconf reads a constant.
    let hz = unsafe { libc::sysconf(libc::_SC_CLK_TCK) }.max(1) as u64;
    Some(Proc { start: ticks * 1000 / hz, zombie: rest.first() == Some(&"Z") })
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
pub fn inspect(pid: u32) -> Option<Proc> {
    // SAFETY: signal 0 only checks that the process exists.
    let found = unsafe { libc::kill(pid as i32, 0) } == 0 || std::io::Error::last_os_error().raw_os_error() == Some(libc::EPERM);
    found.then_some(Proc { start: 0, zombie: false })
}

/// The start time to record for ``pid`` now, or None when it cannot be read.
pub fn start_of(pid: u32) -> Option<u64> {
    inspect(pid).filter(|p| !p.zombie).map(|p| p.start)
}

/// Whether the holder is still the process it was: it exists, is not a zombie, and (when a
/// start time was recorded) started at that moment.
pub fn is_alive(pid: u32, recorded_start: Option<u64>) -> bool {
    match inspect(pid) {
        None => false,
        Some(p) if p.zombie => false,
        Some(p) => recorded_start.is_none_or(|s| s == p.start),
    }
}
