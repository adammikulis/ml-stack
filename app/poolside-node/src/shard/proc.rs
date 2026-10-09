//! The executor's process group: start it as the leader of one, ask it to stop, kill all of it.
//!
//! The test runner and every worker it starts stay in the executor's group, so one kill ends the
//! whole run. Unix signals the group; Windows asks `taskkill /T` for the tree.

use std::process::Command;

#[cfg(unix)]
pub fn group_command(cmd: &mut Command) {
    use std::os::unix::process::CommandExt;
    cmd.process_group(0);
}

#[cfg(windows)]
pub fn group_command(cmd: &mut Command) {
    use std::os::windows::process::CommandExt;
    const CREATE_NEW_PROCESS_GROUP: u32 = 0x0000_0200;
    const CREATE_NO_WINDOW: u32 = 0x0800_0000;
    cmd.creation_flags(CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW);
}

/// Ask the executor (not its children) to stop: it interrupts its runner, then ends the group.
#[cfg(unix)]
pub fn terminate(pid: u32) {
    // SAFETY: kill(2) with a pid this node started and a signal number; it touches no memory.
    unsafe { libc::kill(pid as i32, libc::SIGTERM) };
}

#[cfg(windows)]
pub fn terminate(pid: u32) {
    let _ = Command::new("taskkill").args(["/PID", &pid.to_string()]).output();
}

/// End the executor, its runner and everything they started.
#[cfg(unix)]
pub fn kill_group(pid: u32) {
    // SAFETY: a negative pid names the process group the executor leads; no memory is touched.
    unsafe { libc::kill(-(pid as i32), libc::SIGKILL) };
}

#[cfg(windows)]
pub fn kill_group(pid: u32) {
    let _ = Command::new("taskkill").args(["/T", "/F", "/PID", &pid.to_string()]).output();
}
