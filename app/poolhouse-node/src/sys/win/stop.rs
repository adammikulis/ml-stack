//! A graceful stop for a node that has no terminal: a named event (`Local\poolhouse-node-stop-<key>`,
//! open to the current user alone) that `poolhouse.node_launch` sets, and Ctrl+C / Ctrl+Break /
//! console close for a node run by hand. There is no signal to send a detached Windows process,
//! and a plain terminate would skip the node's exit.

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use windows_sys::core::BOOL;
use windows_sys::Win32::Foundation::WAIT_OBJECT_0;
use windows_sys::Win32::System::Console::SetConsoleCtrlHandler;
use windows_sys::Win32::System::Threading::{CreateEventW, OpenEventW, SetEvent, WaitForSingleObject, EVENT_MODIFY_STATE};

use super::acl::only_me;
use super::{state_key, wide, Handle};

const POLL_MS: u32 = 50;
/// Ctrl+C, Ctrl+Break and the console closing.
const CTRL_EVENTS: [u32; 3] = [0, 1, 2];

static CTRL: AtomicBool = AtomicBool::new(false);

unsafe extern "system" fn on_ctrl(kind: u32) -> BOOL {
    if CTRL_EVENTS.contains(&kind) {
        CTRL.store(true, Ordering::SeqCst);
        return 1;
    }
    0
}

/// The name of the event whose signal stops the node of ``state``.
pub fn stop_event_name(state: &Path) -> String {
    format!(r"Local\poolhouse-node-stop-{}", state_key(state))
}

/// Set ``stop`` when the stop event is signalled or the console asks the process to end; the
/// watcher goes when ``stop`` is set or nothing else holds it any more.
pub fn watch_stop(state: &Path, stop: Arc<AtomicBool>) {
    // SAFETY: registers a handler that only stores to a static.
    unsafe { SetConsoleCtrlHandler(Some(on_ctrl), 1) };
    let Ok(only) = only_me("GA", "") else { return };
    let attributes = only.attributes();
    let name = wide(stop_event_name(state));
    // SAFETY: `name` is NUL-ended and `attributes` points into `only`; both outlive the call.
    let Ok(event) = Handle::checked(unsafe { CreateEventW(&attributes, 1, 0, name.as_ptr()) }) else { return };
    std::thread::spawn(move || {
        let event = event;
        while Arc::strong_count(&stop) > 1 && !stop.load(Ordering::SeqCst) {
            // SAFETY: the event is owned by this thread.
            let signalled = unsafe { WaitForSingleObject(event.0, POLL_MS) } == WAIT_OBJECT_0;
            if signalled || CTRL.load(Ordering::SeqCst) {
                stop.store(true, Ordering::SeqCst);
            }
        }
    });
}

/// Signal the stop event of the node of ``state``; false when no node holds it.
pub fn signal_stop(state: &Path) -> bool {
    let name = wide(stop_event_name(state));
    // SAFETY: `name` is NUL-ended; the handle is ours.
    let Ok(event) = Handle::checked(unsafe { OpenEventW(EVENT_MODIFY_STATE, 0, name.as_ptr()) }) else { return false };
    // SAFETY: the handle is open for the call.
    unsafe { SetEvent(event.0) != 0 }
}
