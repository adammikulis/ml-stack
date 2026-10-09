//! What this device is, which Python runs its shards, and what a finished run left behind.

use std::collections::BTreeMap;
use std::io::Read;
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use serde_json::{json, Value};

use super::host::MOST_RESULT;
use crate::error::{Error, Result};

/// The only Python the tests run on.
pub const PYTHON: &str = "3.13";

/// Where this node runs, in the words `sys.platform` and `platform.machine()` use; `wsl` marks
/// a Linux that is Windows Subsystem for Linux, which is a different device from the Windows
/// around it (and has its own node, key and certificate).
pub fn platform() -> Value {
    let system = match std::env::consts::OS {
        "macos" => "darwin",
        "windows" => "win32",
        other => other,
    };
    let wsl = std::env::consts::OS == "linux"
        && (std::env::var_os("WSL_DISTRO_NAME").is_some()
            || std::fs::read_to_string("/proc/sys/kernel/osrelease").is_ok_and(|t| t.to_lowercase().contains("microsoft")));
    json!({"system": system, "machine": std::env::consts::ARCH, "cpus": std::thread::available_parallelism().map_or(1, |n| n.get()), "wsl": wsl})
}

/// A probe that outlasts this is killed: the node's lock is held while it runs.
const PROBE_LIMIT: Duration = Duration::from_secs(10);
/// A Python that answered 3.13 is not asked again for this long.
const FRESH: Duration = Duration::from_secs(60);
static VERIFIED: OnceLock<Mutex<BTreeMap<String, Instant>>> = OnceLock::new();

fn probe(python: &str) -> std::io::Result<String> {
    let mut child = Command::new(python).args(["-c", "import sys; print('%d.%d' % sys.version_info[:2])"])
        .stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::null()).spawn()?;
    let end = Instant::now() + PROBE_LIMIT;
    while child.try_wait()?.is_none() {
        if Instant::now() > end {
            let _ = child.kill();
            let _ = child.wait();
            return Err(std::io::Error::new(std::io::ErrorKind::TimedOut, "it did not answer in 10 seconds"));
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    let mut text = String::new();
    if let Some(mut out) = child.stdout.take() {
        let _ = out.read_to_string(&mut text);
    }
    Ok(text.trim().to_string())
}

/// ``python``'s `major.minor` when it is 3.13; otherwise the sentence that says what to do.
pub fn python_version(python: &str) -> Result<String> {
    if python.is_empty() {
        return Err(Error::Invalid("no Python is set for shards: run `python -m ml_stack.testfarm.consent on` with this device's Python 3.13".into()));
    }
    let known = VERIFIED.get_or_init(|| Mutex::new(BTreeMap::new()));
    if known.lock().is_ok_and(|k| k.get(python).is_some_and(|at| at.elapsed() < FRESH)) {
        return Ok(PYTHON.into());
    }
    let found = probe(python).map_err(|e| Error::Invalid(format!("Python 3.13 is missing: {python} did not run ({e}); install it and run `python -m ml_stack.testfarm.consent on`")))?;
    if found != PYTHON {
        return Err(Error::Invalid(format!("Python 3.13 is needed and {python} is {}; install 3.13 and run `python -m ml_stack.testfarm.consent on` with it", if found.is_empty() { "not a working Python" } else { &found })));
    }
    if let Ok(mut k) = known.lock() {
        k.insert(python.into(), Instant::now());
    }
    Ok(found)
}

/// The executor's result file, if it wrote one; `Err` when it is larger than a reply may carry.
pub fn read_result(dir: &Path) -> Result<Option<Value>> {
    let path = dir.join("result.json");
    let Ok(meta) = std::fs::metadata(&path) else { return Ok(None) };
    if meta.len() > MOST_RESULT {
        return Err(Error::Quota(format!("the result is {} bytes; a reply carries at most {MOST_RESULT}", meta.len())));
    }
    Ok(serde_json::from_slice(&std::fs::read(path)?).ok())
}

/// The last 2000 bytes of a log, lossy.
pub fn tail(path: &Path) -> String {
    let mut text = Vec::new();
    if let Ok(file) = std::fs::File::open(path) {
        let _ = file.take(1 << 20).read_to_end(&mut text);
    }
    let start = text.len().saturating_sub(2000);
    String::from_utf8_lossy(&text[start..]).trim().to_string()
}
