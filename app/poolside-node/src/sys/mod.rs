//! What differs by operating system: private files and directories, durable replacement, the
//! single-instance lock, who is on the other end of the local API, the local API's own
//! transport (a Unix socket, a Windows named pipe), and whether a process is the one a lease
//! recorded. Everything above this module is the same code on every platform.

use std::path::{Path, PathBuf};

use crate::error::Result;

#[cfg(unix)]
mod unix;
#[cfg(unix)]
pub use unix::*;

#[cfg(windows)]
mod win;
#[cfg(windows)]
pub use win::*;

/// Who is on the other end of the local API: a user id on Unix, a SID on Windows.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Principal(String);

impl Principal {
    pub(crate) fn new(text: String) -> Principal {
        Principal(text)
    }

    /// The user this process runs as.
    pub fn me() -> Result<Principal> {
        current_principal()
    }

    /// Someone who is nobody this process runs as (tests use it to see the refusal).
    pub fn stranger() -> Principal {
        Principal("a stranger".into())
    }
}

/// Where the node of the state directory ``state`` listens: a socket file, or a pipe name.
pub fn endpoint(state: &Path) -> PathBuf {
    endpoint_of(state)
}

/// The 32 hex characters that name the pipe and the stop event of a node: the SHA-256 of its
/// state directory's absolute path with backslashes, lower case, no verbatim prefix and no
/// trailing separator. `ml_stack.node_health.key_of` computes the same.
pub fn key_of(path: &str) -> String {
    let text = path.replace('/', "\\");
    let text = text.strip_prefix(r"\\?\").unwrap_or(&text).trim_end_matches('\\').to_lowercase();
    crate::fsutil::sha256_hex(text.as_bytes())[..32].to_string()
}

#[cfg(test)]
mod tests {
    use super::key_of;

    #[test]
    fn the_key_of_a_path_is_the_one_the_python_side_computes() {
        let want = "303c7027db4da393a0c6c375056106d2";
        assert_eq!(key_of(r"C:\Users\Me\State\node"), want);
        assert_eq!(key_of(r"\\?\c:\users\me\state\node\"), want, "case, a verbatim prefix and a trailing separator do not matter");
        assert_eq!(key_of("C:/Users/Me/State/node"), want);
    }
}
