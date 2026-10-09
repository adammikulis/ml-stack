//! Which project a node works on, so devices of one project find each other and no others.
//!
//! The project is the repository that uses the node: the normalised `origin` remote (the same on
//! every device that cloned it), else the name of the main working tree. It travels as a short
//! key (16 hex digits of a SHA-256), so a remote URL is never put on the network. An owner can
//! name the project instead.

use std::path::Path;
use std::process::Command;

use crate::error::{Error, Result};
use crate::fsutil::sha256_hex;
use crate::project::{common_root, top};

/// A remote written any way, as `host/owner/repo`: no scheme, user, port or `.git`, lower case.
/// `git@example.invalid:Me/Repo.git`, `https://example.invalid/me/repo` and
/// `ssh://git@example.invalid:22/me/repo.git/` all give `example.invalid/me/repo`.
pub fn normalise_remote(url: &str) -> String {
    let url = url.trim();
    let (hostpart, path) = match url.split_once("://") {
        Some((_, rest)) => rest.split_once('/').unwrap_or((rest, "")),
        None => match url.split_once(':') {
            Some((host, path)) if !host.contains(['/', '\\']) && host.len() > 1 => (host, path),
            _ => return url.replace('\\', "/").trim_end_matches('/').trim_end_matches(".git").to_lowercase(),
        },
    };
    let host = hostpart.rsplit('@').next().unwrap_or(hostpart);
    let host = host.split(':').next().unwrap_or(host);
    let path = path.trim_matches('/');
    let path = path.strip_suffix(".git").unwrap_or(path).trim_end_matches('/');
    format!("{host}/{path}").to_lowercase()
}

/// The key a project name travels as.
pub fn project_key(name: &str) -> String {
    sha256_hex(name.trim().as_bytes())[..16].to_string()
}

/// The name of the project containing ``dir``: its normalised `origin` remote, else
/// `local/<name of the main working tree>` (the same from every linked worktree).
pub fn project_name(dir: &Path) -> Result<String> {
    let top = top(dir)?;
    let out = Command::new("git").arg("-C").arg(&top).args(["remote", "get-url", "origin"]).output();
    if let Some(out) = out.ok().filter(|o| o.status.success()) {
        let url = String::from_utf8_lossy(&out.stdout).trim().to_string();
        if !url.is_empty() {
            return Ok(normalise_remote(&url));
        }
    }
    let name = common_root(&top)?.file_name().map(|n| n.to_string_lossy().to_lowercase()).unwrap_or_default();
    if name.is_empty() {
        return Err(Error::Invalid("the repository root has no name to take a project from".into()));
    }
    Ok(format!("local/{name}"))
}

/// The key of the project this device works on: ``over`` when the owner names one, else the
/// project of the repository containing ``dir``.
pub fn project_of(dir: &Path, over: Option<&str>) -> Result<String> {
    match over.map(str::trim).filter(|o| !o.is_empty()) {
        Some(name) => Ok(project_key(name)),
        None => project_name(dir).map(|n| project_key(&n)),
    }
}

/// Whether ``text`` is a project key as it travels.
pub fn valid_key(text: &str) -> bool {
    text.len() == 16 && text.bytes().all(|b| b.is_ascii_hexdigit())
}
