//! An optional repository hint: `.poolhouse/project.toml` with `board = "id"`.
//!
//! The node's project registry (`registry.rs`) is what resolves a directory to a board; this
//! file is never required. It exists so setup can propose an id for a repository (from its root
//! commit) and a repository can carry the id it was registered under. Nothing here chooses a
//! board silently.

use std::path::{Path, PathBuf};
use std::process::Command;

use crate::error::{Error, Result};
use crate::row::valid_name;

pub const PROJECT_FILE: &str = ".poolhouse/project.toml";

/// The top of the working tree containing ``dir`` (the nearest ancestor with a `.git`).
pub(crate) fn top(dir: &Path) -> Result<PathBuf> {
    let start = dir.canonicalize()?;
    start.ancestors().find(|d| d.join(".git").exists()).map(Path::to_path_buf)
        .ok_or_else(|| Error::Invalid("this directory is not in a git repository".into()))
}

/// The main working tree of the repository that ``top`` belongs to: for a linked worktree, the
/// parent of the common git directory.
pub(crate) fn common_root(top: &Path) -> Result<PathBuf> {
    let git = top.join(".git");
    if git.is_dir() {
        return Ok(top.to_path_buf());
    }
    let text = std::fs::read_to_string(&git)?;
    let gitdir = text.trim().strip_prefix("gitdir:").map(|p| PathBuf::from(p.trim()))
        .ok_or_else(|| Error::Invalid("`.git` is neither a directory nor a gitdir file".into()))?;
    let gitdir = if gitdir.is_absolute() { gitdir } else { top.join(gitdir) };
    let common = match std::fs::read_to_string(gitdir.join("commondir")) {
        Ok(rel) => gitdir.join(rel.trim()).canonicalize()?,
        Err(_) => gitdir.canonicalize()?,
    };
    common.parent().map(Path::to_path_buf).ok_or_else(|| Error::Invalid("no repository root".into()))
}

/// The board id written in a project file's text.
pub fn parse_board(text: &str) -> Option<String> {
    let line = text.lines().map(str::trim).find(|l| l.starts_with("board") && l.contains('='))?;
    let value = line.split_once('=')?.1.trim().trim_matches('"');
    valid_name(value).then(|| value.to_string())
}

/// The board of the project that contains ``dir``.
pub fn board_of(dir: &Path) -> Result<String> {
    let top = top(dir)?;
    let roots = [top.clone(), common_root(&top)?];
    for root in &roots {
        if let Ok(text) = std::fs::read_to_string(root.join(PROJECT_FILE)) {
            return parse_board(&text).ok_or_else(|| Error::Invalid(format!("{PROJECT_FILE} has no valid `board = \"id\"`")));
        }
    }
    Err(Error::Invalid(format!("this repository has no {PROJECT_FILE}: run project setup to choose a board id")))
}

/// An id to propose for the repository at ``dir``: `p-` and the first 12 characters of its root
/// commit. Only a proposal; `init_project` writes what the owner or setup confirms.
pub fn propose_id(dir: &Path) -> Result<String> {
    let out = Command::new("git").arg("-C").arg(dir).args(["rev-list", "--max-parents=0", "HEAD"]).output()?;
    let text = String::from_utf8_lossy(&out.stdout);
    let root = text.lines().next().unwrap_or("").trim();
    if !out.status.success() || root.len() < 12 {
        return Err(Error::Invalid("the repository has no commit to take an id from".into()));
    }
    Ok(format!("p-{}", &root[..12]))
}

/// Write the project file in the working tree containing ``dir``; never over an existing one.
pub fn init_project(dir: &Path, id: &str) -> Result<PathBuf> {
    if !valid_name(id) {
        return Err(Error::Invalid("a board id is a short lower-case word".into()));
    }
    let path = top(dir)?.join(PROJECT_FILE);
    if path.exists() {
        return Err(Error::Invalid("this repository already has a board id".into()));
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&path, format!("board = \"{id}\"\n"))?;
    Ok(path)
}
