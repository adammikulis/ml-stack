//! The project registry: which board a directory belongs to.
//!
//! A project is a board id and the places its work lives: a git repository, a plain folder, a
//! cloud-synced folder (its local sync path). A client's working directory
//! resolves to the project whose source contains it; a linked git worktree resolves through
//! the repository's common directory. A directory that matches nothing is not part of any
//! project, and is never given another project's board.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::fsutil::write_atomic;
use crate::row::valid_name;

pub const KINDS: [&str; 4] = ["git", "folder", "gdrive", "onedrive"];
const MAX_SOURCES: usize = 32;

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Source {
    pub kind: String,
    pub path: PathBuf,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Project {
    pub id: String,
    pub sources: Vec<Source>,
}

pub struct Projects {
    path: PathBuf,
    all: Vec<Project>,
}

/// The main working tree of the repository containing ``dir``, if it is in one.
pub fn git_root(dir: &Path) -> Option<PathBuf> {
    let top = dir.ancestors().find(|d| d.join(".git").exists())?;
    let git = top.join(".git");
    if git.is_dir() {
        return Some(top.to_path_buf());
    }
    let text = std::fs::read_to_string(&git).ok()?;
    let gitdir = PathBuf::from(text.trim().strip_prefix("gitdir:")?.trim());
    let gitdir = if gitdir.is_absolute() { gitdir } else { top.join(gitdir) };
    let common = match std::fs::read_to_string(gitdir.join("commondir")) {
        Ok(rel) => gitdir.join(rel.trim()).canonicalize().ok()?,
        Err(_) => gitdir.canonicalize().ok()?,
    };
    common.parent().map(Path::to_path_buf)
}

impl Projects {
    pub fn open(path: &Path) -> Result<Projects> {
        let all = match std::fs::read(path) {
            Ok(bytes) => serde_json::from_slice(&bytes).map_err(|e| Error::Damaged(format!("projects unreadable: {e}")))?,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Vec::new(),
            Err(e) => return Err(e.into()),
        };
        Ok(Projects { path: path.into(), all })
    }

    pub fn all(&self) -> &[Project] {
        &self.all
    }

    pub fn get(&self, id: &str) -> Option<&Project> {
        self.all.iter().find(|p| p.id == id)
    }

    fn save(&self) -> Result<()> {
        write_atomic(&self.path, &serde_json::to_vec(&self.all)?)
    }

    fn source(kind: &str, path: &str) -> Result<Source> {
        if !KINDS.contains(&kind) {
            return Err(Error::Invalid("a source is git, folder, gdrive or onedrive".into()));
        }
        let real = Path::new(path).canonicalize().map_err(|_| Error::Invalid("the source path does not exist".into()))?;
        let path = if kind == "git" { git_root(&real).ok_or_else(|| Error::Invalid("the path is not in a git repository".into()))? } else { real };
        Ok(Source { kind: kind.into(), path })
    }

    /// Register a new project with its first source.
    pub fn add(&mut self, id: &str, kind: &str, path: &str) -> Result<Project> {
        if !valid_name(id) || self.get(id).is_some() {
            return Err(Error::Invalid("a project id is a new short lower-case word".into()));
        }
        let source = Projects::source(kind, path)?;
        self.claim_check(id, &source)?;
        self.all.push(Project { id: id.into(), sources: vec![source] });
        self.save()?;
        Ok(self.all.last().cloned().expect("pushed above"))
    }

    /// Add a source to an existing project.
    pub fn add_source(&mut self, id: &str, kind: &str, path: &str) -> Result<Project> {
        let source = Projects::source(kind, path)?;
        self.claim_check(id, &source)?;
        let project = self.all.iter_mut().find(|p| p.id == id).ok_or_else(|| Error::Invalid("no such project".into()))?;
        if project.sources.len() >= MAX_SOURCES {
            return Err(Error::Quota("a project keeps this many sources at most".into()));
        }
        if !project.sources.contains(&source) {
            project.sources.push(source);
        }
        let done = project.clone();
        self.save()?;
        Ok(done)
    }

    /// The same place may not belong to two projects.
    fn claim_check(&self, id: &str, source: &Source) -> Result<()> {
        match self.all.iter().find(|p| p.id != id && p.sources.iter().any(|s| s.path == source.path)) {
            Some(other) => Err(Error::Invalid(format!("that place already belongs to project {}", other.id))),
            None => Ok(()),
        }
    }

    /// The project whose source contains ``dir`` (the deepest wins), or `Denied` saying the
    /// directory is not part of a project.
    pub fn resolve(&self, dir: &str) -> Result<&Project> {
        let not_part = || Error::Denied("this directory is not part of a project: register it with project_add".into());
        let real = Path::new(dir).canonicalize().map_err(|_| not_part())?;
        let mut places = vec![real.clone()];
        places.extend(git_root(&real));
        let best = self.all.iter().flat_map(|p| p.sources.iter().map(move |s| (p, s)))
            .filter(|(_, s)| places.iter().any(|d| d.starts_with(&s.path)))
            .max_by_key(|(_, s)| s.path.components().count());
        best.map(|(p, _)| p).ok_or_else(not_part)
    }
}
