//! One origin's append-only log on disk: JSON lines, fsynced on append, a torn tail dropped on open.

use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};

use crate::error::{Error, Result};
use crate::fsutil::sync_dir;
use crate::row::Row;
use crate::rules::check_chain;

pub struct Log {
    path: PathBuf,
    file: File,
    rows: Vec<Row>,
    /// Bytes of an unfinished last line dropped when the log was opened.
    pub torn_bytes: u64,
}

impl Log {
    /// Open (creating if need be) the log of ``origin`` on ``board``. A last line without its
    /// newline was never acknowledged: it is cut off. Any other damage is an error.
    pub fn open(path: &Path, board: &str, origin: &str) -> Result<Log> {
        let created = !path.exists();
        let mut file = OpenOptions::new().read(true).append(true).create(true).mode(0o600).open(path)?;
        if created {
            if let Some(parent) = path.parent() {
                sync_dir(parent)?;
            }
        }
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes)?;
        let good = bytes.iter().rposition(|b| *b == b'\n').map_or(0, |i| i + 1);
        let torn_bytes = (bytes.len() - good) as u64;
        if torn_bytes > 0 {
            file.set_len(good as u64)?;
            file.sync_all()?;
        }
        let mut rows = Vec::new();
        for line in bytes[..good].split(|b| *b == b'\n').filter(|l| !l.is_empty()) {
            let row: Row = serde_json::from_slice(line).map_err(|e| Error::Damaged(format!("a stored row is unreadable: {e}")))?;
            rows.push(row);
        }
        check_chain(&rows, board, origin, None)?;
        Ok(Log { path: path.to_path_buf(), file, rows, torn_bytes })
    }

    pub fn rows(&self) -> &[Row] {
        &self.rows
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Add one row; it is on disk when this returns.
    pub fn append(&mut self, row: Row) -> Result<()> {
        self.extend(vec![row])
    }

    /// Add rows in one write and one fsync. On a failed write the file is cut back.
    pub fn extend(&mut self, rows: Vec<Row>) -> Result<()> {
        let mut buf = Vec::new();
        for row in &rows {
            serde_json::to_writer(&mut buf, row)?;
            buf.push(b'\n');
        }
        let before = self.file.metadata()?.len();
        let written = self.file.write_all(&buf).and_then(|_| self.file.sync_data());
        if let Err(e) = written {
            let _ = self.file.set_len(before);
            return Err(e.into());
        }
        self.rows.extend(rows);
        Ok(())
    }

    /// Delete the file; the log is gone.
    pub fn remove(self) -> Result<()> {
        std::fs::remove_file(&self.path)?;
        Ok(())
    }
}
