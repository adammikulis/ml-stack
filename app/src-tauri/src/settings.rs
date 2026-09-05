//! The one setting the window reads and writes: what closing it does.

use std::fs;
use std::path::{Path, PathBuf};

pub const BACKGROUND: &str = "background";
pub const QUIT: &str = "quit";

/// Where the daemon keeps this machine's settings.
pub fn path(root: &Path) -> PathBuf {
    root.join("settings.json")
}

/// The saved answer to the close question: `background`, `quit`, or empty for ask.
pub fn on_close(path: &Path) -> String {
    fs::read_to_string(path)
        .ok()
        .and_then(|text| serde_json::from_str::<serde_json::Value>(&text).ok())
        .and_then(|value| value.get("on_close")?.as_str().map(str::to_string))
        .unwrap_or_default()
}

/// Save the answer, leaving every other setting as it was.
pub fn set_on_close(path: &Path, mode: &str) -> Result<(), String> {
    let mut value = fs::read_to_string(path)
        .ok()
        .and_then(|text| serde_json::from_str::<serde_json::Value>(&text).ok())
        .filter(serde_json::Value::is_object)
        .unwrap_or_else(|| serde_json::json!({}));
    value["on_close"] = serde_json::Value::String(mode.to_string());
    let text = serde_json::to_string_pretty(&value).map_err(|e| e.to_string())?;

    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let temp = path.with_extension("json.tmp");
    fs::write(&temp, text).map_err(|e| e.to_string())?;
    fs::rename(&temp, path).map_err(|e| e.to_string())
}
