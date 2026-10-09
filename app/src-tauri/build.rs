use std::{env, fs, path::Path};

/// Tauri refuses to build while the sidecar named in `externalBin` is missing, so a fresh
/// clone could not even `cargo check`. A check or a debug build never ships it, so it gets an
/// empty placeholder (git-ignored); a release build panics with the command that builds the
/// real one rather than bundling an empty file.
fn ensure_sidecar() {
    let target = env::var("TARGET").unwrap_or_default();
    let suffix = if target.contains("windows") { ".exe" } else { "" };
    let path = Path::new("binaries").join(format!("ml-stack-headless-{target}{suffix}"));
    if path.exists() {
        return;
    }
    if env::var("PROFILE").as_deref() == Ok("release") {
        panic!(
            "the sidecar {} is missing: run `python packaging/build.py --bundle` from the repository root",
            path.display()
        );
    }
    fs::create_dir_all("binaries").expect("could not make src-tauri/binaries");
    fs::write(&path, b"").expect("could not write the placeholder sidecar");
    println!(
        "cargo:warning={} is an empty placeholder so this build can run; `python packaging/build.py --bundle` makes the real one",
        path.display()
    );
}

fn main() {
    ensure_sidecar();
    let commands = tauri_build::AppManifest::new().commands(&["close_choice", "on_closing", "reopen_page"]);
    tauri_build::try_build(tauri_build::Attributes::new().app_manifest(commands))
        .expect("the window's own commands could not be declared")
}
