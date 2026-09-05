fn main() {
    let commands = tauri_build::AppManifest::new().commands(&["close_choice", "on_closing"]);
    tauri_build::try_build(tauri_build::Attributes::new().app_manifest(commands))
        .expect("the window's own commands could not be declared")
}
