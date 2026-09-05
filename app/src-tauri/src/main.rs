#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]
//! A native window on the ml-stack interface, and the daemon that serves it.

mod daemon;
mod settings;

use std::path::PathBuf;
use std::sync::Mutex;

use tauri::utils::config::Color;
use tauri::{AppHandle, Manager, RunEvent, State, WebviewUrl, WebviewWindowBuilder, WindowEvent};
use tauri_plugin_shell::process::CommandChild;

const TITLE: &str = "ml-stack";
const WIDTH: f64 = 1180.0;
const HEIGHT: f64 = 820.0;
const MIN_WIDTH: f64 = 900.0;
const MIN_HEIGHT: f64 = 640.0;
const BACKGROUND: Color = Color(11, 15, 20, 255);
const PORT: u16 = 8770;
const ROOT: &str = ".ml-stack/traind";

/// What the window holds: where the settings are, and the daemon it started.
struct Shell {
    settings: PathBuf,
    daemon: Mutex<Option<CommandChild>>,
    quitting: Mutex<bool>,
}

/// The page's answer to the close question.
#[tauri::command]
fn close_choice(
    app: AppHandle,
    state: State<'_, Shell>,
    mode: String,
    remember: bool,
) -> serde_json::Value {
    if mode != settings::BACKGROUND && mode != settings::QUIT {
        return serde_json::json!({ "ok": false });
    }
    if remember {
        if let Err(why) = settings::set_on_close(&state.settings, &mode) {
            return serde_json::json!({ "ok": false, "error": why });
        }
    }
    act(&app, &state, &mode);
    serde_json::json!({ "ok": true, "mode": mode, "remembered": remember })
}

/// Whether the window may close now, raising the question when nothing is saved.
#[tauri::command]
fn on_closing(app: AppHandle, state: State<'_, Shell>) -> bool {
    if *state.quitting.lock().unwrap() {
        return true;
    }
    let saved = settings::on_close(&state.settings);
    if saved == settings::QUIT {
        act(&app, &state, settings::QUIT);
        return true;
    }
    let Some(window) = app.get_webview_window("main") else {
        return true;
    };
    if saved == settings::BACKGROUND {
        let _ = window.hide();
    } else {
        let _ = window.eval("window.mlStackAskOnClose && window.mlStackAskOnClose()");
    }
    false
}

fn act(app: &AppHandle, state: &State<'_, Shell>, mode: &str) {
    if mode == settings::BACKGROUND {
        if let Some(window) = app.get_webview_window("main") {
            let _ = window.hide();
        }
        return;
    }
    *state.quitting.lock().unwrap() = true;
    app.exit(0);
}

/// The port and settings root named on the command line, or the defaults.
fn asked(home: &std::path::Path) -> (u16, PathBuf) {
    let args: Vec<String> = std::env::args().collect();
    let value = |name: &str| {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    let port = value("--port").and_then(|p| p.parse().ok()).unwrap_or(PORT);
    let root = value("--root")
        .map(PathBuf::from)
        .unwrap_or_else(|| home.join(ROOT));
    (port, root)
}

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .invoke_handler(tauri::generate_handler![close_choice, on_closing])
        .setup(|app| {
            let handle = app.handle().clone();
            let home = app.path().home_dir()?;
            let (port, root) = asked(&home);

            let started = if daemon::healthy(port) {
                None
            } else {
                Some(daemon::start(&handle, port, &root.to_string_lossy())?)
            };
            app.manage(Shell {
                settings: settings::path(&root),
                daemon: Mutex::new(started),
                quitting: Mutex::new(false),
            });

            let url = format!("http://127.0.0.1:{port}/ui/");
            WebviewWindowBuilder::new(app, "main", WebviewUrl::External(url.parse()?))
                .title(TITLE)
                .inner_size(WIDTH, HEIGHT)
                .min_inner_size(MIN_WIDTH, MIN_HEIGHT)
                .background_color(BACKGROUND)
                .build()?;
            Ok(())
        })
        .on_window_event(|window, event| {
            if let WindowEvent::CloseRequested { api, .. } = event {
                let app = window.app_handle().clone();
                let state = app.state::<Shell>();
                if !on_closing(app.clone(), state) {
                    api.prevent_close();
                }
            }
        })
        .build(tauri::generate_context!())
        .expect("the window could not be built")
        .run(|app, event| match event {
            RunEvent::Reopen { .. } => {
                if let Some(window) = app.get_webview_window("main") {
                    let _ = window.show();
                    let _ = window.set_focus();
                }
            }
            RunEvent::Exit => {
                if let Some(child) = app.state::<Shell>().daemon.lock().unwrap().take() {
                    let _ = child.kill();
                }
            }
            _ => {}
        });
}
