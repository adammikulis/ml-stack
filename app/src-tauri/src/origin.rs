//! The one origin the window may show and may ask for native commands: the daemon this app
//! opened, on its port. The static capability names the default port; a daemon on another port
//! is granted at start-up, for that port alone.

use tauri::ipc::CapabilityBuilder;
use tauri::Url;

pub const HOST: &str = "127.0.0.1";

/// What the page on the own origin may ask the window for.
pub const GRANTED: [&str; 4] = [
    "core:default",
    "window-state:default",
    "allow-close-choice",
    "allow-on-closing",
];

/// The address the window loads, and the only one it may stay on.
pub fn address(port: u16) -> String {
    format!("http://{HOST}:{port}/ui/")
}

/// Whether `url` is on the own origin: this scheme, host and port, with no credentials. A
/// `blob:` address the page itself made (a download it built) belongs to that origin too.
pub fn is_own(url: &Url, port: u16) -> bool {
    if url.scheme() == "blob" {
        let prefix = format!("http://{HOST}:{port}/");
        return url.path().starts_with(&prefix) && !url.path()[prefix.len()..].contains(['@', '\\']);
    }
    url.scheme() == "http"
        && url.host_str() == Some(HOST)
        && url.port_or_known_default() == Some(port)
        && url.username().is_empty()
        && url.password().is_none()
}

/// The grant for a daemon on `port`, for the main window and that origin alone.
pub fn capability(port: u16) -> CapabilityBuilder {
    GRANTED.iter().fold(
        CapabilityBuilder::new(format!("own-origin-{port}"))
            .local(false)
            .window("main")
            .remote(format!("http://{HOST}:{port}/*")),
        |built, permission| built.permission(*permission),
    )
}
