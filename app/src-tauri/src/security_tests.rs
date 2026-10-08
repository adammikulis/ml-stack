use crate::origin::{self, GRANTED};
use tauri::ipc::Origin;
use tauri::Url;

fn remote(url: &str) -> Origin {
    Origin::Remote { url: url.parse().unwrap() }
}

#[test]
fn capabilities_reject_remote_shell_and_foreign_windows() {
    let mut context = crate::app_context();
    let authority = context.runtime_authority_mut();
    let local = remote("http://127.0.0.1:8770/ui/");
    assert!(authority.resolve_access("on_closing", "main", "main", &local).is_some());
    for command in ["plugin:shell|execute", "plugin:shell|spawn", "plugin:shell|open", "plugin:fs|read_file"] {
        assert!(authority.resolve_access(command, "main", "main", &local).is_none(), "{command}");
    }
    for url in [
        "https://outside.invalid/",
        "http://127.0.0.1.outside.invalid/ui/",
        "http://192.0.2.1:8770/ui/",
        "http://localhost:8770/ui/",
        "http://127.0.0.1:8771/ui/",
        "http://127.0.0.1:1/ui/",
        "https://127.0.0.1:8770/ui/",
    ] {
        assert!(authority.resolve_access("close_choice", "main", "main", &remote(url)).is_none(), "{url}");
    }
    assert!(authority.resolve_access("close_choice", "foreign", "foreign", &local).is_none());
}

// Tauri matches a grant on the page's origin and does not look at credentials in the address, so
// `http://user@127.0.0.1:8770/` would be granted; the navigation guard below is what keeps the
// window from ever being on such an address.
#[test]
fn another_port_is_granted_that_port_alone() {
    let mut context = crate::app_context();
    let authority = context.runtime_authority_mut();
    authority.add_capability(origin::capability(9123)).unwrap();
    let own = remote("http://127.0.0.1:9123/ui/");
    assert!(authority.resolve_access("close_choice", "main", "main", &own).is_some());
    assert!(authority.resolve_access("plugin:shell|execute", "main", "main", &own).is_none());
    for url in ["http://127.0.0.1:9124/ui/", "http://127.0.0.1:9000/ui/", "http://localhost:9123/ui/"] {
        assert!(authority.resolve_access("close_choice", "main", "main", &remote(url)).is_none(), "{url}");
    }
    assert!(authority.resolve_access("close_choice", "foreign", "foreign", &own).is_none());
}

#[test]
fn the_static_capability_names_the_default_port_and_the_same_permissions() {
    let file: serde_json::Value =
        serde_json::from_str(include_str!("../capabilities/main.json")).unwrap();
    assert_eq!(file["windows"], serde_json::json!(["main"]));
    assert_eq!(file["local"], serde_json::json!(false));
    assert_eq!(
        file["remote"]["urls"],
        serde_json::json!([format!("http://{}:{}/*", origin::HOST, crate::PORT)])
    );
    assert_eq!(file["permissions"], serde_json::json!(GRANTED));
}

#[test]
fn the_window_stays_on_its_own_origin() {
    let url = |text: &str| text.parse::<Url>().unwrap();
    assert!(origin::is_own(&url(&origin::address(8770)), 8770));
    assert!(origin::is_own(&url("http://127.0.0.1:8770/ui/#chat"), 8770));
    assert!(origin::is_own(&url("http://127.0.0.1:8770/ui/projects?x=1"), 8770));
    for outside in [
        "https://outside.invalid/",
        "http://outside.invalid:8770/ui/",
        "http://127.0.0.1:8771/ui/",
        "http://127.0.0.1/ui/",
        "https://127.0.0.1:8770/ui/",
        "http://localhost:8770/ui/",
        "http://127.0.0.1.outside.invalid:8770/ui/",
        "http://127.0.0.1@outside.invalid:8770/",
        "http://127.0.0.1:8770@outside.invalid/",
        "http://user:pass@127.0.0.1:8770/ui/",
        "file:///etc/passwd",
        "javascript:alert(1)",
        "data:text/html,hello",
        "about:blank",
        "ftp://127.0.0.1:8770/",
    ] {
        assert!(!origin::is_own(&url(outside), 8770), "{outside}");
    }
}

#[test]
fn the_configuration_carries_a_content_security_policy() {
    let config: serde_json::Value =
        serde_json::from_str(include_str!("../tauri.conf.json")).unwrap();
    let policy = config["app"]["security"]["csp"].as_str().expect("csp must not be null");
    for directive in ["default-src 'none'", "form-action 'none'", "base-uri 'none'", "object-src 'none'"] {
        assert!(policy.contains(directive), "{directive}");
    }
    assert!(!policy.contains('*') && !policy.contains("unsafe-"), "{policy}");
    let connect = policy.split("; ").find(|part| part.starts_with("connect-src")).unwrap();
    assert_eq!(connect, "connect-src ipc: http://ipc.localhost");
}
