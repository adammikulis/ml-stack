use tauri::ipc::Origin;

#[test]
fn capabilities_reject_remote_shell_and_foreign_windows() {
    let mut context = crate::app_context();
    let authority = context.runtime_authority_mut();
    let local = Origin::Remote {
        url: "http://127.0.0.1:8770/ui/".parse().unwrap(),
    };
    assert!(authority.resolve_access("on_closing", "main", "main", &local).is_some());
    assert!(authority.resolve_access("reopen_page", "main", "main", &local).is_some());
    for command in ["plugin:shell|execute", "plugin:shell|spawn", "plugin:shell|open", "plugin:fs|read_file"] {
        assert!(authority.resolve_access(command, "main", "main", &local).is_none(), "{command}");
    }
    for url in ["https://outside.invalid/", "http://127.0.0.1.outside.invalid/ui/", "http://192.0.2.1:8770/ui/"] {
        let origin = Origin::Remote { url: url.parse().unwrap() };
        assert!(authority.resolve_access("close_choice", "main", "main", &origin).is_none(), "{url}");
        assert!(authority.resolve_access("reopen_page", "main", "main", &origin).is_none(), "{url}");
    }
    assert!(authority.resolve_access("close_choice", "foreign", "foreign", &local).is_none());
    assert!(authority.resolve_access("reopen_page", "foreign", "foreign", &local).is_none());
}
