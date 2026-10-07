use super::valid_health;

#[test]
fn health_requires_complete_daemon_json() {
    let body = r#"{"ok":true,"name":"demo","slots":1,"free":1,"busy":false}"#;
    let response = format!("HTTP/1.0 200 OK\r\nContent-Type: application/json\r\n\r\n{body}");
    assert!(valid_health(response.as_bytes()));
    for bad in [
        "HTTP/1.0 200 OK\r\n\r\nhello",
        "HTTP/1.0 200 OK\r\n\r\n{\"ok\":true}",
        "HTTP/1.0 200 OK\r\n\r\n{",
        "HTTP/1.0 500 200\r\n\r\n{}",
        "HTTP/1.0 200 OK",
    ] {
        assert!(!valid_health(bad.as_bytes()), "{bad}");
    }
    assert!(!valid_health(response.replace("200 OK", "503 Unavailable").as_bytes()));
    assert!(!valid_health(response.replace("true", "false").as_bytes()));
}
