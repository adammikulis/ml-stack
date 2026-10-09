use super::{page_url, ticket_from, token};

#[test]
fn a_ticket_is_read_only_from_a_complete_ok_response() {
    let ok = "HTTP/1.0 200 OK\r\nContent-Type: application/json\r\n\r\n{\"ticket\":\"abc-DEF_123\",\"expires_at\":1.0}";
    assert_eq!(ticket_from(ok.as_bytes()).as_deref(), Some("abc-DEF_123"));
    for bad in [
        "HTTP/1.0 403 Forbidden\r\n\r\n{\"ticket\":\"abc\"}",
        "HTTP/1.0 200 OK\r\n\r\n{\"error\":\"no\"}",
        "HTTP/1.0 200 OK\r\n\r\n{\"ticket\":\"a b\"}",
        "HTTP/1.0 200 OK\r\n\r\n{\"ticket\":\"a&b=c\"}",
        "HTTP/1.0 200 OK\r\n\r\n{",
        "HTTP/1.0 200 OK",
    ] {
        assert!(ticket_from(bad.as_bytes()).is_none(), "{bad}");
    }
}

#[test]
fn tokens_hold_only_url_safe_characters() {
    assert!(token("aZ09-_"));
    for bad in ["", "a b", "a\r\nb", "a/b", "a?b", &"x".repeat(257)] {
        assert!(!token(bad), "{bad:?}");
    }
}

#[test]
fn without_a_recorded_secret_the_window_opens_the_bare_page() {
    let root = std::env::temp_dir().join("poolhouse-launch-test-missing");
    assert_eq!(page_url(&root, 8770), "http://127.0.0.1:8770/ui/");
}
