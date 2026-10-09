use super::{challenge, expected, hmac_sha256, same, secret, SECRET_FILE};
use std::fs;
use std::path::PathBuf;

fn hex_of(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("poolside-identity-{name}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn hmac_matches_the_published_vectors_the_daemon_side_uses() {
    // RFC 4231 test cases 1, 2 and 6 (a key longer than the block)
    assert_eq!(
        hex_of(&hmac_sha256(&[0x0b; 20], &[0x48, 0x69, 0x20, 0x54, 0x68, 0x65, 0x72, 0x65])),
        "b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7"
    );
    assert_eq!(
        hex_of(&hmac_sha256(b"Jefe", b"what do ya want for nothing?")),
        "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843"
    );
    assert_eq!(
        hex_of(&hmac_sha256(
            &[0xaa; 131],
            b"Test Using Larger Than Block-Size Key - Hash Key First"
        )),
        "60e431591ee0b67f0d8a26aacbf5b77f8e0bc6213728c5140546040f0ee37f54"
    );
}

#[test]
fn challenges_are_long_hex_and_never_repeat() {
    let first = challenge().unwrap();
    assert_eq!(first.len(), 32);
    assert!(first.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)));
    assert_ne!(first, challenge().unwrap());
}

#[test]
fn proofs_compare_whole_and_only_when_equal() {
    let proof = expected(&[7; 32], &"ab".repeat(16));
    assert!(same(&proof, &proof));
    assert!(!same(&proof, &expected(&[8; 32], &"ab".repeat(16))));
    assert!(!same(&proof, &proof[1..]));
    assert!(!same(&proof, ""));
}

#[test]
fn only_a_private_plain_file_of_the_right_size_is_a_secret() {
    let root = scratch("secret");
    assert!(secret(&root).is_none());
    let path = root.join(SECRET_FILE);
    fs::write(&path, "ab".repeat(32)).unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&path, fs::Permissions::from_mode(0o644)).unwrap();
        assert!(secret(&root).is_none(), "a file others can read is not a secret");
        fs::set_permissions(&path, fs::Permissions::from_mode(0o600)).unwrap();
    }
    assert_eq!(secret(&root).unwrap(), vec![0xab; 32]);
    fs::write(&path, "ab").unwrap();
    assert!(secret(&root).is_none());
    fs::write(&path, "zz".repeat(32)).unwrap();
    assert!(secret(&root).is_none());
    #[cfg(unix)]
    {
        fs::remove_file(&path).unwrap();
        let real = root.join("elsewhere");
        fs::write(&real, "ab".repeat(32)).unwrap();
        std::os::unix::fs::symlink(&real, &path).unwrap();
        assert!(secret(&root).is_none(), "a link is not the daemon's file");
    }
    let _ = fs::remove_dir_all(&root);
}
