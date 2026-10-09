//! No default test touches the machine's network. A test that listens on every interface, joins a
//! multicast group or broadcasts makes macOS ask the person for Local Network permission, once per
//! built binary; such tests exist only below the OPT-IN marker of `processes.rs`, each `#[ignore]`d.

use std::path::Path;

const MARKER: &str = "// ---- OPT-IN BELOW";
const NETWORK: [&str; 6] = ["0.0.0.0:", "UNSPECIFIED", "239.", "BROADCAST", "\"--network\"", "local_address"];

fn sources(dir: &Path, out: &mut Vec<std::path::PathBuf>) {
    for entry in std::fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            sources(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

#[test]
fn no_default_test_listens_beyond_loopback_or_uses_multicast() {
    let mut files = Vec::new();
    sources(&Path::new(env!("CARGO_MANIFEST_DIR")).join("tests"), &mut files);
    let me = Path::new(file!()).file_name().unwrap().to_owned();
    for file in files.into_iter().filter(|f| f.file_name().unwrap() != me) {
        let text = std::fs::read_to_string(&file).unwrap();
        let default_part = text.split(MARKER).next().unwrap();
        for token in NETWORK {
            assert!(!default_part.contains(token), "{} reaches the machine's network ({token}) outside the opt-in section", file.display());
        }
    }
}

#[test]
fn every_test_below_the_opt_in_marker_is_ignored() {
    let text = std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/processes.rs")).unwrap();
    let optin = text.split(MARKER).nth(1).expect("the marker is there");
    let tests = optin.matches("#[test]").count();
    assert!(tests > 0 && tests == optin.matches("#[test]\n#[ignore").count(), "each opt-in test is #[ignore]d");
}
