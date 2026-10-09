use poolhouse_node::project::{board_of, init_project, propose_id, parse_board};
use std::path::Path;
use std::process::Command;
use tempfile::tempdir;

fn git(dir: &Path, args: &[&str]) {
    let out = Command::new("git").arg("-C").arg(dir).args(["-c", "user.name=t", "-c", "user.email=t@example.invalid", "-c", "commit.gpgsign=false"]).args(args).output().unwrap();
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
}

fn repo(dir: &Path, board: Option<&str>) {
    git(dir, &["init", "-q"]);
    if let Some(id) = board {
        init_project(dir, id).unwrap();
        git(dir, &["add", ".poolhouse/project.toml"]);
    }
    git(dir, &["commit", "-q", "--allow-empty", "-m", "start"]);
}

#[test]
fn two_worktrees_of_one_repository_share_a_board() {
    let dir = tempdir().unwrap();
    let main = dir.path().join("main");
    std::fs::create_dir(&main).unwrap();
    repo(&main, Some("shop"));
    let linked = dir.path().join("linked");
    git(&main, &["worktree", "add", "-q", linked.to_str().unwrap(), "-b", "feature"]);
    std::fs::create_dir_all(linked.join("src/deep")).unwrap();
    assert_eq!(board_of(&main).unwrap(), "shop");
    assert_eq!(board_of(&linked).unwrap(), "shop");
    assert_eq!(board_of(&linked.join("src/deep")).unwrap(), "shop");
}

#[test]
fn a_worktree_without_the_committed_file_still_finds_the_repository_board() {
    let dir = tempdir().unwrap();
    let main = dir.path().join("main");
    std::fs::create_dir(&main).unwrap();
    repo(&main, None);
    let linked = dir.path().join("linked");
    git(&main, &["worktree", "add", "-q", linked.to_str().unwrap(), "-b", "feature"]);
    init_project(&main, "shop").unwrap();
    assert_eq!(board_of(&linked).unwrap(), "shop", "found through the git common directory");
}

#[test]
fn two_repositories_get_different_boards() {
    let dir = tempdir().unwrap();
    let (a, b) = (dir.path().join("a"), dir.path().join("b"));
    std::fs::create_dir(&a).unwrap();
    std::fs::create_dir(&b).unwrap();
    repo(&a, Some("alpha"));
    repo(&b, Some("beta"));
    assert_ne!(board_of(&a).unwrap(), board_of(&b).unwrap());
}

#[test]
fn a_repository_with_no_board_id_is_an_error_never_a_silent_default() {
    let dir = tempdir().unwrap();
    repo(dir.path(), None);
    assert!(board_of(dir.path()).unwrap_err().to_string().contains("project.toml"));
    let proposed = propose_id(dir.path()).unwrap();
    assert!(proposed.starts_with("p-") && proposed.len() == 14, "{proposed}");
    assert!(board_of(dir.path()).is_err(), "proposing writes nothing");
    init_project(dir.path(), &proposed).unwrap();
    assert_eq!(board_of(dir.path()).unwrap(), proposed);
    assert!(init_project(dir.path(), "other").is_err(), "an existing id is never overwritten");
}

#[test]
fn a_directory_outside_any_repository_and_a_bad_file_are_refused() {
    let dir = tempdir().unwrap();
    assert!(board_of(dir.path()).is_err());
    repo(dir.path(), None);
    std::fs::create_dir_all(dir.path().join(".poolhouse")).unwrap();
    std::fs::write(dir.path().join(".poolhouse/project.toml"), "board = \"../../x\"\n").unwrap();
    assert!(board_of(dir.path()).is_err());
    assert_eq!(parse_board("board = \"ok-1\"\n"), Some("ok-1".into()));
    assert_eq!(parse_board("board = \"Bad Id\"\n"), None);
}
