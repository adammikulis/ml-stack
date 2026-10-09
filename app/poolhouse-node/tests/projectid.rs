use std::path::Path;
use std::process::Command;

use poolhouse_node::projectid::{normalise_remote, project_key, project_name, project_of, valid_key};
use tempfile::tempdir;

fn git(dir: &Path, args: &[&str]) {
    let out = Command::new("git").arg("-C").arg(dir).args(["-c", "user.name=t", "-c", "user.email=t@example.invalid", "-c", "commit.gpgsign=false"]).args(args).output().unwrap();
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
}

fn repo(dir: &Path, origin: Option<&str>) {
    git(dir, &["init", "-q"]);
    if let Some(url) = origin {
        git(dir, &["remote", "add", "origin", url]);
    }
}

#[test]
fn every_way_of_writing_a_remote_gives_the_same_project() {
    for url in [
        "https://example.invalid/Owner/Repo.git", "https://example.invalid/owner/repo", "https://example.invalid/owner/repo/",
        "git@example.invalid:Owner/Repo.git", "ssh://git@example.invalid/owner/repo.git", "ssh://git@example.invalid:22/owner/repo.git/",
        "https://user:token@example.invalid/owner/repo.git", "git://example.invalid/owner/repo",
    ] {
        assert_eq!(normalise_remote(url), "example.invalid/owner/repo", "{url}");
    }
    assert_ne!(normalise_remote("https://example.invalid/owner/other"), normalise_remote("https://example.invalid/owner/repo"));
    assert_ne!(normalise_remote("https://other.example.org/owner/repo"), normalise_remote("https://example.invalid/owner/repo"));
}

#[test]
fn a_key_is_sixteen_hex_digits_that_do_not_show_the_name() {
    let key = project_key("example.invalid/owner/repo");
    assert!(valid_key(&key) && !key.contains("owner"));
    assert_eq!(key, project_key(" example.invalid/owner/repo "));
    assert_ne!(key, project_key("example.invalid/owner/other"));
    assert!(!valid_key("short") && !valid_key("zzzzzzzzzzzzzzzz"));
}

#[test]
fn two_clones_with_different_folders_and_remote_spellings_are_one_project() {
    let (a, b) = (tempdir().unwrap(), tempdir().unwrap());
    repo(a.path(), Some("git@example.invalid:Owner/Repo.git"));
    repo(b.path(), Some("https://example.invalid/owner/repo"));
    assert_eq!(project_name(a.path()).unwrap(), "example.invalid/owner/repo");
    assert_eq!(project_of(a.path(), None).unwrap(), project_of(b.path(), None).unwrap());
    let c = tempdir().unwrap();
    repo(c.path(), Some("https://example.invalid/owner/elsewhere"));
    assert_ne!(project_of(a.path(), None).unwrap(), project_of(c.path(), None).unwrap());
}

#[test]
fn a_repository_without_a_remote_is_named_by_its_folder_from_any_depth() {
    let parent = tempdir().unwrap();
    let root = parent.path().join("My-Project");
    std::fs::create_dir_all(root.join("src/deep")).unwrap();
    repo(&root, None);
    assert_eq!(project_name(&root).unwrap(), "local/my-project");
    assert_eq!(project_of(&root.join("src/deep"), None).unwrap(), project_key("local/my-project"));
}

#[test]
fn a_linked_worktree_is_the_project_of_the_repository_it_belongs_to() {
    let parent = tempdir().unwrap();
    let root = parent.path().join("main-tree");
    std::fs::create_dir_all(&root).unwrap();
    repo(&root, None);
    git(&root, &["commit", "-q", "--allow-empty", "-m", "first"]);
    let linked = parent.path().join("elsewhere");
    git(&root, &["worktree", "add", "-q", "-b", "side", linked.to_str().unwrap()]);
    assert_eq!(project_name(&linked).unwrap(), "local/main-tree");
}

#[test]
fn an_explicit_name_overrides_the_repository_and_outside_a_repository_there_is_none() {
    let a = tempdir().unwrap();
    repo(a.path(), Some("https://example.invalid/owner/repo"));
    assert_eq!(project_of(a.path(), Some("my-team")).unwrap(), project_key("my-team"));
    assert_eq!(project_of(a.path(), Some("  ")).unwrap(), project_of(a.path(), None).unwrap(), "a blank override is none");
    let bare = tempdir().unwrap();
    assert!(project_of(bare.path(), None).is_err());
    assert_eq!(project_of(bare.path(), Some("my-team")).unwrap(), project_key("my-team"));
}
