mod kit;

use kit::*;
use serde_json::json;
use std::path::Path;
use std::process::Command;
use tempfile::tempdir;

fn git(dir: &Path, args: &[&str]) {
    let out = Command::new("git").arg("-C").arg(dir).args(["-c", "user.name=t", "-c", "user.email=t@example.invalid", "-c", "commit.gpgsign=false"]).args(args).output().unwrap();
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
}

fn resolve(n: &mut poolhouse_node::node::Node, path: &Path) -> serde_json::Value {
    req(n, "project_resolve", "", "", json!({"path": path.to_str().unwrap()}))
}

fn add(n: &mut poolhouse_node::node::Node, id: &str, kind: &str, path: &Path) -> serde_json::Value {
    req(n, "project_add", "", "", json!({"id": id, "kind": kind, "path": path.to_str().unwrap()}))
}

#[test]
fn a_folder_only_project_resolves_from_inside_it() {
    let (state, work) = (tempdir().unwrap(), tempdir().unwrap());
    let docs = work.path().join("drive/Specs");
    std::fs::create_dir_all(docs.join("2026/q4")).unwrap();
    let mut n = node(state.path());
    ok(add(&mut n, "specs", "gdrive", &docs));
    assert_eq!(ok(resolve(&mut n, &docs.join("2026/q4")))["board"], "specs");
    assert_eq!(ok(resolve(&mut n, &docs))["board"], "specs");
    drop(n);
    let mut n = node(state.path());
    assert_eq!(ok(resolve(&mut n, &docs.join("2026")))["board"], "specs", "the registry survives a restart");
    assert_eq!(ok(req(&mut n, "project_list", "", "", json!({})))[0]["sources"][0]["kind"], "gdrive");
}

#[test]
fn a_repository_and_all_its_worktrees_resolve_to_its_project() {
    let (state, work) = (tempdir().unwrap(), tempdir().unwrap());
    let main = work.path().join("shop");
    std::fs::create_dir(&main).unwrap();
    git(&main, &["init", "-q"]);
    git(&main, &["commit", "-q", "--allow-empty", "-m", "start"]);
    let linked = work.path().join("shop-feature");
    git(&main, &["worktree", "add", "-q", linked.to_str().unwrap(), "-b", "feature"]);
    std::fs::create_dir_all(linked.join("src")).unwrap();
    let mut n = node(state.path());
    ok(add(&mut n, "shop", "git", &linked.join("src")));
    for dir in [main.clone(), linked.clone(), linked.join("src")] {
        assert_eq!(ok(resolve(&mut n, &dir))["board"], "shop", "{dir:?}");
    }
    let sources = ok(req(&mut n, "project_list", "", "", json!({})));
    assert_eq!(Path::new(sources[0]["sources"][0]["path"].as_str().unwrap()), main.canonicalize().unwrap(), "a git source is the repository, not the worktree");
}

#[test]
fn an_unregistered_directory_is_not_part_of_a_project_and_never_gets_another_board() {
    let (state, work) = (tempdir().unwrap(), tempdir().unwrap());
    let (mine, stranger, sibling) = (work.path().join("mine"), work.path().join("stranger"), work.path().join("mine-sibling"));
    for d in [&mine, &stranger, &sibling] {
        std::fs::create_dir(d).unwrap();
    }
    let mut n = node(state.path());
    ok(add(&mut n, "mine", "folder", &mine));
    for dir in [&stranger, &sibling, work.path(), Path::new("/definitely/not/here")] {
        let reply = resolve(&mut n, dir);
        assert_eq!(code(&reply), "denied", "{dir:?}");
        assert!(reply["error"]["message"].as_str().unwrap().contains("not part of a project"));
    }
}

#[test]
fn the_deepest_source_wins_and_one_place_cannot_belong_to_two_projects() {
    let (state, work) = (tempdir().unwrap(), tempdir().unwrap());
    let inner = work.path().join("mono/service");
    std::fs::create_dir_all(&inner).unwrap();
    let mut n = node(state.path());
    ok(add(&mut n, "mono", "folder", &work.path().join("mono")));
    ok(add(&mut n, "service", "folder", &inner));
    assert_eq!(ok(resolve(&mut n, &inner))["board"], "service");
    assert_eq!(ok(resolve(&mut n, &work.path().join("mono")))["board"], "mono");
    assert_eq!(code(&add(&mut n, "thief", "folder", &inner)), "invalid");
    assert_eq!(code(&add(&mut n, "mono", "folder", &inner)), "invalid", "an id is registered once");
    assert_eq!(code(&add(&mut n, "BAD", "folder", &inner)), "invalid");
    assert_eq!(code(&add(&mut n, "k", "dropbox", &inner)), "invalid");
}

#[test]
fn only_a_session_of_the_project_adds_a_place_to_it() {
    let (state, work) = (tempdir().unwrap(), tempdir().unwrap());
    let (a, b, extra) = (work.path().join("a"), work.path().join("b"), work.path().join("extra"));
    for d in [&a, &b, &extra] {
        std::fs::create_dir(d).unwrap();
    }
    let mut n = node(state.path());
    ok(add(&mut n, "alpha", "folder", &a));
    ok(add(&mut n, "beta", "folder", &b));
    let (_, alpha) = session(&mut n, "alpha", "s");
    let (_, beta) = session(&mut n, "beta", "s");
    let source = |n: &mut poolhouse_node::node::Node, token: &str| req(n, "source_add", "", token, json!({"id": "alpha", "kind": "onedrive", "path": extra.to_str().unwrap()}));
    assert_eq!(code(&source(&mut n, "")), "denied");
    assert_eq!(code(&source(&mut n, &beta)), "denied");
    ok(source(&mut n, &alpha));
    assert_eq!(ok(resolve(&mut n, &extra))["board"], "alpha");
}
