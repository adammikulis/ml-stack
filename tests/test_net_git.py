"""git against a remote: the host policy applies and only plain https/ssh addresses are spoken."""

import subprocess

import pytest

from ml_stack import net
from ml_stack.httpguard import Refused
from ml_stack.net import git as netgit, provenance


def local_policy(tmp_path, *allowed):
    return net.Policy(allowed=list(allowed), path=tmp_path / "a.jsonl")


@pytest.mark.parametrize("url", [
    "--upload-pack=touch /tmp/x", "-oProxyCommand=id", "ext::sh -c id", "fd::17",
    "file:///etc", "/local/path", "http://github.com/a/b", "git://github.com/a/b",
    "https://user:pw@github.com/a/b", "", "https://github.com/a/b c", "ftp://github.com/x",
])
def test_an_address_that_is_not_plain_https_is_refused(tmp_path, url):
    with pytest.raises(Refused):
        netgit.guarded(url, local_policy(tmp_path, "github.com"))


def test_ssh_is_spoken_only_when_the_caller_names_it(tmp_path):
    policy = local_policy(tmp_path, "github.com")
    with pytest.raises(Refused, match="https"):
        netgit.guarded("git@github.com:owner/repo.git", policy)
    assert netgit.guarded("git@github.com:owner/repo.git", policy, "https:ssh")


def test_a_host_the_policy_does_not_list_needs_approval(tmp_path):
    policy = local_policy(tmp_path, "github.com")
    with pytest.raises(net.NeedsApproval) as caught:
        netgit.guarded("https://git.example.org/a/b", policy)
    assert caught.value.host == "git.example.org"
    policy.approve("git.example.org", by="person")
    assert netgit.guarded("https://git.example.org/a/b", policy)


def test_a_network_command_is_refused_before_git_runs(tmp_path, monkeypatch):
    ran = []
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: ran.append(a))
    with pytest.raises(net.NeedsApproval):
        netgit.run(["clone", "--", "https://git.example.org/a/b", str(tmp_path / "x")],
                   url="https://git.example.org/a/b", policy=local_policy(tmp_path, "github.com"))
    assert ran == []


def test_git_runs_without_prompts_hooks_or_foreign_protocols(tmp_path):
    env = netgit.environment("https", tmp_path / "hooks")
    assert env["GIT_TERMINAL_PROMPT"] == "0" and env["GIT_ALLOW_PROTOCOL"] == "https"
    pairs = {env[f"GIT_CONFIG_KEY_{n}"]: env[f"GIT_CONFIG_VALUE_{n}"]
             for n in range(int(env["GIT_CONFIG_COUNT"]))}
    assert pairs["core.hooksPath"] == str(tmp_path / "hooks")
    assert pairs["protocol.ext.allow"] == "never" and pairs["protocol.file.allow"] == "never"
    assert pairs["submodule.recurse"] == "false" and pairs["transfer.fsckObjects"] == "true"


@pytest.mark.parametrize("selector", [
    "GIT_DIR", "GIT_WORK_TREE", "GIT_COMMON_DIR", "GIT_INDEX_FILE", "GIT_PREFIX",
    "GIT_NAMESPACE", "GIT_OBJECT_DIRECTORY", "GIT_ALTERNATE_OBJECT_DIRECTORIES",
])
def test_git_repository_selection_cannot_be_overridden_by_environment(tmp_path, monkeypatch, selector):
    monkeypatch.setenv(selector, str(tmp_path / "foreign-repository"))
    assert selector not in netgit.environment("https", tmp_path / "hooks")


def test_a_local_repository_is_not_cloned_through_the_network_path(tmp_path):
    origin = tmp_path / "origin"
    origin.mkdir()
    subprocess.run(["git", "init", "-q", str(origin)], check=True)
    with pytest.raises(Refused):
        netgit.clone(f"file://{origin}", tmp_path / "copy", policy=local_policy(tmp_path))


def test_a_repository_hook_does_not_run_on_a_command_run_through_the_wrapper(tmp_path):
    """git run here has core.hooksPath pointed at an empty directory."""
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    hook = repo / ".git" / "hooks" / "post-checkout"
    hook.write_text(f"#!/bin/sh\ntouch {tmp_path / 'ran'}\n")
    hook.chmod(0o755)
    (repo / "f").write_text("x")
    subprocess.run(["git", "-C", str(repo), "add", "f"], check=True)
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", "commit",
                    "-qm", "m", "--no-verify"], check=True)
    netgit.run(["checkout", "-q", "-b", "other"], cwd=repo)
    assert not (tmp_path / "ran").exists()


def test_the_commit_that_came_down_is_recorded_in_the_download_index(tmp_path, monkeypatch):
    calls = []

    def fake_run(args, **kwargs):
        calls.append(list(args))
        if args[0] == "rev-parse":
            return subprocess.CompletedProcess(args, 0, "abc123\n", "")
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(netgit, "run", fake_run)
    commit = netgit.clone("https://github.com/owner/repo", tmp_path / "repo", branch="main")
    assert commit == "abc123"
    row = provenance.recent(1)[0]
    assert (row["kind"], row["sha256"], row["url"]) == ("git", "abc123",
                                                         "https://github.com/owner/repo")
    assert calls[0][:4] == ["clone", "--depth", "1", "--no-tags"] and "--no-recurse-submodules" in calls[0]
