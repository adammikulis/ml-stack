"""Policy validation, the generated profile text, the bubblewrap argv and the fail-closed
rules. Nothing here needs a sandbox on the machine."""

from __future__ import annotations

import os
import re
import sys

import pytest

from ml_stack import sandbox
from ml_stack.sandbox import AllowUnsandboxed, Limits, Net, Policy, PolicyError, policies, run
from ml_stack.sandbox.bubblewrap import Bubblewrap, arguments
from ml_stack.sandbox.container import Container
from ml_stack.sandbox.seatbelt import ProfileError, profile, quote

SYSTEM_BIN = os.path.realpath("/bin")
SYSTEM_SHELL = os.path.realpath("/bin/sh")


def forms(text: str) -> list[str]:
    """The top-level forms of a profile, reading string literals with their escapes."""
    out, depth, start, i, in_str = [], 0, 0, 0, False
    while i < len(text):
        c = text[i]
        if in_str:
            if c == "\\":
                i += 1
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c == "(":
            if depth == 0:
                start = i
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                out.append(text[start:i + 1])
        i += 1
    assert depth == 0 and not in_str, "unbalanced profile"
    return out


def heads(text: str) -> list[str]:
    return [re.match(r"\((\S+)(?: (\S+))?", f).group(0) for f in forms(text)]


@pytest.fixture
def tree(tmp_path):
    for name in ("a", "b", "c"):
        (tmp_path / name).mkdir()
    return tmp_path


def test_every_path_must_be_absolute_real_and_plain(tree):
    (tree / "link").symlink_to(tree / "a")
    for bad, why in (("rel", "not absolute"), (str(tree / "a") + "/../b", "climbs"),
                     (f"{tree}/a/", "ends with a slash"), (f"{tree}//a", "empty segment"),
                     (f"{tree}/./a", r"\. segment"), (str(tree / "link"), "symlink"),
                     (str(tree / "missing"), "does not exist"),
                     (f"{tree}/a\x00", "control character"), (f"{tree}/a\n", "control character"),
                     (f"{tree}/a\x7f", "control character"), ("", "empty")):
        with pytest.raises(PolicyError, match=why):
            Policy("p", read=(bad,)).validated()
    ok = Policy("p", read=(str(tree / "a"),), write=(str(tree / "b"),)).validated()
    assert ok.read == (str(tree / "a"),) and ok.write == (str(tree / "b"),)


def test_validation_covers_write_exec_and_cache_too(tree):
    for field in ("write", "exec"):
        with pytest.raises(PolicyError, match=field):
            Policy("p", **{field: ("relative",)}).validated()
    with pytest.raises(PolicyError, match="cache"):
        Policy("p", cache="relative").validated()


def test_environment_names_and_values_are_checked():
    for env in ({"1BAD": "x"}, {"A B": "x"}, {"A=B": "x"}, {"A": "x\x00"}, {"": "x"}):
        with pytest.raises(PolicyError, match="env"):
            Policy("p", env=env).validated()
    assert Policy("p", env={"GOOD_1": "v"}).validated().env == {"GOOD_1": "v"}


def test_network_and_limit_values_are_checked():
    for bad in (0, 65536, -1, True, "80"):
        with pytest.raises(PolicyError):
            Net.only(bad)
    with pytest.raises(PolicyError):
        Net(sandbox.NetMode.PORTS)
    with pytest.raises(PolicyError):
        Net(sandbox.NetMode.LOOPBACK, (80,))
    for field in ("wall_seconds", "cpu_seconds", "file_bytes", "open_files", "output_bytes"):
        with pytest.raises(PolicyError, match=field):
            Limits(**{field: 0})
    assert Limits(wall_seconds=None).wall_seconds is None
    assert Net.only(9, 3, 3).ports == (3, 9)


def test_the_policy_can_lose_its_network():
    assert Policy("p", net=Net.loopback()).without_network().net == Net.deny()


def test_quote_escapes_quotes_and_backslashes_and_refuses_control_characters():
    assert quote('a"b\\c') == '"a\\"b\\\\c"'
    for bad in ("a\nb", "a\x00b", "a\rb", "a\x1bb", "a\x7fb"):
        with pytest.raises(ProfileError):
            quote(bad)


def test_the_profile_denies_by_default_and_grants_only_what_the_policy_names(tree):
    pol = Policy("p", read=(str(tree / "a"),), write=(str(tree / "b"),),
                 exec=(SYSTEM_BIN,), env={}).validated()
    text = profile(pol, "/bin/echo", "tag1")
    assert forms(text)[1].startswith("(deny default")
    assert f'(subpath "{tree}/a")' in text and f'(subpath "{tree}/b")' in text
    assert str(tree / "c") not in text
    assert "network" not in text and "iokit" not in text
    writes = [f for f in forms(text) if f.startswith("(allow file-write*")]
    assert any(f'"{tree}/b"' in f for f in writes)
    assert all(f'"{tree}/a"' not in f for f in writes)


def test_network_rules_follow_the_mode(tree):
    base = Policy("p")
    assert "network" not in profile(base.validated(), "/bin/echo", "t")
    loop = profile(Policy("p", net=Net.loopback()).validated(), "/bin/echo", "t")
    assert '(remote ip "localhost:*")' in loop
    ports = profile(Policy("p", net=Net.only(8123, 8124)).validated(), "/bin/echo", "t")
    assert '(remote ip "localhost:8123")' in ports and '"localhost:*"' not in ports
    assert "network-bind" not in ports


def test_the_gpu_rules_are_added_only_when_asked(tree):
    off = profile(Policy("p").validated(), "/bin/echo", "t")
    on = profile(Policy("p", gpu=True).validated(), "/bin/echo", "t")
    assert "iokit" not in off and "MTLCompilerService" not in off
    assert '(iokit-user-client-class "IOGPUDeviceUserClient")' in on
    assert "(allow iokit-open (iokit-user-client-class" in on and "iokit-get-properties" not in on


def test_a_cache_directory_is_readable_and_writable(tree):
    text = profile(Policy("p", cache=str(tree / "c")).validated(), "/bin/echo", "t")
    assert any(f.startswith("(allow file-write*") and str(tree / "c") in f for f in forms(text))


@pytest.mark.parametrize("name", [
    'x") (allow file-read* (regex #".*")) ;',
    "x\\\") (allow default) (",
    "a)(b",
    "semi;colon",
])
def test_a_hostile_path_adds_no_form_to_the_profile(tree, name):
    plain = tree / "plain"
    plain.mkdir()
    hostile = tree / name
    hostile.mkdir()
    a = profile(Policy("p", read=(str(plain),)).validated(), "/bin/echo", "t")
    b = profile(Policy("p", read=(str(hostile),)).validated(), "/bin/echo", "t")
    assert len(forms(a)) == len(forms(b))
    assert heads(a) == heads(b)


def test_a_tag_must_be_a_plain_token(tree):
    for bad in ('a"b', "a b", "", "a\nb", "x" * 80, 'a") (allow default'):
        with pytest.raises(ProfileError):
            profile(Policy("p").validated(), "/bin/echo", bad)


def test_the_command_is_found_on_the_policy_path_and_must_exist(tree):
    from ml_stack.sandbox.backend import program_of

    assert program_of(["sh"], "/bin") == "/bin/sh"
    for argv, path in (([], "/bin"), ([""], "/bin"), (["nosuchprogram"], "/bin"),
                       (["rel/prog"], "/bin")):
        with pytest.raises(PolicyError):
            program_of(argv, path)


def test_bubblewrap_arguments_unshare_everything_and_bind_the_allow_list(tree):
    pol = Policy("p", read=(str(tree / "a"),), write=(str(tree / "b"),),
                 exec=(SYSTEM_SHELL,), env={"PATH": "/usr/bin", "HOME": "/work"}).validated()
    args = arguments(pol, SYSTEM_SHELL)
    assert args[:4] == ["--unshare-all", "--die-with-parent", "--new-session", "--clearenv"]
    assert "--share-net" not in args
    assert ["--ro-bind", str(tree / "a"), str(tree / "a")] == args[args.index(str(tree / "a")) - 1:][:3]
    assert ["--bind", str(tree / "b"), str(tree / "b")] == args[args.index("--bind"):][:3]
    assert args[args.index("--setenv"):][:3] == ["--setenv", "PATH", "/usr/bin"]
    assert str(tree / "c") not in args


def test_bubblewrap_keeps_the_network_only_for_loopback_policies():
    assert "--share-net" in arguments(Policy("p", net=Net.loopback()).validated(), "/bin/sh")
    assert "--share-net" in arguments(Policy("p", net=Net.only(80)).validated(), "/bin/sh")
    assert "--share-net" not in arguments(Policy("p").validated(), "/bin/sh")


def test_without_a_backend_the_command_is_not_run_and_the_refusal_is_an_event(tmp_path):
    marker = tmp_path / "ran"
    events: list[tuple[str, dict]] = []
    pol = Policy("untrusted", read=(str(tmp_path),), write=(str(tmp_path),), exec=(SYSTEM_BIN,),
                 env={"PATH": "/bin"})
    with pytest.raises(sandbox.SandboxUnavailable, match="not run"):
        run(["/usr/bin/touch", str(marker)], pol, via=Container(),
            on_event=lambda n, f: events.append((n, f)))
    assert not marker.exists()
    assert [n for n, _ in events] == ["sandbox.unavailable"]
    with pytest.raises(sandbox.SandboxUnavailable):
        run(["/usr/bin/touch", str(marker)], pol, via=Bubblewrap() if os.uname().sysname == "Darwin"
            else Container())


def test_running_unsandboxed_needs_a_named_reason_and_is_logged(tmp_path, caplog):
    marker = tmp_path / "ran"
    events: list[tuple[str, dict]] = []
    pol = Policy("untrusted", env={"PATH": "/bin"})
    caplog.set_level("WARNING", logger="ml_stack.sandbox")
    result = run(["/usr/bin/touch", str(marker)], pol, via=Container(),
                 unsandboxed=AllowUnsandboxed("test on a host with no sandbox"),
                 on_event=lambda n, f: events.append((n, f)), cwd=str(tmp_path))
    assert marker.exists() and result.returncode == 0 and not result.sandboxed
    assert result.backend == "none"
    assert events[0][0] == "sandbox.unsandboxed"
    assert events[0][1]["reason"] == "test on a host with no sandbox"
    assert any("WITHOUT a sandbox" in r.getMessage() for r in caplog.records)
    with pytest.raises(PolicyError):
        AllowUnsandboxed("  ")


def test_the_container_backend_is_a_documented_stub():
    assert not Container().available().ok
    with pytest.raises(NotImplementedError, match=r"docs/sandbox\.md"):
        Container().wrap(["/bin/true"], Policy("p"))


def test_native_policy_factories_use_resolved_system_paths(tmp_path):
    for name in policies.SYSTEM_EXEC:
        assert name == os.path.realpath(name)
    assert len(policies.SYSTEM_EXEC) == len(set(policies.SYSTEM_EXEC))
    assert policies.bash(tmp_path, tmp_path).validated().exec == policies.SYSTEM_EXEC
    assert policies.mcp_server(os.path.realpath(sys.executable), tmp_path, tmp_path).validated()
