"""One resolver for tokens and keys: its order, what it refuses to trust, and what it never says."""

import io
import json
import logging
import os
import stat
import sys

import pytest

from ml_stack import credentials
from ml_stack.credentials import CredentialError, Secret, cli as credentials_cli, keychain

SECRET = "tok-4f9a1c7e2b8d6035a1c97e"

class MemoryKeychain:
    """A keychain in memory, with the three calls `keyring` has and the errors it raises."""

    class errors:
        KeyringError = type("KeyringError", (Exception,), {})
        PasswordDeleteError = type("PasswordDeleteError", (KeyringError,), {})

    def __init__(self):
        self.held = {}

    def get_password(self, service, name):
        return self.held.get((service, name))

    def set_password(self, service, name, value):
        self.held[(service, name)] = value

    def delete_password(self, service, name):
        try:
            del self.held[(service, name)]
        except KeyError:
            raise self.errors.PasswordDeleteError(name) from None


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_TOKEN_PATH", "HF_TOKEN_FILE",
                 "WIDGET_KEY", "WIDGET_KEY_FILE", "ML_STACK_CREDENTIALS_FILE",
                 credentials.INSECURE_ENV):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ML_STACK_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    monkeypatch.setattr(credentials, "_keyring", lambda: None)
    monkeypatch.setattr(keychain, "_BLOCKED", False)
    return tmp_path


def _file(path, text, mode=0o600):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(mode)
    return path


def _stored(tmp_path, text, mode=0o600):
    return _file(tmp_path / "home" / "credentials.toml", text, mode)


# -- the order ----------------------------------------------------------------------------


def test_each_source_is_found_when_it_is_the_only_one(isolated, monkeypatch):
    assert credentials.get("WIDGET_KEY") is None
    _stored(isolated, 'WIDGET_KEY = "from-file"\n')
    assert credentials.get("WIDGET_KEY") == "from-file"
    assert credentials.status("WIDGET_KEY") == {"present": True, "source": "credentials file"}
    secret = _file(isolated / "mounted", "from-mount\n", 0o644)
    monkeypatch.setenv("WIDGET_KEY_FILE", str(secret))
    assert credentials.get("WIDGET_KEY") == "from-mount"
    monkeypatch.setenv("WIDGET_KEY", "from-env")
    assert credentials.get("WIDGET_KEY") == "from-env"
    assert credentials.get("WIDGET_KEY", "from-argument") == "from-argument"


def test_the_order_is_argument_then_environment_then_named_file_then_file_then_keychain(
        isolated, monkeypatch):
    chain = MemoryKeychain()
    monkeypatch.setattr(credentials, "_keyring", lambda: chain)
    credentials.set("WIDGET_KEY", "from-keychain", keychain=True)
    assert credentials.get("WIDGET_KEY") == "from-keychain"
    _stored(isolated, 'WIDGET_KEY = "from-file"\n')
    assert credentials.get("WIDGET_KEY") == "from-file"
    monkeypatch.setenv("WIDGET_KEY_FILE", str(_file(isolated / "m", "from-mount")))
    assert credentials.get("WIDGET_KEY") == "from-mount"
    monkeypatch.setenv("WIDGET_KEY", "from-env")
    assert credentials.get("WIDGET_KEY") == "from-env"
    assert credentials.get("WIDGET_KEY", explicit="from-argument") == "from-argument"


def test_an_empty_environment_variable_falls_through(isolated, monkeypatch):
    _stored(isolated, 'WIDGET_KEY = "from-file"\n')
    monkeypatch.setenv("WIDGET_KEY", "   ")
    assert credentials.get("WIDGET_KEY") == "from-file"


def test_a_missing_required_credential_says_where_to_put_it_and_nothing_else(isolated):
    with pytest.raises(CredentialError, match=r"WIDGET_KEY_FILE.*ml-stack credentials set"):
        credentials.get("WIDGET_KEY", required=True)


def test_a_name_that_is_not_a_name_is_refused():
    for bad in ("", "1ABC", "A-B", "A B", "A" * 65, "../x"):
        with pytest.raises(CredentialError):
            credentials.get(bad)


# -- Hugging Face's own file ---------------------------------------------------------------


def test_a_token_saved_by_huggingface_login_is_found(isolated):
    _file(isolated / "hf" / "token", f"{SECRET}\n", 0o644)
    assert credentials.get("HF_TOKEN") == SECRET
    assert credentials.status("HF_TOKEN")["source"] == "huggingface token file"


def test_hf_token_path_names_the_file(isolated, monkeypatch):
    monkeypatch.setenv("HF_TOKEN_PATH", str(_file(isolated / "elsewhere", SECRET, 0o600)))
    assert credentials.get("HF_TOKEN") == SECRET


def test_the_legacy_hub_variable_is_an_alias(monkeypatch):
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", SECRET)
    assert credentials.get("HF_TOKEN") == SECRET


def test_the_ml_stack_file_wins_over_the_huggingface_file(isolated):
    _file(isolated / "hf" / "token", "from-hf", 0o644)
    _stored(isolated, 'HF_TOKEN = "from-ml-stack"\n')
    assert credentials.get("HF_TOKEN") == "from-ml-stack"


def test_the_huggingface_file_is_only_for_hf_names(isolated):
    _file(isolated / "hf" / "token", "from-hf", 0o644)
    assert credentials.get("WIDGET_KEY") is None


# -- what is not trusted -------------------------------------------------------------------


@pytest.mark.skipif(sys.platform == "win32", reason="file modes are POSIX")
def test_a_file_other_users_can_read_is_refused_with_the_command_that_fixes_it(isolated):
    path = _stored(isolated, f'WIDGET_KEY = "{SECRET}"\n', 0o644)
    with pytest.raises(CredentialError, match=f"chmod 600 {path}") as raised:
        credentials.get("WIDGET_KEY")
    assert SECRET not in str(raised.value)
    path.chmod(0o640)
    with pytest.raises(CredentialError, match="chmod 600"):
        credentials.get("WIDGET_KEY")


@pytest.mark.skipif(sys.platform == "win32", reason="file modes are POSIX")
def test_the_loud_flag_reads_a_file_others_can_read(isolated, monkeypatch):
    _stored(isolated, 'WIDGET_KEY = "from-file"\n', 0o644)
    monkeypatch.setenv(credentials.INSECURE_ENV, "yes")
    with pytest.raises(CredentialError):
        credentials.get("WIDGET_KEY")
    monkeypatch.setenv(credentials.INSECURE_ENV, "yes-read-it-anyway")
    assert credentials.get("WIDGET_KEY") == "from-file"


@pytest.mark.skipif(sys.platform == "win32", reason="file modes are POSIX")
def test_a_file_owned_by_someone_else_is_refused(isolated, monkeypatch):
    _stored(isolated, 'WIDGET_KEY = "x"\n')
    monkeypatch.setattr(os, "geteuid", lambda: isolated.stat().st_uid + 1)
    with pytest.raises(CredentialError, match="another user"):
        credentials.get("WIDGET_KEY")


@pytest.mark.skipif(sys.platform == "win32", reason="file modes are POSIX")
def test_a_symlink_out_of_the_config_directory_is_refused(isolated):
    outside = _file(isolated / "outside.toml", 'WIDGET_KEY = "elsewhere"\n')
    link = isolated / "home" / "credentials.toml"
    link.parent.mkdir(parents=True)
    link.symlink_to(outside)
    with pytest.raises(CredentialError, match="symlink"):
        credentials.get("WIDGET_KEY")


@pytest.mark.skipif(sys.platform == "win32", reason="file modes are POSIX")
def test_a_symlink_inside_the_config_directory_is_followed(isolated):
    real = _file(isolated / "home" / "real.toml", 'WIDGET_KEY = "inside"\n')
    (isolated / "home" / "credentials.toml").symlink_to(real)
    assert credentials.get("WIDGET_KEY") == "inside"


def test_the_override_variable_moves_the_file(isolated, monkeypatch):
    moved = _file(isolated / "mine" / "keys.toml", 'WIDGET_KEY = "moved"\n')
    monkeypatch.setenv("ML_STACK_CREDENTIALS_FILE", str(moved))
    assert credentials.file_path() == moved
    assert credentials.get("WIDGET_KEY") == "moved"


def test_a_file_over_the_size_cap_is_refused(isolated):
    _stored(isolated, "# " + "x" * (70 * 1024))
    with pytest.raises(CredentialError, match="larger"):
        credentials.get("WIDGET_KEY")


def test_a_named_file_over_the_size_cap_is_refused(isolated, monkeypatch):
    monkeypatch.setenv("WIDGET_KEY_FILE", str(_file(isolated / "big", "x" * (70 * 1024))))
    with pytest.raises(CredentialError, match="larger"):
        credentials.get("WIDGET_KEY")


@pytest.mark.parametrize("text", ["WIDGET_KEY = ", "[[[", 'WIDGET_KEY = "unclosed',
                                  "\x00\x01\x02not toml"])
def test_a_file_that_is_not_toml_is_refused_without_quoting_it(isolated, text):
    _stored(isolated, text + SECRET)
    with pytest.raises(CredentialError, match="not valid TOML") as raised:
        credentials.get("WIDGET_KEY")
    assert SECRET not in str(raised.value) and SECRET not in repr(raised.value.__cause__)


def test_a_file_that_is_not_text_is_refused(isolated):
    path = isolated / "home" / "credentials.toml"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"\xff\xfe\x00\x80")
    path.chmod(0o600)
    with pytest.raises(CredentialError, match="UTF-8"):
        credentials.get("WIDGET_KEY")


def test_a_directory_is_not_a_credentials_file(isolated):
    (isolated / "home" / "credentials.toml").mkdir(parents=True)
    with pytest.raises(CredentialError, match="regular file"):
        credentials.get("WIDGET_KEY")


def test_an_entry_that_is_not_a_string_is_refused(isolated):
    _stored(isolated, "WIDGET_KEY = 5\n")
    with pytest.raises(CredentialError, match="not strings"):
        credentials.get("WIDGET_KEY")


@pytest.mark.parametrize("value", ["two\nlines", "tab\there", "nul\x00byte", "x" * 9000, ""])
def test_a_value_that_is_not_one_printable_line_is_refused(isolated, value):
    with pytest.raises(CredentialError):
        credentials.set("WIDGET_KEY", value)
    assert not (isolated / "home" / "credentials.toml").exists()


def test_surrounding_whitespace_and_newlines_are_stripped(isolated, monkeypatch):
    monkeypatch.setenv("WIDGET_KEY_FILE", str(_file(isolated / "m", f"  {SECRET} \r\n\n")))
    assert credentials.get("WIDGET_KEY") == SECRET


# -- writing -------------------------------------------------------------------------------


@pytest.mark.skipif(sys.platform == "win32", reason="file modes are POSIX")
def test_set_writes_mode_0600_in_a_directory_only_the_owner_enters(isolated):
    where = credentials.set("WIDGET_KEY", SECRET)
    path = isolated / "home" / "credentials.toml"
    assert where == str(path)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    assert sorted(one.name for one in path.parent.iterdir()) == ["credentials.toml"]
    assert credentials.get("WIDGET_KEY") == SECRET


def test_set_keeps_the_other_entries_and_round_trips_awkward_characters(isolated):
    awkward = 'a"b\\c =#d'
    credentials.set("ONE", "1")
    credentials.set("WIDGET_KEY", awkward)
    assert credentials.get("ONE") == "1" and credentials.get("WIDGET_KEY") == awkward


@pytest.mark.skipif(sys.platform == "win32", reason="file modes are POSIX")
def test_a_write_that_fails_leaves_the_old_file_whole(isolated, monkeypatch):
    credentials.set("ONE", "1")
    path = isolated / "home" / "credentials.toml"
    before = path.read_text()
    monkeypatch.setattr("ml_stack.credentials.writing.render",
                        lambda entries: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError):
        credentials.set("TWO", "2")
    assert path.read_text() == before
    assert sorted(one.name for one in path.parent.iterdir()) == ["credentials.toml"]


@pytest.mark.skipif(sys.platform == "win32", reason="file modes are POSIX")
def test_set_refuses_a_directory_other_users_can_write(isolated):
    home = isolated / "home"
    home.mkdir()
    home.chmod(0o777)
    with pytest.raises(CredentialError, match="chmod 700"):
        credentials.set("WIDGET_KEY", SECRET)


@pytest.mark.skipif(sys.platform == "win32", reason="file modes are POSIX")
def test_set_will_not_rewrite_a_file_it_would_not_read(isolated):
    path = _stored(isolated, 'ONE = "1"\n', 0o644)
    with pytest.raises(CredentialError, match="chmod 600"):
        credentials.set("TWO", "2")
    assert path.read_text() == 'ONE = "1"\n'


def test_unset_removes_one_entry_and_says_whether_it_was_there(isolated):
    credentials.set("ONE", "1")
    credentials.set("TWO", "2")
    assert credentials.unset("ONE") is True
    assert credentials.unset("ONE") is False
    assert credentials.get("ONE") is None and credentials.get("TWO") == "2"


def test_the_keychain_is_used_only_when_asked_for(isolated, monkeypatch):
    chain = MemoryKeychain()
    monkeypatch.setattr(credentials, "_keyring", lambda: chain)
    assert credentials.set("WIDGET_KEY", SECRET, keychain=True) == "keychain"
    assert not (isolated / "home" / "credentials.toml").exists()
    assert credentials.get("WIDGET_KEY") == SECRET
    assert credentials.unset("WIDGET_KEY", keychain=True) is True
    assert credentials.get("WIDGET_KEY") is None


def test_the_keychain_without_keyring_installed_says_what_to_install():
    with pytest.raises(CredentialError, match="pip install keyring"):
        credentials.set("WIDGET_KEY", SECRET, keychain=True)


# -- what is never said --------------------------------------------------------------------


def test_a_secret_prints_as_nothing_and_is_not_pickled():
    import pickle

    found = Secret(SECRET)
    assert SECRET not in repr(found) and SECRET not in repr([found]) \
        and SECRET not in repr({"key": found})
    assert f"Bearer {found}" == f"Bearer {SECRET}"
    with pytest.raises(TypeError):
        pickle.dumps(found)


def test_the_resolver_logs_nothing_with_the_value_in_it(isolated, monkeypatch, caplog):
    caplog.set_level(logging.DEBUG)
    _stored(isolated, f'WIDGET_KEY = "{SECRET}"\n')
    monkeypatch.setenv("HF_TOKEN", SECRET)
    credentials.get("WIDGET_KEY")
    credentials.get("HF_TOKEN")
    credentials.status("WIDGET_KEY")
    credentials.describe()
    credentials.set("OTHER", SECRET)
    credentials.unset("OTHER")
    assert SECRET not in caplog.text


def test_describe_and_status_report_sources_and_never_values(isolated, monkeypatch):
    _stored(isolated, f'WIDGET_KEY = "{SECRET}"\n')
    monkeypatch.setenv("HF_TOKEN", SECRET)
    rows = credentials.describe()
    assert {"name": "WIDGET_KEY", "present": True, "source": "credentials file"} in rows
    assert {"name": "HF_TOKEN", "present": True, "source": "environment"} in rows
    assert SECRET not in json.dumps(rows)
    assert SECRET not in json.dumps(credentials.status("WIDGET_KEY"))


@pytest.mark.skipif(sys.platform == "win32", reason="file modes are POSIX")
def test_status_reports_an_untrusted_file_as_an_error_not_a_value(isolated):
    _stored(isolated, f'WIDGET_KEY = "{SECRET}"\n', 0o644)
    got = credentials.status("WIDGET_KEY")
    assert got["present"] is False and "chmod 600" in got["error"] and SECRET not in str(got)


# -- the command ---------------------------------------------------------------------------


def _run(monkeypatch, capsys, argv, stdin=""):
    monkeypatch.setattr(sys, "stdin", io.StringIO(stdin))
    code = credentials_cli.command(argv)
    out = capsys.readouterr()
    return code, out.out, out.err


def test_set_reads_the_value_from_stdin_and_never_prints_it(isolated, monkeypatch, capsys):
    code, out, err = _run(monkeypatch, capsys, ["set", "WIDGET_KEY", "--stdin"], f"{SECRET}\n")
    assert code == 0 and SECRET not in out + err
    assert credentials.get("WIDGET_KEY") == SECRET


def test_list_names_each_credential_and_where_it_comes_from(isolated, monkeypatch, capsys):
    credentials.set("WIDGET_KEY", SECRET)
    code, out, err = _run(monkeypatch, capsys, ["list"])
    assert code == 0 and "WIDGET_KEY" in out and "credentials file" in out
    assert SECRET not in out + err
    code, out, _ = _run(monkeypatch, capsys, ["list", "--json"])
    assert any(row["name"] == "WIDGET_KEY" and row["present"] for row in json.loads(out))
    assert SECRET not in out


def test_unset_and_path(isolated, monkeypatch, capsys):
    credentials.set("WIDGET_KEY", SECRET)
    assert _run(monkeypatch, capsys, ["unset", "WIDGET_KEY"])[0] == 0
    assert credentials.get("WIDGET_KEY") is None
    code, out, _ = _run(monkeypatch, capsys, ["path"])
    assert code == 0 and out.strip() == str(isolated / "home" / "credentials.toml")


def test_a_refusal_is_a_message_and_status_1(isolated, monkeypatch, capsys):
    code, _, _ = _run(monkeypatch, capsys, ["set", "WIDGET_KEY", "--stdin"], "two\nlines\n")
    assert code == 0, "only the first line is the value"
    code, _, err = _run(monkeypatch, capsys, ["set", "bad name", "--stdin"], "x")
    assert code == 1 and "credential name" in err


# -- what a child process is given ---------------------------------------------------------


SECRETS = {"HF_TOKEN": "t1", "ANTHROPIC_API_KEY": "t2", "GITHUB_TOKEN": "t3",
           "AWS_SECRET_ACCESS_KEY": "t4", "DB_PASSWORD": "t5", "MY_SERVICE_CREDENTIALS": "t6",
           "SESSION_COOKIE": "t7", "ML_STACK_CREDENTIALS_FILE": "/x", "WIDGET_KEY_FILE": "/y"}


def test_a_child_gets_this_environment_without_anything_that_looks_like_a_secret(monkeypatch):
    for name, value in SECRETS.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("SSH_AUTH_SOCK", "/agent")
    monkeypatch.setenv("PATH", "/usr/bin")
    env = credentials.child_environment()
    assert not set(SECRETS) & set(env) - {"WIDGET_KEY_FILE"}
    assert env["PATH"] == "/usr/bin" and env["SSH_AUTH_SOCK"] == "/agent"


def test_a_child_is_handed_exactly_the_credential_it_needs():
    env = credentials.child_environment({"HF_TOKEN": "needed"}, source={"HF_TOKEN": "old",
                                                                         "OTHER_TOKEN": "x",
                                                                         "HOME": "/h"})
    assert env == {"HF_TOKEN": "needed", "HOME": "/h"}


def test_llama_server_gets_no_secret_but_the_hugging_face_token_it_downloads_with(
        monkeypatch, isolated):
    from ml_stack.serve.binary import child_env, hub_environment

    monkeypatch.setenv("HF_TOKEN", SECRET)
    monkeypatch.setenv("GITHUB_TOKEN", "other")
    assert "HF_TOKEN" not in child_env("/bin/llama-server")
    assert "GITHUB_TOKEN" not in hub_environment()
    assert hub_environment()["HF_TOKEN"] == SECRET


def test_the_token_comes_from_the_credentials_file_when_the_environment_has_none(isolated):
    from ml_stack.serve.binary import hub_environment

    _stored(isolated, f'HF_TOKEN = "{SECRET}"\n')
    assert hub_environment()["HF_TOKEN"] == SECRET
