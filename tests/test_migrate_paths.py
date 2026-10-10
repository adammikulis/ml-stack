"""`poolhouse migrate` leaves no absolute path into the old directories: links are repointed, a venv's
scripts are rewritten, any other file naming the old path is only reported. Real temporary trees."""

from __future__ import annotations

import stat
import time
from pathlib import Path

from poolhouse import migrate, migrate_paths, migrate_scan
from tests import migrate_support

account = migrate_support.account
old_state = migrate_support.old_state
ring = migrate_support.ring


def build(account: Path) -> tuple[Path, Path]:
    """An old state directory with a dangling-to-be link, a venv, a json, a binary and an outside link."""
    old = old_state(account)
    server = old / "llama.cpp" / "builds" / "b1" / "llama-server"
    server.parent.mkdir(parents=True)
    server.write_text("binary-ish", encoding="utf-8")
    (old / "llama.cpp" / "current").symlink_to(server.parent)
    venv = old / "spec-venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text(f"home = {old}/spec-venv/bin\ncommand = {old}/spec-venv/bin/python -m venv\n")
    tool = venv / "bin" / "tool"
    tool.write_text(f"#!{old}/spec-venv/bin/python\nprint('hi')\n")
    tool.chmod(0o755)
    for name in ("activate", "activate.csh", "activate.fish"):
        (venv / "bin" / name).write_text(f'VIRTUAL_ENV="{old}/spec-venv"\n')
    (old / "settings.json").write_text(f'{{"server": "{old}/llama.cpp/current/llama-server"}}')
    (old / "blob.dat").write_bytes(b"\0\1" + str(old).encode() + b"\0")
    (old / "weights.gguf").write_text(str(old))
    outside = account / "elsewhere"
    outside.mkdir()
    (old / "ext").symlink_to(outside)
    return old, outside


def moved(account: Path, tmp_path: Path) -> Path:
    assert migrate.run(account, ring=ring(tmp_path / "k")) == 0
    return account / ".poolhouse"


def files(root: Path) -> dict[Path, str | bytes]:
    return {p: (str(p.readlink()) if p.is_symlink() else p.read_bytes()) for p in sorted(root.rglob("*"))
            if (p.is_symlink() or p.is_file()) and p.name != "migrate.log"}


def test_a_link_with_an_absolute_old_target_is_repointed(account, tmp_path):
    build(account)
    new = moved(account, tmp_path)
    link = new / "llama.cpp" / "current"
    assert str(link.readlink()) == str(new / "llama.cpp" / "builds" / "b1")
    assert (link / "llama-server").read_text(encoding="utf-8") == "binary-ish"


def test_a_venv_has_its_shebangs_config_and_activate_files_rewritten(account, tmp_path):
    build(account)
    new = moved(account, tmp_path)
    venv = new / "spec-venv"
    assert (venv / "bin" / "tool").read_text(encoding="utf-8").startswith(f"#!{venv}/bin/python\n")
    assert stat.S_IMODE((venv / "bin" / "tool").stat().st_mode) == 0o755
    assert f"home = {venv}/bin" in (venv / "pyvenv.cfg").read_text(encoding="utf-8")
    for name in ("activate", "activate.csh", "activate.fish"):
        assert str(account / ".ml-stack") not in (venv / "bin" / name).read_text(encoding="utf-8")
    assert "backup" in (new / "migrate.log").read_text(encoding="utf-8")
    backups = list((new / migrate_paths.BACKUPS).glob("*.orig"))
    assert len(backups) == 5 and all(str(account / ".ml-stack") in b.read_text(encoding="utf-8") for b in backups)


def test_a_json_with_the_old_path_is_reported_and_not_changed(account, tmp_path, capsys):
    old, _outside = build(account)
    before = (old / "settings.json").read_text(encoding="utf-8")
    new = moved(account, tmp_path)
    assert (new / "settings.json").read_text(encoding="utf-8") == before
    assert f"names an old path, not changed: {new / 'settings.json'}" in capsys.readouterr().out
    assert f"left as it is: {new / 'settings.json'}" in (new / "migrate.log").read_text(encoding="utf-8")


def test_a_binary_and_a_model_file_are_untouched(account, tmp_path, capsys):
    old, _outside = build(account)
    blob, weights = (old / "blob.dat").read_bytes(), (old / "weights.gguf").read_bytes()
    new = moved(account, tmp_path)
    assert (new / "blob.dat").read_bytes() == blob and (new / "weights.gguf").read_bytes() == weights
    out = capsys.readouterr().out
    assert "blob.dat" not in out and "weights.gguf" not in out


def test_a_link_pointing_outside_the_state_directory_is_untouched(account, tmp_path):
    _old, outside = build(account)
    new = moved(account, tmp_path)
    assert str((new / "ext").readlink()) == str(outside)


def test_running_again_changes_nothing(account, tmp_path):
    build(account)
    new = moved(account, tmp_path)
    before = files(new)
    assert migrate.run(account, ring=ring(tmp_path / "k")) == 0
    assert files(new) == before
    assert "0 symlinks to repoint, 0 venv files to rewrite" in (new / "migrate.log").read_text(encoding="utf-8")


def test_plan_counts_what_run_would_do_and_changes_nothing(account, capsys):
    old, _outside = build(account)
    before = files(old)
    assert migrate.run(account, dry=True, ring=lambda: None) == 0
    out = capsys.readouterr().out
    assert "1 symlinks to repoint, 5 venv files to rewrite, 1 other files name an old path" in out
    assert "0 symlinks into an old directory with no new target" in out
    assert files(old) == before and not (account / ".poolhouse").exists()


def test_a_link_into_the_old_directory_with_no_new_target_is_listed_not_invented(account, tmp_path, capsys):
    old = old_state(account)
    (old / "gone").symlink_to(old / "builds" / "missing")
    new = moved(account, tmp_path)
    assert str((new / "gone").readlink()) == str(old / "builds" / "missing")
    assert migrate.verify(account) == 1
    assert f"dangling symlink {new / 'gone'}" in capsys.readouterr().err


def test_verify_is_clean_after_run_and_names_what_is_left_after_a_breakage(account, tmp_path, capsys):
    old, _outside = build(account)
    new = moved(account, tmp_path)
    assert migrate.verify(account) == 0
    (new / "spec-venv" / "bin" / "tool").write_text(f"#!{old}/spec-venv/bin/python\n", encoding="utf-8")
    (new / "dead").symlink_to(new / "nowhere")
    assert migrate.verify(account) == 1
    err = capsys.readouterr().err
    assert f"old path in {new / 'spec-venv' / 'bin' / 'tool'}" in err and f"dangling symlink {new / 'dead'}" in err


def test_a_machine_the_old_code_already_moved_is_repaired_by_run(account, tmp_path):
    old, _outside = build(account)
    old.rename(account / ".poolhouse")
    assert migrate.verify(account) == 1
    assert migrate.run(account, ring=ring(tmp_path / "k")) == 0
    assert migrate.verify(account) == 0


def test_the_command_group_lists_verify():
    assert "verify" in {command.name for command in migrate.GROUP.commands}


def big_tree(account: Path, count: int = 200_000) -> Path:
    """An old state directory with a huge pruned models directory, a dangling link and a venv."""
    old = old_state(account)
    models = old / "models" / "family"
    models.mkdir(parents=True)
    for i in range(count):
        (models / f"{i}.x").write_text(str(old), encoding="utf-8")
    (old / "gone").symlink_to(old / "builds" / "missing")
    venv = old / "spec-venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text(f"home = {old}/spec-venv/bin\n")
    (venv / "bin" / "tool").write_text(f"#!{old}/spec-venv/bin/python\n")
    return old


def test_the_default_scan_skips_a_huge_pruned_directory_and_still_finds_the_link_and_the_venv(account):
    old = big_tree(account)
    started = time.monotonic()
    found = migrate_paths.scan([old], migrate._pairs(account), planning=True)
    assert time.monotonic() - started < 1.0
    assert [p.name for p in found.stuck] == ["gone"]
    assert sorted(p.name for p in found.rewrite) == ["pyvenv.cfg", "tool"]
    assert not found.report and not found.stopped


def test_deep_reads_a_settings_file_the_default_scan_does_not_reach(account):
    old = old_state(account)
    nest = old / "deep" / "er" / "than" / "the" / "default"
    nest.mkdir(parents=True)
    (nest / "a.json").write_text(f'{{"p": "{old}/x"}}', encoding="utf-8")
    pairs = migrate._pairs(account)
    assert not migrate_paths.scan([old], pairs, planning=True).report
    assert migrate_paths.scan([old], pairs, planning=True, budget=migrate_scan.Budget.of(True)).report == [nest / "a.json"]


def test_deep_says_where_it_is_every_five_seconds(account):
    old = old_state(account)
    lines: list[str] = []
    ticks = iter(range(0, 1000, 6))
    clock = migrate_scan.Budget(None, lines.append, clock=lambda: next(ticks))
    list(migrate_scan.candidates(old, clock))
    assert lines and "entries seen, now in" in lines[0]


def test_a_budget_of_zero_trips_and_names_what_it_left(account):
    old = big_tree(account, 10)
    found = migrate_paths.scan([old], migrate._pairs(account), planning=True, budget=migrate_scan.Budget(0))
    assert found.skipped == [old] and str(old) in found.stopped and "--deep" in found.stopped
    assert migrate_paths.scan([old], migrate._pairs(account), planning=True, budget=migrate_scan.Budget.of(True)).stopped == ""


def test_plan_prints_the_elapsed_time_and_deep_is_a_flag_of_every_command(account, capsys):
    build(account)
    assert migrate.run(account, dry=True, ring=lambda: None) == 0
    assert "scanned in" in capsys.readouterr().out
    assert all(any("--deep" in o.flags for o in c.options) for c in migrate.GROUP.commands)
