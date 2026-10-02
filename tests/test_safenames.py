"""A name or an archive from somewhere else never writes outside where it was put."""

import io
import os
import sys
import tarfile
import zipfile

import pytest

from ml_stack.safenames import Unsafe, safe_filename, safe_join, unpack


@pytest.mark.parametrize("name", [
    "model.gguf", "Qwen3-8B-Q4_K_M.gguf", "a b.txt", "v1.2.3", ".hidden", "ümlaut.bin",
    "com10.txt", "console.log",
])
def test_ordinary_names_pass(name):
    assert safe_filename(name) == name


@pytest.mark.parametrize("name", [
    "", " ", ".", "..", "a/b", "a\\b", "/etc/passwd", "C:evil", "a:b", "x" * 201,
    "nul", "NUL", "con.txt", "COM1", "lpt9.log", "aux.tar.gz", "trailing.", " lead", "trail ",
    "tab\there", "nl\nname", "nul\x00byte", "bell\x07", "q?", "star*", "pipe|", "lt<", "gt>",
    'quote"', "\u202eevil.gguf",
])
def test_names_that_are_not_one_plain_file_name_are_refused(name):
    with pytest.raises(Unsafe):
        safe_filename(name)


def test_a_non_string_is_refused():
    with pytest.raises(Unsafe):
        safe_filename(None)


@pytest.mark.parametrize("relative", [
    "../x", "a/../../x", "/abs", "\\abs", "C:\\x", "a/../../../etc", "a\\..\\..\\x", "..",
    "a/nul", "a/b/con.txt",
])
def test_a_path_that_leaves_its_root_or_holds_a_bad_part_is_refused(tmp_path, relative):
    with pytest.raises(Unsafe):
        safe_join(tmp_path, relative)


def test_a_path_inside_its_root_is_joined(tmp_path):
    assert safe_join(tmp_path, "a/b/c.bin") == (tmp_path / "a" / "b" / "c.bin").resolve()


@pytest.mark.skipif(sys.platform == "win32", reason="symlinks need privileges on Windows")
def test_a_symlink_already_on_disk_cannot_carry_a_path_out(tmp_path):
    (tmp_path / "root").mkdir()
    (tmp_path / "root" / "out").symlink_to(tmp_path)
    with pytest.raises(Unsafe):
        safe_join(tmp_path / "root", "out/x.bin")


def _zip(path, entries):
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return path


def _tar(path, build):
    with tarfile.open(path, "w:gz") as tf:
        build(tf)
    return path


def _file(tf, name, data=b"x", mode=0o644):
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mode = mode
    tf.addfile(info, io.BytesIO(data))


def test_a_zip_is_unpacked(tmp_path):
    archive = _zip(tmp_path / "a.zip", {"bin/llama-server": b"abc", "lib/libx.so": b"def"})
    got = unpack(archive, tmp_path / "out")
    assert sorted(p.name for p in got) == ["libx.so", "llama-server"]
    assert (tmp_path / "out" / "bin" / "llama-server").read_bytes() == b"abc"


@pytest.mark.parametrize("name", ["../evil", "a/../../evil", "/abs/evil", "C:/evil", "nul",
                                  "dir/con.txt"])
def test_a_zip_entry_that_would_escape_or_is_a_device_is_refused_and_nothing_is_written(
        tmp_path, name):
    archive = _zip(tmp_path / "a.zip", {"fine.txt": b"ok", name: b"bad"})
    with pytest.raises(Unsafe):
        unpack(archive, tmp_path / "out")
    assert not (tmp_path / "evil").exists() and not (tmp_path.parent / "evil").exists()


def test_a_zip_symlink_entry_is_refused(tmp_path):
    archive = tmp_path / "a.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        info = zipfile.ZipInfo("link")
        info.external_attr = (0o120777 << 16)
        zf.writestr(info, "/etc/passwd")
    with pytest.raises(Unsafe, match="link"):
        unpack(archive, tmp_path / "out")


def test_a_zip_that_unpacks_to_too_much_or_holds_too_many_entries_is_refused(tmp_path):
    archive = _zip(tmp_path / "a.zip", {"big": b"\0" * 10_000})
    with pytest.raises(Unsafe, match="more than"):
        unpack(archive, tmp_path / "o1", max_bytes=1000)
    many = _zip(tmp_path / "m.zip", {f"f{n}": b"" for n in range(20)})
    with pytest.raises(Unsafe, match="entries"):
        unpack(many, tmp_path / "o2", max_entries=10)


def test_a_tar_is_unpacked_with_its_modes_and_relative_symlinks(tmp_path):
    def build(tf):
        _file(tf, "llama-server", b"bin", mode=0o755)
        _file(tf, "libllama.0.dylib", b"lib")
        link = tarfile.TarInfo("libllama.dylib")
        link.type = tarfile.SYMTYPE
        link.linkname = "libllama.0.dylib"
        tf.addfile(link)

    unpack(_tar(tmp_path / "a.tar.gz", build), tmp_path / "out")
    out = tmp_path / "out"
    assert (out / "llama-server").read_bytes() == b"bin"
    if sys.platform != "win32":
        assert os.access(out / "llama-server", os.X_OK)
        assert (out / "libllama.dylib").is_symlink()
        assert (out / "libllama.dylib").read_bytes() == b"lib"


@pytest.mark.parametrize("linkname", ["/etc/passwd", "../../../etc/passwd", "C:\\Windows", "a/../.."])
def test_a_tar_symlink_that_points_outside_is_refused(tmp_path, linkname):
    def build(tf):
        link = tarfile.TarInfo("sneaky")
        link.type = tarfile.SYMTYPE
        link.linkname = linkname
        tf.addfile(link)

    with pytest.raises(Unsafe):
        unpack(_tar(tmp_path / "a.tar.gz", build), tmp_path / "out")


def test_a_tar_hard_link_or_device_is_refused(tmp_path):
    def hard(tf):
        _file(tf, "a")
        link = tarfile.TarInfo("b")
        link.type = tarfile.LNKTYPE
        link.linkname = "a"
        tf.addfile(link)

    def device(tf):
        node = tarfile.TarInfo("dev")
        node.type = tarfile.CHRTYPE
        tf.addfile(node)

    for build in (hard, device):
        with pytest.raises(Unsafe, match="hard link or a device"):
            unpack(_tar(tmp_path / f"{build.__name__}.tar", build), tmp_path / build.__name__)


@pytest.mark.parametrize("name", ["../evil", "/abs/evil", "a/../../evil"])
def test_a_tar_entry_that_would_escape_is_refused(tmp_path, name):
    def build(tf):
        _file(tf, name)

    with pytest.raises(Unsafe):
        unpack(_tar(tmp_path / "a.tar.gz", build), tmp_path / "out")
    assert not (tmp_path / "evil").exists()


def test_a_tar_over_its_caps_is_refused(tmp_path):
    def big(tf):
        _file(tf, "big", b"\0" * 10_000)

    with pytest.raises(Unsafe, match="more than"):
        unpack(_tar(tmp_path / "a.tar.gz", big), tmp_path / "o", max_bytes=100)
