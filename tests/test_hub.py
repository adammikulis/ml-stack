"""What a model card asks for, read out of prose -- and out of the file itself."""
import pytest

from ml_stack.hub import advice


def test_advice_reads_the_settings_a_card_names():
    said = """
    ## Best Practices
    ### 1. Sampling Parameters
    Use the following standardized sampling configuration across all use cases:
    * `temperature=1.0`
    * `top_p=0.95`
    * `top_k=64`
    """
    assert advice(said) == {"temperature": 1.0, "top_p": 0.95, "top_k": 64.0}


def test_advice_reads_the_several_ways_a_card_writes_them():
    assert advice('"temperature": 0.7') == {"temperature": 0.7}
    assert advice("temp: 0.6, top-k: 20") == {"temperature": 0.6, "top_k": 20.0}
    assert advice("repetition_penalty = 1.05") == {"repeat_penalty": 1.05}
    # a card that only shows a command line still reads
    assert advice("llama-server --temp 0.6 --top-k 20") == {"temperature": 0.6, "top_k": 20.0}


def test_the_first_mention_wins_inside_the_recommending_section():
    """A card names its recommendation, then shows variations of it."""
    assert advice("temperature=1.0 ... later, try temperature=0.3") == {"temperature": 1.0}


def test_the_recommending_section_is_read_before_the_rest():
    """Otherwise a card that opens by warning against a setting reads as recommending it.

    "First mention wins" is only safe once you know which part is doing the recommending.
    """
    said = """
    # Notes
    Do not use `temperature=0.0` with this model; it degenerates.

    ## Recommended sampling
    * `temperature=0.8`
    * `top_p=0.9`

    ## Troubleshooting
    If output repeats, try `temperature=1.4`.
    """
    assert advice(said) == {"temperature": 0.8, "top_p": 0.9}


def test_a_card_with_no_marked_section_falls_back_to_the_whole_document():
    assert advice("Just use temperature=0.7 and you'll be fine.") == {"temperature": 0.7}


def test_a_card_that_says_nothing_says_nothing():
    """Empty is a real answer: nobody chose, so the caller's default stands. Inventing a
    number here would be putting words in a publisher's mouth."""
    assert advice("") == {}
    assert advice("This model is good at reasoning.") == {}
    assert advice("gpt-oss supports configurable reasoning effort: low, medium, high.") == {}


def test_a_draft_head_is_found_wherever_the_publisher_put_it(monkeypatch):
    """Not every repository puts it in the same place, and one rule missed two of three."""
    import ml_stack.hub as hub

    shelves = {
        # gemma's QAT repos carry it at the root and again under MTP/
        "maker/gem-GGUF": [("gem-Q4.gguf", 4_000_000_000),
                           ("MTP/mtp-gem-Q4_0.gguf", 56_000_000),
                           ("mtp-gem.gguf", 56_000_000)],
        # Qwen3.8-27B carries it only under MTP/, which a "no slashes" rule skipped entirely
        "maker/big-GGUF": [("big-Q4.gguf", 17_000_000_000),
                           ("MTP/mtp-big-Q4_0.gguf", 1_300_000_000)],
        # nothing at all, and no sibling either
        "maker/bare-GGUF": [("bare-Q4.gguf", 9_000_000_000)],
        # the head shipped as its own repository beside the weights
        "maker/split-GGUF": [("split-Q4.gguf", 9_000_000_000)],
        "maker/split-MTP-GGUF": [("mtp-split.gguf", 300_000_000)],
        # a "-MTP-" repository that is NOT a draft: the whole model rebuilt with the
        # prediction layers in it, to be served with --spec-type, not as a second model
        "maker/moe-GGUF": [("moe-Q4.gguf", 20_000_000_000)],
        "maker/moe-MTP-GGUF": [("moe-UD-Q8_K_XL.gguf", 36_400_000_000)],
    }
    monkeypatch.setattr(hub, "files", lambda repo, **kw: shelves.get(repo, []))

    assert hub.draft_for("maker/gem-GGUF") == "hf:maker/gem-GGUF/mtp-gem.gguf"
    assert hub.draft_for("maker/big-GGUF") == "hf:maker/big-GGUF/MTP/mtp-big-Q4_0.gguf"
    assert hub.draft_for("maker/split-GGUF") == "hf:maker/split-MTP-GGUF/mtp-split.gguf"
    assert hub.draft_for("maker/bare-GGUF") == ""
    # 36G of weights is not a draft head, whatever the repository is called
    assert hub.draft_for("maker/moe-GGUF") == ""


def test_a_repository_that_is_not_there_is_not_a_draft(monkeypatch):
    import ml_stack.hub as hub

    def missing(repo, **kw):
        raise OSError("404")

    monkeypatch.setattr(hub, "files", missing)
    assert hub.draft_for("maker/whatever-GGUF") == ""


def test_shards_are_totalled_into_builds(monkeypatch):
    """A large model is published in shards, one directory per quantisation. Forty lines of
    individual files answers no question anybody has -- what decides whether a model can be
    served is the total of a build, and adding those up by hand is the step this removes."""
    import ml_stack.hub as hub

    shelves = [
        ("BF16/thing-BF16-00001-of-00002.gguf", 60_000_000_000),
        ("BF16/thing-BF16-00002-of-00002.gguf", 40_000_000_000),
        ("UD-Q4_K_XL/thing-UD-Q4_K_XL-00001-of-00002.gguf", 15_000_000_000),
        ("UD-Q4_K_XL/thing-UD-Q4_K_XL-00002-of-00002.gguf", 10_000_000_000),
        ("thing-Q8_0.gguf", 30_000_000_000),
        ("mmproj-F32.gguf", 900_000_000),      # a companion, not a build
        ("MTP/mtp-thing.gguf", 1_000_000_000),
    ]
    monkeypatch.setattr(hub, "files", lambda repo, **kw: shelves)

    got = hub.builds("maker/thing-GGUF")
    assert got == [("BF16", 100_000_000_000, 2),
                   ("thing-Q8_0.gguf", 30_000_000_000, 1),
                   ("UD-Q4_K_XL", 25_000_000_000, 2)]
    assert not any(name.startswith(("mmproj", "MTP")) for name, _s, _n in got)


def test_a_subdirectory_does_not_make_something_a_companion():
    """Calling every sharded weight "alongside" buries the model under its own projector."""
    from ml_stack.hub import aside

    assert aside("UD-Q4_K_XL/thing-00001-of-00004.gguf") == 0
    assert aside("thing-Q4_K_M.gguf") == 0
    assert aside("mmproj-F32.gguf") == 1
    assert aside("MTP/mtp-thing.gguf") == 1
    assert aside("mtp-thing.gguf") == 1
    assert aside("imatrix_unsloth.gguf") == 1


def test_room_is_what_can_be_served_not_what_is_installed(monkeypatch):
    """On unified memory this is the wired limit, not the whole of RAM: a model and its KV
    cache have to fit under what Metal will wire, and `free` does not report that."""
    import ml_stack.hub as hub

    import subprocess

    class Done:
        def __init__(self, out): self.returncode, self.stdout = 0, out

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: Done("112640\n"))
    assert hub.room() == 112640 * 1024 * 1024

    def refused(*a, **k):
        raise OSError("no sysctl here")

    monkeypatch.setattr(subprocess, "run", refused)
    assert hub.room() == 0, "a machine that will not say is not a machine with no memory"


def test_a_listing_says_what_is_already_downloaded(monkeypatch, capsys):
    """Nothing said what was local, so 87G that was already on the disk was nearly fetched
    again. `fleet.models` had known all along; the listing just never asked."""
    import ml_stack.hub as hub

    shelves = [("UD-IQ4_XS/thing-UD-IQ4_XS-00001-of-00002.gguf", 40_000_000_000),
               ("UD-IQ4_XS/thing-UD-IQ4_XS-00002-of-00002.gguf", 46_000_000_000),
               ("BF16/thing-BF16-00001-of-00001.gguf", 300_000_000_000)]
    monkeypatch.setattr(hub, "files", lambda repo, **kw: shelves)
    monkeypatch.setattr(hub, "room", lambda: 110 * 2**30)
    monkeypatch.setattr(hub, "held", lambda: {
        "thing-UD-IQ4_XS-00001-of-00002.gguf": 40_000_000_000,
        "thing-UD-IQ4_XS-00002-of-00002.gguf": 46_000_000_000})

    assert hub.main(["files", "maker/thing-GGUF"]) == 0
    said = capsys.readouterr().out
    assert "ON THIS MACHINE" in said
    # the build that is not here, and could not be served anyway, says both
    assert "TOO BIG" in said
    assert said.index("ON THIS MACHINE") > said.index("TOO BIG"), "largest first"


def test_a_partly_downloaded_build_says_so(monkeypatch, capsys):
    """An interrupted fetch is not the same as having it, and looks identical on disk."""
    import ml_stack.hub as hub

    monkeypatch.setattr(hub, "files", lambda repo, **kw: [
        ("UD-IQ4_XS/thing-UD-IQ4_XS-00001-of-00003.gguf", 1),
        ("UD-IQ4_XS/thing-UD-IQ4_XS-00002-of-00003.gguf", 1),
        ("UD-IQ4_XS/thing-UD-IQ4_XS-00003-of-00003.gguf", 1)])
    monkeypatch.setattr(hub, "room", lambda: 110 * 2**30)
    monkeypatch.setattr(hub, "held", lambda: {"thing-UD-IQ4_XS-00001-of-00003.gguf": 1})

    hub.main(["files", "maker/thing-GGUF"])
    said = capsys.readouterr().out
    assert "1/3 downloaded" in said and "ON THIS MACHINE" not in said


def test_held_resolves_symlinks_because_a_cache_is_made_of_them(tmp_path, monkeypatch):
    """A Hub cache is symlinks into blobs/, so `ls -l` reports 79 bytes for a 46G
    shard. Reading that as "not downloaded" is the same mistake wearing a different hat."""
    import ml_stack.hub as hub

    blob = tmp_path / "blobs" / "abc123"
    blob.parent.mkdir()
    blob.write_bytes(b"x" * 5000)
    link = tmp_path / "thing-Q4.gguf"
    link.symlink_to(blob)
    assert link.lstat().st_size < 200, "the link itself is tiny, which is the trap"

    class Found:
        path = link

    monkeypatch.setattr("ml_stack.fleet.models.Models.all", lambda self: [Found()])
    assert hub.held()["thing-Q4.gguf"] == 5000


def _gguf(tmp_path, pairs):
    """A minimal GGUF carrying only metadata, for reading it back."""
    import struct

    out = bytearray(b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0)
                    + struct.pack("<Q", len(pairs)))
    for key, value in pairs.items():
        out += struct.pack("<Q", len(key)) + key.encode()
        out += struct.pack("<I", 6) + struct.pack("<f", value)     # 6 == float32
    where = tmp_path / "model.gguf"
    where.write_bytes(bytes(out))
    return where


def test_sampling_is_read_from_the_model_file_itself(tmp_path):
    """`general.sampling.*` is written into the GGUF, so it cannot drift from the weights
    and needs no prose parsed out of a README. Qwen3.8 carries it; many models do not."""
    from ml_stack.hub import in_gguf

    where = _gguf(tmp_path, {"general.sampling.temp": 1.0,
                             "general.sampling.top_k": 20.0,
                             "general.sampling.top_p": 0.95,
                             "general.unrelated": 7.0})
    got = in_gguf(where)
    assert set(got) == {"temperature", "top_k", "top_p"}
    assert got["temperature"] == 1.0 and got["top_k"] == 20.0
    # stored as float32, so it reads back as 0.949999988 -- compare as a number, not a string
    assert got["top_p"] == pytest.approx(0.95)


def test_a_file_with_nothing_to_say_says_nothing(tmp_path):
    from ml_stack.hub import in_gguf

    assert in_gguf(_gguf(tmp_path, {"general.file_type": 30.0})) == {}
    assert in_gguf(tmp_path / "does-not-exist.gguf") == {}
    not_a_gguf = tmp_path / "x.gguf"
    not_a_gguf.write_bytes(b"this is not a gguf at all")
    assert in_gguf(not_a_gguf) == {}


def test_a_draft_head_is_found_whatever_method_it_implements(monkeypatch):
    """A head is named by its method, not by being a draft. Knowing only `mtp-` reported
    "no draft" for gpt-oss-120b, which ships two EAGLE3 heads and no mtp- file -- so it was
    served unaccelerated with nothing saying why."""
    import ml_stack.hub as hub

    shelves = {
        "maker/oss-GGUF": [("oss-MXFP4.gguf", 60_000_000_000),
                           ("eagle3-oss-BF16.gguf", 1_500_000_000),
                           ("eagle3-oss-Q8_0.gguf", 810_000_000)],
        "maker/gem-GGUF": [("gem-Q4.gguf", 4_000_000_000),
                           ("mtp-gem.gguf", 56_000_000)],
        "maker/bare-GGUF": [("bare-Q4.gguf", 9_000_000_000)],
    }
    monkeypatch.setattr(hub, "files", lambda repo, **kw: shelves.get(repo, []))

    assert hub.draft_for("maker/oss-GGUF") == "hf:maker/oss-GGUF/eagle3-oss-BF16.gguf"
    assert hub.draft_for("maker/gem-GGUF") == "hf:maker/gem-GGUF/mtp-gem.gguf"
    assert hub.draft_for("maker/bare-GGUF") == ""


def test_a_head_says_which_kind_of_speculation_it_needs():
    """A head implements one method. An EAGLE3 head served as draft-simple is not slower,
    it is being asked to do something it does not do."""
    from ml_stack.hub import spec_for

    assert spec_for("hf:maker/x/eagle3-oss-Q8_0.gguf") == "draft-eagle3"
    assert spec_for("/models/mtp-gemma-4-E4B-it.gguf") == "draft-mtp"
    assert spec_for("MTP/mtp-thing.gguf") == "draft-mtp"
    # a whole model used as a draft implements nothing in particular, and says so
    assert spec_for("/models/gpt-oss-20b-MXFP4.gguf") == ""
    assert spec_for("") == ""


class TestFetch:
    """Downloading an `hf:` reference into the cache without serving it -- what a preflight
    calls so a download never happens inside a benchmark's timed window."""

    def test_every_shard_of_the_named_build_is_downloaded(self, tmp_path, monkeypatch):
        import huggingface_hub
        import ml_stack.hub as hub

        shelves = [
            ("thing-00001-of-00002.gguf", 4_000_000_000),
            ("thing-00002-of-00002.gguf", 3_000_000_000),
            ("mmproj-F32.gguf", 900_000_000),      # a companion, not this build
        ]
        monkeypatch.setattr(hub, "files", lambda repo, **kw: shelves)

        downloaded: list[str] = []

        def fake_download(repo_id, filename, **kw):
            downloaded.append(filename)
            target = tmp_path / filename
            target.write_bytes(b"x")
            return str(target)

        monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)

        got = hub.fetch("hf:maker/thing-GGUF/thing-00001-of-00002.gguf")
        assert downloaded == ["thing-00001-of-00002.gguf", "thing-00002-of-00002.gguf"]
        assert got.name == "thing-00001-of-00002.gguf"

    def test_an_unsharded_reference_downloads_just_the_one_file(self, tmp_path, monkeypatch):
        import huggingface_hub
        import ml_stack.hub as hub

        monkeypatch.setattr(hub, "files",
                            lambda repo, **kw: [("thing-Q4_K_M.gguf", 4_000_000_000)])
        downloaded: list[str] = []

        def fake_download(repo_id, filename, **kw):
            downloaded.append(filename)
            target = tmp_path / filename
            target.write_bytes(b"x")
            return str(target)

        monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)

        got = hub.fetch("hf:maker/thing-GGUF/thing-Q4_K_M.gguf")
        assert downloaded == ["thing-Q4_K_M.gguf"]
        assert got.name == "thing-Q4_K_M.gguf"

    def test_a_reference_with_no_file_is_rejected(self):
        import ml_stack.hub as hub

        with pytest.raises(ValueError, match="hf:owner/repo/file.gguf"):
            hub.fetch("hf:maker/thing-GGUF")
