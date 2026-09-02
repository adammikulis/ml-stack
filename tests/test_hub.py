"""What a model card asks for, read out of prose -- and out of the file itself."""
import pytest

from ml_stack.hub import advice


@pytest.fixture(autouse=True)
def _no_network_for_draft_note(monkeypatch):
    """``draft_for``'s default ``borrows=False`` path asks ``draft_note`` whether a found
    head's own README warns about needing a fork, which would otherwise reach the real Hub
    in every test that finds one. A test that cares about ``draft_note`` itself overrides
    this locally with its own fake."""
    import huggingface_hub
    import ml_stack.hub as hub

    def refuse(*a, **k):
        raise OSError("no network in tests")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", refuse)
    hub._DRAFT_NOTES.clear()   # the cache is process-wide; each test starts with none
    yield


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


class TestDraftForBorrows:
    """`--draft auto` must not offer a head that needs a fork unless the binary about to
    serve it can borrow -- measured for real: every `mtp-` head under
    unsloth/Qwen3.8-Flash-Next-GGUF/MTP/ fails on mainline llama.cpp master with
    `check_tensor_dims: tensor 'output_hc_norm.weight' not found`."""

    SHELVES = {
        "maker/thing-GGUF": [
            ("thing-Q4.gguf", 4_000_000_000),
            ("MTP/mtp-thing-shared-BF16.gguf", 5_000_000_000),
            ("MTP/mtp-thing-shared-Q8_0.gguf", 2_600_000_000),
            ("MTP/mtp-thing-Q8_0.gguf", 3_900_000_000),
        ],
    }

    def _repo(self, monkeypatch, tmp_path, *, note: str = ""):
        import huggingface_hub
        import ml_stack.hub as hub

        hub._DRAFT_NOTES.clear()
        monkeypatch.setattr(hub, "files", lambda repo, **kw: self.SHELVES.get(repo, []))

        readme = tmp_path / "MTP-README.md"
        readme.write_text(note)

        def fake_download(repo, filename, **kw):
            if note and filename == "MTP/README.md":
                return str(readme)
            raise OSError("no readme")

        monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
        return hub

    def test_a_head_whose_readme_warns_is_withheld_by_default(self, monkeypatch, tmp_path):
        hub = self._repo(monkeypatch, tmp_path,
                         note="These do not work on mainline ggml-org/llama.cpp yet.")

        assert hub.draft_for("maker/thing-GGUF") == ""
        assert hub.draft_for("maker/thing-GGUF", borrows=False) == ""

    def test_the_same_head_is_offered_when_the_binary_can_borrow(self, monkeypatch, tmp_path):
        hub = self._repo(monkeypatch, tmp_path,
                         note="These do not work on mainline ggml-org/llama.cpp yet.")

        found = hub.draft_for("maker/thing-GGUF", borrows=True)
        # the recommended shared-Q8_0 head is preferred over its shared-BF16 sibling
        assert found == "hf:maker/thing-GGUF/MTP/mtp-thing-shared-Q8_0.gguf"

    def test_a_head_with_no_warning_is_offered_either_way(self, monkeypatch, tmp_path):
        hub = self._repo(monkeypatch, tmp_path, note="")   # no MTP/README.md at all

        assert hub.draft_for("maker/thing-GGUF", borrows=False) != ""
        assert hub.draft_for("maker/thing-GGUF", borrows=False) == \
            hub.draft_for("maker/thing-GGUF", borrows=True)


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


class TestLocated:
    """A bare model name resolved against the Hub cache -- what fixed 'up' reading a name
    copied out of `ml-stack-models files` as a relative path and reporting shards missing
    for a model that was on the machine the whole time."""

    def test_an_exact_filename_is_found_as_the_link_whose_size_reads_through(self, tmp_path):
        import ml_stack.hub as hub

        cache = tmp_path / "hub"
        blob = cache / "models--maker--thing-GGUF" / "blobs" / "deadbeef"
        blob.parent.mkdir(parents=True)
        blob.write_bytes(b"x" * 4_000_000)
        snapshot = cache / "models--maker--thing-GGUF" / "snapshots" / "abc123"
        snapshot.mkdir(parents=True)
        (snapshot / "thing-Q4_K_M.gguf").symlink_to(blob)

        found = hub.located("thing-Q4_K_M.gguf", cache=cache)
        # the link, by name -- what llama.cpp needs to find a sharded model's other
        # shards -- with the blob's size readable through it
        assert found == snapshot / "thing-Q4_K_M.gguf" and found.is_symlink()
        assert found.stat().st_size == blob.stat().st_size

    def test_a_shard_less_stem_finds_the_first_shard(self, tmp_path):
        import ml_stack.hub as hub

        cache = tmp_path / "hub"
        snapshot = (cache / "models--maker--big-GGUF" / "snapshots" / "abc123"
                   / "UD-Q4_K_XL")
        snapshot.mkdir(parents=True)
        for shard in ("thing-00001-of-00002.gguf", "thing-00002-of-00002.gguf"):
            (snapshot / shard).write_bytes(b"x")

        found = hub.located("thing.gguf", cache=cache)
        assert found is not None and found.name == "thing-00001-of-00002.gguf"

    def test_a_name_that_matches_nothing_is_none_not_an_error(self, tmp_path):
        import ml_stack.hub as hub

        assert hub.located("nowhere.gguf", cache=tmp_path) is None
        assert hub.located("nowhere.gguf", cache=tmp_path / "does-not-exist") is None


class TestDraftNote:
    """The sentence a draft head's own README says about needing a fork -- unsloth's MTP
    heads for Qwen3.8-Flash-Next fail to load on mainline with a tensor error, and their
    README says why in one line."""

    def test_reads_the_sentence_naming_mainline(self, monkeypatch, tmp_path):
        import huggingface_hub
        import ml_stack.hub as hub

        hub._DRAFT_NOTES.clear()
        readme = tmp_path / "MTP-README.md"
        readme.write_text(
            "# MTP heads\n\nThese heads are trained alongside the model. "
            "These do not work on mainline ggml-org/llama.cpp yet. "
            "Use unslothai/llama.cpp instead.\n")

        seen: list[tuple[str, str]] = []

        def fake_download(repo, filename, **kw):
            seen.append((repo, filename))
            if filename == "MTP/README.md":
                return str(readme)
            raise OSError("no such file in this repo")

        monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
        note = hub.draft_note("maker/thing-GGUF")
        assert "mainline" in note.lower()
        assert seen == [("maker/thing-GGUF", "MTP/README.md")]

        # cached -- a second call does not fetch again
        assert hub.draft_note("maker/thing-GGUF") == note
        assert seen == [("maker/thing-GGUF", "MTP/README.md")]

    def test_falls_back_to_the_plain_readme_when_there_is_no_mtp_one(
            self, monkeypatch, tmp_path):
        import huggingface_hub
        import ml_stack.hub as hub

        hub._DRAFT_NOTES.clear()
        readme = tmp_path / "README.md"
        readme.write_text("This repository requires the unsloth fork of llama.cpp.\n")

        def fake_download(repo, filename, **kw):
            if filename == "README.md":
                return str(readme)
            raise OSError("no MTP readme here")

        monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
        note = hub.draft_note("maker/plain-GGUF")
        assert "requires" in note.lower()

    def test_a_card_with_nothing_to_say_about_either_returns_empty(
            self, monkeypatch, tmp_path):
        import huggingface_hub
        import ml_stack.hub as hub

        hub._DRAFT_NOTES.clear()
        readme = tmp_path / "README.md"
        readme.write_text("Just an ordinary model card, nothing special here.\n")

        def fake_download(repo, filename, **kw):
            if filename == "README.md":
                return str(readme)
            raise OSError("no MTP readme here")

        monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
        assert hub.draft_note("maker/quiet-GGUF") == ""


class TestChooseHead:
    """One resolver for `serve up`, the bench and the app, told which binary will serve.

    The bench's own `--serve-draft auto` chose Qwen3.8-Flash-Next's BF16 MTP head for
    mainline twice on 2026-09-01 and paid an 87G load each time to reach `tensor
    'output_hc_norm.weight' not found`. The gate lived in `draft_for(borrows=)` and only
    `serve up` used it. Now `choose_head` reads the build off the binary itself."""

    FORK_ONLY = "These do not work on mainline ggml-org/llama.cpp yet."

    SHELVES = {
        # a sharded model with three heads under MTP/, one of which borrows the target's
        # embeddings and is the publisher's own recommendation -- the Flash-Next shape
        "maker/flash-GGUF": [
            ("UD-IQ4_XS/flash-UD-IQ4_XS-00001-of-00002.gguf", 40_000_000_000),
            ("UD-IQ4_XS/flash-UD-IQ4_XS-00002-of-00002.gguf", 40_000_000_000),
            ("MTP/mtp-flash-shared-BF16.gguf", 5_000_000_000),
            ("MTP/mtp-flash-shared-Q8_0.gguf", 2_600_000_000),
            ("MTP/mtp-flash-Q8_0.gguf", 3_900_000_000),
        ],
        # a QAT repository with its head at the root, the gemma shape
        "maker/gem-GGUF": [
            ("gem-Q4_K_M.gguf", 4_000_000_000),
            ("mtp-gem.gguf", 40_000_000),
        ],
        "maker/bare-GGUF": [("bare-Q4_K_M.gguf", 4_000_000_000)],
    }

    def _hub(self, monkeypatch, tmp_path, *, notes: dict[str, str] | None = None):
        import huggingface_hub
        import ml_stack.hub as hub

        hub._DRAFT_NOTES.clear()
        monkeypatch.setattr(hub, "files", lambda repo, **kw: self.SHELVES.get(repo, []))
        notes = notes or {}

        def fake_download(repo, filename, **kw):
            if filename == "MTP/README.md" and repo in notes:
                readme = tmp_path / f"{repo.replace('/', '--')}-README.md"
                readme.write_text(notes[repo])
                return str(readme)
            raise OSError("no readme")

        monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
        return hub

    @staticmethod
    def _binary(where, manifest: dict | None = None):
        where.mkdir(parents=True, exist_ok=True)
        binary = where / "llama-server"
        binary.write_text("#!/bin/sh\nexit 0\n")
        binary.chmod(0o755)
        if manifest is not None:
            import json

            (where / "BUILD.json").write_text(json.dumps(manifest))
        return binary

    def test_mainline_is_refused_a_head_whose_readme_names_a_fork(self, monkeypatch, tmp_path):
        hub = self._hub(monkeypatch, tmp_path, notes={"maker/flash-GGUF": self.FORK_ONLY})
        mainline = self._binary(tmp_path / "current",
                                {"commit": "abc1234", "repo": "ggml-org/llama.cpp"})

        chosen = hub.choose_head("hf:maker/flash-GGUF/UD-IQ4_XS/flash-UD-IQ4_XS-00001-of-00002.gguf",
                                 binary=mainline)
        assert chosen.path == ""
        assert chosen.spec_type == ""
        assert chosen.borrows is False
        assert chosen.why.startswith("withheld: the repository's README says it needs a fork")
        assert chosen.note == self.FORK_ONLY

        # a brew bottle or anything on PATH has no manifest at all, and is mainline too
        bottle = self._binary(tmp_path / "bottle")
        assert hub.choose_head("hf:maker/flash-GGUF", binary=bottle).path == ""

    def test_a_named_fork_build_is_given_the_shared_q8_head(self, monkeypatch, tmp_path):
        import ml_stack.serve.binary as binary_module

        hub = self._hub(monkeypatch, tmp_path, notes={"maker/flash-GGUF": self.FORK_ONLY})
        named_root = tmp_path / "named"
        monkeypatch.setattr(binary_module, "MANAGED_NAMED", named_root)
        fork = self._binary(named_root / "forkname")

        chosen = hub.choose_head("hf:maker/flash-GGUF", binary=fork)
        assert chosen.path == "hf:maker/flash-GGUF/MTP/mtp-flash-shared-Q8_0.gguf"
        assert chosen.spec_type == "draft-mtp"
        assert chosen.borrows is True
        assert chosen.why == "shipped beside the weights"
        assert chosen.note == self.FORK_ONLY   # still told, so `up` can print it

    def test_a_manifest_naming_a_fork_borrows_wherever_it_lives(self, monkeypatch, tmp_path):
        """`find_binary` resolves the `named/` link into `builds/<name>-<commit>/`, so the
        path handed over is not under MANAGED_NAMED; the BUILD.json beside it says fork."""
        hub = self._hub(monkeypatch, tmp_path, notes={"maker/flash-GGUF": self.FORK_ONLY})
        fork = self._binary(tmp_path / "builds" / "forkname-abc1234",
                            {"commit": "abc1234", "repo": "someone/llama.cpp",
                             "name": "forkname"})

        chosen = hub.choose_head("hf:maker/flash-GGUF", binary=fork)
        assert chosen.borrows is True
        assert chosen.path.endswith("mtp-flash-shared-Q8_0.gguf")

    def test_mainline_avoids_a_shared_head_when_the_readme_does_not_warn(
            self, monkeypatch, tmp_path):
        """A head that borrows its target's embeddings is what mainline cannot load, so
        with no README to say so a mainline build still takes the head that carries its
        own, and a fork takes the recommended shared one."""
        hub = self._hub(monkeypatch, tmp_path)
        mainline = self._binary(tmp_path / "current")
        fork = self._binary(tmp_path / "fork", {"repo": "someone/llama.cpp"})

        assert hub.choose_head("hf:maker/flash-GGUF", binary=mainline).path == \
            "hf:maker/flash-GGUF/MTP/mtp-flash-Q8_0.gguf"
        assert hub.choose_head("hf:maker/flash-GGUF", binary=fork).path == \
            "hf:maker/flash-GGUF/MTP/mtp-flash-shared-Q8_0.gguf"
        # an explicit preference outranks either default
        assert hub.choose_head("hf:maker/flash-GGUF", binary=fork,
                               prefer=("shared-bf16",)).path.endswith("shared-BF16.gguf")

    def test_a_root_head_with_no_warning_is_chosen_on_mainline(self, monkeypatch, tmp_path):
        hub = self._hub(monkeypatch, tmp_path)
        mainline = self._binary(tmp_path / "current")

        chosen = hub.choose_head("hf:maker/gem-GGUF/gem-Q4_K_M.gguf", binary=mainline)
        assert chosen.path == "hf:maker/gem-GGUF/mtp-gem.gguf"
        assert chosen.spec_type == "draft-mtp"
        assert chosen.why == "shipped beside the weights"
        assert chosen.borrows is False
        assert chosen.note == ""

    def test_no_head_is_an_empty_path_and_says_so(self, monkeypatch, tmp_path):
        hub = self._hub(monkeypatch, tmp_path)
        mainline = self._binary(tmp_path / "current")

        chosen = hub.choose_head("hf:maker/bare-GGUF/bare-Q4_K_M.gguf", binary=mainline)
        assert chosen == hub.Chosen("", "", "no head shipped beside the weights", False)

        # a path from nowhere the Hub knows, with nothing beside it, is the same answer
        lone = tmp_path / "elsewhere" / "lone.gguf"
        lone.parent.mkdir()
        lone.write_bytes(b"GGUF")
        assert hub.choose_head(str(lone), binary=mainline).path == ""

    def test_a_bare_name_resolves_through_the_hub_cache(self, monkeypatch, tmp_path):
        """A filename copied out of `ml-stack-models files` names a file in the cache, whose
        directory names the repository -- so the listing, the README and the head are all
        found from nothing but the name."""
        hub = self._hub(monkeypatch, tmp_path, notes={"maker/flash-GGUF": self.FORK_ONLY})
        cache = tmp_path / "hub"
        snapshot = cache / "models--maker--flash-GGUF" / "snapshots" / "abc" / "UD-IQ4_XS"
        snapshot.mkdir(parents=True)
        (snapshot / "flash-UD-IQ4_XS-00001-of-00002.gguf").write_bytes(b"GGUF")
        monkeypatch.setattr(hub, "HUB_CACHE", cache)
        mainline = self._binary(tmp_path / "current")
        fork = self._binary(tmp_path / "fork", {"repo": "someone/llama.cpp"})

        assert hub.repo_of("flash-UD-IQ4_XS.gguf") == "maker/flash-GGUF"
        assert hub.repo_of(str(snapshot / "flash-UD-IQ4_XS-00001-of-00002.gguf")) == \
            "maker/flash-GGUF"
        withheld = hub.choose_head("flash-UD-IQ4_XS.gguf", binary=mainline)
        assert withheld.path == "" and withheld.why.startswith("withheld")
        given = hub.choose_head("flash-UD-IQ4_XS.gguf", binary=fork)
        assert given.path == "hf:maker/flash-GGUF/MTP/mtp-flash-shared-Q8_0.gguf"

    def test_a_listing_that_cannot_be_fetched_is_answered_from_the_disk(
            self, monkeypatch, tmp_path):
        """Offline, the head already downloaded beside the weights is still the head."""
        import huggingface_hub
        import ml_stack.hub as hub

        hub._DRAFT_NOTES.clear()

        def offline(*a, **k):
            raise OSError("no network")

        monkeypatch.setattr(hub, "files", offline)
        monkeypatch.setattr(huggingface_hub, "hf_hub_download", offline)
        snapshot = tmp_path / "hub" / "models--maker--gem-GGUF" / "snapshots" / "abc"
        snapshot.mkdir(parents=True)
        weights = snapshot / "gem-Q4_K_M.gguf"
        weights.write_bytes(b"GGUF")
        (snapshot / "mtp-gem.gguf").write_bytes(b"head")
        mainline = self._binary(tmp_path / "current")

        chosen = hub.choose_head(str(weights), binary=mainline)
        assert chosen.path == str(snapshot / "mtp-gem.gguf")
        assert chosen.spec_type == "draft-mtp"
        assert chosen.why == "found beside the weights on disk"

    def test_the_listing_says_what_each_build_here_would_serve(
            self, monkeypatch, tmp_path, capsys):
        """`ml-stack-models files` names the head, its warning, and the chooser's answer for
        the default build and every named build -- which `--build` a head needs, before a
        load, is the whole point of asking."""
        import ml_stack.serve.binary as binary_module

        hub = self._hub(monkeypatch, tmp_path, notes={"maker/flash-GGUF": self.FORK_ONLY})
        monkeypatch.setattr(hub, "room", lambda: 0)
        monkeypatch.setattr(hub, "held", lambda: {})
        current = self._binary(tmp_path / "current")
        named_root = tmp_path / "named"
        self._binary(named_root / "forkname")
        monkeypatch.setattr(binary_module, "MANAGED_NAMED", named_root)
        monkeypatch.setattr(binary_module, "find_binary", lambda *a, **k: current)

        assert hub.main(["files", "maker/flash-GGUF"]) == 0
        out = capsys.readouterr().out
        assert "draft head shipped with it: hf:maker/flash-GGUF/MTP/mtp-flash-shared-Q8_0.gguf" in out
        assert f"  {self.FORK_ONLY}" in out
        assert "this build (mainline): withheld" in out
        assert "--build forkname (fork): hf:maker/flash-GGUF/MTP/mtp-flash-shared-Q8_0.gguf" in out


class TestPrettyName:
    def test_the_quantisation_moves_into_brackets_and_the_rest_goes(self):
        from ml_stack.hub import pretty_name

        assert pretty_name("hf:maker/thing-GGUF/UD-IQ4_XS/thing-Flash-UD-IQ4_XS-00001-of-00003.gguf") \
            == "thing-Flash (IQ4_XS)"
        assert pretty_name("gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf") == "gemma-4-E4B-it-qat (Q4_K_XL)"
        assert pretty_name("gpt-oss-120b-mxfp4-00001-of-00003.gguf") == "gpt-oss-120b (mxfp4)"
        assert pretty_name("mtp-thing-Flash-shared-Q8_0.gguf") == "mtp-thing-Flash-shared (Q8_0)"
        assert pretty_name("thing-27B-UD-Q4_K_M.gguf") == "thing-27B (Q4_K_M)"
        assert pretty_name("plain-name.gguf") == "plain-name"
        assert pretty_name("") == ""

    def test_the_pages_carry_the_same_rule(self):
        """One rule, written twice: the pages cannot import Python, so the regex is copied
        and held to the Python one by the same answers under node when it is present."""
        import json
        import shutil
        import subprocess
        from pathlib import Path

        import ml_stack.fleet as fleet
        import ml_stack.graph as graph
        from ml_stack.hub import pretty_name

        pages = [Path(fleet.__file__).parent / "web" / "fit.html",
                 Path(graph.__file__).parent / "web" / "graph.html"]
        for page in pages:
            assert "(?:UD-)?((?:IQ|Q)\\d(?:_[A-Z0-9]+)+|mxfp4|BF16|F16|F32)" in page.read_text()
        node = shutil.which("node")
        if not node:
            pytest.skip("no node to run the page's copy")
        names = ["a/b/thing-Flash-UD-IQ4_XS-00001-of-00003.gguf", "gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf",
                 "gpt-oss-120b-mxfp4-00001-of-00003.gguf", "mtp-thing-shared-Q8_0.gguf", "plain.gguf"]
        text = pages[0].read_text()
        start = text.index("function prettyName")
        end = text.index("\n  }\n", start) + 4
        script = text[start:end] + "\nconsole.log(JSON.stringify(" + json.dumps(names) + ".map(prettyName)))"
        said = subprocess.run([node, "-e", script], capture_output=True, text=True, check=True).stdout
        assert json.loads(said) == [pretty_name(n) for n in names]


class TestShardsBeside:
    def test_every_shard_of_the_build_and_nothing_else(self, tmp_path):
        from ml_stack.hub import shards_beside

        for n in (1, 2, 3):
            (tmp_path / f"thing-Q4_K_M-0000{n}-of-00003.gguf").write_bytes(b"x" * n)
        (tmp_path / "other-Q4_K_M-00001-of-00002.gguf").write_bytes(b"y")
        (tmp_path / "thing-Q4_K_M.gguf").write_bytes(b"z")
        got = shards_beside(tmp_path / "thing-Q4_K_M-00001-of-00003.gguf")
        assert [p.name for p in got] == [f"thing-Q4_K_M-0000{n}-of-00003.gguf" for n in (1, 2, 3)]
        assert shards_beside(tmp_path / "thing-Q4_K_M.gguf") == [tmp_path / "thing-Q4_K_M.gguf"]


def test_an_iq_build_is_marked_as_the_slow_choice_on_a_mac_only():
    from ml_stack.hub import iq_on_metal

    assert iq_on_metal("thing-UD-IQ4_XS", platform="darwin")
    assert iq_on_metal("thing-IQ2_M.gguf", platform="darwin")
    assert not iq_on_metal("thing-UD-Q4_K_XL", platform="darwin")
    assert not iq_on_metal("thing-UD-IQ4_XS", platform="linux")
    assert not iq_on_metal("thing-UD-IQ4_XS", platform="win32")
