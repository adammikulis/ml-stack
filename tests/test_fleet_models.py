"""Model files, and getting one from the network before the internet."""

from __future__ import annotations

import http.server
import json
import os
import socket
import threading
from pathlib import Path

import pytest
from ml_stack.fleet.models import ModelError, Models, _resolve


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def store(tmp_path):
    d = tmp_path / "models"
    d.mkdir()
    return Models([d], d)


def a_model(folder, name="qwen3-4b-q4.gguf", mb=2):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    path.write_bytes(os.urandom(mb * 1024 * 1024))
    return path


class TestFinding:
    def test_it_lists_model_files_and_ignores_everything_else(self, store, tmp_path):
        a_model(tmp_path / "models")
        (tmp_path / "models" / "notes.txt").write_text("hello")
        assert [m.name for m in store.all()] == ["qwen3-4b-q4.gguf"]

    def test_a_tiny_file_is_not_a_model(self, store, tmp_path):
        (tmp_path / "models" / "stub.gguf").write_bytes(b"x" * 100)
        assert store.all() == []

    def test_it_matches_on_part_of_the_name(self, store, tmp_path):
        a_model(tmp_path / "models", "Qwen3-4B-Instruct-Q4_K_M.gguf")
        assert store.find("qwen3") is not None
        assert store.find("llama") is None

    def test_the_beacon_carries_names_and_sizes_only(self, store, tmp_path):
        a_model(tmp_path / "models")
        row = store.public()[0]
        assert set(row) == {"name", "size", "modified"}
        assert "path" not in row

    def test_the_digest_is_computed_once_per_file(self, store, tmp_path):
        a_model(tmp_path / "models")
        model = store.all()[0]
        assert store.digest(model) == store.digest(model)
        assert len(store.digest(model)) == 64


class TestSources:
    def test_a_hugging_face_reference_becomes_a_url(self):
        url = _resolve("hf:Qwen/Qwen3-4B-GGUF/qwen3-4b-q4.gguf")
        assert url.startswith("https://huggingface.co/Qwen/Qwen3-4B-GGUF/resolve/")
        assert url.endswith("qwen3-4b-q4.gguf?download=true")

    @pytest.mark.parametrize("bad", ["just-a-name", "hf:owner", "hf:owner/repo",
                                     "ftp://somewhere/x.gguf"])
    def test_something_that_is_not_a_source_is_refused(self, bad):
        with pytest.raises(ModelError):
            _resolve(bad)


class TestGetting:
    def test_a_model_already_here_is_not_fetched_again(self, store, tmp_path):
        a_model(tmp_path / "models")
        got = store.ensure("qwen3", source="http://127.0.0.1:1/never.gguf")
        assert got.name == "qwen3-4b-q4.gguf"

    def test_with_automatic_downloading_off_it_refuses(self, store):
        with pytest.raises(ModelError, match="automatic downloading is off"):
            store.ensure("absent.gguf", autodownload=False)

    def test_with_nobody_holding_it_and_no_source_it_says_so(self, store):
        with pytest.raises(ModelError, match="no machine on this network"):
            store.ensure("absent.gguf")

    def test_it_downloads_when_no_machine_has_it(self, store, tmp_path):
        payload = os.urandom(2 * 1024 * 1024)

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        srv = http.server.ThreadingHTTPServer(("127.0.0.1", free_port()), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            got = store.ensure(
                "far.gguf",
                source=f"http://127.0.0.1:{srv.server_address[1]}/far.gguf")
        finally:
            srv.shutdown()

        assert got.path.read_bytes() == payload
        assert got.path.parent == store.store

    def test_a_short_download_is_left_to_resume_from(self, store):
        payload = os.urandom(1024 * 1024)

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                self.send_response(200)
                # Claims more than it sends.
                self.send_header("Content-Length", str(len(payload) * 2))
                self.end_headers()
                self.wfile.write(payload)

        srv = http.server.ThreadingHTTPServer(("127.0.0.1", free_port()), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            with pytest.raises(ModelError, match="resume"):
                store.ensure("short.gguf",
                             source=f"http://127.0.0.1:{srv.server_address[1]}/s.gguf")
        finally:
            srv.shutdown()
        assert not (store.store / "short.gguf").exists()
        assert (store.store / "short.gguf.part").exists()


class TestResuming:
    """A .part on disk is a claim about a file. Each test breaks that claim."""

    def serve(self, handler):
        srv = http.server.ThreadingHTTPServer(("127.0.0.1", free_port()), handler)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        return srv

    def test_a_server_that_ignores_range_does_not_get_spliced_onto_the_part(
            self, store):
        payload = os.urandom(512 * 1024)
        part = store.store / "m.gguf.part"
        part.write_bytes(b"\xff" * 4096)

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                # Answers 200 with the whole file even though Range was asked for.
                self.send_response(200)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        srv = self.serve(H)
        try:
            got = store.ensure(
                "m.gguf", source=f"http://127.0.0.1:{srv.server_address[1]}/m.gguf")
        finally:
            srv.shutdown()

        assert got.path.read_bytes() == payload
        assert got.size == len(payload)
        assert not part.exists()

    def test_a_part_left_by_a_different_file_is_discarded(self, store):
        payload = os.urandom(256 * 1024)
        part = store.store / "m.gguf.part"
        part.write_bytes(b"\x00" * 8192)
        Path(str(part) + ".from").write_text(json.dumps(
            {"url": "http://elsewhere.invalid/other.gguf", "validator": '"x"'}))
        seen = []

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                seen.append(self.headers.get("Range"))
                self.send_response(200)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        srv = self.serve(H)
        try:
            got = store.ensure(
                "m.gguf", source=f"http://127.0.0.1:{srv.server_address[1]}/m.gguf")
        finally:
            srv.shutdown()

        assert seen == [None], "asked to resume a part belonging to another file"
        assert got.path.read_bytes() == payload

    def test_a_part_longer_than_the_file_is_refused_not_promoted(self, store):
        part = store.store / "m.gguf.part"
        part.write_bytes(b"\x01" * (64 * 1024))

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                self.send_response(416)
                self.send_header("Content-Range", "bytes */1024")
                self.send_header("Content-Length", "0")
                self.end_headers()

        srv = self.serve(H)
        try:
            with pytest.raises(ModelError, match="discarded"):
                store.ensure(
                    "m.gguf",
                    source=f"http://127.0.0.1:{srv.server_address[1]}/m.gguf")
        finally:
            srv.shutdown()

        assert not (store.store / "m.gguf").exists()
        assert not part.exists()

    def test_a_genuine_resume_asks_for_the_rest_and_keeps_what_it_had(self, store):
        head, tail = b"A" * 4096, b"B" * 4096
        part = store.store / "m.gguf.part"
        part.write_bytes(head)

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                assert self.headers.get("Range") == "bytes=4096-"
                self.send_response(206)
                self.send_header("Content-Range", "bytes 4096-8191/8192")
                self.send_header("Content-Length", str(len(tail)))
                self.end_headers()
                self.wfile.write(tail)

        srv = self.serve(H)
        try:
            got = store.ensure(
                "m.gguf", source=f"http://127.0.0.1:{srv.server_address[1]}/m.gguf")
        finally:
            srv.shutdown()

        assert got.path.read_bytes() == head + tail
        assert not part.exists()


class TestChoosingAQuant:
    """Naming a repository and no file should land on the Q4 build."""

    @pytest.mark.parametrize("names,want", [
        (["m-Q8_0.gguf", "m-Q4_K_M.gguf", "m-Q5_K_M.gguf"], "m-Q4_K_M.gguf"),
        (["m-Q8_0.gguf", "m-Q5_K_M.gguf"], "m-Q5_K_M.gguf"),
        (["m-Q4_K_S.gguf", "m-Q8_0.gguf"], "m-Q4_K_S.gguf"),
        (["m-Q4_0.gguf", "m-Q8_0.gguf"], "m-Q4_0.gguf"),
        (["only-f16.gguf"], "only-f16.gguf"),
    ])
    def test_it_prefers_q4(self, names, want, monkeypatch):
        from ml_stack.fleet import models as mod

        monkeypatch.setattr(mod, "_read_repo_files", lambda o, r: names)
        assert mod._quant_in("o", "r") == want

    def test_a_sharded_model_is_not_offered_as_one_file(self, monkeypatch):
        from ml_stack.fleet import models as mod

        monkeypatch.setattr(mod, "_read_repo_files", lambda o, r: [
            "big-Q4_K_M-00001-of-00003.gguf", "big-Q4_K_M-00002-of-00003.gguf",
            "small-Q8_0.gguf"])
        assert mod._quant_in("o", "r") == "small-Q8_0.gguf"

    def test_a_repository_with_no_gguf_says_so(self, monkeypatch):
        from ml_stack.fleet import models as mod

        monkeypatch.setattr(mod, "_read_repo_files", lambda o, r: ["README.md"])
        with pytest.raises(ModelError, match="no single-file gguf"):
            mod._quant_in("o", "r")

    def test_naming_a_file_still_takes_that_file(self, monkeypatch):
        from ml_stack.fleet import models as mod

        def refuse(*a):
            raise AssertionError("went to the network for a reference that named a file")

        monkeypatch.setattr(mod, "_read_repo_files", refuse)
        url = mod._resolve("hf:owner/repo/exact-Q8_0.gguf")
        assert url.endswith("exact-Q8_0.gguf?download=true")


class TestReadingWhatAModelIs:
    @pytest.mark.parametrize("name,total,active", [
        ("Qwen3-Coder-30B-A3B-Instruct", 30.0, 3.0),
        ("Ornith-1.5-35B-A3B", 35.0, 3.0),
        ("Ornith-1.0-9B", 9.0, 0.0),
        ("LFM2.5-2.6B", 2.6, 0.0),
        ("something-with-no-size", 0.0, 0.0),
    ])
    def test_it_reads_the_sizes_out_of_a_name(self, name, total, active):
        from ml_stack.fleet.models import _params_in

        assert _params_in(name) == (total, active)

    @pytest.mark.parametrize("bad", [
        "Bonsai-27B-mmproj-BF16.gguf", "DeepSeek-V4-Flash-MTP-Q4K.gguf",
        "model-draft-Q4_K_M.gguf", "imatrix_unsloth.gguf", "adapter-Q4.gguf",
    ])
    def test_a_file_that_sits_beside_a_model_is_not_the_model(self, bad):
        from ml_stack.fleet.models import _is_beside

        assert _is_beside(bad) is True

    @pytest.mark.parametrize("good", [
        "Qwen3-8B-Q4_K_M.gguf", "Bonsai-27B-dspark-Q4_1.gguf",
    ])
    def test_a_real_build_is_not_mistaken_for_an_accessory(self, good):
        from ml_stack.fleet.models import _is_beside

        assert _is_beside(good) is False

    def test_a_vision_projector_means_it_reads_pictures(self):
        from ml_stack.fleet.models import Suggestion

        seeing = Suggestion("m", "hf:o/r/m.gguf", 1.0, "", takes=("text", "image"))
        assert seeing.public()["takes"] == ["\U0001f4ac", "\U0001f5bc"]

    def test_modalities_come_from_how_the_hub_files_it(self):
        from ml_stack.fleet.models import _modalities

        assert _modalities({"pipeline_tag": "text-generation"}) == (
            ("text",), ("text",))
        takes, gives = _modalities({"pipeline_tag": "image-text-to-text"})
        assert "image" in takes and gives == ("text",)
        takes, _ = _modalities({"tags": ["audio-text-to-text"]})
        assert "audio" in takes

    @pytest.mark.parametrize("name,family", [
        ("Qwen3-Coder-30B-A3B-Instruct", "Qwen"),
        ("Ornith-1.5-9B", "Ornith"),
        ("deepseek-v4", "DeepSeek"),
        ("gpt-oss-20b", "GPT-OSS"),
        ("LFM2.5-2.6B", "LFM"),
        ("Bonsai-27B-gguf", "Bonsai"),
    ])
    def test_the_family_is_read_off_the_name(self, name, family):
        from ml_stack.fleet.models import family_of

        assert family_of(name) == family

    def test_a_model_named_after_no_family_goes_under_the_one_it_came_from(self):
        """Gemmable 4 12B is Gemma 4 12B fine-tuned, and the hub says nothing about
        that: the repository carries no base_model tag."""
        from ml_stack.fleet.models import family_of

        assert family_of("Gemmable-4-12B-MTP") == "Gemma"
        assert family_of("Mia-AiLab/Gemmable-4-12B-MTP-GGUF") == "Gemma"

    def test_a_name_that_merely_starts_the_same_is_not_folded_in(self):
        from ml_stack.fleet.models import family_of

        assert family_of("Gemstone-7B") == "Gemstone"
        assert family_of("Llamafile-3B") == "Llamafile"

    def test_a_family_nobody_has_heard_of_still_groups(self):
        from ml_stack.fleet.models import family_of

        assert family_of("Zarquon-9000-70B-Instruct") == "Zarquon"
        assert family_of("owner/Zarquon-9000-70B-GGUF") == "Zarquon"


class TestDraftModels:
    """A small model kept beside a big one, for guessing ahead."""

    def test_a_draft_is_kept_beside_its_model_and_not_listed_as_one(self, store,
                                                                   tmp_path):
        from ml_stack.fleet.models import draft_beside

        big = a_model(tmp_path / "models", name="big.gguf", mb=2)
        draft = big.with_suffix(".draft.gguf")
        draft.write_bytes(os.urandom(2 * 1024 * 1024))

        assert [m.name for m in store.all()] == ["big.gguf"], "the draft was listed"
        assert draft_beside(big) == draft

    def test_with_no_draft_there_is_nothing_beside_it(self, store, tmp_path):
        from ml_stack.fleet.models import draft_beside

        big = a_model(tmp_path / "models", name="lonely.gguf", mb=2)
        assert draft_beside(big) is None

    def test_getting_a_draft_puts_it_next_to_the_model(self, store, tmp_path):
        payload = os.urandom(2 * 1024 * 1024)

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        srv = http.server.ThreadingHTTPServer(("127.0.0.1", free_port()), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        big = a_model(tmp_path / "models", name="pair.gguf", mb=2)
        model = store.find("pair.gguf")
        try:
            got = store.ensure_draft(
                model, f"http://127.0.0.1:{srv.server_address[1]}/d.gguf")
        finally:
            srv.shutdown()

        assert got == big.with_suffix(".draft.gguf")
        assert got.read_bytes() == payload
        assert [m.name for m in store.all()] == ["pair.gguf"]

    def test_a_draft_is_copied_from_a_machine_that_has_it(self, store, tmp_path,
                                                          monkeypatch):
        """The internet is only for what nobody nearby holds."""
        from http.server import ThreadingHTTPServer

        from ml_stack.fleet.daemon import JobRunner, load_or_create_token, make_handler

        theirs = tmp_path / "theirs"
        payload = a_model(theirs, name="pair.draft.gguf", mb=2).read_bytes()
        a_model(theirs, name="pair.gguf", mb=2)
        root = tmp_path / "traind"
        (root / "files").mkdir(parents=True)
        key = b"a-cluster-key-they-both-know"
        token = load_or_create_token(root, key)
        runner = JobRunner(root)
        port = free_port()
        httpd = ThreadingHTTPServer(
            ("127.0.0.1", port),
            make_handler(runner, root / "files", token,
                         models=Models([theirs], theirs),
                         cluster_key_path=tmp_path / "their.key"))
        threading.Thread(target=httpd.serve_forever, daemon=True).start()

        asked = []
        monkeypatch.setattr(
            Models, "where",
            lambda self, name, k, **kw: (asked.append(name),
                                         [("theirs", f"http://127.0.0.1:{port}", 0)])[1])

        big = a_model(tmp_path / "models", name="pair.gguf", mb=2)
        model = store.find("pair.gguf")
        try:
            got = store.ensure_draft(model, "http://127.0.0.1:1/none.gguf", key=key)
        finally:
            runner.shutdown()
            httpd.shutdown()
            httpd.server_close()

        assert got == big.with_suffix(".draft.gguf")
        assert got.read_bytes() == payload
        assert asked == ["pair.gguf"], "the machines holding the model are the ones asked"

    def test_a_name_pointing_out_of_the_store_finds_nothing(self, store, tmp_path):
        (tmp_path / "secret.draft.gguf").write_bytes(b"x" * 4096)
        a_model(tmp_path / "models", name="here.draft.gguf", mb=2)

        assert store.find_draft("../secret.draft.gguf") is None
        assert store.find_draft("here.draft.gguf") is not None
        assert store.find_draft("here.gguf") is None

    def test_a_draft_already_there_is_not_fetched_again(self, store, tmp_path):
        big = a_model(tmp_path / "models", name="again.gguf", mb=2)
        draft = big.with_suffix(".draft.gguf")
        draft.write_bytes(b"x" * 2048)
        model = store.find("again.gguf")
        # An unreachable source: it must not be asked for.
        assert store.ensure_draft(model, "http://127.0.0.1:1/none.gguf") == draft

    def test_a_repository_that_ships_a_draft_offers_it(self):
        from ml_stack.fleet.models import Suggestion

        pick = Suggestion("m", "hf:o/r/m.gguf", 5.0, "", draft_ref="hf:o/r/m-draft.gguf",
                          draft_gb=0.5)
        assert pick.public()["draft_ref"].endswith("m-draft.gguf")
        assert pick.public()["draft_gb"] == 0.5


class TestUncensoredBuilds:
    @pytest.mark.parametrize("name", [
        "Qwen3.8-27B-Uncensored", "Huihui-DeepSeek-V4-abliterated",
        "Qwen3.8-27B-Heretic-Abliterated-Uncensored", "Model-OBLITERATED",
        "something-nsfw-7B",
    ])
    def test_a_build_with_its_refusals_removed_is_marked(self, name):
        from ml_stack.fleet.models import is_unfiltered

        assert is_unfiltered(name) is True

    @pytest.mark.parametrize("name", [
        "Qwen3-Coder-30B-A3B-Instruct", "Ornith-1.5-9B", "gpt-oss-20b",
    ])
    def test_an_ordinary_build_is_not(self, name):
        from ml_stack.fleet.models import is_unfiltered

        assert is_unfiltered(name) is False

    def test_the_flag_reaches_the_screen(self):
        from ml_stack.fleet.models import Suggestion

        assert Suggestion("Qwen3-Uncensored", "hf:o/r/m.gguf", 1.0, "").public()[
            "unfiltered"] is True
        assert Suggestion("Qwen3-4B", "hf:o/r/m.gguf", 1.0, "").public()[
            "unfiltered"] is False


class TestPagingAndSearching:
    def rows(self, n, unfiltered_every=0):
        from ml_stack.fleet.models import Suggestion

        out = []
        for i in range(n):
            rude = unfiltered_every and i % unfiltered_every == 0
            out.append(Suggestion(
                name=f"{'Rude' if rude else 'Plain'}-{i}-7B",
                ref=f"hf:o/r{i}/m.gguf", gb=1.0 + i, what="", family="Fam",
                unfiltered=rude))
        return out

    def test_a_page_is_cut_after_filtering_not_before(self, monkeypatch):
        """Filtering the page would leave it half empty."""
        from ml_stack.fleet import models as mod

        monkeypatch.setattr(mod, "_popular", (mod.time.time(), self.rows(40, 2)))
        page = mod.popular(free_gb=999, ram_gb=999, limit=10, page=0, rude=False)
        assert len(page) == 10, f"got {len(page)} of 10"
        assert all(not p.public()["unfiltered"] for p in page)

    def test_pages_do_not_repeat_or_skip(self, monkeypatch):
        from ml_stack.fleet import models as mod

        monkeypatch.setattr(mod, "_popular", (mod.time.time(), self.rows(25)))
        seen = []
        for page in range(3):
            seen += [p.name for p in mod.popular(999, 999, limit=10, page=page)]
        assert len(seen) == 25
        assert len(set(seen)) == 25

    def test_showing_uncensored_builds_adds_them_back(self, monkeypatch):
        from ml_stack.fleet import models as mod

        monkeypatch.setattr(mod, "_popular", (mod.time.time(), self.rows(20, 2)))
        assert mod.how_many(999, 999, rude=False) == 10
        assert mod.how_many(999, 999, rude=True) == 20

    def test_families_cover_the_whole_list_not_one_page(self, monkeypatch):
        """A family further down still needs a box on the first page."""
        from ml_stack.fleet import models as mod
        from ml_stack.fleet.models import Suggestion

        rows = [Suggestion(f"Qwen3-{i}B", "hf:o/r/m.gguf", float(i), "", family="Qwen")
                for i in range(1, 20)]
        rows.append(Suggestion("gemma-4-2B", "hf:o/r/g.gguf", 40.0, "", family="Gemma"))
        monkeypatch.setattr(mod, "_popular", (mod.time.time(), rows))

        first = mod.popular(999, 999, limit=5, page=0)
        assert "Gemma" not in {p.family for p in first}
        assert mod.families(999, 999) == ["Gemma", "Qwen"]

    def test_a_search_asks_the_hub_and_is_paged_the_same_way(self, monkeypatch):
        from ml_stack.fleet import models as mod

        asked = []

        def fake_hub(url, timeout=25.0):
            asked.append(url)
            return [{"id": f"owner/thing-{i}-GGUF"} for i in range(6)]

        monkeypatch.setattr(mod, "_hub", fake_hub)
        monkeypatch.setattr(mod, "_resolve_rows",
                            lambda rows: self.rows(len(rows)))
        mod._found.clear()
        got = mod.popular(999, 999, limit=4, page=0, query="thing")
        assert len(got) == 4
        assert any("search=thing" in u for u in asked), asked
        assert mod.searched_count("thing", 999, 999) == 6

    def test_an_empty_search_goes_back_to_the_popular_list(self, monkeypatch):
        from ml_stack.fleet import models as mod

        monkeypatch.setattr(mod, "_popular", (mod.time.time(), self.rows(5)))
        assert len(mod.popular(999, 999, limit=10, query="   ")) == 5


class TestSuggestions:
    def test_every_one_is_a_reference_this_code_can_resolve(self):
        from ml_stack.fleet.models import SUGGESTED, _resolve

        assert SUGGESTED, "nothing offered at all would pass every check below"
        for pick in SUGGESTED:
            assert pick.ref.startswith("hf:"), pick.ref
            url = _resolve(pick.ref)
            assert url.startswith("https://huggingface.co/"), url
            assert pick.file.endswith(".gguf"), pick.file
            assert pick.gb > 0 and pick.what and pick.name

    def test_names_and_files_do_not_repeat(self):
        from ml_stack.fleet.models import SUGGESTED

        assert SUGGESTED, "an empty list has no repeats either"
        assert len({p.name for p in SUGGESTED}) == len(SUGGESTED)
        assert len({p.file for p in SUGGESTED}) == len(SUGGESTED)

    def test_what_will_not_fit_is_not_offered(self):
        from ml_stack.fleet.models import suggestions

        small = suggestions(free_gb=3.0, ram_gb=64.0)
        assert small, "nothing offered on a machine with 3 GB free"
        assert all(p.gb <= 3.0 for p in small)
        assert not suggestions(free_gb=0.001, ram_gb=64.0)

    def test_a_machine_short_of_memory_is_not_offered_a_big_one(self):
        from ml_stack.fleet.models import suggestions

        assert all(p.gb <= 4.0 for p in suggestions(free_gb=999.0, ram_gb=4.0))

    def test_the_smallest_comes_first(self):
        from ml_stack.fleet.models import suggestions

        got = suggestions(free_gb=999.0, ram_gb=999.0)
        assert [p.gb for p in got] == sorted(p.gb for p in got)

    def test_with_no_limits_known_everything_is_offered(self):
        from ml_stack.fleet.models import SUGGESTED, suggestions

        assert len(suggestions()) == len(SUGGESTED)


class TestProgress:
    def test_the_two_kinds_of_progress_do_not_share_a_callback(self, store):
        """ensure() reports a stage as text and bytes as two numbers. One callback
        taking both would be called with a string and then with a pair."""
        payload = os.urandom(512 * 1024)

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        srv = http.server.ThreadingHTTPServer(("127.0.0.1", free_port()), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()

        notes, seen = [], []
        try:
            store.ensure("m.gguf",
                         source=f"http://127.0.0.1:{srv.server_address[1]}/m.gguf",
                         on_note=notes.append,
                         on_progress=lambda done, total: seen.append((done, total)))
        finally:
            srv.shutdown()

        assert notes == ["Downloading m.gguf"]
        assert seen, "no byte counts arrived"
        assert all(isinstance(d, int) and isinstance(t, int) for d, t in seen)
        assert seen[-1][0] == len(payload)
        assert seen[-1][1] == len(payload)


class TestGettingInTheBackground:
    def serve(self, payload, delay=0.0):
        import time as clock

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                for i in range(0, len(payload), 65536):
                    self.wfile.write(payload[i:i + 65536])
                    self.wfile.flush()
                    if delay:
                        clock.sleep(delay)

        srv = http.server.ThreadingHTTPServer(("127.0.0.1", free_port()), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        return srv

    def waited(self, downloads, gid, want, timeout=20.0):
        import time as clock

        until = clock.monotonic() + timeout
        while clock.monotonic() < until:
            row = next(g for g in downloads.active() if g.id == gid)
            if row.state == want:
                return row
            clock.sleep(0.05)
        raise AssertionError(f"still {row.state} after {timeout}s: {row.error}")

    def test_a_download_runs_without_holding_the_caller(self, store):
        from ml_stack.fleet.models import Downloads

        payload = os.urandom(1024 * 1024)
        srv = self.serve(payload)
        downloads = Downloads(store)
        try:
            started = downloads.start(
                "big.gguf",
                source=f"http://127.0.0.1:{srv.server_address[1]}/big.gguf")
            assert started.state == "getting"
            done = self.waited(downloads, started.id, "done")
        finally:
            srv.shutdown()

        assert done.total == len(payload)
        assert done.done == len(payload)
        assert (store.store / "big.gguf").read_bytes() == payload

    def test_how_far_along_it_is_can_be_read_while_it_runs(self, store):
        """Bigger than one CHUNK, or the whole body arrives in a single read and
        the only counts ever seen are nothing and everything."""
        from ml_stack.fleet.models import CHUNK, Downloads

        payload = os.urandom(4 * CHUNK)
        srv = self.serve(payload, delay=0.01)
        downloads = Downloads(store)
        try:
            started = downloads.start(
                "slow.gguf",
                source=f"http://127.0.0.1:{srv.server_address[1]}/slow.gguf")
            import time as clock
            seen = []
            for _ in range(200):
                row = next(g for g in downloads.active() if g.id == started.id)
                seen.append(row.done)
                if row.state != "getting":
                    break
                clock.sleep(0.05)
            self.waited(downloads, started.id, "done")
        finally:
            srv.shutdown()

        assert any(0 < d < len(payload) for d in seen), (
            f"only ever saw {sorted(set(seen))}")

    def test_a_failure_is_reported_rather_than_raised_into_nowhere(self, store):
        from ml_stack.fleet.models import Downloads

        downloads = Downloads(store)
        started = downloads.start("nope.gguf", source="http://127.0.0.1:1/nope.gguf")
        row = self.waited(downloads, started.id, "failed")
        assert row.error
        assert not (store.store / "nope.gguf").exists()

    def test_asking_twice_for_the_same_model_does_not_start_it_twice(self, store):
        from ml_stack.fleet.models import Downloads

        payload = os.urandom(512 * 1024)
        srv = self.serve(payload, delay=0.05)
        downloads = Downloads(store)
        try:
            source = f"http://127.0.0.1:{srv.server_address[1]}/twice.gguf"
            first = downloads.start("twice.gguf", source=source)
            again = downloads.start("twice.gguf", source=source)
            assert again.id == first.id
            self.waited(downloads, first.id, "done")
        finally:
            srv.shutdown()
        assert len([g for g in downloads.active() if g.name == "twice.gguf"]) == 1


class TestUnfinishedDownloads:
    def test_a_part_still_being_written_is_not_offered_for_discard(self, store):
        (store.store / "busy.gguf.part").write_bytes(b"x" * 1024)
        assert store.unfinished() == []

    def test_one_nothing_has_touched_for_an_hour_is(self, store):
        import os
        import time

        part = store.store / "stopped.gguf.part"
        part.write_bytes(b"x" * 2048)
        old = time.time() - 7200
        os.utime(part, (old, old))

        found = store.unfinished()
        assert [r["name"] for r in found] == ["stopped.gguf.part"]
        assert found[0]["size"] == 2048

    def test_discarding_takes_the_part_and_what_it_recorded(self, store):
        part = store.store / "gone.gguf.part"
        part.write_bytes(b"x" * 16)
        stamp = Path(str(part) + ".from")
        stamp.write_text(json.dumps({"url": "http://x/y.gguf", "validator": "t"}))

        assert store.discard("gone.gguf.part") == ["gone.gguf.part"]
        assert not part.exists()
        assert not stamp.exists()

    def test_discarding_leaves_finished_models_alone(self, store, tmp_path):
        a_model(tmp_path / "models", name="keep.gguf")
        (store.store / "drop.gguf.part").write_bytes(b"x")
        store.discard("drop.gguf.part")
        assert [m.name for m in store.all()] == ["keep.gguf"]

    @pytest.mark.parametrize("bad", ["../keep.gguf", "keep.gguf", "a/b.part"])
    def test_discard_reaches_nothing_outside_the_store(self, store, tmp_path, bad):
        a_model(tmp_path / "models", name="keep.gguf")
        outside = store.store.parent / "keep.gguf"
        outside.write_bytes(b"important")
        assert store.discard(bad) == []
        assert outside.exists()
        assert (store.store / "keep.gguf").exists()


class TestRemoving:
    def test_only_models_this_machine_downloaded_can_be_removed(self, tmp_path):
        """A model in someone's own folder is theirs, not this program's to delete."""
        elsewhere = tmp_path / "theirs"
        a_model(elsewhere, "theirs.gguf")
        store = Models([elsewhere, tmp_path / "ours"], tmp_path / "ours")

        assert store.find("theirs") is not None
        assert store.remove("theirs") is False
        assert (elsewhere / "theirs.gguf").exists()

    def test_one_it_downloaded_can_be(self, store, tmp_path):
        a_model(tmp_path / "models")
        assert store.remove("qwen3") is True
        assert store.all() == []


class TestOverHTTP:
    """The routes a peer uses: GET /models and POST /models/get."""

    @pytest.fixture
    def served(self, tmp_path):
        from http.server import ThreadingHTTPServer

        from ml_stack.fleet.daemon import JobRunner, load_or_create_token, make_handler
        from ml_stack.fleet.remote import Peer

        root = tmp_path / "traind"
        files = root / "files"
        files.mkdir(parents=True)
        theirs = tmp_path / "theirs"
        a_model(theirs)
        token = load_or_create_token(root)
        runner = JobRunner(root)
        port = free_port()
        httpd = ThreadingHTTPServer(
            ("127.0.0.1", port),
            make_handler(runner, files, token,
                         models=Models([theirs], theirs),
                         cluster_key_path=tmp_path / "cluster.key"))
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        try:
            yield Peer(f"http://127.0.0.1:{port}", token), theirs
        finally:
            runner.shutdown()
            httpd.shutdown()
            httpd.server_close()

    def test_a_peer_lists_the_models_a_machine_holds(self, served):
        peer, _ = served
        rows = peer.models()
        assert [r["name"] for r in rows] == ["qwen3-4b-q4.gguf"]
        assert "path" not in rows[0]

    def test_a_peer_asks_a_machine_for_a_model_it_already_has(self, served):
        peer, theirs = served
        got = peer.get_model("qwen3")
        assert got["name"] == "qwen3-4b-q4.gguf"
        assert got["size"] == (theirs / "qwen3-4b-q4.gguf").stat().st_size

    def test_asking_for_one_nobody_has_is_refused_not_crashed(self, served):
        from ml_stack.fleet.remote import PeerError

        peer, _ = served
        with pytest.raises(PeerError):
            peer.get_model("nothing-like-this.gguf")


class TestCaches:
    def test_each_existing_root_is_listed_with_its_weight_files_and_bytes(self, tmp_path,
                                                                        monkeypatch):
        from ml_stack.fleet import models as models_module
        from ml_stack.fleet.models import caches, holding, sized

        hub = tmp_path / "hf" / "hub"
        blobs = hub / "models--maker--big-GGUF" / "blobs"
        snapshot = hub / "models--maker--big-GGUF" / "snapshots" / "abc"
        blobs.mkdir(parents=True)
        snapshot.mkdir(parents=True)
        (blobs / ("aa" * 8)).write_bytes(b"x" * 3000)
        (snapshot / "big-Q4_K_M.gguf").symlink_to(blobs / ("aa" * 8))
        (snapshot / "model.safetensors").write_bytes(b"y" * 1000)
        (snapshot / "README.md").write_text("words")
        mine = tmp_path / "models"
        a_model(mine, "small.gguf", mb=1)
        monkeypatch.setattr(models_module, "default_roots",
                            lambda root: [tmp_path / "absent", hub, mine])

        assert holding(hub) == (2, 4000), "the symlink reads through to its blob"
        assert holding(tmp_path / "absent") == (0, 0)
        assert caches(tmp_path) == [(hub, 2, 4000), (mine, 1, 1024 * 1024)]
        assert sized(4000) == "0M" and sized(1024 * 1024) == "1M"
        assert sized(int(86.2 * 2**30)) == "86.2G"

    def test_hf_home_names_the_hub_cache(self, tmp_path, monkeypatch):
        from ml_stack.fleet.models import default_roots

        monkeypatch.setenv("HF_HOME", str(tmp_path / "elsewhere"))
        assert tmp_path / "elsewhere" / "hub" in default_roots(tmp_path)
        assert Path.home() / ".cache" / "huggingface" / "hub" not in default_roots(tmp_path)
        monkeypatch.delenv("HF_HOME")
        assert Path.home() / ".cache" / "huggingface" / "hub" in default_roots(tmp_path)

