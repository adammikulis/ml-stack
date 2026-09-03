"""Putting ml-stack on a machine: what starts it at boot, and what it does about the
models already on the disk.

None of this needs root -- `system_service` only *generates* what root would write, so
every platform's shape is checked on whatever machine happens to be running the suite.
The cache tests build a Hub cache in ``tmp_path``; nothing reads a real one. Every name is
invented.
"""

from __future__ import annotations

import plistlib
from pathlib import Path

from ml_stack.fleet import autostart
from ml_stack.fleet.autostart import (
    ADOPTED,
    IN_PLACE,
    LEFT_ALONE,
    SYSTEM_LABEL,
    choose_model,
    models_in,
    plan_cache,
    system_service,
)

USER = "wrenfield"


def _cache(root: Path, **models: int) -> Path:
    """A Hub cache holding ``models``, each of the given size in bytes."""
    hub = root / "hub"
    hub.mkdir(parents=True)
    for name, size in models.items():
        blobs = hub / f"models--{name}" / "blobs"
        blobs.mkdir(parents=True)
        (blobs / "weights.gguf").write_bytes(b"g" * size)
    return root


class TestStartingAtBoot:
    def test_macos_gets_a_launchdaemon_that_runs_as_the_person_who_installed_it(self):
        """UserName is the whole trick: the service's home is theirs, so the models in
        ~/.cache/huggingface are the models it serves, and nothing is downloaded twice."""
        made = system_service(USER, "/Users/wrenfield", argv=["/opt/ml-stack/bin/traind"],
                              platform="darwin")
        plist = plistlib.loads(made.body.encode())

        assert plist["UserName"] == USER
        assert plist["RunAtLoad"] is True, "it would not start until somebody logged in"
        assert plist["KeepAlive"] == {"SuccessfulExit": False}
        assert plist["EnvironmentVariables"]["HF_HOME"] == \
            "/Users/wrenfield/.cache/huggingface"
        assert made.path == f"/Library/LaunchDaemons/{SYSTEM_LABEL}.plist"

    def test_linux_gets_a_system_unit_with_that_user_and_the_same_cache(self):
        made = system_service(USER, "/home/wrenfield", argv=["/opt/ml-stack/bin/traind"],
                              platform="linux")

        assert f"User={USER}" in made.body
        assert "Environment=HF_HOME=/home/wrenfield/.cache/huggingface" in made.body
        assert "WantedBy=multi-user.target" in made.body, "it would not start at boot"
        assert "Restart=on-failure" in made.body

    def test_windows_gets_a_startup_task_as_that_user_not_as_system(self):
        """SYSTEM has its own profile, so a task running as SYSTEM would have an empty
        model cache and download everything again."""
        made = system_service(USER, r"C:\Users\wrenfield", argv=[r"C:\opt\traind.exe"],
                              platform="win32")

        assert "/SC ONSTART" in made.body, "it would not survive a reboot"
        assert f'/RU "{USER}"' in made.body
        assert "SYSTEM" not in made.body

    def test_the_definition_names_the_one_cache_on_the_machine(self):
        env = autostart.service_environment("/home/wrenfield")
        assert env["HF_HOME"].endswith("/.cache/huggingface")
        assert env["HOME"] == "/home/wrenfield"


class TestTheModelsAlreadyHere:
    def test_the_same_user_reads_their_cache_where_it_is(self, tmp_path):
        mine = _cache(tmp_path / "home" / ".cache" / "huggingface",
                      **{"wrenfield--quince-2b": 2048})
        shared = tmp_path / "shared" / "huggingface"

        got = plan_cache(mine, shared, same_user=True)

        assert got.decision == IN_PLACE
        assert got.service_cache == mine, "the service was pointed somewhere else"
        assert not shared.exists(), "it made a second cache for no reason"
        assert (mine / "hub").is_dir(), "it moved a cache it had no need to move"
        assert "nothing moved" in got.said

    def test_declining_leaves_the_users_cache_alone_and_says_what_it_will_cost(self, tmp_path):
        mine = _cache(tmp_path / "home" / ".cache" / "huggingface",
                      **{"wrenfield--quince-2b": 4096, "wrenfield--larch-9b": 8192})
        shared = tmp_path / "shared" / "huggingface"

        got = plan_cache(mine, shared, same_user=False, adopt=False)

        assert got.decision == LEFT_ALONE
        assert (mine / "hub" / "models--wrenfield--larch-9b").is_dir()
        assert not shared.exists()
        assert "download what it needs again" in got.said
        assert "larch/9b" in got.said or "wrenfield/larch-9b" in got.said

    def test_adopting_moves_the_cache_and_leaves_a_link_back(self, tmp_path):
        mine = _cache(tmp_path / "home" / ".cache" / "huggingface",
                      **{"wrenfield--quince-2b": 4096})
        shared = tmp_path / "shared" / "huggingface"

        got = plan_cache(mine, shared, same_user=False, adopt=True)

        assert got.decision == ADOPTED
        assert mine.is_symlink(), "the user's own tools now look at nothing"
        assert mine.resolve() == shared.resolve()
        moved = shared / "hub" / "models--wrenfield--quince-2b" / "blobs" / "weights.gguf"
        assert moved.is_file() and moved.stat().st_size == 4096
        # Through the link as well: one set of files, reachable from both paths.
        assert (mine / "hub" / "models--wrenfield--quince-2b").is_dir()

    def test_adopting_twice_is_a_no_op(self, tmp_path):
        mine = _cache(tmp_path / "home" / ".cache" / "huggingface",
                      **{"wrenfield--quince-2b": 4096})
        shared = tmp_path / "shared" / "huggingface"
        plan_cache(mine, shared, same_user=False, adopt=True)

        again = plan_cache(mine, shared, same_user=False, adopt=True)

        assert again.decision == ADOPTED and "already points at" in again.said
        assert (shared / "hub" / "models--wrenfield--quince-2b").is_dir()

    def test_a_shared_cache_that_already_exists_is_never_written_over(self, tmp_path):
        mine = _cache(tmp_path / "home" / ".cache" / "huggingface",
                      **{"wrenfield--quince-2b": 4096})
        shared = _cache(tmp_path / "shared" / "huggingface",
                        **{"wrenfield--larch-9b": 4096})

        got = plan_cache(mine, shared, same_user=False, adopt=True)

        assert got.error and "already exists" in got.error
        assert (mine / "hub").is_dir(), "it moved the cache anyway"

    def test_the_models_are_listed_biggest_first_with_their_sizes(self, tmp_path):
        mine = _cache(tmp_path / "hf", **{"wrenfield--quince-2b": 100,
                                          "wrenfield--larch-9b": 900})
        assert models_in(mine) == [("wrenfield/larch-9b", 900),
                                   ("wrenfield/quince-2b", 100)]

    def test_a_machine_with_no_cache_is_not_an_error(self, tmp_path):
        assert models_in(tmp_path / "nothing") == []


class TestWhatToStartWith:
    """A first model is chosen from what was measured, not from what is newest."""

    class _Profile:
        def __init__(self, model: str, draft: str = "", build: str = "") -> None:
            self.model, self.draft, self.build = model, draft, build

    PROFILES = [_Profile("larch-104b-Q4_K_XL.gguf", "mtp-larch.gguf", "unsloth"),
                _Profile("quince-2b-qat.gguf", "mtp-quince.gguf")]
    FITS = [{"model": "larch-104b-Q4_K_XL.gguf", "weights": 104_000_000_000,
             "context": 32768, "per_token": 0},
            {"model": "quince-2b-qat.gguf", "weights": 2_600_000_000,
             "context": 32768, "per_token": 0}]

    def test_a_big_machine_gets_the_best_measured_model(self):
        got = choose_model(200_000_000_000, profiles=self.PROFILES, fits=self.FITS)
        assert got["model"] == "larch-104b-Q4_K_XL.gguf"
        assert got["build"] == "unsloth", "the fork its profile needs was not named"

    def test_a_small_machine_gets_the_one_it_has_room_for(self):
        got = choose_model(8_000_000_000, profiles=self.PROFILES, fits=self.FITS)
        assert got["model"] == "quince-2b-qat.gguf"

    def test_a_machine_with_room_for_nothing_measured_is_told_so(self):
        assert choose_model(1_000_000_000, profiles=self.PROFILES, fits=self.FITS) is None

    def test_none_picks_nothing_and_asks_no_questions(self):
        assert choose_model(200_000_000_000, want="none") is None

    def test_a_word_narrows_it_to_that_model(self):
        got = choose_model(200_000_000_000, want="quince", profiles=self.PROFILES,
                           fits=self.FITS)
        assert got["model"] == "quince-2b-qat.gguf"

    def test_the_default_is_the_small_one_that_still_answers(self):
        """A first install finishes while the person is still watching."""
        assert "E2B" in autostart.DEFAULT_MODEL
