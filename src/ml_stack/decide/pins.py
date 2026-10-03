"""The released pointer-head checkpoint and the base model it sits on, pinned file by file."""

from __future__ import annotations

from dataclasses import dataclass

from ml_stack.decide.fetch import Pin

STRANDS = "StrandsAgents/strands-decider-2B-hobson-v19"
STRANDS_REV = "bb282d786bc251fd4e3068de3ada9ddbb38127cd"
BASE = "Qwen/Qwen3.5-2B-Base"
BASE_REV = "b1485b2fa6dfa1287294f269f5fb618e03d52d7c"


@dataclass(frozen=True, slots=True)
class Checkpoint:
    """The files a pointer decider loads."""

    name: str
    files: tuple[Pin, ...]
    licence: str
    base_repo: str = ""

    @property
    def base_files(self) -> tuple[Pin, ...]:
        """The pins of the base model alone: every file when ``base_repo`` is not set."""
        return tuple(p for p in self.files if not self.base_repo or p.repo == self.base_repo)

    @property
    def total_bytes(self) -> int:
        """Download size of every file."""
        return sum(p.size for p in self.files)


def _strands(name: str, sha: str, size: int) -> Pin:
    return Pin(STRANDS, STRANDS_REV, name, sha, size)


def _base(name: str, sha: str, size: int) -> Pin:
    return Pin(BASE, BASE_REV, name, sha, size)


STRANDS_V19 = Checkpoint(
    "strands-decider-2B-hobson-v19",
    (
        _strands("hobson_config.json",
                 "2ae86f2ed56975f68e8f2f368104df8ce0ff4b7d9d146dcaf8eec5626d62449d", 743),
        _strands("head.safetensors",
                 "daad0727152b6185447cee36f78230c144312feb7b19f02749b19645238b5287", 4213224),
        _strands("lora/adapter_config.json",
                 "eb48e4ff81569664c4dd2a504b53da598eeaa390c265c2269ce4fe7526eab38a", 1271),
        _strands("lora/adapter_model.safetensors",
                 "701bdb895887097f7954ec7eb06f7937d3b790035b5462195eb26abd80a4aebc", 67324872),
        _strands("tokenizer.json",
                 "a2cdd2e108566b09079afa8d266e9e65e7c280f27b771218237a06de5ba9cd86", 19989592),
        _strands("tokenizer_config.json",
                 "8671bed7c852ce9e661be94f179a7b4ffd091c2a65aea0363e5501c20318ee45", 1128),
        _base("config.json",
              "ed1c1723241f23f7f4e23430759cbd7dcfb4103cbdfe052bfe7626b57c2615b4", 2908),
        _base("model.safetensors.index.json",
              "74d2ddfe79f10f35b27b498632f02b97b60dd9ec39b35d7c5a890c399284e319", 64460),
        _base("model.safetensors-00001-of-00001.safetensors",
              "928acbf11878c32185bbd863514d191769285065ab9ea14fbfe431303f5fdf2d", 4548221488),
    ),
    "Apache-2.0 (checkpoint and Qwen3.5-2B-Base)",
    BASE,
)


TINY_BASE_REPO = "Qwen/Qwen3.5-0.8B-Base"
TINY_BASE_REV = "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"


def _tiny(name: str, sha: str, size: int) -> Pin:
    return Pin(TINY_BASE_REPO, TINY_BASE_REV, name, sha, size)


QWEN35_0_8B_BASE = Checkpoint(
    "qwen3.5-0.8b-base",
    (
        _tiny("config.json",
              "b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204", 2907),
        _tiny("model.safetensors.index.json",
              "ce9a885efdf27d3664fdef5d512ad365216f1074051ef840c7cd8e5431495d0a", 50900),
        _tiny("model.safetensors-00001-of-00001.safetensors",
              "c2b1e5a17d9c1e27685d92ed9b382911ebb99955ecd89052d1721241adfbab6c", 1746942600),
        _tiny("tokenizer.json",
              "fe000e3ed39ed12b8d2481d527d44f93c65d37e87645d2dcc80d1bf9d50d2927", 12807196),
        _tiny("tokenizer_config.json",
              "e611fbccc7c29ef3b1cafb1cb7ea548d189968632901d678fd62be68c47885de", 16712),
    ),
    "Apache-2.0",
)


QWEN35_2B_BASE = Checkpoint("qwen3.5-2b-base", STRANDS_V19.base_files, "Apache-2.0")

GEMMA4_E2B_REPO = "google/gemma-4-E2B"
GEMMA4_E2B_REV = "d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f"


def _gemma(name: str, sha: str, size: int) -> Pin:
    return Pin(GEMMA4_E2B_REPO, GEMMA4_E2B_REV, name, sha, size)


GEMMA4_E2B = Checkpoint(
    "gemma-4-e2b",
    (
        _gemma("config.json",
               "e5faef0dd1a8f2437f6010721146b85433eaa90e679ef011e803c7ffefae73b8", 4914),
        _gemma("model.safetensors",
               "76dc84a5a805a2c8b91e9ccc00b8dbf8f4a99bf0d56ab25832f6e6addd4f7f57", 10246621918),
        _gemma("tokenizer.json",
               "12bac982b793c44b03d52a250a9f0d0b666813da566b910c24a6da0695fd11e6", 32170070),
        _gemma("tokenizer_config.json",
               "12754e3442e47c1a1c55d550b617fc47ae227e8156ed1b1de0d8428195768b5c", 906),
    ),
    "Apache-2.0",
)

BASES = {c.name: c for c in (QWEN35_0_8B_BASE, QWEN35_2B_BASE, GEMMA4_E2B)}
"""The pinned base models a decider can be trained on, by the name `--base` takes."""
