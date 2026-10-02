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
)
