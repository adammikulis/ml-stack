"""The pointer backend: a language-model torso with its output head removed and a small
pointer head that scores each option against the answer position.

One forward pass over a prompt that lists the options; the hidden state at ``<answer>`` is the
query and the hidden state at the end of each option's line is its key. Runs the released
checkpoints in `ml_stack.decide.pins`, or one trained by `ml_stack.train.decider`.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from ml_stack.decide import pointer_prompt
from ml_stack.decide.base import Asked, BaseDecider
from ml_stack.decide.calibrate import Calibration
from ml_stack.decide.pins import STRANDS_V19, Checkpoint
from ml_stack.decide.sources import Source, local_source, strands_source, temperature_for
from ml_stack.decide.types import BackendUnavailable, DecideError

MAX_TOKENS = 4096


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise BackendUnavailable("the pointer backend needs torch, transformers and peft: "
                                 "pip install 'ml-stack[decide-pointer]'") from exc
    return torch


def device_name(requested: str = "auto") -> str:
    """``cuda`` or ``mps`` if present, else ``cpu``, unless a device is named."""
    if requested != "auto":
        return requested
    torch = _torch()
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def build_head(torch: Any, hidden: int, dim: int) -> Any:
    """The pointer readout: a shared LayerNorm, then a query and a key projection."""
    nn = torch.nn

    class Pointer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.norm = nn.LayerNorm(hidden)
            self.q = nn.Linear(hidden, dim)
            self.k = nn.Linear(hidden, dim)
            self.scale = dim ** -0.5

        def forward(self, decide: Any, options: Any) -> Any:
            """decide [B, d], options [B, K, d] -> logits [B, K]."""
            q = self.q(self.norm(decide)).unsqueeze(-1)
            k = self.k(self.norm(options))
            return (k @ q).squeeze(-1) * self.scale

    return Pointer()


def load_torso(base_dir: Path, lora_dir: Path | None, dtype: str, device: str, *,
               trainable: bool = False) -> Any:
    """The base model's decoder with its adapter applied, from safetensors only."""
    torch = _torch()
    import transformers
    config = json.loads((base_dir / "config.json").read_text())
    if "auto_map" in config:
        raise DecideError(f"{base_dir}/config.json asks for custom code; it is not loaded")
    cfg = transformers.AutoConfig.from_pretrained(base_dir, local_files_only=True,
                                                  trust_remote_code=False)
    cls = (transformers.Qwen3_5ForCausalLM if cfg.model_type in ("qwen3_5", "qwen3_5_text")
           else None)
    if cfg.model_type == "gemma4":
        # Gemma4ForCausalLM does not map `model.language_model.*` and would leave it random.
        full = transformers.Gemma4Model.from_pretrained(
            base_dir, dtype=getattr(torch, dtype), local_files_only=True,
            trust_remote_code=False, use_safetensors=True)
        torso = full.language_model
    elif cls is None:
        torso = transformers.AutoModel.from_pretrained(
            base_dir, dtype=getattr(torch, dtype), local_files_only=True,
            trust_remote_code=False, use_safetensors=True)
    else:
        lm = cls.from_pretrained(base_dir, config=cfg.get_text_config(),
                                 dtype=getattr(torch, dtype), local_files_only=True,
                                 trust_remote_code=False, use_safetensors=True)
        torso = lm.model
    if lora_dir is not None:
        from peft import PeftModel
        torso = PeftModel.from_pretrained(torso, str(lora_dir), is_trainable=trainable,
                                          use_safetensors=True)
    torso = torso.to(device)
    return torso if trainable else torso.eval()


class PointerDecider(BaseDecider):
    """Runs a pointer-head checkpoint on this machine's best device.

    ``source`` is the pinned released checkpoint (default), another pinned `Checkpoint`, or
    the directory a training run wrote. Files load lazily on first use; nothing is fetched
    unless ``download`` is set.
    """

    name = "pointer"

    def __init__(self, source: Checkpoint | Path | str = STRANDS_V19, *, device: str = "auto",
                 download: bool = False, calibration: Calibration | None = None) -> None:
        self.source = source
        self.model = source.name if isinstance(source, Checkpoint) else Path(source).name
        self.device_request = device
        self.download = download
        self.calibration = calibration
        self._lock = threading.Lock()
        self._loaded: tuple[Any, ...] | None = None

    def resolve(self) -> Source:
        """The verified local files."""
        if isinstance(self.source, Checkpoint):
            return strands_source(self.source, download=self.download)
        return local_source(self.source, download=self.download)

    def load(self) -> None:
        """Load the tokenizer, torso and head if they are not loaded."""
        with self._lock:
            if self._loaded is not None:
                return
            torch = _torch()
            from safetensors.torch import load_file
            from transformers import AutoTokenizer
            files = self.resolve()
            device = device_name(self.device_request)
            tok = AutoTokenizer.from_pretrained(files.tokenizer_dir, local_files_only=True,
                                                trust_remote_code=False)
            torso = load_torso(files.base_dir, files.lora_dir, files.dtype, device)
            state = load_file(str(files.head))
            head = build_head(torch, state["q.weight"].shape[1], files.pointer_dim)
            head.load_state_dict(state)
            head = head.to(device).float().eval()
            self._loaded = (tok, torso, head, device, files,
                            {"device": device, "dtype": files.dtype})

    def probabilities(self, asked: Asked) -> tuple[list[float], dict[str, Any]]:
        self.load()
        torch = _torch()
        if self._loaded is None:
            raise DecideError("the pointer model did not load")
        tok, torso, head, device, files, info = self._loaded
        temperature = temperature_for(files, asked.question)
        rendered = pointer_prompt.render(asked.question, asked.state, asked.options)
        enc = tok(rendered.text, return_offsets_mapping=True, add_special_tokens=False,
                  return_tensors="pt")
        length = int(enc["input_ids"].shape[1])
        if length > MAX_TOKENS:
            raise DecideError(f"the prompt is {length} tokens; the limit is {MAX_TOKENS}")
        spots = pointer_prompt.positions([tuple(o) for o in enc["offset_mapping"][0].tolist()],
                                         rendered.spans)
        with self._lock, torch.inference_mode():
            hidden = torso(input_ids=enc["input_ids"].to(device),
                           attention_mask=enc["attention_mask"].to(device),
                           use_cache=False).last_hidden_state[0]
            logits = head(hidden[-1:].float(), hidden[spots].float().unsqueeze(0))[0]
            probs = torch.softmax(logits / temperature, dim=-1).tolist()
        return probs, {"tokens": length, **info}
