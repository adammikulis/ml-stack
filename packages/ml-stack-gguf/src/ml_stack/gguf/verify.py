"""Prove an exported GGUF reads the same sequences the model was trained on.

Train/inference tokenizer divergence is the failure this exists for, and it is
the worst kind: nothing raises, generation stays fluent, and the model is simply
reading different token ids than it learned from. It is usually found weeks
later as "the model got worse for no reason".

Three checks, cheapest first, because an early failure explains the later ones:

1. **The file opens.** A GGUF writer that copies a reader's SYNTHETIC header
   fields (``GGUF.version``, ``GGUF.tensor_count``, ``GGUF.kv_count``) emits a
   duplicate key. llama.cpp tolerates it; the Python ``gguf`` reader refuses the
   file outright. That asymmetry is how such a file ships unnoticed.
2. **``add_space_prefix`` is present.** Absent is not neutral -- llama.cpp
   defaults it to TRUE, which inserts a space after every special token that a
   SentencePiece tokenizer never produced.
3. **The served model agrees with a reference tokenizer, id for id.** The only
   check that can observe a divergence rather than infer one from metadata.

``encode`` is any callable ``str -> list[int]``, so this is tokenizer-agnostic:
pass ``sentencepiece``'s, a HF tokenizer's, or your own.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from ml_stack.gguf.vocab import ADD_SPACE_PREFIX, read_metadata

__all__ = ["Check", "FidelityReport", "verify_metadata", "verify_tokenizer_fidelity"]


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str = ""


@dataclass
class FidelityReport:
    checks: list[Check] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(c.ok for c in self.checks)

    @property
    def failures(self) -> list[Check]:
        return [c for c in self.checks if not c.ok]

    def __str__(self) -> str:
        return "\n".join(
            f"{'ok  ' if c.ok else 'FAIL'}  {c.name}" + (f": {c.detail}" if c.detail else "")
            for c in self.checks
        )


def verify_metadata(gguf: Path | str, *, expect_space_prefix: bool | None = False) -> FidelityReport:
    """Checks 1 and 2. No server, no tokenizer, no model load."""
    report = FidelityReport()
    try:
        meta = read_metadata(gguf)
    except Exception as exc:
        report.checks.append(Check(
            "opens", False,
            f"{type(exc).__name__}: {exc} -- a duplicate GGUF.* header field does this, "
            "and llama.cpp tolerates what this reader refuses"))
        return report

    report.checks.append(Check("opens", True, f"{len(meta)} metadata keys"))
    if expect_space_prefix is None:
        return report

    value = meta.get(ADD_SPACE_PREFIX)
    if value is None:
        report.checks.append(Check(
            ADD_SPACE_PREFIX, False,
            "ABSENT -- llama.cpp defaults it to TRUE, so the runtime inserts a space "
            "after every special token that the training tokenizer never produced"))
    elif bool(value) is not bool(expect_space_prefix):
        report.checks.append(Check(ADD_SPACE_PREFIX, False,
                                   f"{value!r}, expected {expect_space_prefix!r}"))
    else:
        report.checks.append(Check(ADD_SPACE_PREFIX, True, str(bool(value))))
    return report


def verify_tokenizer_fidelity(gguf: Path | str, encode: Callable[[str], Sequence[int]],
                              probes: Sequence[str], *, context: int = 1024,
                              bos_id: int | None = None,
                              expect_space_prefix: bool | None = False,
                              serve_fn=None, client_cls=None) -> FidelityReport:
    """All three checks. Serves the GGUF and compares /tokenize against ``encode``.

    ``probes`` should be REAL sequences from the training corpus -- tagged turns,
    structured output, accented text -- not lorem. Divergence concentrates
    exactly on the special tokens and punctuation that lorem does not contain.

    ``serve_fn`` and ``client_cls`` are injectable so this is testable without a
    model; they default to ``ml_stack.serve.serve`` and ``ml_stack.client.Client``.
    """
    report = verify_metadata(gguf, expect_space_prefix=expect_space_prefix)
    if not report.checks or not report.checks[0].ok:
        return report                      # it does not open; nothing else can run

    if serve_fn is None:
        from ml_stack.serve import serve as serve_fn
    if client_cls is None:
        from ml_stack.client import Client as client_cls

    with serve_fn(gguf, context=context) as server:
        client = client_cls(server.base_url)
        for text in probes:
            want = list(encode(text))
            got = list(client.tokenize(text))
            # llama.cpp prepends BOS; most reference tokenizers do not.
            if bos_id is not None and got and got[0] == bos_id:
                got = got[1:]
            if got == want:
                report.checks.append(Check(f"tokenize {text[:44]!r}", True, f"{len(want)} ids"))
                continue
            pieces = client.tokenize(text, with_pieces=True)
            report.checks.append(Check(
                f"tokenize {text[:44]!r}", False,
                f"reference={want} server={got} "
                f"pieces={[p.get('piece') for p in pieces if isinstance(p, dict)]}"))
    return report
