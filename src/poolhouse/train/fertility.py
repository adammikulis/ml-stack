"""How many tokens a language costs, and what a vocabulary costs to hold."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

__all__ = ["Fertility", "embedding_params", "measure", "report_markdown"]


@dataclass(frozen=True)
class Fertility:
    lang: str
    vocab: int
    n_bytes: int
    n_words: int
    n_tokens: int

    @property
    def bytes_per_token(self) -> float:
        return self.n_bytes / self.n_tokens if self.n_tokens else 0.0

    @property
    def tokens_per_word(self) -> float:
        return self.n_tokens / self.n_words if self.n_words else 0.0


def measure(encode: Callable[[str], Sequence[int]],
            samples: Mapping[str, Sequence[str]], *, vocab: int) -> list[Fertility]:
    """Fertility per language for one tokenizer."""
    out: list[Fertility] = []
    for lang, texts in samples.items():
        n_bytes = n_words = n_tokens = 0
        for t in texts:
            if not t:
                continue
            n_bytes += len(t.encode("utf-8"))
            n_words += max(1, len(t.split()))
            n_tokens += len(encode(t))
        out.append(Fertility(lang=lang, vocab=vocab, n_bytes=n_bytes,
                             n_words=n_words, n_tokens=n_tokens))
    return out


def embedding_params(vocab: int, d_model: int, *, tied: bool = True) -> int:
    """Parameters held by the embedding (and output projection, if untied)."""
    return vocab * d_model * (1 if tied else 2)


def report_markdown(rows: Sequence[Fertility], *, d_model: int,
                    non_embedding_params: int, tied: bool = True,
                    baseline_lang: str | None = None) -> str:
    """A table meant to be READ before a training run, not filed after one."""
    if not rows:
        return "no measurements\n"

    vocabs = sorted({r.vocab for r in rows})
    langs = list(dict.fromkeys(r.lang for r in rows))
    base = baseline_lang or langs[0]

    lines = [
        "# Tokenizer fertility",
        "",
        f"d_model {d_model}, {'tied' if tied else 'untied'} embeddings, "
        f"{non_embedding_params:,} non-embedding parameters.",
        "",
        "`bytes/token` higher is better -- more text per token. `rel` is "
        f"tokens-per-word against **{base}** at the same vocabulary: 1.20 means "
        "this language needs 20% more tokens to say the same thing.",
        "",
        "| vocab | lang | bytes/token | tokens/word | rel | embed params | % of model |",
        "|---|---|---|---|---|---|---|",
    ]
    for v in vocabs:
        emb = embedding_params(v, d_model, tied=tied)
        total = emb + non_embedding_params
        at_v = {r.lang: r for r in rows if r.vocab == v}
        ref = at_v.get(base)
        for lang in langs:
            r = at_v.get(lang)
            if r is None:
                continue
            rel = (r.tokens_per_word / ref.tokens_per_word
                   if ref and ref.tokens_per_word else 1.0)
            lines.append(
                f"| {v:,} | {lang} | {r.bytes_per_token:.2f} | "
                f"{r.tokens_per_word:.2f} | {rel:.2f} | {emb/1e6:.1f}M | "
                f"{100*emb/total:.0f}% |")
    lines.append("")

    worst = max(rows, key=lambda r: r.tokens_per_word)
    best = min(rows, key=lambda r: r.tokens_per_word)
    lines += [
        "## Reading it",
        "",
        f"- Worst served: **{worst.lang}** at vocab {worst.vocab:,} "
        f"({worst.tokens_per_word:.2f} tokens/word).",
        f"- Best served: **{best.lang}** at vocab {best.vocab:,} "
        f"({best.tokens_per_word:.2f} tokens/word).",
        "- A `rel` far from 1.00 means one language is paying for the others. "
        "Growing the vocabulary usually narrows it -- at a cost visible in the "
        "last two columns.",
        "",
    ]
    return "\n".join(lines)
