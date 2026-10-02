"""The text a pointer-head model reads, and where in it each option is scored from."""

from __future__ import annotations

from dataclasses import dataclass

from ml_stack.decide.types import Option

HEADER = "Select exactly one option."


@dataclass(frozen=True, slots=True)
class Rendered:
    """A prompt and the character span of each option's line within it."""

    text: str
    spans: tuple[tuple[int, int], ...]


def render(question: str, state: str, options: tuple[Option, ...]) -> Rendered:
    """The prompt for a choice question: the state, then the numbered options, then ``<answer>``.

    Each option is one line, ``1. name - description``, with whitespace in the description
    collapsed so that a line is a span.
    """
    prefix = (f"<state>\n{state}\n</state>\n"
              f'<question type="choice">\n{HEADER}\n{question}\n<options>\n')
    lines, spans, cursor = [], [], len(prefix)
    for i, option in enumerate(options):
        desc = " ".join(option.description.split())
        line = f"{i + 1}. {option.name}" + (f" — {desc}" if desc else "")
        lines.append(line)
        spans.append((cursor, cursor + len(line)))
        cursor += len(line) + 1
    return Rendered(prefix + "\n".join(lines) + "\n</options>\n</question>\n<answer>",
                    tuple(spans))


def positions(offsets: list[tuple[int, int]], spans: tuple[tuple[int, int], ...]) -> list[int]:
    """For each span, the index of the last token that ends inside it."""
    found = []
    for start, end in spans:
        inside = [i for i, (a, b) in enumerate(offsets) if b > a and a >= start and b <= end]
        if not inside:
            raise ValueError(f"no token covers the option line at characters {start}-{end}")
        found.append(inside[-1])
    return found
