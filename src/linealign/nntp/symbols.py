"""PyLaia symbol table helpers and NNTP label filtering."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path


CTC_TOKEN = "<ctc>"
SPACE_TOKEN = "<space>"
NNTP_SPACE = "sp"


@dataclass(frozen=True)
class SymbolTable:
    """Represents the PyLaia symbol table used for both filtering and observations."""

    raw_by_index: dict[int, str]

    @property
    def allowed_text_chars(self) -> set[str]:
        chars = set()
        for symbol in self.raw_by_index.values():
            if symbol == SPACE_TOKEN:
                chars.add(" ")
            elif symbol != CTC_TOKEN:
                chars.add(symbol)
        return chars

    @property
    def observation_symbols(self) -> list[str]:
        symbols: list[str] = []
        for index in sorted(self.raw_by_index):
            symbol = self.raw_by_index[index]
            if symbol == CTC_TOKEN:
                continue
            if symbol == SPACE_TOKEN:
                symbols.append(NNTP_SPACE)
            else:
                symbols.append(symbol)
        return symbols

    def symbol_for_index(self, index: int) -> str | None:
        symbol = self.raw_by_index[index]
        if symbol == CTC_TOKEN:
            return None
        if symbol == SPACE_TOKEN:
            return NNTP_SPACE
        return symbol


@dataclass(frozen=True)
class FilteredLabel:
    """One transcription filtered down to the PyLaia symbol set."""

    sample_id: str
    original_text: str
    filtered_text: str
    tokens: list[str]
    stripped_counts: dict[str, int]


def load_symbol_table(path: Path) -> SymbolTable:
    """Load a PyLaia syms.txt file."""

    raw_by_index: dict[int, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        symbol, index = line.split()
        raw_by_index[int(index)] = symbol
    return SymbolTable(raw_by_index=raw_by_index)


def filter_transcription_text(sample_id: str, text: str, symbol_table: SymbolTable) -> FilteredLabel:
    """Strip unsupported characters and convert spaces to NNTP tokens."""

    normalized = " ".join(text.split())
    allowed_chars = symbol_table.allowed_text_chars
    filtered_chars: list[str] = []
    stripped = Counter()

    for char in normalized:
        if char == " " or char in allowed_chars:
            filtered_chars.append(char)
        else:
            stripped[char] += 1

    filtered_text = "".join(filtered_chars)
    filtered_text = " ".join(filtered_text.split())
    tokens = [NNTP_SPACE if char == " " else char for char in filtered_text]

    return FilteredLabel(
        sample_id=sample_id,
        original_text=text,
        filtered_text=filtered_text,
        tokens=tokens,
        stripped_counts=dict(sorted(stripped.items())),
    )
