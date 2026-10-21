"""Immutable source-sequence split handling for the CRUW public protocol."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CRUWProtocol:
    """The protocol exposes only fit and validation source sequences."""

    fit_sequences: tuple[int, ...]
    validation_sequences: tuple[int, ...]

    def __post_init__(self) -> None:
        fit = set(self.fit_sequences)
        validation = set(self.validation_sequences)
        overlap = sorted(fit & validation)
        if overlap:
            raise ValueError(f"fit/validation overlap: {overlap}")
        if not fit or not validation:
            raise ValueError("fit and validation splits must both be non-empty")

    @property
    def all_sequences(self) -> tuple[int, ...]:
        return tuple(sorted((*self.fit_sequences, *self.validation_sequences)))

    def sequence_ids(self, split: str) -> tuple[int, ...]:
        if split == "fit":
            return self.fit_sequences
        if split == "validation":
            return self.validation_sequences
        if split == "test":
            raise ValueError("official test access is forbidden by this protocol")
        raise ValueError(f"unsupported protocol split: {split}")


def load_protocol(path: str | Path) -> CRUWProtocol:
    """Load the frozen CRUW public-sequence 12/4 protocol."""

    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    return CRUWProtocol(
        fit_sequences=tuple(int(value) for value in payload["fit_sequences"]),
        validation_sequences=tuple(int(value) for value in payload["validation_sequences"]),
    )
