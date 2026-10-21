"""Immutable complete-sequence split handling for CARRADA."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CARRADAProtocol:
    fit_sequences: tuple[str, ...]
    development_sequences: tuple[str, ...]
    locked_test_sequences: tuple[str, ...]

    def __post_init__(self) -> None:
        fit = set(self.fit_sequences)
        development = set(self.development_sequences)
        locked_test = set(self.locked_test_sequences)
        overlap = sorted((fit & development) | (fit & locked_test) | (development & locked_test))
        if overlap:
            raise ValueError(f"fit/development overlap: {overlap}")
        if not fit or not development:
            raise ValueError("train and validation splits must be non-empty")
        if (
            len(fit) != len(self.fit_sequences)
            or len(development) != len(self.development_sequences)
            or len(locked_test) != len(self.locked_test_sequences)
        ):
            raise ValueError("duplicate sequence in protocol")

    @property
    def all_sequences(self) -> tuple[str, ...]:
        return tuple(sorted((*self.fit_sequences, *self.development_sequences, *self.locked_test_sequences)))

    def sequence_ids(self, split: str) -> tuple[str, ...]:
        if split in {"fit", "train"}:
            return self.fit_sequences
        if split in {"development", "validation"}:
            return self.development_sequences
        if split in {"test", "locked_test"}:
            return self.locked_test_sequences
        raise ValueError(f"unsupported protocol split: {split}")


def load_protocol(path: str | Path) -> CARRADAProtocol:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    fit = payload.get("train_sequences", payload.get("fit_sequences"))
    development = payload.get("validation_sequences", payload.get("development_sequences"))
    locked_test = payload.get("locked_test_sequences", ())
    if fit is None or development is None:
        raise ValueError("protocol must define train/validation sequence lists")
    return CARRADAProtocol(
        fit_sequences=tuple(str(value) for value in fit),
        development_sequences=tuple(str(value) for value in development),
        locked_test_sequences=tuple(str(value) for value in locked_test),
    )
