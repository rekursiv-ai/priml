"""Read authentic competition CSVs without changing text or prompt groups."""

from __future__ import annotations

from dataclasses import field
from pathlib import Path
from typing import Final, cast

import csv
import hashlib
import io

from configgle import Fig

import numpy as np


LABELS: Final = (
    "claude-fable-5",
    "claude-opus-4-6",
    "claude-opus-4-8",
    "claude-opus-5",
    "claude-sonnet-5",
    "gemini-3.1-pro-preview",
    "gemini-3.6-flash",
    "gemini-3.7-flash",
    "gemini-3.8-flash",
    "glm-5.3",
    "glm-5.3-flash",
    "gpt-5.5",
    "gpt-5.6-luna",
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-6-astra",
    "kimi-k3",
    "qwen3.8-max",
)


class CompetitionData:
    """Competition rows in their original CSV order."""

    class Config(Fig["CompetitionData"]):
        """Authentic input location and competition schema."""

        directory: Path | str = "/datasets/name-that-llm/raw"
        """Directory containing competition train.csv and test.csv."""

        labels: list[str] = field(default_factory=lambda: list(LABELS))
        """Sorted official class vocabulary."""

        expected_rows: int = 2699
        """Number of labeled competition passages."""

        expected_prompts: int = 150
        """Number of distinct labeled prompt groups."""

        max_prompts: int | None = None
        """Smoke-only cap applied after full competition-input validation."""

    def __init__(self, config: Config) -> None:
        directory = Path(config.directory)
        train, train_sha256 = _read(
            directory / "train.csv",
            columns=["id", "prompt_id", "text", "model"],
        )
        test, test_sha256 = _read(directory / "test.csv", columns=["id", "text"])
        self.sha256 = {"train.csv": train_sha256, "test.csv": test_sha256}
        self.ids = [row["id"] for row in train]
        self.texts = [row["text"] for row in train]
        self.test_ids = [row["id"] for row in test]
        self.test_texts = [row["text"] for row in test]
        self.groups = np.array([row["prompt_id"] for row in train], dtype=str)
        labels = [row["model"] for row in train]
        if len(set(self.ids)) != len(train) or len(set(self.test_ids)) != len(test):
            raise ValueError("Competition passage IDs must be unique.")
        if set(self.ids) & set(self.test_ids):
            raise ValueError("Training and test passage IDs must be disjoint.")
        if any(not text for text in self.texts) or any(
            not row["prompt_id"] for row in train
        ):
            raise ValueError("Training text and prompt_id must be nonempty.")
        if len(train) != config.expected_rows:
            raise ValueError(f"Expected {config.expected_rows} labeled passages.")
        if len(set(self.groups)) != config.expected_prompts:
            raise ValueError(f"Expected {config.expected_prompts} distinct prompts.")
        if config.labels != sorted(set(labels)):
            raise ValueError("Training labels must match the sorted class vocabulary.")
        self.targets = np.array(
            [config.labels.index(label) for label in labels],
            dtype=np.int64,
        )
        if config.max_prompts is not None:
            keep = sorted(set(self.groups))[: config.max_prompts]
            indices = [
                index for index, row in enumerate(train) if row["prompt_id"] in keep
            ]
            self.ids = [self.ids[index] for index in indices]
            self.texts = [self.texts[index] for index in indices]
            self.groups = self.groups[indices]
            self.targets = self.targets[indices]


def _read(path: Path, columns: list[str]) -> tuple[list[dict[str, str]], str]:
    with path.open(encoding="utf-8", newline="") as stream:
        text = stream.read()
    reader = csv.DictReader(io.StringIO(text, newline=""))
    if reader.fieldnames != columns:
        raise ValueError(f"Expected columns {columns} in {path}.")
    rows = list(reader)
    if not rows or any(
        set(row) != set(columns) or None in row.values() for row in rows
    ):
        raise ValueError(f"Malformed or empty competition CSV: {path}.")
    # Strict UTF-8 without newline translation re-encodes to the file's bytes.
    return cast("list[dict[str, str]]", rows), hashlib.sha256(text.encode()).hexdigest()
