"""Packed token rows, served from device memory.

The prepared dataset is one array per split plus the metadata a bits-per-byte
score needs::

    all__tokens.npy      [n_rows, max_seq_len + 1] packed token ids
    all__token_bytes.npy [vocab_size] UTF-8 byte length of each token
    dataset.json         {"vocab_size": int, "max_seq_len": int}

Each row is a full context with no padding: documents are packed end to end
during preparation, so every position carries a real target and the loss needs
no mask. A row is one training example, and the input/target pair is the row
offset by one -- which is why a row holds ``max_seq_len + 1`` tokens.

``all__token_bytes.npy`` is what makes the score comparable across tokenizers.
Cross-entropy per TOKEN falls simply by making tokens larger, so the metric
divides by bytes instead; a token's byte length is a property of the vocabulary
and is therefore recorded beside it. Special tokens are zero, excluding them
from the denominator.

``scripts/prepare_data.py`` downloads, trains the tokenizer, and packs; this
module only reads the result, so constructing a config never touches the
network -- and the tokenizer libraries are needed by the script alone.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any, Self, override

import json
import logging
import math

from configgle import Fig
from torch import Tensor

import numpy as np
import torch

from priml.paths import resolve_working_dir
from priml.runtime import get_device


logger = logging.getLogger(__name__)


class NanoChatData:
    """Packed token rows held in device memory, yielding ``media`` batches.

    Batches carry ``media`` (inputs) and ``label`` (the same rows shifted by
    one). The evaluation split is walked in order and the training split in a
    seeded shuffle, so a run replays from its seed and pass count alone.

    Raises:
      FileNotFoundError: If the prepared arrays are absent. Run
        ``uv --quiet run --frozen python -m
        priml.baselines.nanochat.scripts.prepare_data`` first.

    """

    class Config(Fig["NanoChatData"]):
        """Where the prepared arrays live, and how batches are drawn."""

        base_dir: Path | str | None = None
        """Resource root supplied during parent finalization."""

        working_dir: Path | str = "/datasets/nanochat"
        """Directory holding the ``train/`` and ``val/`` splits.

        Resolved beneath ``base_dir`` at finalize, so it names a location
        within the resource root rather than an absolute filesystem path."""

        batch_size: int = 32
        """Rows per training batch."""

        eval_batch_size: int | None = None
        """Rows per evaluation batch; ``None`` reuses ``batch_size``."""

        device: str = "auto"
        """Device holding the resident arrays ("auto" picks the best)."""

        seed: int = 0
        """Seeds the shuffle; each pass draws from ``seed + pass``."""

        num_eval_rows: int | None = None
        """Evaluation rows to score; ``None`` scores all of them.

        The full validation split is large, so mid-training evaluation normally
        reads a fixed prefix as a proxy and the reported number comes from an
        uncapped final pass. The two populations are not comparable."""

        @override
        def finalize(self) -> Self:
            self.working_dir = resolve_working_dir(self.base_dir, self.working_dir)
            return super().finalize()

    def __init__(self, config: Config) -> None:
        if config.batch_size <= 0:
            raise ValueError(f"batch_size must be positive; got {config.batch_size}.")
        self.config = config
        self.dataset_dir = Path(config.working_dir)
        self.batch_size = config.batch_size
        self.eval_batch_size = config.eval_batch_size or config.batch_size
        self._passes = 0
        self._live: _TokenBatches | None = None

    @property
    def token_bytes(self) -> Tensor:
        """UTF-8 byte length of every token id, for the bits-per-byte score."""
        return _load_split(self.dataset_dir, "val", device=self.config.device)[1]

    def train_dataloader(self) -> _TokenBatches:
        """Build the re-iterable training stream."""
        # Snapshot any prior stream's counter first, so re-creating the loader
        # continues the sequence rather than restarting it.
        if self._live is not None:
            self._passes = self._live.passes
        stream = _TokenBatches(
            dataset_dir=self.dataset_dir,
            device=self.config.device,
            batch_size=self.batch_size,
            split="train",
            shuffle=True,
            num_rows=None,
            seed=self.config.seed,
            passes=self._passes,
        )
        self._live = stream
        return stream

    def eval_dataloader(self) -> _TokenBatches:
        """Build the evaluation stream: disk order, never shuffled.

        Every candidate must be scored on the identical token stream, which a
        shuffle would break.
        """
        return _TokenBatches(
            dataset_dir=self.dataset_dir,
            device=self.config.device,
            batch_size=self.eval_batch_size,
            split="val",
            shuffle=False,
            num_rows=self.config.num_eval_rows,
            seed=self.config.seed,
            passes=0,
        )

    def state_dict(self) -> dict[str, Any]:
        """Snapshot the completed-pass count.

        The counter advances when a pass BEGINS, so a mid-pass checkpoint
        resumes with the next pass's order rather than replaying the
        interrupted one.
        """
        return {"passes": self._live.passes if self._live is not None else self._passes}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore state produced by :meth:`state_dict`."""
        if "passes" in state_dict:
            self._passes = int(state_dict["passes"])
            if self._live is not None:
                self._live.passes = self._passes


class _TokenBatches:
    """One split, resident on device, iterated in fixed-size batches."""

    def __init__(
        self,
        *,
        dataset_dir: Path,
        device: torch.device | str,
        batch_size: int,
        split: str,
        shuffle: bool,
        num_rows: int | None,
        seed: int,
        passes: int,
    ) -> None:
        rows, token_bytes = _load_split(dataset_dir, split, device=device)
        self.device = get_device(device)
        self.rows: Tensor = rows if num_rows is None else rows[:num_rows]
        self.token_bytes = token_bytes
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.passes = passes

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Yield every row once as an ``(input, target)`` pair.

        A short final batch is dropped rather than padded: every row is a full
        context by construction, so a partial batch would be the only place in
        the run where the token count per step changes.
        """
        count = self.rows.shape[0]
        if self.shuffle:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(self.seed + self.passes)
            order = torch.randperm(count, device=self.device, generator=generator)
        else:
            order = torch.arange(count, device=self.device)
        self.passes += 1
        for start in range(0, count - self.batch_size + 1, self.batch_size):
            rows = self.rows[order[start : start + self.batch_size]]
            yield {
                "media": rows[:, :-1].long(),
                "label": rows[:, 1:].long(),
                "token_bytes": self.token_bytes,
            }

    def __len__(self) -> int:
        """Whole batches per pass; a short final batch is dropped."""
        return math.floor(self.rows.shape[0] / self.batch_size)


def _load_split(
    dataset_dir: Path,
    split: str,
    *,
    device: torch.device | str,
) -> tuple[Tensor, Tensor]:
    """Read one prepared split onto ``device``.

    Args:
      dataset_dir: Dataset root holding ``train/`` and ``val/``.
      split: Which one to read.
      device: Where the resident arrays live.

    Returns:
      rows: ``[n_rows, max_seq_len + 1]`` packed token ids.
      token_bytes: ``[vocab_size]`` UTF-8 byte length per token id.

    Raises:
      FileNotFoundError: If the split or its metadata is missing.

    """
    path = Path(dataset_dir).expanduser() / split
    metadata_path = path / "dataset.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"no prepared nanochat data at {path}; build it with "
            "`uv --quiet run --frozen python -m "
            "priml.baselines.nanochat.scripts.prepare_data`.",
        )
    metadata = json.loads(metadata_path.read_text())
    logger.info("loading nanochat split %r from %s", split, path)
    resolved = get_device(device)
    # Prepared as uint16, which torch cannot hold; int32 is the smallest dtype
    # that represents the whole vocabulary and still indexes an embedding.
    rows = torch.from_numpy(
        np.load(path / "all__tokens.npy").astype(np.int32),
    ).to(resolved)
    token_bytes = torch.from_numpy(
        np.load(path / "all__token_bytes.npy").astype(np.int32),
    ).to(resolved)
    if token_bytes.shape[0] != int(metadata["vocab_size"]):
        raise ValueError(
            f"{path} declares vocab_size {metadata['vocab_size']} but its byte "
            f"table holds {token_bytes.shape[0]} entries.",
        )
    logger.info(
        "nanochat %r: %d rows of %d tokens",
        split,
        rows.shape[0],
        rows.shape[1] - 1,
    )
    return rows, token_bytes
