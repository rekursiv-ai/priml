"""Packed token rows, served from device memory.

The prepared dataset is one directory per split, each holding the arrays and
the metadata a bits-per-byte score needs::

    <split>/all__tokens.npy      [n_rows, max_seq_len + 1] packed token ids
    <split>/all__token_bytes.npy [vocab_size] UTF-8 byte length of each token
    <split>/dataset.json         geometry + the byte table's fingerprint

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

import hashlib
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

        vocab_size: int = -1
        """Token vocabulary the prepared arrays must hold; -1 skips the check.

        Declared rather than read from disk: a config must build without
        touching the filesystem. The model pushes its own value down, and
        ``__init__`` verifies the arrays against it -- so a prepared directory
        that disagrees fails at load, naming both numbers, instead of surfacing
        later as an out-of-range embedding index."""

        max_seq_len: int = -1
        """Context length the prepared rows must carry; -1 skips the check.

        A row holds ``max_seq_len + 1`` tokens, since the inputs and targets
        are the row offset by one."""

        @override
        def finalize(self) -> Self:
            self.working_dir = resolve_working_dir(self.base_dir, self.working_dir)
            return super().finalize()

    def __init__(self, config: Config) -> None:
        if config.batch_size <= 0:
            raise ValueError(f"batch_size must be positive; got {config.batch_size}.")
        if config.eval_batch_size is not None and config.eval_batch_size <= 0:
            raise ValueError(
                f"eval_batch_size must be positive; got {config.eval_batch_size}.",
            )
        if config.num_eval_rows is not None and config.num_eval_rows <= 0:
            raise ValueError(
                f"num_eval_rows must be positive; got {config.num_eval_rows}.",
            )
        self.config = config
        self.dataset_dir = Path(config.working_dir)
        self.batch_size = config.batch_size
        # ``is None``, not ``or``: zero is falsy, so an ``or`` would absorb an
        # explicit 0 into the training batch size and still report a score.
        self.eval_batch_size = (
            config.batch_size
            if config.eval_batch_size is None
            else config.eval_batch_size
        )
        self._passes = 0
        self._live: _TokenBatches | None = None

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
            vocab_size=self.config.vocab_size,
            max_seq_len=self.config.max_seq_len,
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
            vocab_size=self.config.vocab_size,
            max_seq_len=self.config.max_seq_len,
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
        vocab_size: int,
        max_seq_len: int,
    ) -> None:
        rows, token_bytes = _load_split(
            dataset_dir,
            split,
            device=device,
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
        )
        self.device = get_device(device)
        self.rows: Tensor = rows if num_rows is None else rows[:num_rows]
        self.token_bytes = token_bytes
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.passes = passes
        # A short batch is dropped, so too few rows yields NOTHING -- and an
        # empty stream is invisible until it surfaces far away: training as a
        # generic epoch-reset error, evaluation as a metric with no samples.
        # Both are the same misconfiguration, so it is named once, here, where
        # the row count and the batch size are both in hand.
        available = int(self.rows.shape[0])
        if available < batch_size:
            raise ValueError(
                f"the {split!r} split yields no batches: {available} rows "
                f"available (of {rows.shape[0]}) against a batch size of "
                f"{batch_size}. Lower the batch size, raise the row cap, or "
                "prepare more data.",
            )

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


def token_bytes_fingerprint(token_bytes: np.ndarray) -> str:
    """Return the identity of one byte-length table.

    The table is the DENOMINATOR of every reported score, so two tables of the
    same length are two different metrics wearing one name. Recording the
    fingerprint beside the data is what lets a reader tell which accounting
    produced a number -- a distinction worth roughly the size of a real
    candidate effect, and invisible to any shape check.

    Args:
      token_bytes: Per-token-id byte lengths.

    Returns:
      fingerprint: Hex SHA-256 over the table's canonical int64 bytes.

    """
    return hashlib.sha256(
        np.ascontiguousarray(token_bytes, dtype=np.int64).tobytes(),
    ).hexdigest()


def _load_split(
    dataset_dir: Path,
    split: str,
    *,
    device: torch.device | str,
    vocab_size: int = -1,
    max_seq_len: int = -1,
) -> tuple[Tensor, Tensor]:
    """Read one prepared split onto ``device``, verifying its geometry.

    Args:
      dataset_dir: Dataset root holding ``train/`` and ``val/``.
      split: Which one to read.
      device: Where the resident arrays live.
      vocab_size: Vocabulary the arrays must hold; -1 skips the check.
      max_seq_len: Context the rows must carry; -1 skips the check.

    Returns:
      rows: ``[n_rows, max_seq_len + 1]`` packed token ids.
      token_bytes: ``[vocab_size]`` UTF-8 byte length per token id.

    Raises:
      FileNotFoundError: If the split or its metadata is missing.
      ValueError: If the arrays disagree with each other or with the declared
        geometry.

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
    raw_token_bytes = np.load(path / "all__token_bytes.npy")

    # The split's OWN metadata is checked unconditionally: a caller that never
    # declares geometry still must not be handed arrays contradicting the file
    # beside them. The caller's declaration, when given, is then checked
    # against that verified metadata rather than against the arrays -- so every
    # path through this function validates, and none of them only-sometimes.
    for field in ("vocab_size", "max_seq_len", "token_bytes_sha256"):
        if field not in metadata:
            raise ValueError(
                f"{path}/dataset.json declares no {field!r}; it predates the "
                "current preparer, so what it holds cannot be established. "
                "Re-prepare the split.",
            )
    observed = token_bytes_fingerprint(raw_token_bytes)
    if metadata["token_bytes_sha256"] != observed:
        raise ValueError(
            f"{path} records byte-table fingerprint "
            f"{metadata['token_bytes_sha256']} but its table hashes to "
            f"{observed}; the score's denominator changed, so a number measured "
            "here is not comparable with one measured before. Re-prepare.",
        )
    # Prepared as uint16, which torch cannot hold; int32 is the smallest dtype
    # that represents the whole vocabulary and still indexes an embedding.
    raw_rows = np.load(path / "all__tokens.npy")
    if raw_rows.ndim != 2 or raw_rows.shape[1] < 2:
        raise ValueError(
            f"{path}/all__tokens.npy has shape {raw_rows.shape}; it must be "
            "two-dimensional with at least two columns, since inputs and "
            "targets are one row offset by one.",
        )
    rows = torch.from_numpy(raw_rows.astype(np.int32)).to(resolved)
    token_bytes = torch.from_numpy(raw_token_bytes.astype(np.int32)).to(resolved)

    declared_vocab = int(metadata["vocab_size"])
    declared_seq = int(metadata["max_seq_len"])
    if token_bytes.shape[0] != declared_vocab:
        raise ValueError(
            f"{path} declares vocab_size {declared_vocab} but its byte "
            f"table holds {token_bytes.shape[0]} entries.",
        )
    if rows.shape[1] != declared_seq + 1:
        raise ValueError(
            f"{path} declares max_seq_len {declared_seq}, so its rows must "
            f"hold {declared_seq + 1} tokens; they hold {rows.shape[1]}.",
        )
    largest = int(rows.max()) if rows.numel() else 0
    if largest >= declared_vocab:
        raise ValueError(
            f"{path} holds token id {largest}, outside its own declared "
            f"vocab_size {declared_vocab}; it would index past the embedding.",
        )
    # The caller's declaration, against the metadata just verified. Without it
    # a disagreement surfaces inside the forward, naming only the model's side
    # and never the directory the rows came from.
    if max_seq_len > 0 and declared_seq != max_seq_len:
        raise ValueError(
            f"{path} holds rows of {declared_seq} tokens but the model "
            f"declares max_seq_len {max_seq_len}; prepare the data with "
            f"--max-seq-len {max_seq_len}, or set the model to {declared_seq}.",
        )
    if vocab_size > 0 and declared_vocab != vocab_size:
        raise ValueError(
            f"{path} was prepared for vocab_size {declared_vocab} but the "
            f"model declares {vocab_size}.",
        )
    logger.info(
        "nanochat %r: %d rows of %d tokens",
        split,
        rows.shape[0],
        rows.shape[1] - 1,
    )
    return rows, token_bytes
