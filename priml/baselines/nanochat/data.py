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
        # A prepared split is immutable, and the loop builds a fresh eval
        # loader per evaluation. Re-reading would re-verify bytes that cannot
        # have changed: measured at 107 ms against the 46 ms the eval pass
        # itself takes, so the read costs more than twice the work it feeds.
        self._splits: dict[str, PreparedSplit] = {}

    def _split(self, name: str) -> PreparedSplit:
        """The named split, verified once and retained."""
        if name not in self._splits:
            self._splits[name] = PreparedSplit(
                self.dataset_dir / name,
                device=self.config.device,
            )
        return self._splits[name]

    def train_dataloader(self) -> _TokenBatches:
        """Build the re-iterable training stream."""
        # Snapshot any prior stream's counter first, so re-creating the loader
        # continues the sequence rather than restarting it.
        if self._live is not None:
            self._passes = self._live.passes
        stream = _TokenBatches(
            prepared=self._split("train"),
            name="train",
            device=self.config.device,
            batch_size=self.batch_size,
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
            prepared=self._split("val"),
            name="val",
            device=self.config.device,
            batch_size=self.eval_batch_size,
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
        prepared: PreparedSplit,
        name: str,
        device: torch.device | str,
        batch_size: int,
        shuffle: bool,
        num_rows: int | None,
        seed: int,
        passes: int,
        vocab_size: int,
        max_seq_len: int,
    ) -> None:
        prepared.agrees_with(vocab_size=vocab_size, max_seq_len=max_seq_len)
        self.device = get_device(device)
        rows = prepared.rows
        self.rows: Tensor = rows if num_rows is None else rows[:num_rows]
        self.token_bytes = prepared.token_bytes
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
                f"the {name!r} split yields no batches: {available} rows "
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


class PreparedSplit:
    """One split's arrays, verified against the metadata written beside them.

    Constructing this IS the guarantee. The alternative -- returning bare
    tensors and checking them at each use -- is what this class replaces: those
    checks accumulated as a list of conditionals inside the loader, each having
    to re-decide when it applied, and every one of them was at some point
    written to skip precisely the case it existed to catch.

    So there are no optional checks here and nothing to opt into. A split that
    disagrees with its own ``dataset.json`` cannot be built, and a caller
    holding one of these need not ask whether it was validated.

    Attributes:
      rows: ``[n_rows, max_seq_len + 1]`` packed token ids.
      token_bytes: ``[vocab_size]`` UTF-8 byte length per token id.
      vocab_size: Vocabulary the split declares.
      max_seq_len: Context length the split declares.

    """

    def __init__(
        self,
        directory: Path,
        *,
        device: torch.device | str,
    ) -> None:
        if not (directory / "dataset.json").is_file():
            raise FileNotFoundError(
                f"no prepared nanochat data at {directory}; build it with "
                "`uv --quiet run --frozen python -m "
                "priml.baselines.nanochat.scripts.prepare_data`.",
            )
        metadata = json.loads((directory / "dataset.json").read_text())
        for field in ("vocab_size", "max_seq_len", "token_bytes_sha256"):
            if field not in metadata:
                raise ValueError(
                    f"{directory}/dataset.json declares no {field!r}; it "
                    "predates the current preparer, so what it holds cannot be "
                    "established. Re-prepare the split.",
                )
        self.vocab_size = int(metadata["vocab_size"])
        self.max_seq_len = int(metadata["max_seq_len"])

        raw_token_bytes = np.load(directory / "all__token_bytes.npy")
        observed = token_bytes_fingerprint(raw_token_bytes)
        if metadata["token_bytes_sha256"] != observed:
            raise ValueError(
                f"{directory} records byte-table fingerprint "
                f"{metadata['token_bytes_sha256']} but its table hashes to "
                f"{observed}; the score's denominator changed, so a number "
                "measured here is not comparable with one measured before. "
                "Re-prepare.",
            )
        raw_rows = np.load(directory / "all__tokens.npy")
        if raw_rows.ndim != 2 or raw_rows.shape[1] < 2:
            raise ValueError(
                f"{directory}/all__tokens.npy has shape {raw_rows.shape}; it "
                "must be two-dimensional with at least two columns, since "
                "inputs and targets are one row offset by one.",
            )
        resolved = get_device(device)
        # Prepared as uint16, which torch cannot hold; int32 is the smallest
        # dtype spanning the vocabulary that still indexes an embedding.
        self.rows: Tensor = torch.from_numpy(raw_rows.astype(np.int32)).to(resolved)
        self.token_bytes: Tensor = torch.from_numpy(
            raw_token_bytes.astype(np.int32),
        ).to(resolved)

        if self.token_bytes.shape[0] != self.vocab_size:
            raise ValueError(
                f"{directory} declares vocab_size {self.vocab_size} but its "
                f"byte table holds {self.token_bytes.shape[0]} entries.",
            )
        if self.rows.shape[1] != self.max_seq_len + 1:
            raise ValueError(
                f"{directory} declares max_seq_len {self.max_seq_len}, so its "
                f"rows must hold {self.max_seq_len + 1} tokens; they hold "
                f"{self.rows.shape[1]}.",
            )
        largest = int(self.rows.max()) if self.rows.numel() else 0
        if largest >= self.vocab_size:
            raise ValueError(
                f"{directory} holds token id {largest}, outside its own "
                f"declared vocab_size {self.vocab_size}; it would index past "
                "the embedding.",
            )
        logger.info(
            "nanochat %r: %d rows of %d tokens",
            directory.name,
            self.rows.shape[0],
            self.max_seq_len,
        )

    def agrees_with(self, *, vocab_size: int, max_seq_len: int) -> None:
        """Raise unless a caller's declared geometry matches this split.

        Args:
          vocab_size: Vocabulary the caller declares; -1 declares nothing.
          max_seq_len: Context the caller declares; -1 declares nothing.

        Raises:
          ValueError: The caller's declaration contradicts the split. Without
            this the disagreement surfaces inside the forward, naming only the
            caller's side and never the directory the rows came from.

        """
        if max_seq_len > 0 and self.max_seq_len != max_seq_len:
            raise ValueError(
                f"the split holds rows of {self.max_seq_len} tokens but the "
                f"model declares max_seq_len {max_seq_len}; prepare the data "
                f"with --max-seq-len {max_seq_len}, or set the model to "
                f"{self.max_seq_len}.",
            )
        if vocab_size > 0 and self.vocab_size != vocab_size:
            raise ValueError(
                f"the split was prepared for vocab_size {self.vocab_size} but "
                f"the model declares {vocab_size}.",
            )


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
