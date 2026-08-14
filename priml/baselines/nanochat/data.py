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
from typing import Any, Final, Self, override

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

_IGNORED_TARGET: Final = -1
"""Target marking a padded evaluation row.

Negative, so the metric's byte-table lookup cannot reach it and the row leaves
both of the score's sums."""


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

        eval_batch_size: int = 16
        """Rows per evaluation batch.

        Fixed rather than tracking ``batch_size``: a short final batch is
        dropped, so the batch width decides WHICH rows are scored. Training's
        batch follows device memory, and a score whose row set moved with the
        card it ran on would not be the comparison this baseline exists to
        make."""

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
        if config.eval_batch_size <= 0:
            raise ValueError(
                f"eval_batch_size must be positive; got {config.eval_batch_size}.",
            )
        if config.num_eval_rows is not None and config.num_eval_rows <= 0:
            raise ValueError(
                f"num_eval_rows must be positive; got {config.num_eval_rows}.",
            )
        # Deliberately NOT a divisibility rule on the cap. An earlier revision
        # required ``num_eval_rows % eval_batch_size == 0`` because the
        # remainder would be dropped -- but the UNCAPPED pass drops the
        # identical remainder, so the rule policed the proxy eval while waiving
        # the number actually reported. The remainder is a property of the row
        # count, not of the cap, so evaluation scores its tail instead (see
        # ``_TokenBatches.__iter__``) and neither case silently loses rows.
        self.config = config
        self.dataset_dir = Path(config.working_dir)
        self.batch_size = config.batch_size
        self.eval_batch_size = config.eval_batch_size
        self._passes = 0
        self._live: _TokenBatches | None = None
        # A prepared split is immutable, and the loop builds a fresh eval
        # loader per evaluation. Re-reading would re-verify bytes that cannot
        # have changed: measured at 107 ms against the 46 ms the eval pass
        # itself takes, so the read costs more than twice the work it feeds.
        #
        # The cost is memory: each split stays resident on the training device
        # for the whole run rather than only during an eval. Token ids are
        # int32, so a 300k-row split at 2048 context is ~2.5 GiB -- material
        # beside the weights and activations on a small card, and the reason
        # ``num_eval_rows`` exists.
        self._splits: dict[str, _PreparedSplit] = {}

    def _split(self, name: str) -> _PreparedSplit:
        """The named split, verified once and retained."""
        if name not in self._splits:
            self._splits[name] = _PreparedSplit(
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
            device=self.config.device,
            batch_size=self.batch_size,
            shuffle=True,
            score_every_row=False,
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
            device=self.config.device,
            batch_size=self.eval_batch_size,
            shuffle=False,
            score_every_row=True,
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


class _PreparedSplit:
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
        self.name = directory.name
        self.vocab_size = int(metadata["vocab_size"])
        self.max_seq_len = int(metadata["max_seq_len"])

        raw_token_bytes = np.load(directory / "all__token_bytes.npy")
        if raw_token_bytes.ndim != 1:
            raise ValueError(
                f"{directory}/all__token_bytes.npy has shape "
                f"{raw_token_bytes.shape}; it must be one-dimensional, one "
                "byte length per token id.",
            )
        # Before the fingerprint too: it canonicalizes to int64, so a float
        # table hashes identically to its own truncation and the identity check
        # cannot tell the two apart.
        _require_integers(
            raw_token_bytes,
            what=f"{directory}/all__token_bytes.npy",
        )
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
        # Checked BEFORE the cast: ``astype(np.int32)`` truncates rather than
        # refusing, so a float array would silently become different tokens --
        # 1.9 read as token 1 -- and every range check below would then pass.
        _require_integers(raw_rows, what=f"{directory}/all__tokens.npy")
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
        # Both ends of the id range: a negative id indexes an embedding from
        # the BACK, which is a silently wrong row rather than a failure.
        largest = int(self.rows.max()) if self.rows.numel() else 0
        smallest = int(self.rows.min()) if self.rows.numel() else 0
        if largest >= self.vocab_size or smallest < 0:
            raise ValueError(
                f"{directory} holds token ids in [{smallest}, {largest}], "
                f"outside its own declared vocab_size {self.vocab_size}; they "
                "would index past the embedding or wrap to its end.",
            )
        # The metric's scoring mask is ``lengths > 0``, so a negative length
        # drops that token from BOTH sums -- a token silently excluded from the
        # score rather than a rejected table.
        if int(self.token_bytes.min()) < 0:
            raise ValueError(
                f"{directory} holds a negative byte length; the score's "
                "denominator counts bytes, and a negative one would silently "
                "drop its token from the measurement.",
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


class _TokenBatches:
    """One split, resident on device, iterated in fixed-size batches."""

    def __init__(
        self,
        *,
        prepared: _PreparedSplit,
        device: torch.device | str,
        batch_size: int,
        shuffle: bool,
        score_every_row: bool,
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
        self.score_every_row = score_every_row
        self.seed = seed
        self.passes = passes
        # A short batch is dropped, so too few rows yields NOTHING -- and an
        # empty stream is invisible until it surfaces far away: training as a
        # generic epoch-reset error, evaluation as a metric with no samples.
        # Both are the same misconfiguration, so it is named once, here, where
        # the row count and the batch size are both in hand.
        available = int(self.rows.shape[0])
        if available < batch_size and not score_every_row:
            raise ValueError(
                f"the {prepared.name!r} split yields no batches: {available} rows "
                f"available (of {rows.shape[0]}) against a batch size of "
                f"{batch_size}. Lower the batch size, raise the row cap, or "
                "prepare more data.",
            )

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Yield every row once as an ``(input, target)`` pair.

        Training DROPS a short final batch: the token count per optimizer step
        is what the recipe is tuned against, and a narrower step would be the
        one place in the run where it moves.

        Evaluation SCORES it. A dropped tail means the reported number covers
        fewer rows than the split holds -- silently, and by an amount set by
        the batch width -- so the short batch is padded to full width and the
        padding is marked with ``-1`` targets, which the metric's byte table
        excludes from both of its sums.
        """
        count = self.rows.shape[0]
        if self.shuffle:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(self.seed + self.passes)
            order = torch.randperm(count, device=self.device, generator=generator)
        else:
            order = torch.arange(count, device=self.device)
        self.passes += 1
        last = count if self.score_every_row else count - self.batch_size + 1
        for start in range(0, last, self.batch_size):
            index = order[start : start + self.batch_size]
            rows = self.rows[index]
            valid = int(rows.shape[0])
            if valid < self.batch_size:
                # Padded with the FIRST row rather than zeros, so the model
                # runs on a real context; the -1 targets are what remove it
                # from the score.
                pad = self.rows[:1].expand(self.batch_size - valid, -1)
                rows = torch.cat([rows, pad])
            media = rows[:, :-1].long()
            label = rows[:, 1:].long()
            if valid < self.batch_size:
                label = label.clone()
                label[valid:] = _IGNORED_TARGET
            yield {
                "media": media,
                "label": label,
                "token_bytes": self.token_bytes,
                "valid_count": valid,
            }

    def __len__(self) -> int:
        """Batches per pass, counting a padded final batch when scored."""
        rows = self.rows.shape[0]
        if self.score_every_row:
            return math.ceil(rows / self.batch_size)
        return math.floor(rows / self.batch_size)


def _require_integers(array: np.ndarray, *, what: str) -> None:
    """Raise unless the array holds integers.

    Args:
      array: Array loaded from a prepared split.
      what: Path named in the message.

    Raises:
      ValueError: The dtype is not integral. Casting instead would TRUNCATE --
        a token id of 1.9 becomes 1, a different token entirely -- and every
        later range check would pass on the truncated values.

    """
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError(
            f"{what} has dtype {array.dtype}; it must hold integers, since "
            "casting a fractional value would silently change which token it "
            "names.",
        )
