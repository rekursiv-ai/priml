"""Token rows packed exactly as the reference recipe packs them.

A prepared directory holds the source corpus and the vocabulary fitted on it::

    shard_00000.parquet ... shard_00007.parquet
    tokenizer/tokenizer.pkl           pickled tiktoken encoding
    tokenizer/token_bytes.npy         [vocab_size] UTF-8 byte length per id
    tokenizer/tokenizer_recipe.json   what the vocabulary was fitted on

Rows are packed HERE, at read time, rather than written to disk by the
preparer. The packing is a stateful stream, and reproducing the reference means
reproducing that stream rather than producing an equivalent-looking arrangement
of the same tokens: every row begins with the document-start token, documents
are placed largest-fits-first out of a thousand-document buffer, and when
nothing fits the space that remains the shortest buffered document is cropped to
fill it exactly. Utilization is 100% and there is no padding, so every position
carries a real target and the loss needs no mask.

Which document is chosen depends on what the buffer HOLDS, and the buffer is
refilled 128 documents at a time -- so the refill granularity is part of the
packing rather than an implementation detail. So is the buffer's low-water mark.

``token_bytes.npy`` is what makes the score comparable across tokenizers.
Cross-entropy per TOKEN falls simply by making tokens larger, so the metric
divides by bytes instead; a token's byte length is a property of the vocabulary
and is therefore recorded beside it. Special tokens are zero, excluding them
from the denominator.

``scripts/prepare_data.py`` downloads the shards and fits the vocabulary; this
module only reads them, so the tokenizer TRAINER is needed by the script alone.

References:
    https://github.com/karpathy/autoresearch
      ``prepare.py``, commit b11d6f283f866eb7e10fb776a4b8553fef873fd5.

"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Self, override

import hashlib
import json
import logging
import pickle

from configgle import Fig
from torch import Tensor

import numpy as np
import torch

from priml.paths import resolve_working_dir
from priml.runtime import get_device
from priml.timer import CheckpointableStepTimer


if TYPE_CHECKING:
    # Annotation-only: unpickling the encoding imports tiktoken itself.
    import tiktoken


logger = logging.getLogger(__name__)

IGNORED_TARGET: Final = -1
"""Target the loss is told to skip.

No row the packer emits carries it -- packing leaves no padding to mark. It is
stated because the reference states it (``train.py``: ``ignore_index=-1``), so
a fork that introduces padding inherits the same marker rather than choosing
one. It is NOT self-excluding: a negative index reads the byte table from the
BACK and lands on a real length, so anything marked with it must also be
excluded from the metric by ``valid_count``."""

DOCUMENTS_PER_REFILL: Final = 128
"""Documents encoded per buffer refill.

Part of the packing, not a batching convenience: best-fit chooses the largest
document that fits out of whatever the buffer currently holds, so how many
arrive at a time decides which document that is."""


class NanoChatData:
    """Prepared shards and vocabulary, served as packed ``media`` batches.

    Batches carry ``media`` (inputs) and ``label`` (the same rows shifted by
    one). The training stream is infinite and the evaluation stream is capped at
    a fixed token count, so every candidate is scored on the identical prefix of
    the pinned validation shard.

    Raises:
      FileNotFoundError: If the shards or the vocabulary are absent. Run
        ``uv --quiet run --frozen python -m
        priml.baselines.nanochat.scripts.prepare_data`` first.

    """

    class Config(Fig["NanoChatData"]):
        """Where the corpus lives, and how batches are drawn from it."""

        base_dir: Path | str | None = None
        """Resource root supplied during parent finalization."""

        working_dir: Path | str = "/datasets/nanochat"
        """Directory holding the ``shard_*.parquet`` files.

        Resolved beneath ``base_dir`` at finalize, so it names a location within
        the resource root rather than an absolute filesystem path."""

        tokenizer_dir: Path | str = ""
        """Directory holding the fitted vocabulary; empty is ``<data>/tokenizer``."""

        num_train_shards: int = 7
        """Shards forming the training split, numbered from zero."""

        val_shard: int = 7
        """The pinned validation shard; no run trains on it."""

        batch_size: int = 32
        """Rows per training batch."""

        eval_batch_size: int = 128
        """Rows per evaluation batch.

        Fixed rather than tracking ``batch_size``: the scored token count is
        fixed, so the batch width decides HOW MANY batches are scored and, with
        the packer's stream, which rows fall in them. Training's batch follows
        device memory, and a score whose row set moved with the card it ran on
        would not be the comparison this baseline exists to make."""

        eval_tokens: int = 40 * 524_288
        """Validation tokens to score; the reference's fixed evaluation size."""

        buffer_size: int = 1_000
        """Documents held for best-fit selection before a row is packed."""

        device: str = "auto"
        """Device batches land on ("auto" picks the best)."""

        vocab_size: int = -1
        """Vocabulary the fitted tokenizer must report; -1 skips the check.

        Declared rather than read from disk: a config must build without
        touching the filesystem. The model pushes its own value down, and
        ``__init__`` verifies the tokenizer against it -- so a prepared
        directory that disagrees fails at load, naming both numbers, instead of
        surfacing later as an out-of-range embedding index."""

        max_seq_len: int = 2_048
        """Tokens per packed row.

        A row is packed to ``max_seq_len + 1`` tokens, since the inputs and the
        targets are the row offset by one."""

        @override
        def finalize(self) -> Self:
            self.working_dir = resolve_working_dir(self.base_dir, self.working_dir)
            if not self.tokenizer_dir:
                self.tokenizer_dir = Path(self.working_dir) / "tokenizer"
            return super().finalize()

    def __init__(self, config: Config) -> None:
        if config.batch_size <= 0:
            raise ValueError(f"batch_size must be positive; got {config.batch_size}.")
        if config.eval_batch_size <= 0:
            raise ValueError(
                f"eval_batch_size must be positive; got {config.eval_batch_size}.",
            )
        if config.max_seq_len < 2:
            raise ValueError(
                f"max_seq_len must be at least two; got {config.max_seq_len}.",
            )
        if config.buffer_size <= 0:
            raise ValueError(f"buffer_size must be positive; got {config.buffer_size}.")
        tokens_per_eval_batch = config.eval_batch_size * config.max_seq_len
        if config.eval_tokens <= 0 or config.eval_tokens % tokens_per_eval_batch:
            raise ValueError(
                f"eval_tokens={config.eval_tokens} must be positive and a whole "
                f"number of eval batches of {tokens_per_eval_batch} tokens; "
                "otherwise the reported score covers a different token count "
                "than it names.",
            )
        self.config = config
        self.device = get_device(config.device)
        self.dataset_dir = Path(config.working_dir)
        self.batch_size = config.batch_size
        self.eval_batch_size = config.eval_batch_size
        self.timer_epoch = CheckpointableStepTimer()
        """Passes over the corpus; ticked by the loop when the shards wrap.

        A budgeted run rarely reaches one -- the stream wraps rather than
        ending, and the recipe stops on time long before the corpus is
        exhausted."""

        self.train_paths = _shard_paths(
            self.dataset_dir,
            indices=range(config.num_train_shards),
        )
        self.val_paths = _shard_paths(self.dataset_dir, indices=[config.val_shard])
        self.tokenizer = Tokenizer.from_directory(Path(config.tokenizer_dir))
        if 0 < config.vocab_size != self.tokenizer.vocab_size:
            raise ValueError(
                f"the fitted vocabulary holds {self.tokenizer.vocab_size} tokens "
                f"but the model declares vocab_size {config.vocab_size}; prepare "
                f"with --vocab-size {config.vocab_size}, or set the model to "
                f"{self.tokenizer.vocab_size}.",
            )
        self.token_bytes: Tensor = torch.from_numpy(
            self.tokenizer.token_bytes.astype(np.int32),
        ).to(self.device)
        # The live training stream, held so ``state_dict`` reports how far the
        # corpus was actually consumed. Read from the stream rather than
        # mirrored into a counter here: a copy updated at the call sites would
        # be right only where someone remembered to update it, and the resume
        # guard that reads it would then pass on exactly the runs it exists to
        # refuse.
        self._live: _PackedStream | None = None
        logger.info(
            "nanochat: %d train shards, val shard %d, vocab %d",
            len(self.train_paths),
            config.num_train_shards,
            self.tokenizer.vocab_size,
        )

    def train_dataloader(self) -> _PackedStream:
        """Build the training stream: the corpus, packed, without end.

        Unbounded because the run stops on its time budget rather than on a pass
        over the data. A row is a full context by construction, so a pass is not
        a meaningful boundary -- and cutting the stream at one would end the
        packer's document buffer mid-row, which is a different row than the
        reference produces there.
        """
        self._live = _PackedStream(
            paths=self.train_paths,
            tokenizer=self.tokenizer,
            token_bytes=self.token_bytes,
            batch_size=self.batch_size,
            max_seq_len=self.config.max_seq_len,
            buffer_size=self.config.buffer_size,
            device=self.device,
            max_batches=None,
        )
        return self._live

    def eval_dataloader(self) -> _PackedStream:
        """Build the evaluation stream: the pinned shard, from its start.

        Rebuilt per evaluation rather than continued, so every score covers the
        identical tokens: a stream that carried on would score a later part of
        the shard each time and report the difference as progress.
        """
        return _PackedStream(
            paths=self.val_paths,
            tokenizer=self.tokenizer,
            token_bytes=self.token_bytes,
            batch_size=self.eval_batch_size,
            max_seq_len=self.config.max_seq_len,
            buffer_size=self.config.buffer_size,
            device=self.device,
            max_batches=self.config.eval_tokens
            // (self.eval_batch_size * self.config.max_seq_len),
        )

    def state_dict(self) -> dict[str, Any]:
        """Snapshot how far the training stream has advanced."""
        return {
            "batches": self._live.served if self._live is not None else 0,
            "timer_epoch": self.timer_epoch.state_dict(),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Refuse to resume a stream that cannot be positioned.

        The packer's state is a document buffer built by tokenizing the corpus
        from its start, so there is no seek: reaching batch N means re-doing the
        work of N batches. Restarting instead would silently retrain on the
        opening of the corpus while the schedules carried on from where the
        checkpoint left them.

        Raises:
          ValueError: The checkpoint had advanced the stream.

        """
        served = int(state_dict.get("batches", 0))
        if served:
            raise ValueError(
                f"this checkpoint had served {served} batches, and the packed "
                "stream cannot be positioned without re-tokenizing the corpus "
                "up to that point; resuming would silently replay the start of "
                "the data. Start a fresh run.",
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


class Tokenizer:
    """The fitted vocabulary, with the byte table that scores it.

    Attributes:
      encoding: The tiktoken encoding written by the preparer.
      bos_token_id: Id of the document-start token every row begins with.
      token_bytes: ``[vocab_size]`` UTF-8 byte length per token id.

    """

    def __init__(
        self,
        encoding: tiktoken.Encoding,
        *,
        bos_token: str,
        token_bytes: np.ndarray,
    ) -> None:
        self.encoding = encoding
        self.bos_token_id = encoding.encode_single_token(bos_token)
        self.token_bytes = token_bytes

    @classmethod
    def from_directory(cls, tokenizer_dir: Path) -> Tokenizer:
        """Load the vocabulary, verified against the recipe beside it.

        Args:
          tokenizer_dir: Directory the preparer wrote.

        Returns:
          tokenizer: The fitted vocabulary and its byte table.

        Raises:
          FileNotFoundError: The vocabulary is absent.
          ValueError: The byte table contradicts its recorded fingerprint, or
            holds a length no score can use.

        """
        pickled = tokenizer_dir / "tokenizer.pkl"
        recipe_path = tokenizer_dir / "tokenizer_recipe.json"
        if not pickled.is_file() or not recipe_path.is_file():
            raise FileNotFoundError(
                f"no prepared nanochat vocabulary at {tokenizer_dir}; build it "
                "with `uv --quiet run --frozen python -m "
                "priml.baselines.nanochat.scripts.prepare_data`.",
            )
        with pickled.open("rb") as file:
            encoding = pickle.load(file)  # noqa: S301 -- our own prepared artifact
        recipe = json.loads(recipe_path.read_text())
        for field in ("bos_token", "token_bytes_sha256"):
            if field not in recipe:
                raise ValueError(
                    f"{recipe_path} declares no {field!r}; it predates the "
                    "current preparer, so what it holds cannot be established. "
                    "Re-prepare the vocabulary.",
                )
        raw = np.load(tokenizer_dir / "token_bytes.npy")
        if raw.ndim != 1:
            raise ValueError(
                f"{tokenizer_dir}/token_bytes.npy has shape {raw.shape}; it must "
                "be one-dimensional, one byte length per token id.",
            )
        # Before the fingerprint too: it canonicalizes to int64, so a float
        # table hashes identically to its own truncation and the identity check
        # cannot tell the two apart.
        if not np.issubdtype(raw.dtype, np.integer):
            raise ValueError(
                f"{tokenizer_dir}/token_bytes.npy has dtype {raw.dtype}; it must "
                "hold integers, since a fractional length would silently change "
                "the score's denominator.",
            )
        observed = token_bytes_fingerprint(raw)
        if recipe["token_bytes_sha256"] != observed:
            raise ValueError(
                f"{tokenizer_dir} records byte-table fingerprint "
                f"{recipe['token_bytes_sha256']} but its table hashes to "
                f"{observed}; the score's denominator changed, so a number "
                "measured here is not comparable with one measured before. "
                "Re-prepare.",
            )
        # The metric's scoring mask is ``lengths > 0``, so a negative length
        # drops that token from BOTH sums -- a token silently excluded from the
        # score rather than a rejected table.
        if int(raw.min(initial=0)) < 0:
            raise ValueError(
                f"{tokenizer_dir} holds a negative byte length; the score's "
                "denominator counts bytes, and a negative one would silently "
                "drop its token from the measurement.",
            )
        tokenizer = cls(
            encoding,
            bos_token=recipe["bos_token"],
            token_bytes=raw,
        )
        if raw.shape[0] != tokenizer.vocab_size:
            raise ValueError(
                f"{tokenizer_dir} fitted {tokenizer.vocab_size} tokens but its "
                f"byte table holds {raw.shape[0]} entries.",
            )
        return tokenizer

    @property
    def vocab_size(self) -> int:
        """Tokens the vocabulary holds, reserved tokens included."""
        return int(self.encoding.n_vocab)

    def encode_batch(
        self, texts: list[str], *, num_threads: int = 8
    ) -> list[list[int]]:
        """Encode documents, prepending the document-start token to each.

        Args:
          texts: Raw document strings.
          num_threads: Threads for the batched encoder.

        Returns:
          token_lists: One BOS-prefixed token id list per document.

        """
        token_lists = self.encoding.encode_ordinary_batch(
            texts,
            num_threads=num_threads,
        )
        for row in token_lists:
            row.insert(0, self.bos_token_id)
        return token_lists


class _PackedStream:
    """One split, packed into fixed-width rows and batched.

    The yielded ``media`` and ``label`` are VIEWS of a reused buffer, as the
    reference's are: a batch is consumed before the next is requested, and
    copying instead would add a per-step allocation the recipe's throughput was
    not measured with. A consumer that must retain a batch clones it.
    """

    def __init__(
        self,
        *,
        paths: list[Path],
        tokenizer: Tokenizer,
        token_bytes: Tensor,
        batch_size: int,
        max_seq_len: int,
        buffer_size: int,
        device: torch.device,
        max_batches: int | None,
    ) -> None:
        self.paths = paths
        self.tokenizer = tokenizer
        self.token_bytes = token_bytes
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.buffer_size = buffer_size
        self.device = device
        self.max_batches = max_batches
        self.served = 0
        """Batches drawn from this stream, across every iteration of it."""

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Yield ``(input, target)`` pairs, packed to full rows without padding.

        Rows are cut from a document stream rather than from a token stream: a
        row begins with the document-start token, takes the largest buffered
        document that fits the space remaining, and -- when none does -- crops
        the shortest to fill it exactly. So no position is padding, no target is
        masked, and the token count per optimizer step never moves.
        """
        rows = self.max_seq_len + 1
        documents = _document_batches(self.paths)
        buffer: list[list[int]] = []

        pinned = self.device.type == "cuda"
        row_buffer = torch.empty(self.batch_size, rows, dtype=torch.long)
        # One staging allocation holding inputs and targets back to back, so a
        # batch reaches the device in ONE transfer rather than two.
        width = self.batch_size * self.max_seq_len
        staged = torch.empty(2 * width, dtype=torch.long, pin_memory=pinned)
        resident = (
            torch.empty(2 * width, dtype=torch.long, device=self.device)
            if pinned
            else staged
        )
        # Viewed as rows on BOTH sides of the transfer. Copying into a flat
        # staging buffer instead would force ``row_buffer[:, :-1]`` -- a strided
        # view -- to be materialized contiguous first, which is a whole extra
        # batch-sized allocation and copy per step; ``copy_`` between two
        # equally-shaped tensors handles the stride itself.
        staged_media = staged[:width].view(self.batch_size, self.max_seq_len)
        staged_label = staged[width:].view(self.batch_size, self.max_seq_len)
        media = resident[:width].view(self.batch_size, self.max_seq_len)
        label = resident[width:].view(self.batch_size, self.max_seq_len)

        drawn = 0
        while self.max_batches is None or drawn < self.max_batches:
            for index in range(self.batch_size):
                position = 0
                while position < rows:
                    while len(buffer) < self.buffer_size:
                        buffer.extend(self.tokenizer.encode_batch(next(documents)))
                    position = _pack_row(row_buffer[index], buffer, position=position)
            staged_media.copy_(row_buffer[:, :-1])
            staged_label.copy_(row_buffer[:, 1:])
            if pinned:
                resident.copy_(staged, non_blocking=True)
            drawn += 1
            self.served += 1
            yield {
                "media": media,
                "label": label,
                "token_bytes": self.token_bytes,
                "valid_count": self.batch_size,
            }

    def __len__(self) -> int:
        """Batches in one evaluation pass.

        Raises:
          TypeError: The stream is unbounded, which is the training case. The
            loop asks and falls back to counting, so this states the absence
            rather than inventing a horizon.

        """
        if self.max_batches is None:
            raise TypeError("the training stream is unbounded and has no length.")
        return self.max_batches


def _pack_row(row: Tensor, buffer: list[list[int]], *, position: int) -> int:
    """Place one document into ``row`` at ``position``; return the new position.

    Args:
      row: The row being filled, ``max_seq_len + 1`` wide.
      buffer: Encoded documents available for selection; the chosen one is
        removed.
      position: Where the next document starts.

    Returns:
      position: Where the document after it starts.

    """
    remaining = row.numel() - position
    best_index = -1
    best_length = 0
    for index, document in enumerate(buffer):
        if best_length < len(document) <= remaining:
            best_index = index
            best_length = len(document)
    if best_index >= 0:
        document = buffer.pop(best_index)
        row[position : position + len(document)] = torch.tensor(
            document,
            dtype=torch.long,
        )
        return position + len(document)
    # Nothing fits, so the space cannot be filled by a whole document and
    # leaving it empty would be padding. The SHORTEST is cropped: it is the one
    # losing the least of itself to the cut.
    shortest = min(range(len(buffer)), key=lambda index: len(buffer[index]))
    document = buffer.pop(shortest)
    row[position : position + remaining] = torch.tensor(
        document[:remaining],
        dtype=torch.long,
    )
    return position + remaining


def _document_batches(paths: list[Path]) -> Iterator[list[str]]:
    """Yield document batches from parquet shards, wrapping at the end.

    Unbounded: the stream is what a budgeted run draws from, and a run that
    outlasts the corpus continues from its start rather than ending.
    """
    # Imported here rather than at module scope: parquet is the corpus's own
    # format, and nothing but this reader touches it.
    from pyarrow import parquet  # noqa: PLC0415 -- corpus-only dependency

    while True:
        for path in paths:
            shard = parquet.ParquetFile(path)
            for group in range(shard.num_row_groups):
                texts: list[str] = (
                    shard.read_row_group(group).column("text").to_pylist()
                )
                for start in range(0, len(texts), DOCUMENTS_PER_REFILL):
                    yield texts[start : start + DOCUMENTS_PER_REFILL]


def _shard_paths(directory: Path, *, indices: range | list[int]) -> list[Path]:
    """Return the named shards, in index order.

    Args:
      directory: Where the corpus was prepared.
      indices: Shard numbers the split is made of.

    Returns:
      paths: One path per index.

    Raises:
      FileNotFoundError: A named shard is absent, reported by name so the
        missing one is identifiable rather than the directory merely being
        called unprepared.

    """
    paths = [directory / f"shard_{index:05d}.parquet" for index in indices]
    missing = [path.name for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"{directory} is missing {missing}; prepare the corpus with "
            "`uv --quiet run --frozen python -m "
            "priml.baselines.nanochat.scripts.prepare_data`.",
        )
    return paths
