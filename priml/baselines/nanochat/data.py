"""NanoChat online packing and immutable prepared token rows."""

from __future__ import annotations

from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Final,
    NotRequired,
    Protocol,
    Self,
    TypedDict,
    cast,
    override,
)

import contextlib
import hashlib
import logging
import math
import pickle
import queue
import threading

from configgle import Fig
from numpy.lib.npyio import NpzFile
from numpy.typing import NDArray
from torch import Tensor

import numpy as np
import torch

from priml.lib.custom_json import DictCodec, IntCodec, StrCodec, loads
from priml.paths import resolve_working_dir
from priml.runtime import get_device
from priml.timer import CheckpointableStepTimer


if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from pyarrow import parquet

    import tiktoken
else:
    from wrapt import lazy_import

    parquet = lazy_import("pyarrow.parquet")
    tiktoken = lazy_import("tiktoken")


logger = logging.getLogger(__name__)

IGNORED_TARGET: Final = -1
"""Target excluded by the loss. Metrics must mask it before byte-table indexing."""

DOCUMENTS_PER_REFILL: Final = 128
"""Documents encoded per refill; changing this changes best-fit packing order."""


class NanoChatBatch(TypedDict):
    """Packed token rows and the byte accounting needed to score them."""

    media: Tensor
    label: Tensor
    token_bytes: Tensor
    valid_count: int
    score_mask: NotRequired[Tensor]
    reference_bytes: NotRequired[int]
    literal_bytes: NotRequired[int]
    evaluation_batch: NotRequired[int]
    evaluation_batches: NotRequired[int]


class TokenEncoder(Protocol):
    """Encode documents with a leading document-start token."""

    def encode_batch(
        self,
        texts: list[str],
        *,
        num_threads: int = 8,
    ) -> list[list[int]]:
        """Return one BOS-prefixed token list for each input document."""
        ...


class NanoChatData:
    """Serve packed token batches from parquet shards or prepared arrays.

    Online training wraps; prepared training raises on exhaustion. Evaluation
    restarts at a fixed token count, which covers different text across tokenizers.
    """

    class Config(Fig["NanoChatData"]):
        """Configure corpus paths, tokenizer identity, and packing geometry."""

        base_dir: Path | str | None = None
        """Resource root supplied during parent finalization."""

        working_dir: Path | str = "/datasets/nanochat"
        """Parquet shard directory, resolved beneath ``base_dir``."""

        tokenizer_dir: Path | str = ""
        """Directory holding the fitted vocabulary; empty is ``<data>/tokenizer``."""

        prepared_train_manifest: Path | str = ""
        """Frozen token-row manifest; empty uses online parquet packing."""

        prepared_eval_manifest: Path | str = ""
        """Separate packed evaluation manifest, resolved beneath ``base_dir``."""

        reference_evaluation: ReferenceEvaluation.Config | None = None
        """Optional byte-matched reference replay; training remains unchanged."""

        num_train_shards: int = 7
        """Shards forming the training split, numbered from zero."""

        train_shard_indices: tuple[int, ...] = ()
        """Explicit training shards; empty uses the first ``num_train_shards``."""

        val_shard: int = 7
        """The pinned validation shard; no run trains on it."""

        batch_size: int = 32
        """Rows per training batch."""

        eval_batch_size: int = 128
        """Evaluation rows per batch; changing this can change packed row selection."""

        eval_tokens: int = 40 * 524_288
        """Physical validation token positions to score, including special tokens."""

        buffer_size: int = 1_000
        """Documents held for best-fit selection before a row is packed."""

        train_buffer_size: int | None = None
        """Training-only packing buffer; None inherits ``buffer_size``."""

        device: torch.device | str | None = "auto"
        """Device batches land on.

        ``"auto"`` probes the hardware, ``None`` defers to
        ``torch.get_default_device()``; see :func:`get_device`."""

        vocab_size: int = -1
        """Expected tokenizer vocabulary size; -1 skips validation at load time."""

        max_seq_len: int = 2_048
        """Input tokens per row; packing adds one token for shifted targets."""

        @override
        def finalize(self) -> Self:
            self.working_dir = resolve_working_dir(self.base_dir, self.working_dir)
            if not self.tokenizer_dir:
                self.tokenizer_dir = Path(self.working_dir) / "tokenizer"
            if self.prepared_train_manifest:
                self.prepared_train_manifest = resolve_working_dir(
                    self.base_dir,
                    self.prepared_train_manifest,
                )
            if self.prepared_eval_manifest:
                self.prepared_eval_manifest = resolve_working_dir(
                    self.base_dir,
                    self.prepared_eval_manifest,
                )
            return super().finalize()

    class StateDict(TypedDict):
        """Consumed training batches and persisted epoch timing."""

        batches: int
        timer_epoch: CheckpointableStepTimer.StateDict

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
        if config.train_buffer_size is not None and config.train_buffer_size <= 0:
            raise ValueError("train_buffer_size must be positive.")
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

        # Read consumption from the stream itself so the resume guard cannot
        # silently replay already-trained rows with advanced schedules.
        self._live: PackedTokenStream | None = None
        self._prepared: PreparedTokenRows | None = None
        self._tokenizer: Tokenizer | None = None
        self._reference = (
            config.reference_evaluation.make()
            if config.reference_evaluation is not None
            else None
        )
        if self._reference is not None and (
            self._reference.inputs.shape[1] != config.max_seq_len
            or self._reference.batch_size != config.eval_batch_size
            or (
                config.vocab_size > 0
                and self._reference.vocab_size != config.vocab_size
            )
        ):
            raise ValueError("Reference evaluation geometry differs from the model.")
        prepared_fields = (
            config.prepared_train_manifest,
            config.prepared_eval_manifest,
        )
        if any(prepared_fields) and not all(prepared_fields):
            raise ValueError("Prepared data requires both manifests.")
        if config.prepared_train_manifest:
            self._prepared = PreparedTokenRows(config)
            self.train_paths: list[Path] = []
            self.val_paths: list[Path] = []
            self.token_bytes = torch.from_numpy(
                self._prepared.token_bytes.astype(np.int32),
            ).to(self.device)
            return

        if not config.train_shard_indices and config.num_train_shards <= 0:
            raise ValueError(
                "num_train_shards must be positive without explicit shards.",
            )
        train_indices = list(
            config.train_shard_indices or range(config.num_train_shards),
        )
        if len(train_indices) != len(set(train_indices)):
            raise ValueError("train_shard_indices names a shard twice.")
        if config.val_shard in train_indices:
            raise ValueError("The validation shard must be excluded from training.")
        self.train_paths = _shard_paths(self.dataset_dir, indices=train_indices)
        self.val_paths = _shard_paths(self.dataset_dir, indices=[config.val_shard])
        self._tokenizer = Tokenizer.from_directory(Path(config.tokenizer_dir))
        if 0 < config.vocab_size != self.tokenizer.vocab_size:
            raise ValueError(
                f"the fitted vocabulary holds {self.tokenizer.vocab_size} tokens "
                f"but the model declares vocab_size {config.vocab_size}; prepare "
                f"with --vocab-size {config.vocab_size}, or set the model to "
                f"{self.tokenizer.vocab_size}.",
            )
        self.token_bytes = torch.from_numpy(
            self.tokenizer.token_bytes.astype(np.int32),
        ).to(self.device)
        logger.info(
            "nanochat: %d train shards, val shard %d, vocab %d",
            len(self.train_paths),
            config.num_train_shards,
            self.tokenizer.vocab_size,
        )

    @property
    def tokenizer(self) -> Tokenizer:
        """Return the loaded online tokenizer.

        Raises:
          RuntimeError: The dataset uses prepared token rows.

        """
        if self._tokenizer is None:
            raise RuntimeError("Prepared token rows have no online tokenizer.")
        return self._tokenizer

    def train_dataloader(self) -> PackedTokenStream:
        """Build an ordered training stream with optional prefetching.

        Returns:
          stream: Wrapping online batches or finite prepared batches.

        """
        self._live = PackedTokenStream(
            paths=self.train_paths,
            tokenizer=self._tokenizer,
            prepared=self._prepared,
            token_bytes=self.token_bytes,
            batch_size=self.batch_size,
            max_seq_len=self.config.max_seq_len,
            buffer_size=self.config.train_buffer_size
            if self.config.train_buffer_size is not None
            else self.config.buffer_size,
            device=self.device,
            max_batches=None,
            prefetch=True,
        )
        return self._live

    def eval_dataloader(self) -> PackedTokenStream | Iterator[NanoChatBatch]:
        """Restart the fixed-size validation stream from its beginning.

        Returns:
          stream: Packed validation batches in deterministic order.

        """
        if self._reference is not None:
            return self._reference.batches(
                device=self.device,
                vocab_size=len(self.token_bytes),
            )
        return PackedTokenStream(
            paths=self.val_paths,
            tokenizer=self._tokenizer,
            prepared=self._prepared,
            token_bytes=self.token_bytes,
            batch_size=self.eval_batch_size,
            max_seq_len=self.config.max_seq_len,
            buffer_size=self.config.buffer_size,
            device=self.device,
            max_batches=self.config.eval_tokens
            // (self.eval_batch_size * self.config.max_seq_len),
        )

    def state_dict(self) -> StateDict:
        """Snapshot how far the training stream has advanced.

        Returns:
          state: Serializable training-stream progress.

        """
        return {
            "batches": self._live.served if self._live is not None else 0,
            "timer_epoch": self.timer_epoch.state_dict(),
        }

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        """Reject checkpoints whose training stream has already advanced.

        The packer cannot seek; restarting would replay previously consumed batches.

        Args:
          state_dict: Checkpoint containing the consumed batch count.

        Raises:
          ValueError: The checkpoint has consumed batches.

        """
        served = cast(NanoChatData.StateDict, state_dict)["batches"]
        if served:
            raise ValueError(
                f"this checkpoint had served {served} batches, and the packed "
                "stream cannot be positioned without re-tokenizing the corpus "
                "up to that point; resuming would silently replay the start of "
                "the data. Start a fresh run.",
            )


def token_bytes_fingerprint(token_bytes: np.ndarray) -> str:
    """Hash the canonical byte-length table used as the BPB denominator.

    Args:
      token_bytes: Per-token byte lengths.

    Returns:
      fingerprint: SHA-256 of contiguous int64 table bytes.

    """
    return hashlib.sha256(
        np.ascontiguousarray(token_bytes, dtype=np.int64).tobytes(),
    ).hexdigest()


class Tokenizer:
    """Load a fitted tiktoken vocabulary and its verified byte-length table.

    Attributes:
      encoding: Fitted tiktoken encoder.
      bos_token_id: Document-start token ID.
      token_bytes: Per-token UTF-8 byte lengths, including zero-valued specials.

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
            encoding = cast(object, pickle.load(file))  # noqa: S301 -- The artifact is a trusted tokenizer file created by this pipeline.
        assert isinstance(encoding, tiktoken.Encoding)
        recipe = DictCodec.coerce(loads(recipe_path.read_text()), default=None)
        for field in ("bos_token", "token_bytes_sha256"):
            if field not in recipe:
                raise ValueError(
                    f"{recipe_path} declares no {field!r}; it predates the "
                    "current preparer, so what it holds cannot be established. "
                    "Re-prepare the vocabulary.",
                )
        raw = cast(object, np.load(tokenizer_dir / "token_bytes.npy"))
        assert isinstance(raw, np.ndarray)
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
        if np.any(raw < 0):
            raise ValueError(
                f"{tokenizer_dir} holds a negative byte length; the score's "
                "denominator counts bytes, and a negative one would silently "
                "drop its token from the measurement.",
            )
        tokenizer = cls(
            encoding,
            bos_token=StrCodec.coerce(recipe["bos_token"], default=None),
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
        """Return the vocabulary size, including reserved tokens."""
        return int(self.encoding.n_vocab)

    def encode_batch(
        self,
        texts: list[str],
        *,
        num_threads: int = 8,
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


class PackedTokenStream:
    """Batch packed rows using reusable input and target buffers.

    Clone batches that must survive the next fetch; yielded tensors share storage.
    """

    def __init__(
        self,
        *,
        paths: list[Path],
        tokenizer: TokenEncoder | None,
        token_bytes: Tensor,
        batch_size: int,
        max_seq_len: int,
        buffer_size: int,
        device: torch.device,
        max_batches: int | None,
        prefetch: bool = False,
        prepared: PreparedTokenRows | None = None,
    ) -> None:
        self.paths = paths
        self.tokenizer = tokenizer
        self.token_bytes = token_bytes
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.buffer_size = buffer_size
        self.device = device
        self.max_batches = max_batches
        self.prepared = prepared
        # Prefetch requires pinned host memory for asynchronous transfers.
        self.prefetch = prefetch and self._pins_host_memory
        self.served = 0
        """Batches drawn from this stream, across every iteration of it."""

    # Pinning is what makes the host-to-device copy asynchronous, so it is also the
    # precondition for overlapping the copy with compute. Only CUDA supports it here,
    # and the prefetch path is built around it -- keep this the single source of that
    # answer so the staging allocation, the resident buffer, and the prefetch decision
    # cannot disagree.
    @property
    def _pins_host_memory(self) -> bool:
        """Return whether the device supports this loader's pinned-memory transfers."""
        return self.device.type == "cuda"

    def __iter__(self) -> Iterator[NanoChatBatch]:
        """Yield full packed batches, optionally prefetched one batch ahead.

        Yields:
          batch: Input tokens, shifted targets, byte lengths, and valid row count.

        """
        if self.prefetch:
            yield from self._prefetched()
            return
        yield from self._packed()

    def _packed(self) -> Iterator[NanoChatBatch]:
        """Pack and yield batches serially. See :meth:`__iter__`."""
        pinned = self._pins_host_memory
        width = self.batch_size * self.max_seq_len
        staged = torch.empty(
            2 * width,
            dtype=torch.long,
            device="cpu",
            pin_memory=pinned,
        )
        resident = (
            torch.empty(2 * width, dtype=torch.long, device=self.device)
            if self.device.type != "cpu"
            else staged
        )
        # Copy into shaped views directly to avoid materializing an extra contiguous
        # batch.
        staged_media = staged[:width].view(self.batch_size, self.max_seq_len)
        staged_label = staged[width:].view(self.batch_size, self.max_seq_len)
        media = resident[:width].view(self.batch_size, self.max_seq_len)
        label = resident[width:].view(self.batch_size, self.max_seq_len)

        for inputs, targets in self._row_pairs():
            staged_media.copy_(inputs)
            staged_label.copy_(targets)
            if self.device.type != "cpu":
                resident.copy_(staged, non_blocking=pinned)
            self.served += 1
            yield {
                "media": media,
                "label": label,
                "token_bytes": self.token_bytes,
                "valid_count": self.batch_size,
            }

    # The worker packs and stages into PINNED memory; this thread issues the device copy
    # and yields. Two staging slots, alternating, so the worker fills one while the step
    # consumes the other -- and a queue of depth one, so it can never run more than a
    # batch ahead and the packing order is the serial one.
    #
    # The packing is pure Python over a document buffer, so it releases the GIL only
    # inside the tokenizer; what it overlaps is the DEVICE, which is where the step's
    # 1.5 s/step lives.
    def _prefetched(self) -> Iterator[NanoChatBatch]:
        """Pack into two pinned slots on one worker; issue copies on the consumer."""
        width = self.batch_size * self.max_seq_len
        pinned = self._pins_host_memory
        slots = [
            torch.empty(2 * width, dtype=torch.long, device="cpu", pin_memory=pinned)
            for _ in range(2)
        ]
        resident = torch.empty(2 * width, dtype=torch.long, device=self.device)
        media = resident[:width].view(self.batch_size, self.max_seq_len)
        label = resident[width:].view(self.batch_size, self.max_seq_len)

        # A one-slot queue limits prefetch to one batch, keeping two staging buffers
        # sufficient.
        ready: queue.Queue[tuple[int, BaseException | None]] = queue.Queue(maxsize=1)
        done = threading.Event()
        # Wait on the worker; waiting on the consumer would stall queued device work.
        copied = [threading.Event() for _ in slots]
        for event in copied:
            event.set()
        copy_done = [torch.cuda.Event() for _ in slots] if pinned else None

        def pack() -> None:
            pairs = self._row_pairs()
            drawn = 0
            try:
                while self.max_batches is None or drawn < self.max_batches:
                    if done.is_set():
                        return
                    slot = drawn % len(slots)
                    # Wait before reusing pinned storage; the asynchronous copy may
                    # still be reading it.
                    copied[slot].wait()
                    copied[slot].clear()
                    if copy_done is not None:
                        copy_done[slot].synchronize()
                    inputs, targets = next(pairs)
                    staged = slots[slot]
                    staged[:width].view(self.batch_size, self.max_seq_len).copy_(
                        inputs,
                    )
                    staged[width:].view(self.batch_size, self.max_seq_len).copy_(
                        targets,
                    )
                    ready.put((slot, None))
                    drawn += 1
            except BaseException as error:  # noqa: BLE001 -- Dataset iteration contains malformed-record failures per the loader contract.
                ready.put((-1, error))
                return
            ready.put((-1, None))

        worker = threading.Thread(target=pack, name="nanochat-packer", daemon=True)
        worker.start()
        try:
            while True:
                slot, error = ready.get()
                if error is not None:
                    raise error
                if slot < 0:
                    return
                resident.copy_(slots[slot], non_blocking=True)
                if copy_done is not None:
                    copy_done[slot].record()
                copied[slot].set()
                self.served += 1
                yield {
                    "media": media,
                    "label": label,
                    "token_bytes": self.token_bytes,
                    "valid_count": self.batch_size,
                }
        finally:
            # Release a worker blocked on a slot or full queue when the consumer stops
            # early.
            done.set()
            for event in copied:
                event.set()  # Unblock a worker parked on a slot it cannot refill.
            with contextlib.suppress(queue.Empty):
                ready.get_nowait()

    def _row_pairs(self) -> Iterator[tuple[Tensor, Tensor]]:
        if self.prepared is not None:
            yield from self.prepared.batches(
                batch_size=self.batch_size,
                training=self.max_batches is None,
            )
            return
        if self.tokenizer is None:
            raise ValueError("Expected self.tokenizer is not None.")
        rows = self.max_seq_len + 1
        documents = _document_batches(self.paths)
        buffer: list[list[int]] = []
        row_buffer = torch.empty(self.batch_size, rows, dtype=torch.long, device="cpu")
        drawn = 0
        while self.max_batches is None or drawn < self.max_batches:
            for index in range(self.batch_size):
                position = 0
                while position < rows:
                    while len(buffer) < self.buffer_size:
                        buffer.extend(self.tokenizer.encode_batch(next(documents)))
                    position = _pack_row(row_buffer[index], buffer, position=position)
            drawn += 1
            yield row_buffer[:, :-1], row_buffer[:, 1:]

    def __len__(self) -> int:
        """Return the number of evaluation batches.

        Raises:
          TypeError: The training stream has no declared length.

        """
        if self.max_batches is None:
            raise TypeError("the training stream is unbounded and has no length.")
        return self.max_batches


def _pack_row(row: Tensor, buffer: list[list[int]], *, position: int) -> int:
    """Append the largest fitting document, or crop the shortest to fill the row."""
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
            device=row.device,
        )
        return position + len(document)
    shortest = min(range(len(buffer)), key=lambda index: len(buffer[index]))
    document = buffer.pop(shortest)
    row[position : position + remaining] = torch.tensor(
        document[:remaining],
        dtype=torch.long,
        device=row.device,
    )
    return position + remaining


# Unbounded: the stream is what a budgeted run draws from, and a run that outlasts the
# corpus continues from its start rather than ending.
def _document_batches(paths: list[Path]) -> Iterator[list[str]]:
    """Yield parquet documents in shard order, wrapping after the final shard."""
    while True:
        productive = False
        for path in paths:
            shard = parquet.ParquetFile(path)
            for group in range(shard.num_row_groups):
                texts = cast(
                    list[str],
                    (shard.read_row_group(group).column("text").to_pylist()),
                )
                for start in range(0, len(texts), DOCUMENTS_PER_REFILL):
                    productive = True
                    yield texts[start : start + DOCUMENTS_PER_REFILL]
        if not productive:
            raise ValueError("The online corpus contains no documents.")


def _shard_paths(directory: Path, *, indices: range | list[int]) -> list[Path]:
    """Return existing shard paths in the requested index order."""
    paths = [directory / f"shard_{index:05d}.parquet" for index in indices]
    missing = [path.name for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"{directory} is missing {missing}; prepare the corpus with "
            "`uv --quiet run --frozen python -m "
            "priml.baselines.nanochat.scripts.prepare_data`.",
        )
    return paths


class PreparedTokenRows:
    """Memory-map prepared token rows without loading an online encoder."""

    def __init__(self, config: NanoChatData.Config) -> None:
        train_path = Path(config.prepared_train_manifest)
        eval_path = Path(config.prepared_eval_manifest)
        train = _mapping(loads(train_path.read_text()))
        evaluation = _mapping(loads(eval_path.read_text()))
        self.vocab_size = _integer(train["vocab_size"])
        self.bos_token_id = _integer(train["bos_id"])
        geometry = _mapping(train["train"])
        if (
            evaluation["protocol"] != "standard-packed-shard7-tokensub-v1"
            or evaluation["vocab_size"] != self.vocab_size
            or evaluation["bos_token_id"] != self.bos_token_id
            or geometry["batch_size"] != config.batch_size
            or evaluation["eval_batch_size"] != config.eval_batch_size
            or geometry["seq_len"] != config.max_seq_len
            or evaluation["max_seq_len"] != config.max_seq_len
            or evaluation["physical_positions"] != config.eval_tokens
            or (config.vocab_size > 0 and config.vocab_size != self.vocab_size)
            or evaluation["val_shard"] != config.val_shard
            or evaluation["buffer_size"] != config.buffer_size
            or geometry["buffer_size"]
            != (config.train_buffer_size or config.buffer_size)
            or geometry["train_shard_indices"]
            != list(config.train_shard_indices or range(config.num_train_shards))
        ):
            raise ValueError(
                "Prepared geometry or evaluation protocol differs from config.",
            )
        self.train_rows = _array(
            train_path.parent / "train_rows.npy",
            shape=(_integer(geometry["total_rows"]), config.max_seq_len + 1),
            dtype=np.dtype(np.uint16),
        )
        shape = (_integer(evaluation["rows"]), config.max_seq_len)
        if (
            shape[0] * shape[1] != config.eval_tokens
            or shape[0] != _integer(evaluation["batches"]) * config.eval_batch_size
            or len(self.train_rows) % config.batch_size
        ):
            raise ValueError("Prepared array geometry does not contain whole batches.")
        self.eval_inputs = _array(
            eval_path.parent / "eval_x.npy",
            shape=shape,
            dtype=np.dtype(np.uint16),
        )
        self.eval_targets = _array(
            eval_path.parent / "eval_y.npy",
            shape=shape,
            dtype=np.dtype(np.uint16),
        )
        tables = _mapping(evaluation["byte_tables"])
        self.token_bytes = _byte_table(
            eval_path.parent,
            metadata=_mapping(tables["primary"]),
            vocab=self.vocab_size,
        )
        literal = _byte_table(
            eval_path.parent,
            metadata=_mapping(tables["literal"]),
            vocab=self.vocab_size,
        )
        if (
            np.any(self.token_bytes[: self.bos_token_id] <= 0)
            or np.any(
                cast(NDArray[np.bool_], self.token_bytes[self.bos_token_id :] != 0),
            )
            or np.any(
                cast(
                    NDArray[np.bool_],
                    (self.token_bytes > 0) != (literal > 0),
                ),
            )
            or np.any(literal < 0)
        ):
            raise ValueError("Prepared byte tables disagree on the scoring mask.")
        totals = [0, 0]
        scored = 0
        for start in range(0, shape[0], config.eval_batch_size):
            labels = self.eval_targets[start : start + config.eval_batch_size]
            inputs = self.eval_inputs[start : start + config.eval_batch_size]
            if np.any(labels >= self.vocab_size) or np.any(inputs >= self.vocab_size):
                raise ValueError(
                    "Prepared evaluation contains an out-of-vocabulary token.",
                )
            lengths = self.token_bytes[labels]
            totals[0] += int(lengths.sum())
            totals[1] += int(literal[labels].sum())
            scored += int(np.count_nonzero(lengths))
        if (
            totals[0] != _mapping(tables["primary"])["total_on_eval_y"]
            or totals[1] != _mapping(tables["literal"])["total_on_eval_y"]
            or scored != tables["scored_positions"]
        ):
            raise ValueError(
                "Prepared evaluation byte totals or scored positions differ.",
            )

    def batches(
        self,
        *,
        batch_size: int,
        training: bool,
    ) -> Iterator[tuple[Tensor, Tensor]]:
        """Yield host input/target pairs in the frozen order.

        Args:
          batch_size: Rows per batch, already checked against the manifest.
          training: Select training rows; otherwise replay the packed evaluation.

        Yields:
          inputs: CPU int64 input tokens.
          targets: CPU int64 target tokens.

        Raises:
          RuntimeError: Training consumes all prepared rows. Never wrap or tokenize.

        """
        rows = self.train_rows if training else self.eval_inputs
        for start in range(0, len(rows), batch_size):
            block = torch.from_numpy(rows[start : start + batch_size].astype(np.int64))
            if training:
                yield block[:, :-1], block[:, 1:]
            else:
                targets = torch.from_numpy(
                    self.eval_targets[start : start + batch_size].astype(np.int64),
                )
                yield block, targets
        if training:
            raise RuntimeError("PREPARED_EXHAUSTED: no wrapping or online fallback.")


class ReferenceEvaluation:
    """Load byte-matched evaluation inputs independently of training data."""

    class Config(Fig["ReferenceEvaluation"]):
        """Locate a prepared evaluation archive."""

        path: Path = Path("/opt/scratch/datasets/nanochat/reference-eval/unigram.npz")
        """Archive containing token rows, masks, and byte denominators."""

    def __init__(self, config: Config) -> None:
        loaded = cast(object, np.load(config.path, allow_pickle=False))
        assert isinstance(loaded, NpzFile)
        with loaded as archive:
            if str(archive["protocol"]) != "karpathy-reference-bytes-v1":
                raise ValueError("Unsupported reference evaluation protocol.")
            # The dtypes are claims about the archive; ``_validate`` checks each.
            self.inputs = cast(NDArray[np.int64], archive["inputs"])
            self.targets = cast(NDArray[np.int64], archive["targets"])
            self.mask = cast(NDArray[np.bool_], archive["score_mask"])
            self.token_bytes = cast(NDArray[np.int64], archive["token_bytes"])
            self.reference_bytes = cast(NDArray[np.int64], archive["reference_bytes"])
            self.literal_bytes = cast(NDArray[np.int64], archive["literal_bytes"])
            self.batch_size = int(archive["batch_size"])
            self.vocab_size = int(archive["vocab_size"])
            self.bos_token_id = int(archive["bos_token_id"])
        self._validate()

    def batches(
        self,
        *,
        device: torch.device | str,
        vocab_size: int,
    ) -> Iterator[NanoChatBatch]:
        """Replay the archive with its original batch boundaries.

        Args:
          device: Destination device for model inputs and masks.
          vocab_size: Model vocabulary, which must match the prepared tokenizer.

        Yields:
          batch: Tokens, scoring mask, byte denominators, and completeness counters.

        """
        if vocab_size != self.vocab_size:
            raise ValueError("Reference evaluation vocabulary differs from model.")
        count = math.ceil(len(self.inputs) / self.batch_size)
        table = torch.tensor(self.token_bytes, dtype=torch.int32, device=device)
        for index, start in enumerate(range(0, len(self.inputs), self.batch_size)):
            end = start + self.batch_size
            yield {
                "media": torch.tensor(self.inputs[start:end], device=device),
                "label": torch.tensor(self.targets[start:end], device=device),
                "token_bytes": table,
                "valid_count": len(self.inputs[start:end]),
                "score_mask": torch.tensor(self.mask[start:end], device=device),
                "reference_bytes": int(self.reference_bytes[start:end].sum()),
                "literal_bytes": int(self.literal_bytes[start:end].sum()),
                "evaluation_batch": index,
                "evaluation_batches": count,
            }

    def _validate(self) -> None:
        if (
            self.inputs.ndim != 2
            or len(self.inputs) == 0
            or self.inputs.shape != self.targets.shape
            or self.inputs.shape != self.mask.shape
            or self.inputs.dtype != np.int64
            or self.targets.dtype != np.int64
            or self.mask.dtype != np.bool_
            or self.batch_size <= 0
            or self.reference_bytes.shape != (len(self.inputs),)
            or self.literal_bytes.shape != (len(self.inputs),)
            or self.reference_bytes.dtype != np.int64
            or self.literal_bytes.dtype != np.int64
            or self.token_bytes.shape != (self.vocab_size,)
            or self.token_bytes.dtype != np.int64
        ):
            raise ValueError("Invalid reference evaluation array geometry or dtype.")
        if (
            self.bos_token_id < 0
            or self.bos_token_id >= self.vocab_size
            or np.any(self.inputs < 0)
            or np.any(self.inputs >= self.vocab_size)
            or np.any(self.targets < 0)
            or np.any(self.targets >= self.vocab_size)
            or np.any(self.mask & (self.targets >= self.bos_token_id))
            or np.any(self.reference_bytes < 0)
            or np.any(self.literal_bytes < 0)
            or int(self.literal_bytes.sum()) <= 0
            or int(self.reference_bytes.sum()) <= 0
        ):
            raise ValueError("Invalid reference targets, scoring mask, or byte counts.")


def _array(path: Path, *, shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
    array = cast(object, np.load(path, mmap_mode="r", allow_pickle=False))
    assert isinstance(array, np.ndarray)
    if array.shape != shape or array.dtype != dtype or not array.flags.c_contiguous:
        raise ValueError(f"Prepared array geometry/dtype mismatch: {path}.")
    return array


def _byte_table(
    directory: Path,
    *,
    metadata: dict[str, object],
    vocab: int,
) -> NDArray[np.int64]:
    name = metadata["file"]
    assert isinstance(name, str)
    array = cast(object, np.load(directory / name, allow_pickle=False))
    assert isinstance(array, np.ndarray)
    if array.shape != (vocab,) or not np.issubdtype(array.dtype, np.integer):
        raise ValueError("Prepared byte table must contain one integer per token.")
    return np.ascontiguousarray(array, dtype=np.int64)


def _mapping(value: object) -> dict[str, object]:
    return dict(DictCodec.coerce(value, default=None))


def _integer(value: object) -> int:
    return IntCodec.coerce(value, default=None)
