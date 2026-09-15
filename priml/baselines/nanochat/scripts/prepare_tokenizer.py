"""Prepare fitting samples and reversible byte tokenizers."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, cast

import hashlib
import json
import logging
import math
import tempfile

from configgle import Fig

import rustbpe
import tokenizers

from priml.lib.custom_json import DictCodec
from priml.paths import validated_output_path


if TYPE_CHECKING:
    from numpy import array, save, zeros
    from pyarrow import Table, parquet

    import tiktoken
else:
    from wrapt import lazy_import

    array = lazy_import("numpy", "array")
    save = lazy_import("numpy", "save")
    zeros = lazy_import("numpy", "zeros")
    Table = lazy_import("pyarrow", "Table")
    parquet = lazy_import("pyarrow.parquet")
    tiktoken = lazy_import("tiktoken")


logger = logging.getLogger(__name__)


def read_mapping(path: Path) -> dict[str, object]:
    """Read a JSON object without interpreting its nested schema.

    Args:
      path: JSON input.

    Returns:
      mapping: The decoded object.

    """
    return dict(DictCodec.coerce(json.loads(path.read_text()), default=None))


def write_mapping(path: Path, *, value: object) -> None:
    """Publish deterministic JSON after its complete contents have been written.

    Args:
      path: Destination, outside protected inputs.
      value: JSON-compatible preparation metadata.

    """
    destination = validated_output_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=destination.parent, delete=False, prefix="nanochat-json-"
    ) as output:
        output.write(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        staged = Path(output.name)
    staged.replace(destination)


def document_rows(path: Path, *, shard: int) -> Iterator[tuple[str, str]]:
    """Yield original row identities and literal text in source order.

    Args:
      path: Original Parquet shard.
      shard: Original shard index.

    Yields:
      row: Original identity and unmodified text.

    """
    offset = 0
    for batch in parquet.ParquetFile(path).iter_batches(
        batch_size=1_024, columns=["text"]
    ):
        for text in cast(list[str], batch.column(0).to_pylist()):
            yield f"{shard}:{offset}", text
            offset += 1


def usage_scores(pieces: list[bytes], counts: list[int]) -> list[float]:
    """Rank pieces by hard-EM occurrence count.

    Args:
      pieces: Ordered byte pieces.
      counts: Occurrences from the final Viterbi pass.

    Returns:
      scores: Occurrence counts, one per piece.

    """
    assert len(pieces) == len(counts)
    return [float(count) for count in counts]


class SamplePreparation:
    """Reproduce the train-only stratified byte windows used for vocabulary fitting."""

    class Config(Fig["SamplePreparation"]):
        """Declare sample selection and its source identities."""

        raw_dir: Path = Path("/opt/scratch/datasets/nanochat/raw")
        """Original unmodified shards, before model-training deduplication."""

        working_dir: Path = Path("/opt/scratch/datasets/nanochat/sample")
        """Destination for fitting text."""

        val_shard: int = 7
        """Heldout shard, excluded from vocabulary fitting."""

        train_shard_indices: tuple[int, ...] = (*range(7), *range(8, 15))
        """Ordered training sources; validation shard7 is excluded."""

        rows_per_shard: int = 4_096
        """Number of equal-population midpoint strata in each shard."""

        max_bytes: int = 8_192
        """Maximum UTF-8-safe window, positioned by the document SHA-256."""

    def __init__(self, config: Config) -> None:
        self.config = config
        if config.val_shard in config.train_shard_indices:
            raise ValueError("The fitting sample must exclude validation shard7.")
        if config.rows_per_shard <= 0 or config.max_bytes < 4:
            raise ValueError(
                "Sample size must be positive and windows at least four bytes.",
            )

    def build(self) -> None:
        """Write selected literal text in the original shard and row order."""
        output = validated_output_path(
            self.config.working_dir,
            protected=[self.config.raw_dir],
        )
        output.mkdir(parents=True, exist_ok=False)
        texts: list[str] = []
        for shard in self.config.train_shard_indices:
            path = self.config.raw_dir / f"shard_{shard:05d}.parquet"
            total = parquet.read_metadata(path).num_rows
            count = min(total, self.config.rows_per_shard)
            selected = {
                (2 * index + 1) * total // (2 * count) for index in range(count)
            }
            for index, (_, text) in enumerate(document_rows(path, shard=shard)):
                if index not in selected:
                    continue
                raw = text.encode()
                start, end = sample_window(raw, max_bytes=self.config.max_bytes)
                sample = raw[start:end]
                if sample:
                    texts.append(sample.decode())
        parquet.write_table(
            Table.from_pydict({"text": texts}),
            output / "sample.parquet",
        )


class UnigramPreparation:
    """Fit the overshot BPE seed, two Viterbi passes, and a pruned byte Unigram."""

    class Config(Fig["UnigramPreparation"]):
        """Keep every fitting choice visible in the preparation recipe."""

        sample_dir: Path = Path("/opt/scratch/datasets/nanochat/sample")
        """Fitting sample containing the selected literal text."""

        working_dir: Path = Path("/opt/scratch/datasets/nanochat/unigram16k")
        """Tokenizer assets and fitted piece counts."""

        vocab_size: int = 16_384
        """Model vocabulary including reserved IDs."""

        reserved_count: int = 16
        """IDs appended after ordinary byte pieces; the first is BOS."""

        overshoot_learned: float = 1.15
        """Multiplier applied to learned seed pieces beyond the 256 bytes."""

        num_passes: int = 2
        """Hard-EM Viterbi recount passes before one-shot pruning."""

        batch_size: int = 256
        """Documents per deterministic tokenizer counting batch."""

        num_threads: int = 8
        """Threads used by the initial tiktoken counting pass."""

        split_pattern: str = r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"
        """Inherited byte-BPE pretokenization expression."""

        pruning: Callable[[list[bytes], list[int]], list[float]] = usage_scores
        """Injected pruning objective; ties retain the earlier seed piece."""

    def __init__(self, config: Config) -> None:
        self.config = config
        if config.vocab_size - config.reserved_count < 256:
            raise ValueError("The ordinary vocabulary must contain every byte.")

    def build(self) -> None:
        """Fit the tokenizer from the selected text and save its piece counts."""
        texts = load_sample(self.config.sample_dir)
        output = validated_output_path(
            self.config.working_dir,
            protected=[self.config.sample_dir],
        )
        output.mkdir(parents=True, exist_ok=False)
        ordinary = self.config.vocab_size - self.config.reserved_count
        trainer = rustbpe.Tokenizer()
        trainer.train_from_iterator(
            iter(texts),
            256 + math.ceil((ordinary - 256) * self.config.overshoot_learned),
            pattern=self.config.split_pattern,
        )
        ranks = {bytes(piece): rank for piece, rank in trainer.get_mergeable_ranks()}
        pieces = [piece for piece, _ in sorted(ranks.items(), key=lambda pair: pair[1])]
        seed = tiktoken.Encoding(
            name="nanochat-reproduction-seed",
            pat_str=self.config.split_pattern,
            mergeable_ranks=ranks,
            special_tokens={},
        )
        frequencies: Counter[int] = Counter()
        for start in range(0, len(texts), self.config.batch_size):
            for ids in seed.encode_ordinary_batch(
                texts[start : start + self.config.batch_size],
                num_threads=self.config.num_threads,
            ):
                frequencies.update(ids)
        counts = [frequencies[index] for index in range(len(pieces))]
        for iteration in range(self.config.num_passes):
            model = frequency_model(
                pieces,
                counts=counts,
                split_pattern=self.config.split_pattern,
            )
            frequencies = Counter()
            for start in range(0, len(texts), self.config.batch_size):
                for encoded in model.encode_batch(
                    texts[start : start + self.config.batch_size],
                    add_special_tokens=False,
                ):
                    frequencies.update(encoded.ids)
            counts = [frequencies[index] for index in range(len(pieces))]
            logger.info("Hard-EM pass %d: %d tokens", iteration + 1, sum(counts))
        scores = self.config.pruning(pieces, counts)
        keep = sorted(
            [
                *range(256),
                *sorted(
                    range(256, len(pieces)),
                    key=lambda index: (-scores[index], index),
                )[: ordinary - 256],
            ],
        )
        model = frequency_model(
            [pieces[index] for index in keep],
            counts=[counts[index] for index in keep],
            split_pattern=self.config.split_pattern,
        )
        model.save(str(output / "tokenizer.json"))
        save(
            output / "counts.npy",
            array([counts[index] for index in keep], dtype="int64"),
        )


def load_sample(directory: Path) -> list[str]:
    """Read fitting text in its prepared row order.

    Args:
      directory: Prepared sample Parquet directory.

    Returns:
      texts: Exact fitting texts in the original order.

    """
    return cast(
        list[str],
        parquet.read_table(directory / "sample.parquet").column("text").to_pylist(),
    )


def sample_window(raw: bytes, *, max_bytes: int) -> tuple[int, int]:
    """Choose the original SHA-positioned UTF-8-safe sample window.

    Args:
      raw: Complete UTF-8 document.
      max_bytes: Maximum window size.

    Returns:
      start: Inclusive byte offset.
      end: Exclusive byte offset.

    """
    digest = hashlib.sha256(raw).hexdigest()
    start = int(digest[:16], 16) % max(1, len(raw) - max_bytes + 1)
    while start < len(raw) and raw[start] & 0xC0 == 0x80:
        start += 1
    end = min(start + max_bytes, len(raw))
    while end < len(raw) and raw[end] & 0xC0 == 0x80:
        end -= 1
    return start, end


def frequency_model(
    pieces: list[bytes],
    *,
    counts: list[int],
    split_pattern: str,
) -> tokenizers.Tokenizer:
    """Construct the exact byte Unigram encoding and pretokenization pipeline.

    Args:
      pieces: Ordered byte pieces; order defines ordinary token IDs.
      counts: Hard-EM occurrence counts, floored at one for log probabilities.
      split_pattern: Frozen regular-expression pretokenizer.

    Returns:
      model: Unnormalized, reversible ByteLevel Unigram tokenizer.

    """
    alphabet = byte_alphabet()
    total = sum(max(1, count) for count in counts)
    vocab = [
        ("".join(alphabet[value] for value in piece), math.log(max(1, count) / total))
        for piece, count in zip(pieces, counts, strict=True)
    ]
    model = tokenizers.Tokenizer(
        tokenizers.models.Unigram(vocab, unk_id=None, byte_fallback=False),
    )
    model.pre_tokenizer = tokenizers.pre_tokenizers.Sequence(
        [
            tokenizers.pre_tokenizers.Split(
                tokenizers.Regex(split_pattern),
                behavior="isolated",
            ),
            tokenizers.pre_tokenizers.ByteLevel(
                add_prefix_space=False,
                use_regex=False,
            ),
        ],
    )
    model.decoder = tokenizers.decoders.ByteLevel()
    return model


def byte_alphabet() -> dict[int, str]:
    """Return the reversible GPT-2 ByteLevel alphabet.

    Returns:
      alphabet: One distinct Unicode character for each byte value.

    """
    values = [*range(33, 127), *range(161, 173), *range(174, 256)]
    mapping = dict(zip(values, map(chr, values), strict=True))
    for value in range(256):
        if value not in mapping:
            mapping[value] = chr(256 + len(mapping) - len(values))
    return mapping


class ByteLevelTokenizer:
    """Load an ordinary byte tokenizer and add the separate BOS boundary."""

    class Config(Fig["ByteLevelTokenizer"]):
        """Identify the fitted model and reserved vocabulary."""

        path: Path = Path("/opt/scratch/datasets/nanochat/unigram16k/tokenizer.json")
        """Frozen HF tokenizer JSON."""

        reserved_count: int = 16
        """Reserved IDs outside the ordinary tokenizer; first is BOS."""

    def __init__(self, config: Config) -> None:
        self.backend = tokenizers.Tokenizer.from_file(str(config.path))
        document = read_mapping(config.path)
        if (
            document.get("normalizer") is not None
            or document.get("post_processor") is not None
            or document.get("added_tokens")
        ):
            raise ValueError(
                "Expected ordinary byte pieces without normalization or added tokens."
            )
        self.bos_token_id = self.backend.get_vocab_size(with_added_tokens=False)
        self.vocab_size = self.bos_token_id + config.reserved_count
        self.token_bytes = zeros(self.vocab_size, dtype="int32")
        self.token_bytes_literal = zeros(self.vocab_size, dtype="int32")
        for index in range(self.bos_token_id):
            piece = self.backend.id_to_token(index)
            assert piece is not None
            self.token_bytes_literal[index] = len(piece)
            self.token_bytes[index] = len(
                self.backend.decode([index], skip_special_tokens=False).encode()
            )

    def encode_batch(
        self, texts: list[str], *, num_threads: int = 8
    ) -> list[list[int]]:
        """Encode unmodified documents and prepend exactly one BOS.

        Args:
          texts: Literal UTF-8 documents.
          num_threads: Native stream compatibility; HF owns its encoder pool.

        Returns:
          rows: BOS-prefixed IDs, verified to preserve each document's bytes.

        """
        del num_threads
        output: list[list[int]] = []
        for text, encoded in zip(
            texts,
            self.backend.encode_batch(texts, add_special_tokens=False),
            strict=True,
        ):
            ids = encoded.ids
            if self.backend.decode(ids, skip_special_tokens=False) != text:
                raise ValueError("The tokenizer failed literal document round-trip.")
            if int(self.token_bytes_literal[ids].sum()) != len(text.encode()):
                raise ValueError("Token pieces do not conserve literal UTF-8 bytes.")
            output.append([self.bos_token_id, *ids])
        return output
