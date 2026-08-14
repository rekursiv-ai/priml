"""Download the corpus, train a tokenizer, and pack it into token rows.

Run once before the first experiment. Idempotent: a split already present is
left alone, so re-running costs nothing.

Three stages, in order, because each consumes the previous one's output:

1. **Download.** Parquet text shards, fetched once and reused.
2. **Train the tokenizer.** A byte-pair vocabulary fitted on a prefix of the
   training text. The vocabulary is part of the recipe -- it decides what a
   token IS -- so it is built here and frozen, never refitted per run.
3. **Pack.** Documents are concatenated end to end and cut into fixed-length
   rows, so every position carries a real target and training needs no padding
   mask. A document longer than a row spans several; one shorter shares a row
   with its neighbours, separated by the document-start token.

The build is deterministic: the shard order is fixed and the tokenizer's
training text is a prefix, so the same flags over the same shards produce
byte-identical arrays.

The tokenizer libraries are imported HERE and nowhere else. Preparation happens
once, on a machine with a network; training reads flat arrays, so a published
install needs neither.

Examples:
  uv --quiet run --frozen python -m priml.baselines.nanochat.scripts.prepare_data
  uv --quiet run --frozen python -m priml.baselines.nanochat.scripts.prepare_data --directory /datasets/my-nanochat

"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import argparse
import json
import logging
import os
import pickle
import shutil
import tempfile
import urllib.request

import numpy as np

from priml.baselines.nanochat.data import NanoChatData, token_bytes_fingerprint
from priml.train.train_loop import TrainLoop


if TYPE_CHECKING:
    import tiktoken


logger = logging.getLogger(__name__)

SOURCE_URL: Final = (
    "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve"
)
"""Base URL of the source shards.

Fetched over plain HTTP rather than through a Hugging Face client: a handful of
files, nothing a dependency would add. Keeping it stdlib means preparing data
needs no optional extra."""

SOURCE_REVISION: Final = "main"
"""Revision the shards are fetched at.

NOT a pin: this tracks the branch, so a corpus that moves upstream produces
different shards under the same name. What IS pinned is the artifact those
shards become -- ``dataset.json`` records the tokenizer's byte-length table by
digest, so a score measured against one preparation cannot be silently compared
with a score measured against another. Digest-pinning the shards themselves
would mean listing every one of them here."""


def main() -> int:
    """Prepare the dataset; return the process exit code."""
    parser = argparse.ArgumentParser(
        description=(__doc__ or "").strip(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_arguments(parser)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    prepare(
        args.directory,
        num_train_shards=args.num_train_shards,
        vocab_size=args.vocab_size,
        max_seq_len=args.max_seq_len,
        tokenizer_train_chars=args.tokenizer_train_chars,
    )
    return 0


def default_directory() -> Path:
    """Return the dataset directory a default ``TrainLoop`` would resolve."""
    config = NanoChatData.Config()
    config.base_dir = TrainLoop.Config().base_dir
    return Path(config.copy_tree().finalize().working_dir)


def prepare(
    directory: Path | str | None = None,
    *,
    num_train_shards: int = 7,
    vocab_size: int = 8_192,
    max_seq_len: int = 2_048,
    tokenizer_train_chars: int = 200_000_000,
    text_directory: Path | str | None = None,
) -> Path:
    """Build both splits under ``directory`` if they are not already there.

    Args:
      directory: Destination; ``None`` uses :func:`default_directory`.
      num_train_shards: Source shards forming the training split. The shard
        after them is the validation split, so every candidate is scored on
        text no run trained on.
      vocab_size: Tokenizer vocabulary, including the reserved tokens.
      max_seq_len: Tokens per packed row.
      tokenizer_train_chars: Characters the tokenizer is fitted on.
      text_directory: Local ``shard_*.parquet`` files to read instead of
        downloading, each carrying a ``text`` column. Lets a test build the
        pipeline hermetically.

    Returns:
      directory: Where the splits were written.

    """
    out = Path(directory) if directory is not None else default_directory()
    out.mkdir(parents=True, exist_ok=True)
    prepared = [out / split / "dataset.json" for split in ("train", "val")]
    if all(path.is_file() for path in prepared):
        # Reusing a split prepared under DIFFERENT flags would hand back data
        # that silently is not what was asked for, so the recorded geometry has
        # to agree before the early return.
        for path in prepared:
            recorded = json.loads(path.read_text())
            asked = {"vocab_size": vocab_size, "max_seq_len": max_seq_len}
            differing = {
                key: (recorded.get(key), value)
                for key, value in asked.items()
                if recorded.get(key) != value
            }
            if differing:
                raise ValueError(
                    f"{path.parent} was prepared with {differing} (recorded, "
                    "requested); prepare into a different --directory, or "
                    "delete this one to rebuild it.",
                )
        logger.info("nanochat data already prepared at %s", out)
        return out

    source = Path(text_directory) if text_directory is not None else out / "source"
    shards = (
        _shard_paths(source, count=num_train_shards + 1)
        if text_directory is not None
        else _download(out, count=num_train_shards + 1)
    )
    encoding = _tokenizer(
        shards[:num_train_shards],
        out=out,
        vocab_size=vocab_size,
        train_chars=tokenizer_train_chars,
    )
    for split, split_shards in (
        ("train", shards[:num_train_shards]),
        ("val", shards[num_train_shards:]),
    ):
        _pack_split(
            split,
            shards=split_shards,
            out=out,
            encoding=encoding,
            max_seq_len=max_seq_len,
            vocab_size=vocab_size,
        )
    logger.info("nanochat data ready at %s", out)
    return out


def _download(out: Path, *, count: int) -> list[Path]:
    """Fetch the source shards, skipping any already present."""
    source = out / "source"
    source.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for index in range(count):
        name = f"shard_{index:05d}.parquet"
        path = source / name
        if not path.is_file():
            url = f"{SOURCE_URL}/{SOURCE_REVISION}/{name}"
            logger.info("downloading %s", url)
            handle, staged = tempfile.mkstemp(dir=source, prefix=f".{name}.")
            os.close(handle)
            staging = Path(staged)
            # Stream rather than read whole: a shard is hundreds of MB.
            with (
                urllib.request.urlopen(url) as response,  # noqa: S310 -- fixed https URL from pinned constants
                staging.open("wb") as file,
            ):
                shutil.copyfileobj(response, file)
            staging.replace(path)
        paths.append(path)
    return paths


def _shard_paths(source: Path, *, count: int) -> list[Path]:
    """Return locally staged shards, newest naming scheme first.

    Raises:
      FileNotFoundError: Fewer shards are present than the split needs.

    """
    paths = sorted(source.glob("shard_*"))[:count]
    if len(paths) < count:
        raise FileNotFoundError(
            f"{source} holds {len(paths)} shards but the splits need {count}.",
        )
    return paths


def _tokenizer(
    shards: list[Path],
    *,
    out: Path,
    vocab_size: int,
    train_chars: int,
) -> tiktoken.Encoding:
    """Fit a byte-pair vocabulary, or load the one already fitted.

    Args:
      shards: Training shards supplying the text.
      out: Dataset root; the encoding is cached at ``tokenizer.pkl``.
      vocab_size: Vocabulary size, including the reserved tokens.
      train_chars: Characters to fit on.

    Returns:
      encoding: The fitted tokenizer.

    """
    # Imported here, not at module scope: training reads flat arrays, so a
    # published install must not need a BPE trainer to import this package.
    import rustbpe  # noqa: PLC0415 -- preparation-only dependency
    import tiktoken  # noqa: PLC0415 -- preparation-only dependency

    path = out / "tokenizer.pkl"
    if path.is_file():
        with path.open("rb") as file:
            cached = pickle.load(file)  # noqa: S301 -- our own prepared artifact
        # A file at this path written by anything else would otherwise surface
        # as an AttributeError deep inside packing.
        assert isinstance(cached, tiktoken.Encoding), type(cached).__name__
        return cached

    logger.info("fitting a %d-token vocabulary", vocab_size)
    trainer = rustbpe.Tokenizer()
    trainer.train_from_iterator(
        _documents(shards, max_chars=train_chars),
        vocab_size - len(RESERVED_TOKENS),
        pattern=SPLIT_PATTERN,
    )
    ranks = {bytes(token): rank for token, rank in trainer.get_mergeable_ranks()}
    encoding = tiktoken.Encoding(
        name="nanochat",
        pat_str=trainer.get_pattern(),
        mergeable_ranks=ranks,
        special_tokens={
            name: len(ranks) + index for index, name in enumerate(RESERVED_TOKENS)
        },
    )
    staging = path.with_suffix(".pkl.partial")
    with staging.open("wb") as file:
        pickle.dump(encoding, file)
    staging.replace(path)
    return encoding


def _pack_split(
    split: str,
    *,
    shards: list[Path],
    out: Path,
    encoding: tiktoken.Encoding,
    max_seq_len: int,
    vocab_size: int,
) -> None:
    """Tokenize a split's shards and cut them into fixed-length rows."""
    destination = out / split
    if (destination / "dataset.json").is_file():
        logger.info("nanochat %r already prepared; skipping", split)
        return
    stream: list[int] = []
    for document in _documents(shards, max_chars=None):
        stream.append(encoding.encode_single_token(BOS_TOKEN))
        stream.extend(encoding.encode_ordinary(document))
    # A row holds one extra token: the inputs are the row without its last
    # token and the targets the row without its first.
    width = max_seq_len + 1
    rows = len(stream) // width
    if not rows:
        raise ValueError(
            f"nanochat {split!r} tokenized to {len(stream)} tokens, fewer than "
            f"the {width} one row needs.",
        )
    packed = np.array(stream[: rows * width], dtype=np.uint16).reshape(rows, width)

    destination.mkdir(parents=True, exist_ok=True)
    token_bytes = _token_bytes(encoding, vocab_size=vocab_size)
    np.save(destination / "all__tokens.npy", packed)
    np.save(destination / "all__token_bytes.npy", token_bytes)
    (destination / "dataset.json").write_text(
        json.dumps(
            {
                "vocab_size": vocab_size,
                "max_seq_len": max_seq_len,
                # The score's denominator, recorded so a later change to byte
                # accounting is caught at load rather than silently repricing
                # every number measured against this split.
                "token_bytes_sha256": token_bytes_fingerprint(token_bytes),
            },
        ),
    )
    logger.info("nanochat %r: %d rows of %d tokens", split, rows, max_seq_len)


def _token_bytes(encoding: tiktoken.Encoding, *, vocab_size: int) -> np.ndarray:
    """UTF-8 byte length of every token id; reserved tokens count as zero.

    The bits-per-byte score divides by these, so a reserved token contributing
    zero is what keeps document boundaries out of the denominator.
    """
    reserved = set(RESERVED_TOKENS)
    lengths = [
        0 if (text := encoding.decode([token])) in reserved else len(text.encode())
        for token in range(vocab_size)
    ]
    return np.array(lengths, dtype=np.int32)


def _documents(shards: list[Path], *, max_chars: int | None) -> Iterator[str]:
    """Yield documents from parquet shards, stopping after ``max_chars``."""
    # Imported here for the same reason as the tokenizer: parquet is a
    # preparation format, and training never reads one.
    from pyarrow import parquet  # noqa: PLC0415 -- preparation-only dependency

    seen = 0
    for path in shards:
        shard = parquet.ParquetFile(path)
        for group in range(shard.num_row_groups):
            column: list[Any] = shard.read_row_group(group).column("text").to_pylist()
            for document in column:
                text = str(document)
                yield text
                seen += len(text)
                if max_chars is not None and seen >= max_chars:
                    return


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register flags on ``parser``."""
    parser.add_argument(
        "--directory",
        default=None,
        help=f"destination (default: {default_directory()})",
    )
    parser.add_argument(
        "--num-train-shards",
        type=int,
        default=7,
        help="source shards forming the training split",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=8_192,
        help="tokenizer vocabulary, including reserved tokens",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=2_048,
        help="tokens per packed row",
    )
    parser.add_argument(
        "--tokenizer-train-chars",
        type=int,
        default=200_000_000,
        help="characters the tokenizer is fitted on",
    )


RESERVED_TOKENS: Final = tuple(f"<|reserved_{index}|>" for index in range(16))
"""Tokens appended after the byte-pair merges.

Sixteen rather than one because a vocabulary cannot be extended after the fact
without renumbering every id: a later task needing a turn separator or a tool
marker takes one of these instead of invalidating every checkpoint."""

BOS_TOKEN: Final = RESERVED_TOKENS[0]
"""Marks a document's start; every packed row's first document begins with it."""

SPLIT_PATTERN: Final = (
    r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}"""
    r"""| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
)
"""Regex splitting text before the byte-pair merges are applied.

Caps numbers at two digits and keeps a leading space with its word, both of
which bound how much the vocabulary spends on rare literals."""


if __name__ == "__main__":
    raise SystemExit(main())
